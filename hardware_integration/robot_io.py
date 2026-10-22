"""The real robot's sensors and wheels in the form already used by this projet's pipeline

    Waffle     RosbridgeRobot (external/turtlebot_rosbridge, UTS-RI) plus a /odom subscription. The package is used
               unchanged- The subscription goes through its own _add_subscriber helper and cleanup() removes it.
    RobotIO    wraps a Waffle (or sim_robot.SimRosbridgeRobot, which has the same interface) and converts:
                 /joint_states  wheel angles -> measured (v, omega) and the interval dt between the two readings,
                                from the message time stamps (EncoderOdometry's role in main.py), or /odom's twist
                 /scan          ROS LaserScan -> ranges on the project's 360 beams (cfg.LIDAR_ANGLES), invalid -> inf
                                (expected by perception.process_scan)
                 camera         BGR -> RGB, undistorted with the calibration (hw_config), as perception expects
                 /cmd_vel       (v, omega) clipped to hw_config's limits. A watchdog stops the robot on stale data
               The TurtleBot's firmware runs the wheel speed loops, so (v, omega) are sent as they are. The
               simulation's deadband compensation (model.unicycle_to_wheels) is a MuJoCo actuator model and is
               not used here.
"""
import threading
import time

import cv2
import numpy as np
from rosbridge_api import RosbridgeRobot

from src import config as cfg
from src.control import model
from hardware_integration import hw_config as hw


class Waffle(RosbridgeRobot):
    """RosbridgeRobot with the TurtleBot3's own odometry (/odom, nav_msgs/Odometry) as well."""

    def __init__(self, host, port=hw.PORT, odom_topic="/odom", **kw):
        super().__init__(host, port=port, **kw)
        self.odom_topic = odom_topic
        self._odom, self._odom_lock = None, threading.Lock()

    def setup(self):
        super().setup()
        self._add_subscriber("odom", self.odom_topic, "nav_msgs/msg/Odometry", self._on_odom)

    def _on_odom(self, message):
        with self._odom_lock: # runs on roslibpy's thread
            self._odom = message

    def query_odom(self):
        """The latest /odom message (dict), or None."""
        self._require_connection()
        with self._odom_lock:
            return self._odom


def stamp_seconds(message):
    """ROS 2 header stamp -> seconds (None if the message has none)."""
    try:
        s = message["header"]["stamp"]
        return float(s["sec"]) + 1e-9 * float(s["nanosec"])
    except (KeyError, TypeError):
        return None


def scan_to_beams(scan, angles=cfg.LIDAR_ANGLES):
    """A ROS LaserScan -> ranges on the project's beam angles (nearest scan beam within half an increment) with
    every invalid reading (0, nan, inf, outside [range_min, range_max] or the project's limits) as inf."""
    r = np.asarray(scan.ranges, float)
    ok = np.isfinite(r) & (r >= max(scan.range_min, cfg.LIDAR_RANGE_MIN)) & (r <= min(scan.range_max,
                                                                                    cfg.LIDAR_RANGE_MAX))
    r = np.where(ok, r, np.inf)
    a = np.mod(np.asarray(scan.angles, float), 2 * np.pi)
    out = np.full(len(angles), np.inf)
    if len(a) == 0:
        return out
    order = np.argsort(a)
    a, r = a[order], r[order]
    # nearest scan beam to each project beam, on the circle
    idx = np.searchsorted(a, angles)
    lo, hi = (idx - 1) % len(a), idx % len(a)
    d_lo = np.abs(model.wrap(angles - a[lo]))
    d_hi = np.abs(model.wrap(angles - a[hi]))
    best = np.where(d_lo <= d_hi, lo, hi)
    near = np.minimum(d_lo, d_hi) <= 0.5 * abs(scan.angle_increment) + 1e-6
    out[near] = r[best[near]]
    return out


class RobotIO:
    """Sensor readings and commands for the pipeline, from a RosbridgeRobot-like object (see the module doc)."""

    def __init__(self, robot, odom_source="joint_states"):
        self.robot = robot
        self.odom_source = odom_source
        self.K, self.dist, self.calibrated = hw.load_calibration()
        self._undistort = bool(np.any(self.dist != 0))
        self._prev = None # (stamp, left, right) of the last wheel reading used
        self._prev_odom_stamp = None
        self.last_odom_t = self.last_scan_t = None # wall (or sim) clock of the last fresh data, for the watchdog
        self.stats = {"odometry": 0, "scans": 0, "images": 0, "stale_stops": 0}

    # clock: the real robot runs on the wall clock. The simulated one supplies its own
    def now(self):
        return self.robot.now() if hasattr(self.robot, "now") else time.monotonic()

    def wait_until(self, t):
        if hasattr(self.robot, "wait_until"):
            self.robot.wait_until(t)
        else:
            time.sleep(max(0.0, t - time.monotonic()))

    def read_odometry(self):
        """(u = (v, omega), dt) for the motion since the previous call or None if there is no new reading."""
        if self.odom_source == "odom":
            return self._read_odom_twist()
        js = self.robot.query_joint_states()
        if not js:
            return None
        stamp = stamp_seconds(js)
        try:
            names = list(js["name"])
            pos = js["position"]
            left = hw.WHEEL_SIGNS[0] * float(pos[names.index(hw.WHEEL_JOINTS[0])])
            right = hw.WHEEL_SIGNS[1] * float(pos[names.index(hw.WHEEL_JOINTS[1])])
        except (KeyError, ValueError, IndexError):
            return None
        if stamp is None or (self._prev is not None and stamp <= self._prev[0]):
            return None # no new message
        prev, self._prev = self._prev, (stamp, left, right)
        self.last_odom_t = self.now()
        if prev is None:
            return None
        dt = stamp - prev[0]
        if dt > 1.0: # a gap in the data- do not integrate a guess over it
            return None
        self.stats["odometry"] += 1
        return model.wheels_to_unicycle((left - prev[1]) / dt, (right - prev[2]) / dt), dt

    def _read_odom_twist(self):
        od = self.robot.query_odom()
        if not od:
            return None
        stamp = stamp_seconds(od)
        if stamp is None or (self._prev_odom_stamp is not None and stamp <= self._prev_odom_stamp):
            return None
        prev, self._prev_odom_stamp = self._prev_odom_stamp, stamp
        self.last_odom_t = self.now()
        if prev is None or stamp - prev > 1.0:
            return None
        tw = od["twist"]["twist"]
        self.stats["odometry"] += 1
        return (float(tw["linear"]["x"]), float(tw["angular"]["z"])), stamp - prev

    def read_scan(self):
        """Ranges on the project's 360 beams (inf = no return) from a scan not seen before, or None."""
        scan = self.robot.query_laser_scan(timeout=0, wait_for_new=True)
        if scan is None:
            return None
        self.last_scan_t = self.now()
        self.stats["scans"] += 1
        return scan_to_beams(scan)

    def read_image(self):
        """An undistorted RGB camera frame not seen before, or None."""
        bgr = self.robot.query_image(timeout=0, wait_for_new=True)
        if bgr is None:
            return None
        if (bgr.shape[1], bgr.shape[0]) != hw.CAMERA_SIZE:
            bgr = cv2.resize(bgr, hw.CAMERA_SIZE) # the calibration and perception are for 640x480
        if self._undistort:
            bgr = cv2.undistort(bgr, self.K, self.dist) # keeps K: the pinhole model then holds
        self.stats["images"] += 1
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    def stale(self):
        """True if odometry or LiDAR data is older than hw.WATCHDOG (never received counts as stale)."""
        t = self.now()
        return any(last is None or t - last > hw.WATCHDOG for last in (self.last_odom_t, self.last_scan_t))

    def command(self, v, w):
        v = float(np.clip(v, -hw.V_LIMIT, hw.V_LIMIT))
        w = float(np.clip(w, -hw.W_LIMIT, hw.W_LIMIT))
        self.robot.send_command(v, w)
        return v, w

    def stop(self):
        self.robot.stop()
