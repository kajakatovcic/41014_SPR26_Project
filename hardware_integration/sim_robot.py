"""A stand-in for the real robot- the RosbridgeRobot interface backed by the MuJoCo simulation.

It answers in the REAL robot's formats so the hardware pipeline (robot_io, run_hardware) can be tested end to end
before going near the robot:
    query_image        BGR uint8, 640 x 480 (as RosbridgeRobot decodes camera_ros's compressed images), 10 Hz
    query_laser_scan   rosbridge_api.LaserScan as the LDS-01 publishes it: angle_min 0, 1 deg steps, CCW, and 0.0
                       for "no return", 5 Hz
    query_joint_states sensor_msgs/JointState as a dict, with the simulated encoders' odometry errors
    query_odom         nav_msgs/Odometry as a dict (twist from the same encoders, as the TurtleBot's node does)
    send_command       (v, omega) -> the simulated wheels. model.unicycle_to_wheels plays the part of the TurtleBot's
                       firmware speed loop (it compensates the MuJoCo actuators' deadband)
Time is the simulation's: wait_until(t) steps the physics to t, so the loop runs as if in real time however long
its processing takes. 
true_pose() is ground truth for scoring a test run only.
"""
import cv2
import mujoco
import mujoco.viewer
import numpy as np
from rosbridge_api import LaserScan

from src import config as cfg
from src.control import model
from src.simulation import world as sim


class SimRosbridgeRobot:
    CAMERA_PERIOD, SCAN_PERIOD, JOINT_PERIOD = 0.1, 0.2, 0.02

    def __init__(self, host="sim", seed=0, start=cfg.EXPLORE_START, encoder_errors=None, viewer=False,
                 extra_obstacles=(), obstacle_appears=None, **_):
        self.host, self.rng = host, np.random.default_rng(seed)
        self.start, self.encoder_errors, self.viewer = start, encoder_errors, viewer
        self.extra_obstacles, self.obstacle_appears = extra_obstacles, obstacle_appears
        self.is_connected = False

    # connection (as RosbridgeRobot)
    def setup(self):
        self.m = sim.build_world(extra_obstacles=self.extra_obstacles, popup=self.obstacle_appears is not None)
        self.d = mujoco.MjData(self.m)
        sim.set_pose(self.m, self.d, *self.start)
        for _ in range(int(0.5 / self.m.opt.timestep)):
            mujoco.mj_step(self.m, self.d)
        self.t0 = self.d.time
        self.camera = sim.Camera(self.m)
        self.encoders = (sim.Encoders(self.m, self.d) if self.encoder_errors is None
                         else sim.Encoders(self.m, self.d, *self.encoder_errors))
        self.view = mujoco.viewer.launch_passive(self.m, self.d) if self.viewer else None
        self._next = {"image": 0.0, "scan": 0.0, "joints": 0.0}
        self._joint_msg = self._odom_msg = None
        self._prev_wheels = None
        self._odom_pose = np.zeros(3)
        self.popup_xy = None
        self.is_connected = True

    def cleanup(self):
        if self.is_connected:
            self.stop()
            self.camera.close()
            if self.view is not None:
                self.view.close()
        self.is_connected = False

    def __enter__(self):
        self.setup()
        return self

    def __exit__(self, *exc):
        self.cleanup()

    # clock
    def now(self):
        return self.d.time - self.t0

    def wait_until(self, t):
        while self.now() < t - 1e-9:
            mujoco.mj_step(self.m, self.d)
            if self.obstacle_appears is not None and self.popup_xy is None and self.now() >= self.obstacle_appears[0]:
                x = sim.true_pose(self.m, self.d)
                self.popup_xy = x[:2] + self.obstacle_appears[1] * np.array([np.cos(x[2]), np.sin(x[2])])
                sim.place_popup(self.m, self.d, self.popup_xy)
            if self.now() >= self._next["joints"]:
                self._publish_joints()
                self._next["joints"] += self.JOINT_PERIOD
        if self.view is not None and self.view.is_running():
            self.view.sync()

    # topics
    def _stamp(self):
        s = self.now()
        return {"sec": int(s), "nanosec": int(round((s - int(s)) * 1e9))}

    def _publish_joints(self):
        q = self.encoders.read(self.d, self.rng)
        header = {"stamp": self._stamp(), "frame_id": ""}
        self._joint_msg = {"header": header, "name": ["wheel_left_joint", "wheel_right_joint"],
                           "position": [float(q[0]), float(q[1])], "velocity": [], "effort": []}
        if self._prev_wheels is not None:
            v, w = model.wheels_to_unicycle(*((q - self._prev_wheels) / self.JOINT_PERIOD))
            self._odom_pose = model.f_uni(self._odom_pose, (v, w), self.JOINT_PERIOD) # TurtleBot-style dead reckoning
            x, y, th = self._odom_pose
            self._odom_msg = {"header": header,
                              "pose": {"pose": {"position": {"x": x, "y": y, "z": 0.0},
                                                "orientation": {"x": 0.0, "y": 0.0, "z": np.sin(th / 2),
                                                                "w": np.cos(th / 2)}}},
                              "twist": {"twist": {"linear": {"x": v, "y": 0.0, "z": 0.0},
                                                  "angular": {"x": 0.0, "y": 0.0, "z": w}}}}
        self._prev_wheels = q.copy()

    def query_joint_states(self):
        return self._joint_msg

    def query_odom(self):
        return self._odom_msg

    def query_laser_scan(self, timeout=None, *, wait_for_new=False):
        if self.now() < self._next["scan"]:
            return None
        self._next["scan"] = self.now() + self.SCAN_PERIOD
        rho = sim.read_lidar(self.m, self.d, self.rng)
        rho = np.where(np.isfinite(rho), rho, 0.0).astype(np.float32) # the LDS-01 reports "no return" as 0
        angles = (np.arange(len(rho)) * (2 * np.pi / len(rho))).astype(np.float32)
        return LaserScan(angles=angles, ranges=rho, intensities=np.zeros_like(rho), angle_min=0.0,
                         angle_max=float(angles[-1]), angle_increment=float(2 * np.pi / len(rho)),
                         time_increment=0.0, scan_time=self.SCAN_PERIOD, range_min=cfg.LIDAR_RANGE_MIN,
                         range_max=cfg.LIDAR_RANGE_MAX)

    def query_image(self, timeout=None, *, wait_for_new=False):
        if self.now() < self._next["image"]:
            return None
        self._next["image"] = self.now() + self.CAMERA_PERIOD
        return cv2.cvtColor(self.camera.render(self.d, self.rng), cv2.COLOR_RGB2BGR)

    def send_command(self, linear=0.0, angular=0.0):
        self.d.ctrl[:] = model.unicycle_to_wheels(linear, angular)

    def stop(self):
        self.send_command(0.0, 0.0)

    # ground truth: scoring a test run only
    def true_pose(self):
        return sim.true_pose(self.m, self.d)

    def collisions(self):
        return sim.robot_collisions(self.m, self.d)
