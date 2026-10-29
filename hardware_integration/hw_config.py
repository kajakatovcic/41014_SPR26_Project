"""Parameters of the REAL robot and room for the hardware pipeline (hardware_integration/run_hardware.py).

Everything the simulation took from src/config.py that is different, or has to be measured, on the real TurtleBot3
Waffle Pi and its room. Values marked MEASURE are placeholders copied from the simulatio. Measure them before
trusting a run. apply() copies the few values that src/ reads from src.config at call time.

The robot side (external/turtlebot_rosbridge/misc/start_robot.sh): ROS 2 Humble, TURTLEBOT3_MODEL=waffle_pi,
LDS-01 LiDAR on /scan, camera_ros at 640x480 on /camera/image_raw/compressed, /joint_states, /odom, /cmd_vel,
rosbridge on port 9090.
"""
import json
from pathlib import Path

import numpy as np
from machinevisiontoolbox import CentralCamera
from spatialmath import SE3, SO3

from src import config as cfg

HERE = Path(__file__).resolve().parent

# Connection
PORT = 9090
WHEEL_JOINTS = ("wheel_left_joint", "wheel_right_joint") # names in /joint_states (TurtleBot3 bringup)
WHEEL_SIGNS = (1.0, 1.0) # +1 if the joint angle increases when the robot drives FORWARD. check with check_sensors.py

# Markers and room (the robot's only prior knowledge, as in the simulation's explore mission)
ARUCO_DICT = "4x4_50" # @TODO MEASURE: the dictionary the markers were printed from
MARKER_SIDE = 0.1 # m, side of the BLACK square of the printed markers. @TODO MEASURE
MARKER_HEIGHT = 0.1 # m, height of the marker centres above the floor (rejects floor reflections). @TODO MEASURE
REF_MARKER_ID = 1 # the marker the robot localises from: must be on a straight wall >= 1m long (wall-line yaw)
DOCK_ID = 0 # the marker on the charging station
# The dock marker's pose in the reference marker's frame {A}: x along the reference marker's normal (out of the
# wall, into the room), y to the left of it, angle of the dock marker's normal. @TODO MEASURE with a tape: from the
# reference marker's centre, the distance out from its wall and along it to the dock marker's centre.
DOCK_OFFSET = (1.0, -3.5, np.pi / 2)
PREDOCK_DISTANCE = 0.5 # m, robot origin to the dock marker at the pre-dock pose
DOCK_DISTANCE = 0.25 # m, camera to marker when docked (the camera must still decode the marker)
DOCK_TOL = (0.01, 0.01) # m

# Sensor mounts on the robot {R} (x forward, y left, z up, origin on the floor between the wheels). @TODO MEASURE
CAMERA_POS = (0.07, 0.0, 0.107) # m, camera centre (simulation value from the menagerie model)
T_RC = SE3(*CAMERA_POS) * SE3(SO3(np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]])))
T_RS = SE3(-0.049, 0.0, 0.1255) # LiDAR scan origin (simulation value). scan angle 0 = robot forward, CCW positive

# Camera intrinsics: run hardware_integration/calibrate_camera.py, which writes camera_calibration.json (loaded
# below). Until then the simulation's calibration is used: wrong for the real lens.
CAMERA_SIZE = (cfg.CAMERA_WIDTH, cfg.CAMERA_HEIGHT) # camera_ros is started at 640x480
CALIBRATION_FILE = HERE / "camera_calibration.json"
BIAS_PX = 0.0 # corner bias fitted in simulation (0.64px) is a property of the renderer: 0 until fitted on hardware

# Driving limits for the real robot (the Waffle Pi can do 0.26m/s and 1.82rad/s. start gently)
V_NOMINAL = 0.12 # m/s, path-following speed
V_LIMIT = 0.15 # m/s, every command is clipped to these
W_LIMIT = 0.4 # rad/s
CONTROL_DT = 0.05 # s, loop period (20Hz)
RANSAC_ITERS = 150 # wall-fit hypotheses per wall (simulation: 500). With ~350 scan points and walls holding a large
# share of them, 150 still finds each wall almost
#  surely, at under a third of the cost. the scan must fit the loop
WATCHDOG = 0.5 # s, stop if odometry or lidar data is older than this
TURNING_FOR_WALL_YAW = 0.2 # rad/s, use the wall-line heading update only when turning slower than this (the scan
# and the camera frame are up to ~50ms apart on the real robot, and the scan itself takes 0.2sec)


def load_calibration():
    """(K, dist) from camera_calibration.json, or the simulation's pinhole calibration if there is none yet."""
    if CALIBRATION_FILE.exists():
        c = json.loads(CALIBRATION_FILE.read_text())
        return np.array(c["K"], float), np.array(c["dist"], float), True
    K = np.array([[cfg.CAMERA_CAL_F[0], 0, cfg.CAMERA_CAL_PP[0]],
                  [0, cfg.CAMERA_CAL_F[1], cfg.CAMERA_CAL_PP[1]], [0, 0, 1]])
    return K, np.zeros(5), False


def camera_model():
    """The calibrated pinhole camera for perception (images are undistorted first so no distortion here)."""
    K, _, _ = load_calibration()
    return CentralCamera(f=[K[0, 0], K[1, 1]], pp=(K[0, 2], K[1, 2]), imagesize=CAMERA_SIZE, rho=1,
                         name="Waffle Pi camera")


def apply():
    """Copy the values src/ looks up in src.config at call time (marker height and dictionary, dock id)."""
    cfg.MARKER_HEIGHT = MARKER_HEIGHT
    cfg.ARUCO_DICT = ARUCO_DICT
    cfg.DOCK_ID = DOCK_ID
    cfg.REF_MARKER_ID = REF_MARKER_ID
    cfg.DOCK_OFFSET = DOCK_OFFSET
