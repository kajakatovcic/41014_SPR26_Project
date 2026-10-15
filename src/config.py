"""Parameters shared by the simulation and the perception/estimation/control subsystems

Frames (as following the tutorials):
    {W} world: x, y on the floor, z up
    {R} robot (TurtleBot3 base): x forward, y left, z up, origin on the floor between the wheels
    {S} lidar: same axes as {R}, at the LDS-01 scan origin (Week 4 Part 5)
    {C} camera: z forward (optical axis), x right, y down (Week 3 + Week 9 D3)
    {M} ArUco marker: x right and y up when facing the marker and z out of its face (Week 9 tag_corners)
"""
from pathlib import Path

import numpy as np
from spatialmath import SE3, SO3

ROOT = Path(__file__).resolve().parents[1]
TB3_SCENE_XML = ROOT / "robotis_mujoco_menagerie" / "robotis_tb3" / "scene_turtlebot3_waffle_pi.xml"

# Robot (TurtleBot3 Waffle Pi)
WHEEL_RADIUS = 0.033 # m
WHEEL_TRACK = 0.288 # m, distance between the wheels
WHEEL_SPEED_MAX = 7.88 # rad/s, actuator limit in the model (0.26m/s)
ENCODER_TICKS = 4096  # per wheel revolution (Dynamixel XM430)
ROBOT_RADIUS = 0.25 # m, circle around the robot origin that contains the chassis (x -0.197..0.086, y +/-0.153)
CONTROL_DT = 0.05 # sec, control and estimation period (20Hz) - under half the wheel actuators' 0.11sec time constant

# Odometry errors injected by the SIMULATION's encoder driver (the robot's software does not know these values).
# The MuJoCo wheels roll without slipping so without them dead reckoning would be unrealistically perfect.
#   systematic: each wheel's effective radius differs from the nominal WHEEL_RADIUS the software assumes (tyre
#               wear, load, manufacturing) by a different amount left and right causing heading drift
#   random:     slip, a random walk on each wheel angle whose variance grows with the angle turned
SIM_WHEEL_SCALE_ERROR = (+0.008, -0.004) # (left, right) fractional radius error, typical 0.1-1 % (UMBmark)
SIM_WHEEL_SLIP_SIGMA = 0.02 # rad per sqrt(rad turned)

# EKF (criterion C). What the FILTER assumes about odometry and its starting point. This is tuned in tests/test_estimation.py.
EKF_WHEEL_SCALE_SIGMA = 0.01 # fractional uncertainty of each wheel's effective radius
EKF_WHEEL_SLIP_SIGMA = 0.02 # rad per sqrt(rad turned)
# Noise scales, tuned by NEES over repeated missions (tests/test_estimation.py, Week 7 D.3/D.4: "pick Q_d by NEES,
# and if you cannot settle it, err high"). With both at 1 the filter was overconfident (NEES 6.7 against 3):
#   Q x3: the wheel-radius error is a constant bias but the noise model treats it as fresh noise each step, so
#         its variance grows like N instead of N^2 over a stretch without markers
#   R range x5: consecutive camera frames see the same marker with nearly the same RANGE error (corner bias and
#         calibration repeat frame to frame. the innovations' autocorrelation is +0.31 one frame apart), so the
#         10Hz range measurements are not independent. Errors correlated over ~n frames carry about 1/n of the
#         information each: R_eff ~ n R. Bearing errors are uncorrelated (+0.02) so bearing is left alone.
EKF_Q_SCALE = 3.0 # overall scale on the odometry noise
EKF_R_SCALE = (5.0, 1.0) # scales on perception's per-frame (range, bearing) variances
EKF_P0 = (0.05, 0.05, np.deg2rad(3.0)) # prior std devs on the start pose (x [m], y [m], theta [rad])
EKF_GATE = 11.83 # chi-square 2 dof, 99.7 %: reject marker measurements beyond this NIS

# 2D lidar (LDS-01)
LIDAR_NUM_RAYS = 360
LIDAR_ANGLES = np.deg2rad(np.arange(LIDAR_NUM_RAYS) * 360.0 / LIDAR_NUM_RAYS)  # CCW from forward
LIDAR_RANGE_MIN, LIDAR_RANGE_MAX = 0.12, 3.5   # m
LIDAR_SIGMA = 0.01 # m, range noise std dev (Wk4 Part 5)
LIDAR_RATE = 5.0 # Hz
T_RS = SE3(-0.049, 0.0, 0.1255) # lidar in the robot frame at the model's LDS mesh

# RGB camera (pinhole, Raspberry Pi camera v2 FOV)
CAMERA_WIDTH, CAMERA_HEIGHT = 640, 480
CAMERA_FOVY = 48.8 # deg, vertical FOV
CAMERA_RATE = 10.0 # Hz
CAMERA_NOISE = 2.0  # grey levels, std dev of the Gaussian pixel noise added in simulation
CAMERA_POS = (0.0855, 0.0, 0.107) # camera centre in the robot frame - front of the model's camera housing
R_RC = SO3(np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]]))   # robot (x fwd, z up) -> camera (z fwd, y down) from wk9
T_RC = SE3(*CAMERA_POS) * SE3(R_RC)

# True intrinsics implied by the simulated camera: GROUND TRUTH, for evaluating the calibration only. MuJoCo's
# principal point is the image centre. In pixel coordinates with pixel centres at integers (OpenCV) that is
# ((W - 1) / 2, (H - 1) / 2). MuJoCo renders an ideal pinhole camera, so there is no lens distortion.
CAMERA_F_PX = (CAMERA_HEIGHT / 2) / np.tan(np.deg2rad(CAMERA_FOVY) / 2)
CAMERA_PP = ((CAMERA_WIDTH - 1) / 2, (CAMERA_HEIGHT - 1) / 2)

# Calibrated intrinsics used by perception (Week 3 Part B). pinhole model w/ 30 checkerboard views, from
# tests/test_calibration.py (seed 0, reprojection RMS 0.11px). Rerun that script and update these if the camera
# changes. The full distortion model fits no better and its parameters are unstable, so lens distortion is not
# modelled.
CAMERA_CAL_F = (529.71, 529.88) # (f/rho_u, f/rho_v) [px]
CAMERA_CAL_PP = (319.32, 239.22) # (u0, v0) [px]

# Checkerboard calibration target (Week 3 Part B). 10 x 7 squares therefore 9 x 6 internal corners. One odd and one even
# dimension makes the corner order unambiguous.
CHECKERBOARD_GRID = (9, 6) # internal corners (columns, rows)
CHECKERBOARD_SQUARE = 0.04 # m

# ArUco markers
ARUCO_DICT = "4x4_50"
MARKER_SIDE = 0.15 # m, side of the black square
MARKER_HEIGHT = 0.13 # m, height of the marker centre above the floor
DOCK_ID = 0 # marker on the charging station


def marker_pose(x, y, facing, z=MARKER_HEIGHT, standoff=0.005):
    """World pose of a vertical marker mounted on a surface at (x, y), facing along the yaw angle `facing`.
    The marker stands `standoff` proud of the surface so the renderer does not mix the two up (z-fighting)."""
    n = np.array([np.cos(facing), np.sin(facing), 0.0]) # marker z: out of the face
    up = np.array([0.0, 0.0, 1.0]) # marker y: up
    return SE3.Rt(np.column_stack([np.cross(up, n), up, n]), [x + standoff * n[0], y + standoff * n[1], z])


# Room: walls at +/- ROOM_HALF, obstacles, charging station + landmark markers (the known map)
ROOM_HALF = 2.0 # m, inner faces of the walls are at +/- (ROOM_HALF - WALL_THICK)
WALL_THICK = 0.05 # m, half thickness
WALL_HEIGHT = 0.3 # m
_IN = ROOM_HALF - WALL_THICK

DOCK_SIZE = (0.05, 0.15, 0.125) # half sizes of the charging station box
DOCK_POS = (_IN - DOCK_SIZE[0], 1.0) # against the +x wall, facing -x
MARKERS = {
    DOCK_ID: marker_pose(DOCK_POS[0] - DOCK_SIZE[0], DOCK_POS[1], np.pi),
    1: marker_pose(-_IN, -0.8, 0.0),
    2: marker_pose(0.5, _IN, -np.pi / 2),
    3: marker_pose(-0.5, -_IN, np.pi / 2),
    4: marker_pose(_IN, -1.0, np.pi),
}

OBSTACLES = [   # name, kind, centre (x, y), size (box half sizes or cylinder radius), height
    ("box_red", "box", (0.6, 0.5), (0.15, 0.15), 0.3),
    ("cyl_green", "cylinder", (0.4, -0.9), 0.12, 0.4),
    ("box_blue", "box", (-0.9, 0.9), (0.25, 0.1), 0.4),
    ("cyl_yellow", "cylinder", (-0.7, -0.3), 0.08, 0.4),
]

# MISSION: start pose and waypoints. Routed so the robot (ROBOT_RADIUS) clears every obstacle and wall by >= 0.14m.
# The last waypoint is the pre-dock pose, on the dock marker's normal with the camera ~0.6 m from the marker, where
# the robot turns to face the station and switches to visual docking.
START_POSE = (-1.5, -1.5, 0.0) # x, y [m], theta [rad]: the estimator's prior; the robot is placed here
WAYPOINTS = [(1.2, -1.45), (1.2, 0.2), (1.1, 1.0)]
WAYPOINT_TOL = 0.15 # m, switch to the next waypoint inside this distance (placeholder WaypointFollower)
GOAL_TOL = 0.05 # m, the last waypoint counts as reached inside this distance (placeholder WaypointFollower)

# Docking (Week 9 D3)- stop with the camera DOCK_DISTANCE in front of the dock marker, centred on it. The camera can
# only decode the marker while its 0.2m plate fits in the 480 x image height: Z >= 0.2m x 530px / 480px = 0.22m
DOCK_DISTANCE = 0.25 # m, camera to marker when docked
DOCK_TOL = (0.01, 0.01) # m, docked when |depth - DOCK_DISTANCE| and |lateral offset| are both inside these