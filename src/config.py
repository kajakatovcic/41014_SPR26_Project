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

# Nominal intrinsics implied by the simulated camera. MuJoCo's principal point is the image centre. In pixel
# coordinates with pixel centres at integers (OpenCV) that is ((W - 1) / 2, (H - 1) / 2).
CAMERA_F_PX = (CAMERA_HEIGHT / 2) / np.tan(np.deg2rad(CAMERA_FOVY) / 2)
CAMERA_PP = ((CAMERA_WIDTH - 1) / 2, (CAMERA_HEIGHT - 1) / 2)

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
