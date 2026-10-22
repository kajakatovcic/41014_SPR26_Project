"""Perception for Track A.
Incorporates the 2D LiDAR and RGB camera measurements for the estimator and controller.

Everything here works on sensor data only (lidar ranges, camera images) and never on simulator state.
This way it can be pointed at the real TurtleBot3 without making any changes.

Tutorial work used in this subsystem: 
    Lidar   Week 4: Range-bearing model, frames with SE3, voxel + outlier filtering, RANSAC, DBSCAN
    Camera  Week 3: CentralCamera, ArUco detection and PnP pose.
            Week 7: Range and bearing to a known marker.
            Week 9: Tag position in the camera frame for docking
"""
from dataclasses import dataclass

import numpy as np
import open3d as o3d
from machinevisiontoolbox import CentralCamera, Image
from spatialmath import SE3

from src import config as cfg


def pose_to_SE3(x):
    """Planar pose (x, y, theta) -> SE3, e.g. the EKF estimate of the robot in the world frame."""
    return SE3(x[0], x[1], 0) * SE3.Rz(x[2])


def wrap(a):
    """Wrap an angle back into (-pi, pi]."""
    return (a + np.pi) % (2 * np.pi) - np.pi

# Lidar (Week 4)

# Unit beam directions in the sensor frame - Week 4 Part 5 with elevation angle at zero
LIDAR_DIRS_S = np.stack([np.cos(cfg.LIDAR_ANGLES), np.sin(cfg.LIDAR_ANGLES), np.zeros(cfg.LIDAR_NUM_RAYS)], axis=1)


def lidar_points(rho, T_RS=cfg.T_RS):
    """Scan ranges -> N x 3 points in the robot frame {R}, keeping only valid returns."""
    ok = np.isfinite(rho)
    P_S = LIDAR_DIRS_S[ok] * rho[ok, None]
    return (T_RS * P_S.T).T


def clean_cloud(P, voxel=0.02, remove_outliers=False, nb_neighbors=5, std_ratio=2.0):
    """
    Voxel downsampling. Optionally followed by statistical outlier removal. (from wk4 part 3)

    Outlier removel is off by default for the LiDAR. In wk4 it removes the flying pixels of RGBD and stereo.
    SInce the LiDAR has constant 1cm range noise and no flying points, and its beams thin out with range (1deg is
    6cm at 3.5m), the filter would mainly delete real returns from distant obstacles.
    Using >300 random turtlebot poses (tests/test_perception.py) it detectd 92.5% of visible obstacles with the 
    filter and 96.9% without. With no false clusters eithe rway .
    """
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P)).voxel_down_sample(voxel)
    if remove_outliers:
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    return np.asarray(pcd.points)


def ransac_line(P, thresh, iters, rng):
    """
    Uses RANSAC with a line model unlike wk4 which uses a plane. 
    Samples 2 points, fits the line through them, counters inliers within the threshold, keeps the best 
    hypothesis and refits it to its inliers.
    Returns the line (a,b,c) of ax + by + c = 0 with a unit normal and the inlier mask.
    """
    best = np.zeros(len(P), bool)
    for _ in range(iters):
        p, q = P[rng.choice(len(P), 2, replace=False)]
        n = np.array([-(q - p)[1], (q - p)[0]])
        if np.linalg.norm(n) < 1e-9:
            continue
        n /= np.linalg.norm(n)
        inl = np.abs((P - p) @ n) < thresh
        if inl.sum() > best.sum():
            best = inl
    c = P[best].mean(axis=0)
    n = np.linalg.svd(P[best] - c)[2][1] # normal = direction of least spread
    return np.array([n[0], n[1], -n @ c]), best


@dataclass
class Obstacle:
    centroid: np.ndarray # (x, y) of the visible surface, robot frame
    extent: np.ndarray # (dx, dy) of the visible surface
    points: np.ndarray # M x 2


@dataclass
class LidarResult:
    points: np.ndarray # N x 2, every valid return in the robot frame
    walls: list # [(line (a, b, c), inlier points M x 2)], robot frame
    obstacles: list # [Obstacle]


def process_scan(rho, rng, wall_thresh=0.02, wall_min_points=40, wall_min_length=1.0, eps=0.1, min_points=4,
                 T_RS=cfg.T_RS, ransac_iters=500):
    """
    SPlit a scan into walls (sequential RANSAC lines) + obstacles (DBSCAN on what is left), robot frame. 

    wall_thresh is 2sigma of the range noise. DBSCAN parameters follow the sensor (seen in wk4 5.2). 'eps' must
    exceed the beam spacing at max range (6cm at 3.5m) plus noise and a 16cm cylinder 1.5m away returns only ~6 
    points, so min_points is small. It is tuned in test/tests_perception.py.
    A flat obstacle face close to the robot also returns many collinear points so a line only counts as a wall if
    its inliers span wall_min_length (longer than any obstacle face). Shorter lines go back to the obstacle points.
    T_RS is the LiDAR's mount on the robot (the real robot's can differ from the simulated one)- ransac_iters the
    RANSAC hypotheses per wall (fewer on the real robot, where the scan must be processed in real time).
    """
    P_R = lidar_points(rho, T_RS)
    P = clean_cloud(P_R)[:, :2]

    walls, rest, not_walls = [], P, []
    while len(rest) > wall_min_points:
        line, inl = ransac_line(rest, wall_thresh, ransac_iters, rng)
        if inl.sum() < wall_min_points:
            break
        # The inliers of an infinite line can include a few points where other surfaces cross it, so keep only
        # the longest unbroken run of points along the line as the wall segment
        idx = np.flatnonzero(inl)
        along = rest[idx] @ np.array([-line[1], line[0]]) # position along the line
        order = np.argsort(along)
        runs = np.split(order, np.flatnonzero(np.diff(along[order]) > 0.2) + 1)
        seg = idx[max(runs, key=lambda r: np.ptp(along[r]))]
        if np.ptp(rest[seg] @ np.array([-line[1], line[0]])) >= wall_min_length and len(seg) >= wall_min_points:
            walls.append((line, rest[seg]))
        else:
            not_walls.append(rest[seg])
        keep = np.ones(len(rest), bool)
        keep[seg] = False
        rest = rest[keep]
    rest = np.vstack([rest] + not_walls)

    obstacles = []
    if len(rest):
        cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.column_stack([rest, np.zeros(len(rest))])))
        labels = np.array(cloud.cluster_dbscan(eps=eps, min_points=min_points))
        for i in range(labels.max() + 1):
            p = rest[labels == i]
            obstacles.append(Obstacle(p.mean(axis=0), p.max(axis=0) - p.min(axis=0), p))
    return LidarResult(P_R[:, :2], walls, obstacles)


def to_world(P_R, x_hat):
    """Points in the robot frame -> world frame, using the ESTIMATED robot pose (x, y, theta)."""
    P = np.column_stack([P_R, np.zeros(len(P_R))]) if P_R.shape[1] == 2 else P_R
    return (pose_to_SE3(x_hat) * P.T).T[:, :P_R.shape[1]]


# Camera (Weeks 3, 7, 9)

# Calibrated intrinsics (Week 3 Part B - tests/test_calibration.py): pinhole model with no lens distortion
CAMERA = CentralCamera(f=list(cfg.CAMERA_CAL_F), imagesize=(cfg.CAMERA_WIDTH, cfg.CAMERA_HEIGHT),
                       pp=cfg.CAMERA_CAL_PP, rho=1, name="robot camera")

# Corner localisation model for the Waffle Pi camera with the calibrated intrinsics as fitted by
# tests/test_perception.py over seeds 0-4 (2500 poses, ~2200 detections).
# The detected corners sit BIAS_PX inside the true marker outline on average (per-seed fits 0.60-0.68px). The
# scatter is fitted separately for range (from the apparent size so it absorbs PnP errors at oblique vies - fits
# 0.40-0.45px) and bearing (from the image position of the marker centre - fits 0.21-0.25px), rounded up slightly
BIAS_PX = 0.64
SIGMA_RANGE_PX = 0.45
SIGMA_BEARING_PX = 0.24

# Calibration uncertainty of the principal point u0 [px]. tests/test_calibration.py Part 2 gives a spread of
# 0.06-0.13 px between independent calibrations and the calibration in use is 0.18px out, so 0.2px. An error in u0
# shifts EVERY bearing by the same angle: 0.18 px biases bearings by 0.02 deg, as large as the bearing noise itself. The truth is unknown
# on a real robot so the bias cannot be removed. Instead it is added to the bearing variance. This is derived from
# week 7 - when in doubt, inflate the noise rather than trust an overconfident filter.
SIGMA_PP_PX = 0.2


@dataclass
class MarkerObservation:
    id: int
    T_CM: SE3 # marker pose in the camera frame from PnP
    T_RM: SE3 # marker pose in the robot frame
    z: np.ndarray # [range, bearing] from the robot origin to the marker centre (WK7, h_rb)
    R: np.ndarray # 2 x 2 measurement covariance of z
    size_px: float # apparent side length in the image [px]
    corners: np.ndarray # 2 x 4 image corners [px]


def range_bearing(T_RM):
    """Range and bearing of the marker centre in the robot frame, measured in the floor plane (from wk7 C.4)."""
    x, y = T_RM.t[0], T_RM.t[1]
    return np.array([np.hypot(x, y), np.arctan2(y, x)])


def measurement_covariance(Z, f=CAMERA.K[0, 0], side=cfg.MARKER_SIDE,
                           sigma_range_px=SIGMA_RANGE_PX, sigma_bearing_px=SIGMA_BEARING_PX, sigma_pp_px=SIGMA_PP_PX):
    """Noise model for z = [range, bearing] (Week 3):
         range comes from apparent size, sigma_Z ~ Z^2 sigma_px / (f s) so it grows with the SQUARE of the range.
         bearing comes from the image position of the marker centre, sigma_alpha ~ sigma_px / f, constant, plus the
         principal point's calibration uncertainty, which acts on the bearing in the same way."""
    return np.diag([(Z ** 2 * sigma_range_px / (f * side)) ** 2, (sigma_bearing_px ** 2 + sigma_pp_px ** 2) / f ** 2])


def correct_range(T_CM, f, side, bias_px):
    """Remove the range bias caused by corners detected bias_px inside the marker outline. The apparent size is
    n = f s / Z, so the measured depth Z' = f s / (n - bias). 
    Scaling the position along the viewing ray leaves the bearing untouched."""
    Z_meas = T_CM.t[2]
    Z = f * side / (f * side / Z_meas + bias_px)
    return SE3.Rt(T_CM.R, T_CM.t * Z / Z_meas)


def detect_markers(rgb, camera=CAMERA, side=cfg.MARKER_SIDE, T_RC=cfg.T_RC, bias_px=BIAS_PX,
                   known=cfg.MARKERS, height_tol=0.1):
    """Detects ArUco markers and measure each one's pose (wk7 part E) and expresses it in the robot frame.

    Detections are checked against the known map (wk7). The id must be in it and the marker centre must be at its
    mapped height (within height_tol). The second check rejects reflections of markers in a shiny floor which appear
    below the flow and can decode as a valid id - i.e. a mirrored marker ca nread as a different code.
    known=None (the explore mission w/ no map): any id is accepted, at the markers' mounting height cfg.MARKER_HEIGHT.

    Uses the marker POSITION (z=range, bearing) for EKF. The orientation from PnP suffers from the planar pose 
    ambiguity discussed in wk3 (part E) - for a small marker seen nearly face on, it can flip by ~10deg."""
    out = []
    f = camera.K[0, 0]
    for fid in Image(rgb).mono().fiducial(dict=cfg.ARUCO_DICT, K=camera.K, side=side):
        if known is not None and int(fid.id) not in known:
            continue
        T_CM = correct_range(fid.pose, f, side, bias_px)
        T_RM = T_RC * T_CM
        height = cfg.MARKER_HEIGHT if known is None else known[int(fid.id)].t[2]
        if abs(T_RM.t[2] - height) > height_tol:
            continue
        corners = np.asarray(fid.corners)
        size_px = np.mean(np.linalg.norm(corners - np.roll(corners, 1, axis=1), axis=0))
        out.append(MarkerObservation(int(fid.id), T_CM, T_RM, range_bearing(T_RM),
                                     measurement_covariance(T_CM.t[2], f, side),
                                     size_px, corners))
    return out


def docking_error(obs):
    """
    Tag position in the camera frame (x_M lateral, z_M depth), the inputs of the docking law (wk9 D3), and the
    angle between the robot's heading and the marker's face normal (zero when square on).
    """
    n = obs.T_RM.R[:, 2] # marker z axis (out of the face) in {R}
    return obs.T_CM.t[0], obs.T_CM.t[2], wrap(np.arctan2(-n[1], -n[0]))
