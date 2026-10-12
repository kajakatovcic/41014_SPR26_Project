"""Camera calibration from checkerboard images (Week 3 Parts B & C).

Works on images from any source (the simulated camera or the real one), so the same procedure calibrates both.

    calibrate()             Zhang's multi-plane method (same OpenCV calls as CentralCamera.images2C) (Week 3 B),
                            with the option of a pinhole model without lens distortion
    reprojection_errors()   per-image reprojection error (Week 3 C check of whether a calibration is any good)
"""
from dataclasses import dataclass

import cv2
import numpy as np
from machinevisiontoolbox import CentralCamera
from spatialmath import SE3

from src import config as cfg

_SUBPIX = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)


def board_points(gridshape=cfg.CHECKERBOARD_GRID, square=cfg.CHECKERBOARD_SQUARE):
    """Internal corners in the target frame, N x 3 in OpenCV's order - the plane Z = 0, origin at the first corner
    found, x along the gridshape[0] direction (Week 3 Part C - P_board)."""
    P = np.zeros((gridshape[0] * gridshape[1], 3))
    P[:, :2] = np.mgrid[0:gridshape[0], 0:gridshape[1]].T.reshape(-1, 2) * square
    return P


def measured_corners(rgb, gridshape=cfg.CHECKERBOARD_GRID):
    """Sub-pixel checkerboard corners as an N x 2 array, or None if the board is not found (Week 3 Part C)."""
    grey = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY) if rgb.ndim == 3 else rgb
    ok, c = cv2.findChessboardCorners(grey, gridshape)
    if not ok:
        return None
    return cv2.cornerSubPix(grey, c, (11, 11), (-1, -1), _SUBPIX).reshape(-1, 2)


@dataclass
class Calibration:
    K: np.ndarray # 3 x 3 intrinsic matrix
    dist: np.ndarray # (k1, k2, p1, p2, k3), all zero for the pinhole model
    poses: list # camera pose in the target frame for each image used, SE3 (as images2C frames)
    corners: list # measured corners for each image used, N x 2
    ids: list # index of each image used in the input list
    rms: float # overall reprojection RMS [px] reported by OpenCV

    def camera(self):
        """The calibrated camera as a CentralCamera (pinhole part only)."""
        return CentralCamera(f=[self.K[0, 0], self.K[1, 1]], pp=(self.K[0, 2], self.K[1, 2]),
                             imagesize=(cfg.CAMERA_WIDTH, cfg.CAMERA_HEIGHT), rho=1)


def calibrate(images, distortion=True, gridshape=cfg.CHECKERBOARD_GRID, square=cfg.CHECKERBOARD_SQUARE):
    """Calibrate from a list of RGB or grey images of the target.

    distortion=True estimates the full (k1, k2, p1, p2, k3) model as CentralCamera.images2C does in Week 3
    distortion=False fixes them at zero - a pure pinhole model with only f/rho_u, f/rho_v, u0 and v0 to estimate.
    Images where the board is not found are skipped. Always check how many were used (Week 3 B1)."""
    P = board_points(gridshape, square).astype(np.float32)
    obj, img, ids, corners = [], [], [], []
    for i, im in enumerate(images):
        c = measured_corners(im, gridshape)
        if c is not None:
            obj.append(P); img.append(c.astype(np.float32).reshape(-1, 1, 2)); ids.append(i); corners.append(c)
    if len(obj) < 3:
        raise ValueError(f"only {len(obj)} images with a detectable board. Need at least 3")

    flags = 0 if distortion else (cv2.CALIB_ZERO_TANGENT_DIST | cv2.CALIB_FIX_K1 | cv2.CALIB_FIX_K2
                                  | cv2.CALIB_FIX_K3)
    size = (images[0].shape[1], images[0].shape[0])
    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(obj, img, size, None, None, flags=flags)
    poses = [SE3(SE3(t.flatten()) * SE3.EulerVec(r.flatten())).inv() for r, t in zip(rvecs, tvecs)]  # as images2C
    return Calibration(K, dist.flatten(), poses, corners, ids, rms)


def reprojection_errors(cal, gridshape=cfg.CHECKERBOARD_GRID, square=cfg.CHECKERBOARD_SQUARE):
    """Week 3 Part C - project the known board corners through the calibrated model, using the pose calibration
    recovered for each image, and compare with where the corners were measured. Returns one N-vector of errors
    [px] per image and the residual vectors (measured - projected) - N x 2 per image."""
    P = board_points(gridshape, square)
    errors, residuals = [], []
    for T_BC, c in zip(cal.poses, cal.corners):
        T_CB = T_BC.inv()
        rvec = cv2.Rodrigues(T_CB.R)[0]
        proj, _ = cv2.projectPoints(P, rvec, T_CB.t, cal.K, cal.dist) # applies the distortion model too
        r = c - proj.reshape(-1, 2)
        residuals.append(r)
        errors.append(np.linalg.norm(r, axis=1))
    return errors, residuals
