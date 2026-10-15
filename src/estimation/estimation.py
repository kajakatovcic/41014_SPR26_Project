"""Estimation for Track A: EKF fusing wheel-encoder odometry with range-bearing measurements of ArUco markers.

State:   x = [x, y, theta]   robot pose in the world frame {W} (origin = robot frame {R} origin)
Input:   wheel angle increments from the encoders (predict step)
Measure: z = [range, bearing] from the robot origin to a known marker (update step), from
         perception.detect_markers (MarkerObservation.z and .R). The camera offset T_RC is already applied there,
         so h(x) below must NOT include a camera offset.

Only sensor data goes in here. Ground truth (world.true_pose) is NEVER used by this module. It is for evaluation only.

Sections (see the work breakdown):
    S2  odometry characterisation   -> tests/odometry_characterisation.py (sets K_SLIP and SIGMA_FLOOR)
    S3  process model, F, G, Q      -> EKF.predict, motion_model, motion_jacobians
    S4  measurement model, H        -> measurement_model, measurement_jacobian
    S5  validate R                  -> tests/validate_R.py (TODO, not in this file)
    S6  EKF implementation          -> EKF
    S7-S11 tests, tuning, reporting -> separate scripts that use this module
"""
import numpy as np

from src import config as cfg
from src.perception.perception import wrap


ODOM_RADIUS_SCALE = 0.9978   # straight-line test: odometry over-read distance by 0.22%
ODOM_TRACK_SCALE = 1.0024    # remaining heading over-read (~0.24%) after the radius fix
RADIUS = -0.4

R_W = cfg.WHEEL_RADIUS * ODOM_RADIUS_SCALE
B = cfg.WHEEL_TRACK * ODOM_TRACK_SCALE
# R_W = cfg.WHEEL_RADIUS
# B = cfg.WHEEL_TRACK

# Odometry noise (Section 2 sets these from data). Per-wheel travel noise: sigma = K_SLIP * |ds| + SIGMA_FLOOR
"""Set K_SLIP to cover the bias with margin: around 0.01, which covers the 0.25% straight error and the roughly 1% 
    turning error. Round up a bit rather than down. You can justify it in the report as “measured systematic error, 
    inflated to absorb unmodelled slip”."""
K_SLIP = 0.03  # TODO: replace with the value measured in odometry_characterisation.py
SIGMA_FLOOR = 2e-5  # m, about the encoder quantisation (2*pi/ENCODER_TICKS * R_W / sqrt(12))

# 99% chi-square gate for 2 degrees of freedom (range, bearing)
GATE_CHI2 = 9.21


# Encoder bookkeeping

class EncoderOdometry:
    """Turns raw encoder angles (world.read_encoders) into wheel angle increments.

    The first reading only initialises the reference, so an arbitrary encoder offset at start-up does not matter.
    TODO (Section 1): check the wheel joint signs in the model. If driving forward gives a negative increment on a
    wheel, flip SIGN for that wheel.
    """
    SIGN = np.array([1.0, 1.0])  # [left, right] both possitive as as proven in WHEEL_test.py

    def __init__(self):
        self.prev = None

    def increments(self, enc):
        """enc = [left, right] wheel angles [rad]. Returns (dphi_L, dphi_R) [rad]."""
        enc = np.asarray(enc, float)
        if self.prev is None:
            self.prev = enc
            return 0.0, 0.0
        d = (enc - self.prev) * self.SIGN
        self.prev = enc
        return d[0], d[1]


# Section 3: process model

def motion_model(x, dphi_L, dphi_R):
    """Midpoint unicycle model. Returns the predicted state and the intermediate (ds, dth, theta_mid)."""
    ds_R, ds_L = R_W * dphi_R, R_W * dphi_L
    ds = 0.5 * (ds_R + ds_L)
    dth = (ds_R - ds_L) / B
    th_m = x[2] + 0.5 * dth
    x_new = np.array([x[0] + ds * np.cos(th_m),
                      x[1] + ds * np.sin(th_m),
                      wrap(x[2] + dth)])
    return x_new, ds, dth, th_m


def motion_jacobians(x, dphi_L, dphi_R):
    """F = df/dx (3x3) and G = df/du (3x2) with u = [ds_R, ds_L]."""
    _, ds, _, th_m = motion_model(x, dphi_L, dphi_R)
    s, c = np.sin(th_m), np.cos(th_m)
    F = np.array([[1, 0, -ds * s],
                  [0, 1, ds * c],
                  [0, 0, 1]])
    G = np.array([[0.5 * c - ds / (2 * B) * s, 0.5 * c + ds / (2 * B) * s],
                  [0.5 * s + ds / (2 * B) * c, 0.5 * s - ds / (2 * B) * c],
                  [1 / B, -1 / B]])
    return F, G


def wheel_noise_cov(dphi_L, dphi_R, k=K_SLIP, floor=SIGMA_FLOOR):
    """Sigma_u: covariance of u = [ds_R, ds_L]. Noise grows with distance travelled (slip, radius error)."""
    ds_R, ds_L = R_W * dphi_R, R_W * dphi_L
    return np.diag([(k * abs(ds_R)) ** 2 + floor ** 2,
                    (k * abs(ds_L)) ** 2 + floor ** 2])


# Section 4: measurement model

def measurement_model(x, marker_xy):
    """Predicted [range, bearing] from the robot origin to a marker at world position marker_xy."""
    dx, dy = marker_xy[0] - x[0], marker_xy[1] - x[1]
    return np.array([np.hypot(dx, dy), wrap(np.arctan2(dy, dx) - x[2])])


def measurement_jacobian(x, marker_xy):
    """H = dh/dx (2x3)."""
    dx, dy = marker_xy[0] - x[0], marker_xy[1] - x[1]
    rho2 = dx * dx + dy * dy
    rho = np.sqrt(rho2)
    return np.array([[-dx / rho, -dy / rho, 0.0],
                     [dy / rho2, -dx / rho2, -1.0]])


def numeric_jacobian(f, x, eps=1e-6):
    """Finite-difference Jacobian, for checking F, G and H (Sections 3 and 4 deliverable)."""
    x = np.asarray(x, float)
    f0 = np.asarray(f(x))
    J = np.zeros((f0.size, x.size))
    for i in range(x.size):
        dx = np.zeros_like(x)
        dx[i] = eps
        J[:, i] = (np.asarray(f(x + dx)) - np.asarray(f(x - dx))) / (2 * eps)
    return J


# Section 6: the EKF

class EKF:
    """Extended Kalman filter for the planar pose.

    Usage (each control step):
        ekf.predict(dphi_L, dphi_R)
        for obs in detect_markers(img):       # only when a camera frame arrives (10 Hz)
            ekf.update(obs.id, obs.z, obs.R)
        x_hat, P = ekf.x, ekf.P               # what the controller is allowed to use
    """

    def __init__(self, x0, P0, gate=GATE_CHI2, k=K_SLIP, floor=SIGMA_FLOOR):
        self.x = np.asarray(x0, float).copy()
        self.P = np.asarray(P0, float).copy()
        self.gate = gate
        self.k, self.floor = k, floor
        self.log = []  # one dict per update, for NIS and gating statistics (Sections 9 and 11)

    def predict(self, dphi_L, dphi_R):
        F, G = motion_jacobians(self.x, dphi_L, dphi_R)
        Q = G @ wheel_noise_cov(dphi_L, dphi_R, self.k, self.floor) @ G.T
        self.x, *_ = motion_model(self.x, dphi_L, dphi_R)
        self.P = F @ self.P @ F.T + Q
        self.P = 0.5 * (self.P + self.P.T)  # keep symmetric

    def update(self, marker_id, z, R):
        """Range-bearing update from one marker. Returns True if accepted, False if gated out.

        TODO (Section 6): make sure this update uses the state predicted to the image timestamp.
        """
        m = cfg.MARKERS[marker_id].t[:2]
        H = measurement_jacobian(self.x, m)
        nu = np.asarray(z, float) - measurement_model(self.x, m)
        nu[1] = wrap(nu[1])
        S = H @ self.P @ H.T + R
        nis = float(nu @ np.linalg.solve(S, nu))
        accepted = nis <= self.gate
        self.log.append(dict(id=marker_id, nu=nu, S=S, nis=nis, accepted=accepted))
        if not accepted:
            return False

        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ nu
        self.x[2] = wrap(self.x[2])
        I_KH = np.eye(3) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T  # Joseph form
        self.P = 0.5 * (self.P + self.P.T)
        return True



if __name__ == "__main__":
    rng = np.random.default_rng(0)
    x = np.array([0.3, -0.2, 0.7])
    dL, dR = 0.05, 0.08
    F, G = motion_jacobians(x, dL, dR)
    F_num = numeric_jacobian(lambda s: motion_model(s, dL, dR)[0], x)
    G_num = numeric_jacobian(lambda u: motion_model(x, u[1] / R_W, u[0] / R_W)[0],
                             np.array([R_W * dR, R_W * dL]))
    print("max |F - F_num| =", np.abs(F - F_num).max())
    print("max |G - G_num| =", np.abs(G - G_num).max())

    m = cfg.MARKERS[cfg.DOCK_ID].t[:2]
    H = measurement_jacobian(x, m)
    H_num = numeric_jacobian(lambda s: measurement_model(s, m), x)
    print("max |H - H_num| =", np.abs(H - H_num).max())