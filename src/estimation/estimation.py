# Estimation: KF/EKF or observer design, tuning and reported performance
"""Pose estimation for Track A (criterion C)

    EncoderOdometry   wheel encoder angles -> measured (v, omega) over each control period
    EKF               extended Kalman filter (Week 7 C)- predicts with the unicycle motion model driven by the
                      encoder odometry, corrects with ArUco range-and-bearing measurements to the known markers
    DeadReckoning     odometry only, no corrections (Week 7 C.4), kept as the baseline that the EKF is judged against

Both estimators share one interface (predict(u, dt), update(observations), x, P), so either can drive main.py.
"""
import numpy as np

from src import config as cfg
from src.control import model


# Week 7 C.3- the EKF algorithm, and Week 7 C.4- the range-and-bearing sensor model

def ekf_predict(xh, P, u, f, F_jac, Qd):
    xh_pred = f(xh, u) # nonlinear f propagates the state
    F = F_jac(xh, u)  # Jacobian propagates the covariance
    P_pred = F @ P @ F.T + Qd
    return xh_pred, P_pred


def ekf_update(xh_pred, P_pred, y, h, H_jac, Rn, residual=None):
    y_hat = h(xh_pred) # nonlinear h gives the expected measurement
    H = H_jac(xh_pred)
    S = H @ P_pred @ H.T + Rn # innovation covariance
    K = P_pred @ H.T @ np.linalg.inv(S) # Kalman gain
    nu = (y - y_hat) if residual is None else residual(y, y_hat)
    xh = xh_pred + K @ nu
    P = (np.eye(len(xh_pred)) - K @ H) @ P_pred
    return xh, P, nu, S


def h_rb(xh, mk):
    """Range and bearing from pose xh to marker mk."""
    dx, dy = mk[0] - xh[0], mk[1] - xh[1]
    return np.array([np.hypot(dx, dy), model.wrap(np.arctan2(dy, dx) - xh[2])])


def H_rb(xh, mk):
    """Jacobian H_t = dh/dx evaluated at the predicted estimate."""
    dx, dy = mk[0] - xh[0], mk[1] - xh[1]
    q = dx ** 2 + dy ** 2
    r = np.sqrt(q)
    return np.array([[-dx / r, -dy / r, 0.0],
                     [dy / q, -dx / q, -1.0]])


def rb_residual(y, y_hat):
    """Innovation with the bearing wrapped (Week 7 C.5-  "one missing wrap() and the filter lurches")."""
    return np.array([y[0] - y_hat[0], model.wrap(y[1] - y_hat[1])])


# Odometry

class EncoderOdometry:
    """Turns successive encoder readings into the robot's measured (v, omega) (Week 7-dead reckoning input)."""

    def __init__(self, angles0):
        self.prev = np.array(angles0, float)

    def update(self, angles, dt):
        angles = np.array(angles, float)
        w_l, w_r = (angles - self.prev) / dt
        self.prev = angles
        return model.wheels_to_unicycle(w_l, w_r)


def odometry_noise(x, u, dt, scale_sigma=cfg.EKF_WHEEL_SCALE_SIGMA, slip_sigma=cfg.EKF_WHEEL_SLIP_SIGMA):
    """Process noise Q_d for one prediction step, from the wheel odometry error model.

    Wk7 uses a constant Q_d while here it comes from the errors of the odometry that drives the prediction. Each
    wheel's distance ds has variance (scale_sigma ds)^2 (radius uncertainty) + slip_sigma^2 r |ds| (slip). This is
    mapped onto the state throug hthe Jacobian of the motion model with respect to the two wheel distances. A robot
    standing still thus gains no uncertainty and a fast turn gains mostly heading uncertainty"""
    v, w = u
    ds_l, ds_r = (v - w * model.B / 2) * dt, (v + w * model.B / 2) * dt
    var = (scale_sigma * np.array([ds_l, ds_r])) ** 2 + slip_sigma ** 2 * model.R * np.abs([ds_l, ds_r])
    var += (model.R * 2 * np.pi / cfg.ENCODER_TICKS) ** 2 / 6 # encoder quantisation, both readings

    # Jacobian of f_uni with respect to (ds_l, ds_r)- numerical via (v, omega)
    D = np.array([[1 / (2 * dt), 1 / (2 * dt)], [-1 / (model.B * dt), 1 / (model.B * dt)]]) # d(v, w)/d(ds)
    eps = 1e-6
    cols = []
    for a, b in D.T:
        d = model.f_uni(x, (v + eps * a, w + eps * b), dt) - model.f_uni(x, (v - eps * a, w - eps * b), dt)
        d[2] = model.wrap(d[2]) # heading difference across +/- pi
        cols.append(d / (2 * eps))
    G = np.column_stack(cols)
    return G @ np.diag(var) @ G.T


# Estimators

class EKF:
    """Week 7 C.4 EKF on x = (x, y, theta) with the project's sensors:
        predict: model.f_uni/F_uni driven by the encoder odometry, Q_d from odometry_noise() x q_scale
        update:  one ArUco range-and-bearing measurement at a time against the known marker map with the
                 covariance perception reports for it. The range and bearing variances are scaled by r_scale, the
                 bearing innovation wrapped, and a chi-square gate on the normalised innovation (NIS) to reject 
                 outliers e.g. mis-identified marker
    q_scale and r_scale are explained and set in src/config.py.
    """

    def __init__(self, x0, P0_sigma=cfg.EKF_P0, q_scale=cfg.EKF_Q_SCALE, r_scale=cfg.EKF_R_SCALE,
                 markers=cfg.MARKERS, gate=cfg.EKF_GATE):
        self.x = np.array(x0, float)
        self.P = np.diag(np.square(P0_sigma))
        self.q_scale, self.gate = q_scale, gate
        self.r_sqrt = np.diag(np.sqrt(np.broadcast_to(r_scale, 2))) # R -> D R D scales (range, bearing) variances
        self.markers = {i: T.t[:2] for i, T in markers.items()}
        # diagnostics for tuning and consistency checks
        self.nis, self.rejected = [], 0
        self.innovations = [] # (camera frame, marker id, innovation whitened by S^-1/2)
        self.frame = 0 # camera frames passed to update()
        self.reject_streak = 0 # consecutive camera frames whose marker measurements were ALL rejected
        self.recoveries = [] # camera frames at which the covariance was re-inflated

    # Lock-out recovery (criterion E). The gate trusts P. If the robot is moved in a way odometry cannot see (a push,
    # a wrong start pose), P stays small, every correct measurement fails the gate, and the filter never corrects
    # itself. When every marker measurement has been rejected for RECOVER_FRAMES frames in a row, the filter
    # concludes that it, not the camera, is wrong, and inflates P so the measurements are accepted again.
    RECOVER_FRAMES = 5 # 0.5 s at 10 Hz
    RECOVER_SIGMA = (0.3, 0.3, np.deg2rad(30.0))

    def predict(self, u, dt):
        Qd = self.q_scale * odometry_noise(self.x, u, dt)
        self.x, self.P = ekf_predict(self.x, self.P, u, lambda s, u_: model.f_uni(s, u_, dt),
                                     lambda s, u_: model.F_uni(s, u_, dt), Qd)
        self.x[2] = model.wrap(self.x[2])

    def update(self, observations):
        """Correct with the marker observations of one camera frame (perception.MarkerObservation list)."""
        self.frame += 1
        known = [o for o in observations if o.id in self.markers]
        accepted = self._update(known)
        if known and accepted == 0:
            self.reject_streak += 1
            if self.reject_streak >= self.RECOVER_FRAMES:
                self.P = self.P + np.diag(np.square(self.RECOVER_SIGMA))
                self.recoveries.append(self.frame)
                self.reject_streak = 0
                self._update(known)
        elif accepted:
            self.reject_streak = 0

    def _update(self, observations):
        """Sequential updates with gating; returns how many measurements were accepted."""
        accepted = 0
        for obs in observations:
            mk = self.markers.get(obs.id)
            if mk is None:
                continue
            # Gate before committing- NIS = nu^T S^-1 nu is chi-square with 2 dof for a consistent filter
            R = self.r_sqrt @ obs.R @ self.r_sqrt
            H = H_rb(self.x, mk)
            nu = rb_residual(obs.z, h_rb(self.x, mk))
            S = H @ self.P @ H.T + R
            nis = float(nu @ np.linalg.solve(S, nu))
            if nis > self.gate:
                self.rejected += 1
                continue
            self.nis.append(nis)
            self.innovations.append((self.frame, obs.id, np.linalg.solve(np.linalg.cholesky(S), nu)))
            self.x, self.P, _, _ = ekf_update(self.x, self.P, obs.z, lambda s: h_rb(s, mk), lambda s: H_rb(s, mk),
                                              R, residual=rb_residual)
            self.x[2] = model.wrap(self.x[2])
            self.P = (self.P + self.P.T) / 2 # keep it symmetric against round-off
            accepted += 1
        return accepted


class DeadReckoning:
    """Odometry only: x_{k+1} = f(x_k, u_k), no corrections (Week 7 C.4 'dead reckoning'). This is the baseline."""

    def __init__(self, x0, **_):
        self.x = np.array(x0, float)
        self.P = np.zeros((3, 3)) # not propagated

    def predict(self, u, dt):
        self.x = model.f_uni(self.x, u, dt)

    def update(self, observations):
        return None
