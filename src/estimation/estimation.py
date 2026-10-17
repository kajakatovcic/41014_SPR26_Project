# Estimation: KF/EKF or observer design, tuning and reported performance
"""Pose estimation for Track A (criterion C)

    EncoderOdometry   wheel encoder angles -> measured (v, omega) over each control period
    EKF               extended Kalman filter (Week 7 C)- predicts with the unicycle motion model driven by the
                      encoder odometry, corrects with ArUco range-and-bearing measurements to the known markers
    MarkerSLAM        the explore mission's EKF. No map, localises from the reference marker (its frame {A} is the
                      reference frame), knows only the dock marker's offset from it, maps other markers as it sees them
    DeadReckoning     odometry only, no corrections (Week 7 C.4), kept as the baseline that the EKF is judged against

All estimators share one interface (predict(u, dt), update(observations), x, P), so any can drive main.py.
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
    RECOVER_FRAMES = 5 # 0.5s at 10Hz
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


def pose_from_marker(T_RM):
    """Robot pose in a marker's floor frame (origin at the marker centre, x along its normal out of the face) from
    the marker's PnP pose in the robot frame (Week 3 E). The marker normal seen at angle psi in {R} means the robot
    heads -psi in the marker frame, and the marker origin = robot position + Rot(theta) p_R."""
    p_R, n_R = T_RM.t[:2], T_RM.R[:2, 2]
    th = -np.arctan2(n_R[1], n_R[0])
    c, s = np.cos(th), np.sin(th)
    return np.array([*(-np.array([[c, -s], [s, c]]) @ p_R), th])


def _pose_from_rb_yaw(r, a, psi):
    """The same in terms of the measured range r, bearing a and marker yaw psi (for the prior's Jacobian)."""
    th = -psi
    return np.array([-r * np.cos(th + a), -r * np.sin(th + a), th])


class MarkerSLAM:
    """EKF for the explore mission in the frame {A} of the reference marker (cfg.REF_MARKER_ID). 
    This extends wk7 C.4:

      initialise  the robot stands still and averages INIT_FRAMES views of the reference marker. Its PnP pose gives
                  the robot's pose in {A}. The marker's yaw (which way it faces) comes from the LiDAR wall
                  line it hangs on (Week 4 RANSAC, set through `walls`) when there is one since the PnP yaw can
                  flip (planar pose ambiguity), and from PnP otherwise. The prior covariance comes from the Jacobian
                  of the pose with respect to (range, bearing, marker yaw) so the yaw's uncertainty appears as a
                  rotation about the marker - the direction one marker cannot observe (Part A).
      known       the reference marker at the origin and the dock marker at cfg.DOCK_OFFSET (measured by hand).
                  Range-and-bearing updates exactly as in the Week 7 EKF with its gate and lock-out recovery.
                  PLUS a heading update when a LiDAR wall line from the same moment passes through a known marker.
                  The wall's normal in the robot frame is the marker's facing direction psi, and its direction in
                  {A} is known, so z = psi = phi_M - theta (sigma 1 deg). One marker plus its wall observes the full
                  pose, which one marker alone cannot. This is what lets the estimate recover at the pre-dock pose
                  after a disturbance where only the dock marker is in view.
      mapped      any other marker is added to the state the first time it is seen (EKF-SLAM, Week 7 C), and updates 
                  both the robot and its own position from then on. Only once the dock marker has been seen. Before 
                  that the rotation about the reference marker is unobservable and the yaw from PnP can be several
                  degrees out so markers mapped then are placed incorrectly and later pull the estimate away from 
                  the dock marker. Mapped markers are also dropped when the lockout recovery starts since they may
                  have been mapped from the wrong pose it recovers from.
    correct=False runs the same initialisation, and then odometry only (dead reckoning from the marker).
    x and P are the robot's pose and is 3x3 covariance (None until initialised) like the other estimators."""

    RECOVER_FRAMES = EKF.RECOVER_FRAMES
    RECOVER_SIGMA = EKF.RECOVER_SIGMA

    def __init__(self, ref_id=cfg.REF_MARKER_ID, dock_id=cfg.DOCK_ID, dock_offset=cfg.DOCK_OFFSET,
                 q_scale=cfg.EKF_Q_SCALE, r_scale=cfg.EKF_R_SCALE, gate=cfg.EKF_GATE, init_frames=cfg.INIT_FRAMES,
                 sigma_yaw=cfg.INIT_SIGMA_YAW, correct=True):
        self.ref_id, self.dock_id = ref_id, dock_id
        self.known = {ref_id: np.zeros(2), dock_id: np.array(dock_offset[:2], float)}
        self.normals = {ref_id: 0.0, dock_id: float(dock_offset[2])} # facing directions in {A}
        self.walls_fresh = False # walls are from the same moment as this camera frame (set by the main loop)
        self.yaw_nis, self.yaw_rejected, self.yaw_reject_streak = [], 0, 0
        self.anchored = False # the dock marker (the second known marker) has corrected the estimate
        self.q_scale, self.gate, self.correct = q_scale, gate, correct
        self.r_sqrt = np.diag(np.sqrt(np.broadcast_to(r_scale, 2)))
        self.init_frames, self.sigma_yaw = init_frames, sigma_yaw
        self.s, self.S = None, None # full state (robot pose + mapped markers) and its covariance
        self.mapped = [] # ids of mapped markers, in state order
        self._views = [] # reference marker views while standing still, before initialisation
        self._moving = False
        # diagnostics as EKF
        self.nis, self.rejected, self.innovations = [], 0, []
        self.frame, self.reject_streak, self.recoveries = 0, 0, []
        self.init_frame = None
        self.walls = [] # latest LiDAR wall lines in the robot frame (perception.process_scan), for initialisation
        self.init_yaw_source = None # "wall" or "pnp"

    @property
    def ready(self):
        return self.s is not None

    @property
    def x(self):
        return None if self.s is None else self.s[:3]

    @property
    def P(self):
        return None if self.S is None else self.S[:3, :3]

    def predict(self, u, dt):
        self._moving = abs(u[0]) > 1e-3 or abs(u[1]) > 1e-3
        if not self.ready:
            return
        n = len(self.s)
        F = np.eye(n)
        F[:3, :3] = model.F_uni(self.s[:3], u, dt)
        self.s[:3] = model.f_uni(self.s[:3], u, dt)
        self.s[2] = model.wrap(self.s[2])
        self.S = F @ self.S @ F.T
        self.S[:3, :3] += self.q_scale * odometry_noise(self.s[:3], u, dt)

    def update(self, observations):
        self.frame += 1
        if not self.ready:
            self._initialise(observations)
            return
        if not self.correct:
            return
        usable = [o for o in observations if o.id in self.known or o.id in self.mapped]
        accepted = self._update(usable)
        if usable and accepted == 0:
            self.reject_streak += 1
            if self.reject_streak >= self.RECOVER_FRAMES:
                self.s, self.S, self.mapped = self.s[:3], self.S[:3, :3], [] # forget markers mapped from it
                self.S += np.diag(np.square(self.RECOVER_SIGMA))
                self.recoveries.append(self.frame)
                self.reject_streak = 0
                self._update([o for o in usable if o.id in self.known])
        elif accepted:
            self.reject_streak = 0
        if self.walls_fresh:
            self._wall_yaw_updates([o for o in usable if o.id in self.known])
        if not self.anchored:
            return
        for o in observations:
            if o.id not in self.known and o.id not in self.mapped:
                self._add_marker(o)

    def _initialise(self, observations):
        ref = [o for o in observations if o.id == self.ref_id]
        if self._moving or not ref:
            self._views = [] # the views must all be from the same pose
            return
        self._views.append(ref[0])
        if len(self._views) < self.init_frames:
            return
        p_R = np.mean([o.T_RM.t[:2] for o in self._views], axis=0)
        psi = self._wall_yaw(p_R)
        if psi is not None:
            th, sigma_yaw, self.init_yaw_source = -psi, cfg.INIT_SIGMA_YAW_WALL, "wall"
        else: # PnP yaw, median over the views (robust to an occasional flipped frame)
            th = float(np.median([pose_from_marker(o.T_RM)[2] for o in self._views]))
            sigma_yaw, self.init_yaw_source = self.sigma_yaw, "pnp"
        c, s = np.cos(th), np.sin(th)
        self.s = np.array([*(-np.array([[c, -s], [s, c]]) @ p_R), th])
        # prior: Jacobian of the pose with respect to (range, bearing, yaw) with the frame's range/bearing noise
        r, a = np.hypot(*p_R), np.arctan2(p_R[1], p_R[0])
        z = np.array([r, a, -th])
        J = np.column_stack([(_pose_from_rb_yaw(*(z + dz)) - _pose_from_rb_yaw(*(z - dz))) / 2e-6
                             for dz in np.eye(3) * 1e-6])
        R = self.r_sqrt @ self._views[-1].R @ self.r_sqrt
        self.S = J @ np.diag([R[0, 0], R[1, 1], sigma_yaw ** 2]) @ J.T + np.diag([1e-4, 1e-4, 1e-5])
        self.init_frame = self.frame

    YAW_GATE = 8.81 # chi-square 1 dof, 99.7 %

    def _wall_yaw_updates(self, observations, max_offset=0.15, agree=np.deg2rad(30.0)):
        """Heading update from the wall line behind each known marker in view. The wall must pass within 
        max_offset of the marker (dock marker stands ~0.1m proud of its wall) and agree with the PnP normal to
        within `agree` (the right wall and not a corner). The update is gated like the others."""
        for o in observations:
            p_R, n_R = o.T_RM.t[:2], o.T_RM.R[:2, 2]
            psi = self._wall_yaw(p_R, max_offset=max_offset)
            if psi is None or abs(model.wrap(psi - np.arctan2(n_R[1], n_R[0]))) > agree:
                continue
            phi = self.normals[o.id]
            H = np.zeros((1, len(self.s)))
            H[0, 2] = -1.0
            R = np.array([[cfg.INIT_SIGMA_YAW_WALL ** 2]])
            nu = model.wrap(psi - (phi - self.s[2]))
            nis = float(nu ** 2 / (H @ self.S @ H.T + R)[0, 0])
            if nis > self.YAW_GATE:
                # a turn the odometry did not see (a push) fails this gate every time. The same lock-out recovery
                self.yaw_rejected += 1
                self.yaw_reject_streak += 1
                if self.yaw_reject_streak < self.RECOVER_FRAMES:
                    continue
                self.s, self.S, self.mapped = self.s[:3], self.S[:3, :3], []
                self.S += np.diag(np.square(self.RECOVER_SIGMA))
                self.recoveries.append(self.frame)
                H = np.array([[0.0, 0.0, -1.0]])
            self.yaw_reject_streak = 0
            self.yaw_nis.append(nis)
            self.s, self.S, _, _ = ekf_update(self.s, self.S, np.array([psi]),
                                              lambda s_: np.array([model.wrap(phi - s_[2])]), lambda s_: H, R,
                                              residual=lambda y, yh: model.wrap(y - yh))
            self.s[2] = model.wrap(self.s[2])
            self.S = (self.S + self.S.T) / 2

    def _wall_yaw(self, p_R, max_offset=0.1, max_gap=0.3):
        """Direction of the reference marker's normal in the robot frame from the LiDAR wall line it hangs on.
        A RANSAC wall (wk4) passing within max_offset of marker centre with returns within max_gap of it.The
        wall normal is turned to face the robot (the marker faces the robot that sees it). None if no wall."""
        best = None
        for (a, b, c), pts in self.walls:
            off = abs(a * p_R[0] + b * p_R[1] + c)
            if off < max_offset and np.min(np.linalg.norm(pts - p_R, axis=1)) < max_gap:
                if best is None or off < best[0]:
                    n = np.array([a, b]) if np.dot([a, b], -p_R) > 0 else -np.array([a, b])
                    best = (off, float(np.arctan2(n[1], n[0])))
        return None if best is None else best[1]

    def _lm(self, s, i):
        """Position of marker i: fixed if known or from the state s if mapped."""
        return self.known[i] if i in self.known else s[3 + 2 * self.mapped.index(i):][:2]

    def _H(self, i, H_r):
        """Full-state Jacobian- the robot block from H_rb and minus its position part on a mapped marker."""
        H = np.zeros((2, len(self.s)))
        H[:, :3] = H_r
        if i in self.mapped:
            k = 3 + 2 * self.mapped.index(i)
            H[:, k:k + 2] = -H_r[:, :2]
        return H

    def _update(self, observations):
        accepted = 0
        for obs in observations:
            mk = self._lm(self.s, obs.id).copy()
            R = self.r_sqrt @ obs.R @ self.r_sqrt
            H = self._H(obs.id, H_rb(self.s[:3], mk))
            nu = rb_residual(obs.z, h_rb(self.s[:3], mk))
            Sn = H @ self.S @ H.T + R
            nis = float(nu @ np.linalg.solve(Sn, nu))
            if nis > self.gate:
                self.rejected += 1
                continue
            self.nis.append(nis)
            self.innovations.append((self.frame, obs.id, np.linalg.solve(np.linalg.cholesky(Sn), nu)))
            i = obs.id
            self.s, self.S, _, _ = ekf_update(self.s, self.S, obs.z, lambda s_: h_rb(s_[:3], self._lm(s_, i)),
                                              lambda s_: self._H(i, H_rb(s_[:3], self._lm(s_, i))), R,
                                              residual=rb_residual)
            self.s[2] = model.wrap(self.s[2])
            self.S = (self.S + self.S.T) / 2
            accepted += 1
            self.anchored |= obs.id == self.dock_id
        return accepted

    def _add_marker(self, obs):
        """EKF-SLAM landmark initialisation. Inverse of the range-and-bearing model with its Jacobians."""
        x, y, th = self.s[:3]
        r, a = obs.z
        c, s = np.cos(th + a), np.sin(th + a)
        G_x = np.array([[1, 0, -r * s], [0, 1, r * c]])
        G_z = np.array([[c, -r * s], [s, r * c]])
        R = self.r_sqrt @ obs.R @ self.r_sqrt
        S_ll = G_x @ self.S[:3, :3] @ G_x.T + G_z @ R @ G_z.T
        S_lx = G_x @ self.S[:3, :]
        self.s = np.concatenate([self.s, [x + r * c, y + r * s]])
        self.S = np.block([[self.S, S_lx.T], [S_lx, S_ll]])
        self.mapped.append(obs.id)


class DeadReckoning:
    """Odometry only: x_{k+1} = f(x_k, u_k) w/ no corrections (Week 7 C.4 'dead reckoning'). 
    This is used as the baseline."""

    def __init__(self, x0, **_):
        self.x = np.array(x0, float)
        self.P = np.zeros((3, 3)) # not propagated

    def predict(self, u, dt):
        self.x = model.f_uni(self.x, u, dt)

    def update(self, observations):
        return None
