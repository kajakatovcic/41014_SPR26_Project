import numpy as np
from scipy.interpolate import CubicSpline
from scipy.linalg import solve_discrete_are
from src.config import WHEEL_RADIUS, WHEEL_SPEED_MAX, WHEEL_TRACK

V_MAX = WHEEL_SPEED_MAX * WHEEL_RADIUS # m/s
V_LINEAR_MIN = 0.05

def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi

def saturate(v, w, v_wheel_max=V_MAX, track=WHEEL_TRACK):
    peak = abs(v) + abs(w) * track / 2 # linear speed of the fastest wheel
    if peak > v_wheel_max:
        s = v_wheel_max / peak
        v, w = v * s, w * s
    return v, w

def to_wheel_speeds(v, w): # angular velocities of left/right wheels in rad/s
    return ((v - w * WHEEL_TRACK / 2) / WHEEL_RADIUS,
            (v + w * WHEEL_TRACK / 2) / WHEEL_RADIUS)

class Trajectory:

    def __init__(self, waypoints, v_ref=0.15, start_heading=None, end_heading=None):
        waypoint = np.asanyarray(waypoints, float)
        segment = np.hypot(*np.diff(waypoint, axis=0).T)
        sum = np.concatenate([[0], np.cumsum(segment)])
        bc = "natural"
        if start_heading is not None or end_heading is not None:
            def end(h):
                return (2, np.zeros(2)) if h is None else (1, np.array([np.cos(h), np.sin(h)]))
            bc = (end(start_heading), end(end_heading))
        self.spline = CubicSpline(sum, waypoint, bc_type=bc)
        self.length, self.v_ref = sum[-1], v_ref
        self.duration = self.length / v_ref

    def sample(self, t):
        sum = np.clip(t * self.v_ref, 0, self.length)
        d1, d2 = self.spline(sum, 1), self.spline(sum, 2)
        x, y = self.spline(sum)
        theta = np.arctan2(d1[1], d1[0])
        kappa = (d1[0] * d2[1] - d1[1] * d2[0]) / np.hypot(*d1) ** 3 # curvature of path
        if t >= self.duration:
            return x, y, theta, 0, 0
        return x, y, theta, self.v_ref, self.v_ref * kappa

class LQRTracker:

    def __init__(self, dt, Q=(40, 40, 8), R=(5, 2)):
        self.dt, self.Q, self.R = dt, np.diag(Q), np.diag(R)
        self._cache = {}

    def gain(self, v_r, w_r):
        v_linear = max(abs(v_r), V_LINEAR_MIN)
        key = (round(v_linear, 3), round(w_r, 3))
        if key not in self._cache:
            A = np.array([[0, w_r, 0], [-w_r, 0, v_linear], [0, 0, 0]])
            B = np.array([[-1, 0], [0, 0], [0, -1]])
            Ad, Bd = np.eye(3) + A * self.dt, B * self.dt # Euler discretisation
            P = solve_discrete_are(Ad, Bd, self.Q, self.R)
            self._cache[key] = np.linalg.solve(self.R + Bd.T @ P @ Bd, Bd.T @ P @ Ad)
        return self._cache[key]

    def error(self, x_hat, ref):
        c, s = np.cos(x_hat[2]), np.sin(x_hat[2])
        dx, dy = ref[0] - x_hat[0], ref[1] - x_hat[1]
        return np.array([c * dx + s * dy, -s * dx + c * dy, wrap(ref[2] - x_hat[2])])

    def command(self, x_hat, ref):
        e = self.error(x_hat, ref)
        v, w = np.array([ref[3], ref[4]]) - self.gain(ref[3], ref[4]) @ e
        v, w = saturate(v, w)
        return v, w, e