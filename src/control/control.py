"""Controllers for Track A (criterion D).

    MissionController  the mission: follow the waypoint path then dock at the charging station. There are three modes/states:
                        Track: The LQR path tracking on the cross-track/heading error model (wks 5,6,8). If the LiDAR 
                               blocks the path for 1sec it plans a detour around the obstacle.
                        Turn:  Turn on the spot with the Lyapunov heading law omega = -k e_theta (wk6 B6) at corners and 
                               whenever the heading error leaves the linear model's range.
                        Dock:  Position-based visual servoing (PBVS) on the dock marker (wk9).
    WaypointFollower   The earlier placeholder (derived from wk 8 C4) kept as a baseline to compare against.
    obstacle_ahead     LiDAR's safety check - is there anything in the corridor that the robot is about to drive through? 

    Controllers receive the ESTIMATED pose + sensory data and NEVER ground truth.
"""
import control as ct
import numpy as np

from src import config as cfg
from src.control import model
from src.perception import perception as per

# LiDAR safety corridor in the robot frame - the chassis half width plus a margin, out to a stopping distance ahead
# of the camera housing (the front of the robot, x = 0.086m)
CORRIDOR_HALF_WIDTH = 0.153 + 0.05
STOP_DISTANCE = 0.086 + 0.20


def obstacle_ahead(points_R, half_width=CORRIDOR_HALF_WIDTH, stop=STOP_DISTANCE):
    """True if any LiDAR return (N x 2, robot frame) lies in the corridor straight ahead of the robot."""
    if points_R is None or len(points_R) == 0:
        return False
    x, y = points_R[:, 0], points_R[:, 1]
    return bool(np.any((x > 0) & (x < stop) & (np.abs(y) < half_width)))


def _segment_clearance(P, a, b):
    """Smallest distance from points P (N x 2) to the segment a -> b."""
    ab = b - a
    s = np.clip(((P - a) @ ab) / max(ab @ ab, 1e-12), 0, 1)
    return float(np.min(np.linalg.norm(P - (a + s[:, None] * ab), axis=1))) if len(P) else np.inf


# LQR path tracking (Week 6 A.4-A.5 used on the Week 5 D/Week 8 B error model)

# Bryson's rule (wk6) - Weight each state and input by 1/(largest acceptable value)^2
E_Y_MAX = 0.05 # m, cross-track error
E_TH_MAX = np.deg2rad(10.0) # rad, heading error
DW_MAX = 0.5 # rad/s, steering effort (a third of the 1.58rad/s the wheels can give)
Q_LQR = np.diag([1 / E_Y_MAX ** 2, 1 / E_TH_MAX ** 2])
R_LQR = np.array([[1 / DW_MAX ** 2]])


def lqr_gain(v, Q=Q_LQR, R=R_LQR, dt=cfg.CONTROL_DT):
    """Discrete LQR gain for the cross-track/heading error model at forward speed v: d_omega = -K [e_y, e_theta].
    The speed sits inside A (Week 5 D) so the gain is recomputed for the speed being driven (gain scheduling)."""
    Ad, Bd = model.discretise(*model.reduced_error_model(max(v, 0.02)), dt)
    K, _, _ = ct.dlqr(Ad, Bd, Q, R)
    return K


class MissionController:
    """Waypoint path, then visual docking. Calls command() once per control period."""

    TURN_ENTER = np.deg2rad(30) # heading error beyond which the linear error model is not trusted (Week 5 B2)
    TURN_EXIT = np.deg2rad(3) # turn on the spot until the heading error is this small

    def __init__(self, waypoints=cfg.WAYPOINTS, start=cfg.START_POSE, v_nominal=0.15, k_turn=1.5,
                 lam_v=0.5, lam_w=1.5, dock_distance=cfg.DOCK_DISTANCE, dock_tol=cfg.DOCK_TOL, Q=Q_LQR, R=R_LQR):
        self.points = [np.array(start[:2], float)] + [np.array(w, float) for w in waypoints]
        self.v_nominal, self.k_turn = v_nominal, k_turn
        self.lam_v, self.lam_w = lam_v, lam_w # Week 9 D3 docking gains
        self.dock_distance, self.dock_tol = dock_distance, dock_tol
        self.Q, self.R = Q, R
        self._gains = {}
        # dock marker - the robot must face against its normal
        T = cfg.MARKERS[cfg.DOCK_ID]
        self.dock_marker = T.t[:2]
        self.dock_heading = model.wrap(np.arctan2(T.R[1, 2], T.R[0, 2]) + np.pi)
        # STATE
        self.segment = 0
        self.mode = "turn"
        self.blocked = False
        self.blocked_since = None  # time the lidar first blocked the current segment
        self.detours = 0 # detours planned around unexpected obstacles
        self.errors = (np.nan, np.nan) # (e_y, e_theta) of the last tracking step (for logging)
        self.last_dock_obs, self.last_dock_t = None, -np.inf

    @property
    def done(self):
        return self.mode == "done"

    @property
    def index(self): # waypoints completed (for logging)
        return self.segment

    def gain(self, v):
        key = round(v, 2)
        if key not in self._gains:
            self._gains[key] = lqr_gain(key, self.Q, self.R)
        return self._gains[key]

    def _segment(self):
        a, b = self.points[self.segment], self.points[self.segment + 1]
        d = b - a
        L = np.linalg.norm(d)
        return a, d / L, L, np.arctan2(d[1], d[0])

    def _target_heading(self):
        if self.segment < len(self.points) - 1:
            return self._segment()[3]
        return self.dock_heading

    # Detour around an obstacle the map does not have. The route was planned around the known
    # obstacles. Anything else is found by the LiDAR. 
    # This is not from tutorial work, rather an extension using a minimal local replanner.
    DETOUR_AFTER = 1.0  # sseconds blocked before replanning (lets a passing obstacle clear)
    DETOUR_CLEARANCE = cfg.ROBOT_RADIUS + 0.05

    def _plan_detour(self, x_hat, scan_points):
        """Sidestep the blocked segment. Inserts two waypoints (one beside the robot and one beside the far side of
        the obstacle) then rejoin the route. The side and the offset come from the LiDAR points. The detour is
        only accepted if every new leg keeps DETOUR_CLEARANCE from every LiDAR return."""
        _, u, _, _ = self._segment()
        n = np.array([-u[1], u[0]]) # left of the path
        p = x_hat[:2]
        P = per.to_world(scan_points, x_hat) # LiDAR points in the world via the estimate
        rel = P - p
        s, lat = rel @ u, rel @ n
        # The obstacle - the returns inside the corridor ahead, grown by anything within 0.3m of them
        core = (s > 0) & (s < 1.0) & (np.abs(lat) < CORRIDOR_HALF_WIDTH)
        if not core.any():
            return False
        near = np.min(np.linalg.norm(P[:, None] - P[core][None], axis=2), axis=1) < 0.3
        obst = near & (s > 0)
        end = self.points[self.segment + 1]
        to_end = (end - p) @ u
        need = cfg.ROBOT_RADIUS + 0.02
        # The robot has stopped close to the obstacle, so the first (sideways) leg starts inside that margin. It
        # only has to not get any closer than the robot already is
        need_first = min(need, float(np.min(np.linalg.norm(P - p, axis=1))) - 0.005)
        best = None
        for side in (1, -1):
            offset = max(0.0, (side * lat[obst]).max()) + self.DETOUR_CLEARANCE
            P1 = p + side * offset * n
            P2 = P1 + min(s[obst].max() + self.DETOUR_CLEARANCE, max(to_end, 0.0)) * u
            first = _segment_clearance(P, p, P1)
            rest = min(_segment_clearance(P, P1, P2), _segment_clearance(P, P2, end))
            if first >= need_first and rest >= need and (best is None or rest > best[0]):
                best = (rest, P1, P2)
        if best is None:
            return False
        self.points = self.points[:self.segment + 1] + [p.copy(), best[1], best[2]] + self.points[self.segment + 1:]
        self.segment += 1 # now on the leg from the robot to P1
        self.mode = "turn"
        self.detours += 1
        return True

    def command(self, x_hat, scan_points=None, observations=(), t=0.0):
        """(v, omega) from the estimated pose, the latest LiDAR points (robot frame) and this period's marker
        observations (empty when there is no camera frame)."""
        self.blocked = False
        for obs in observations:
            if obs.id == cfg.DOCK_ID:
                self.last_dock_obs, self.last_dock_t = obs, t

        if self.mode == "turn":
            e = model.wrap(x_hat[2] - self._target_heading())
            if abs(e) > self.TURN_EXIT:
                return 0.0, float(np.clip(-self.k_turn * e, -0.8 * model.W_MAX, 0.8 * model.W_MAX))
            self.mode = "track" if self.segment < len(self.points) - 1 else "dock"

        if self.mode == "track":
            a, u, L, th = self._segment()
            p = x_hat[:2] - a
            s = p @ u # progress along the segment
            if s >= L - 0.01: # end of segment: next one, or the dock
                self.segment += 1
                self.mode = "turn"
                return self.command(x_hat, scan_points, (), t)
            e_y = u[0] * p[1] - u[1] * p[0] # cross-track error - left positive
            e_th = model.wrap(x_hat[2] - th)
            self.errors = (e_y, e_th)
            if abs(e_th) > self.TURN_ENTER: # if outside the linear model - realign first
                self.mode = "turn"
                return self.command(x_hat, scan_points, (), t)
            v = min(self.v_nominal, max(0.04, 0.6 * (L - s)))   # slow down for the corner
            w = float(-(self.gain(v) @ np.array([e_y, e_th]))[0])
            w = float(np.clip(w, -model.W_MAX, model.W_MAX))
            if obstacle_ahead(scan_points):
                v, w, self.blocked = 0.0, 0.0, True
                self.blocked_since = t if self.blocked_since is None else self.blocked_since
                if t - self.blocked_since >= self.DETOUR_AFTER and self._plan_detour(x_hat, scan_points):
                    self.blocked_since = None
                    return self.command(x_hat, None, (), t)
            else:
                self.blocked_since = None
            return v, w

        if self.mode == "dock":
            # Week 9 D3- v = lam_v (z_M - d*), omega = -lam_w atan2(x_M, z_M), on the marker seen by the camera.
            # No lidar stop here: the dock itself is the obstacle being approached.
            if self.last_dock_obs is not None and t - self.last_dock_t < 0.25:
                x_m, z_m, _ = per.docking_error(self.last_dock_obs)
                if abs(z_m - self.dock_distance) < self.dock_tol[0] and abs(x_m) < self.dock_tol[1]:
                    self.mode = "done"
                    return 0.0, 0.0
                v = float(np.clip(self.lam_v * (z_m - self.dock_distance), -0.05, 0.1))
                w = float(np.clip(-self.lam_w * np.arctan2(x_m, z_m), -0.8, 0.8))
                return v, w
            # marker not in view: stop and turn towards where the estimate says it is
            e = model.wrap(x_hat[2] - np.arctan2(*(self.dock_marker - x_hat[:2])[::-1]))
            return 0.0, float(np.clip(-self.k_turn * e, -0.8, 0.8))

        return 0.0, 0.0


# Baseline: the placeholder (from wk8 c.4)

class WaypointFollower:
    """Placeholder waypoint controller kept as the baseline.

    omega = k_heading * heading error to the current waypoint.
    v = v_nominal * cos(heading error) so the robot turns on the spot when facing away and drives straight when 
    lined up. It slows down approaching the final waypoint and stops if the LiDAR sees an obstacle in the corridor 
    ahead. No docking."""

    def __init__(self, waypoints=cfg.WAYPOINTS, v_nominal=0.15, k_heading=1.5, k_goal=0.5,
                 tol=cfg.WAYPOINT_TOL, goal_tol=cfg.GOAL_TOL):
        self.waypoints = [np.array(w, float) for w in waypoints]
        self.v_nominal, self.k_heading, self.k_goal = v_nominal, k_heading, k_goal
        self.tol, self.goal_tol = tol, goal_tol
        self.index = 0 # current waypoint
        self.blocked = False # last command was stopped by the LiDAR safety check
        self.mode = "track"
        self.errors = (np.nan, np.nan)

    @property
    def done(self):
        return self.index >= len(self.waypoints)

    def command(self, x_hat, scan_points=None, observations=(), t=0.0):
        """(v, omega) from the estimated pose x_hat = (x, y, theta) and the latest lidar points (robot frame)."""
        self.blocked = False
        if self.done:
            self.mode = "done"
            return 0.0, 0.0
        last = self.index == len(self.waypoints) - 1
        d = np.hypot(*(self.waypoints[self.index] - x_hat[:2]))
        if d < (self.goal_tol if last else self.tol):
            self.index += 1
            return self.command(x_hat, scan_points)

        target = self.waypoints[self.index]
        e_th = model.wrap(np.arctan2(target[1] - x_hat[1], target[0] - x_hat[0]) - x_hat[2])
        w = np.clip(self.k_heading * e_th, -model.W_MAX, model.W_MAX)
        v = self.v_nominal * max(np.cos(e_th), 0.0)
        if last:
            v = min(v, self.k_goal * d)
        if v > 0 and obstacle_ahead(scan_points):
            v, self.blocked = 0.0, True
        return v, w
