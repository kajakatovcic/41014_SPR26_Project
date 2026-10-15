"""Controllers for Track A (criterion D).

    MissionController  the mission: follow the route then dock at the charging station. There are three modes/states:
                        Track: LQR path following on the cross-track/heading error model (wks 5,6,8) along a smooth
                               spline through the route, with a curvature feed-forward (spline geometry and
                               feed-forward from Anthony Fava's trajectory tracker, src/control/trajectory.py).
                               If the LiDAR blocks the path for 1sec it plans a detour around the obstacle.
                        Turn:  Turn on the spot with the Lyapunov heading law omega = -k e_theta (wk6 B6) at the
                               start, at the corners of straight (detour) legs, before docking, and whenever the
                               heading error leaves the linear model's range.
                        Dock:  Position-based visual servoing (PBVS) on the dock marker (wk9).
    Path               the reference path: a cubic spline or straight segments
    WaypointFollower   The earlier placeholder (derived from wk 8 C4) kept as a baseline to compare against.
    obstacle_ahead     LiDAR's safety check - is there anything in the corridor that the robot is about to drive through?

    Controllers receive the ESTIMATED pose + sensory data and NEVER ground truth.
"""
import control as ct
import numpy as np

from src import config as cfg
from src.control import model
from src.control.trajectory import Trajectory
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


class Path:
    """The reference path, sampled every ds metres: positions, tangent headings, curvatures and arc length.

    smooth=True: a cubic spline through the points with optional start/end headings, using Trajectory 
    (src/control/trajectory.py) for the geomtry. smooth=False: straight segments (curvature 0, the heading jumps
    at each corner), used for detours, whos straight legs are what the clearance check assumes."""

    def __init__(self, points, start_heading=None, end_heading=None, smooth=True, ds=0.01):
        pts = np.asarray(points, float)
        if smooth:
            spline = Trajectory(pts, start_heading=start_heading, end_heading=end_heading).spline
            u = np.linspace(0, spline.x[-1], max(2, int(np.ceil(spline.x[-1] / ds)) + 1))
            xy, d1, d2 = spline(u), spline(u, 1), spline(u, 2)
            th = np.arctan2(d1[:, 1], d1[:, 0])
            kappa = (d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]) / np.hypot(d1[:, 0], d1[:, 1]) ** 3
        else:
            xy, th = [], []
            for a, b in zip(pts[:-1], pts[1:]):
                n = max(2, int(np.ceil(np.linalg.norm(b - a) / ds)) + 1)
                seg = a + np.linspace(0, 1, n)[:, None] * (b - a)
                xy.append(seg if not xy else seg[1:])
                th.append(np.full(len(xy[-1]), np.arctan2(*(b - a)[::-1])))
            xy, th = np.vstack(xy), np.concatenate(th)
            kappa = np.zeros(len(xy))
        self._set(xy, th, kappa, pts)

    def _set(self, xy, th, kappa, points):
        self.xy, self.th, self.kappa = xy, th, kappa
        self.s = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))])
        self.points = np.asarray(points, float)

    def join(self, other, j):
        """This path, then `other` from its sample j onwards (e.g. straight detour legs rejoining the spline)."""
        joined = Path.__new__(Path)
        rest = [w for w in other.points if other.s[np.argmin(np.linalg.norm(other.xy - w, axis=1))] > other.s[j]]
        joined._set(np.vstack([self.xy, other.xy[j + 1:]]), np.concatenate([self.th, other.th[j + 1:]]),
                    np.concatenate([self.kappa, other.kappa[j + 1:]]), list(self.points) + rest)
        return joined

    def __len__(self):
        return len(self.xy)


class MissionController:
    """Route, then visual docking. Calls command() once per control period."""

    TURN_ENTER = np.deg2rad(30) # heading error beyond which the linear error model is not trusted (Week 5 B2)
    TURN_EXIT = np.deg2rad(3) # turn on the spot until the heading error is this small
    W_TRACK_MAX = 0.8 # rad/s, turn rate the feed-forward may ask for: the speed drops on tight curves to respect it
    LOOKAHEAD = 0.2 # m, how far ahead the speed limit looks for curvature and corners
    SEARCH = 0.5 # m, the nearest path point is searched this far ahead of the current one (progress is monotonic)

    def __init__(self, waypoints=cfg.WAYPOINTS, start=cfg.START_POSE, v_nominal=0.15, k_turn=1.5,
                 lam_v=0.5, lam_w=1.5, dock_distance=cfg.DOCK_DISTANCE, dock_tol=cfg.DOCK_TOL, Q=Q_LQR, R=R_LQR,
                 smooth=True):
        self.v_nominal, self.k_turn = v_nominal, k_turn
        self.lam_v, self.lam_w = lam_v, lam_w # Week 9 D3 docking gains
        self.dock_distance, self.dock_tol = dock_distance, dock_tol
        self.Q, self.R = Q, R
        self._gains = {}
        # dock marker - the robot must face against its normal
        T = cfg.MARKERS[cfg.DOCK_ID]
        self.dock_marker = T.t[:2]
        self.dock_heading = model.wrap(np.arctan2(T.R[1, 2], T.R[0, 2]) + np.pi)
        # route: a spline arriving at the pre-dock pose already facing the dock or straight segments. With no
        # waypoints the robot docks straight from where it starts.
        if len(waypoints):
            self.path = Path([start[:2], *waypoints], start_heading=start[2], end_heading=self.dock_heading,
                             smooth=smooth)
        else:
            self.path = Path([start[:2], np.array(start[:2]) + 0.01 * np.array([np.cos(start[2]), np.sin(start[2])])],
                             smooth=False)
        self._history = [] # path driven before each detour, for reference_path()
        # STATE
        self.idx = 0 # index of the nearest path point (progress along the path)
        self.route_done = not len(waypoints)
        self.mode = "turn"
        self.blocked = False
        self.blocked_since = None  # time the lidar first blocked the path
        self.detours = 0 # detours planned around unexpected obstacles
        self.errors = (np.nan, np.nan) # (e_y, e_theta) of the last tracking step (for logging)
        self.last_dock_obs, self.last_dock_t = None, -np.inf

    @property
    def done(self):
        return self.mode == "done"

    @property
    def index(self): # waypoints passed (for logging)
        s_now = self.path.s[self.idx]
        return sum(1 for w in self.path.points[1:] if self._arc(w) <= s_now + 1e-6) + (1 if self.route_done else 0)

    def _arc(self, p):
        return self.path.s[np.argmin(np.linalg.norm(self.path.xy - p, axis=1))]

    def gain(self, v):
        key = round(v, 2)
        if key not in self._gains:
            self._gains[key] = lqr_gain(key, self.Q, self.R)
        return self._gains[key]

    def _ahead(self, distance):
        """Index of the path point `distance` metres ahead of the current one."""
        return min(int(np.searchsorted(self.path.s, self.path.s[self.idx] + distance)), len(self.path) - 1)

    def _target_heading(self):
        if not self.route_done:
            return self.path.th[self._ahead(0.03)]
        return self.dock_heading

    def reference_path(self):
        """The path planned and driven (dense points N x 2), for evaluating tracking against ground truth."""
        return np.vstack(self._history + [self.path.xy])

    # Detour around an obstacle the map does not have. The route was planned around the known
    # obstacles. Anything else is found by the LiDAR.
    # This is not from tutorial work, rather an extension using a minimal local replanner.
    DETOUR_AFTER = 1.0  # sseconds blocked before replanning (lets a passing obstacle clear)
    DETOUR_CLEARANCE = cfg.ROBOT_RADIUS + 0.05

    def _plan_detour(self, x_hat, scan_points):
        """Sidestep the blocked path- Two new waypoints (one beside the robot and one beside the far side of the
        obstacle), then straight to the next waypoint beyond the obstacle, where the robot rejoins the planned
        path. The side and the offset come from the LiDAR points. The detour is only accepted if every new leg keeps
        clear of every LiDAR return."""
        th = self.path.th[self.idx]
        u = np.array([np.cos(th), np.sin(th)])
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
        # rejoin at the first waypoint beyond the obstacle
        s_now = self.path.s[self.idx]
        rest = [w for w in self.path.points[1:] if self._arc(w) > s_now + s[obst].max() + self.DETOUR_CLEARANCE]
        if not rest:
            return False
        end = rest[0]
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
            rest_clear = min(_segment_clearance(P, P1, P2), _segment_clearance(P, P2, end))
            if first >= need_first and rest_clear >= need and (best is None or rest_clear > best[0]):
                best = (rest_clear, P1, P2)
        if best is None:
            return False
        # straight legs to the rejoin waypoint, then the rest of the planned path (the spline) as before
        self._history.append(self.path.xy[:self.idx + 1])
        j = int(np.argmin(np.linalg.norm(self.path.xy - end, axis=1)))
        self.path = Path([p.copy(), best[1], best[2], end], smooth=False).join(self.path, j)
        self.idx = 0
        self.mode = "turn" # face the first leg
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
            if not self.route_done:
                self.idx = self._ahead(0.03) # past a straight-leg corner, onto the leg just turned to
            self.mode = "dock" if self.route_done else "track"

        if self.mode == "track":
            return self._track(x_hat, scan_points, t)
        if self.mode == "dock":
            return self._dock(x_hat, t)
        return 0.0, 0.0

    def _track(self, x_hat, scan_points, t):
        """LQR path following: nearest path point -> (e_y, e_theta), d_omega = -K e plus the curvature
        feed-forward omega_r = v kappa so curves are driven rather than corrected."""
        path = self.path
        p = x_hat[:2]
        stop = int(np.searchsorted(path.s, path.s[self.idx] + self.SEARCH)) + 1
        self.idx += int(np.argmin(np.linalg.norm(path.xy[self.idx:stop] - p, axis=1)))
        remaining = path.s[-1] - path.s[self.idx]
        if remaining < 0.01: # end of the route: turn to the dock heading, then dock
            self.route_done = True
            self.mode = "turn"
            return self.command(x_hat, scan_points, (), t)
        # A corner of straight legs (the heading jumps by more than TURN_ENTER) reached: step past it and turn on
        # the spot onto the next leg. Without this, overshooting a sharp corner keeps the nearest path point AT the
        # corner, with the old heading, and the robot carries on straight.
        ahead = slice(self.idx, self._ahead(0.015) + 1)
        jump = np.flatnonzero(np.abs(model.wrap(path.th[ahead] - path.th[self.idx])) > self.TURN_ENTER)
        if len(jump):
            self.idx += int(jump[0])
            self.mode = "turn"
            return self.command(x_hat, scan_points, (), t)
        th = path.th[self.idx]
        d = p - path.xy[self.idx]
        e_y = np.cos(th) * d[1] - np.sin(th) * d[0] # cross-track error - left positive
        e_th = model.wrap(x_hat[2] - th)
        self.errors = (e_y, e_th)
        if abs(e_th) > self.TURN_ENTER: # outside the linear model (or a straight-leg corner) - realign first
            self.mode = "turn"
            return self.command(x_hat, scan_points, (), t)
        # speed: nominal, slower for the curvature ahead (|v kappa| <= W_TRACK_MAX) and for a corner or the end
        ahead = slice(self.idx, self._ahead(self.LOOKAHEAD) + 1)
        k_max = float(np.max(np.abs(path.kappa[ahead])))
        corner = np.max(np.abs(model.wrap(path.th[ahead] - th))) > self.TURN_ENTER
        to_stop = remaining if not corner else path.s[self.idx + int(np.argmax(np.abs(model.wrap(path.th[ahead] - th)) > self.TURN_ENTER))] - path.s[self.idx]
        v = min(self.v_nominal, self.W_TRACK_MAX / max(k_max, 1e-6), max(0.04, 0.6 * to_stop))
        w = v * path.kappa[self.idx] - float((self.gain(v) @ np.array([e_y, e_th]))[0])
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

    def _dock(self, x_hat, t):
        """Week 9 D3: v = lam_v (z_M - d*), omega = -lam_w atan2(x_M, z_M), on the marker seen by the camera.
        No LiDAR stop here- the dock itself is the obstacle being approached."""
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
