"""Explore mission controller (criterion D - DEFAULT mission in main.py).

    Navigator   The TurtleBot is given no route and no map. There are 6 modes:
                  search    turn on the spot until the reference marker is in view
                  localise  stand still while the estimator initialises from the reference marker (Week 3 E)
                  look      turn to where the dock marker should be (from its known offset) and look for it. One
                            marker leaves the rotation about it unobservable and its PnP yaw can be several degrees
                            out (planar ambiguity). A second known marker makes the full pose observable (Part A).
                            Moves on after LOOK_HOLD seconds whether or not the dock marker was seen.
                  plan      plan from the estimate to the pre-dock pose on the LiDAR occupancy grid (src/control/
                            planning.py: A* -> spline), held still until a plan exists
                  turn/track/dock   as MissionController: Lyapunov turn on the spot (Week 6 B.6), LQR path
                            following with the curvature feed-forward (Weeks 5, 6, 8), Week 9 D3 docking
                  find      the dock marker has not been in view for FIND_AFTER sseconds at the pre-dock pose (the estimate
                            has drifted, e.g. after a long detour with no known marker in view). Turn on the spot
                            until it is seen, hold still for FOUND_HOLD sseconds while the EKF corrects, then replan to the
                            pre-dock pose from the corrected estimate. Similarly, if the estimate puts the robot off the
                            dock marker's normal while docking (a push), it backs off and re-approaches
                Every LiDAR scan updates the grid. The robot replans whenever the path ahead runs within its radius of
                a LiDAR return, when it has drifted off the path (an estimate correction), or when the LiDAR safety
                corridor stops it. A large estimate correction restarts the grid since it was drawn from a wrong pose. The pre-dock pose comes from the dock marker's offset from the reference marker
                (cfg.DOCK_OFFSET) on the dock marker's normal.

The controller receives the ESTIMATED pose in the reference marker's frame {A} (None before localisation). The LiDAR
points and the marker observations - NEVER ground truth.
"""
import numpy as np

from src import config as cfg
from src.control import model, planning
from src.control.control import MissionController, obstacle_ahead


class Navigator(MissionController):
    SEARCH_RATE = 0.7 # rad/s, turn rate while looking for the reference marker (4 deg per camera frame)
    LOCALISE_HOLD = 0.3 # s, stay still this long after the reference marker was last seen
    LOOK_HOLD = 1.5 # s, how long to look towards the dock marker once facing it
    LOOK_RANGE = 4.5 # m, only look if the dock marker should be within camera range
    REPLAN_LOOKAHEAD = 1.5 # m of path ahead checked against the grid after every scan
    REPLAN_OFF_PATH = 0.15 # m, replan when the estimate is this far from the path (e.g. after a correction)
    DETOUR_AFTER = 0.5 # s blocked by the lidar corridor before replanning
    MAP_RESET_JUMP = 0.05 # m, an estimate correction this large restarts the lidar map
    FIND_AFTER = 3.0 # s without the dock marker in view while docking before searching for it
    FOUND_HOLD = 1.0 # s to hold still once it is found (the EKF's gate and lock-out recovery need ~0.5s)
    DOCK_ABORT_LATERAL = 0.04 # m off the dock marker's normal (estimated) while docking. Back off and re-approach
    DOCK_RETRIES = 3
    BACK_OFF_TO = cfg.PREDOCK_DISTANCE + 0.4 # m from the dock marker before re-approaching
    BACK_OFF_SPEED = 0.08 # m/s, reversing
    BACK_OFF_MAX_T = 12.0 # s

    def __init__(self, ref_id=cfg.REF_MARKER_ID, dock_offset=cfg.DOCK_OFFSET, predock=cfg.PREDOCK_DISTANCE,
                 smooth=True, **kw):
        super().__init__(waypoints=[], start=(0.0, 0.0, 0.0), smooth=smooth, **kw)
        self.ref_id, self.smooth = ref_id, smooth
        dx, dy, phi = dock_offset
        n = np.array([np.cos(phi), np.sin(phi)]) # dock marker normal in {A}
        self.dock_marker = np.array([dx, dy], float)
        self.dock_heading = model.wrap(phi + np.pi)
        self.goal = self.dock_marker + predock * n
        self.grid = planning.OccupancyGrid()
        self.path, self.route_done, self.mode = None, False, "search"
        self.replans, self.plan_failures = 0, 0
        self.replan_reasons = {"start": 0, "path blocked": 0, "off path": 0, "corridor": 0, "dock found": 0,
                               "dock abort": 0}
        self._last_scan, self._last_ref_t, self._last_x = None, -np.inf, None
        self.map_resets = 0
        self._look_since, self.dock_seen_at_start = None, False
        self._dock_lost_since, self._found_t, self.dock_searches = None, None, 0
        self.dock_retries, self._back_since = 0, None

    @property
    def index(self): # planned paths so far (for logging)
        return self.replans

    def reference_path(self):
        return np.vstack(self._history + ([self.path.xy] if self.path is not None else [np.zeros((0, 2))]))

    def command(self, x_hat, scan_points=None, observations=(), t=0.0):
        self.blocked = False
        if any(o.id == self.ref_id for o in observations):
            self._last_ref_t = t
        if x_hat is None: # not localised yet
            if t - self._last_ref_t < self.LOCALISE_HOLD:
                self.mode = "localise"
                return 0.0, 0.0
            self.mode = "search"
            return 0.0, self.SEARCH_RATE
        # A correction that moves the estimate further than the robot can drive in one period (0.011m) means the
        # map so far was drawn from a wrong pose. Start it again (the 360deg LiDAR refills it within a scan)
        if self._last_x is not None and np.linalg.norm(x_hat[:2] - self._last_x[:2]) > self.MAP_RESET_JUMP:
            self.grid = planning.OccupancyGrid()
            self.map_resets += 1
            if self.path is not None:
                self._planned_clear = np.full(len(self.path), np.inf) # judge the path on the new map alone
        self._last_x = x_hat.copy()
        if scan_points is not None and scan_points is not self._last_scan:
            self._last_scan = scan_points
            self.grid.update(x_hat, scan_points)
            if self.mode in ("track", "turn") and not self.route_done:
                reason = self._needs_replan(x_hat)
                if reason:
                    self._replan(x_hat, reason)
        if self.mode in ("search", "localise"):
            to_dock = self.dock_marker - x_hat[:2]
            self.mode = "look" if np.linalg.norm(to_dock) < self.LOOK_RANGE else "plan"
        if self.mode == "look":
            to_dock = self.dock_marker - x_hat[:2]
            e = model.wrap(x_hat[2] - np.arctan2(to_dock[1], to_dock[0]))
            if any(o.id == cfg.DOCK_ID for o in observations):
                self.dock_seen_at_start = True
            if abs(e) > self.TURN_EXIT and self._look_since is None:
                return 0.0, float(np.clip(-self.k_turn * e, -0.8 * model.W_MAX, 0.8 * model.W_MAX))
            self._look_since = t if self._look_since is None else self._look_since
            if not self.dock_seen_at_start and t - self._look_since < self.LOOK_HOLD:
                return 0.0, 0.0
            self.mode = "plan"
        if self.mode == "back off":
            return self._back_off(x_hat, t)
        if self.mode == "find":
            if any(o.id == cfg.DOCK_ID for o in observations):
                self._found_t = t if self._found_t is None else self._found_t
            if self._found_t is None:
                return 0.0, self.SEARCH_RATE
            if t - self._found_t < self.FOUND_HOLD:
                return 0.0, 0.0
            self._found_t = None
            if not self._replan(x_hat, "dock found"):
                self.mode = "plan"
                return 0.0, 0.0
        if self.mode == "plan":
            if not self._replan(x_hat, "start"):
                self.mode = "plan"
                return 0.0, 0.0
        return super().command(x_hat, scan_points, observations, t)

    def _needs_replan(self, x_hat):
        """'path blocked' when the path ahead now runs within the robot's radius of a LiDAR return AND closer than
        when it was planned (i.e. something new appeared). A path that had to start or end inside the margin (the
        robot stopped near something or the goal is next to an obstacle) would otherwise be replanned on every
        scan."""
        ahead = slice(self.idx, self._ahead(self.REPLAN_LOOKAHEAD) + 1)
        now = self.grid.clearance_at(self.path.xy[ahead])
        if np.any(now < np.minimum(cfg.ROBOT_RADIUS, self._planned_clear[ahead] - 0.03)):
            return "path blocked"
        if np.linalg.norm(self.path.xy[self.idx] - x_hat[:2]) > self.REPLAN_OFF_PATH and self.mode == "track":
            return "off path"
        return None

    def _replan(self, x_hat, reason):
        path = planning.plan(self.grid, x_hat, self.goal, self.dock_heading, smooth=self.smooth)
        if path is None:
            self.plan_failures += 1
            return False
        if self.path is not None:
            self._history.append(self.path.xy[:self.idx + 1])
        self.path, self.idx, self.route_done = path, 0, False
        self._planned_clear = self.grid.clearance_at(path.xy)
        self.mode = "turn" # onto the new path (exits at once if already lined up)
        self.blocked_since = None
        self.replans += 1
        self.replan_reasons[reason] = self.replan_reasons.get(reason, 0) + 1
        return True

    def _dock(self, x_hat, t):
        """
        Wk9 D3 docking (see MissionController._dock). If the dock marker stays out of view the robot will go and
        find it. If the estimate puts the robot more than DOCK_ABORT_LATERAL off the dock marker's normal (i.e.
        a push while docking- the docking law nulls range and bearing ONLY so it would dock off centre), back off 
        and reapproach from the pre-dock pose. This is done maximum DOCK_RETRIES times."""
        if self.last_dock_obs is not None and t - self.last_dock_t < 0.25:
            self._dock_lost_since = None
            n = np.array([np.cos(self.dock_heading + np.pi), np.sin(self.dock_heading + np.pi)]) # marker normal
            d = x_hat[:2] - self.dock_marker
            lateral = float(n[0] * d[1] - n[1] * d[0])
            if abs(lateral) > self.DOCK_ABORT_LATERAL and self.dock_retries < self.DOCK_RETRIES:
                self.dock_retries += 1
                self.mode, self._back_since = "back off", t
                return self._back_off(x_hat, t)
        else:
            self._dock_lost_since = t if self._dock_lost_since is None else self._dock_lost_since
            if t - self._dock_lost_since >= self.FIND_AFTER:
                self.mode, self._dock_lost_since = "find", None
                self.dock_searches += 1
                return 0.0, self.SEARCH_RATE
        return super()._dock(x_hat, t)

    def _back_off(self, x_hat, t):
        """Reverse straight out from the dock until BACK_OFF_TO from the dock marker (to give room to line up
        on its normal again) and then replan to the pre-dock pose. Stops early if the LiDAR sees anything close
        behind the robot."""
        far = np.linalg.norm(x_hat[:2] - self.dock_marker) >= self.BACK_OFF_TO
        behind = self._last_scan is not None and obstacle_ahead(-self._last_scan, stop=0.197 + 0.15)
        if far or behind or t - self._back_since > self.BACK_OFF_MAX_T:
            if self._replan(x_hat, "dock abort"):
                return 0.0, 0.0
            self.mode = "plan"
            return 0.0, 0.0
        return -self.BACK_OFF_SPEED, 0.0

    def _plan_detour(self, x_hat, scan_points):
        """Called by MissionController._track when the LiDAR corridor has been blocked for DETOUR_AFTER. Replan
        on the grid (which already holds what is blocking)."""
        return self._replan(x_hat, "corridor")
