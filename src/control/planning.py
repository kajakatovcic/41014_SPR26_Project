"""Planning on a map the robot builds from its LiDAR - for the explore mission.

This content is extended past the tutorials. The tuts give LiDAR measurement model + frames (wk4), the estimate
(wk7) and the path-following controller (wk5/6/8) but no planner. This is a standard minimal extension so that the
robot can find its own way given no previous map.

    OccupancyGrid  log-odds occupancy grid in the reference frame {A}. Each LiDAR return marks the cell it hits as
                   occupied and the cells along the beam as free (ray casting), placed with the ESTIMATED pose so
                   obstacles seen from a slightly wrong estimate fade as better scans overwrite them.
    astar          A* on the grid, 8-connected. Cells closer to an obstacle than the robot's radius + margin are
                   blocked and nearby cells cost more so the path keeps to open space.
    plan           A* path -> a few waypoints (line-of-sight shortcuts) -> the spline through them
                   (control.Path) ending at the goal heading, checked for clearance on the grid. Falls back to 
                   straight segments if no spline fits.
Unknown cells count as free (optimistic). The robot plans straight at the goal and replans as the LiDAR sees more.
"""
import heapq

import numpy as np
from scipy.ndimage import distance_transform_edt

from src import config as cfg
from src.control import model


class OccupancyGrid:
    L_OCC, L_FREE, L_MIN, L_MAX = 0.85, -0.4, -2.0, 3.5 # log-odds increments and limits
    L_THRESH = 0.5 # occupied above this (two hits or one hit not yet seen through)

    def __init__(self, extent=cfg.GRID_EXTENT, res=cfg.GRID_RES):
        (x0, x1), (y0, y1) = extent
        self.origin, self.res = np.array([x0, y0]), res
        self.L = np.zeros((int(round((x1 - x0) / res)), int(round((y1 - y0) / res)))) # [ix, iy]
        self._clear = None

    def cells(self, p):
        return np.floor((np.asarray(p) - self.origin) / self.res).astype(int)

    def centres(self, ij):
        return self.origin + (np.asarray(ij) + 0.5) * self.res

    def _inside(self, ij):
        return (ij[..., 0] >= 0) & (ij[..., 0] < self.L.shape[0]) & (ij[..., 1] >= 0) & (ij[..., 1] < self.L.shape[1])

    def update(self, x_hat, points_R, sensor_R=cfg.T_RS.t[:2]):
        """Add one scan: LiDAR returns in the robot frame (N x 2), placed with the estimated pose x_hat in {A}."""
        if points_R is None or len(points_R) == 0:
            return
        c, s = np.cos(x_hat[2]), np.sin(x_hat[2])
        Rot = np.array([[c, -s], [s, c]])
        o = x_hat[:2] + Rot @ sensor_R
        P = x_hat[:2] + points_R @ Rot.T
        d = np.linalg.norm(P - o, axis=1)
        step = self.res / 2
        n_free = np.maximum(((d - self.res) / step).astype(int), 0) # samples short of the hit cell
        ray = np.repeat(np.arange(len(P)), n_free)
        k = np.concatenate([np.arange(n) for n in n_free]) if n_free.sum() else np.zeros(0, int)
        free_pts = o + (P[ray] - o) * ((k * step) / d[ray])[:, None]
        flat = self.L.reshape(-1) # each cell counts once per scan (unique flat indices)
        occ = self.cells(P)
        occ = np.unique(np.ravel_multi_index(occ[self._inside(occ)].T, self.L.shape))
        free = self.cells(free_pts)
        free = np.unique(np.ravel_multi_index(free[self._inside(free)].T, self.L.shape))
        free = np.setdiff1d(free, occ, assume_unique=True) # a cell with a return in it is not free this scan
        flat[free] += self.L_FREE
        flat[occ] += self.L_OCC
        np.clip(self.L, self.L_MIN, self.L_MAX, out=self.L)
        self._clear = None

    @property
    def occupied(self):
        return self.L > self.L_THRESH

    @property
    def clearance(self):
        """Distance [m] from each cell centre to the nearest occupied cell (cached until the next scan)."""
        if self._clear is None:
            occ = self.occupied
            self._clear = distance_transform_edt(~occ) * self.res if occ.any() else np.full(self.L.shape, np.inf)
        return self._clear

    def clearance_at(self, p):
        ij = self.cells(np.atleast_2d(p))
        out = np.full(len(ij), np.inf)
        ok = self._inside(ij)
        out[ok] = self.clearance[ij[ok, 0], ij[ok, 1]]
        return out


KAPPA_MAX = 15.0 # 1/m, tightest spline curvature accepted (0.067m radius: driven at >= 0.053 m/s). Splines that
                 # hug an obstacle's corner reach 3-14/m. A U-turn squeezed into a few cm reaches hundreds.

NEIGHBOURS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def astar(cost, start, goal):
    """Cheapest 8-connected path on a cost grid (np.inf = blocked) from cell start to cell goal, or None."""
    nx, ny = cost.shape
    g = np.full(cost.shape, np.inf)
    parent = -np.ones(cost.shape + (2,), int)
    g[start] = 0.0
    h = lambda c: np.hypot(c[0] - goal[0], c[1] - goal[1]) 
    heap = [(h(start), 0.0, start)]
    while heap:
        _, gc, c = heapq.heappop(heap)
        if c == goal:
            path = [c]
            while tuple(parent[c]) != (-1, -1):
                c = tuple(parent[c])
                path.append(c)
            return path[::-1]
        if gc > g[c]:
            continue
        for di, dj in NEIGHBOURS:
            n = (c[0] + di, c[1] + dj)
            if not (0 <= n[0] < nx and 0 <= n[1] < ny) or not np.isfinite(cost[n]):
                continue
            gn = gc + np.hypot(di, dj) * cost[n]
            if gn < g[n]:
                g[n], parent[n] = gn, c
                heapq.heappush(heap, (gn + h(n), gn, n))
    return None


def _segment_clear(grid, a, b, need):
    n = max(2, int(np.ceil(np.linalg.norm(b - a) / (grid.res / 2))) + 1)
    pts = a + np.linspace(0, 1, n)[:, None] * (b - a)
    return bool(np.all(grid.clearance_at(pts) >= need))


def shortcut(grid, pts, need):
    """Fewest waypoints along a dense path such that each straight leg keeps `need` clearance (string pulling)."""
    out, i = [pts[0]], 0
    while i < len(pts) - 1:
        j = len(pts) - 1
        while j > i + 1 and not _segment_clear(grid, pts[i], pts[j], need):
            j -= 1
        out.append(pts[j])
        i = j
    return np.array(out)


def plan(grid, x_hat, goal, goal_heading, smooth=True, clearance=cfg.PLAN_CLEARANCE, prefer=0.6, weight=4.0,
         approach=0.3):
    """A path (control.Path) from the estimated pose to the goal position, arriving at goal_heading or None.
    The robot first turns on the spot onto the path unless the path starts within 30deg of its heading. The last
    `approach` metres run straight along goal_heading (A* plans to the point that far back) so the robot arrives
    lined up instead of hooking round onto the heading at the last moment."""
    from src.control.control import Path 

    clear = grid.clearance
    cost = 1.0 + weight * np.clip((prefer - clear) / prefer, 0, None)
    cost[clear < clearance] = np.inf
    final = goal
    pre = goal - approach * np.array([np.cos(goal_heading), np.sin(goal_heading)])
    if (approach > 0 and np.linalg.norm(pre - x_hat[:2]) > approach and _segment_clear(grid, pre, goal, clearance)
            and grid._inside(grid.cells(pre)) and np.isfinite(cost[tuple(grid.cells(pre))])):
        goal = pre # A* to the approach point. The straight approach is appended below
    start, end = tuple(grid.cells(x_hat[:2])), grid.cells(goal)
    if not (grid._inside(np.array(start)) and grid._inside(end)):
        return None
    # The robot may already be inside the margin (it stopped close to something). Let it out at a high cost
    near = np.linalg.norm(grid.centres(np.indices(cost.shape).transpose(1, 2, 0)) - x_hat[:2], axis=2) < 0.4
    cost[near & ~np.isfinite(cost)] = 50.0
    if not np.isfinite(cost[tuple(end)]): # goal inside the margin- The nearest free cell within 0.3m
        free = np.argwhere(np.isfinite(cost))
        if not len(free):
            return None
        k = np.argmin(np.linalg.norm(free - end, axis=1))
        if np.linalg.norm(free[k] - end) * grid.res > 0.3:
            return None
        end = free[k]
    cells = astar(cost, start, tuple(end))
    if cells is None:
        return None
    dense = np.vstack([x_hat[:2], grid.centres(np.array(cells[1:-1], int).reshape(-1, 2)), goal])
    first = dense[min(len(dense) - 1, 4)] - x_hat[:2] # initial direction of the path (~0.2m ahead)
    start_heading = x_hat[2] if abs(model.wrap(np.arctan2(first[1], first[0]) - x_hat[2])) < np.deg2rad(30) else None
    tail = [final] if goal is not final else [] # the straight approach
    # Fewer waypoints give a smoother spline. More keep it closer to the straight legs. Use the fewest that keep
    # the spline itself clear of every return and no tighter than KAPPA_MAX (the controller slows to keep
    # v kappa <= 0.8rad/s, so an extremely tight curve is crawled at ~0 speed). Straight legs with turns on the spot
    # (always clear and drivable) are used as a last resort.
    # Starting along the robot's heading avoids a stop to turn, but can hook if the first waypoint is close. A free
    # start makes the robot turn on the spot onto the path first.
    starts = (start_heading, None) if start_heading is not None else (None,)
    for margin in (0.08, 0.15, 0.25) if smooth else ():
        pts = shortcut(grid, dense, clearance + margin)
        if len(pts) < 2:
            continue
        for sh in starts:
            path = Path(np.vstack([pts] + tail), start_heading=sh, end_heading=goal_heading)
            if (np.all(grid.clearance_at(path.xy[3:]) >= cfg.ROBOT_RADIUS + 0.03)
                    and np.max(np.abs(path.kappa)) <= KAPPA_MAX):
                return path
    pts = shortcut(grid, dense, clearance)
    pts = np.vstack([pts] + tail)
    return Path(pts, smooth=False) if len(pts) >= 2 else None # the controller turns to goal_heading at the end
