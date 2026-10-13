"""Controllers for Track A.

    WaypointFollower   PLACEHOLDER (Week 8 C.4): "point towards the next waypoint, drive and turn", proportional
                       heading control plus a constant forward speed, switching waypoint inside a tolerance.
                       To be replaced by the LQR tracker on the tracking-error model (src/control/model.py).
    obstacle_ahead     lidar safety check: is anything in the corridor the robot is about to drive through?

Controllers receive the ESTIMATED pose only, never ground truth.
"""
import numpy as np

from src import config as cfg
from src.control import model

# Lidar safety corridor in the robot frame - the chassis half width plus a margin, out to a stopping distance ahead
# of the camera housing (the front of the robot, x = 0.086m)
CORRIDOR_HALF_WIDTH = 0.153 + 0.05
STOP_DISTANCE = 0.086 + 0.20


def obstacle_ahead(points_R, half_width=CORRIDOR_HALF_WIDTH, stop=STOP_DISTANCE):
    """True if any lidar return (N x 2, robot frame) lies in the corridor straight ahead of the robot."""
    if points_R is None or len(points_R) == 0:
        return False
    x, y = points_R[:, 0], points_R[:, 1]
    return bool(np.any((x > 0) & (x < stop) & (np.abs(y) < half_width)))


class WaypointFollower:
    """Placeholder waypoint controller (Week 8 C.4).

    omega = k_heading * heading error to the current waypoint; v = v_nominal * cos(heading error), so the robot
    turns on the spot when facing away and drives straight when lined up. It slows down approaching the final
    waypoint, and stops if the lidar sees an obstacle in the corridor ahead."""

    def __init__(self, waypoints=cfg.WAYPOINTS, v_nominal=0.15, k_heading=1.5, k_goal=0.5,
                 tol=cfg.WAYPOINT_TOL, goal_tol=cfg.GOAL_TOL):
        self.waypoints = [np.array(w, float) for w in waypoints]
        self.v_nominal, self.k_heading, self.k_goal = v_nominal, k_heading, k_goal
        self.tol, self.goal_tol = tol, goal_tol
        self.index = 0                     # current waypoint
        self.blocked = False               # last command was stopped by the lidar safety check

    @property
    def done(self):
        return self.index >= len(self.waypoints)

    def command(self, x_hat, scan_points=None):
        """(v, omega) from the estimated pose x_hat = (x, y, theta) and the latest lidar points (robot frame)."""
        self.blocked = False
        if self.done:
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
