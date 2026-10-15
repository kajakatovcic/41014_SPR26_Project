"""Route clearance checks against the known map.

ROUTE is now config.WAYPOINTS. the clearance check is unchanged.
    clearance   smallest distance from a path to the obstacles and walls, beyond the robot's radius
"""
import numpy as np

from src.config import OBSTACLES, ROBOT_RADIUS, ROOM_HALF, WALL_THICK

WALL_IN = ROOM_HALF - WALL_THICK

def obstacle_distance(p, obstacles=OBSTACLES):

    best = (np.inf, None)
    for name, kind, (cx, cy), size, _ in obstacles:
        if kind == "cylinder":
            d = np.hypot(p[0] - cx, p[1] - cy) - size
        else:
            qx, qy = abs(p[0] - cx) - size[0], abs(p[1] - cy) - size[1]
            d = np.hypot(max(qx, 0), max(qy, 0)) + min(max(qx, qy), 0)
        if d < best[0]:
            best = (d, name)
    return best

def wall_distance(p):

    return WALL_IN - max(abs(p[0]), abs(p[1]))

def clearance(xy):

    xy = np.asarray(xy)
    obs = min(obstacle_distance(p)[0] for p in xy) - ROBOT_RADIUS
    wall = min(wall_distance(p) for p in xy if p[0] < 1.4) - ROBOT_RADIUS
    return obs, wall
