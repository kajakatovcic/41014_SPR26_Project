import numpy as np

from src.config import DOCK_POS, OBSTACLES, ROBOT_RADIUS, ROOM_HALF, WALL_THICK

WALL_IN = ROOM_HALF - WALL_THICK
DOCK_STOP = (1.55, DOCK_POS[1])
DOCK_HEADING = 0
ROUTE = [(-0.6, -1.3), (-0.1, -0.7), (-0.05, 0.2), (0.1, 0.85), (0.5, 1), (1, 1), DOCK_STOP]

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