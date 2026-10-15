"""Spline route geometry.

originally src/control/control.py.
Renamed to trajectory.py when integrated with kaja-branch whose src/control/control.py holds the mission
controller.

    Trajectory   cubic spline through the waypoints (parametrised by arc length along the chords), with optional
                 start/end headings

How it is used in the integrated system (src/control/control.py):
    - Trajectory's spline is the route geometry of MissionController (control.Path)- the spline through the
      waypoints with the end heading set to face the dock, and its curvature as a turn-rate feed-forward
      (omega_r = v kappa) added to the 2-state Bryson LQR. The route is now config.WAYPOINTS.
    - Only the geometry is kept here. 
"""
import numpy as np
from scipy.interpolate import CubicSpline


class Trajectory:

    def __init__(self, waypoints, start_heading=None, end_heading=None):
        waypoint = np.asanyarray(waypoints, float)
        segment = np.hypot(*np.diff(waypoint, axis=0).T)
        sum = np.concatenate([[0], np.cumsum(segment)])
        bc = "natural"
        if start_heading is not None or end_heading is not None:
            def end(h):
                return (2, np.zeros(2)) if h is None else (1, np.array([np.cos(h), np.sin(h)]))
            bc = (end(start_heading), end(end_heading))
        self.spline = CubicSpline(sum, waypoint, bc_type=bc)
        self.length = sum[-1]
