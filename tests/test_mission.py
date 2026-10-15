import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.config import OBSTACLES
from src.control.control import LQRTracker, Trajectory, wrap
from src.control.mission import DOCK_HEADING, DOCK_STOP, ROUTE, clearance, obstacle_distance

DT = 0.02

def run(start, est_sigma=(0, 0, 0), gain_err=(1, 1), Q=(40, 40, 8), R=(5, 2), seed=0):

    rng = np.random.default_rng(seed)
    traj = Trajectory([start[:2], *ROUTE], v_ref=0.15, end_heading=DOCK_HEADING)
    ctrl = LQRTracker(DT, Q, R)
    x, log, t = np.array(start, float), [], 0
    while t < traj.duration + 4:
        x_hat = x + rng.normal(0, est_sigma)
        ref = traj.sample(t)
        v, w, _ = ctrl.command(x_hat, ref)
        v, w, = v * gain_err[0], w * gain_err[1]
        x = x + DT * np.array([v * np.cos(x[2]), v * np.sin(x[2]), w]); x[2] = wrap(x[2])
        log.append([t, *x, *ref[:3]])
        t += DT
    L = np.array(log)
    final_pos = np.hypot(*(x[:2] - np.array(DOCK_STOP)))
    final_head = abs(wrap(x[2] - DOCK_HEADING))
    obs_clear = min(obstacle_distance(p)[0] for p in L[:, 1:3])
    return L, dict(final_pos=final_pos, final_head=np.degrees(final_head), obs_clear=obs_clear, success=final_pos < 0.03 and final_head < np.radians(5) and obs_clear > 0.2)

def random_start(rng):
    return (rng.uniform(-1.7, -1.1), rng.uniform(-1.7, -1.2), rng.uniform(-np.pi, np.pi))

if __name__ == "__main__":
    xy = Trajectory([(-1.5, -1.5), *ROUTE], end_heading=DOCK_HEADING)
    path = np.array([xy.sample(t)[:2] for t in np.arange(0, xy.duration, 0.05)])
    obs, wall = clearance(path)
    print(f"route clearance beyond robot radius: obstacles {obs*100:.1f} cm, walls {wall*100:.1f} cm")

    rng = np.random.default_rng(1)
    for label, sig, ge in [("clean estimate        ", (0, 0, 0), (1, 1)),
                           ("noisy est + 5% gain err", (0.02, 0.02, 0.03), (0.95, 1.05))]:
        res = [run(random_start(rng), sig, ge, seed=i)[1] for i in range(50)]
        ok = np.mean([r["success"] for r in res])
        print(f"{label}: success {ok*100:.0f}% of 50 | final pos err median {np.median([r['final_pos'] for r in res])*100:.1f} cm,"
              f" worst {max(r['final_pos'] for r in res)*100:.1f} cm | min obstacle clearance {min(r['obs_clear'] for r in res)*100:.1f} cm")
    L, _ = run((-1.5, -1.5, 0))
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(L[:, 4], L[:, 5], "k--", label="reference"); ax.plot(L[:, 1], L[:, 2], label="actual")
    for n, k, c, s, _ in OBSTACLES:
        ax.add_patch(plt.Circle(c, s, color="r", alpha=0.4) if k == "cylinder"
                     else plt.Rectangle((c[0]-s[0], c[1]-s[1]), 2*s[1], 2*s[1], color="r", alpha=0.4))
    ax.plot(*DOCK_STOP, "g*", ms=14, label="dock stop"); ax.set_xlim(-2, 2); ax.set_ylim(-2, 2)
    ax.set_aspect("equal"); ax.legend(); plt.savefig("tests/mission_tracking.png", dpi=120)