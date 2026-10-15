## Main code file for integrating all control/estimation/perception
import argparse
import time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np

from src import config as cfg
from src.control.control import LQRTracker, Trajectory, to_wheel_speeds, wrap
from src.control.mission import DOCK_HEADING, DOCK_STOP, ROUTE, obstacle_distance
from src.simulation import world

CONTROL_DT = 0.02
SETTLE = 4
DEFAULT_START = (-1.5, -1.5, 0)

class DeadReckoning:

    def __init__(self, x0, enc0):
        self.x = np.array(x0, float)
        self.enc = np.array(enc0, float)

    def update(self, enc):
        dl, dr = (np.asarray(enc) - self.enc) * cfg.WHEEL_RADIUS
        self.enc = np.array(enc, float)
        ds, dth = (dl + dr) / 2, (dr - dl) / cfg.WHEEL_TRACK
        mid = self.x[2] + dth / 2
        self.x += [ds * np.cos(mid), ds * np.sin(mid), dth]
        self.x[2] = wrap(self.x[2])
        return self.x.copy()

def random_start(rng):
    return (rng.uniform(-1.7, -1.1), rng.uniform(-1.7, -1.2), rng.uniform(-np.pi, np.pi))

def run_mission(model, data, start, viewer=None):

    mujoco.mj_resetData(model, data)
    world.set_pose(model, data, *start)
    steps = max(1, round(CONTROL_DT / model.opt.timestep))
    traj = Trajectory([start[:2], *ROUTE], v_ref=0.15, end_heading=DOCK_HEADING)
    ctrl = LQRTracker(CONTROL_DT)
    est = DeadReckoning(start, world.read_encoders(model, data))
    log = []
    for k in range(int((traj.duration + SETTLE) / CONTROL_DT)):
        t = k * CONTROL_DT
        x_hat = est.update(world.read_encoders(model, data))
        ref = traj.sample(t)
        v, w, _ = ctrl.command(x_hat, ref)
        data.ctrl[0], data.ctrl[1] = to_wheel_speeds(v, w)

        for _ in range(steps):
            mujoco.mj_step(model, data)
        x_true = world.true_pose(model, data)
        log.append([t, *x_true, *x_hat, *ref[:3], v, w])
        if viewer is not None:
            if not viewer.is_running():
                break
            viewer.sync()
            time.sleep(CONTROL_DT)
    L = np.array(log)
    final_pos = np.hypot(*(L[-1, 1:3] - np.array(DOCK_STOP)))
    final_head = np.degrees(abs(wrap(L[-1, 3] - DOCK_HEADING)))
    obs_clear = min(obstacle_distance(p)[0] for p in L[:, 1:3])
    est_err = np.hypot(L[:, 1] - L[:, 4], L[:, 2] - L[:, 5])
    return L, dict(final_pos=final_pos, final_head=final_head, obs_clear=obs_clear,
                   est_err_final=est_err[-1], est_err_max=est_err.max(),
                   success=final_pos < 0.03 and final_head < 5 and obs_clear > 0.2)

def report(m):
    return (f"final pos err {m['final_pos']*100:.1f} cm, heading err {m['final_head']:.1f} deg,"
            f"min obstacle clearance {m['obs_clear']*100:.1f} cm, odometry drift final {m['est_err_final']*100:.1f} cm "
            f"(max {m['est_err_max']*100:.1f} cm) -> {'SUCCESS' if m['success'] else 'miss'}")

def plot(L, path = "mission_sim.png"):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(L[:, 7], L[:, 8], "k--", label="reference")
    ax.plot(L[:, 1], L[:, 2], label="true (ground truth)")
    ax.plot(L[:, 4], L[:, 5], ":", label="odometry estimate")
    for n, k, c, s, _ in cfg.OBSTACLES:
        ax.add_patch(plt.Circle(c, s, color="r", alpha=0.4) if k == "cylinder"
                     else plt.Rectangle((c[0] - s[0], c[1] - s[1]), 2 * s[0], 2 * s[1], color="r", alpha=0.4))
    ax.plot(*DOCK_STOP, "g*", ms=14, label="dock stop")
    ax.set_xlim(-2, 2); ax.set_ylim(-2, 2); ax.set_aspect("equal"); ax.legend()
    plt.savefig(path, dpi=120)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--headless", action="store_true", help="no viewer (run as fast as possible)")
    ap.add_argument("--random", action="store_true", help="random start pose")
    ap.add_argument("--trials", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    model = world.build_world()
    data = mujoco.MjData(model)
    rng = np.random.default_rng(args.seed)

    if args.headless or args.trials > 1:
        results = []
        for i in range(args.trials):
            start = random_start(rng) if args.random else DEFAULT_START
            L, m = run_mission(model, data, start)
            results.append(m)
            print(f"trial {i}l start ({start[0]:.2f}, {start[1]:.2f}, {np.degrees(start[2]):.0f}deg) | {report(m)}")
        print(f"success {100 * np.mean([r['success'] for r in results]):.0f}% of {len(results)} | "
                  f"median final pos err {100 * np.median([r['final_pos'] for r in results]):.1f} cm")
        plot(L)
        return

    import mujoco.viewer as mj_viewer
    start = random_start(rng) if args.random else DEFAULT_START
    with mj_viewer.launch_passive(model, data) as viewer:
        L, m = run_mission(model, data, start, viewer)
        print(report(m))
        plot(L)
        while viewer.is_running():
            time.sleep(0.1)

if __name__ == "__main__":
    main()
