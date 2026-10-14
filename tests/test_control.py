import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__). resolve(). parents[1]))
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

from src.control.control import LQRTracker, Trajectory, wrap, to_wheel_speeds


DT = 0.02
WAYPOINTS = [(-1.2, -1.2), (1.2, -1.2), (1.2, 1.2), (-1.2, 1.2), (-1.2, -0.6)]

def run(Q=(40, 40, 8), R=(5, 2), x0=(-1.3, -1.3, 0.4), est_sigma=(0, 0, 0), seed=0):
    range = np.random.default_rng(seed)
    traj, ctrl = Trajectory(WAYPOINTS, v_ref=0.15), LQRTracker(DT, Q, R)
    x = np.array(x0, float)
    log, t = [], 0
    while t < traj.duration + 3:
        x_hat = x + range.normal(0, est_sigma) # controller only sees estimate
        ref = traj.sample(t)
        v, w, _ = ctrl.command(x_hat, ref)
        x = x + DT * np.array([v * np.cos(x[2]), v * np.sin(x[2]), w]) # true plant
        x[2] = wrap(x[2])
        e_true = ctrl.error(x, traj.sample(t)) # evaluate
        log.append([t, *x, *ref[:3], *e_true, v, w])
        t += DT
    return np.array(log)

if __name__ == "__main__":
    L = run()
    ct, he = L[:, 7], L[:, 9]
    print(f"clean estimate : RMS position erro {np.sqrt(np.mean(L[:,7]**2 + L[:,8]**2)):.4f} m, "
          f"max {np.max(np.hypot(L[:,7], L[:,8])):.3f} m, final {np.hypot(L[-1,7], L[-1,8]):.4f} m, "
          f"RMS heading {np.degrees(np.sqrt(np.mean(he**2))):.2f} deg")
    L2 = run(est_sigma=(0.02, 0.02, 0.03))
    print(f"noisy estimate : RMS pos error {np.sqrt(np.mean(L2[:,7]**2 + L2[:,8]**2)):.4f} m "
          f"(est. sigma 2 cm, 0.03 rad)")
    print("max wheel speed [rad/s]:", np.max(np.abs(to_wheel_speeds(L[:, 10], L[:, 11]))))
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    ax[0].plot(L[:, 4], L[:, 5], "k--", label="reference"); ax[0].plot(L[:, 1], L[:, 2], label="actual")
    ax[0].set_aspect("equal"); ax[0].legend(); ax[0].set_title("Path tracking")
    ax[1].plot(L[:, 0], np.hypot(L[:,7], L[:, 8])); ax[1].set_xlabel("t [s]"); ax[1].set_ylabel("position error [m]")
    plt.tight_layout(); plt.savefig("tests/control_tracking.png", dpi=120)

