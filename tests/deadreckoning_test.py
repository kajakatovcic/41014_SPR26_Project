"""Section 8: dead-reckoning baseline (predict-only EKF) on a repeatable path.

The robot drives open-loop (constant wheel speeds, no feedback), so no ground truth enters the control. The filter
only sees encoders. The true pose is used for EVALUATION only: error plots and RMSE.

What it produces (for the report):
  - path: true vs estimated
  - error in x, y, theta over time, with the +/-3 sigma bounds from the filter's covariance P
  - RMSE and final error, for the sim as-is (noise_k = 0) and with extra odometry noise injected (noise_k > 0)

Two cases are run because the simulator's encoders have almost no random noise, so the "as-is" baseline looks
better than any real robot. The injected-noise case is the more honest one for tuning and for part E.

Run from the project root:  python -m tests.baseline_deadreckoning
"""
from pyexpat import model
import sys
from pathlib import Path
from xml.parsers.expat import model

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import mujoco
import numpy as np

from src import config as cfg
from src.simulation import world
from src.estimation.estimation import EKF, EncoderOdometry
from src.perception.perception import wrap

PREDICT_DT = 0.05            # s, the filter predicts at 20 Hz on the increments accumulated since the last predict
START = (0.4, -0.2, np.pi / 2)   # start on a circle of radius 0.4 m centred at (0, -0.2), heading counter-clockwise
RADIUS = 0.4                 # m. This circle clears all four obstacles by >0.2 m (checked by hand, re-check if
                             # you move obstacles in config.py)
SPEED = 0.132                # m/s (about 4 rad/s on the wheels)


def circle_wheel_speeds(v, radius):
    """Wheel speeds [rad/s] (left, right) for a constant-curvature counter-clockwise arc."""
    w = v / radius
    wl = (v - w * cfg.WHEEL_TRACK / 2) / cfg.WHEEL_RADIUS
    wr = (v + w * cfg.WHEEL_TRACK / 2) / cfg.WHEEL_RADIUS
    return wl, wr


def set_wheel_speeds(data, wl, wr):
    data.ctrl[0] = np.clip(wl, -cfg.WHEEL_SPEED_MAX, cfg.WHEEL_SPEED_MAX)   # wheel_left
    data.ctrl[1] = np.clip(wr, -cfg.WHEEL_SPEED_MAX, cfg.WHEEL_SPEED_MAX)   # wheel_right


def run_baseline(model, data, laps=3, noise_k=0.0, seed=0, clockwise = False):
    """Drive `laps` laps open-loop. Returns arrays t, truth (N x 3), est (N x 3), sigma (N x 3)."""
    rng = np.random.default_rng(seed)
    radius = -RADIUS if clockwise else RADIUS
    start = (-0.4, -0.2, np.pi / 2) if clockwise else START
    world.set_pose(model, data, *start)

    # The start pose is known (we placed the robot), so it is a legitimate initial estimate. It is NOT ground
    # truth leaking in later: after this the filter only sees encoders.
    x0 = np.array(start, float)
    P0 = np.diag([0.01 ** 2, 0.01 ** 2, np.deg2rad(1.0) ** 2])
    ekf = EKF(x0, P0)

    odo = EncoderOdometry()
    odo.increments(world.read_encoders(model, data))

    wl, wr = circle_wheel_speeds(SPEED, radius)
    set_wheel_speeds(data, wl, wr)
    duration = laps * 2 * np.pi * abs(radius) / SPEED
    steps_per_predict = int(round(PREDICT_DT / model.opt.timestep))
    n_predicts = int(duration / PREDICT_DT)

    acc = np.zeros(2)            # accumulated wheel angle increments [left, right]
    t, truth, est, sigma = [], [], [], []
    for k in range(n_predicts):
        for _ in range(steps_per_predict):
            mujoco.mj_step(model, data)
            acc += odo.increments(world.read_encoders(model, data))

        # Extra odometry noise (multiplicative, on each predict interval) so it matches the model in wheel_noise_cov:
        # sigma_travel = noise_k * |travel| per predict step. Off by default.
        noisy = acc * (1.0 + noise_k * rng.standard_normal(2))
        ekf.predict(noisy[0], noisy[1])
        acc[:] = 0

        t.append((k + 1) * PREDICT_DT)
        truth.append(world.true_pose(model, data))       # EVALUATION ONLY
        est.append(ekf.x.copy())
        sigma.append(np.sqrt(np.diag(ekf.P)))
    set_wheel_speeds(data, 0.0, 0.0)
    return np.array(t), np.array(truth), np.array(est), np.array(sigma)


def errors(truth, est):
    e = est - truth
    e[:, 2] = wrap(e[:, 2])
    return e


def report(label, t, truth, est, sigma):
    e = errors(truth, est)
    rmse = np.sqrt((e ** 2).mean(axis=0))
    pos_final = np.hypot(e[-1, 0], e[-1, 1])
    dist = np.sum(np.hypot(*np.diff(truth[:, :2], axis=0).T))
    print(f"[{label}] distance driven {dist:.1f} m")
    print(f"   RMSE  x {rmse[0]*1000:.1f} mm | y {rmse[1]*1000:.1f} mm | theta {np.rad2deg(rmse[2]):.3f} deg")
    print(f"   final position error {pos_final*1000:.1f} mm "
          f"(3 sigma x {3*sigma[-1,0]*1000:.1f} mm, y {3*sigma[-1,1]*1000:.1f} mm), "
          f"final heading error {np.rad2deg(e[-1,2]):+.3f} deg (3 sigma {np.rad2deg(3*sigma[-1,2]):.3f} deg)")
    # Fraction of the time the true error is inside the filter's own 3 sigma bound (should be ~99.7% if consistent)
    inside = (np.abs(e) <= 3 * sigma).mean(axis=0)
    print(f"   inside 3 sigma: x {inside[0]*100:.0f}% | y {inside[1]*100:.0f}% | theta {inside[2]*100:.0f}%")


def plot(label, t, truth, est, sigma, fname):
    e = errors(truth, est)
    fig, ax = plt.subplots(2, 2, figsize=(11, 8))
    ax[0, 0].plot(truth[:, 0], truth[:, 1], label="true")
    ax[0, 0].plot(est[:, 0], est[:, 1], "--", label="dead reckoning")
    ax[0, 0].set_aspect("equal"); ax[0, 0].set_xlabel("x [m]"); ax[0, 0].set_ylabel("y [m]")
    ax[0, 0].set_title(f"Path ({label})"); ax[0, 0].legend()
    for a, i, name, scale, unit in [(ax[0, 1], 0, "x", 1000, "mm"), (ax[1, 0], 1, "y", 1000, "mm"),
                                    (ax[1, 1], 2, "theta", 180 / np.pi, "deg")]:
        a.plot(t, e[:, i] * scale, label="error")
        a.fill_between(t, -3 * sigma[:, i] * scale, 3 * sigma[:, i] * scale, alpha=0.25, label="+/-3 sigma (from P)")
        a.set_xlabel("time [s]"); a.set_ylabel(f"{name} error [{unit}]"); a.set_title(f"{name} error"); a.legend()
    fig.tight_layout()
    fig.savefig(fname, dpi=150)


if __name__ == "__main__":
    model = world.build_world()
    data = mujoco.MjData(model)

#     for label, k in [("sim as-is", 0.0), ("injected noise k=0.02", 0.02)]:
#         t, truth, est, sigma = run_baseline(model, data, laps=3, noise_k=k, seed=0)
#         report(label, t, truth, est, sigma)
#         plot(label, t, truth, est, sigma, f"baseline_{'asis' if k == 0 else 'noisy'}.png")
#     plt.show()
    for label, k, cw in [("sim as-is", 0.0, False), ("injected noise k=0.02", 0.02, False),
                     ("clockwise as-is", 0.0, True), ("clockwise injected k=0.02", 0.02, True)]:
            t, truth, est, sigma = run_baseline(model, data, laps=3, noise_k=k, seed=0, clockwise=cw)
            report(label, t, truth, est, sigma)
    plot(label, t, truth, est, sigma, f"baseline_{label.replace(' ', '_').replace('=', '')}.png")
    plt.show()
    # TODO (next):
    #  - repeat the noisy case over several seeds and report mean +/- std of the final error
    #  - add a second path (e.g. a figure-eight or square) so the result isn't specific to one circle
    #  - the same loop is where the update step plugs in later: after ekf.predict, call detect_markers on a
    #    rendered frame every 0.1 s and pass each observation to ekf.update(obs.id, obs.z, obs.R)