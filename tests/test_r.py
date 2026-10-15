"""Sections 6, 7, 9, 10 (first pass): the full EKF (odometry + ArUco range-bearing) against predict-only, on the
open-loop circle from the baseline.

Per run, one simulation drives the robot and TWO filters see exactly the same encoder data:
  - predict-only : dead reckoning (the baseline)
  - full         : predict at 20 Hz, plus an update from every detected marker every 0.1 s (camera rate)
The true pose is used for EVALUATION only (errors, NEES). The filters never see it. The start pose is the known
placed pose (plus an optional deliberate error for the convergence test).

Reports: RMSE, number of updates and gated detections, longest predict-only stretch, NIS and NEES consistency.
Saves one figure per case.

Run from the project root:  python -m tests.full_filter_circle
(it imports helpers from tests/baseline_deadreckoning.py, so keep both files in tests/)
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import mujoco
import numpy as np

from src import config as cfg
from src.simulation import world
from src.estimation.estimation import EKF, EncoderOdometry
from src.perception.perception import detect_markers, wrap
from tests.deadreckoning_test import (PREDICT_DT, RADIUS, SPEED, START, circle_wheel_speeds, errors,
                                          set_wheel_speeds)

CAMERA_PERIOD = 1.0 / cfg.CAMERA_RATE
START_CW = (-0.4, -0.2, np.pi / 2)

# 95% two-sided chi-square bounds, for judging consistency
NIS_BOUNDS = (0.0506, 7.378)    # 2 dof (range, bearing)
NEES_BOUNDS = (0.216, 9.348)    # 3 dof (x, y, theta)


def run_filter(model, data, camera, laps=3, noise_k=0.0, seed=0, clockwise=False,
               init_error=(0.0, 0.0, 0.0), P0_sigma=(0.01, 0.01, np.deg2rad(1.0)), dropout=0.0):
    """Drive `laps` laps open-loop with two filters. dropout = probability that a whole camera frame is discarded."""
    rng = np.random.default_rng(seed)
    start = np.array(START_CW if clockwise else START, float)
    radius = -RADIUS if clockwise else RADIUS
    world.set_pose(model, data, *start)

    x0 = start + np.asarray(init_error, float)           # the filters' initial belief (may be deliberately wrong)
    x0[2] = wrap(x0[2])
    P0 = np.diag(np.square(P0_sigma))
    ekf_full, ekf_pred = EKF(x0.copy(), P0.copy()), EKF(x0.copy(), P0.copy())

    odo = EncoderOdometry()
    odo.increments(world.read_encoders(model, data))
    wl, wr = circle_wheel_speeds(SPEED, radius)
    set_wheel_speeds(data, wl, wr)

    steps_per_predict = int(round(PREDICT_DT / model.opt.timestep))
    frame_every = int(round(CAMERA_PERIOD / PREDICT_DT))
    n_predicts = int(laps * 2 * np.pi * abs(radius) / SPEED / PREDICT_DT)

    acc = np.zeros(2)
    out = dict(t=[], truth=[], est_full=[], est_pred=[], P_full=[], updates=[], frames=[])
    t_cam = 0.0
    for k in range(n_predicts):
        for _ in range(steps_per_predict):
            mujoco.mj_step(model, data)
            acc += odo.increments(world.read_encoders(model, data))
        noisy = acc * (1.0 + noise_k * rng.standard_normal(2))   # same noisy increments to both filters
        ekf_full.predict(*noisy)
        ekf_pred.predict(*noisy)
        acc[:] = 0
        t = (k + 1) * PREDICT_DT

        # Camera frame at 10 Hz. The frame is rendered at this same simulation time, so the filter state (just
        # predicted to t) matches the image timestamp. Latency would break that (a robustness test for part E).
        if (k + 1) % frame_every == 0:
            n_obs = 0
            if rng.random() >= dropout:
                t0 = time.perf_counter()
                obs_list = detect_markers(camera.render(data, rng))
                t_cam += time.perf_counter() - t0
                for obs in obs_list:
                    ekf_full.update(obs.id, obs.z, obs.R)
                    e = ekf_full.log[-1]
                    out["updates"].append((t, obs.id, e["nis"], e["accepted"]))
                    n_obs += 1
            out["frames"].append((t, n_obs))

        out["t"].append(t)
        out["truth"].append(world.true_pose(model, data))        # EVALUATION ONLY
        out["est_full"].append(ekf_full.x.copy())
        out["est_pred"].append(ekf_pred.x.copy())
        out["P_full"].append(ekf_full.P.copy())
    set_wheel_speeds(data, 0.0, 0.0)

    for key in ("t", "truth", "est_full", "est_pred", "P_full", "frames"):
        out[key] = np.array(out[key])
    out["updates"] = np.array(out["updates"], float).reshape(-1, 4)  # t, id, nis, accepted
    out["sec_per_frame"] = t_cam / max(len(out["frames"]), 1)
    return out


def analyse(label, r):
    t, truth = r["t"], r["truth"]
    e_full, e_pred = errors(truth, r["est_full"]), errors(truth, r["est_pred"])
    sigma = np.sqrt(np.array([np.diag(P) for P in r["P_full"]]))
    nees = np.array([e @ np.linalg.solve(P, e) for e, P in zip(e_full, r["P_full"])])

    last = t > t[-1] * 2 / 3                                    # last third of the run (filter has settled)
    rm = lambda e, s=slice(None): np.sqrt((e[s] ** 2).mean(axis=0))
    print(f"\n[{label}]")
    for name, e in (("predict-only", e_pred), ("full filter ", e_full)):
        a, b = rm(e), rm(e, last)
        print(f"  {name}  RMSE all : x {a[0]*1e3:6.1f} mm  y {a[1]*1e3:6.1f} mm  theta {np.rad2deg(a[2]):6.3f} deg"
              f" | last third: x {b[0]*1e3:6.1f}  y {b[1]*1e3:6.1f}  theta {np.rad2deg(b[2]):6.3f}")
        print(f"  {name}  final position error {np.hypot(e[-1, 0], e[-1, 1])*1e3:.1f} mm, "
              f"final heading error {np.rad2deg(e[-1, 2]):+.3f} deg")

    u = r["updates"]
    frames = r["frames"]
    seen = frames[:, 1] > 0
    gaps, run = [], 0
    for s in seen:
        run = 0 if s else run + 1
        gaps.append(run)
    print(f"  frames with a detection: {100 * seen.mean():.0f}% ({int(seen.sum())}/{len(seen)}); "
          f"longest predict-only stretch {max(gaps) * CAMERA_PERIOD:.1f} s; "
          f"{len(u)} detections, {int((u[:, 3] == 0).sum()) if len(u) else 0} gated out")
    if len(u):
        nis = u[u[:, 3] == 1, 2]
        inside = ((nis >= NIS_BOUNDS[0]) & (nis <= NIS_BOUNDS[1])).mean()
        print(f"  NIS  mean {nis.mean():.2f} (expect 2.0), {100 * inside:.0f}% inside the 95% bounds (expect ~95%)")
    inside_n = ((nees >= NEES_BOUNDS[0]) & (nees <= NEES_BOUNDS[1])).mean()
    print(f"  NEES mean {nees.mean():.2f} (expect 3.0), {100 * inside_n:.0f}% inside the 95% bounds (expect ~95%)")
    print(f"  inside own 3 sigma: x {100 * (np.abs(e_full[:, 0]) <= 3 * sigma[:, 0]).mean():.0f}% "
          f"y {100 * (np.abs(e_full[:, 1]) <= 3 * sigma[:, 1]).mean():.0f}% "
          f"theta {100 * (np.abs(e_full[:, 2]) <= 3 * sigma[:, 2]).mean():.0f}%")
    print(f"  render+detect {1e3 * r['sec_per_frame']:.0f} ms per frame")
    return e_full, e_pred, sigma, nees


def plot(label, r, e_full, e_pred, sigma, nees, fname):
    t = r["t"]
    fig, ax = plt.subplots(2, 3, figsize=(16, 8))
    a = ax[0, 0]
    a.plot(r["truth"][:, 0], r["truth"][:, 1], label="true")
    a.plot(r["est_pred"][:, 0], r["est_pred"][:, 1], ":", label="predict-only")
    a.plot(r["est_full"][:, 0], r["est_full"][:, 1], "--", label="full filter")
    for mid, T in cfg.MARKERS.items():
        if abs(T.t[0]) < 2.2 and abs(T.t[1]) < 2.2:
            a.plot(T.t[0], T.t[1], "k^")
    a.set_aspect("equal"); a.set_title("Path (markers are on the walls)"); a.legend(fontsize=8)
    for a, i, name, sc, unit in [(ax[0, 1], 0, "x", 1e3, "mm"), (ax[0, 2], 1, "y", 1e3, "mm"),
                                 (ax[1, 0], 2, "theta", 180 / np.pi, "deg")]:
        a.plot(t, e_pred[:, i] * sc, color="gray", alpha=0.7, label="predict-only error")
        a.plot(t, e_full[:, i] * sc, label="full filter error")
        a.fill_between(t, -3 * sigma[:, i] * sc, 3 * sigma[:, i] * sc, alpha=0.2, label="+/-3 sigma (full)")
        a.set_xlabel("time [s]"); a.set_ylabel(f"{name} error [{unit}]"); a.set_title(f"{name} error"); a.legend(fontsize=8)
    a = ax[1, 1]
    u = r["updates"]
    if len(u):
        ok = u[:, 3] == 1
        a.scatter(u[ok, 0], u[ok, 2], s=8, label="accepted")
        a.scatter(u[~ok, 0], u[~ok, 2], s=18, c="r", marker="x", label="gated out")
    a.axhline(NIS_BOUNDS[1], color="g", ls="--", label="95% bounds"); a.axhline(NIS_BOUNDS[0], color="g", ls="--")
    a.set_yscale("symlog", linthresh=1); a.set_xlabel("time [s]"); a.set_ylabel("NIS"); a.set_title("Innovation consistency (NIS)")
    a.legend(fontsize=8)
    a = ax[1, 2]
    a.plot(t, nees)
    a.axhline(NEES_BOUNDS[1], color="g", ls="--"); a.axhline(NEES_BOUNDS[0], color="g", ls="--")
    a.set_yscale("symlog", linthresh=1); a.set_xlabel("time [s]"); a.set_ylabel("NEES"); a.set_title("State consistency (NEES)")
    fig.suptitle(label)
    fig.tight_layout()
    fig.savefig(fname, dpi=150)


if __name__ == "__main__":
    model = world.build_world()
    data = mujoco.MjData(model)
    camera = world.Camera(model)

    cases = [
        ("ccw, exact start", dict(clockwise=False)),
        ("cw, exact start", dict(clockwise=True)),
        ("ccw, wrong start (0.15 m, -0.10 m, 10 deg)",
         dict(clockwise=False, init_error=(0.15, -0.10, np.deg2rad(10)), P0_sigma=(0.2, 0.2, np.deg2rad(15)))),
    ]
    for label, kw in cases:
        r = run_filter(model, data, camera, laps=3, noise_k=0.02, seed=0, **kw)
        e_full, e_pred, sigma, nees = analyse(label, r)
        plot(label, r, e_full, e_pred, sigma, nees, f"fullfilter_{label.split(',')[0]}_{label.split(', ')[1].split(' ')[0]}.png")
    camera.close()
    plt.show()

    # TODO (next):
    #  - several seeds per case, report mean +/- std of RMSE and of the NIS/NEES means
    #  - robustness (part E): dropout=0.3, noise_k=0.05, a deliberate 2% wheel-radius error, camera latency
    #  - tuning (Section 9): vary K_SLIP and an R inflation factor, pick values from NIS/NEES, not from this one run