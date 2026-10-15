"""Section 11 (with part E): robustness of the full EKF to disturbances, compared with predict-only dead reckoning.

Replays the recordings cached by tune_filter.py (sweep_recordings.pkl), so it needs no simulator and runs in about
a minute. Each scenario injects a disturbance into the RECORDED data before the filter sees it:

  heavy noise        more multiplicative noise on the encoder increments
  dropout            whole camera frames discarded at random
  blackout           no detections for a stretch of time (long predict-only)
  wheel scale        the encoders over-read on one or both wheels (an unmodelled wheel-radius / slip error)
  outliers           some detections replaced by wrong range/bearing; shown with the gate on and off
  camera delay       detections arrive late and are applied to the CURRENT state (uncompensated)
  wrong start        the filter starts from a wrong pose with a matching larger covariance
  combined           several of these at once

Truth is used for evaluation only. Disturbances are injected into the data (not physical), which is a limitation to
state in the report: real wheel slip or collisions would need the simulator (part E).

Run from the project root:  python -m tests.robustness_tests   (after tune_filter.py has created the cache)
"""
import csv
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np

from src.estimation.estimation import EKF, GATE_CHI2
from src.perception.perception import wrap
from tests.deadreckoning_test import PREDICT_DT

REC_FILE = Path("sweep_recordings.pkl")
K_VALUES = (0.03, 0.05)       # K_SLIP values to compare (0.03 = chosen, 0.05 = more conservative)
N_SEEDS = 3                   # disturbance/noise draws per recording
P0_DEFAULT = (0.01, 0.01, np.deg2rad(1.0))
INIT_ERR = (0.15, -0.10, np.deg2rad(10))

SCENARIOS = [
    ("nominal (2% noise)", {}),
    ("heavy noise (5%)", dict(noise_k=0.05)),
    ("30% frames dropped", dict(dropout=0.3)),
    ("60% frames dropped", dict(dropout=0.6)),
    ("detection blackout 15-30 s", dict(blackout=(15.0, 30.0))),
    ("left wheel +2% scale", dict(wheel_scale=(1.02, 1.0))),
    ("both wheels +3% scale", dict(wheel_scale=(1.03, 1.03))),
    ("5% outliers, gate ON", dict(outliers=0.05)),
    ("5% outliers, gate OFF", dict(outliers=0.05, gate=np.inf)),
    ("camera delay 100 ms", dict(delay=2)),
    ("camera delay 300 ms", dict(delay=6)),
    ("wrong start", dict(init_error=INIT_ERR, P0_sigma=(0.2, 0.2, np.deg2rad(15)))),
    ("combined", dict(noise_k=0.05, dropout=0.3, outliers=0.02, wheel_scale=(1.02, 1.0))),
]


def replay(rec, k, sc, seed, updates=True):
    n = len(rec["inc"])
    odo_rng = np.random.default_rng(1000 + seed)      # same odometry noise in every scenario
    ev_rng = np.random.default_rng(5000 + seed)       # dropout / outlier draws
    ws = np.array(sc.get("wheel_scale", (1.0, 1.0)))
    noisy = rec["inc"] * ws * (1.0 + sc.get("noise_k", 0.02) * odo_rng.standard_normal(rec["inc"].shape))

    events = {}
    if updates:
        for i in sorted(rec["frames"]):
            t = (i + 1) * PREDICT_DT
            if ev_rng.random() < sc.get("dropout", 0.0):
                continue
            b = sc.get("blackout")
            if b and b[0] <= t < b[1]:
                continue
            for mid, z, R in rec["frames"][i]:
                z = z.copy()
                if ev_rng.random() < sc.get("outliers", 0.0):
                    z += np.array([ev_rng.choice([-1, 1]) * ev_rng.uniform(0.3, 1.0),
                                   ev_rng.choice([-1, 1]) * np.deg2rad(ev_rng.uniform(5, 20))])
                j = i + sc.get("delay", 0)             # a late detection is applied to the CURRENT state
                if j < n:
                    events.setdefault(j, []).append((mid, z, R))

    x0 = rec["start"] + np.asarray(sc.get("init_error", (0, 0, 0)), float)
    x0[2] = wrap(x0[2])
    P0 = np.diag(np.square(sc.get("P0_sigma", P0_DEFAULT)))
    ekf = EKF(x0, P0, gate=sc.get("gate", GATE_CHI2), k=k)
    est, Ps = [], []
    for j in range(n):
        ekf.predict(*noisy[j])
        for mid, z, R in events.get(j, ()):
            ekf.update(mid, z, R)
        est.append(ekf.x.copy()); Ps.append(ekf.P.copy())
    return np.array(est), np.array(Ps), ekf


def errors(rec, est):
    e = est - rec["truth"]
    e[:, 2] = wrap(e[:, 2])
    return e


def metrics(rec, est, Ps, ekf, init_scenario):
    e = errors(rec, est)
    n = len(e)
    last, mid = slice(2 * n // 3, None), slice(n // 3, None)
    out = dict(xy=np.sqrt((e[last, :2] ** 2).sum(axis=1).mean()), th=np.sqrt((e[last, 2] ** 2).mean()),
               maxpos=np.hypot(e[mid, 0], e[mid, 1]).max())
    if Ps is not None:
        out["nees"] = np.mean([ei @ np.linalg.solve(P, ei) for ei, P in zip(e, Ps)])
        nis = [l["nis"] for l in ekf.log if l["accepted"]]
        out["nis"] = np.mean(nis) if nis else np.nan
        out["gate"] = 1 - len(nis) / max(len(ekf.log), 1)
    if init_scenario:   # time until position < 20 mm and heading < 1 deg for good
        ok = (np.hypot(e[:, 0], e[:, 1]) < 0.02) & (np.abs(e[:, 2]) < np.deg2rad(1))
        bad = np.flatnonzero(~ok)
        out["conv"] = np.nan if len(bad) and bad[-1] == n - 1 else ((bad[-1] + 1) if len(bad) else 0) * PREDICT_DT
    return out


def mean_of(ms, key):
    vals = [m[key] for m in ms if key in m]
    return float(np.nanmean(vals)) if vals else np.nan


def run_all(recs, k):
    rows = []
    print(f"\n================ K_SLIP = {k} ({len(recs)} recordings x {N_SEEDS} draws per scenario) ================")
    print("RMSE over the last third of each run; max error over the last two thirds (after settling).")
    print(f"{'scenario':32s} | predict-only xy[mm] th[deg] | full filter xy[mm] th[deg] max[mm] | NEES   NIS  gate% | conv[s]")
    for name, sc in SCENARIOS:
        full, pred = [], []
        for ri, rec in enumerate(recs):
            for s in range(N_SEEDS):
                seed = ri * 10 + s
                est, Ps, ekf = replay(rec, k, sc, seed, updates=True)
                full.append(metrics(rec, est, Ps, ekf, "init_error" in sc))
                est_p, _, _ = replay(rec, k, sc, seed, updates=False)
                pred.append(metrics(rec, est_p, None, None, False))
        r = dict(scenario=name, k=k,
                 pred_xy=1e3 * mean_of(pred, "xy"), pred_th=np.rad2deg(mean_of(pred, "th")),
                 xy=1e3 * mean_of(full, "xy"), th=np.rad2deg(mean_of(full, "th")), maxpos=1e3 * mean_of(full, "maxpos"),
                 nees=mean_of(full, "nees"), nis=mean_of(full, "nis"), gate=100 * mean_of(full, "gate"),
                 conv=mean_of(full, "conv"))
        rows.append(r)
        conv = f"{r['conv']:5.1f}" if not np.isnan(r["conv"]) else "  -  "
        print(f"{name:32s} | {r['pred_xy']:16.1f} {r['pred_th']:7.2f} | {r['xy']:15.1f} {r['th']:7.2f} {r['maxpos']:7.1f} | "
              f"{r['nees']:5.1f} {r['nis']:5.2f} {r['gate']:5.1f} | {conv}")
    return rows


if __name__ == "__main__":
    if not REC_FILE.exists():
        sys.exit(f"{REC_FILE} not found: run tests/tune_filter.py first to record the data")
    recs = pickle.load(open(REC_FILE, "rb"))
    all_rows = []
    for k in K_VALUES:
        all_rows += run_all(recs, k)

    with open("robustness_results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0]))
        w.writeheader(); w.writerows(all_rows)

    rows = [r for r in all_rows if r["k"] == K_VALUES[0]]
    x = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x - 0.2, [r["pred_xy"] for r in rows], 0.4, label="predict-only", color="gray")
    ax.bar(x + 0.2, [r["xy"] for r in rows], 0.4, label="full filter")
    ax.set_yscale("log"); ax.set_ylabel("position RMSE, last third [mm]")
    ax.set_xticks(x); ax.set_xticklabels([r["scenario"] for r in rows], rotation=40, ha="right")
    ax.set_title(f"Robustness (K_SLIP = {K_VALUES[0]})"); ax.legend()
    fig.tight_layout(); fig.savefig("robustness_summary.png", dpi=150)
    plt.show()

    # Reading the table:
    #  - the full filter should stay at a few mm in every row except where the data is genuinely missing
    #  - predict-only shows what the camera is saving you from (wheel-scale rows, wrong start)
    #  - 'outliers, gate OFF' vs 'gate ON' is the evidence that the gate works (the sweep never exercised it)
    #  - camera delay is applied uncompensated: error growth with delay motivates keeping a short state history and
    #    applying each detection at its own timestamp (retrodiction), a natural 'future work' item
    #  - NEES >> 3 in a row means the filter is overconfident under that disturbance; compare the two K_SLIP blocks