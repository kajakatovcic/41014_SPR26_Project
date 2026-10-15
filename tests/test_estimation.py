# Evaluate the EKF (src/estimation/estimation.py) against ground truth over repeated missions (marking criterion C).
# Each trial runs the full loop of main.py with the EKF driving the controller, and draws new odometry errors
# (each wheel's radius off by up to +/-0.8%) and new sensor noise so the filter is judged across conditions it
# was not tuned for.
#   Part 1 - accuracy: EKF against dead reckoning on the same odometry and whether each mission reached its goal
#   Part 2 - consistency (Week 7 D.3): NEES which should average n = 3 for a consistent filter, NIS (2 per marker
#            update), 2-sigma coverage, and whether consecutive camera measurements are actually independent
#   Part 3 - tuning (Week 7 D.4): the recorded sensor data is replayed through the EKF with the odometry noise
#            (Q) and the camera noise (R) scaled up and down, to see what accuracy and consistency each choice gives
# Usage: python tests/test_estimation.py [--trials 5] [--no-plot]
import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import chi2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import main                                           
from src import config as cfg                               
from src.control import model                             
from src.estimation.estimation import EKF                    

Q_SCALES = [0.1, 0.3, 1.0, 3.0, 10.0]


def errors(est, true):
    e = est - true
    e[:, 2] = model.wrap(e[:, 2])
    return e


def nees(e, P):
    return np.array([ei @ np.linalg.solve(Pi, ei) for ei, Pi in zip(e, P)])


def replay(log, q_scale=cfg.EKF_Q_SCALE, r_scale=cfg.EKF_R_SCALE):
    """Run the EKF again on a mission's recorded odometry and marker observations with other noise scales."""
    ekf = EKF(cfg.START_POSE, q_scale=q_scale, r_scale=r_scale)
    camera_every = int(round(1 / cfg.CAMERA_RATE / cfg.CONTROL_DT))
    X, P = [], []
    for k, (u, obs) in enumerate(zip(log.odo, log.obs)):
        ekf.predict(u, cfg.CONTROL_DT)
        if k % camera_every == 0:
            ekf.update(obs)
        X.append(ekf.x.copy()); P.append(ekf.P.copy())
    return np.array(X), np.array(P), ekf


def innovation_autocorrelation(ekfs, max_lag=5):
    """Correlation between the whitened innovations of the same marker 1, 2, ... camera frames apart. Independent
    measurements give ~0. The EKF assumes 0."""
    pairs = {lag: [] for lag in range(1, max_lag + 1)}
    for ekf in ekfs:
        by_id = {}
        for frame, mid, w in ekf.innovations:
            by_id.setdefault(mid, {})[frame] = w
        for seq in by_id.values():
            for frame, w in seq.items():
                for lag in pairs:
                    if frame + lag in seq:
                        pairs[lag].append((w, seq[frame + lag]))
    out = {}
    for lag, p in pairs.items():
        a, b = np.array([x for x, _ in p]), np.array([y for _, y in p])
        out[lag] = [np.corrcoef(a[:, i], b[:, i])[0, 1] for i in range(2)] # range, bearing
    return out


def run_trials(n, seed0=100):
    rng = np.random.default_rng(seed0)
    trials = []
    for i in range(n):
        scale_err = rng.uniform(-0.008, 0.008, 2)
        log, ctrl = main.run(seed=seed0 + i, encoder_errors=(scale_err, cfg.SIM_WHEEL_SLIP_SIGMA))
        trials.append((log, ctrl, scale_err))
        a = log.arrays()
        dist, lat, _ = main.docking_errors(a["true"][-1])
        print(f"trial {i}: wheel radius errors ({100 * scale_err[0]:+.2f}%, {100 * scale_err[1]:+.2f}%), "
              f"{a['t'][-1]:.0f} s, {'docked' if ctrl.done else 'NOT docked'}, true docking error "
              f"{1000 * np.hypot(dist, lat):.0f} mm")
    return trials


def part1(trials):
    print("\n=== Part 1: accuracy, EKF against dead reckoning on the same odometry")
    print(f'{"trial":>6} | {"EKF RMS":>16} {"EKF max":>9} | {"dead reckoning RMS":>20} {"final":>18}')
    rms_ekf, rms_dr, ok = [], [], True
    for i, (log, ctrl, _) in enumerate(trials):
        a = log.arrays()
        e, d = errors(a["est"], a["true"]), errors(a["dr"], a["true"])
        ep, dp = np.linalg.norm(e[:, :2], axis=1), np.linalg.norm(d[:, :2], axis=1)
        rms_ekf.append(np.sqrt(np.mean(ep ** 2))); rms_dr.append(np.sqrt(np.mean(dp ** 2)))
        print(f"{i:>6} | {1000 * rms_ekf[-1]:>6.1f} mm {np.degrees(np.sqrt(np.mean(e[:, 2] ** 2))):>5.2f} deg "
              f"{1000 * ep.max():>6.1f} mm | {1000 * rms_dr[-1]:>8.0f} mm {np.degrees(np.sqrt(np.mean(d[:, 2] ** 2))):>6.1f} deg "
              f"{1000 * dp[-1]:>8.0f} mm {np.degrees(abs(d[-1, 2])):>5.1f} deg")
        ok &= bool(ctrl.done)
    print(f"mean RMS position error: EKF {1000 * np.mean(rms_ekf):.1f} mm, dead reckoning "
          f"{1000 * np.mean(rms_dr):.0f} mm")
    ok &= np.mean(rms_ekf) < 0.25 * np.mean(rms_dr)
    print("every mission docked and the EKF is more than 4x better than dead reckoning: "
          + ("PASS" if ok else "FAIL"))
    return ok


def consistency(trials, q_scale=cfg.EKF_Q_SCALE, r_scale=cfg.EKF_R_SCALE):
    """NEES (mean over all steps of all trials), 2-sigma coverage, NIS, RMS errors for one pair of noise scales."""
    all_nees, cov_pos, cov_th, nis, rms_p, rms_t, rej = [], [], [], [], [], [], 0
    per_time, ekfs = [], []
    for log, _, _ in trials:
        X, P, ekf = replay(log, q_scale, r_scale)
        ekfs.append(ekf)
        e = errors(X, np.array(log.true))
        n = nees(e, P)
        all_nees.append(n); per_time.append(n)
        ep = np.linalg.norm(e[:, :2], axis=1)
        cov_pos.append(np.mean(ep < 2 * np.sqrt(P[:, 0, 0] + P[:, 1, 1])))
        cov_th.append(np.mean(np.abs(e[:, 2]) < 2 * np.sqrt(P[:, 2, 2])))
        nis += ekf.nis; rej += ekf.rejected
        rms_p.append(np.sqrt(np.mean(ep ** 2))); rms_t.append(np.sqrt(np.mean(e[:, 2] ** 2)))
    return dict(nees=np.mean(np.concatenate(all_nees)), cov_pos=np.mean(cov_pos), cov_th=np.mean(cov_th),
                nis=np.mean(nis), rejected=rej, rms_pos=np.mean(rms_p), rms_th=np.mean(rms_t), per_time=per_time,
                ekfs=ekfs)


def part2(trials):
    print(f"\n=== Part 2: consistency (Week 7 D.3), Q x{cfg.EKF_Q_SCALE:g}, R (range, bearing) x{cfg.EKF_R_SCALE}")
    c = consistency(trials)
    n_steps = sum(len(p) for p in c["per_time"])
    lo, hi = chi2.ppf([0.025, 0.975], 3)
    inside = np.mean(np.concatenate(c["per_time"]) < hi)
    print(f"mean NEES {c['nees']:.2f} (consistent filter: 3; above means overconfident, below conservative)")
    print(f"NEES below the 97.5% chi-square bound ({hi:.2f}) on {100 * inside:.0f}% of {n_steps} steps (expect 97.5%)")
    print(f"mean NIS {c['nis']:.2f} (expect 2), {c['rejected']} marker updates rejected by the gate")
    print(f"2-sigma coverage: position {100 * c['cov_pos']:.0f}%, heading {100 * c['cov_th']:.0f}% (expect ~95%)")
    verdict = "OVERCONFIDENT" if c["nees"] > 4.5 else "conservative" if c["nees"] < 1.5 else "consistent"
    print(f"verdict: {verdict}")

    ac = innovation_autocorrelation(c["ekfs"])
    print("autocorrelation of the same marker's whitened innovations, k camera frames apart (independent: 0):")
    print("   " + "   ".join(f"k={lag}: range {r:+.2f} bearing {b:+.2f}" for lag, (r, b) in ac.items() if lag <= 3))
    ok = verdict != "OVERCONFIDENT" and c["cov_pos"] > 0.85 and c["cov_th"] > 0.85
    print("PASS" if ok else "FAIL")
    return ok, c


def sweep_row(label, c):
    print(f"{label:>10} {1000 * c['rms_pos']:>13.1f} {np.degrees(c['rms_th']):>18.2f} {c['nees']:>10.2f} "
          f"{100 * c['cov_pos']:>11.0f}% {100 * c['cov_th']:>15.0f}% {c['nis']:>6.2f} {c['rejected']:>9d}")


def part3(trials):
    print("\n=== Part 3: tuning (Week 7 D.4) replayed on the recorded missions")
    header = (f'{"RMS pos [mm]":>13} {"RMS heading [deg]":>18} {"mean NEES":>10} {"2-sigma pos":>12} '
              f'{"2-sigma heading":>16} {"NIS":>6} {"rejected":>9}')
    print(f'odometry noise, R (range, bearing) x{cfg.EKF_R_SCALE}:\n{"Q scale":>10} ' + header)
    rows = []
    for s in Q_SCALES:
        c = consistency(trials, q_scale=s)
        rows.append((s, c))
        sweep_row(f"{s:g}", c)
    print(f'camera noise, Q x{cfg.EKF_Q_SCALE:g}:\n{"R scale":>10} ' + header)
    for r in [(1, 1), (3, 1), (5, 1), (10, 1), (3, 3), (5, 5)]:
        sweep_row(f"{r[0]:g}, {r[1]:g}", consistency(trials, r_scale=r))
    return rows


def plot(trials, c2, rows):
    fig, ax = plt.subplots(2, 2, figsize=(13, 9))

    a = ax[0, 0]
    for i, (log, _, _) in enumerate(trials):
        arr = log.arrays()
        e, d = errors(arr["est"], arr["true"]), errors(arr["dr"], arr["true"])
        a.plot(arr["t"], 1000 * np.linalg.norm(e[:, :2], axis=1), color="tab:blue", lw=1,
               label="EKF" if i == 0 else None)
        a.plot(arr["t"], 1000 * np.linalg.norm(d[:, :2], axis=1), ":", color="tab:purple", lw=1,
               label="dead reckoning" if i == 0 else None)
    a.set_yscale("log"); a.set_xlabel("time [s]"); a.set_ylabel("position error [mm]"); a.legend()
    a.set_title("Part 1: position error, every trial")

    a = ax[0, 1]
    for i, n in enumerate(c2["per_time"]):
        a.plot(np.arange(len(n)) * cfg.CONTROL_DT, n, lw=0.8, alpha=0.7)
    a.axhline(3, color="k", label="expected (n = 3)")
    a.axhline(chi2.ppf(0.975, 3), color="k", ls="--", label="97.5% bound")
    a.set_yscale("log"); a.set_xlabel("time [s]"); a.set_ylabel("NEES"); a.legend()
    a.set_title("Part 2: NEES over time, every trial")

    a = ax[1, 0]
    s = [r[0] for r in rows]
    a.plot(s, [1000 * r[1]["rms_pos"] for r in rows], "o-", label="RMS position [mm]")
    a.set_xscale("log"); a.set_xlabel("Q_d scale"); a.set_ylabel("RMS position error [mm]")
    a2 = a.twinx()
    a2.plot(s, [r[1]["nees"] for r in rows], "s--", color="tab:red", label="mean NEES")
    a2.axhline(3, color="tab:red", lw=0.8, label="NEES = 3 (consistent)"); a2.set_yscale("log")
    a2.set_ylabel("mean NEES")
    a.legend(loc="upper left"); a2.legend(loc="upper right")
    a.set_title("Part 3: accuracy and consistency against Q_d scale")

    a = ax[1, 1]
    a.plot(s, [100 * r[1]["cov_pos"] for r in rows], "o-", label="position")
    a.plot(s, [100 * r[1]["cov_th"] for r in rows], "s-", label="heading")
    a.axhline(95, color="k", ls="--", lw=0.8)
    a.set_xscale("log"); a.set_xlabel("Q_d scale"); a.set_ylabel("2-sigma coverage [%]"); a.legend()
    a.set_title("Part 3: 2-sigma coverage against Q_d scale")
    plt.tight_layout(); plt.show()


def cli():
    parser = argparse.ArgumentParser(description="EKF evaluation over repeated missions")
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    trials = run_trials(args.trials)
    ok1 = part1(trials)
    ok2, c2 = part2(trials)
    rows = part3(trials)
    print(f'\nestimation test: {"PASS" if ok1 and ok2 else "FAIL"}')
    if not args.no_plot:
        plot(trials, c2, rows)


if __name__ == "__main__":
    cli()
