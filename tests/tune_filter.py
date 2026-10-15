"""Section 9: tune Q (and check R and the gate) from consistency statistics, not by guesswork.

Design (this keeps the sweep cheap):
  1. RECORD once per (path, direction, camera seed): the clean encoder increments per 20 Hz predict step, the true
     pose (evaluation only), and every marker detection (id, z, R) per camera frame. None of that depends on the
     filter settings, so the simulator only has to run once for each recording.
  2. REPLAY offline for every setting: add the odometry noise (same random numbers for every setting, so the
     comparison is fair), run the EKF, and compute NEES, NIS, gate rejections and RMSE. No simulator needed.

Two sets of recordings:
  tune       circle radius 0.4 m, both directions, 3 camera seeds   -> choose settings here
  validation circle radius 0.3 m at a different speed, both directions -> check the choice on a path it never saw

Odometry noise levels (noise_k): injected multiplicative noise on each predict interval.
  0.00  the simulator as it is (after calibration). Whatever NEES excess remains here is UNMODELLED error.
  0.02  moderate extra noise, 0.05 heavy.
Careful: for noise_k > 0 the "right" K_SLIP is just noise_k (the filter's noise model matches the injected noise by
construction), so those rows check the machinery, they do not choose K_SLIP for you. The value to defend in the
report is the one that keeps the filter consistent at the disturbance level you claim to handle.

Settings swept: K_SLIP, plus a constant process-noise floor (position m/sqrt(s), heading rad/sqrt(s)) that is added
every predict step to represent unmodelled disturbance. Also R_scale (obs.R multiplier) is available but starts at 1,
since the validation of obs.R showed it is already conservative.

Run from the project root:  python -m tests.tune_filter
The recordings take a few minutes the first time and are cached in sweep_recordings.pkl.
"""
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mujoco
import numpy as np

from src import config as cfg
from src.simulation import world
from src.estimation.estimation import EKF, EncoderOdometry
from src.perception.perception import detect_markers, wrap
from tests.deadreckoning_test import PREDICT_DT, circle_wheel_speeds, set_wheel_speeds

REC_FILE = Path("sweep_recordings.pkl")
CAMERA_EVERY = int(round((1.0 / cfg.CAMERA_RATE) / PREDICT_DT))
CENTRE = (0.0, -0.2)

NEES_BOUNDS = (0.216, 9.348)    # 95%, 3 dof
NIS_BOUNDS = (0.0506, 7.378)    # 95%, 2 dof

K_GRID = [0.005, 0.01, 0.02, 0.03, 0.05]
FLOORS = {"none": (0.0, 0.0), "low": (0.001, np.deg2rad(0.1)), "med": (0.003, np.deg2rad(0.3))}  # (m/sqrt(s), rad/sqrt(s))
NOISE_LEVELS = [0.0, 0.02, 0.05]
R_SCALE = 1.0


# Recording

def record_run(model, data, camera, radius, speed, clockwise, seed, laps=3):
    rng = np.random.default_rng(seed)
    start = np.array([CENTRE[0] - radius if clockwise else CENTRE[0] + radius, CENTRE[1], np.pi / 2])
    world.set_pose(model, data, *start)
    odo = EncoderOdometry()
    odo.increments(world.read_encoders(model, data))
    wl, wr = circle_wheel_speeds(speed, -radius if clockwise else radius)
    set_wheel_speeds(data, wl, wr)

    steps = int(round(PREDICT_DT / model.opt.timestep))
    n = int(laps * 2 * np.pi * radius / speed / PREDICT_DT)
    acc = np.zeros(2)
    inc, truth, frames = [], [], {}
    for i in range(n):
        for _ in range(steps):
            mujoco.mj_step(model, data)
            acc += odo.increments(world.read_encoders(model, data))
        inc.append(acc.copy()); acc[:] = 0
        truth.append(world.true_pose(model, data))               # EVALUATION ONLY
        if (i + 1) % CAMERA_EVERY == 0:
            obs = detect_markers(camera.render(data, rng))
            if obs:
                frames[i] = [(o.id, o.z.copy(), o.R.copy()) for o in obs]
    set_wheel_speeds(data, 0.0, 0.0)
    return dict(start=start, inc=np.array(inc), truth=np.array(truth), frames=frames,
                radius=radius, clockwise=clockwise, seed=seed)


def record_all():
    model = world.build_world()
    data = mujoco.MjData(model)
    camera = world.Camera(model)
    recs = []
    for cw in (False, True):
        for seed in (0, 1, 2):
            print(f"recording tune  r=0.40 cw={cw} seed={seed}")
            r = record_run(model, data, camera, 0.4, 0.132, cw, seed)
            r["set"] = "tune"; recs.append(r)
    for cw in (False, True):
        for seed in (10, 11):
            print(f"recording valid r=0.30 cw={cw} seed={seed}")
            r = record_run(model, data, camera, 0.3, 0.10, cw, seed)
            r["set"] = "validation"; recs.append(r)
    camera.close()
    return recs


# Replay

class TunedEKF(EKF):
    """EKF plus a constant process-noise floor (per sqrt-second), added at every predict step."""

    def __init__(self, x0, P0, k, qfloor=(0.0, 0.0)):
        super().__init__(x0, P0, k=k)
        self.q_extra = np.diag([qfloor[0] ** 2, qfloor[0] ** 2, qfloor[1] ** 2]) * PREDICT_DT

    def predict(self, dphi_L, dphi_R):
        super().predict(dphi_L, dphi_R)
        self.P = self.P + self.q_extra


def replay(rec, k, qfloor, noise_k, noise_seed, r_scale=R_SCALE):
    rng = np.random.default_rng(1000 + noise_seed)
    noisy = rec["inc"] * (1.0 + noise_k * rng.standard_normal(rec["inc"].shape))
    P0 = np.diag(np.square([0.01, 0.01, np.deg2rad(1.0)]))
    ekf = TunedEKF(rec["start"].copy(), P0, k, qfloor)
    est, Ps = [], []
    for i in range(len(noisy)):
        ekf.predict(*noisy[i])
        for mid, z, R in rec["frames"].get(i, ()):
            ekf.update(mid, z, R * r_scale)
        est.append(ekf.x.copy()); Ps.append(ekf.P.copy())
    est, Ps = np.array(est), np.array(Ps)

    e = est - rec["truth"]
    e[:, 2] = wrap(e[:, 2])
    nees = np.array([ei @ np.linalg.solve(P, ei) for ei, P in zip(e, Ps)])
    acc = [l for l in ekf.log if l["accepted"]]
    nis = np.array([l["nis"] for l in acc]) if acc else np.array([np.nan])
    last = slice(len(e) * 2 // 3, None)
    return dict(nees=nees.mean(), nees_in=((nees >= NEES_BOUNDS[0]) & (nees <= NEES_BOUNDS[1])).mean(),
                nis=nis.mean(), gate=1 - len(acc) / max(len(ekf.log), 1),
                rmse_xy=np.sqrt((e[last, :2] ** 2).sum(axis=1).mean()), rmse_th=np.sqrt((e[last, 2] ** 2).mean()))


def sweep(recs, label):
    print(f"\n=========== {label}: {len(recs)} recordings ===========")
    for noise_k in NOISE_LEVELS:
        print(f"\n--- injected odometry noise k = {noise_k} ---")
        print("   K_SLIP  floor |  NEES  in95% |   NIS | gate% | RMSE xy [mm]  theta [deg]   (last third; * = NEES 2.5-4, gate <= 2%)")
        for fname, qf in FLOORS.items():
            for k in K_GRID:
                m = [replay(r, k, qf, noise_k, i) for i, r in enumerate(recs)]
                avg = {key: float(np.nanmean([x[key] for x in m])) for key in m[0]}
                good = "*" if 2.5 <= avg["nees"] <= 4.0 and avg["gate"] <= 0.02 else " "
                print(f"   {k:6.3f}  {fname:5s} | {avg['nees']:5.2f}  {100 * avg['nees_in']:4.0f}% | {avg['nis']:5.2f} | "
                      f"{100 * avg['gate']:4.1f}  | {1e3 * avg['rmse_xy']:8.2f}   {np.rad2deg(avg['rmse_th']):8.3f}   {good}")


if __name__ == "__main__":
    if REC_FILE.exists():
        recs = pickle.load(open(REC_FILE, "rb"))
        print(f"loaded {len(recs)} cached recordings from {REC_FILE}")
    else:
        recs = record_all()
        pickle.dump(recs, open(REC_FILE, "wb"))
    sweep([r for r in recs if r["set"] == "tune"], "TUNE (r = 0.4 m)")
    sweep([r for r in recs if r["set"] == "validation"], "VALIDATION (r = 0.3 m, held out)")

    # How to choose:
    #  - Expected values for a consistent filter: mean NEES about 3, mean NIS about 1.9 (accepted updates only, so a
    #    touch under 2), gate rejections about 1%, ~95% of samples inside the 95% bounds.
    #  - Pick the SMALLEST process noise that gets there at the disturbance level you will claim (noise_k = 0.02 is a
    #    sensible headline), then check the same setting still behaves at 0.00 and 0.05.
    #  - Confirm the choice in the VALIDATION table (a path it was not tuned on).
    #  - Put the chosen K_SLIP and floor into estimation.py, then rerun full_filter_circle.py for the final results.