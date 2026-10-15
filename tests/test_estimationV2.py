"""Section 2 (v2): odometry characterisation with signed errors, unwrapped angles, and a sweep over speeds and
directions.

Changes from v1:
  1. Straight test reports the SIGNED along-track error (positive = odometry over-reads) and the cross-track error.
  2. Angles are accumulated step by step (unwrapped), so any amount of turning gives a clean angle and percentage.
  3. Sweeps several speeds, and spins in both directions.
Also: the odometry keeps integrating for a short settle time after the wheels are commanded to stop, so the
true pose and the dead-reckoned pose cover the same motion (the robot coasts a little after ctrl goes to 0).

The true pose is used for EVALUATION only (it is how the odometry error is measured).

Run from the project root:  python -m tests.test_estimation   (or run the file directly, see the path line below)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # so `from src...` works however the file is launched

import mujoco
import numpy as np

from src import config as cfg
from src.simulation import world
from src.estimation.estimation import EncoderOdometry, motion_model
from src.perception.perception import wrap


def set_wheel_speeds(model, data, w_left, w_right):
    """Command the wheels [rad/s]. Actuators are ['wheel_left', 'wheel_right'] (checked)."""
    data.ctrl[0] = np.clip(w_left, -cfg.WHEEL_SPEED_MAX, cfg.WHEEL_SPEED_MAX)
    data.ctrl[1] = np.clip(w_right, -cfg.WHEEL_SPEED_MAX, cfg.WHEEL_SPEED_MAX)


def run_segment(model, data, w_left, w_right, duration, settle=1.0):
    """Drive at constant wheel speeds for `duration` s, then stop and keep recording for `settle` s.

    Returns a dict with the true start/end pose, the dead-reckoned end pose (started from the true start pose),
    and the unwrapped heading change according to the truth and according to the odometry.
    """
    odo = EncoderOdometry()
    p0 = world.true_pose(model, data)
    x_odo = p0.copy()
    odo.increments(world.read_encoders(model, data))  # initialise the encoder reference

    th_true_prev = p0[2]
    dth_true = 0.0  # unwrapped true heading change
    dth_odo = 0.0   # unwrapped odometry heading change

    def advance(seconds):
        nonlocal x_odo, th_true_prev, dth_true, dth_odo
        for _ in range(int(round(seconds / model.opt.timestep))):
            mujoco.mj_step(model, data)
            dL, dR = odo.increments(world.read_encoders(model, data))
            x_odo, _, dth, _ = motion_model(x_odo, dL, dR)
            dth_odo += dth
            th = world.true_pose(model, data)[2]
            dth_true += wrap(th - th_true_prev)
            th_true_prev = th

    set_wheel_speeds(model, data, w_left, w_right)
    advance(duration)
    set_wheel_speeds(model, data, 0.0, 0.0)
    advance(settle)

    return dict(p0=p0, p1=world.true_pose(model, data), x_odo=x_odo, dth_true=dth_true, dth_odo=dth_odo)


def straight_test(model, data, speeds=(2.0, 4.0, 6.0), distance=1.3, trials=3):
    """Straight line from (-1, 0) heading +x. Rows: speed, distance, along-track err, cross-track err, heading err, %.

    along-track error > 0 means the odometry thinks it travelled FURTHER than it really did.
    """
    rows = []
    for w in speeds:
        duration = distance / (w * cfg.WHEEL_RADIUS)
        for _ in range(trials):
            world.set_pose(model, data, -1.0, 0.0, 0.0)
            r = run_segment(model, data, w, w, duration)
            d = r["p1"][:2] - r["p0"][:2]
            dist = np.linalg.norm(d)
            u = d / dist                           # true direction of travel
            e = r["x_odo"][:2] - r["p1"][:2]       # odometry minus truth (position)
            along = e @ u
            cross = u[0] * e[1] - u[1] * e[0]
            rows.append((w, dist, along, cross, r["dth_odo"] - r["dth_true"], 100 * along / dist))
    return np.array(rows)


def spin_test(model, data, speeds=(1.0, 2.0, 3.0), duration=8.0, trials=3):
    """Spin on the spot, both directions. Rows: wheel speed (signed, + = counter-clockwise), true angle turned
    (unwrapped), heading error (odometry - truth), error as a % of the angle turned."""
    rows = []
    for w in speeds:
        for sign in (+1, -1):
            for _ in range(trials):
                world.set_pose(model, data, 0.0, 0.0, 0.0)
                r = run_segment(model, data, -sign * w, sign * w, duration)
                err = r["dth_odo"] - r["dth_true"]
                rows.append((sign * w, r["dth_true"], err, 100 * err / abs(r["dth_true"])))
    return np.array(rows)


def summarise(rows, key_col, value_cols, names):
    """Print the mean and std of the chosen columns for each distinct value in key_col."""
    for key in np.unique(rows[:, key_col]):
        sel = rows[rows[:, key_col] == key]
        parts = [f"{n}: {sel[:, c].mean():+.4g} +/- {sel[:, c].std():.2g}" for c, n in zip(value_cols, names)]
        print(f"  speed {key:+.1f} rad/s | " + " | ".join(parts))


if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True, linewidth=120)
    model = world.build_world()
    data = mujoco.MjData(model)

    print("Straight (speed, distance, along err, cross err, heading err, along err %):")
    straight = straight_test(model, data)
    print(straight)
    print("Summary by speed:")
    summarise(straight, 0, (5, 3, 4), ("along %", "cross m", "heading rad"))

    print("\nSpin (speed, angle turned, heading err, err %):")
    spin = spin_test(model, data)
    print(spin)
    print("Summary by speed and direction:")
    summarise(spin, 0, (3,), ("err %",))

    # Interpreting the output:
    #  - same % at every speed and direction  -> a pure scale factor (calibrate it, or cover it with K_SLIP)
    #  - % changes with speed                 -> something dynamic (slip, ramp-up)
    #  - different sign/size CW vs CCW        -> asymmetry between the wheels
    #  - cross-track error ~ 0                -> left/right wheels are matched