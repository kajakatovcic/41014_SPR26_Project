"""Section 2: odometry characterisation.

Drive a straight line and a spin on the spot, integrate the encoder odometry, and compare it with the true pose.
The true pose is used here for EVALUATION only (it is how we measure the odometry error), never inside the filter.

Outputs (for the report): odometry error versus distance travelled and angle turned, and a justified value for
K_SLIP in estimation.py.

Run from the project root:  python -m tests.odometry_characterisation
"""
import mujoco
import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config as cfg
from src.simulation import world
from src.estimation.estimation import EncoderOdometry, motion_model
from src.perception.perception import wrap


def set_wheel_speeds(model, data, w_left, w_right):
    data.ctrl[0] = np.clip(w_left, -cfg.WHEEL_SPEED_MAX, cfg.WHEEL_SPEED_MAX)
    data.ctrl[1] = np.clip(w_right, -cfg.WHEEL_SPEED_MAX, cfg.WHEEL_SPEED_MAX)



def run_segment(model, data, w_left, w_right, duration):
    """Drive with constant wheel speeds. Returns (odometry_pose_delta, true_pose_delta) over the segment."""
    odo = EncoderOdometry()
    x_odo = np.array(world.true_pose(model, data))  # start the dead reckoning from the true start pose
    x_true0 = world.true_pose(model, data)
    odo.increments(world.read_encoders(model, data))  # initialise the encoder reference

    set_wheel_speeds(model, data, w_left, w_right)
    n = int(duration / model.opt.timestep)
    for _ in range(n):
        mujoco.mj_step(model, data)
        dL, dR = odo.increments(world.read_encoders(model, data))
        x_odo, *_ = motion_model(x_odo, dL, dR)
    set_wheel_speeds(model, data, 0.0, 0.0)

    x_true1 = world.true_pose(model, data)
    err = x_odo - x_true1
    err[2] = wrap(err[2])
    return err, x_true0, x_true1


def straight_test(model, data, speed=5.0, duration=10.0, trials=5):
    """Straight line: error in position and heading versus distance travelled."""
    results = []
    for _ in range(trials):
        world.set_pose(model, data, -1.0, 0.0, 0.0)
        err, p0, p1 = run_segment(model, data, speed, speed, duration)
        dist = np.hypot(*(p1[:2] - p0[:2]))
        results.append((dist, np.hypot(err[0], err[1]), err[2]))
    return np.array(results)  # columns: distance, position error, heading error


def spin_test(model, data, speed=3.0, duration=10.0, trials=5):
    """Spin on the spot: heading error versus angle turned (this is likely the weakest point of the odometry)."""
    results = []
    for _ in range(trials):
        world.set_pose(model, data, 0.0, 0.0, 0.0)
        err, p0, p1 = run_segment(model, data, -speed, speed, duration)
        results.append((abs(wrap(p1[2] - p0[2])), abs(err[2])))
    return np.array(results)  # columns: angle turned (wrapped, so keep the spin below pi), heading error


if __name__ == "__main__":
    model = world.build_world()
    data = mujoco.MjData(model)

    print("Straight line (distance, position error, heading error):")
    print(straight_test(model, data))
    print("Spin (angle, heading error):")
    print(spin_test(model, data))

    # TODO:
    #  - repeat at several speeds and distances, and turn on the spot in both directions
    #  - fit error ~ k * distance (and k_theta * angle), and put the result into estimation.K_SLIP
    #  - decide with part B whether extra noise needs injecting if the sim has almost no slip