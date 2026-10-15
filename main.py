## Main code file for integrating all control/estimation/perception
"""Track A end-to-end loop: simulation -> sensors -> perception -> estimator -> controller -> wheels.

Rates: control and estimation every cfg.CONTROL_DT (20Hz), camera at cfg.CAMERA_RATE (10 Hz), lidar at
cfg.LIDAR_RATE (5 Hz). The estimator and controller only ever see sensor data and the estimate. Ground truth from
MuJoCo goes into the log for evaluation and plots only.

Estimator: the EKF (needed for criterion C) drives the controller. Dead reckoning runs alongside on the same odometry for
comparison only. Controller (criterion D): MissionController, LQR path tracking then visual docking, or the
WaypointFollower placeholder as a baseline.

Usage: python main.py [--duration 120] [--seed 0] [--estimator ekf|dr] [--controller lqr|simple] [--viewer]
                      [--no-plot]
"""
import argparse
import time
from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import mujoco
import mujoco.viewer
import numpy as np

from src import config as cfg
from src.control import model
from src.control.control import MissionController, WaypointFollower
from src.estimation.estimation import EKF, DeadReckoning, EncoderOdometry
from src.perception import perception as per
from src.simulation import world as sim


@dataclass
class Log:
    """Everything recorded during a run. 'true' is ground truth for EVALUATION only."""
    t: list = field(default_factory=list)
    true: list = field(default_factory=list)
    est: list = field(default_factory=list)
    P: list = field(default_factory=list) # estimator covariance
    dr: list = field(default_factory=list) # dead reckoning on the same odometry (comparison only)
    cmd: list = field(default_factory=list)
    odo: list = field(default_factory=list) # measured (v, omega) from the encoders
    waypoint: list = field(default_factory=list)
    mode: list = field(default_factory=list) # controller mode (track/turn/dock/done)
    track_err: list = field(default_factory=list) # controller's (e_y, e_theta), computed from the ESTIMATE
    blocked: list = field(default_factory=list)
    markers: list = field(default_factory=list) # (t, id, z) for every marker detection
    obs: list = field(default_factory=list) # marker observations used at each control step (for replay)
    obstacles: list = field(default_factory=list) # obstacle centroids in the world frame via the ESTIMATE

    collisions: set = field(default_factory=set) # what the robot touched other than the floor (ground truth)

    def arrays(self):
        return {k: np.array(v) for k, v in self.__dict__.items()
                if isinstance(v, list) and k not in ("markers", "obs", "obstacles")}


def run(duration=120.0, seed=0, viewer=False, start=cfg.START_POSE, waypoints=cfg.WAYPOINTS, estimator="ekf",
        controller="lqr", true_start=None, encoder_errors=None, camera_noise=cfg.CAMERA_NOISE,
        camera_fovy=cfg.CAMERA_FOVY, lidar_sigma=cfg.LIDAR_SIGMA, world_markers=None, extra_obstacles=(),
        push_at=None):
    """One mission. The keyword arguments after controller change the WORLD (never the robot's software) for the
    robustness trials:
        true_start       the robot is placed here instead of at the start pose its software assumes
        encoder_errors   (scale_error, slip_sigma) for the simulated odometry
        camera_noise     pixel noise [grey levels]-  camera_fovy the real camera, against the calibrated one
        lidar_sigma      lidar range noise [m]
        world_markers    the markers actually in the room (the robot's map keeps all of cfg.MARKERS)
        extra_obstacles  obstacles the route does not know about (cfg.OBSTACLES format)
        push_at          (t [s], sideways force [N]): a 0.2sec shove that the encoders do not see
    """
    rng = np.random.default_rng(seed)
    mj_model = sim.build_world(camera_fovy=camera_fovy, markers=world_markers, extra_obstacles=extra_obstacles)
    data = mujoco.MjData(mj_model)
    sim.set_pose(mj_model, data, *(start if true_start is None else true_start))
    for _ in range(int(0.5 / mj_model.opt.timestep)): # settle on the wheels
        mujoco.mj_step(mj_model, data)
    camera = sim.Camera(mj_model, noise=camera_noise)
    encoders = sim.Encoders(mj_model, data) if encoder_errors is None else sim.Encoders(mj_model, data, *encoder_errors)

    # The robot's software- starts from the known start pose (the prior) and not from the simulator
    odometry = EncoderOdometry(encoders.read(data, rng))
    estimator = EKF(start) if estimator == "ekf" else DeadReckoning(start)
    dead_reckoning = DeadReckoning(start)
    controller = MissionController(waypoints, start) if controller == "lqr" else WaypointFollower(waypoints)

    dt = cfg.CONTROL_DT
    n_sub = int(round(dt / mj_model.opt.timestep))
    camera_every = int(round(1 / cfg.CAMERA_RATE / dt))
    lidar_every = int(round(1 / cfg.LIDAR_RATE / dt))
    log, scan_points, t_done = Log(), None, None

    view = mujoco.viewer.launch_passive(mj_model, data) if viewer else None
    wall_start = time.time()
    t0 = data.time
    for k in range(int(round(duration / dt))):
        t = data.time - t0

        # Sense and estimate
        u_odo = odometry.update(encoders.read(data, rng), dt) # motion over the last period
        estimator.predict(u_odo, dt)
        dead_reckoning.predict(u_odo, dt)
        observations = []
        if k % camera_every == 0:
            observations = per.detect_markers(camera.render(data, rng))
            estimator.update(observations)
            log.markers += [(t, o.id, o.z) for o in observations]
        if k % lidar_every == 0:
            scan = per.process_scan(sim.read_lidar(mj_model, data, rng, lidar_sigma), rng)
            scan_points = scan.points
            log.obstacles += [per.to_world(o.centroid[None], estimator.x)[0] for o in scan.obstacles]

        # Control on the estimate only
        v, w = controller.command(estimator.x.copy(), scan_points, observations, t)
        data.ctrl[:] = model.unicycle_to_wheels(v, w)

        # Log (ground truth for EVALUATION only)
        log.t.append(t); log.true.append(sim.true_pose(mj_model, data)); log.est.append(estimator.x.copy())
        log.P.append(estimator.P.copy()); log.dr.append(dead_reckoning.x.copy())
        log.cmd.append((v, w)); log.odo.append(u_odo); log.waypoint.append(controller.index)
        log.mode.append(controller.mode); log.track_err.append(controller.errors)
        log.blocked.append(controller.blocked); log.obs.append(observations)

        # Advance the simulation one control period (disturbance and collision check: world side, evaluation only)
        if push_at is not None and t <= push_at[0] < t + dt:
            sim.push(mj_model, data, push_at[1])
        if push_at is not None and t <= push_at[0] + sim.PUSH_DURATION < t + dt:
            sim.push_end(mj_model, data)
        for _ in range(n_sub):
            mujoco.mj_step(mj_model, data)
        log.collisions |= sim.robot_collisions(mj_model, data)
        if view is not None:
            if not view.is_running():
                break
            view.sync()
            time.sleep(max(0.0, (data.time - t0) - (time.time() - wall_start))) # real time
        if controller.done: # stop 1sec after the last waypoint
            t_done = t if t_done is None else t_done
            if t - t_done >= 1.0:
                break

    camera.close()
    if view is not None:
        view.close()
    log.ekf_rejected = getattr(estimator, "rejected", 0)
    log.ekf_nis = getattr(estimator, "nis", [])
    log.detours = getattr(controller, "detours", 0)
    log.ekf_recoveries = len(getattr(estimator, "recoveries", []))
    return log, controller


def estimation_errors(est, true):
    """Position error [m] and signed heading error [rad], estimate - truth."""
    return np.linalg.norm(est[:, :2] - true[:, :2], axis=1), model.wrap(est[:, 2] - true[:, 2])


def docked_pose(dock_distance=cfg.DOCK_DISTANCE):
    """Target robot pose when docked: facing the dock marker, centred on it, camera dock_distance from it."""
    T = cfg.MARKERS[cfg.DOCK_ID]
    n = T.R[:2, 2] # marker normal, pointing out of the dock
    th = np.arctan2(-n[1], -n[0])
    cam = T.t[:2] + dock_distance * n
    return np.array([cam[0] - cfg.CAMERA_POS[0] * np.cos(th), cam[1] - cfg.CAMERA_POS[0] * np.sin(th), th])


def docking_errors(x_true):
    """True docking errors (evaluation only): distance along the marker normal (+ too far out), lateral offset
    [m], heading error [rad]."""
    target = docked_pose()
    e = model.tracking_error(x_true, target)
    return -e[0], e[1], e[2]


def path_errors(xy, start=cfg.START_POSE, waypoints=cfg.WAYPOINTS):
    """Distance of each point from the waypoint path (polyline), evaluation only."""
    pts = [np.array(start[:2])] + [np.array(w) for w in waypoints]
    d = np.full(len(xy), np.inf)
    for a, b in zip(pts[:-1], pts[1:]):
        ab = b - a
        s = np.clip(((xy - a) @ ab) / (ab @ ab), 0, 1)
        d = np.minimum(d, np.linalg.norm(xy - (a + s[:, None] * ab), axis=1))
    return d


def summarise(log, controller, waypoints=cfg.WAYPOINTS):
    a = log.arrays()
    status = "DOCKED" if controller.done and isinstance(controller, MissionController) else (
        "finished" if controller.done else "NOT finished")
    print(f"run time {a['t'][-1]:.1f} s, {status} (according to the robot)")
    tracking = a["mode"] == "track"
    if tracking.any():
        d = path_errors(a["true"][tracking, :2])
        print(f"true distance from the path while tracking: RMS {1000 * np.sqrt(np.mean(d ** 2)):.0f} mm, "
              f"max {1000 * d.max():.0f} mm")
    if isinstance(controller, MissionController):
        dist, lat, hd = docking_errors(a["true"][-1])
        print(f"true docking error: distance {1000 * dist:+.0f} mm, lateral {1000 * lat:+.0f} mm, "
              f"heading {np.degrees(hd):+.1f} deg")
    else:
        goal_err = np.hypot(*(np.array(waypoints[-1]) - a["true"][-1, :2]))
        print(f"true distance from the final waypoint: {1000 * goal_err:.0f} mm")
    for name, key in [("estimator", "est"), ("dead reckoning", "dr")]:
        e_pos, e_th = estimation_errors(a[key], a["true"])
        print(f"{name:>15} error: final {1000 * e_pos[-1]:4.0f} mm / {np.degrees(abs(e_th[-1])):4.1f} deg, "
              f"RMS {1000 * np.sqrt(np.mean(e_pos ** 2)):4.0f} mm / {np.degrees(np.sqrt(np.mean(e_th ** 2))):4.1f} deg, "
              f"max {1000 * e_pos.max():4.0f} mm")
    if log.ekf_nis:
        print(f"EKF: {len(log.ekf_nis)} marker updates, {log.ekf_rejected} rejected by the gate, "
              f"mean NIS {np.mean(log.ekf_nis):.2f} (2 for a consistent filter)")
    ids, counts = np.unique([m[1] for m in log.markers], return_counts=True) if log.markers else ([], [])
    print(f"marker detections: {len(log.markers)} " + str(dict(zip([int(i) for i in ids], [int(c) for c in counts]))))
    print(f"lidar safety stops: {int(a['blocked'].sum())} control periods")


def plot(log, waypoints=cfg.WAYPOINTS):
    a = log.arrays()
    fig = plt.figure(figsize=(15, 7))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.1, 1])
    ax = [fig.add_subplot(gs[:, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 1])]

    m = ax[0]
    _in = cfg.ROOM_HALF - cfg.WALL_THICK
    m.add_patch(plt.Rectangle((-_in, -_in), 2 * _in, 2 * _in, fill=False, ec="0.3", lw=2))
    for _, kind, c, size, _ in cfg.OBSTACLES:
        if kind == "box":
            m.add_patch(plt.Rectangle(np.array(c) - size, *(2 * np.array(size)), fc="0.8", ec="0.4"))
        else:
            m.add_patch(plt.Circle(c, size, fc="0.8", ec="0.4"))
    m.add_patch(plt.Rectangle(np.array(cfg.DOCK_POS) - cfg.DOCK_SIZE[:2], *(2 * np.array(cfg.DOCK_SIZE[:2])),
                              fc="0.3", ec="k"))
    for mid, T in cfg.MARKERS.items():
        m.plot(*T.t[:2], "s", color="tab:red", ms=7)
        m.annotate(str(mid), T.t[:2], xytext=(4, 4), textcoords="offset points", color="tab:red")
    if log.obstacles:
        ob = np.array(log.obstacles)
        m.plot(ob[:, 0], ob[:, 1], ".", ms=2, color="tab:orange", alpha=0.4, label="lidar obstacles (via estimate)")
    wp = np.array([cfg.START_POSE[:2]] + list(waypoints))
    m.plot(wp[:, 0], wp[:, 1], "k:", lw=1)
    m.plot(wp[1:, 0], wp[1:, 1], "kx", ms=9, mew=2, label="waypoints")
    m.plot(a["true"][:, 0], a["true"][:, 1], "k", lw=2.5, label="true path")
    m.plot(a["est"][:, 0], a["est"][:, 1], "--", color="tab:blue", lw=1.8, label="estimated path")
    m.plot(a["dr"][:, 0], a["dr"][:, 1], ":", color="tab:purple", lw=1.5, label="dead reckoning")
    m.set_aspect("equal"); m.set_xlabel("x [m]"); m.set_ylabel("y [m]"); m.legend(loc="lower right", fontsize=8)
    m.set_title("map, waypoints and paths")

    e_pos, e_th = estimation_errors(a["est"], a["true"])
    d_pos, d_th = estimation_errors(a["dr"], a["true"])
    sig_pos = np.sqrt(a["P"][:, 0, 0] + a["P"][:, 1, 1])
    sig_th = np.sqrt(a["P"][:, 2, 2])

    e = ax[1]
    e.plot(a["t"], 1000 * e_pos, color="tab:blue", label="estimator position error")
    e.plot(a["t"], 1000 * d_pos, ":", color="tab:purple", label="dead reckoning position error")
    e.fill_between(a["t"], 0, 2000 * sig_pos, color="tab:blue", alpha=0.15, label="estimator 2 sigma")
    if log.markers:
        e.plot([mk[0] for mk in log.markers], np.zeros(len(log.markers)), "|", color="tab:green", ms=12,
               label="marker detections")
    e.set_xlabel("time [s]"); e.set_ylabel("position error [mm]"); e.legend(loc="upper left", fontsize=8)
    e.set_title("estimation error (estimate - truth)")

    h = ax[2]
    h.plot(a["t"], np.degrees(e_th), color="tab:red", label="estimator heading error")
    h.plot(a["t"], np.degrees(d_th), ":", color="tab:purple", label="dead reckoning heading error")
    h.fill_between(a["t"], -2 * np.degrees(sig_th), 2 * np.degrees(sig_th), color="tab:red", alpha=0.15,
                   label="estimator +/- 2 sigma")
    h.set_xlabel("time [s]"); h.set_ylabel("heading error [deg]"); h.legend(loc="upper left", fontsize=8)
    plt.tight_layout(); plt.show()


def main():
    parser = argparse.ArgumentParser(description="Track A end-to-end loop")
    parser.add_argument("--duration", type=float, default=120.0, help="maximum run time [s]")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--estimator", choices=["ekf", "dr"], default="ekf",
                        help="which estimator drives the controller (dr = dead reckoning)")
    parser.add_argument("--controller", choices=["lqr", "simple"], default="lqr",
                        help="lqr = LQR tracking + docking; simple = the Week 8 C.4 placeholder")
    parser.add_argument("--viewer", action="store_true", help="show the MuJoCo viewer, in real time")
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    log, controller = run(args.duration, args.seed, args.viewer, estimator=args.estimator,
                          controller=args.controller)
    summarise(log, controller)
    if not args.no_plot:
        plot(log)


if __name__ == "__main__":
    main()
