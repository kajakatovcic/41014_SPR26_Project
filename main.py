## Main code file for integrating all control/estimation/perception
"""Track A end-to-end loop: simulation -> sensors -> perception -> estimator -> controller -> wheels.

Rates: control and estimation every cfg.CONTROL_DT (20Hz), camera at cfg.CAMERA_RATE (10 Hz), lidar at
cfg.LIDAR_RATE (5 Hz). The estimator and controller only ever see sensor data and the estimate. Ground truth from
MuJoCo goes into the log for evaluation and plots only.

Missions:
    explore (default)  no map and no route. The robot finds and localises from the reference marker (its frame {A}
                       becomes the reference frame), knows only the dock marker's offset from it, plans to the dock on
                       an occupancy grid built from its lidar and replans as it goes (MarkerSLAM + Navigator).
    route              the earlier mission, kept as a baseline: known start pose, known marker map, fixed waypoint
                       route (EKF + MissionController, or the WaypointFollower placeholder).
Estimator: the EKF drives the controller. Dead reckoning runs alongside on the same odometry for comparison only
(in the explore mission from the moment the robot localises).

Usage: python main.py [--mission explore|route] [--duration 120] [--seed 0] [--estimator ekf|dr]
                      [--controller lqr|straight|simple] [--viewer] [--record [PATH]] [--no-plot]
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
from src.control.mission import clearance
from src.control.navigation import Navigator
from src.estimation.estimation import EKF, DeadReckoning, EncoderOdometry, MarkerSLAM
from src.perception import perception as per
from src.simulation import video
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
    walls: list = field(default_factory=list) # lidar wall lines the estimator had at each camera update (replay)

    collisions: set = field(default_factory=set) # what the robot touched other than the floor (ground truth)

    def arrays(self):
        return {k: np.array(v) for k, v in self.__dict__.items()
                if isinstance(v, list) and k not in ("markers", "obs", "obstacles", "walls")}


# The explore mission estimates in the reference marker's frame {A}. For EVALUATION ONLY (logs, plots, errors against
# ground truth), estimates are converted to the world frame through the reference marker's true world pose.

def marker_frame(marker_id):
    """World pose (x, y, phi) of a marker's floor frame: origin at its centre, x along its normal."""
    T = cfg.MARKERS[marker_id]
    return np.array([*T.t[:2], np.arctan2(T.R[1, 2], T.R[0, 2])])


def pose_to_world(x, F):
    """Pose (x, y, theta) in frame F = (x, y, phi) -> world."""
    c, s = np.cos(F[2]), np.sin(F[2])
    return np.array([F[0] + c * x[0] - s * x[1], F[1] + s * x[0] + c * x[1], model.wrap(x[2] + F[2])])


def points_to_world(P, F):
    c, s = np.cos(F[2]), np.sin(F[2])
    return np.asarray(P).reshape(-1, 2) @ np.array([[c, s], [-s, c]]) + F[:2]


def cov_to_world(P, F):
    c, s = np.cos(F[2]), np.sin(F[2])
    Rot = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    return Rot @ P @ Rot.T


def run(duration=120.0, seed=0, viewer=False, mission="explore", start=cfg.START_POSE, waypoints=cfg.WAYPOINTS,
        estimator="ekf", controller="lqr", true_start=None, encoder_errors=None, camera_noise=cfg.CAMERA_NOISE,
        camera_fovy=cfg.CAMERA_FOVY, lidar_sigma=cfg.LIDAR_SIGMA, world_markers=None, extra_obstacles=(),
        push_at=None, obstacle_appears=None, record=None):
    """One mission ("explore" or "route", see the module docstring). With viewer=True the robot's camera feed is
    shown live in its own window next to the viewer; record = a path saves an MP4 of an overview of the room beside
    the camera feed (src/simulation/video.py). start and waypoints are the route mission's (its known start pose).

    The keyword arguments from true_start on change the WORLD (never the robot's software) for the
    robustness trials:
        true_start       where the robot is placed (default: cfg.EXPLORE_START, or the route's start pose)
        encoder_errors   (scale_error, slip_sigma) for the simulated odometry
        camera_noise     pixel noise [grey levels]-  camera_fovy the real camera, against the calibrated one
        lidar_sigma      lidar range noise [m]
        world_markers    the markers actually in the room (the robot's map keeps all of cfg.MARKERS)
        extra_obstacles  obstacles the route does not know about (cfg.OBSTACLES format)
        push_at          (t [s], sideways force [N]): a 0.2sec shove that the encoders do not see; or
                         ("dock", delay [s], force [N]): the same, delay seconds after docking starts
        obstacle_appears (t [s], distance [m]): a 0.1 m radius cylinder appears that distance straight ahead of the
                         robot at time t (someone puts something in its way)
    """
    explore = mission == "explore"
    if true_start is None:
        true_start = cfg.EXPLORE_START if explore else start
    rng = np.random.default_rng(seed)
    mj_model = sim.build_world(camera_fovy=camera_fovy, markers=world_markers, extra_obstacles=extra_obstacles,
                               popup=obstacle_appears is not None)
    data = mujoco.MjData(mj_model)
    sim.set_pose(mj_model, data, *true_start)
    for _ in range(int(0.5 / mj_model.opt.timestep)): # settle on the wheels
        mujoco.mj_step(mj_model, data)
    camera = sim.Camera(mj_model, noise=camera_noise)
    encoders = sim.Encoders(mj_model, data) if encoder_errors is None else sim.Encoders(mj_model, data, *encoder_errors)

    odometry = EncoderOdometry(encoders.read(data, rng))
    if explore:
        # The robot's software knows no map and no start pose: it localises from the reference marker
        if controller == "simple":
            raise ValueError("the simple placeholder needs the route mission (--mission route)")
        estimator = MarkerSLAM(correct=(estimator == "ekf"))
        dead_reckoning = None # starts from the pose the robot localises at
        controller = Navigator(smooth=(controller != "straight"))
        known_markers = None # perception has no marker map either
        frame_A = marker_frame(cfg.REF_MARKER_ID) # EVALUATION only: {A} -> world
    else:
        # The robot's software starts from the known start pose (the prior) and not from the simulator
        estimator = EKF(start) if estimator == "ekf" else DeadReckoning(start)
        dead_reckoning = DeadReckoning(start)
        controller = {"lqr": lambda: MissionController(waypoints, start),
                      "straight": lambda: MissionController(waypoints, start, smooth=False),
                      "simple": lambda: WaypointFollower(waypoints)}[controller]()
        known_markers = cfg.MARKERS
        frame_A = np.zeros(3) # the route mission estimates in the world frame
    nan3, nan33 = np.full(3, np.nan), np.full((3, 3), np.nan)

    dt = cfg.CONTROL_DT
    n_sub = int(round(dt / mj_model.opt.timestep))
    camera_every = int(round(1 / cfg.CAMERA_RATE / dt))
    lidar_every = int(round(1 / cfg.LIDAR_RATE / dt))
    log, scan_points, t_done, t_dock = Log(), None, None, None

    view = mujoco.viewer.launch_passive(mj_model, data) if viewer else None
    cam_window = None
    if view is not None:
        video.setup_viewer(view)
        try:
            cam_window = video.CameraWindow()  # the robot's camera, live, in its own window
        except Exception as err:  # macOS: mjpython runs off the main thread, so OpenCV can't open windows
            print(f"camera window unavailable ({type(err).__name__}); continuing without it")
    recorder = video.Recorder(record, mj_model) if record else None
    wall_start = time.time()
    t0 = data.time
    for k in range(int(round(duration / dt))):
        t = data.time - t0

        # Sense and estimate
        u_odo = odometry.update(encoders.read(data, rng), dt) # motion over the last period
        estimator.predict(u_odo, dt)
        if dead_reckoning is not None:
            dead_reckoning.predict(u_odo, dt)
        observations, frame, scan = [], None, None
        # The explore mission's estimator uses the lidar wall lines with the camera frame of the SAME step (the
        # marker's yaw), so the lidar is read first there; the route mission keeps its original order
        if explore and k % lidar_every == 0:
            scan = per.process_scan(sim.read_lidar(mj_model, data, rng, lidar_sigma), rng)
            estimator.walls, estimator.walls_fresh = scan.walls, True
        log.walls.append((getattr(estimator, "walls", None), getattr(estimator, "walls_fresh", False)))
        if k % camera_every == 0:
            frame = camera.render(data, rng)
            observations = per.detect_markers(frame, known=known_markers)
            estimator.update(observations)
            log.markers += [(t, o.id, o.z) for o in observations]
            if dead_reckoning is None and estimator.x is not None: # just localised
                dead_reckoning = DeadReckoning(estimator.x.copy())
        if explore:
            estimator.walls_fresh = False # only for the camera frame taken with that scan
        if not explore and k % lidar_every == 0:
            scan = per.process_scan(sim.read_lidar(mj_model, data, rng, lidar_sigma), rng)
        x_hat = None if estimator.x is None else estimator.x.copy()
        if scan is not None:
            scan_points = scan.points
            if x_hat is not None:
                log.obstacles += [points_to_world(per.to_world(o.centroid[None], x_hat), frame_A)[0]
                                  for o in scan.obstacles]

        # Control on the estimate only
        v, w = controller.command(x_hat, scan_points, observations, t)
        data.ctrl[:] = model.unicycle_to_wheels(v, w)

        # Show / record the camera frame the robot just used (demonstration only)
        if frame is not None and (cam_window is not None or recorder is not None):
            shown = video.annotate(frame, observations, t, controller.mode)
            if cam_window is not None:
                cam_window.show(shown)
            if recorder is not None:
                recorder.write(data, shown, t, controller.mode)

        # Log (ground truth for EVALUATION only)
        log.t.append(t); log.true.append(sim.true_pose(mj_model, data))
        log.est.append(nan3 if x_hat is None else pose_to_world(x_hat, frame_A))
        log.P.append(nan33 if x_hat is None else cov_to_world(estimator.P, frame_A))
        log.dr.append(nan3 if dead_reckoning is None else pose_to_world(dead_reckoning.x, frame_A))
        log.cmd.append((v, w)); log.odo.append(u_odo); log.waypoint.append(controller.index)
        log.mode.append(controller.mode); log.track_err.append(controller.errors)
        log.blocked.append(controller.blocked); log.obs.append(observations)

        # Advance the simulation one control period (disturbance and collision check: world side, evaluation only)
        if push_at is not None:
            if push_at[0] == "dock": # ("dock", delay, force): relative to docking start
                if controller.mode == "dock" and t_dock is None:
                    t_dock = t
                t_push = None if t_dock is None else t_dock + push_at[1]
            else:
                t_push = push_at[0]
            if t_push is not None and t <= t_push < t + dt:
                sim.push(mj_model, data, push_at[-1])
            if t_push is not None and t <= t_push + sim.PUSH_DURATION < t + dt:
                sim.push_end(mj_model, data)
        if obstacle_appears is not None and t <= obstacle_appears[0] < t + dt:
            x_true = sim.true_pose(mj_model, data)
            log.popup = x_true[:2] + obstacle_appears[1] * np.array([np.cos(x_true[2]), np.sin(x_true[2])])
            sim.place_popup(mj_model, data, log.popup)
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
    if recorder is not None:
        recorder.close()
        print(f"video: {recorder.frames} frames ({recorder.frames / cfg.CAMERA_RATE:.1f} s) -> {recorder.path}")
    if cam_window is not None:
        cam_window.close()
    if view is not None:
        view.close()
    log.ekf_rejected = getattr(estimator, "rejected", 0)
    log.ekf_nis = getattr(estimator, "nis", [])
    log.detours = getattr(controller, "detours", 0)
    log.ekf_recoveries = len(getattr(estimator, "recoveries", []))
    log.ref_path = (points_to_world(controller.reference_path(), frame_A) if hasattr(controller, "reference_path")
                    else np.array([start[:2]] + list(waypoints)))
    log.mission, log.frame_A = mission, frame_A
    if explore:
        log.grid = controller.grid # the lidar map the robot built, in {A}
        log.replans, log.plan_failures = controller.replans, controller.plan_failures
        log.replan_reasons = dict(controller.replan_reasons)
        log.map_resets = controller.map_resets
        log.dock_searches = controller.dock_searches
        log.dock_retries = controller.dock_retries
        log.yaw_updates, log.yaw_rejected = len(estimator.yaw_nis), estimator.yaw_rejected
        log.init_frame = estimator.init_frame
        log.init_yaw_source = estimator.init_yaw_source
        log.mapped = {i: points_to_world(estimator.s[3 + 2 * k:5 + 2 * k], frame_A)[0]
                      for k, i in enumerate(estimator.mapped)} if estimator.ready else {}
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


def path_errors(xy, path=None):
    """Distance of each point from a planned path given as a polyline (N x 2): the controller's own reference
    (log.ref_path) or by default the waypoint route in src/config.py. FOr evaluation only."""
    pts = [np.array(cfg.START_POSE[:2])] + [np.array(w) for w in cfg.WAYPOINTS] if path is None else list(path)
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
        d = path_errors(a["true"][tracking, :2], log.ref_path)
        print(f"true distance from the planned path while tracking: RMS {1000 * np.sqrt(np.mean(d ** 2)):.0f} mm, "
              f"max {1000 * d.max():.0f} mm")
    if isinstance(controller, MissionController):
        dist, lat, hd = docking_errors(a["true"][-1])
        print(f"true docking error: distance {1000 * dist:+.0f} mm, lateral {1000 * lat:+.0f} mm, "
              f"heading {np.degrees(hd):+.1f} deg")
    else:
        goal_err = np.hypot(*(np.array(waypoints[-1]) - a["true"][-1, :2]))
        print(f"true distance from the final waypoint: {1000 * goal_err:.0f} mm")
    if log.mission == "explore":
        ready = np.flatnonzero(~np.isnan(a["est"][:, 0]))
        if not len(ready):
            print("never localised: the reference marker was not found")
            return
        e0, th0 = estimation_errors(a["est"][ready[:1]], a["true"][ready[:1]])
        print(f"localised from marker {cfg.REF_MARKER_ID} at t = {a['t'][ready[0]]:.1f} s (marker yaw from the "
              f"{log.init_yaw_source}): initial error {1000 * e0[0]:.0f} mm / {np.degrees(abs(th0[0])):.1f} deg")
        print(f"paths planned: {log.replans} {log.replan_reasons}, failed plans: {log.plan_failures}, "
              f"map restarts: {log.map_resets}, searches for the dock marker: {log.dock_searches}, docking "
              f"re-approaches: {log.dock_retries}")
        print(f"wall-line heading updates: {log.yaw_updates} accepted, {log.yaw_rejected} rejected by the gate")
        if log.mapped:
            print("markers mapped on the way (error of the mapped position): " + ", ".join(
                f"{i}: {1000 * np.linalg.norm(p - cfg.MARKERS[i].t[:2]):.0f} mm" for i, p in log.mapped.items()))
        obs_c, wall_c = clearance(a["true"][:, :2])
        print(f"closest true approach beyond the robot's radius: obstacles {100 * obs_c:.1f} cm, walls "
              f"{100 * wall_c:.1f} cm; collisions: {sorted(log.collisions) or 'none'}")
    for name, key in [("estimator", "est"), ("dead reckoning", "dr")]:
        e_pos, e_th = estimation_errors(a[key], a["true"])
        ok = ~np.isnan(e_pos)
        e_pos, e_th = e_pos[ok], e_th[ok]
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
    if getattr(log, "popup", None) is not None:
        m.add_patch(plt.Circle(log.popup, sim.POPUP_RADIUS, fc="tab:purple", ec="k", alpha=0.6,
                               label="obstacle that appeared"))
    if log.mission == "explore":
        g = log.grid
        occ = points_to_world(g.centres(np.argwhere(g.occupied)), log.frame_A)
        m.plot(occ[:, 0], occ[:, 1], "s", ms=2, color="tab:orange", alpha=0.6, label="lidar occupancy grid (robot's)")
        m.plot(log.ref_path[:, 0], log.ref_path[:, 1], "-", color="tab:green", lw=1, label="planned paths")
        m.plot(*a["true"][0, :2], "ko", ms=6, label="start")
        title = "explore mission: the robot's lidar map, planned and driven paths"
    else:
        if log.obstacles:
            ob = np.array(log.obstacles)
            m.plot(ob[:, 0], ob[:, 1], ".", ms=2, color="tab:orange", alpha=0.4, label="lidar obstacles (via estimate)")
        wp = np.array([cfg.START_POSE[:2]] + list(waypoints))
        m.plot(wp[:, 0], wp[:, 1], "k:", lw=1)
        m.plot(wp[1:, 0], wp[1:, 1], "kx", ms=9, mew=2, label="waypoints")
        title = "map, waypoints and paths"
    m.plot(a["true"][:, 0], a["true"][:, 1], "k", lw=2.5, label="true path")
    m.plot(a["est"][:, 0], a["est"][:, 1], "--", color="tab:blue", lw=1.8, label="estimated path")
    m.plot(a["dr"][:, 0], a["dr"][:, 1], ":", color="tab:purple", lw=1.5, label="dead reckoning")
    m.set_aspect("equal"); m.set_xlabel("x [m]"); m.set_ylabel("y [m]"); m.legend(loc="lower right", fontsize=8)
    m.set_title(title)

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
    parser.add_argument("--mission", choices=["explore", "route"], default="explore",
                        help="explore = localise from the reference marker, plan on the lidar map (default); "
                             "route = the earlier fixed route with a known map and start pose")
    parser.add_argument("--duration", type=float, default=120.0, help="maximum run time [s]")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--estimator", choices=["ekf", "dr"], default="ekf",
                        help="which estimator drives the controller (dr = dead reckoning)")
    parser.add_argument("--controller", choices=["lqr", "straight", "simple"], default="lqr",
                        help="lqr = LQR path following on the spline route + docking; straight = the same on "
                             "straight segments (turning on the spot at corners); simple = the Week 8 C.4 placeholder")
    parser.add_argument("--viewer", action="store_true",
                        help="show the MuJoCo viewer in real time, and the robot's camera feed in its own window")
    parser.add_argument("--record", nargs="?", const="videos/mission.mp4", default=None, metavar="PATH",
                        help="save an MP4 of an overview of the room beside the camera feed (default "
                             "videos/mission.mp4)")
    parser.add_argument("--obstacle-appears", type=float, default=None, metavar="T",
                        help="put an obstacle 0.8 m in front of the robot at time T [s], to see it replan")
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    log, controller = run(args.duration, args.seed, args.viewer, mission=args.mission, estimator=args.estimator,
                          controller=args.controller, record=args.record,
                          obstacle_appears=None if args.obstacle_appears is None else (args.obstacle_appears, 0.8))
    summarise(log, controller)
    if not args.no_plot:
        plot(log)


if __name__ == "__main__":
    main()
