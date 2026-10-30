"""The explore mission on the real TurtleBot3 Waffle Pi through rosbridge (or on the simulated stand-in using --sim).

Unchanged software from main.py's explore mission: 
    perception   lidar walls/obstacles (Week 4) and ArUco range-and-bearing (Weeks 3, 7, 9)
    estimation   MarkerSLAM: localise from the reference marker, the dock marker at its measured offset, other
                 markers mapped as seen, wall-line heading updates (src/estimation/estimation.py)
    control      Navigator: search, localise, look, plan on the LiDAR map (A* -> spline), LQR path following with
                 the curvature feed-forward, Week 9 docking, find / back off (src/control/navigation.py)
What differs is only the robot layer (robot_io.py), i.e. Real sensors on their own clocks, wheel odometry from
/joint_states time stamps, commands sent as (v, omega), and a 20Hz loop on the wall clock.

Usage:
    python hardware_integration/run_hardware.py 192.168.1.2XX --dry-run   # sensors, localisation, plans. NO motion
    python hardware_integration/run_hardware.py 192.168.1.2XX             # the full mission
    python hardware_integration/run_hardware.py --sim                     # pipeline on the MuJoCo stand-in
Options: --odom-source joint_states|odom, --straight, --max-time, --show-camera, --video PATH, --log PATH, --no-plot.
Stop at any time with Ctrl+C. The robot is sent a stop command on the way out.
"""


import argparse
import pickle
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config as cfg  
from src.control.navigation import Navigator  
from src.estimation.estimation import MarkerSLAM 
from src.perception import perception as per  
from src.simulation import video  
from hardware_integration import hw_config as hw 
from hardware_integration.robot_io import RobotIO  


def run(io, max_time=180.0, dry_run=False, smooth=True, show_camera=False, video_path=None, seed=0):
    """One explore mission; returns the log (dict) and the navigator."""
    hw.apply()
    rng = np.random.default_rng(seed) # RANSAC sampling in perception
    camera = hw.camera_model()
    estimator = MarkerSLAM(ref_id=hw.REF_MARKER_ID, dock_id=hw.DOCK_ID, dock_offset=hw.DOCK_OFFSET)
    nav = Navigator(ref_id=hw.REF_MARKER_ID, dock_offset=hw.DOCK_OFFSET, predock=hw.PREDOCK_DISTANCE,
                    smooth=smooth, v_nominal=hw.V_NOMINAL, dock_distance=hw.DOCK_DISTANCE, dock_tol=hw.DOCK_TOL)
    window = video.CameraWindow("Robot camera (hardware pipeline)") if show_camera else None
    writer = None
    if video_path:
        Path(video_path).parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 10, hw.CAMERA_SIZE)

    log = {k: [] for k in ("t", "x", "P", "u", "dt", "cmd", "mode", "obs", "rho", "walls", "loop_time", "stale")}
    scan_points, last_w, t_done, k = None, 0.0, None, 0
    t0 = io.now()
    try:
        while True:
            t_loop = time.perf_counter()
            t = io.now() - t0
            if t > max_time:
                print(f"time limit ({max_time:.0f} s) reached")
                break

            # odometry: the wheel motion since the last reading over the interval between the readings
            od = io.read_odometry()
            if od is not None:
                (v_m, w_m), dt = od
                estimator.predict((v_m, w_m), dt)
                last_w = w_m
            # lidar
            rho = io.read_scan()
            walls_fresh = False
            if rho is not None:
                scan = per.process_scan(rho, rng, T_RS=hw.T_RS, ransac_iters=hw.RANSAC_ITERS)
                scan_points = scan.points
                estimator.walls = scan.walls
                walls_fresh = abs(last_w) < hw.TURNING_FOR_WALL_YAW # scan and frame are not simultaneous here
            # camera
            observations, rgb = [], io.read_image()
            if rgb is not None:
                observations = per.detect_markers(rgb, camera=camera, side=hw.MARKER_SIDE, T_RC=hw.T_RC,
                                                  bias_px=hw.BIAS_PX, known=None)
                estimator.walls_fresh = walls_fresh
                estimator.update(observations)
                estimator.walls_fresh = False

            # control on the estimate only
            x_hat = None if estimator.x is None else estimator.x.copy()
            v, w = nav.command(x_hat, scan_points, observations, t)
            stale = io.stale()
            if stale and t > 2.0: # give the topics a moment to start
                v, w = 0.0, 0.0
            if not dry_run:
                v, w = io.command(v, w)

            if rgb is not None and (window is not None or writer is not None):
                shown = video.annotate(rgb, observations, t, nav.mode)
                if window is not None:
                    window.show(shown)
                if writer is not None:
                    writer.write(cv2.cvtColor(shown, cv2.COLOR_RGB2BGR))

            log["t"].append(t); log["x"].append(x_hat); log["P"].append(None if x_hat is None else estimator.P.copy())
            log["u"].append(None if od is None else od[0]); log["dt"].append(None if od is None else od[1])
            log["cmd"].append((v, w)); log["mode"].append(nav.mode); log["obs"].append(observations)
            log["rho"].append(rho); log["walls"].append((estimator.walls, walls_fresh) if rho is not None else None)
            log["stale"].append(stale)
            log["loop_time"].append(time.perf_counter() - t_loop)
            if hasattr(io.robot, "true_pose"): # the simulated stand-in only. ground truth for scoring
                log.setdefault("true", []).append(io.robot.true_pose())

            if nav.done:
                t_done = t if t_done is None else t_done
                if t - t_done > 1.0:
                    break
            k += 1
            io.wait_until(t0 + k * hw.CONTROL_DT)
    except KeyboardInterrupt:
        print("\nstopped by the user")
    finally:
        if not dry_run:
            io.stop()
        if window is not None:
            window.close()
        if writer is not None:
            writer.release()
    log["estimator"] = {"init_frame": estimator.init_frame, "init_yaw_source": estimator.init_yaw_source,
                        "mapped": {i: estimator.s[3 + 2 * j:5 + 2 * j].copy() for j, i in enumerate(estimator.mapped)}
                        if estimator.ready else {}, "recoveries": len(estimator.recoveries),
                        "rejected": estimator.rejected, "yaw_updates": len(estimator.yaw_nis),
                        "yaw_rejected": estimator.yaw_rejected}
    log["navigator"] = {"replans": nav.replans, "reasons": dict(nav.replan_reasons), "map_resets": nav.map_resets,
                        "dock_searches": nav.dock_searches, "dock_retries": nav.dock_retries,
                        "paths": nav.reference_path()}
    log["io"] = dict(io.stats, calibrated=io.calibrated)
    return log, nav


def summarise(log, nav):
    t = np.array(log["t"])
    ready = [i for i, x in enumerate(log["x"]) if x is not None]
    lt = 1000 * np.array(log["loop_time"])
    print(f"\nrun time {t[-1]:.1f} s, final mode '{nav.mode}', {'DOCKED' if nav.done else 'NOT docked'} "
          "(according to the robot)")
    if ready:
        e = log["estimator"]
        print(f"localised at t = {t[ready[0]]:.1f} s (marker yaw from the {e['init_yaw_source']}); "
              f"markers mapped: {sorted(e['mapped'])}; lock-out recoveries {e['recoveries']}, gate rejections "
              f"{e['rejected']}; wall-line heading updates {e['yaw_updates']} ({e['yaw_rejected']} rejected)")
    else:
        print("never localised: was the reference marker in view, and are MARKER_HEIGHT / ARUCO_DICT right?")
    n = log["navigator"]
    print(f"paths planned {n['replans']} {n['reasons']}, map restarts {n['map_resets']}, dock searches "
          f"{n['dock_searches']}, docking re-approaches {n['dock_retries']}")
    s = log["io"]
    print(f"data: {s['odometry']} odometry readings ({s['odometry'] / t[-1]:.0f}/s), {s['scans']} scans "
          f"({s['scans'] / t[-1]:.1f}/s), {s['images']} camera frames ({s['images'] / t[-1]:.1f}/s); camera "
          f"calibration {'from camera_calibration.json' if s['calibrated'] else 'NOT done (simulation values)'}")
    print(f"loop processing time: mean {lt.mean():.0f} ms, 95th percentile {np.percentile(lt, 95):.0f} ms, max "
          f"{lt.max():.0f} ms (budget {1000 * hw.CONTROL_DT:.0f} ms); stale-data stops in "
          f"{int(np.sum(log['stale']))} periods")


def plot(log, nav, truth=None):
    """The robot's own view in its reference frame {A}: lidar map, planned paths, estimated path, markers."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 8))
    g = nav.grid
    occ = g.centres(np.argwhere(g.occupied))
    ax.plot(occ[:, 0], occ[:, 1], "s", ms=2, color="tab:orange", alpha=0.6, label="lidar occupancy grid")
    paths = log["navigator"]["paths"]
    if len(paths):
        ax.plot(paths[:, 0], paths[:, 1], "-", color="tab:green", lw=1, label="planned paths")
    xs = np.array([x for x in log["x"] if x is not None])
    if len(xs):
        ax.plot(xs[:, 0], xs[:, 1], "--", color="tab:blue", lw=1.8, label="estimated path")
    if truth is not None:
        ax.plot(truth[:, 0], truth[:, 1], "k", lw=1.2, label="true path (simulation)")
    ax.plot(0, 0, "s", color="tab:red", ms=8)
    ax.annotate(f"ref {hw.REF_MARKER_ID}", (0, 0), xytext=(5, 5), textcoords="offset points", color="tab:red")
    ax.plot(*hw.DOCK_OFFSET[:2], "s", color="tab:red", ms=8)
    ax.annotate(f"dock {hw.DOCK_ID}", hw.DOCK_OFFSET[:2], xytext=(5, 5), textcoords="offset points", color="tab:red")
    for i, p in log["estimator"]["mapped"].items():
        ax.plot(*p, "^", color="tab:purple", ms=7)
        ax.annotate(f"mapped {i}", p, xytext=(5, 5), textcoords="offset points", color="tab:purple")
    ax.set_aspect("equal"); ax.set_xlabel("x in {A} [m]"); ax.set_ylabel("y in {A} [m]"); ax.legend(fontsize=8)
    ax.set_title("hardware pipeline: the robot's map and paths in its reference frame {A}")
    plt.tight_layout(); plt.show()


def main():
    parser = argparse.ArgumentParser(description="Explore mission on the real TurtleBot3 (rosbridge)")
    parser.add_argument("host", nargs="?", help="robot IP address, e.g. 192.168.1.213 (omit with --sim)")
    parser.add_argument("--sim", action="store_true", help="run on the MuJoCo stand-in instead of the robot")
    parser.add_argument("--dry-run", action="store_true",
                        help="never send a command: run sensing, localisation and planning while the robot stays "
                             "put (or is driven by hand / teleop)")
    parser.add_argument("--odom-source", choices=["joint_states", "odom"], default="joint_states")
    parser.add_argument("--straight", action="store_true", help="straight segments instead of splines")
    parser.add_argument("--max-time", type=float, default=180.0)
    parser.add_argument("--show-camera", action="store_true", help="live annotated camera window")
    parser.add_argument("--video", default=None, metavar="PATH", help="save the annotated camera feed")
    parser.add_argument("--log", default=None, metavar="PATH", help="save the run (pickle) for offline analysis")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sim-viewer", action="store_true", help="with --sim: show the MuJoCo viewer")
    parser.add_argument("--obstacle-appears", type=float, default=None, metavar="T",
                        help="with --sim: an obstacle appears 0.8 m ahead of the robot at time T")
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()
    if not args.sim and not args.host:
        parser.error("give the robot's IP address, or --sim")

    if args.sim:
        from hardware_integration.sim_robot import SimRosbridgeRobot
        robot = SimRosbridgeRobot(seed=args.seed, viewer=args.sim_viewer,
                                  obstacle_appears=None if args.obstacle_appears is None
                                  else (args.obstacle_appears, 0.8))
    else:
        from hardware_integration.robot_io import Waffle
        robot = Waffle(args.host)
    print(f"connecting to {'the simulated robot' if args.sim else args.host} ...")
    with robot:
        io = RobotIO(robot, odom_source=args.odom_source)
        if not io.calibrated:
            print("WARNING: no camera_calibration.json: using the simulation's camera intrinsics "
                  "(run hardware_integration/calibrate_camera.py first)")
        log, nav = run(io, args.max_time, args.dry_run, not args.straight, args.show_camera, args.video, args.seed)
        truth = None
        if args.sim: # scoring against ground truth (the stand-in only)
            import main as sim_main
            dist, lat, hd = sim_main.docking_errors(robot.true_pose())
            F = sim_main.marker_frame(hw.REF_MARKER_ID)
            c, s = np.cos(F[2]), np.sin(F[2])
            d = np.array(log["true"])[:, :2] - F[:2]
            truth = np.column_stack([c * d[:, 0] + s * d[:, 1], -s * d[:, 0] + c * d[:, 1]]) # world -> {A}
            est = [(i, x) for i, x in enumerate(log["x"]) if x is not None]
            err = [np.linalg.norm(x[:2] - truth[i]) for i, x in est]
            print(f"(simulation) true docking error: distance {1000 * dist:+.0f} mm, lateral {1000 * lat:+.0f} mm, "
                  f"heading {np.degrees(hd):+.1f} deg; estimate error RMS {1000 * np.sqrt(np.mean(np.square(err))):.0f}"
                  f" mm, max {1000 * max(err):.0f} mm; collisions: {sorted(robot.collisions()) or 'none'}")
    summarise(log, nav)
    if args.log:
        Path(args.log).parent.mkdir(parents=True, exist_ok=True)
        with open(args.log, "wb") as f:
            pickle.dump({k: v for k, v in log.items()}, f)
        print(f"log saved to {args.log}")
    if not args.no_plot:
        plot(log, nav, truth)


if __name__ == "__main__":
    main()
