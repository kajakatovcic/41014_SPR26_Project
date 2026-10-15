# Robustness of the whole system (for marking criteria E- integration and robustness under disturbance and noise).
# Every trial runs the full mission of main.py: perception -> EKF -> LQR tracking -> visual docking. Each scenario
# changes the WORLD only (never the robot's software) and draws its disturbance at random per trial on top of the
# random wheel-radius errors every trial has. Scored against ground truth:
#   success:   docked, within 30mm and 5deg of the docked pose, no collision, inside the time limit
#   per trial docking error, path error, EKF error and NEES, measurement rejections, filter recoveries, detours
#   failures:  each one is given a cause (estimator, controller, perception or obstacle), from the logged data
# Usage: python tests/test_robustness.py [--trials 5] [--workers N] [--scenarios name ...] [--no-plot]
import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import main                                              
from src import config as cfg                               
from src.control import model                                

TIME_LIMIT = 150.0 #sec
SUCCESS = dict(distance=0.03, lateral=0.03, heading=np.deg2rad(5.0))


def _start_offset(rng, radius, heading_deg):
    r, a = radius * np.sqrt(rng.uniform()), rng.uniform(-np.pi, np.pi)
    x, y, th = cfg.START_POSE
    return (x + r * np.cos(a), y + r * np.sin(a), th + np.deg2rad(rng.uniform(-heading_deg, heading_deg)))


def _odometry(rng, scale, slip):
    return (rng.uniform(-scale, scale, 2), slip)


# name -> (description, function(rng) -> keyword arguments for main.run)
SCENARIOS = {
    "nominal": ("wheel radius errors up to +/-0.8 %, default noise",
                lambda r: {}),
    "start_small": ("robot placed up to 0.1m and 5deg from where its software thinks",
                    lambda r: dict(true_start=_start_offset(r, 0.10, 5))),
    "start_large": ("robot placed up to 0.3m and 30deg from where its software thinks",
                    lambda r: dict(true_start=_start_offset(r, 0.30, 30))),
    "odometry_x3": ("wheel radius errors up to +/-2.5%, 3x slip",
                    lambda r: dict(encoder_errors=_odometry(r, 0.025, 3 * cfg.SIM_WHEEL_SLIP_SIGMA))),
    "camera_noise_x4": ("camera pixel noise 8 grey levels (4x)",
                        lambda r: dict(camera_noise=4 * cfg.CAMERA_NOISE)),
    "calibration_error": ("real camera's field of view 1 deg off the calibrated one (focal length ~2% off)",
                          lambda r: dict(camera_fovy=cfg.CAMERA_FOVY + r.choice([-1.0, 1.0]))),
    "lidar_noise_x3": ("lidar range noise 3cm (3x)",
                       lambda r: dict(lidar_sigma=3 * cfg.LIDAR_SIGMA)),
    "landmarks_missing": ("wall markers 2 and 4 missing from the room (still in the robot's map)",
                          lambda r: dict(world_markers={i: T for i, T in cfg.MARKERS.items() if i not in (2, 4)})),
    "push": ("shoved sideways (15-20N for 0.2sec, either side) at a random time in 10-40sec",
             lambda r: dict(push_at=(r.uniform(10, 40), r.choice([-1, 1]) * r.uniform(15, 20)))),
    "unexpected_obstacle": ("a cylinder (r 8-12cm) not in the map, on the route between waypoints 1 and 2",
                            lambda r: dict(extra_obstacles=[("cyl_grey", "cylinder",
                                                             (1.2 + r.uniform(-0.05, 0.05), r.uniform(-1.0, -0.2)),
                                                             r.uniform(0.08, 0.12), 0.4)])),
    "combined": ("start up to 0.1m / 5deg off + 3x odometry errors + 2x camera noise + a push",
                 lambda r: dict(true_start=_start_offset(r, 0.10, 5),
                                encoder_errors=_odometry(r, 0.025, 3 * cfg.SIM_WHEEL_SLIP_SIGMA),
                                camera_noise=2 * cfg.CAMERA_NOISE,
                                push_at=(r.uniform(10, 40), r.choice([-1, 1]) * r.uniform(15, 20)))),
}


def diagnose(m):
    """Why a trial failed based on what was logged."""
    if m["collisions"]:
        return f"collision with {', '.join(m['collisions'])} (controller/route)"
    if not m["docked"]:
        if m["final_ekf_err"] > 0.10:
            return f"estimator lost: EKF {1000 * m['final_ekf_err']:.0f}mm off at the end"
        if m["final_mode"] == "track" and m["blocked_end"]:
            return "blocked by an obstacle, no detour found (controller)"
        if m["final_mode"] == "dock":
            return "docking did not converge (perception/docking law)"
        return f"timeout in {m['final_mode']} mode"
    if m["predock_ekf_err"] > 0.03:
        return f"docked off target: EKF {1000 * m['predock_ekf_err']:.0f}mm off at the pre-dock pose (estimator)"
    return "docked off target with a good estimate at pre-dock (perception/docking law)"


def trial(args):
    """One mission- returns plain metrics (runs in a worker process)"""
    scenario, i, seed = args
    rng = np.random.default_rng(seed)
    kwargs = SCENARIOS[scenario][1](rng)
    kwargs.setdefault("encoder_errors", _odometry(rng, 0.008, cfg.SIM_WHEEL_SLIP_SIGMA))
    t0 = time.time()
    log, c = main.run(duration=TIME_LIMIT, seed=seed, **kwargs)
    a = log.arrays()
    e = a["est"] - a["true"]
    e[:, 2] = model.wrap(e[:, 2])
    e_pos = np.linalg.norm(e[:, :2], axis=1)
    nees = np.array([ei @ np.linalg.solve(Pi, ei) for ei, Pi in zip(e, a["P"])])
    tracking = a["mode"] == "track"
    path = main.path_errors(a["true"][tracking, :2]) if tracking.any() else np.array([np.nan])
    dock_start = np.flatnonzero(a["mode"] == "dock")
    dist, lat, hd = main.docking_errors(a["true"][-1])
    m = dict(scenario=scenario, trial=i, seed=seed, docked=bool(c.done), time=float(a["t"][-1]),
             dist=float(dist), lat=float(lat), heading=float(hd),
             path_rms=float(np.sqrt(np.nanmean(path ** 2))), ekf_rms=float(np.sqrt(np.mean(e_pos ** 2))),
             ekf_max=float(e_pos.max()), final_ekf_err=float(e_pos[-1]), nees=float(np.mean(nees)),
             predock_ekf_err=float(e_pos[dock_start[0]]) if len(dock_start) else np.nan,
             rejected=int(log.ekf_rejected), recoveries=int(log.ekf_recoveries), detours=int(log.detours),
             collisions=sorted(log.collisions), final_mode=c.mode, blocked_end=bool(a["blocked"][-1]),
             wall_time=time.time() - t0)
    m["success"] = (m["docked"] and not m["collisions"] and abs(dist) < SUCCESS["distance"]
                    and abs(lat) < SUCCESS["lateral"] and abs(hd) < SUCCESS["heading"])
    m["cause"] = "" if m["success"] else diagnose(m)
    return m


def report(results):
    print(f'\n{"scenario":>19} | {"success":>8} {"docked":>7} | {"dock dist":>10} {"lateral":>9} {"heading":>8} | '
          f'{"path RMS":>9} | {"EKF RMS":>8} {"EKF max":>8} {"NEES":>6} | {"recov.":>6} {"detours":>7} {"time":>6}')
    for name in SCENARIOS:
        rs = [m for m in results if m["scenario"] == name]
        if not rs:
            continue
        d = [m for m in rs if m["docked"]]
        f = lambda key, k=1000: np.mean([abs(m[key]) for m in d]) * k if d else np.nan       # noqa: E731
        print(f"{name:>19} | {sum(m['success'] for m in rs):>3}/{len(rs):<4} {sum(m['docked'] for m in rs):>3}/{len(rs):<3} | "
              f"{f('dist'):>7.1f} mm {f('lat'):>6.1f} mm {f('heading', np.degrees(1)):>4.1f} deg | "
              f"{1000 * np.nanmean([m['path_rms'] for m in rs]):>6.1f} mm | "
              f"{1000 * np.mean([m['ekf_rms'] for m in rs]):>5.1f} mm {1000 * np.max([m['ekf_max'] for m in rs]):>5.0f} mm "
              f"{np.mean([m['nees'] for m in rs]):>6.2f} | {np.mean([m['recoveries'] for m in rs]):>6.1f} "
              f"{np.mean([m['detours'] for m in rs]):>7.1f} {np.mean([m['time'] for m in rs]):>5.0f}s")
    fails = [m for m in results if not m["success"]]
    print(f"\noverall: {len(results) - len(fails)}/{len(results)} successful "
          f"(docked within {1000 * SUCCESS['distance']:.0f}mm / {np.degrees(SUCCESS['heading']):.0f} deg, "
          f"no collision, under {TIME_LIMIT:.0f} s)")
    if fails:
        print("failures:")
        for m in fails:
            print(f"   {m['scenario']} trial {m['trial']} (seed {m['seed']}): {m['cause']}")
    print("(dock distance, lateral and heading are means of |error| over the docked trials; EKF max is the worst "
          "over all trials; NEES ~3 = consistent)")


def plot(results):
    import matplotlib.pyplot as plt
    names = [n for n in SCENARIOS if any(m["scenario"] == n for m in results)]
    fig, ax = plt.subplots(2, 2, figsize=(14, 9))

    a = ax[0, 0]
    rate = [100 * np.mean([m["success"] for m in results if m["scenario"] == n]) for n in names]
    a.barh(names, rate, color=["tab:green" if r == 100 else "tab:orange" if r >= 80 else "tab:red" for r in rate])
    a.set_xlim(0, 100); a.set_xlabel("success rate [%]"); a.invert_yaxis(); a.set_title("success rate by scenario")

    a = ax[0, 1]
    colours = plt.get_cmap("tab20")
    for k, n in enumerate(names):
        d = [m for m in results if m["scenario"] == n and m["docked"]]
        a.plot([1000 * m["lat"] for m in d], [1000 * m["dist"] for m in d], "o", color=colours(k), label=n, alpha=0.8)
    a.add_patch(plt.Rectangle((-30, -30), 60, 60, fill=False, ls="--", ec="k"))
    a.set_xlabel("lateral docking error [mm]"); a.set_ylabel("distance docking error [mm]")
    a.legend(fontsize=7, ncol=2); a.set_title("docking errors (dashed: success box)")

    a = ax[1, 0]
    a.boxplot([[1000 * m["ekf_max"] for m in results if m["scenario"] == n] for n in names], vert=False,
              tick_labels=names)
    a.set_xscale("log"); a.set_xlabel("worst EKF position error in the mission [mm]"); a.invert_yaxis()
    a.set_title("estimation error by scenario")

    a = ax[1, 1]
    a.boxplot([[m["time"] for m in results if m["scenario"] == n] for n in names], vert=False, tick_labels=names)
    a.set_xlabel("mission time [s]"); a.invert_yaxis(); a.set_title("mission time by scenario")
    plt.tight_layout(); plt.show()


def cli():
    parser = argparse.ArgumentParser(description="Robustness trials for the whole system")
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    parser.add_argument("--scenarios", nargs="*", default=list(SCENARIOS), choices=list(SCENARIOS))
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    for n in args.scenarios:
        print(f"{n:>19}: {SCENARIOS[n][0]}")
    jobs = [(n, i, 1000 * k + i) for k, n in enumerate(SCENARIOS) if n in args.scenarios for i in range(args.trials)]
    print(f"\n{len(jobs)} missions on {args.workers} worker processes...")
    t0 = time.time()
    with ProcessPoolExecutor(args.workers) as pool:
        results = list(pool.map(trial, jobs))
    print(f"done in {time.time() - t0:.0f} s")
    report(results)
    if not args.no_plot:
        plot(results)


if __name__ == "__main__":
    cli()
