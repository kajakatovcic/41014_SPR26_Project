# Robustness of the whole system (for marking criteria E- integration and robustness under disturbance and noise).
# Every trial runs the full mission of main.py: perception -> EKF -> LQR tracking -> visual docking. Each scenario
# changes the WORLD only (never the robot's software) and draws its disturbance at random per trial on top of the
# random wheel-radius errors every trial has. Scored against ground truth:
#   success:   docked, within 30mm and 5deg of the docked pose, no collision, inside the time limit
#   per trial docking error, path error, EKF error and NEES, measurement rejections, filter recoveries, detours
#   failures:  each one is given a cause (estimator, controller, perception or obstacle), from the logged data
# --mission explore (default) runs the explore mission's scenarios, --mission route the earlier route mission's.
# Usage: python tests/test_robustness.py [--mission explore|route] [--trials 5] [--workers N]
#                                        [--scenarios name ...] [--controller lqr|straight] [--no-plot]
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
from src.control.mission import clearance, obstacle_distance, wall_distance

TIME_LIMIT = 150.0 #sec
SUCCESS = dict(distance=0.03, lateral=0.03, heading=np.deg2rad(5.0))


def _start_offset(rng, radius, heading_deg):
    r, a = radius * np.sqrt(rng.uniform()), rng.uniform(-np.pi, np.pi)
    x, y, th = cfg.START_POSE
    return (x + r * np.cos(a), y + r * np.sin(a), th + np.deg2rad(rng.uniform(-heading_deg, heading_deg)))


def _odometry(rng, scale, slip):
    return (rng.uniform(-scale, scale, 2), slip)


def _force(rng):
    return rng.choice([-1, 1]) * rng.uniform(15, 20)


def _obstacle_on_route(rng, controller):
    """An unmapped cylinder on the controller's own planned path, 15-22 % of the way along (the first leg, which has
    open floor beside it), up to 5 cm either side of it. Further along, the route threads the gap between the
    yellow and green cylinders, where an obstacle leaves no room for a local detour (0.52 m of free space for a
    0.5 m wide robot that keeps 2 cm clearance): getting past there needs global replanning, which this controller
    does not do."""
    from src.control.control import MissionController
    path = MissionController(smooth=(controller != "straight")).path
    i = int(np.searchsorted(path.s, rng.uniform(0.15, 0.22) * path.s[-1]))
    th = path.th[i]
    c = path.xy[i] + rng.uniform(-0.05, 0.05) * np.array([-np.sin(th), np.cos(th)])
    return [("cyl_grey", "cylinder", tuple(c), rng.uniform(0.08, 0.12), 0.4)]


def _visible_start(rng, facing=True):
    """Explore mission: a start 0.6-1.4 m in front of the reference marker, up to 40 deg off its normal, clear of the
    room's obstacles, facing the marker to within 15 deg (facing=True) or at any heading (the robot must search)."""
    F = main.marker_frame(cfg.REF_MARKER_ID)
    while True:
        d, a = rng.uniform(0.6, 1.4), np.deg2rad(rng.uniform(-40, 40))
        p = F[:2] + d * np.array([np.cos(F[2] + a), np.sin(F[2] + a)])
        if obstacle_distance(p)[0] > cfg.ROBOT_RADIUS + 0.05 and wall_distance(p) > cfg.ROBOT_RADIUS + 0.05:
            th = F[2] + a + np.pi + np.deg2rad(rng.uniform(-15, 15)) if facing else rng.uniform(-np.pi, np.pi)
            return (p[0], p[1], model.wrap(th))


def _obstacle_in_the_way(rng):
    """Explore mission: an extra cylinder (r 8-12 cm) near the straight line from the start to the pre-dock pose,
    25-60 % of the way and up to 10 cm either side, kept 0.3 m clear of the room's obstacles so it is a separate
    obstacle. The robot knows no obstacles at all, so this is simply a different room."""
    T = cfg.MARKERS[cfg.DOCK_ID]
    goal = T.t[:2] + cfg.PREDOCK_DISTANCE * T.R[:2, 2]
    start = np.array(cfg.EXPLORE_START[:2])
    u = (goal - start) / np.linalg.norm(goal - start)
    while True:
        r = rng.uniform(0.08, 0.12)
        c = start + rng.uniform(0.25, 0.6) * (goal - start) + rng.uniform(-0.1, 0.1) * np.array([-u[1], u[0]])
        if obstacle_distance(c)[0] > r + 0.3:
            return [("cyl_grey", "cylinder", tuple(c), r, 0.4)]


# name -> (description, function(rng, controller) -> keyword arguments for main.run)
# Pushes in "push" and "combined" happen while driving the ROUTE (10-25 s; the route ends at ~30 s on the spline,
# later on straight segments); "push_docking" pushes during docking.
ROUTE_SCENARIOS = {
    "nominal": ("wheel radius errors up to +/-0.8 %, default noise",
                lambda r, c: {}),
    "start_small": ("robot placed up to 0.1m and 5deg from where its software thinks",
                    lambda r, c: dict(true_start=_start_offset(r, 0.10, 5))),
    "start_large": ("robot placed up to 0.3m and 30deg from where its software thinks",
                    lambda r, c: dict(true_start=_start_offset(r, 0.30, 30))),
    "odometry_x3": ("wheel radius errors up to +/-2.5%, 3x slip",
                    lambda r, c: dict(encoder_errors=_odometry(r, 0.025, 3 * cfg.SIM_WHEEL_SLIP_SIGMA))),
    "camera_noise_x4": ("camera pixel noise 8 grey levels (4x)",
                        lambda r, c: dict(camera_noise=4 * cfg.CAMERA_NOISE)),
    "calibration_error": ("real camera's field of view 1 deg off the calibrated one (focal length ~2% off)",
                          lambda r, c: dict(camera_fovy=cfg.CAMERA_FOVY + r.choice([-1.0, 1.0]))),
    "lidar_noise_x3": ("lidar range noise 3cm (3x)",
                       lambda r, c: dict(lidar_sigma=3 * cfg.LIDAR_SIGMA)),
    "landmarks_missing": ("wall markers 2 and 4 missing from the room (still in the robot's map)",
                          lambda r, c: dict(world_markers={i: T for i, T in cfg.MARKERS.items() if i not in (2, 4)})),
    "push": ("shoved sideways (15-20N for 0.2sec, either side) at a random time in 10-25sec, on the route",
             lambda r, c: dict(push_at=(r.uniform(10, 25), _force(r)))),
    "push_docking": ("the same shove 2-5sec after visual docking starts",
                     lambda r, c: dict(push_at=("dock", r.uniform(2, 5), _force(r)))),
    "unexpected_obstacle": ("a cylinder (r 8-12cm) not in the map, on the controller's own planned route",
                            lambda r, c: dict(extra_obstacles=_obstacle_on_route(r, c))),
    "combined": ("start up to 0.1m / 5deg off + 3x odometry errors + 2x camera noise + a push on the route",
                 lambda r, c: dict(true_start=_start_offset(r, 0.10, 5),
                                   encoder_errors=_odometry(r, 0.025, 3 * cfg.SIM_WHEEL_SLIP_SIGMA),
                                   camera_noise=2 * cfg.CAMERA_NOISE,
                                   push_at=(r.uniform(10, 25), _force(r)))),
}

# Explore mission (the default): the robot localises from the reference marker (~1 s), looks for the dock marker
# (~3 s), then drives (~5-40 s) and docks. Pushes on the way come at 12-30 s.
EXPLORE_SCENARIOS = {
    "nominal": ("default start facing the reference marker, wheel radius errors up to +/-0.8 %, default noise",
                lambda r, c: {}),
    "start_random": ("random start 0.6-1.4 m in front of the reference marker, up to 40 deg off its normal",
                     lambda r, c: dict(true_start=_visible_start(r))),
    "start_search": ("the same positions at a random heading: the robot must turn to find the marker",
                     lambda r, c: dict(true_start=_visible_start(r, facing=False))),
    "odometry_x3": ("wheel radius errors up to +/-2.5%, 3x slip",
                    lambda r, c: dict(encoder_errors=_odometry(r, 0.025, 3 * cfg.SIM_WHEEL_SLIP_SIGMA))),
    "camera_noise_x4": ("camera pixel noise 8 grey levels (4x)",
                        lambda r, c: dict(camera_noise=4 * cfg.CAMERA_NOISE)),
    "calibration_error": ("real camera's field of view 1 deg off the calibrated one (focal length ~2% off)",
                          lambda r, c: dict(camera_fovy=cfg.CAMERA_FOVY + r.choice([-1.0, 1.0]))),
    "lidar_noise_x3": ("lidar range noise 3cm (3x)",
                       lambda r, c: dict(lidar_sigma=3 * cfg.LIDAR_SIGMA)),
    "landmarks_missing": ("only the reference and dock markers in the room (no other markers to map)",
                          lambda r, c: dict(world_markers={i: T for i, T in cfg.MARKERS.items()
                                                           if i in (cfg.REF_MARKER_ID, cfg.DOCK_ID)})),
    "push": ("shoved sideways (15-20N for 0.2sec, either side) at a random time in 12-30sec, while driving",
             lambda r, c: dict(push_at=(r.uniform(12, 30), _force(r)))),
    "push_docking": ("the same shove 2-5sec after visual docking starts",
                     lambda r, c: dict(push_at=("dock", r.uniform(2, 5), _force(r)))),
    "extra_obstacle": ("an extra cylinder (r 8-12cm) near the straight line from the start to the dock",
                       lambda r, c: dict(extra_obstacles=_obstacle_in_the_way(r))),
    "obstacle_appears": ("a cylinder (r 10cm) appears 0.6-1.0m straight ahead of the robot at 12-25sec, while driving",
                         lambda r, c: dict(obstacle_appears=(r.uniform(12, 25), r.uniform(0.6, 1.0)))),
    "combined": ("random start + 3x odometry errors + 2x camera noise + a push while driving",
                 lambda r, c: dict(true_start=_visible_start(r),
                                   encoder_errors=_odometry(r, 0.025, 3 * cfg.SIM_WHEEL_SLIP_SIGMA),
                                   camera_noise=2 * cfg.CAMERA_NOISE,
                                   push_at=(r.uniform(12, 30), _force(r)))),
}
SCENARIOS = {"explore": EXPLORE_SCENARIOS, "route": ROUTE_SCENARIOS}


def diagnose(m):
    """Why a trial failed based on what was logged."""
    if m["collisions"]:
        return f"collision with {', '.join(m['collisions'])} (controller/route)"
    if np.isnan(m["t_localised"]):
        return "never localised: the reference marker was not found (search)"
    if not m["docked"] and m["final_mode"] == "plan":
        return "no path found on the lidar map (planner)"
    if not m["docked"]:
        if m["final_ekf_err"] > 0.10:
            return f"estimator lost: EKF {1000 * m['final_ekf_err']:.0f}mm off at the end"
        if m["final_mode"] == "track" and m["blocked_end"]:
            return "blocked by an obstacle, no detour found (controller)"
        if m["final_mode"] == "dock":
            return "docking did not converge (perception/docking law)"
        return f"timeout in {m['final_mode']} mode"
    if m["pushed_while_docking"]:
        return ("pushed while docking: the docking law cannot remove a sideways offset, and one marker cannot "
                "observe it (controller + sensing layout)")
    if m["predock_ekf_err"] > 0.03:
        return f"docked off target: EKF {1000 * m['predock_ekf_err']:.0f}mm off at the pre-dock pose (estimator)"
    return "docked off target with a good estimate at pre-dock (perception/docking law)"


def trial(args):
    """One mission- returns plain metrics (runs in a worker process)"""
    scenario, i, seed, controller, mission = args
    rng = np.random.default_rng(seed)
    kwargs = SCENARIOS[mission][scenario][1](rng, controller)
    kwargs.setdefault("encoder_errors", _odometry(rng, 0.008, cfg.SIM_WHEEL_SLIP_SIGMA))
    t0 = time.time()
    log, c = main.run(duration=TIME_LIMIT, seed=seed, mission=mission, controller=controller, **kwargs)
    a = log.arrays()
    ok = ~np.isnan(a["est"][:, 0]) # localised (always, in the route mission)
    e = a["est"] - a["true"]
    e[:, 2] = model.wrap(e[:, 2])
    e_pos = np.linalg.norm(e[:, :2], axis=1) # NaN before localisation
    nees = np.array([ei @ np.linalg.solve(Pi, ei) for ei, Pi in zip(e[ok], a["P"][ok])])
    tracking = a["mode"] == "track"
    path = main.path_errors(a["true"][tracking, :2], log.ref_path) if tracking.any() else np.array([np.nan])
    dock_start = np.flatnonzero(a["mode"] == "dock")
    dist, lat, hd = main.docking_errors(a["true"][-1])
    first = np.flatnonzero(ok)[:1]
    m = dict(scenario=scenario, trial=i, seed=seed, mission=mission, docked=bool(c.done), time=float(a["t"][-1]),
             dist=float(dist), lat=float(lat), heading=float(hd),
             path_rms=float(np.sqrt(np.nanmean(path ** 2))),
             ekf_rms=float(np.sqrt(np.mean(e_pos[ok] ** 2))) if ok.any() else np.nan,
             ekf_max=float(np.nanmax(e_pos)) if ok.any() else np.nan, final_ekf_err=float(e_pos[-1]),
             nees=float(np.mean(nees)) if len(nees) else np.nan,
             predock_ekf_err=float(e_pos[dock_start[0]]) if len(dock_start) else np.nan,
             rejected=int(log.ekf_rejected), recoveries=int(log.ekf_recoveries),
             detours=int(log.replans if mission == "explore" else log.detours),
             t_localised=float(a["t"][first[0]]) if len(first) else np.nan,
             init_err=float(e_pos[first[0]]) if len(first) else np.nan,
             clearance=float(clearance(a["true"][:, :2])[0]),
             collisions=sorted(log.collisions), final_mode=c.mode, blocked_end=bool(a["blocked"][-1]),
             wall_time=time.time() - t0)
    push = kwargs.get("push_at")
    m["pushed_while_docking"] = bool(push is not None and len(dock_start) and
                                     (push[0] == "dock" or push[0] >= a["t"][dock_start[0]]))
    m["success"] = (m["docked"] and not m["collisions"] and abs(dist) < SUCCESS["distance"]
                    and abs(lat) < SUCCESS["lateral"] and abs(hd) < SUCCESS["heading"])
    m["cause"] = "" if m["success"] else diagnose(m)
    return m


def report(results):
    mission = results[0]["mission"]
    print(f'\n{"scenario":>19} | {"success":>8} {"docked":>7} | {"dock dist":>10} {"lateral":>9} {"heading":>8} | '
          f'{"path RMS":>9} | {"EKF RMS":>8} {"EKF max":>8} {"NEES":>6} | {"recov.":>6} '
          f'{"plans" if mission == "explore" else "detours":>7} {"clear":>7} {"time":>6}')
    for name in SCENARIOS[mission]:
        rs = [m for m in results if m["scenario"] == name]
        if not rs:
            continue
        d = [m for m in rs if m["docked"]]
        f = lambda key, k=1000: np.mean([abs(m[key]) for m in d]) * k if d else np.nan       # noqa: E731
        print(f"{name:>19} | {sum(m['success'] for m in rs):>3}/{len(rs):<4} {sum(m['docked'] for m in rs):>3}/{len(rs):<3} | "
              f"{f('dist'):>7.1f} mm {f('lat'):>6.1f} mm {f('heading', np.degrees(1)):>4.1f} deg | "
              f"{1000 * np.nanmean([m['path_rms'] for m in rs]):>6.1f} mm | "
              f"{1000 * np.nanmean([m['ekf_rms'] for m in rs]):>5.1f} mm {1000 * np.nanmax([m['ekf_max'] for m in rs]):>5.0f} mm "
              f"{np.nanmean([m['nees'] for m in rs]):>6.2f} | {np.mean([m['recoveries'] for m in rs]):>6.1f} "
              f"{np.mean([m['detours'] for m in rs]):>7.1f} {100 * min(m['clearance'] for m in rs):>4.1f} cm "
              f"{np.mean([m['time'] for m in rs]):>5.0f}s")
    if mission == "explore":
        loc = [m for m in results if not np.isnan(m["t_localised"])]
        print(f"\nlocalised in {len(loc)}/{len(results)} missions, after {np.mean([m['t_localised'] for m in loc]):.1f} s "
              f"on average (max {max(m['t_localised'] for m in loc):.1f} s); position error at localisation: median "
              f"{1000 * np.median([m['init_err'] for m in loc]):.0f} mm, max {1000 * max(m['init_err'] for m in loc):.0f} mm")
    fails = [m for m in results if not m["success"]]
    print(f"\noverall: {len(results) - len(fails)}/{len(results)} successful "
          f"(docked within {1000 * SUCCESS['distance']:.0f}mm / {np.degrees(SUCCESS['heading']):.0f} deg, "
          f"no collision, under {TIME_LIMIT:.0f} s)")
    if fails:
        print("failures:")
        for m in fails:
            print(f"   {m['scenario']} trial {m['trial']} (seed {m['seed']}): {m['cause']}")
    print("(dock distance, lateral and heading are means of |error| over the docked trials; EKF max is the worst "
          "over all trials; NEES ~3 = consistent; clear = the closest any trial came to an obstacle, beyond the "
          "robot's 0.25 m radius)")


def plot(results):
    import matplotlib.pyplot as plt
    names = [n for n in SCENARIOS[results[0]["mission"]] if any(m["scenario"] == n for m in results)]
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
    parser.add_argument("--mission", choices=["explore", "route"], default="explore")
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    parser.add_argument("--scenarios", nargs="*", default=None,
                        choices=sorted(set(EXPLORE_SCENARIOS) | set(ROUTE_SCENARIOS)))
    parser.add_argument("--controller", choices=["lqr", "straight"], default="lqr",
                        help="lqr = paths as splines; straight = the same controller on straight segments")
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    scenarios = SCENARIOS[args.mission]
    chosen = args.scenarios or list(scenarios)
    print(f"mission: {args.mission}, controller: {args.controller}")
    for n in chosen:
        print(f"{n:>19}: {scenarios[n][0]}")
    jobs = [(n, i, 1000 * k + i, args.controller, args.mission)
            for k, n in enumerate(scenarios) if n in chosen for i in range(args.trials)]
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
