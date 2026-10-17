# Evaluate the explore mission (main.py default)- Localisation from one marker then planning on the LiDAR map and
# docking, with nothing known but the dock marker's offset from the reference marker.
#   Part 0 - the robot's prior knowledge: cfg.DOCK_OFFSET (the hand-measured offset) against the simulated room
#   Part 1 - localisation from one marker (MarkerSLAM's initialisation): the robot stands still at a grid of
#            distances and angles in front of the reference marker. Pose error and its consistency with the prior
#            covariance (NEES), with the marker's yaw from the lidar wall line (used) and from PnP alone (Week 3 E,
#            which can flip: planar pose ambiguity)
#   Part 2 - missions from random starts in front of the reference marker, with random wheel-radius errors, for the
#            spline paths and for straight segments (same seeds): success, time, localisation, estimation,
#            replanning, how close the robot came to obstacles (ground truth), docking
# Usage: python tests/test_navigation.py [--trials 10] [--workers N] [--parts 0 1 2] [--no-plot]
#        (all parts take ~10 min on 16 cores; the figure needs parts 1 and 2 in the same run)
import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import mujoco
import numpy as np
from scipy.stats import chi2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import main
from src import config as cfg
from src.control import model
from src.control.mission import clearance, obstacle_distance
from src.estimation.estimation import MarkerSLAM
from src.perception import perception as per
from src.simulation import world as sim
from test_robustness import SUCCESS, _visible_start

TIME_LIMIT = 150.0


def world_to_frame(x, F):
    """Inverse of main.pose_to_world: world pose -> pose in frame F = (x, y, phi)."""
    c, s = np.cos(F[2]), np.sin(F[2])
    d = np.asarray(x[:2]) - F[:2]
    return np.array([c * d[0] + s * d[1], -s * d[0] + c * d[1], model.wrap(x[2] - F[2])])


def part0():
    print("\n=== Part 0: the robot's prior knowledge (cfg.DOCK_OFFSET) against the simulated room")
    F_ref, F_dock = main.marker_frame(cfg.REF_MARKER_ID), main.marker_frame(cfg.DOCK_ID)
    true = world_to_frame(F_dock, F_ref)
    err = np.array(cfg.DOCK_OFFSET) - true
    err[2] = model.wrap(err[2])
    print(f"dock marker in the reference marker's frame: config ({cfg.DOCK_OFFSET[0]:.3f}, {cfg.DOCK_OFFSET[1]:.3f}, "
          f"{np.degrees(cfg.DOCK_OFFSET[2]):.1f} deg), room ({true[0]:.3f}, {true[1]:.3f}, {np.degrees(true[2]):.1f} deg)")
    ok = np.all(np.abs(err[:2]) < 0.001) and abs(err[2]) < np.deg2rad(0.1)
    print("within 1 mm and 0.1 deg: " + ("PASS" if ok else "FAIL"))
    return ok


def localise_at(m, d, cam, rng, walls):
    """MarkerSLAM's initialisation with the robot standing still (camera at 10 Hz, lidar at 5 Hz)."""
    est = MarkerSLAM()
    for k in range(3 * cfg.INIT_FRAMES):
        if walls and k % 2 == 0:
            est.walls = per.process_scan(sim.read_lidar(m, d, rng), rng).walls
        est.predict((0.0, 0.0), cfg.CONTROL_DT)
        est.update(per.detect_markers(cam.render(d, rng), known=None))
        if est.ready:
            return est
    return None


def part1(seed=0, distances=(0.6, 0.9, 1.2, 1.5), angles=(-40, -20, 0, 20, 40)):
    print("\n=== Part 1: localisation from the reference marker alone (robot still, "
          f"{cfg.INIT_FRAMES} frames): marker yaw from the lidar wall line (used) against PnP alone (Week 3 E)")
    m = sim.build_world()
    d = mujoco.MjData(m)
    cam = sim.Camera(m)
    rng = np.random.default_rng(seed)
    F = main.marker_frame(cfg.REF_MARKER_ID)
    rows = {"wall": [], "pnp": []}
    print(f'{"dist [m]":>8} {"angle":>6} | {"wall: pos err":>13} {"heading err":>12} {"in 2-sigma":>11} {"NEES":>6} | '
          f'{"PnP: pos err":>12} {"heading err":>12} {"in 2-sigma":>11} {"NEES":>6}')
    for dist in distances:
        for ang in angles:
            a = F[2] + np.deg2rad(ang)
            p = F[:2] + dist * np.array([np.cos(a), np.sin(a)])
            if obstacle_distance(p)[0] < cfg.ROBOT_RADIUS:
                continue
            sim.set_pose(m, d, p[0], p[1], a + np.pi)
            for _ in range(200):
                mujoco.mj_step(m, d)
            truth = world_to_frame(sim.true_pose(m, d), F)
            line = f"{dist:>8.1f} {ang:>5d}  |"
            for kind in ("wall", "pnp"):
                est = localise_at(m, d, cam, rng, walls=(kind == "wall"))
                if est is None:
                    rows[kind].append((dist, ang, np.nan, np.nan, np.nan, False, False))
                    line += f" {'marker not detected':>45} |"
                    continue
                e = est.x - truth
                e[2] = model.wrap(e[2])
                P = est.P
                n = float(e @ np.linalg.solve(P, e))
                in_pos = np.linalg.norm(e[:2]) < 2 * np.sqrt(P[0, 0] + P[1, 1])
                in_th = abs(e[2]) < 2 * np.sqrt(P[2, 2])
                rows[kind].append((dist, ang, np.linalg.norm(e[:2]), e[2], n, in_pos, in_th))
                src = "" if est.init_yaw_source == kind else f" ({est.init_yaw_source})"
                line += (f" {1000 * np.linalg.norm(e[:2]):>10.0f} mm {np.degrees(e[2]):>+8.2f} deg "
                         f"{'yes' if in_pos and in_th else 'NO':>6}{src:<5} {n:>6.2f} |")
            print(line)
    cam.close()
    ok = True
    for kind, label in (("wall", "yaw from the wall line"), ("pnp", "yaw from PnP alone")):
        r = np.array([row for row in rows[kind] if not np.isnan(row[2])], dtype=float)
        print(f"{label}: {len(r)} of {len(rows[kind])} views localised; heading within 2 deg in "
              f"{100 * np.mean(np.abs(np.degrees(r[:, 3])) < 2):.0f}%, median position error "
              f"{1000 * np.median(r[:, 2]):.0f} mm, worst {1000 * r[:, 2].max():.0f} mm / "
              f"{np.degrees(np.abs(r[:, 3]).max()):.1f} deg; inside the prior's 2-sigma bounds: position "
              f"{100 * r[:, 5].mean():.0f}%, heading {100 * r[:, 6].mean():.0f}%; NEES below the 97.5% bound "
              f"({chi2.ppf(0.975, 3):.2f}) in {100 * np.mean(r[:, 4] < chi2.ppf(0.975, 3)):.0f}%")
        if kind == "wall":
            ok = len(r) >= 0.9 * len(rows[kind]) and np.mean(np.abs(np.degrees(r[:, 3])) < 2) >= 0.9
    print("wall line: localised from >= 90% of views with the heading within 2 deg in >= 90%: "
          + ("PASS" if ok else "FAIL"))
    return ok, rows


def mission(args):
    """One explore mission from a random start (worker process): plain metrics."""
    i, seed, controller = args
    rng = np.random.default_rng(seed)
    start = _visible_start(rng)
    errs = (rng.uniform(-0.008, 0.008, 2), cfg.SIM_WHEEL_SLIP_SIGMA)
    t0 = time.time()
    log, c = main.run(duration=TIME_LIMIT, seed=seed, controller=controller, true_start=start, encoder_errors=errs)
    a = log.arrays()
    ok = ~np.isnan(a["est"][:, 0])
    e = a["est"] - a["true"]
    e[:, 2] = model.wrap(e[:, 2])
    ep = np.linalg.norm(e[:, :2], axis=1)
    nees = [ei @ np.linalg.solve(Pi, ei) for ei, Pi in zip(e[ok], a["P"][ok])]
    first = np.flatnonzero(ok)[0] if ok.any() else None
    dist, lat, hd = main.docking_errors(a["true"][-1])
    oc, wc = clearance(a["true"][:, :2])
    mode = a["mode"]
    t_of = lambda name: float(a["t"][np.flatnonzero(mode == name)[0]]) if np.any(mode == name) else np.nan  # noqa
    m = dict(i=i, seed=seed, controller=controller, start=start, docked=bool(c.done), time=float(a["t"][-1]),
             t_loc=float(a["t"][first]) if first is not None else np.nan,
             t_drive=t_of("track"), t_dock=t_of("dock"),
             init_pos=float(ep[first]) if first is not None else np.nan,
             init_th=float(e[first, 2]) if first is not None else np.nan,
             ekf_rms=float(np.sqrt(np.mean(ep[ok] ** 2))) if ok.any() else np.nan,
             ekf_max=float(np.nanmax(ep)) if ok.any() else np.nan, nees=float(np.mean(nees)) if nees else np.nan,
             replans=log.replans, reasons=log.replan_reasons, failures=log.plan_failures, resets=log.map_resets,
             clear_obs=float(oc), clear_wall=float(wc), collisions=sorted(log.collisions),
             dist=float(dist), lat=float(lat), heading=float(hd), true_xy=a["true"][:, :2],
             wall=time.time() - t0)
    m["success"] = (m["docked"] and not m["collisions"] and abs(dist) < SUCCESS["distance"]
                    and abs(lat) < SUCCESS["lateral"] and abs(hd) < SUCCESS["heading"])
    return m


def part2(n, workers):
    print(f"\n=== Part 2: explore missions from {n} random starts in front of the reference marker "
          "(0.6-1.4 m, up to 40 deg off its normal), spline paths and straight segments on the same seeds")
    jobs = [(i, 700 + i, c) for c in ("lqr", "straight") for i in range(n)]
    with ProcessPoolExecutor(min(workers, len(jobs))) as pool:
        res = list(pool.map(mission, jobs))
    print(f'{"ctrl":>8} {"trial":>5} {"ok":>3} {"time":>6} | {"localised":>9} {"init err":>16} | '
          f'{"EKF RMS":>8} {"max":>6} {"NEES":>6} | {"plans":>5} {"resets":>6} | {"clear obs":>9} {"walls":>6} | '
          f'{"dock dist":>9} {"lateral":>8} {"heading":>8}')
    for m in res:
        print(f"{m['controller']:>8} {m['i']:>5} {'yes' if m['success'] else 'NO':>3} {m['time']:>5.1f}s | "
              f"{m['t_loc']:>8.1f}s {1000 * m['init_pos']:>6.0f} mm {np.degrees(m['init_th']):>+5.1f} deg | "
              f"{1000 * m['ekf_rms']:>5.0f} mm {1000 * m['ekf_max']:>3.0f} mm {m['nees']:>6.2f} | {m['replans']:>5} "
              f"{m['resets']:>6} | {100 * m['clear_obs']:>6.1f} cm {100 * m['clear_wall']:>3.0f} cm | "
              f"{1000 * m['dist']:>+6.0f} mm {1000 * m['lat']:>+5.0f} mm {np.degrees(m['heading']):>+5.1f} deg"
              + (f"  collisions: {m['collisions']}" if m["collisions"] else ""))
    ok = True
    for c in ("lqr", "straight"):
        rs = [m for m in res if m["controller"] == c]
        mean = lambda k, s=1.0: s * np.nanmean([m[k] for m in rs])  # noqa: E731
        reasons = {}
        for m in rs:
            for k, v in m["reasons"].items():
                reasons[k] = reasons.get(k, 0) + v
        print(f"\n{c}: {sum(m['success'] for m in rs)}/{len(rs)} successful, {sum(m['docked'] for m in rs)}/{len(rs)} "
              f"docked, collisions in {sum(bool(m['collisions']) for m in rs)}")
        print(f"   time {mean('time'):.1f} s (localised at {mean('t_loc'):.1f} s, driving from {mean('t_drive'):.1f} s, "
              f"docking from {mean('t_dock'):.1f} s)")
        print(f"   EKF RMS {mean('ekf_rms', 1000):.0f} mm, worst {1000 * max(m['ekf_max'] for m in rs):.0f} mm, "
              f"mean NEES {mean('nees'):.2f}")
        print(f"   paths planned per mission {mean('replans'):.1f} (in total: {reasons}), failed plans "
              f"{sum(m['failures'] for m in rs)}, map restarts {mean('resets'):.1f}")
        print(f"   closest approach beyond the robot's radius: obstacles {100 * min(m['clear_obs'] for m in rs):.1f} cm "
              f"(mean of the per-mission minimum {mean('clear_obs', 100):.1f} cm), walls "
              f"{100 * min(m['clear_wall'] for m in rs):.1f} cm")
        print(f"   docking |error|: distance {1000 * np.mean([abs(m['dist']) for m in rs]):.1f} mm, lateral "
              f"{1000 * np.mean([abs(m['lat']) for m in rs]):.1f} mm, heading "
              f"{np.degrees(np.mean([abs(m['heading']) for m in rs])):.1f} deg")
        if c == "lqr":
            ok = sum(m["success"] for m in rs) >= 0.9 * len(rs) and not any(m["collisions"] for m in rs)
    print("\nspline paths: >= 90% successful, no collisions: " + ("PASS" if ok else "FAIL"))
    return ok, res


def plot(rows, res):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(18, 6))

    a = ax[0]
    for kind, mk, label in (("pnp", "x", "PnP yaw"), ("wall", "o", "wall-line yaw (used)")):
        r = np.array([row for row in rows[kind] if not np.isnan(row[2])], dtype=float)
        a.plot(r[:, 1] + (0.8 if kind == "wall" else -0.8), np.degrees(r[:, 3]), mk, ms=7, label=label)
    a.axhspan(-2 * np.degrees(cfg.INIT_SIGMA_YAW), 2 * np.degrees(cfg.INIT_SIGMA_YAW), color="0.9",
              label="PnP prior +/- 2 sigma")
    a.set_xlabel("view angle off the marker's normal [deg] (all distances)"); a.set_ylabel("initial heading error [deg]")
    a.legend(fontsize=8); a.set_title("Part 1: heading from one marker (PnP planar ambiguity vs lidar wall)")

    a = ax[1]
    _in = cfg.ROOM_HALF - cfg.WALL_THICK
    a.add_patch(plt.Rectangle((-_in, -_in), 2 * _in, 2 * _in, fill=False, ec="0.3", lw=2))
    for _, kind, c, size, _ in cfg.OBSTACLES:
        a.add_patch(plt.Rectangle(np.array(c) - size, *(2 * np.array(size)), fc="0.8", ec="0.4") if kind == "box"
                    else plt.Circle(c, size, fc="0.8", ec="0.4"))
    for mid, T in cfg.MARKERS.items():
        a.plot(*T.t[:2], "s", color="tab:red", ms=7)
        a.annotate(str(mid), T.t[:2], xytext=(4, 4), textcoords="offset points", color="tab:red")
    for m in res:
        if m["controller"] == "lqr":
            a.plot(m["true_xy"][:, 0], m["true_xy"][:, 1], lw=1.2, color="tab:green" if m["success"] else "tab:red")
            a.plot(*m["start"][:2], "ko", ms=3)
    a.set_aspect("equal"); a.set_title("Part 2: true paths from every start (spline; red = failed)")

    a = ax[2]
    for c, mk in (("lqr", "o"), ("straight", "s")):
        rs = [m for m in res if m["controller"] == c]
        a.plot([1000 * m["lat"] for m in rs], [1000 * m["dist"] for m in rs], mk, alpha=0.7,
               label="spline" if c == "lqr" else "straight segments")
    a.add_patch(plt.Rectangle((-30, -30), 60, 60, fill=False, ls="--", ec="k"))
    a.set_xlabel("lateral docking error [mm]"); a.set_ylabel("distance docking error [mm]"); a.legend()
    a.set_title("Part 2: docking errors (dashed: success box)")
    plt.tight_layout(); plt.show()


def cli():
    parser = argparse.ArgumentParser(description="Explore mission: localisation, planning and docking")
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    parser.add_argument("--parts", nargs="*", type=int, default=[0, 1, 2], help="which parts to run (default all)")
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()
    ok, rows, res = True, None, None
    if 0 in args.parts:
        ok &= part0()
    if 1 in args.parts:
        ok1, rows = part1()
        ok &= ok1
    if 2 in args.parts:
        ok2, res = part2(args.trials, args.workers)
        ok &= ok2
    print(f'\nnavigation test (parts {args.parts}): {"PASS" if ok else "FAIL"}')
    if not args.no_plot and rows is not None and res is not None:
        plot(rows, res)


if __name__ == "__main__":
    cli()
