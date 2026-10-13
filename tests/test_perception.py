# Evaluate the perception module (src/perception) against ground truth in the Track A world ,for marking criterion B.
# The robot is placed at random collision-free poses. At each one the lidar and camera are read through the
# simulation drivers and processed exactly as the robot would. Ground truth from MuJoCo is used only to score them.
#   Part 1 (LiDAR): are obstacles found when they are visible, and do the clusters sit on real surfaces?
#   Part 2 (CAMERA): ArUco detection rate against range and viewing angle, range-bearing errors against the
#            Week 3 noise model sigma_Z = Z^2 sigma_px / (f s), fitted corner bias and scatter, and whether the
#            covariance handed to the EKF is consistent with the errors actually made
# Usage: python tests/test_perception.py [--poses 500] [--seed 0] [--no-plot]
import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from machinevisiontoolbox import CentralCamera
from src import config as cfg
from src.perception import perception as per
from src.simulation import world as sim

_IN = cfg.ROOM_HALF - cfg.WALL_THICK
F, S = per.CAMERA.K[0, 0], cfg.MARKER_SIDE # the calibrated focal length the module uses

# The camera the simulator actually renders- ground truth used only to decide which markers are in view
TRUE_CAMERA = CentralCamera(f=cfg.CAMERA_F_PX, imagesize=(cfg.CAMERA_WIDTH, cfg.CAMERA_HEIGHT), pp=cfg.CAMERA_PP,
                            rho=1)

# Ground-truth geometry (evaluation only)

def solid_objects():
    """Obstacles and the charging station as 2D shapes in the world frame."""
    objs = []
    for name, kind, c, size, _ in cfg.OBSTACLES:
        objs.append(dict(name=name, kind=kind, c=np.array(c), size=np.array(size)))
    objs.append(dict(name="dock", kind="box", c=np.array(cfg.DOCK_POS), size=np.array(cfg.DOCK_SIZE[:2])))
    return objs


def surface_distance(P, o):
    """Distance from points P (N x 2) to the boundary of object o. Negative inside."""
    if o["kind"] == "cylinder":
        return np.linalg.norm(P - o["c"], axis=1) - o["size"]
    q = np.abs(P - o["c"]) - o["size"]
    return np.linalg.norm(np.maximum(q, 0), axis=1) + np.minimum(q.max(axis=1), 0)


def wall_distance(P):
    return _IN - np.abs(P).max(axis=1)


def sample_free_poses(n, rng, margin=cfg.ROBOT_RADIUS + 0.05):
    """Random robot poses at least `margin` from every wall and object. Three quarters of them face a random
    marker (+/-30deg) so the camera part has enough data."""
    objs, ids, poses = solid_objects(), list(cfg.MARKERS), []
    while len(poses) < n:
        p = rng.uniform(-_IN + margin, _IN - margin, 2)
        if min(surface_distance(p[None], o)[0] for o in objs) < margin:
            continue
        if rng.random() < 0.75:
            m = cfg.MARKERS[ids[rng.integers(len(ids))]].t
            th = np.arctan2(m[1] - p[1], m[0] - p[0]) + rng.uniform(-0.5, 0.5)
        else:
            th = rng.uniform(-np.pi, np.pi)
        poses.append((p[0], p[1], per.wrap(th)))
    return poses


def geom_names(model):
    return [model.geom(i).name or model.body(model.geom_bodyid[i]).name for i in range(model.ngeom)]


# Part 1: LiDAR

def lidar_trial(model, data, names, x_true, rng, min_points=4):
    """Process one scan and score it. Returns per-object (visible, detected) and the cluster outcomes.
    An object counts as visible when at least min_points (the DBSCAN minimum) noise-free beams would hit it."""
    res = per.process_scan(sim.read_lidar(model, data, rng), rng, min_points=min_points)

    # Truth - which object each beam would hit, from the true sensor pose
    T_WS = per.pose_to_SE3(x_true) * cfg.T_RS
    dirs = (T_WS.R @ per.LIDAR_DIRS_S.T).T
    gid, dist = np.zeros(len(dirs), np.int32), np.zeros(len(dirs))
    mujoco.mj_multiRay(model, data, T_WS.t, dirs.flatten(), None, 1, model.body("base").id,
                       gid, dist, None, len(dirs), cfg.LIDAR_RANGE_MAX)
    hit = [(names[g], r) for g, r in zip(gid, dist) if g >= 0 and 0 < r < cfg.LIDAR_RANGE_MAX]
    objs = solid_objects()
    beams = {o["name"]: [r for n, r in hit if n == o["name"]] for o in objs}
    visible = {n for n, rs in beams.items() if len(rs) >= min_points}

    # Score each cluster point by the surface it lies on. An object is detected if some cluster holds at least
    # min_points of its returns (a cluster may also hold the wall behind it).
    # A cluster is false if most of its points lie on nothing.
    detected, outcomes, surf = set(), [], []
    for ob in res.obstacles:
        P_W = per.to_world(ob.points, x_true)               # true pose: scoring only
        d = np.abs(np.array([surface_distance(P_W, o) for o in objs]))
        on_obj = d.min(axis=0) < 0.05
        on_wall = np.abs(wall_distance(P_W)) < 0.05
        for k, o in enumerate(objs):
            if (d[k] < 0.05).sum() >= min_points:
                detected.add(o["name"])
        if on_obj.sum() >= on_wall.sum() and on_obj.mean() > 0.5:
            outcomes.append("object"); surf.append(d.min(axis=0)[on_obj].mean())
        elif (on_obj | on_wall).mean() > 0.5:
            outcomes.append("wall")
        else:
            outcomes.append("false")
    return ({o["name"]: (o["name"] in visible, o["name"] in detected, min(beams[o["name"]], default=np.inf))
             for o in objs}, outcomes, surf, res)


def report_lidar(trials):
    print("\n=== Part 1: lidar obstacle detection")
    names = list(trials[0][0])
    bins = [(0, 1), (1, 2), (2, 2.5), (2.5, 3), (3, cfg.LIDAR_RANGE_MAX)]
    print("detection rate of visible obstacles (count) by distance to their nearest point, "
          "required >= 95% within 2.5 m:")
    print(f'{"object":>12} ' + " ".join(f"{f'{lo}-{hi} m':>12}" for lo, hi in bins))
    for n in names:
        cells = []
        for lo, hi in bins:
            b = [t[0][n][1] for t in trials if t[0][n][0] and lo <= t[0][n][2] < hi]
            cells.append(f"{100 * np.mean(b):3.0f}% ({len(b):3d})" if b else "-")
        print(f"{n:>12} " + " ".join(f"{c:>12}" for c in cells))
    near = [t[0][n][1] for t in trials for n in names if t[0][n][0] and t[0][n][2] < 2.5]
    ok = np.mean(near) >= 0.95
    print(f"{'all':>12} {100 * np.mean(near):.1f}% within 2.5 m")
    outcomes = [o for t in trials for o in t[1]]
    surf = 1000 * np.array([s for t in trials for s in t[2]])
    n = len(outcomes)
    print(f"{n} clusters: {outcomes.count('object')} on objects, {outcomes.count('wall')} on wall sections "
          f"RANSAC left behind, {outcomes.count('false')} on nothing")
    print(f"cluster points lie {surf.mean():.1f} mm from the true surface on average")
    ok &= outcomes.count("false") == 0
    print("PASS" if ok else "FAIL")
    return ok


# Part 2: CAMERA

def marker_expected(model, data, x_true, marker_id):
    """Is the whole marker inside the image and without obstacle in hte way. Returns (expected, range, viewing angle)."""
    T_WC = per.pose_to_SE3(x_true) * cfg.T_RC
    T_WM = cfg.MARKERS[marker_id]
    s = cfg.MARKER_SIDE / 2
    corners = T_WM * np.array([[-s, s, 0], [s, s, 0], [s, -s, 0], [-s, -s, 0]]).T
    p_C = T_WC.inv() * corners
    to_cam = T_WC.t - T_WM.t
    rng_ = np.linalg.norm(to_cam[:2])
    angle = np.degrees(np.arccos(np.clip(T_WM.R[:, 2] @ to_cam / np.linalg.norm(to_cam), -1, 1)))
    if np.any(p_C[2] <= 0) or angle >= 90:
        return False, rng_, angle
    uv = TRUE_CAMERA.project_point(corners, pose=T_WC)
    if np.any(uv < 0) or np.any(uv[0] > cfg.CAMERA_WIDTH - 1) or np.any(uv[1] > cfg.CAMERA_HEIGHT - 1):
        return False, rng_, angle
    gid = np.zeros(1, np.int32)
    d = mujoco.mj_ray(model, data, T_WC.t, T_WM.t - T_WC.t, None, 1, model.body("base").id, gid)
    occluded = d < 1 - 0.01 and model.geom_bodyid[gid[0]] != model.body(f"aruco_{marker_id}").id
    return not occluded, rng_, angle


def camera_trial(model, data, cam, x_true, rng):
    obs = per.detect_markers(cam.render(data, rng), bias_px=0.0) # raw: no bias correction, to fit it
    seen = {o.id: o for o in obs}
    rows, false_ids = [], [o.id for o in obs if o.id not in cfg.MARKERS]
    for mid in cfg.MARKERS:
        exp, rng_, angle = marker_expected(model, data, x_true, mid)
        o = seen.get(mid)
        row = dict(id=mid, expected=exp, detected=o is not None, range=rng_, angle=angle)
        if o is not None:
            m = cfg.MARKERS[mid].t
            dx, dy = m[0] - x_true[0], m[1] - x_true[1]
            row.update(Z=o.T_CM.t[2], e_rho=o.z[0] - np.hypot(dx, dy),
                       e_alpha=per.wrap(o.z[1] - (np.arctan2(dy, dx) - x_true[2])), T_CM=o.T_CM)
            if not exp and abs(row["e_rho"]) > 0.3:          # a partly visible marker is fine, a wrong one is not
                false_ids.append(mid)
        rows.append(row)
    return rows, false_ids


def report_camera(rows, false_ids):
    print("\n=== Part 2: camera, ArUco range and bearing")
    ok = True

    exp = [r for r in rows if r["expected"]]
    print("detection rate of fully visible, unoccluded markers (count), required >= 95% within 2m and 60deg:")
    print(f'{"range [m]":>10} {"<30deg":>10} {"30-60deg":>10} {">60deg":>10}')
    rate_ok = []
    for lo, hi in [(0, 1), (1, 2), (2, 3), (3, 4), (4, 6)]:
        cells = []
        for alo, ahi in [(0, 30), (30, 60), (60, 90)]:
            b = [r for r in exp if lo <= r["range"] < hi and alo <= r["angle"] < ahi]
            rate = np.mean([r["detected"] for r in b]) if b else np.nan
            cells.append(f"{100 * rate:3.0f}% ({len(b):3d})" if b else f'{"-":>10}')
            if hi <= 2 and ahi <= 60 and b:
                rate_ok.append(rate)
        print(f"{lo:>4}-{hi:<5} " + " ".join(f"{c:>10}" for c in cells))
    ok &= min(rate_ok) >= 0.95
    print(f"false or wrong-id detections: {len(false_ids)}")
    ok &= not false_ids

    det = [r for r in rows if r["detected"] and r["expected"]]
    Z = np.array([r["Z"] for r in det])
    e_rho = np.array([r["e_rho"] for r in det])
    e_alpha = np.array([r["e_alpha"] for r in det])

    # Wk3: a constant corner error delta [px] gives a range error Z^2 delta / (f s). Fit delta (bias) by least
    # squares, then the scatter about it in the same units. Bearing scatter is sigma_px / f.
    g = Z ** 2 / (F * S)
    bias_px = (g @ e_rho) / (g @ g)                         # corners inside the outline make the range too long
    sigma_range_px = np.sqrt(np.mean(((e_rho - g * bias_px) / g) ** 2))
    sigma_bearing_px = e_alpha.std() * F
    print(f"\n{len(det)} detections used for the noise model")
    print(f"range error before correction: mean {1000 * e_rho.mean():+.1f} mm, RMS {1000 * np.sqrt(np.mean(e_rho**2)):.1f} mm")
    print(f"fitted corner bias      BIAS_PX          = {bias_px:.3f} px  (range error = Z^2 x bias / (f s))")
    print(f"fitted range scatter    SIGMA_RANGE_PX   = {sigma_range_px:.3f} px")
    print(f"fitted bearing scatter  SIGMA_BEARING_PX = {sigma_bearing_px:.3f} px")

    # Apply the correction the module would apply and check the covariance it reports
    e_rho_c = np.array([r["e_rho"] + (per.range_bearing(cfg.T_RC * per.correct_range(r["T_CM"], F, S, bias_px))[0]
                                      - per.range_bearing(cfg.T_RC * r["T_CM"])[0]) for r in det])
    print(f"range error after correction:  mean {1000 * e_rho_c.mean():+.1f} mm, RMS {1000 * np.sqrt(np.mean(e_rho_c**2)):.1f} mm")
    print(f"bearing error: mean {np.degrees(e_alpha.mean()):+.3f} deg, std {np.degrees(e_alpha.std()):.3f} deg")

    def coverage(b_px, s_r, s_b):
        """Fraction of errors inside the 2-sigma bounds the module reports, after its bias correction."""
        e_r = np.array([r["e_rho"] + (per.range_bearing(cfg.T_RC * per.correct_range(r["T_CM"], F, S, b_px))[0]
                                      - per.range_bearing(cfg.T_RC * r["T_CM"])[0]) for r in det])
        sig = np.sqrt(np.array([np.diag(per.measurement_covariance(z, F, S, s_r, s_b)) for z in Z]))
        return np.mean(np.abs(e_r) < 2 * sig[:, 0]), np.mean(np.abs(e_alpha) < 2 * sig[:, 1])

    # The module's values must give an unbiased range and an honest covariance (Wk7: an overconfident filter is
    # the dangerous one). A refit from a different set of poses will differ a little so it is shown for reference.
    print("errors inside the reported 2-sigma bounds (expect ~95%):")
    for label, params in [("module values", (per.BIAS_PX, per.SIGMA_RANGE_PX, per.SIGMA_BEARING_PX)),
                          ("this run's fit", (bias_px, sigma_range_px, sigma_bearing_px))]:
        c_r, c_b = coverage(*params)
        print(f"   {label:14s}: range {100 * c_r:.0f}%, bearing {100 * c_b:.0f}%")
        if label == "module values":
            e_mod = np.array([r["e_rho"] + (per.range_bearing(cfg.T_RC * per.correct_range(r["T_CM"], F, S, per.BIAS_PX))[0]
                                            - per.range_bearing(cfg.T_RC * r["T_CM"])[0]) for r in det])
            print(f"   module values : range error after correction mean {1000 * e_mod.mean():+.1f} mm")
            ok &= abs(e_mod.mean()) < 0.01 and c_r >= 0.9 and c_b >= 0.9
    print("PASS" if ok else "FAIL")
    return ok, dict(rng=np.array([r["range"] for r in det]), e_rho=e_rho, e_rho_c=e_mod, e_alpha=e_alpha, exp=exp)


def plot(lidar_trials, cam):
    fig, ax = plt.subplots(2, 2, figsize=(12, 10))

    a = ax[0, 0]
    for o in solid_objects():
        if o["kind"] == "box":
            a.add_patch(plt.Rectangle(o["c"] - o["size"], *(2 * o["size"]), fc="0.85", ec="0.4"))
        else:
            a.add_patch(plt.Circle(o["c"], o["size"], fc="0.85", ec="0.4"))
    a.add_patch(plt.Rectangle((-_IN, -_IN), 2 * _IN, 2 * _IN, fill=False, ec="0.4"))
    for _, _, _, res, x in lidar_trials[:40]:
        for ob in res.obstacles:
            P = per.to_world(ob.points, x)
            a.plot(P[:, 0], P[:, 1], ".", ms=2, color="tab:red")
    a.set_aspect("equal"); a.set_title("Part 1: obstacle clusters from 40 scans (world frame)")

    a = ax[0, 1]
    zs = np.linspace(0, cam["rng"].max(), 100)
    a.plot(cam["rng"], 1000 * cam["e_rho"], ".", ms=3, color="0.6", label="raw")
    a.plot(cam["rng"], 1000 * cam["e_rho_c"], ".", ms=3, color="tab:blue", label="bias corrected (module)")
    a.plot(zs, 1000 * zs ** 2 * per.BIAS_PX / (F * S), "k--", lw=1, label="bias model, BIAS_PX")
    env = 2000 * zs ** 2 * per.SIGMA_RANGE_PX / (F * S)
    a.fill_between(zs, -env, env, color="tab:blue", alpha=0.15, label="reported 2 sigma, SIGMA_RANGE_PX")
    a.set_xlabel("range [m]"); a.set_ylabel("range error [mm]"); a.legend(); a.set_title("Part 2: range error")

    a = ax[1, 0]
    a.hist(np.degrees(cam["e_alpha"]), bins=40, color="0.6")
    a.set_xlabel("bearing error [deg]"); a.set_title("Part 2: bearing error")

    a = ax[1, 1]
    edges = np.arange(0, 5.5, 0.5)
    for (alo, ahi), c in zip([(0, 30), (30, 60), (60, 90)], ["tab:green", "tab:orange", "tab:red"]):
        b = [r for r in cam["exp"] if alo <= r["angle"] < ahi]
        rates = [np.mean([r["detected"] for r in b if lo <= r["range"] < lo + 0.5] or [np.nan]) for lo in edges[:-1]]
        a.plot(edges[:-1] + 0.25, 100 * np.array(rates), "o-", color=c, label=f"{alo}-{ahi} deg")
    a.set_xlabel("range [m]"); a.set_ylabel("detected [%]"); a.set_ylim(-5, 105); a.legend(title="viewing angle")
    a.set_title("Part 2: detection rate of visible markers")
    plt.tight_layout(); plt.show()


def main():
    parser = argparse.ArgumentParser(description="Perception evaluation against ground truth")
    parser.add_argument("--poses", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    model = sim.build_world()
    data = mujoco.MjData(model)
    cam = sim.Camera(model)
    names = geom_names(model)

    lidar_trials, rows, false_ids = [], [], []
    for x, y, th in sample_free_poses(args.poses, rng):
        sim.set_pose(model, data, x, y, th)
        x_true = sim.true_pose(model, data)
        vis, outcomes, surf, res = lidar_trial(model, data, names, x_true, rng)
        lidar_trials.append((vis, outcomes, surf, res, x_true))
        r, f = camera_trial(model, data, cam, x_true, rng)
        rows += r; false_ids += f
    cam.close()

    ok1 = report_lidar(lidar_trials)
    ok2, cam_res = report_camera(rows, false_ids)
    print(f'\nperception test: {"PASS" if ok1 and ok2 else "FAIL"}')
    if not args.no_plot:
        plot(lidar_trials, cam_res)


if __name__ == "__main__":
    main()
