# Use this script to test the TurtleBot3 Burger 2D lidar in MuJoCo against ground truth.
# It follows the methods within the subject tutorials using the robot + sensors built in test_burger_sensors.py:
#   Part 1:  Range-bearing measurement model, SE3 sensor/robot frames, constant range noise (Week 4, Part 5),
#            checked against an analytic ray cast of the arena geometry (Week 4, Part 2 scene simulator). 
#   Part 2:  Voxel downsampling + statistical outlier removal (Week 4, Part 3), RANSAC wall fitting and
#            DBSCAN obstacle clustering (Week 4, Part 4), compared with the true walls and obstacles.
#   Part 3:  The robot drives a unicycle circle (Week 7, C.4), and sequential point to plane ICP scan matching
#            (Week 4 bonus) is chained into LiDAR odometry and compared with ground truth and with
#            dead reckoning from the wheel encoders through the Week 7 motion model
# Requirements: everything for test_burger_sensors.py, plus matplotlib, open3d and spatialmath-python.
# Usage: python setup_tests/test_burger_lidar.py [--sigma 0.01] [--no-plot]
import argparse

import matplotlib.pyplot as plt
import mujoco
import numpy as np
import open3d as o3d
from spatialmath import SE3

from test_burger_sensors import LIDAR_ANGLES, LIDAR_POS, LIDAR_RANGE_MAX, build_model, read_lidar

np.seterr(divide='ignore', invalid='ignore')

# Sensor and robot
T_RS = SE3(*LIDAR_POS)                  # lidar pose in the robot frame {R}: x forward, y left, z up
WHEEL_RADIUS, WHEEL_TRACK = 0.033, 0.16
SCAN_PERIOD = 0.2                       # s, the LDS-01 spins at 5 Hz

# Unit beam directions in the sensor frame (Week 4 Part 5 with elevation = 0)
DIRS_S = np.stack([np.cos(LIDAR_ANGLES), np.sin(LIDAR_ANGLES), np.zeros_like(LIDAR_ANGLES)], axis=1)

# ---------------------------------------------------------------------------
# Ground truth: the arena geometry read from the MuJoCo model, and Week 4
# ray caster (_box, _sphere) reduced to the 2D scan plane.
# ---------------------------------------------------------------------------

def arena_from_model(model, data):
    """Walls and obstacles as 2D shapes in the world frame, from the compiled model."""
    scene = []
    for i in range(model.ngeom):
        name = model.geom(i).name
        if not name.startswith(('wall_', 'box_', 'cyl_')):
            continue
        c, size = data.geom_xpos[i][:2].copy(), model.geom_size[i]
        if model.geom_type[i] == mujoco.mjtGeom.mjGEOM_BOX:
            scene.append(dict(kind='box', lo=c - size[:2], hi=c + size[:2], c=c, name=name))
        else:
            scene.append(dict(kind='circle', c=c, r=size[0], name=name))
    return scene


def _box(o, d, lo, hi):
    t1, t2 = (np.asarray(lo) - o) / d, (np.asarray(hi) - o) / d
    tmin = np.nanmax(np.minimum(t1, t2), axis=1)
    tmax = np.nanmin(np.maximum(t1, t2), axis=1)
    hit = (tmax > 0) & (tmin <= tmax)
    return np.where(hit, np.where(tmin > 0, tmin, tmax), np.inf)


def _circle(o, d, c, r):
    oc = o - np.asarray(c)
    a, b = np.sum(d * d, axis=1), 2 * np.sum(oc * d, axis=1)
    disc = b ** 2 - 4 * a * (np.sum(oc * oc, axis=1) - r ** 2)
    sq = np.sqrt(np.where(disc >= 0, disc, 0))
    t = np.where((-b - sq) > 0, (-b - sq), (-b + sq)) / (2 * a)
    return np.where((disc >= 0) & (t > 0), t, np.inf)


def raycast(origins, dirs, scene):
    """Return (distance along ray, index of surface hit) for every ray."""
    best = np.full(dirs.shape[0], np.inf)
    idx = np.full(dirs.shape[0], -1)
    for k, s in enumerate(scene):
        t = _box(origins, dirs, s['lo'], s['hi']) if s['kind'] == 'box' else _circle(origins, dirs, s['c'], s['r'])
        m = t < best
        best, idx = np.where(m, t, best), np.where(m, k, idx)
    return best, idx


def surface_distance(P, s):
    """Unsigned 2D distance from points P (N x 2) to the boundary of shape s."""
    if s['kind'] == 'circle':
        return np.abs(np.linalg.norm(P - s['c'], axis=1) - s['r'])
    q = np.abs(P - s['c']) - (s['hi'] - s['lo']) / 2
    return np.abs(np.linalg.norm(np.maximum(q, 0), axis=1) + np.minimum(q.max(axis=1), 0))


# ---------------------------------------------------------------------------
# Robot: ground-truth pose from MuJoCo, unicycle commands (Week 5/7)
# ---------------------------------------------------------------------------

def wrap(a):
    """Fold an angle back into (-pi, pi]."""
    return (a + np.pi) % (2 * np.pi) - np.pi


def T_WR(model, data):
    """True robot pose in the world frame {W}."""
    b = model.body('base').id
    return SE3.Rt(data.xmat[b].reshape(3, 3), data.xpos[b], check=False)


def drive(data, v, w):
    """Unicycle command (v, omega) -> left/right wheel speeds of the differential drive."""
    data.ctrl[0] = (v - w * WHEEL_TRACK / 2) / WHEEL_RADIUS
    data.ctrl[1] = (v + w * WHEEL_TRACK / 2) / WHEEL_RADIUS


def f_uni(xh, u, dt):
    """Nonlinear motion model f(x, u), u = (v, omega) (Week 7, C.4)."""
    v, w = u
    x, y, th = xh
    if abs(w) < 1e-6:
        return np.array([x + v * dt * np.cos(th), y + v * dt * np.sin(th), wrap(th)])
    th_n = th + w * dt
    return np.array([x + v / w * (np.sin(th_n) - np.sin(th)),
                     y + v / w * (-np.cos(th_n) + np.cos(th)),
                     wrap(th_n)])


def pose2d(T):
    return np.array([T.t[0], T.t[1], T.rpy()[2]])


def step_for(model, data, seconds):
    for _ in range(int(round(seconds / model.opt.timestep))):
        mujoco.mj_step(model, data)


# ---------------------------------------------------------------------------
# Lidar: range-bearing measurement z = [rho, alpha], p^S = rho [cos a, sin a, 0]
# ---------------------------------------------------------------------------

def scan(model, data, rng, sigma):
    """Ranges from the MuJoCo rangefinders with constant range noise (Week 4, Part 5)."""
    rho = read_lidar(model, data) + rng.normal(0, sigma, LIDAR_ANGLES.shape)
    ok = np.isfinite(rho) & (rho < LIDAR_RANGE_MAX)
    return rho, ok


def to_points(rho, ok):
    """Polar to Cartesian in the sensor frame, N x 3."""
    return DIRS_S[ok] * rho[ok, None]


def ransac_line(P, thresh, iters, rng):
    """RANSAC with a line model: sample 2 points, count inliers, keep the best, refit on its inliers."""
    best = np.zeros(len(P), bool)
    for _ in range(iters):
        p, q = P[rng.choice(len(P), 2, replace=False)]
        n = np.array([-(q - p)[1], (q - p)[0]])
        if np.linalg.norm(n) < 1e-9:
            continue
        n /= np.linalg.norm(n)
        inl = np.abs((P - p) @ n) < thresh
        if inl.sum() > best.sum():
            best = inl
    c = P[best].mean(axis=0)
    n = np.linalg.svd(P[best] - c)[2][1] # normal = direction of least spread
    return np.array([n[0], n[1], -n @ c]), best   # line a x + b y + c = 0, unit normal


# ---------------------------------------------------------------------------

def part1_measurement_model(model, data, scene, rng, sigma):
    print('\n=== Part 1: measurement model and frames vs analytic ray cast')
    T_WS = T_WR(model, data) * T_RS
    rho, ok = scan(model, data, rng, sigma)

    # Truth: cast the same beams through the known geometry, from the true sensor pose
    dirs_W = (T_WS.R @ DIRS_S.T).T[:, :2]
    rho_true, hit = raycast(np.tile(T_WS.t[:2], (len(dirs_W), 1)), dirs_W, scene)
    both = ok & (rho_true < LIDAR_RANGE_MAX)
    resid = rho[both] - rho_true[both]
    gross = np.abs(resid) > 5 * sigma

    print(f'{ok.sum()} returns from {len(rho)} beams, {both.sum()} compared with truth')
    print(f'range residual: mean {1000 * resid.mean():+.2f} mm, std {1000 * resid.std():.2f} mm '
          f'(sigma = {1000 * sigma:.1f} mm), {gross.sum()} beams beyond 5 sigma')
    passed = abs(resid.mean()) < 0.3 * sigma and 0.7 < resid.std() / sigma < 1.3 and gross.mean() < 0.02
    print('PASS' if passed else 'FAIL')

    P_W = (T_WS * to_points(rho, ok).T).T
    beams_per_shape = {s['name']: int((hit[both] == k).sum()) for k, s in enumerate(scene)}
    return passed, dict(P_W=P_W, T_WS=T_WS, resid=resid, beams=beams_per_shape)


def part2_walls_and_objects(P_W, beams, scene, rng, sigma, min_points=5):
    print('\n=== Part 2: clean the cloud, RANSAC walls, DBSCAN obstacles')
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P_W))
    down = pcd.voxel_down_sample(voxel_size=0.02)
    # Parameters follow the sensor (Week 4 5.2): at 1deg resolution a 16 cm cylinder 1.2m away returns
    # only 8 points, so the neighbourhood used to judge outliers has to be smaller than that.
    clean, _ = down.remove_statistical_outlier(nb_neighbors=min_points, std_ratio=2.0)
    print(f'{len(pcd.points)} points, {len(down.points)} after voxel downsample, '
          f'{len(clean.points)} after outlier removal')

    # Sequential RANSAC: peel off the dominant line until what is left is too small to be a wall
    P = np.asarray(clean.points)[:, :2]
    walls_true = [s for s in scene if s['name'].startswith('wall_')]
    lines, rest, passed = [], P, True
    while len(rest) > 40:
        line, inl = ransac_line(rest, thresh=2 * sigma, iters=500, rng=rng)
        if inl.sum() < 40:
            break
        lines.append((line, rest[inl]))
        rest = rest[~inl]

    print(f'{"wall":>8} {"inliers":>8} {"offset err [mm]":>16} {"angle err [deg]":>16}')
    found = set()
    for line, pts in lines:
        s = min(walls_true, key=lambda w: surface_distance(pts, w).mean())
        off = 1000 * surface_distance(pts.mean(axis=0, keepdims=True), s)[0]
        axis = np.argmin(s['hi'] - s['lo'])              # the wall's thin axis is its normal
        ang = np.rad2deg(np.arccos(min(1.0, abs(line[axis]))))
        found.add(s['name'])
        passed &= off < 20 and ang < 1.0
        print(f'{s["name"]:>8} {len(pts):>8} {off:>16.1f} {ang:>16.2f}')
    passed &= len(found) == len(walls_true)

    # Everything that is not a wall gets clustered
    objects = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.column_stack([rest, np.zeros(len(rest))])))
    labels = np.array(objects.cluster_dbscan(eps=0.06, min_points=min_points))
    obstacles = [s for s in scene if not s['name'].startswith('wall_')]
    visible = {o['name'] for o in obstacles if beams[o['name']] >= min_points}
    print(f'\n{labels.max() + 1} obstacles found, {len(visible)} visible in the true scan')
    print(f'{"obstacle":>10} {"points":>7} {"centroid [m]":>16} {"true centre [m]":>16} {"surface dist [mm]":>18}')
    clusters = []
    for i in range(labels.max() + 1):
        p = rest[labels == i]
        s = min(obstacles, key=lambda o: surface_distance(p, o).mean())
        d = 1000 * surface_distance(p, s).mean()
        passed &= d < 2000 * sigma
        clusters.append((p, s))
        print(f'{s["name"]:>10} {len(p):>7} {str(np.round(p.mean(axis=0), 2)):>16} '
              f'{str(np.round(s["c"], 2)):>16} {d:>18.1f}')
    missed = visible - {s['name'] for _, s in clusters}
    if missed:
        print(f'missed: {", ".join(sorted(missed))}')
    passed &= not missed
    print('PASS' if passed else 'FAIL')
    return passed, dict(lines=lines, clusters=clusters, rest=rest)


def scan_cloud(P_S, height=0.1):
    """A 2D scan as a cloud for point to plane ICP. The beams hit vertical surfaces, so the scan is extruded
    upwards: without it every neighbourhood is a line and the estimated normals are undefined."""
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.vstack([P_S, P_S + [0, 0, height]])))
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.2, max_nn=30))
    return pcd


def wheel_angles(model, data):
    return data.qpos[[model.joint('wheel_left').qposadr[0], model.joint('wheel_right').qposadr[0]]].copy()


def part3_icp_odometry(model, data, rng, sigma, v=0.15, w=0.3):
    print(f'\n=== Part 3: lidar odometry by sequential ICP, unicycle circle v = {v} m/s, omega = {w} rad/s')
    n_scans = int(round(2 * np.pi / w / SCAN_PERIOD))
    T_icp = T_WR(model, data) * T_RS # start the odometry at the true sensor pose
    x_dr = pose2d(T_WR(model, data)) # dead reckoning from the wheel encoders
    phi = wheel_angles(model, data)

    rho, ok = scan(model, data, rng, sigma)
    prev = scan_cloud(to_points(rho, ok))
    T_prev_true = T_icp
    TRUE, ICP, DR, step_err = [pose2d(T_WR(model, data))], [pose2d(T_icp * T_RS.inv())], [x_dr.copy()], []

    drive(data, v, w)
    for _ in range(n_scans):
        step_for(model, data, SCAN_PERIOD)
        T_true = T_WR(model, data) * T_RS

        # encoders -> measured (v, omega) -> Week 7 motion model
        ds_l, ds_r = WHEEL_RADIUS * (wheel_angles(model, data) - phi)
        phi = wheel_angles(model, data)
        u_enc = ((ds_r + ds_l) / 2 / SCAN_PERIOD, (ds_r - ds_l) / WHEEL_TRACK / SCAN_PERIOD)
        x_dr = f_uni(x_dr, u_enc, SCAN_PERIOD)

        rho, ok = scan(model, data, rng, sigma)
        cur = scan_cloud(to_points(rho, ok))
        result = o3d.pipelines.registration.registration_icp(
            cur, prev, 0.3, np.eye(4), o3d.pipelines.registration.TransformationEstimationPointToPlane())

        T_rel = SE3(result.transformation, check=False)
        T_rel_true = T_prev_true.inv() * T_true
        step_err.append([1000 * np.linalg.norm(T_rel.t - T_rel_true.t),
                         abs(np.rad2deg(wrap(T_rel.rpy()[2] - T_rel_true.rpy()[2])))])

        T_icp = SE3((T_icp * T_rel).A, check=False)
        TRUE.append(pose2d(T_true * T_RS.inv())); ICP.append(pose2d(T_icp * T_RS.inv())); DR.append(x_dr.copy())
        prev, T_prev_true = cur, T_true
    drive(data, 0, 0)

    TRUE, ICP, DR, step_err = map(np.array, (TRUE, ICP, DR, step_err))
    path = np.sum(np.linalg.norm(np.diff(TRUE[:, :2], axis=0), axis=1))
    e_icp = np.linalg.norm(ICP[-1, :2] - TRUE[-1, :2])
    e_dr = np.linalg.norm(DR[-1, :2] - TRUE[-1, :2])
    print(f'{n_scans} scan pairs over {path:.2f} m')
    print(f'per-step ICP error: translation {step_err[:, 0].mean():.1f} mm mean, {step_err[:, 0].max():.1f} mm max; '
          f'yaw {step_err[:, 1].mean():.2f} deg mean, {step_err[:, 1].max():.2f} deg max')
    print(f'final position error: ICP odometry {1000 * e_icp:.0f} mm ({100 * e_icp / path:.1f}% of path), '
          f'dead reckoning {1000 * e_dr:.0f} mm')
    print(f'final heading error:  ICP odometry {np.rad2deg(abs(wrap(ICP[-1, 2] - TRUE[-1, 2]))):.2f} deg, '
          f'dead reckoning {np.rad2deg(abs(wrap(DR[-1, 2] - TRUE[-1, 2]))):.2f} deg')
    passed = step_err[:, 0].mean() < 5 and step_err[:, 1].mean() < 0.5 and e_icp < 0.03 * path
    print('PASS' if passed else 'FAIL')
    return passed, dict(true=TRUE, icp=ICP, dr=DR)


def draw_scene(ax, scene):
    for s in scene:
        if s['kind'] == 'box':
            (x0, y0), (x1, y1) = s['lo'], s['hi']
            ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fc='0.85', ec='0.4', lw=0.8))
        else:
            ax.add_patch(plt.Circle(s['c'], s['r'], fc='0.85', ec='0.4', lw=0.8))
    ax.set_aspect('equal'); ax.set_xlabel('x [m]'); ax.set_ylabel('y [m]')


def plot(scene, r1, r2, r3, sigma):
    fig, ax = plt.subplots(2, 2, figsize=(12, 11))

    a = ax[0, 0]
    draw_scene(a, scene)
    a.plot(r1['P_W'][:, 0], r1['P_W'][:, 1], '.', ms=3, color='tab:orange', label='lidar returns')
    t, R = r1['T_WS'].t, r1['T_WS'].R
    a.arrow(t[0], t[1], 0.25 * R[0, 0], 0.25 * R[1, 0], width=0.02, color='tab:green', label='sensor pose')
    a.set_title('Part 1: scan in the world frame over the true geometry'); a.legend(loc='upper right')

    a = ax[0, 1]
    a.hist(1000 * r1['resid'], bins=40, density=True, color='0.7', label='range residual')
    xs = np.linspace(-4 * sigma, 4 * sigma, 200)
    a.plot(1000 * xs, np.exp(-0.5 * (xs / sigma) ** 2) / (sigma * 1000 * np.sqrt(2 * np.pi)), 'b', label='N(0, sigma)')
    a.set_xlabel('measured - true range [mm]'); a.set_title('Part 1: range residuals'); a.legend()

    a = ax[1, 0]
    draw_scene(a, scene)
    cmap = plt.get_cmap('tab10')
    for i, (_, pts) in enumerate(r2['lines']):
        a.plot(pts[:, 0], pts[:, 1], '.', ms=3, color=cmap(i % 10))
    for p, s in r2['clusters']:
        a.plot(p[:, 0], p[:, 1], 'kx', ms=4)
        a.annotate(s['name'], p.mean(axis=0), textcoords='offset points', xytext=(5, 5), fontsize=8)
    a.set_title('Part 2: RANSAC walls (colours), DBSCAN obstacles (x)')

    a = ax[1, 1]
    draw_scene(a, scene)
    a.plot(r3['true'][:, 0], r3['true'][:, 1], 'k', lw=2.5, label='true path')
    a.plot(r3['icp'][:, 0], r3['icp'][:, 1], 'r', lw=1.5, label='ICP lidar odometry')
    a.plot(r3['dr'][:, 0], r3['dr'][:, 1], 'b--', lw=1.5, label='dead reckoning (encoders)')
    a.set_title('Part 3: lidar odometry vs ground truth'); a.legend(loc='upper right')

    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='TurtleBot3 Burger 2D lidar test against ground truth')
    parser.add_argument('--sigma', type=float, default=0.01, help='range noise std dev [m] (Week 4 uses 1 cm)')
    parser.add_argument('--no-plot', action='store_true', help='print results only')
    args = parser.parse_args()

    rng = np.random.default_rng(0)
    model = build_model()
    data = mujoco.MjData(model)
    step_for(model, data, 0.5) # let the robot settle on its wheels
    scene = arena_from_model(model, data)

    ok1, r1 = part1_measurement_model(model, data, scene, rng, args.sigma)
    ok2, r2 = part2_walls_and_objects(r1['P_W'], r1['beams'], scene, rng, args.sigma)
    ok3, r3 = part3_icp_odometry(model, data, rng, args.sigma)

    print(f'\nlidar test: {"PASS" if ok1 and ok2 and ok3 else "FAIL"}')
    if not args.no_plot:
        plot(scene, r1, r2, r3, args.sigma)


if __name__ == '__main__':
    main()
