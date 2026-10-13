# Calibrate the robot camera from simulated checkerboard views and check the result against ground truth
# (marking criterion B - calibration, noise handling, frame management).
# The robot stands still while a checkerboard is held in front of it at random distances and tilts, like a person
# holding the board in front of the real robot. Images come from the simulated camera with its pixel noise.
#   Part 1 - calibration (Week 3 B) with the full distortion model, as CentralCamera.images2C does, and with a
#            pinhole model. Per-image reprojection errors (Week 3 C). Intrinsics are compared with the true values
#   Part 2 - repeatability: independent calibrations from 10, 20 and 40 views to see how accurate and stable each
#            model's parameters are, and which model the data supports
#   Part 3 - camera mount (extrinsics): the board poses from calibration give the camera pose in the robot frame,
#            compared with T_RC in src/config.py (frame management)
# To run: python tests/test_calibration.py [--views 30] [--trials 5] [--seed 0] [--no-plot]
import argparse
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import mujoco
import numpy as np
from machinevisiontoolbox import CentralCamera
from spatialmath import SE3

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config as cfg                                  
from src.perception import calibration as calib             
from src.perception.perception import pose_to_SE3              
from src.simulation import world as sim                        

ROBOT_POSE = (-1.4, -1.4, 0.0) # robot stands here facing +x during calibration
W, H = cfg.CAMERA_WIDTH, cfg.CAMERA_HEIGHT

# NOTE ground truth only used for tests. Not in any real sim/hardware integration
# Ground truth-  the camera the simulator renders. Used only to place the board in view and to score the results.
TRUE_CAMERA = CentralCamera(f=cfg.CAMERA_F_PX, pp=cfg.CAMERA_PP, imagesize=(W, H), rho=1)
SQ = cfg.CHECKERBOARD_SQUARE
_hx, _hy = [(n + 3) * SQ / 2 for n in cfg.CHECKERBOARD_GRID]
PLATE = np.array([[sx * _hx, sy * _hy, 0] for sx in (-1, 1) for sy in (-1, 1)]).T # board outline in {B}



# Capturing calibration images

def sample_board_poses(n, rng, T_WC):
    """Board poses (world frame) 0.45-1.0 m in front of the camera, tilted up to ~35 deg about each axis, fully in
    view and above the floor."""
    poses = []
    while len(poses) < n:
        z = rng.uniform(0.45, 1.0)
        T_CB = (SE3(rng.uniform(-0.15, 0.15) * z, rng.uniform(-0.1, 0.1) * z, z) * SE3.Rx(np.pi)   # facing camera
                * SE3.Rz(rng.uniform(-0.4, 0.4)) * SE3.Rx(rng.uniform(-0.6, 0.6)) * SE3.Ry(rng.uniform(-0.6, 0.6)))
        T_WB = T_WC * T_CB
        if (T_WB * PLATE)[2].min() < 0.01:
            continue
        uv = TRUE_CAMERA.project_point(T_CB * PLATE)
        if np.any(uv < 5) or np.any(uv[0] > W - 6) or np.any(uv[1] > H - 6):
            continue
        poses.append(T_WB)
    return poses


def capture(model, data, cam, board_poses, rng):
    images = []
    for T_WB in board_poses:
        sim.set_checkerboard(model, data, T_WB)
        images.append(cam.render(data, rng))
    return images


def true_camera_pose(model, data):
    """World pose of the camera frame {C} (z forward, y down) from the simulator. Ground truth."""
    c = model.camera(sim.CAMERA_NAME).id
    return SE3.Rt(data.cam_xmat[c].reshape(3, 3), data.cam_xpos[c], check=False) * SE3.Rx(np.pi)


# Scoring

def intrinsic_errors(K):
    return (100 * (K[0, 0] / cfg.CAMERA_F_PX - 1), 100 * (K[1, 1] / cfg.CAMERA_F_PX - 1),
            K[0, 2] - cfg.CAMERA_PP[0], K[1, 2] - cfg.CAMERA_PP[1])


def max_distortion_px(cal):
    """Largest pixel shift the fitted distortion model applies anywhere in the image (0 for the pinhole model)."""
    u, v = np.meshgrid(np.linspace(0, W - 1, 33), np.linspace(0, H - 1, 25))
    xy1 = np.linalg.inv(cal.K) @ np.stack([u.ravel(), v.ravel(), np.ones(u.size)])
    P = xy1.T.astype(np.float64)
    zero = np.zeros(3)
    with_d, _ = cv2.projectPoints(P, zero, zero, cal.K, cal.dist)
    without, _ = cv2.projectPoints(P, zero, zero, cal.K, np.zeros(5))
    return np.linalg.norm(with_d - without, axis=2).max()


def opencv_board_frame(T_CB, corners):
    """OpenCV numbers the corners from whichever end it finds first, so find which corner of the board is the origin
    of its target frame: try the four possibilities and keep the one whose projection matches the measured
    corners. Returns the target frame in the board centre frame {B}."""
    gx, gy = cfg.CHECKERBOARD_GRID
    ax, ay = (gx - 1) * SQ / 2, (gy - 1) * SQ / 2
    best, best_err = None, np.inf
    for sx in (-1, 1):
        for sy in (-1, 1):
            # origin at (sx ax, sy ay), x towards the other end of the row, y towards the other end of the column
            x, y = np.array([-sx, 0, 0.0]), np.array([0, -sy, 0.0])
            T = SE3.Rt(np.column_stack([x, y, np.cross(x, y)]), [sx * ax, sy * ay, 0])
            uv = TRUE_CAMERA.project_point(T_CB * T * calib.board_points().T)
            err = np.linalg.norm(uv.T - corners, axis=1).mean()
            if err < best_err:
                best, best_err = T, err
    return best



def part1(images):
    print("\n=== Part 1: calibration from one set of views (Week 3 B, C)")
    cals = {}
    for name, dist in [("full distortion", True), ("pinhole", False)]:
        c = calib.calibrate(images, distortion=dist)
        errs, _ = calib.reprojection_errors(c)
        per_image = np.array([np.sqrt(np.mean(e ** 2)) for e in errs])
        fu, fv, du, dv = intrinsic_errors(c.K)
        print(f"\n{name}: {len(c.ids)} of {len(images)} images used")
        print(f"   f/rho_u = {c.K[0, 0]:.2f} px ({fu:+.3f}%), f/rho_v = {c.K[1, 1]:.2f} px ({fv:+.3f}%), "
              f"true {cfg.CAMERA_F_PX:.2f}")
        print(f"   u0 = {c.K[0, 2]:.2f} px ({du:+.2f}), v0 = {c.K[1, 2]:.2f} px ({dv:+.2f}), "
              f"true ({cfg.CAMERA_PP[0]}, {cfg.CAMERA_PP[1]})")
        print(f"   distortion (k1, k2, p1, p2, k3) = {np.round(c.dist, 4)}, "
              f"moves pixels by up to {max_distortion_px(c):.2f} px")
        print(f"   reprojection RMS {c.rms:.3f} px; per image: best {per_image.min():.3f}, worst "
              f"{per_image.max():.3f} px (image {c.ids[int(per_image.argmax())]})")
        cals[name] = (c, per_image)
    return cals


def part2(model, data, cam, T_WC, rng, trials, view_counts=(10, 20, 40)):
    print(f"\n=== Part 2: repeatability over {trials} independent calibrations")
    print(f'{"views":>6} {"model":>16} {"f error [%]":>16} {"u0 error [px]":>16} {"v0 error [px]":>16} '
          f'{"max distortion [px]":>20}')
    rows = []
    for n in view_counts:
        res = {True: [], False: []}
        for _ in range(trials):
            images = capture(model, data, cam, sample_board_poses(n, rng, T_WC), rng)
            for dist in (True, False):
                c = calib.calibrate(images, distortion=dist)
                fu, fv, du, dv = intrinsic_errors(c.K)
                res[dist].append([(fu + fv) / 2, du, dv, max_distortion_px(c)])
        for dist, label in [(True, "full distortion"), (False, "pinhole")]:
            a = np.array(res[dist])
            m, s = a.mean(axis=0), a.std(axis=0)
            print(f"{n:>6} {label:>16} " + " ".join(f"{f'{m[k]:+.3f} +/- {s[k]:.3f}':>16}" for k in range(3))
                  + f" {f'{m[3]:.2f} +/- {s[3]:.2f}':>20}")
            rows.append(dict(n=n, distortion=dist, mean=m, std=s))
    return rows


def part3(cal, board_poses, model, data):
    print("\n=== Part 3: camera mount from the calibration poses (frame management)")
    T_WR = pose_to_SE3(ROBOT_POSE)
    T_WC_true = true_camera_pose(model, data)
    t_err, r_err = [], []
    for T_BcvC, i in zip(cal.poses, cal.ids):
        T_WB = board_poses[i]
        T_B_Bcv = opencv_board_frame(T_WC_true.inv() * T_WB, cal.corners[cal.ids.index(i)])
        T_RC_est = T_WR.inv() * T_WB * T_B_Bcv * T_BcvC
        t_err.append(1000 * (T_RC_est.t - cfg.T_RC.t))
        r_err.append(np.degrees(np.linalg.norm((cfg.T_RC.inv() * T_RC_est).rpy())))
    t_err, r_err = np.array(t_err), np.array(r_err)
    m = t_err.mean(axis=0)
    print(f"camera position in the robot frame: config {np.round(cfg.T_RC.t, 4)} m")
    print(f"   from calibration: mean error ({m[0]:+.1f}, {m[1]:+.1f}, {m[2]:+.1f}) mm, "
          f"per-view scatter {np.linalg.norm(t_err - m, axis=1).mean():.1f} mm")
    print(f"camera orientation: mean error {r_err.mean():.2f} deg, worst view {r_err.max():.2f} deg")
    return np.linalg.norm(m) < 5 and r_err.mean() < 0.5


def plot(images, cals, rows):
    fig, ax = plt.subplots(2, 2, figsize=(13, 10))
    c, per_image = cals["pinhole"]

    k = int(np.argmax(per_image))
    img = images[c.ids[k]].copy()
    cv2.drawChessboardCorners(img, cfg.CHECKERBOARD_GRID, c.corners[k].astype(np.float32).reshape(-1, 1, 2), True)
    ax[0, 0].imshow(img); ax[0, 0].set_title(f"worst image for the pinhole model (image {c.ids[k]}), corners found")
    ax[0, 0].axis("off")

    a = ax[0, 1]
    _, residuals = calib.reprojection_errors(c)
    r = residuals[k]
    a.imshow(img, alpha=0.4)
    a.quiver(c.corners[k][:, 0], c.corners[k][:, 1], r[:, 0], r[:, 1], color="r", angles="xy",
             scale_units="xy", scale=0.01)
    a.set_title("reprojection residuals magnified 100x (Week 3 C)"); a.axis("off")

    a = ax[1, 0]
    w = 0.4
    x = np.arange(len(per_image))
    a.bar(x - w / 2, cals["full distortion"][1], w, color="tab:orange", label="full distortion")
    a.bar(x + w / 2, per_image, w, color="tab:blue", label="pinhole")
    a.set_xlabel("calibration image"); a.set_ylabel("reprojection RMS [px]"); a.legend()
    a.set_title("Part 1: per-image reprojection error")

    a = ax[1, 1]
    for dist, colour, label in [(True, "tab:orange", "full distortion"), (False, "tab:blue", "pinhole")]:
        rr = [q for q in rows if q["distortion"] == dist]
        ns = np.array([q["n"] for q in rr]) + (0.4 if dist else -0.4)
        a.errorbar(ns, [q["mean"][0] for q in rr], [q["std"][0] for q in rr], fmt="o-", color=colour,
                   capsize=4, label=f"{label}: focal length")
        a.errorbar(ns, [q["mean"][1] / 10 for q in rr], [q["std"][1] / 10 for q in rr], fmt="s--",
                   color=colour, capsize=4, alpha=0.6, label=f"{label}: u0 error / 10 px")
    a.axhline(0, color="k", lw=0.8)
    a.set_xlabel("views per calibration"); a.set_ylabel("error [%] or [px / 10]")
    a.set_title("Part 2: accuracy and repeatability"); a.legend(fontsize=8)
    plt.tight_layout(); plt.show()


def main():
    parser = argparse.ArgumentParser(description="Camera calibration in simulation checked against ground truth")
    parser.add_argument("--views", type=int, default=30)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    model = sim.build_world(checkerboard=True)
    data = mujoco.MjData(model)
    cam = sim.Camera(model)
    sim.set_pose(model, data, *ROBOT_POSE)
    T_WC = true_camera_pose(model, data)

    board_poses = sample_board_poses(args.views, rng, T_WC)
    images = capture(model, data, cam, board_poses, rng)

    cals = part1(images)
    c = cals["pinhole"][0]
    ok1 = c.rms < 0.2 and len(c.ids) == len(images)
    print("PASS" if ok1 else "FAIL")

    rows = part2(model, data, cam, T_WC, rng, args.trials)
    p20 = next(q for q in rows if q["n"] == 20 and not q["distortion"])
    ok2 = abs(p20["mean"][0]) < 0.2 and np.all(np.abs(p20["mean"][1:3]) < 1.0)
    print("pinhole model, 20 views: " + ("PASS" if ok2 else "FAIL") + " (focal length within 0.2%, principal "
          "point within 1 px)")

    ok3 = part3(c, board_poses, model, data)
    print("PASS" if ok3 else "FAIL")
    cam.close()

    print("\n=== Calibrated values for src/config.py (pinhole model)")
    f_cal, pp_cal = (c.K[0, 0], c.K[1, 1]), (c.K[0, 2], c.K[1, 2])
    print(f"this run:  CAMERA_CAL_F = ({f_cal[0]:.2f}, {f_cal[1]:.2f}), CAMERA_CAL_PP = ({pp_cal[0]:.2f}, "
          f"{pp_cal[1]:.2f})")
    print(f"in config: CAMERA_CAL_F = {cfg.CAMERA_CAL_F}, CAMERA_CAL_PP = {cfg.CAMERA_CAL_PP}")
    df = cfg.CAMERA_CAL_F[0] / cfg.CAMERA_F_PX - 1
    print(f"config focal length error {100 * df:+.3f}% -> every marker range scaled by the same amount "
          f"({1000 * df:+.2f} mm at 1 m)")
    ok4 = (np.allclose(cfg.CAMERA_CAL_F, f_cal, atol=1.0) and np.allclose(cfg.CAMERA_CAL_PP, pp_cal, atol=1.0))
    if not ok4:
        print("-> config values differ from this calibration by more than 1 px: update them")
    print(f'\ncalibration test: {"PASS" if ok1 and ok2 and ok3 and ok4 else "FAIL"}')
    if not args.no_plot:
        plot(images, cals, rows)


if __name__ == "__main__":
    main()
