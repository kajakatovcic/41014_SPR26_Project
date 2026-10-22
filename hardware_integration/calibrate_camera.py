"""Calibrate the real robot's camera (Week 3 Parts B and C) for the hardware pipeline.

The same procedure as tests/test_calibration.py (src/perception/calibration.py- Zhang's method with OpenCV, then the
per-image reprojection error) on images from the robot's own camera. Unlike the simulated camera a real lens has
distortion, so the full (k1, k2, p1, p2, k3) model is fitted. The pipeline undistorts every frame with it
(robot_io.RobotIO.read_image) and then uses the pinhole model as in simulation.

Live from the robot (sends no commands):
    python hardware_integration/calibrate_camera.py 192.168.1.2XX [--grid 9 6] [--square 0.04]
  A window shows the camera with the detected board. Hold the printed checkerboard in front of the robot at
  different distances (0.4-1.2m), positions across the image (including the corners) and tilts (up to ~40deg).
    SPACE  keep this view (only when the board is detected)     C  calibrate with the views kept (>= 12)
    Q/Esc  quit
From saved images instead:
    python hardware_integration/calibrate_camera.py --folder hardware_integration/calibration_images
The result goes to hardware_integration/camera_calibration.json, which hw_config loads.
--grid is the number of INTERNAL corners (columns, rows) and --square the side of one square [m]: @TODO measure them.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.perception import calibration as calib  
from hardware_integration import hw_config as hw  

MIN_VIEWS = 12


def run_calibration(images_rgb, grid, square, out=hw.CALIBRATION_FILE):
    """Calibrate with distortion, report the Week 3 C checks, and save the result."""
    cal = calib.calibrate(images_rgb, distortion=True, gridshape=grid, square=square)
    errors, _ = calib.reprojection_errors(cal, gridshape=grid, square=square)
    per_image = np.array([np.sqrt(np.mean(e ** 2)) for e in errors])
    K, d = cal.K, cal.dist
    print(f"\n{len(cal.ids)} of {len(images_rgb)} images used; reprojection RMS {cal.rms:.3f} px "
          f"(per image {per_image.min():.3f}-{per_image.max():.3f} px, worst image {cal.ids[int(np.argmax(per_image))]})")
    print(f"f/rho_u {K[0, 0]:.1f} px, f/rho_v {K[1, 1]:.1f} px, principal point ({K[0, 2]:.1f}, {K[1, 2]:.1f}) px")
    print(f"distortion (k1, k2, p1, p2, k3) = {np.round(d, 4).tolist()}")
    if cal.rms > 0.5:
        print("RMS above 0.5px: check the board is flat and the square size, and drop blurred views")
    if per_image.max() > 3 * np.median(per_image):
        print("one view is much worse than the rest: consider removing it (Week 3 C)")
    h, w = images_rgb[0].shape[:2]
    out.write_text(json.dumps({"K": K.tolist(), "dist": d.tolist(), "rms": float(cal.rms), "views": len(cal.ids),
                               "image_size": [w, h], "grid": list(grid), "square": square,
                               "date": time.strftime("%Y-%m-%d %H:%M")}, indent=2))
    print(f"saved to {out}")
    return cal


def live(robot, grid, square, save_dir):
    save_dir.mkdir(parents=True, exist_ok=True)
    kept = []
    print(__doc__.split("Live from the robot (sends no commands):")[1].split("From saved")[0])
    while True:
        bgr = robot.query_image(timeout=1.0, wait_for_new=True)
        if bgr is None:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        corners = calib.measured_corners(rgb, grid)
        shown = bgr.copy()
        if corners is not None:
            cv2.drawChessboardCorners(shown, grid, corners.reshape(-1, 1, 2).astype(np.float32), True)
        cv2.putText(shown, f"views kept: {len(kept)}  (SPACE keep, C calibrate, Q quit)", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0) if corners is not None else (0, 0, 255), 2)
        cv2.imshow("calibration", shown)
        key = cv2.waitKey(1) & 0xFF
        if key == ord(" ") and corners is not None:
            kept.append(rgb)
            cv2.imwrite(str(save_dir / f"view_{len(kept):02d}.png"), bgr)
            print(f"kept view {len(kept)}")
        elif key == ord("c"):
            if len(kept) < MIN_VIEWS:
                print(f"keep at least {MIN_VIEWS} views first ({len(kept)} so far)")
                continue
            run_calibration(kept, grid, square)
        elif key in (ord("q"), 27):
            break
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Calibrate the robot's camera from checkerboard views")
    parser.add_argument("host", nargs="?", help="robot IP address (live capture)")
    parser.add_argument("--folder", default=None, help="calibrate from the images in this folder instead")
    parser.add_argument("--grid", type=int, nargs=2, default=list(hw.cfg.CHECKERBOARD_GRID),
                        help="internal corners (columns rows)")
    parser.add_argument("--square", type=float, default=hw.cfg.CHECKERBOARD_SQUARE, help="square side [m]")
    parser.add_argument("--save-dir", default=str(hw.HERE / "calibration_images"))
    args = parser.parse_args()
    grid = tuple(args.grid)
    if args.folder:
        files = sorted(p for p in Path(args.folder).iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg"))
        images = [cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB) for p in files]
        print(f"{len(images)} images in {args.folder}")
        run_calibration(images, grid, args.square)
    elif args.host:
        from hardware_integration.robot_io import Waffle
        with Waffle(args.host) as robot:
            live(robot, grid, args.square, Path(args.save_dir))
    else:
        parser.error("give the robot's IP address, or --folder")


if __name__ == "__main__":
    main()
