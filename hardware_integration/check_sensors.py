"""First check with real robot: is every topic arriving, in the format the pipeline expects? Sends no commands.

For a few seconds it collects /joint_states, /odom, /scan and the camera, then reports:
    joint states  rate, joint names (must include hw_config.WHEEL_JOINTS), and how far each wheel turned. Drive
                  the robot FORWARD during the check (teleop in another terminal, or roll it by hand). Both wheels
                  should turn the same way, a negative one means its WHEEL_SIGNS entry should be -1
    odom          rate and the TurtleBot's own pose estimate
    scan          rate, beam count, angle_min / increment, range limits, how "no return" is encoded, and the
                  nearest return's bearing: put an object straight in front, it should read ~0 deg
                  (scan angle 0 = robot forward, counter-clockwise positive, as the pipeline assumes)
    camera        rate, image size, and the ArUco markers detected with their range and bearing
Usage: python hardware_integration/check_sensors.py 192.168.1.2XX [--seconds 10]   (or --sim)
"""

# py hardware_integration\check_sensors.py 192.168.1.210 --seconds 10
# === /joint_states: 197 messages (20/s)
# joint names: ['wheel_left_joint', 'wheel_right_joint']
# wheel angles turned during the check: left +0.000 rad, right +0.000 rad
#    (the wheels barely moved: drive the robot forward during the check to verify the signs)

# === /odom: 198 messages (20/s)
# TurtleBot's own odometry position: (1.051, 0.450) m

# === /scan: 49 scans (4.9/s; the LDS-01 runs at ~5/s)
# 360 beams, angle_min 0.0 deg, increment 1.00 deg, range 0.12-3.50 m
# 'no return' encoded as: 202 zeros, 0 nan, 0 inf, 0 below range_min
# after conversion to the pipeline's 360 beams: 135 valid returns
# nearest return 0.25 m at bearing +21 deg (an object straight ahead should read ~0, one on the robot's left ~+90)
# perception: 0 walls, 7 obstacles

# === camera: 297 frames (29.7/s)
# image size 640 x 480 (pipeline: 640 x 480)
# no markers detected in the last frame (point the camera at one to check ARUCO_DICT and size)

# all topics present


import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.perception import perception as per  
from hardware_integration import hw_config as hw 
from hardware_integration.robot_io import RobotIO, scan_to_beams, stamp_seconds  


def check(robot, seconds):
    hw.apply()
    io = RobotIO(robot)
    joints, odoms, scans, images = [], [], [], []
    t_end = io.now() + seconds
    last_js = last_od = None
    while io.now() < t_end:
        js = robot.query_joint_states()
        if js and (last_js is None or stamp_seconds(js) != stamp_seconds(last_js)):
            joints.append(js)
            last_js = js
        od = robot.query_odom() if hasattr(robot, "query_odom") else None
        if od and (last_od is None or stamp_seconds(od) != stamp_seconds(last_od)):
            odoms.append(od)
            last_od = od
        s = robot.query_laser_scan(timeout=0, wait_for_new=True)
        if s is not None:
            scans.append(s)
        im = robot.query_image(timeout=0, wait_for_new=True)
        if im is not None:
            images.append(im)
        io.wait_until(io.now() + 0.01)
    ok = True

    print(f"\n=== /joint_states: {len(joints)} messages ({len(joints) / seconds:.0f}/s)")
    if joints:
        names = list(joints[-1]["name"])
        print(f"joint names: {names}")
        if all(n in names for n in hw.WHEEL_JOINTS):
            p0 = np.array([joints[0]["position"][names.index(n)] for n in hw.WHEEL_JOINTS])
            p1 = np.array([joints[-1]["position"][names.index(n)] for n in hw.WHEEL_JOINTS])
            d = p1 - p0
            print(f"wheel angles turned during the check: left {d[0]:+.3f} rad, right {d[1]:+.3f} rad")
            if np.all(np.abs(d) < 0.05):
                print("   (the wheels barely moved: drive the robot forward during the check to verify the signs)")
            else:
                print("   driven forward, both should be positive. Suggested WHEEL_SIGNS = "
                      f"({1.0 if d[0] >= 0 else -1.0}, {1.0 if d[1] >= 0 else -1.0})")
        else:
            print(f"MISSING: expected wheel joints {hw.WHEEL_JOINTS}: set hw_config.WHEEL_JOINTS")
            ok = False
    else:
        print("NO joint states: odometry will not work (is the TurtleBot bringup running?)")
        ok = False

    print(f"\n=== /odom: {len(odoms)} messages ({len(odoms) / seconds:.0f}/s)")
    if odoms:
        try:
            p = odoms[-1]["pose"]["pose"]["position"]
            print(f"TurtleBot's own odometry position: ({p['x']:.3f}, {p['y']:.3f}) m")
        except (KeyError, TypeError):
            print("(no pose in the message)")
    else:
        print("no /odom (fine: the pipeline uses /joint_states by default)")

    print(f"\n=== /scan: {len(scans)} scans ({len(scans) / seconds:.1f}/s; the LDS-01 runs at ~5/s)")
    if scans:
        s = scans[-1]
        r = np.asarray(s.ranges, float)
        print(f"{len(r)} beams, angle_min {np.degrees(s.angle_min):.1f} deg, increment "
              f"{np.degrees(s.angle_increment):.2f} deg, range {s.range_min:.2f}-{s.range_max:.2f} m")
        print(f"'no return' encoded as: {int(np.sum(r == 0))} zeros, {int(np.sum(np.isnan(r)))} nan, "
              f"{int(np.sum(np.isinf(r)))} inf, {int(np.sum((r > 0) & (r < s.range_min)))} below range_min")
        rho = scan_to_beams(s)
        valid = np.isfinite(rho)
        print(f"after conversion to the pipeline's 360 beams: {int(valid.sum())} valid returns")
        if valid.any():
            i = int(np.argmin(np.where(valid, rho, np.inf)))
            print(f"nearest return {rho[i]:.2f} m at bearing {np.degrees(per.wrap(np.deg2rad(i))):+.0f} deg "
                  "(an object straight ahead should read ~0, one on the robot's left ~+90)")
            res = per.process_scan(rho, np.random.default_rng(0), T_RS=hw.T_RS)
            print(f"perception: {len(res.walls)} walls, {len(res.obstacles)} obstacles")
    else:
        print("NO scans: the LiDAR will not work")
        ok = False

    print(f"\n=== camera: {len(images)} frames ({len(images) / seconds:.1f}/s)")
    if images:
        h, w = images[-1].shape[:2]
        print(f"image size {w} x {h} (pipeline: {hw.CAMERA_SIZE[0]} x {hw.CAMERA_SIZE[1]})"
              + ("" if (w, h) == hw.CAMERA_SIZE else ": it will be resized; better to start camera_ros at 640 x 480"))
        import cv2
        rgb = cv2.cvtColor(images[-1], cv2.COLOR_BGR2RGB)
        obs = per.detect_markers(rgb, camera=hw.camera_model(), side=hw.MARKER_SIDE, T_RC=hw.T_RC,
                                 bias_px=hw.BIAS_PX, known=None, height_tol=10.0)
        for o in obs:
            print(f"marker {o.id}: range {o.z[0]:.2f} m, bearing {np.degrees(o.z[1]):+.1f} deg, centre height "
                  f"{o.T_RM.t[2]:.2f} m (hw_config.MARKER_HEIGHT = {hw.MARKER_HEIGHT:.2f})")
        if not obs:
            print("no markers detected in the last frame (point the camera at one to check ARUCO_DICT and size)")
    else:
        print("NO camera frames: the camera will not work")
        ok = False
    print("\nall topics present" if ok else "\nsomething is missing: see above")
    return ok


def main():
    parser = argparse.ArgumentParser(description="Check the robot's topics for the hardware pipeline")
    parser.add_argument("host", nargs="?")
    parser.add_argument("--sim", action="store_true")
    parser.add_argument("--seconds", type=float, default=10.0)
    args = parser.parse_args()
    if args.sim:
        from hardware_integration.sim_robot import SimRosbridgeRobot
        robot = SimRosbridgeRobot()
    elif args.host:
        from hardware_integration.robot_io import Waffle
        robot = Waffle(args.host)
    else:
        parser.error("give the robot's IP address, or --sim")
    with robot:
        check(robot, args.seconds)


if __name__ == "__main__":
    main()
