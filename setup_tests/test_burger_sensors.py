# Use this script to test the TurtleBot3 Burger model in MuJoCo with a 2D lidar and an RGB camera.
# The sensors are added to the stock robotis_mujoco_menagerie model at load time (the submodule is not modified):
#   * 2D lidar - 360 rangefinders (1 deg resolution, 0.12-3.5 m) on the LDS, like the real LDS-01
#   * RGB camera - forward-facing camera on the front of the robot, rendered offscreen
# The robot wanders around a walled arena, using the lidar to avoid obstacles.
# A MuJoCo viewer shows the scene (lidar rays drawn in yellow) and an OpenCV window shows the camera image + lidar scan.
# Requirements: Python 3.8+, mujoco>=3.2.0, numpy, opencv-python, and the robotis_mujoco_menagerie submodule.
# Usage: python setup_tests/test_burger_sensors.py [--duration 60] [--headless]
import argparse
import time
from pathlib import Path

import cv2
import mujoco
import mujoco.viewer
import numpy as np

XML = Path(__file__).resolve().parents[1] / "robotis_mujoco_menagerie" / "robotis_tb3" / "scene_turtlebot3_burger.xml"

# Lidar (LDS-01 spec)
LIDAR_POS = (-0.032, 0.0, 0.182)  # scan origin in the base frame (base_scan in the ROS URDF)
LIDAR_NUM_RAYS = 360
LIDAR_RANGE_MIN = 0.12
LIDAR_RANGE_MAX = 3.5
LIDAR_ANGLES = np.deg2rad(np.arange(LIDAR_NUM_RAYS) * 360.0 / LIDAR_NUM_RAYS)  # CCW from forward (ROS convention)

# RGB camera
CAMERA_NAME = "camera_rgb"
CAMERA_POS = (0.05, 0.0, 0.12)
CAMERA_WIDTH, CAMERA_HEIGHT = 640, 480
CAMERA_FOVY = 48.8  # Raspberry Pi camera v2 vertical FOV (deg)

# Wheel control (velocity actuators, rad/s)
WHEEL_FORWARD = 5.0
WHEEL_TURN = 3.0
OBSTACLE_DIST = 0.45  # m, turn away if anything in the front sector is closer than this


def ray_quat(theta):
    """Quaternion that points a site's +z axis horizontally at yaw angle theta (rangefinders measure along +z)."""
    q_pitch = np.array([np.cos(np.pi / 4), 0.0, np.sin(np.pi / 4), 0.0])  # +z -> +x
    q_yaw = np.array([np.cos(theta / 2), 0.0, 0.0, np.sin(theta / 2)])
    q = np.zeros(4)
    mujoco.mju_mulQuat(q, q_yaw, q_pitch)
    return q


def camera_quat():
    """MuJoCo cameras look along -z with +y up; point it along the robot's +x with +z up."""
    x_axis, y_axis = np.array([0.0, -1.0, 0.0]), np.array([0.0, 0.0, 1.0])
    mat = np.column_stack([x_axis, y_axis, np.cross(x_axis, y_axis)])
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, mat.flatten())
    return q


def add_sensors(spec):
    base = spec.body("base")

    # 2D lidar: one site + rangefinder per beam. Rangefinders ignore the body they are attached to,
    # so the robot's own LDS/base geoms don't block the beams.
    for i, theta in enumerate(LIDAR_ANGLES):
        base.add_site(name=f"lidar_{i}", pos=LIDAR_POS, quat=ray_quat(theta), size=[0.002], group=4)
        spec.add_sensor(
            name=f"lidar_{i}",
            type=mujoco.mjtSensor.mjSENS_RANGEFINDER,
            objtype=mujoco.mjtObj.mjOBJ_SITE,
            objname=f"lidar_{i}",
            intprm=[1, 0, 0],  # output distance only (what <rangefinder> in XML defaults to)
        )

    # RGB camera with a small visual-only housing
    base.add_camera(name=CAMERA_NAME, pos=CAMERA_POS, quat=camera_quat(), fovy=CAMERA_FOVY)
    base.add_geom(
        type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.008, 0.015, 0.01], pos=CAMERA_POS,
        rgba=[0.1, 0.5, 0.1, 1], contype=0, conaffinity=0, density=0, group=2,
    )


def add_arena(spec):
    world = spec.worldbody
    half, height, thick = 2.0, 0.3, 0.05
    wall_rgba = [0.8, 0.8, 0.75, 1]
    for name, pos, size in [
        ("wall_n", [half, 0, height / 2], [thick, half, height / 2]),
        ("wall_s", [-half, 0, height / 2], [thick, half, height / 2]),
        ("wall_e", [0, -half, height / 2], [half, thick, height / 2]),
        ("wall_w", [0, half, height / 2], [half, thick, height / 2]),
    ]:
        world.add_geom(name=name, type=mujoco.mjtGeom.mjGEOM_BOX, pos=pos, size=size, rgba=wall_rgba)

    # Coloured obstacles so the camera has something to look at
    world.add_geom(name="box_red", type=mujoco.mjtGeom.mjGEOM_BOX,
                   pos=[1.0, 0.4, 0.15], size=[0.15, 0.15, 0.15], rgba=[0.85, 0.1, 0.1, 1])
    world.add_geom(name="cyl_green", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                   pos=[0.6, -0.9, 0.2], size=[0.12, 0.2], rgba=[0.1, 0.7, 0.2, 1])
    world.add_geom(name="box_blue", type=mujoco.mjtGeom.mjGEOM_BOX,
                   pos=[-0.9, 0.8, 0.2], size=[0.25, 0.1, 0.2], rgba=[0.1, 0.3, 0.85, 1])
    world.add_geom(name="cyl_yellow", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                   pos=[-1.0, -0.8, 0.2], size=[0.08, 0.2], rgba=[0.9, 0.8, 0.1, 1])


def fix_caster(spec):
    """The burger_base mesh includes the ball caster, so its convex hull touches the floor at the same point as the
    model's low-friction caster sphere, but with friction 1. MuJoCo also takes the larger friction of the two geoms
    in contact, so even the caster sphere gets the floor's friction. Together these drag the rear of the robot: it
    covers ~75% of the distance the wheels roll and barely turns. Give the chassis collision mesh the caster's
    contact properties, and a higher priority so they are used instead of the floor's."""
    caster = next(g for g in spec.geoms if g.type == mujoco.mjtGeom.mjGEOM_SPHERE and g.parent.name == "base")
    chassis = [g for g in spec.geoms if g.meshname == "burger_base" and g.contype]
    for g in [caster] + chassis:
        g.friction, g.condim, g.priority = caster.friction, caster.condim, 1


def build_model():
    spec = mujoco.MjSpec.from_file(str(XML))
    fix_caster(spec)
    add_sensors(spec)
    add_arena(spec)
    return spec.compile()


def render_camera(renderer, data, scene_option):
    """Render the onboard RGB camera, returns an (H, W, 3) uint8 RGB image."""
    renderer.update_scene(data, camera=CAMERA_NAME, scene_option=scene_option)
    return renderer.render()


def read_lidar(model, data):
    """Return the scan as an array of ranges (m), with misses / out-of-range beams set to inf."""
    adr = model.sensor("lidar_0").adr[0]
    ranges = data.sensordata[adr:adr + LIDAR_NUM_RAYS].copy()
    ranges[(ranges < LIDAR_RANGE_MIN) | (ranges > LIDAR_RANGE_MAX)] = np.inf  # rangefinder returns -1 on a miss
    return ranges


def wander(ranges):
    """Simple reactive controller: drive forward, turn towards the more open side when blocked."""
    front = np.concatenate([ranges[:30], ranges[-30:]])
    if front.min() > OBSTACLE_DIST:
        return WHEEL_FORWARD, WHEEL_FORWARD
    left, right = ranges[30:90], ranges[270:330]
    left_space = np.mean(np.minimum(left, LIDAR_RANGE_MAX))
    right_space = np.mean(np.minimum(right, LIDAR_RANGE_MAX))
    return (-WHEEL_TURN, WHEEL_TURN) if left_space > right_space else (WHEEL_TURN, -WHEEL_TURN)


def draw_scan(ranges, size=CAMERA_HEIGHT):
    """Top-down lidar plot, robot at the centre facing up."""
    img = np.full((size, size, 3), 30, np.uint8)
    c = size // 2
    scale = (size / 2 - 10) / LIDAR_RANGE_MAX
    for r in (1.0, 2.0, 3.0):
        cv2.circle(img, (c, c), int(r * scale), (70, 70, 70), 1)
    hit = np.isfinite(ranges)
    xs, ys = ranges[hit] * np.cos(LIDAR_ANGLES[hit]), ranges[hit] * np.sin(LIDAR_ANGLES[hit])
    for x, y in zip(xs, ys):
        cv2.circle(img, (int(c - y * scale), int(c - x * scale)), 2, (0, 220, 255), -1)
    cv2.circle(img, (c, c), 5, (0, 255, 0), -1)
    cv2.line(img, (c, c), (c, c - 15), (0, 255, 0), 2)
    cv2.putText(img, "lidar (1 m rings)", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    return img


def main():
    parser = argparse.ArgumentParser(description="TurtleBot3 Burger 2D lidar + RGB camera test")
    parser.add_argument("--duration", type=float, default=60.0, help="seconds to run")
    parser.add_argument("--headless", action="store_true", help="no windows, just step and print sensor stats")
    args = parser.parse_args()

    model = build_model()
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, CAMERA_HEIGHT, CAMERA_WIDTH)
    camera_option = mujoco.MjvOption()
    camera_option.flags[mujoco.mjtVisFlag.mjVIS_RANGEFINDER] = False  # don't draw the lidar beams in the camera image

    print(f"Actuators: {[model.actuator(i).name for i in range(model.nu)]}")
    print(f"Lidar: {LIDAR_NUM_RAYS} beams, {LIDAR_RANGE_MIN}-{LIDAR_RANGE_MAX} m | "
          f"Camera: {CAMERA_WIDTH}x{CAMERA_HEIGHT}, fovy={CAMERA_FOVY} deg")

    sensor_period = 0.1  # s, update lidar/camera/controller at 10 Hz
    next_sensor_time = 0.0

    if args.headless:
        while data.time < args.duration:
            if data.time >= next_sensor_time:
                ranges = read_lidar(model, data)
                data.ctrl[:] = wander(ranges)
                next_sensor_time += sensor_period
            mujoco.mj_step(model, data)
        rgb = render_camera(renderer, data, camera_option)
        ranges = read_lidar(model, data)
        print(f"t={data.time:.1f}s  pos=({data.qpos[0]:.2f}, {data.qpos[1]:.2f})  "
              f"lidar hits={np.isfinite(ranges).sum()}/{LIDAR_NUM_RAYS}  min={ranges.min():.2f} m  "
              f"camera frame={rgb.shape} mean={rgb.mean():.1f}")
        renderer.close()
        return

    with mujoco.viewer.launch_passive(model, data) as viewer:
        start = time.time()
        while viewer.is_running() and data.time < args.duration:
            # Step physics until the sim catches up with wall-clock time
            while data.time < time.time() - start:
                mujoco.mj_step(model, data)

            if data.time >= next_sensor_time:
                ranges = read_lidar(model, data)
                data.ctrl[:] = wander(ranges)

                rgb = render_camera(renderer, data, camera_option)
                frame = np.hstack([cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), draw_scan(ranges)])
                cv2.imshow("TurtleBot3 Burger - RGB camera | 2D lidar", frame)
                cv2.waitKey(1)
                next_sensor_time = data.time + sensor_period

            viewer.sync()
            time.sleep(0.002)

    renderer.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
