"""MuJoCo world for Track A. 
Includes the TurtleBot3 Waffle Pi with its lidar and camera, in a room with obstacles, a charging
station and ArUco landmarks + the sensor drivers that turn the simulation into noisy sensor data.

Only the drivers (read_lidar, render_camera, read_encoders) may feed perception, estimation and control.
true_pose() is ground truth and is for evaluation and plotting only.
"""
import cv2
import mujoco
import numpy as np
from spatialmath import SE3, UnitQuaternion

from src import config as cfg

CAMERA_NAME = "camera_rgb"


# Build the modle

def _fix_caster(spec):
    """The waffle_pi_base mesh includes the ball casters so its convex hull touches the floor where the model's
    low-friction caster spheres do, but with friction 1. 
    MuJoCo also takes the larger friction of the two geoms in contact so even the caster spheres get the floor's 
    friction. Together, these drag the robot's rear. It covers ~60% of the distance the wheels roll and cannot
    tunr on the spot. Give the chassis collision mesh the casters' contact priorities + a higher priority so they
    are used instead of the floor's."""
    casters = [g for g in spec.geoms if g.type == mujoco.mjtGeom.mjGEOM_SPHERE and g.parent.name == "base"]
    chassis = [g for g in spec.geoms if g.meshname == "waffle_pi_base" and g.contype]
    for g in casters + chassis:
        g.friction, g.condim, g.priority = casters[0].friction, casters[0].condim, 1


def _quat(T):
    return UnitQuaternion(T.R).vec


def _add_sensors(spec):
    base = spec.body("base")

    # 2D lidar: one rangefinder per beam measuring along the site's +z axis. Rangefinders ignore the body they
    # are attached to so the robot's own geoms don't block the beams
    for i, a in enumerate(cfg.LIDAR_ANGLES):
        T = cfg.T_RS * SE3.Rz(a) * SE3.Ry(np.pi / 2) # site +z along the beam
        base.add_site(name=f"lidar_{i}", pos=T.t, quat=_quat(T), size=[0.002], group=4)
        spec.add_sensor(name=f"lidar_{i}", type=mujoco.mjtSensor.mjSENS_RANGEFINDER,
                        objtype=mujoco.mjtObj.mjOBJ_SITE, objname=f"lidar_{i}",
                        intprm=[1, 0, 0]) # output distance only

    # Camera looking out of the Waffle's camera housing. 
    # MuJoCo cameras look along -z with +y up so rotate {C} (z forward, y down) by Rx(pi)
    T = cfg.T_RC * SE3.Rx(np.pi)
    base.add_camera(name=CAMERA_NAME, pos=T.t, quat=_quat(T), fovy=cfg.CAMERA_FOVY)


def marker_image(marker_id, px=600, pad=100):
    """An ArUco marker as a greyscale image with a white quiet zone (seen in wk3, marker_image)."""
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    tag = cv2.aruco.generateImageMarker(dictionary, marker_id, px)
    return cv2.copyMakeBorder(tag, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=255), px, pad


def _add_marker(spec, marker_id, T_WM):
    img, px, pad = marker_image(marker_id)
    rgb = np.ascontiguousarray(np.repeat(img[:, :, None], 3, axis=2))
    tex = spec.add_texture(name=f"aruco_{marker_id}", type=mujoco.mjtTexture.mjTEXTURE_2D,
                           width=rgb.shape[1], height=rgb.shape[0], nchannel=3)
    tex.data = rgb.tobytes()
    mat = spec.add_material(name=f"aruco_{marker_id}")
    mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = f"aruco_{marker_id}"

    # The texture lands the right way round on the box face pointing along the body's +z so the body frame is
    # the marker frame {M}. The plate sits just behind the marker plane.
    half = cfg.MARKER_SIDE * (px + 2 * pad) / px / 2
    body = spec.worldbody.add_body(name=f"aruco_{marker_id}", pos=T_WM.t, quat=_quat(T_WM))
    body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[half, half, 0.002], pos=[0, 0, -0.002],
                  material=f"aruco_{marker_id}", contype=0, conaffinity=0)


def _add_room(spec):
    world = spec.worldbody
    h, t, z = cfg.ROOM_HALF, cfg.WALL_THICK, cfg.WALL_HEIGHT / 2
    for name, pos, size in [("wall_xp", [h, 0, z], [t, h, z]), ("wall_xn", [-h, 0, z], [t, h, z]),
                            ("wall_yp", [0, h, z], [h, t, z]), ("wall_yn", [0, -h, z], [h, t, z])]:
        world.add_geom(name=name, type=mujoco.mjtGeom.mjGEOM_BOX, pos=pos, size=size, rgba=[0.8, 0.8, 0.75, 1])

    colours = {"red": [0.85, 0.1, 0.1, 1], "green": [0.1, 0.7, 0.2, 1],
               "blue": [0.1, 0.3, 0.85, 1], "yellow": [0.9, 0.8, 0.1, 1]}
    for name, kind, (x, y), size, height in cfg.OBSTACLES:
        rgba = colours.get(name.split("_")[-1], [0.5, 0.5, 0.5, 1])
        if kind == "box":
            world.add_geom(name=name, type=mujoco.mjtGeom.mjGEOM_BOX, pos=[x, y, height / 2],
                           size=[size[0], size[1], height / 2], rgba=rgba)
        else:
            world.add_geom(name=name, type=mujoco.mjtGeom.mjGEOM_CYLINDER, pos=[x, y, height / 2],
                           size=[size, height / 2], rgba=rgba)

    sx, sy, sz = cfg.DOCK_SIZE
    world.add_geom(name="dock", type=mujoco.mjtGeom.mjGEOM_BOX, pos=[*cfg.DOCK_POS, sz], size=[sx, sy, sz],
                   rgba=[0.25, 0.25, 0.28, 1])
    for marker_id, T_WM in cfg.MARKERS.items():
        _add_marker(spec, marker_id, T_WM)


def checkerboard_image(px=60):
    """The calibration target as a greyscale image: cfg.CHECKERBOARD_GRID internal corners, px pixels per square,
    with a one-square white border."""
    nx, ny = cfg.CHECKERBOARD_GRID[0] + 1, cfg.CHECKERBOARD_GRID[1] + 1        # squares
    img = np.full(((ny + 2) * px, (nx + 2) * px), 255, np.uint8)
    for r in range(ny):
        for c in range(nx):
            if (r + c) % 2 == 0:
                img[(r + 1) * px:(r + 2) * px, (c + 1) * px:(c + 2) * px] = 0
    return img


def _add_checkerboard(spec):
    """A checkerboard on a mocap body, so a test can hold it anywhere in front of the camera, like a person holding
    the board in front of the robot. As with the markers, the body frame {B} has x right, y up and z out of the
    printed face, with its origin at the centre of the board."""
    img = checkerboard_image()
    rgb = np.ascontiguousarray(np.repeat(img[:, :, None], 3, axis=2))
    tex = spec.add_texture(name="checkerboard", type=mujoco.mjtTexture.mjTEXTURE_2D,
                           width=rgb.shape[1], height=rgb.shape[0], nchannel=3)
    tex.data = rgb.tobytes()
    mat = spec.add_material(name="checkerboard")
    mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "checkerboard"
    half = [(n + 3) * cfg.CHECKERBOARD_SQUARE / 2 for n in cfg.CHECKERBOARD_GRID]   # squares + border
    body = spec.worldbody.add_body(name="checkerboard", mocap=True, pos=[0, 0, -1])   # parked under the floor
    body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[*half, 0.002], pos=[0, 0, -0.002],
                  material="checkerboard", contype=0, conaffinity=0)


def build_world(checkerboard=False):
    """The Track A world. checkerboard=True adds the calibration target (see set_checkerboard)."""
    spec = mujoco.MjSpec.from_file(str(cfg.TB3_SCENE_XML))
    _fix_caster(spec)
    _add_sensors(spec)
    _add_room(spec)
    if checkerboard:
        _add_checkerboard(spec)
    return spec.compile()


# Sensor drivers: simulation -> noisy sensor data

def read_lidar(model, data, rng, sigma=cfg.LIDAR_SIGMA):
    """One lidar scan: ranges [m] for each beam in cfg.LIDAR_ANGLES, inf where there is no valid return."""
    adr = model.sensor("lidar_0").adr[0]
    rho = data.sensordata[adr:adr + cfg.LIDAR_NUM_RAYS].copy()
    rho[rho < 0] = np.inf # rangefinder returns -1 on a miss
    rho = rho + rng.normal(0, sigma, rho.shape)
    rho[(rho < cfg.LIDAR_RANGE_MIN) | (rho > cfg.LIDAR_RANGE_MAX)] = np.inf
    return rho


class Camera:
    """Offscreen renderer for the robot's RGB camera with Gaussian pixel noise."""

    def __init__(self, model, noise=cfg.CAMERA_NOISE):
        self.renderer = mujoco.Renderer(model, cfg.CAMERA_HEIGHT, cfg.CAMERA_WIDTH)
        self.option = mujoco.MjvOption()
        self.option.flags[mujoco.mjtVisFlag.mjVIS_RANGEFINDER] = False # don't draw the lidar beams
        self.noise = noise

    def render(self, data, rng):
        """An (H, W, 3) uint8 RGB image."""
        self.renderer.update_scene(data, camera=CAMERA_NAME, scene_option=self.option)
        img = self.renderer.render().astype(float)
        if self.noise > 0:
            img += rng.normal(0, self.noise, img.shape)
        return np.clip(img, 0, 255).astype(np.uint8)

    def close(self):
        self.renderer.close()


def read_encoders(model, data):
    """Left and right wheel angles [rad], quantised to the encoder resolution."""
    q = data.qpos[[model.joint("wheel_left").qposadr[0], model.joint("wheel_right").qposadr[0]]]
    step = 2 * np.pi / cfg.ENCODER_TICKS
    return np.round(q / step) * step


# NOTE for tests only. Ground truth and test helpers. NOT used by the controller or estimator

def true_pose(model, data):
    """True robot pose (x, y, theta) in the world frame. For evaluation only."""
    b = model.body("base").id
    return np.array([data.xpos[b][0], data.xpos[b][1], np.arctan2(data.xmat[b][3], data.xmat[b][0])])


def set_checkerboard(model, data, T_WB):
    """Move the calibration target so its centre frame {B} is at world pose T_WB."""
    mid = model.body("checkerboard").mocapid[0]
    data.mocap_pos[mid] = T_WB.t
    data.mocap_quat[mid] = _quat(T_WB)
    mujoco.mj_forward(model, data)


def set_pose(model, data, x, y, theta):
    """Place the robot at (x, y, theta), at rest. 
    Used for tests and for setting the start pose."""
    adr = model.joint("base_joint").qposadr[0]
    data.qpos[adr:adr + 7] = [x, y, 0, np.cos(theta / 2), 0, 0, np.sin(theta / 2)]
    data.qvel[:] = 0
    mujoco.mj_forward(model, data)
