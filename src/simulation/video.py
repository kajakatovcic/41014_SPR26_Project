"""Showing the robot's camera: a live window next to the MuJoCo viewer, and an MP4 recording of a mission.

For demonstration only. Nothing here feeds back into perception, estimation or control. The frames are the same
noisy images perception receives, annotated with what the robot's software made of them (the marker detections
and their range and bearing, the controller mode). No ground truth is drawn.

    annotate   draw the marker detections (outline, id, range and bearing, Week 3 / Week 7 h_rb) on a camera frame
    setup_viewer  start the passive MuJoCo viewer on an overview of the room
    CameraWindow  a separate window streaming the annotated camera frames live, from the robot's point of view
    Recorder      MP4 of a fixed overview of the room side by side with the annotated camera feed
"""
from pathlib import Path

import cv2
import mujoco
import numpy as np

from src import config as cfg

GREEN, WHITE, BLACK = (40, 220, 40), (255, 255, 255), (0, 0, 0)


def _text(img, s, org, scale=0.5, colour=WHITE):
    """Text with a dark outline so it reads on any background."""
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, BLACK, 3, cv2.LINE_AA)
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, colour, 1, cv2.LINE_AA)


def annotate(rgb, observations, t, mode):
    """Copy of an (H, W, 3) RGB camera frame with each detected marker outlined and labelled with its id, range
    and bearing (perception's measurement z = [rho, alpha]), and a banner with the time and controller mode."""
    img = rgb.copy()
    for o in observations:
        pts = np.round(o.corners.T).astype(np.int32)
        cv2.polylines(img, [pts], True, GREEN, 2, cv2.LINE_AA)
        x = pts[:, 0].min()
        y = pts[:, 1].min() - 26 if pts[:, 1].min() > 56 else pts[:, 1].max() + 18 # above, or below if no room
        _text(img, f"id {o.id}{' (dock)' if o.id == cfg.DOCK_ID else ''}", (x, y), colour=GREEN)
        _text(img, f"{o.z[0]:.2f} m, {np.degrees(o.z[1]):+.1f} deg", (x, y + 18), colour=GREEN)
    cv2.rectangle(img, (0, 0), (img.shape[1], 22), BLACK, -1)
    _text(img, f"robot camera   t {t:5.1f} s   mode: {mode}   markers seen: {len(observations)}", (6, 16))
    return img


def overview_camera(cam):
    """Point a free MjvCamera at the whole room from the -y side, looking down, so x points right and y up the
    screen as in the map plots."""
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = (0.0, -0.05, 0.0)
    cam.distance, cam.azimuth, cam.elevation = 5.4, 90.0, -65.0


def setup_viewer(view):
    """Start the passive viewer on the room overview (the lidar rays stay on)."""
    with view.lock():
        overview_camera(view.cam)


class CameraWindow:
    """A separate window streaming the robot's camera, from the robot's point of view, next to the MuJoCo viewer.
    Closing it just stops the stream; the mission carries on."""

    def __init__(self, title="Robot camera (Waffle Pi RGB)", position=(20, 20)):
        self.title = title
        cv2.namedWindow(title, cv2.WINDOW_AUTOSIZE)
        cv2.moveWindow(title, *position)
        self.open = True

    def show(self, frame):
        if not self.open:
            return
        cv2.imshow(self.title, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        cv2.waitKey(1) # lets the window draw and handle events
        if cv2.getWindowProperty(self.title, cv2.WND_PROP_VISIBLE) < 1: # closed by the user
            self.open = False

    def close(self):
        if self.open:
            cv2.destroyWindow(self.title)
            cv2.waitKey(1)
        self.open = False


class Recorder:
    """MP4 video of a mission: a fixed overview of the room (left) and the annotated robot camera (right), one video
    frame per camera frame, so it plays back in real time at cfg.CAMERA_RATE."""

    def __init__(self, path, model, fps=cfg.CAMERA_RATE, size=(cfg.CAMERA_WIDTH, cfg.CAMERA_HEIGHT)):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.w, self.h = size
        self.renderer = mujoco.Renderer(model, self.h, self.w)
        self.cam = mujoco.MjvCamera()
        overview_camera(self.cam)
        self.option = mujoco.MjvOption()
        self.option.flags[mujoco.mjtVisFlag.mjVIS_RANGEFINDER] = False # the lidar rays would hide the room
        self.writer = cv2.VideoWriter(str(self.path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (2 * self.w, self.h))
        if not self.writer.isOpened():
            raise RuntimeError(f"could not open {self.path} for writing")
        self.frames = 0

    def write(self, data, camera_frame, t, mode):
        self.renderer.update_scene(data, camera=self.cam, scene_option=self.option)
        overview = self.renderer.render().copy()
        cv2.rectangle(overview, (0, 0), (self.w, 22), BLACK, -1)
        _text(overview, f"overview   t {t:5.1f} s", (6, 16))
        frame = np.hstack([overview, cv2.resize(camera_frame, (self.w, self.h))])
        self.writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        self.frames += 1

    def close(self):
        self.writer.release()
        self.renderer.close()
