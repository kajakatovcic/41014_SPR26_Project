import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mujoco, numpy as np
from src import  config as cfg
from src.simulation import world

model = world.build_world()
data = mujoco.MjData(model)

print([model.actuator(i).name for i in range(model.nu)])   # actuator names and order
print(model.joint("wheel_left").axis, model.joint("wheel_right").axis)  # joint axes

world.set_pose(model, data, 0, 0, 0)
data.ctrl[:] = 3.0          # same positive command on every actuator (check it's velocity control)
enc0 = world.read_encoders(model, data)
p0 = world.true_pose(model, data)
for _ in range(int(1.0 / model.opt.timestep)):
    mujoco.mj_step(model, data)
print("encoder change [L, R]:", world.read_encoders(model, data) - enc0)
print("true pose change     :", world.true_pose(model, data) - p0)


world.set_pose(model, data, 0, 0, 0)
data.ctrl[:] = [-2.0, 2.0]      # left reverse, right forward
enc0 = world.read_encoders(model, data)
p0 = world.true_pose(model, data)
for _ in range(int(1.0 / model.opt.timestep)):
    mujoco.mj_step(model, data)
enc1 = world.read_encoders(model, data)
p1 = world.true_pose(model, data)

dL, dR = enc1 - enc0
print("encoder change [L, R]:", dL, dR)
print("true heading change  :", p1[2] - p0[2])
print("predicted heading    :", cfg.WHEEL_RADIUS * (dR - dL) / cfg.WHEEL_TRACK)
print("true position change :", p1[:2] - p0[:2])