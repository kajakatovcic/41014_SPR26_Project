# Use this script to test the TurtleBot3 Waffle Pi model in MuJoCo.
# It drives the robot straight for 5 seconds, then spins in place for 15 seconds.
# This will help you test that the model is working correctly in your MuJoCo installation.
# Requirements: Python 3.8+, mujoco>=2.4.0, mujoco-viewer>=2.4.0, and the robotis_mujoco_menagerie package.
import time
import mujoco
import mujoco.viewer

XML = r"robotis_mujoco_menagerie\robotis_tb3\scene_turtlebot3_waffle_pi.xml"  # adjust to your file

model = mujoco.MjModel.from_xml_path(XML)
data = mujoco.MjData(model)

print("Actuators:")
for i in range(model.nu):
    print(f"  {i}: {model.actuator(i).name}  ctrlrange={model.actuator_ctrlrange[i]}")
print("Sensors:")
for i in range(model.nsensor):
    print(f"  {i}: {model.sensor(i).name}")

with mujoco.viewer.launch_passive(model, data) as viewer:
    start = time.time()
    while viewer.is_running() and time.time() - start < 20:
        t = time.time() - start
        if model.nu >= 2:
            if t < 5:            # drive straight
                data.ctrl[0] = data.ctrl[1] = 5.0
            else:                # spin in place
                data.ctrl[0], data.ctrl[1] = 3.0, -3.0
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(model.opt.timestep)