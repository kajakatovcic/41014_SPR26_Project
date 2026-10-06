import mujoco
import mujoco.viewer

# Launch a passive viewer session
# You can also pass a loaded model: mujoco.MjModel.from_xml_path("path_to_model.xml")
with mujoco.viewer.launch_passive(mujoco.MjModel.from_xml_string('<mujoco></mujoco>'), mujoco.MjData(mujoco.MjModel.from_xml_string('<mujoco></mujoco>'))) as viewer:
    while viewer.is_running():
        pass