# Use this file to test your mujoco installation
import mujoco  # type: ignore
from mujoco import viewer  # type: ignore

# Launch a passive viewer session
# You can also pass a loaded model: mujoco.MjModel.from_xml_path("path_to_model.xml")
with viewer.launch_passive(
    mujoco.MjModel.from_xml_string('<mujoco></mujoco>'),
    mujoco.MjData(mujoco.MjModel.from_xml_string('<mujoco></mujoco>')),
) as viewer_session:
    while viewer_session.is_running():
        pass