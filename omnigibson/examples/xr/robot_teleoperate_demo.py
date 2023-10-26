"""
Example script for vr system.
"""

import omnigibson as og
import omnigibson.utils.transform_utils as T
from omnigibson.utils.xr_utils import VRSys
from omnigibson.utils.ui_utils import choose_from_options

SCENES = ["Empty Scene", "Rs_int"]

SCENES = {
    "Empty Scene": "Empty scene with only a ground plane (default)",
    "Rs_int": "Interactive indoor scene with realistic objects",
}


ROBOTS = {
    "FrankaPanda": "Franka Emika Panda (default)",
    "Fetch": "Mobile robot with one arm",
    "Tiago": "Mobile robot with two arms",
}

def main():
    scene_model = choose_from_options(options=SCENES, name="scene")
    robot_name = choose_from_options(options=ROBOTS, name="robot")

    # Create the config for generating the environment we want
    scene_cfg = dict()
    if scene_model == "Empty Scene":
        scene_cfg["type"] = "Scene"
    else:
        scene_cfg["type"] = "InteractiveTraversableScene"
        scene_cfg["scene_model"] = scene_model
    # Add the robot we want to load
    robot0_cfg = {
        "type": robot_name,
        "obs_modalities": ["rgb", "depth", "seg_instance", "normal", "scan", "occupancy_grid"],
        "action_normalize": False,
        "controller_config": {
            "arm_0": {
                "name": "InverseKinematicsController",
                "mode": "pose_absolute_ori",
                "kv": 10.0,
            },
            "gripper_0": {
                "name": "MultiFingerGripperController", 
                "command_input_limits": (0.0, 1.0),
                "mode": "smooth", 
                "inverted": True
            }
        }
    }
    cfg = dict(scene=scene_cfg, robots=[robot0_cfg])

    # Create the environment
    env = og.Environment(configs=cfg, action_timestep=1/60., physics_timestep=1/240.)
    env.reset()
    # Start vrsys 
    vr_robot = env.robots[0]
    vrsys = VRSys(system="SteamVR", vr_robot=vr_robot, show_controller=True, disable_display_output=True)
    vrsys.start()
    # we need to update the anchor because we are spawned at the origin
    vrsys.update_anchor(offset=[0, 0, 1])

    # main simulation loop
    for _ in range(10000):
        if og.sim.is_playing():
            vr_data = vrsys.step()
            action = vr_robot.gen_action_from_vr_data(vr_data)
            env.step(action)     

    # Always shut down the environment cleanly at the end
    vrsys.stop()
    env.close()

if __name__ == "__main__":
    main()