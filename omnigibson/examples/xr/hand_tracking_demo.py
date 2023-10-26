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
    "FrankaAllegro": "Franka Panda with Allegro hand (default)",
    "Behaviorbot": "Humanoid robot with two hands",
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
        "obs_modalities": ["rgb", "depth", "normal", "scan", "occupancy_grid"],
        "action_normalize": False,
        "controller_config": {
            "arm_0": {
                "name": "InverseKinematicsController",
                "mode": "pose_absolute_ori",
                "command_input_limits": None,
                "motor_type": "position",
            },
        }
    }
    object_cfg = [{
        "type": "PrimitiveObject",
        "prim_path": f"/World/marker_{i}",
        "name": f"marker_{i}",
        "primitive_type": "Cube",
        "size": 0.01,
        "visual_only": True,
        "rgba": [0.0, 1.0, 0.0, 1.0],
    } for i in range(26)]
    cfg = dict(scene=scene_cfg, robots=[robot0_cfg], objects=object_cfg)

    # Create the environment
    env = og.Environment(configs=cfg, action_timestep=1/60., physics_timestep=1/240.)
    env.reset()
    # Start vrsys 
    vr_robot = env.robots[0]
    vrsys = VRSys(vr_robot=vr_robot, use_hand_tracking=True)
    vrsys.start()
    # we need to update the anchor because otherwise we are spawned at the origin
    vrsys.update_anchor(offset=[0, 0, 1])
    markers = [env.scene.object_registry("name", f"marker_{i}") for i in range(26)]
    
    for _ in range(10000):
        if og.sim.is_playing():
            vr_data = vrsys.step()
            if "right" in vr_data["hand_data"]:
                for i in range(26):
                    pos, orn = T.mat2pose(vr_data["hand_data"]["right"]["raw"][i])
                    markers[i].set_position_orientation(pos, orn)
            action = vr_robot.gen_action_from_vr_data(vr_data)
            env.step(action)                

    # Always shut down the environment cleanly at the end
    vrsys.stop()
    env.close()

if __name__ == "__main__":
    main()