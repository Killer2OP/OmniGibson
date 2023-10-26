"""
Example script for interacting with OmniGibson scenes with Behaviorbot.
"""

import omnigibson as og
from omnigibson.utils.xr_utils import VRSys

def main():
    # Create the config for generating the environment we want
    scene_cfg = dict()
    scene_cfg["type"] = "InteractiveTraversableScene"
    scene_cfg["scene_model"] = "Rs_int"
    scene_cfg["load_object_categories"] = ["floors", "walls", "ceilings"]
    robot0_cfg = dict()
    robot0_cfg["type"] = "Behaviorbot"
    cfg = dict(scene=scene_cfg, robots=[robot0_cfg])

    # Create the environment
    env = og.Environment(configs=cfg, action_timestep=1/60., physics_timestep=1/240.)
    env.reset()
    # start vrsys 
    vr_robot = env.robots[0]
    vrsys = VRSys(system="SteamVR", vr_robot=vr_robot, enable_anchor_movement=True)
    vrsys.start()

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