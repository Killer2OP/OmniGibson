import gym
import numpy as np
from omnigibson.envs.env_wrapper import EnvironmentWrapper
import omnigibson as og

class RLEnv(EnvironmentWrapper):
    def __init__(self, env_config):
        cfg = env_config['cfg']
        self.env_config = env_config
        self.env = og.Environment(configs=cfg, action_timestep=1 / 10., physics_timestep=1 / 60.)
        self.reset_positions = env_config['reset_positions']
        super().__init__(self.env)
        self._update_action_space()
        # self._update_observation_space()

    def reset(self):
        for name, position in enumerate(self.reset_positions):
            obj = self.env.scene.object_registry("name", name)
            if obj is not None:
                self.env.scene.object_registry("name", name).set_position_orientation(*position)
        return self.env.reset()

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        truncated = False
        return obs, reward, done, truncated, info
    
    def transform_action(self, action):
        idx = np.array([])
        for c in self.env_config['action_space_controllers']:
            idx = np.concatenate([idx, self.env.robots[0].controller_action_idx[c]]) 

        idx = np.sort(idx).astype(int)
        return action[idx]
    
    # def _update_observation_space(self):
    #     robot = self.env.robots[0]
    #     self.observation_space = robot.load_observation_space()
    
    def _update_action_space(self):
        action_space = self.env.robots[0].action_space
        idx = np.array([])
        for c in self.env_config['action_space_controllers']:
            idx = np.concatenate([idx, self.env.robots[0].controller_action_idx[c]]) 

        idx = np.sort(idx).astype(int)
        lows = []
        highs = []
        for i in idx:
            lows.append(action_space.low[i])
            highs.append(action_space.high[i]) 

        self.action_space = gym.spaces.Box(np.array(lows), np.array(highs), dtype=np.float32)
        
    

