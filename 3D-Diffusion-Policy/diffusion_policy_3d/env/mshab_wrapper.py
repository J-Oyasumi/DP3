from termcolor import cprint
from gymnasium import spaces
import gymnasium as gym
import numpy as np
from mani_skill.utils import gym_utils
from mani_skill.utils import common
from mani_skill.envs.sapien_env import BaseEnv
import mshab.envs

class MSHABEnv(gym.Wrapper):
    def __init__(self, 
                 env: gym.Env,
                 max_episode_steps=200,   
                 ):
        super().__init__(env)
        self.env = env
        self.cur_step = 0
        self.max_episode_steps = max_episode_steps

        self.action_space = self.base_env.action_space
        self.obs_state_dim = 12
        self.observation_space = spaces.Dict({
            'agent_pos': spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(12,),
                dtype=np.float32
            ),
            'point_cloud': spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(1024, 6),
                dtype=np.float32
            ),
        })


    @property
    def base_env(self) -> BaseEnv:
        return self.env.unwrapped
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.cur_step += 1
        done = terminated or truncated or info['success'] or self.cur_step >= self.max_episode_steps
        
        obs_dict = dict(agent_pos=obs['agent']['qpos'], point_cloud=obs['pointcloud'])

        return obs_dict, reward, done, info
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.cur_step = 0
        obs_dict = dict(agent_pos=obs['agent']['qpos'], point_cloud=obs['pointcloud'])
        return obs_dict, info