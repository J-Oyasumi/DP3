from termcolor import cprint
from gymnasium import spaces
import gymnasium as gym
import numpy as np
from mani_skill.utils import gym_utils
from mani_skill.utils import common
import mshab.envs

class MSHABEnv(gym.Wrapper):
    def __init__(self, 
                 env: gym.Env,
                 max_episode_steps=200,   
                 ):
        super(MSHABEnv, self).__init__(env)
        self.env = env
        self.cur_step = 0
        self.max_episode_steps = max_episode_steps
    
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