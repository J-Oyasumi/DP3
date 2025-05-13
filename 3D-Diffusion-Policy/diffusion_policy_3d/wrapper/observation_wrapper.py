import copy
from typing import Dict

import gymnasium as gym
import gymnasium.spaces.utils
import numpy as np
import torch
from gymnasium.vector.utils import batch_space

from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils import common

class FlattenPoindCloudObservationWrapper(gym.ObservationWrapper):
    """
    Flattens the pointcloud mode observations into a dictionary with two keys, "poindcloud" and "state"

    Args:
        pointcloud (bool): Whether to include pointcloud in the observation
        state (bool): Whether to include state data in the observation
    """

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)

        new_obs = self.observation(self.base_env._init_raw_obs)
        self.base_env.update_obs_space(new_obs)

    @property
    def base_env(self) -> BaseEnv:
        return self.env.unwrapped

    def observation(self, observation: Dict):
        pointcloud = observation['pointcloud']
        state = observation['agent']['qpos']
        
        ret = dict(state=state, pointcloud=pointcloud)
        
        return ret