import wandb
import numpy as np
import torch
from tqdm import tqdm
import gymnasium as gym
import pathlib
import os
import random

from diffusion_policy_3d.env.mshab_wrapper import MSHABEnv
from diffusion_policy_3d.policy.base_policy import BasePolicy
from diffusion_policy_3d.common.utils import dict_apply
from diffusion_policy_3d.env_runner.base_runner import BaseRunner
import diffusion_policy_3d.common.logger_utils as logger_util
from termcolor import cprint

from mshab.envs.planner import plan_data_from_file

class MSHABRunner(BaseRunner):
    def __init__(
            self,
            task_name,
            eval_episodes,
            output_dir,
            **kwargs,
            ):
        super().__init__(output_dir)

        self.eval_episodes = eval_episodes
    
        ASSET_DIR = pathlib.Path(os.environ['MS_ASSET_DIR'])
        REARRANGE_DIR = ASSET_DIR / "data/scene_datasets/replica_cad_dataset/rearrange"
        TASK = "set_table"
        split = "val"

        subtask, obj = task_name.split("_", 1)
        env_id = f'{subtask.capitalize()}SubtaskTrain-v0'
        plan_data = plan_data_from_file(
        REARRANGE_DIR / "task_plans" / TASK / subtask / split / f'{obj}.json'
        )
        spawn_data_fp = REARRANGE_DIR / "spawn_data" / TASK / subtask / split / "spawn_data.pt"

        self.env = gym.make(
            env_id,
            num_envs=1,
            robot_uid="fetch",
            obs_mode="kwargs",
            control_mode="pd_joint_delta_pos",
            sensor_config=dict(shader_pack="default"),
            sim_backend="auto",
            max_episode_steps=200,
            task_plans=plan_data.plans,
            scene_builder_cls=plan_data.dataset,
            spawn_data_fp=spawn_data_fp,
            require_build_configs_repeated_equally_across_envs=False,
        )
        self.env = MSHABEnv(self.env)
    
    def run(self, policy, seed=0):
        random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        device = policy.device
        dtype = policy.dtype

        all_success_rates = []
        env = self.env
        base_seed = 2024

        for episode in tqdm(range(self.eval_episodes), desc="Evaluating"):
            obs, _ = env.reset(seed=base_seed + episode)
            policy.reset()

            done = False

            while not done:
                obs_dict = dict_apply(obs, lambda x: torch.from_numpy(x).to(device=device))
                with torch.no_grad():
                    obs_dict['point_cloud'] = obs_dict['point_cloud'].unsqueeze(0)
                    obs_dict['agent_pos'] = obs_dict['agent_pos'].unsqueeze(0)

                    action_dict = policy.predict_action(obs_dict)
                
                action_dict = dict_apply(action_dict, lambda x: x.detach().cpu().numpy())
                action = action_dict['action'].squeeze(0)

                obs, reward, done, info = env.step(action)
            
            all_success_rates.append(info['success'])
            success_rate = np.mean(all_success_rates)

        cprint(f"Eval episodes: {self.eval_episodes} Success rate: {success_rate:.4f}", 'green')

        log_data = dict(
            success_rate=success_rate,
            eval_episodes=self.eval_episodes,
        )

        return log_data



            