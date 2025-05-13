import wandb
import numpy as np
import torch
from tqdm import tqdm
import gymnasium as gym
import pathlib
import os
import random
from termcolor import cprint
from diffusion_policy_3d.env.mshab_wrapper import MSHABEnv
from diffusion_policy_3d.policy.base_policy import BasePolicy
from diffusion_policy_3d.common.utils import dict_apply
from diffusion_policy_3d.env_runner.base_runner import BaseRunner
import diffusion_policy_3d.common.logger_util as logger_util
from diffusion_policy_3d.wrapper.multistep_wrapper import MultiStepWrapper
from diffusion_policy_3d.wrapper.observation_wrapper import FlattenPoindCloudObservationWrapper
from mshab.envs.planner import plan_data_from_file
from collections import deque
import imageio

class MSHABRunner(BaseRunner):
    def __init__(
            self,
            task_name,
            eval_episodes,
            output_dir,
            n_obs_steps,
            n_action_steps,
            max_episode_steps,
            **kwargs,
            ):
        super().__init__(output_dir)

        self.output_dir = output_dir
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
            robot_uids="fetch",
            obs_mode="rgbd",
            control_mode="pd_joint_delta_pos",
            sensor_configs=dict(shader_pack="default"),
            sim_backend="auto",
            max_episode_steps=max_episode_steps,
            task_plans=plan_data.plans,
            scene_builder_cls=plan_data.dataset,
            spawn_data_fp=spawn_data_fp,
            require_build_configs_repeated_equally_across_envs=False,
        )
        
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.max_episode_steps = max_episode_steps
        cprint(f"[Env]: n_obs_steps: {n_obs_steps} n_action_steps: {n_action_steps} max_episode_steps: {max_episode_steps}", 'yellow')
        
    
    def run(self, policy, seed=0, epoch=None):
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
        base_seed = 2024
        
        debug = False


        for episode in tqdm(range(self.eval_episodes), desc="Evaluating"):
            if debug:
                video_frames_head = []
                video_frames_hand = []
        
            state_window = deque(maxlen=self.n_obs_steps)
            pcd_window = deque(maxlen=self.n_obs_steps)

            obs, _ = self.env.reset(seed = base_seed + episode)
            
            for _ in range(self.n_obs_steps):
                state = torch.cat([
                                    obs['agent']['qpos'], obs['extra']['obj_pose_wrt_base'], 
                                    obs['extra']['tcp_pose_wrt_base'], obs['extra']['goal_pos_wrt_base'],
                                    obs['extra']['is_grasped'][:, None]
                                    ], dim=1).squeeze(0)
                pointcloud = obs['pointcloud'].squeeze(0)
                
                state_window.append(state)
                pcd_window.append(pointcloud)

            obs = dict(
                agent_pos=torch.stack(list(state_window), dim=0).unsqueeze(0).to(device=device),
                point_cloud=torch.stack(list(pcd_window), dim=0).unsqueeze(0).to(device=device),
            )
            

            policy.reset()
            

            step = 0
            done = False
            success = 0
            while not done and step <= self.max_episode_steps:
                with torch.no_grad():
                    action_dict = policy.predict_action(obs)
                action_dict = dict_apply(action_dict, lambda x: x.detach().cpu().numpy())
                actions = action_dict['action'].squeeze(0)
                for action in actions:
                    obs, reward, terminated, truncated, info = self.env.step(action)
                    step += 1
                    
                    if debug:
                        head_img = obs['sensor_data']['fetch_head']['rgb'].squeeze(0).cpu().numpy()
                        hand_img = obs['sensor_data']['fetch_hand']['rgb'].squeeze(0).cpu().numpy()
                        
                        video_frames_head.append(head_img)
                        video_frames_hand.append(hand_img)
                    
                    state = torch.cat([
                                        obs['agent']['qpos'], obs['extra']['obj_pose_wrt_base'], 
                                        obs['extra']['tcp_pose_wrt_base'], obs['extra']['goal_pos_wrt_base'],
                                        obs['extra']['is_grasped'][:, None]
                                        ], dim=1)
                                        
                    state_window.append(state.squeeze(0))
                    pcd_window.append(obs['pointcloud'].squeeze(0))

                    obs = dict(
                        agent_pos=torch.stack(list(state_window), dim=0).unsqueeze(0).to(device=device),
                        point_cloud=torch.stack(list(pcd_window), dim=0).unsqueeze(0).to(device=device),
                    )
                    if info['success']:
                        success = 1
                        done = True
                        break
                    if terminated or truncated:
                        done = True
                        break
            
            cprint('success', 'green') if success else cprint('fail', 'green')
            cprint(step, 'green')

            if debug:
                with imageio.get_writer(f'debug_video_{episode}.mp4', mode='I', fps=20) as writer:
                    for frame in zip(video_frames_head, video_frames_hand):
                        head_img = frame[0]
                        hand_img = frame[1]
                        combined_img = np.concatenate([head_img, hand_img], axis=1)
                        writer.append_data(combined_img)

            all_success_rates.append(success) 
        
        test_mean_score = np.mean(all_success_rates)
        
        if epoch is not None:
            with open(os.path.join(self.output_dir, 'eval_result.txt'), 'a') as f:
                f.write(f"eval_episodes: {self.eval_episodes} SN: {np.sum(all_success_rates)} Epoch:{epoch}\n")
                cprint(f'eval res appendeded', 'green')
        

        log_data = dict(
            test_mean_score=test_mean_score,
        )

        return log_data



            
