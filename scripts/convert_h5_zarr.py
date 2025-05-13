import argparse
import h5py
import zarr
import os
from termcolor import cprint
import numpy as np
from tqdm import tqdm
import shutil

def main(args):
    save_dir = args.zarr_path
    if os.path.exists(save_dir):
        cprint('Data already exists at {}'.format(save_dir), 'yellow')
        cprint('Do you want to overwrite it? (y/n)', 'yellow')
        user_input = input()
        if user_input.lower() == 'y':
            cprint('Overwriting data...', 'yellow')
            shutil.rmtree(save_dir)
        else:
            cprint('Exiting...', 'yellow')
            return
    
    os.makedirs(save_dir, exist_ok=True)

    total_steps = 0
    all_states = []
    all_actions = []
    all_pointclouds = []
    episode_ends = []

    with h5py.File(args.h5_path, 'r') as f:
        num_trajs = len(f.keys())
        cprint(f'Num of episodes: {num_trajs}', 'green')

        for i in tqdm(range(num_trajs)):
            traj = f[f'traj_{i}']
            success_idx = np.argmax(traj['success'])

            assert success_idx != 0, 'Failure trajectory'

            steps = success_idx + 1
            total_steps += steps
            episode_ends.append(total_steps)

            state = traj['obs']['agent']['qpos'][:steps]
            action = traj['actions'][:steps]
            pointcloud = traj['obs']['pointcloud'][:steps]
            
            all_states.append(state)
            all_actions.append(action)
            all_pointclouds.append(pointcloud)
    
    zarr_root = zarr.group(save_dir)
    zarr_data = zarr_root.create_group('data')
    zarr_meta = zarr_root.create_group('meta')

    states = np.vstack(all_states)
    actions = np.vstack(all_actions)
    pointclouds = np.vstack(all_pointclouds)
    episode_ends = np.array(episode_ends)

    compressor = zarr.Blosc(cname='zstd', clevel=3, shuffle=1)
    states_chunk_size = (100, states.shape[1])
    actions_chunk_size = (100, actions.shape[1])
    pointclouds_chunk_size = (100, pointclouds.shape[1], pointclouds.shape[2])

    zarr_data.create_dataset('state', data=states, chunks=states_chunk_size, shape=states.shape, dtype=states.dtype, overwrite=True, compressor=compressor)
    zarr_data.create_dataset('action', data=actions, chunks=actions_chunk_size, shape=actions.shape, dtype=actions.dtype, overwrite=True, compressor=compressor)
    zarr_data.create_dataset('point_cloud', data=pointclouds, chunks=pointclouds_chunk_size, shape=pointclouds.shape, dtype=pointclouds.dtype, overwrite=True, compressor=compressor)
    zarr_meta.create_dataset('episode_ends', data=episode_ends, dtype=episode_ends.dtype, overwrite=True, compressor=compressor)
    
    cprint(f'-' * 50, 'cyan')
    cprint(f'point_cloud shape: {pointclouds.shape}','green')
    cprint(f'state shape: {states.shape}', 'green')
    cprint(f'action shape: {actions.shape}', 'green')
    cprint(f'Saved zarr file to {save_dir}', 'green')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--h5_path', type=str)
    parser.add_argument('--zarr_path', type=str)
    args = parser.parse_args()
    main(args)


