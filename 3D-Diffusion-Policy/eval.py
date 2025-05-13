if __name__ == "__main__":
    import sys
    import os
    import pathlib

    import argparse

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=400, help='Epoch to evaluate')
    args, unknown_args = parser.parse_known_args()
    
    # 将剩余参数传递给hydra
    sys.argv = [sys.argv[0]] + unknown_args


import os
import hydra
import torch
import dill
from omegaconf import OmegaConf
import pathlib
from train import TrainDP3Workspace

OmegaConf.register_new_resolver("eval", eval, replace=True)
    

@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.joinpath(
        'diffusion_policy_3d', 'config'))
)
def main(cfg):
    workspace = TrainDP3Workspace(cfg)
    workspace.eval(args.epoch)

if __name__ == "__main__":
    main()
