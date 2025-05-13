import os
import subprocess
from pathlib import Path
from termcolor import cprint
import argparse

def main(args):
    output_dir = Path('/data/jh/mshab/checkpoints')

    for exp in output_dir.iterdir():
        if not os.path.isdir(exp):
            continue
        cprint(exp.name, 'green')
        task, _, addition = str(exp.name).split('-')
        info, seed = addition.split('_')
        seed = seed[-1]

        if task != args.task or info != args.info:
            continue

        ckpts_dir = exp / "checkpoints"

        for ckpt_file in ckpts_dir.glob('*.ckpt'):
            epoch = str(ckpt_file.name).split('.')[0]

            cmd = ["bash", "scripts/eval_policy.sh", "dp3", task, info, seed, args.gpu, epoch]
            cprint(f"GPU id: {args.gpu} Epoch: {epoch}", "cyan")
            cprint(" ".join(cmd), "red")
            
            subprocess.run(cmd, check=True)
        


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('task', type=str)
    parser.add_argument('info', type=str)
    parser.add_argument('gpu', type=str)

    args = parser.parse_args()
    main(args)
