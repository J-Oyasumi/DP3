import os
import subprocess
from pathlib import Path
from termcolor import cprint

def main():
    task_dir = Path("/data/jh/mshab")
    
    for subtask in task_dir.iterdir():
        print(subtask.name)
        if not subtask.is_dir():
            continue
        for zarr in subtask.iterdir():
            if not zarr.is_dir():
                continue
            cmd = ["python", "-m", "scripts.process_zarr", str(zarr)]
            cprint(" ".join(cmd), "red")
            # 调用 Python 脚本生成数据
            subprocess.run(cmd, check=True)

if __name__ == "__main__":
    main()
