import zarr
import numpy as np
import argparse
from termcolor import cprint

def main(args):
    try:
        # 打开 Zarr 文件（'a' 模式允许读写）
        zarr_file = zarr.open(args.zarr_path, mode='a')
        
        # 检查路径是否存在
        if 'data' not in zarr_file or 'point_cloud' not in zarr_file['data']:
            raise ValueError("Zarr file does not contain 'data/point_cloud'")
        
        # 获取点云数据（确保可写）
        point_cloud_data = zarr_file['data']['point_cloud']
        
        # 修改数据并强制写入
        point_cloud_data[:, :, 3:6] = point_cloud_data[:, :, 3:6] * 255.0
        test_pcd = zarr_file['data']['point_cloud'][0, 0, 3:6]

        cprint(f"Test point cloud data: {test_pcd}", 'green')
        
        print("Successfully updated point cloud data.")
    
    except Exception as e:
        print(f"Error processing Zarr file: {e}")
    
    finally:
        # 确保文件被关闭（Zarr 通常不需要显式关闭，但显式调用更安全）
        if 'zarr_file' in locals():
            zarr_file.store.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scale point cloud RGB values by 255.")
    parser.add_argument("zarr_path", type=str, help="Path to the zarr file.")
    args = parser.parse_args()
    main(args)