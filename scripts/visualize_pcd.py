import zarr
import argparse
import visualizer
from termcolor import cprint

def main(args):
    # Load the zarr file
    zarr_path = args.zarr_path
    zarr_file = zarr.open(zarr_path, mode='r')

    # Extract the point cloud data
    point_cloud_data = zarr_file['data']['point_cloud'][4]
    cprint(point_cloud_data[0], 'green')
    # print(point_cloud_data[0, 3:6])
      
    # point_cloud_data[:, 3:6] = point_cloud_data[:, 3:6] * 255
    print(f"Point cloud data shape: {point_cloud_data.shape}, dtype: {point_cloud_data.dtype}")
    visualizer.visualize_pointcloud(point_cloud_data)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize point cloud data from a zarr file.")
    parser.add_argument("zarr_path", type=str, help="Path to the zarr file containing point cloud data.")
    args = parser.parse_args()

    main(args)
