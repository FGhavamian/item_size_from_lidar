from pathlib import Path
import argparse

import pandas as pd
import numpy as np
import open3d as o3d


THRESHOLD_DISTANCE = 10
THRESHOLD_CONF = 150
NB_NEIGHBORS = 20
STD_RATIO = 0.01


def read_data_from_csv(item_pcd_file_path, floor_pcd_file_path):
    item = pd.read_csv(item_pcd_file_path, index_col=0)
    floor = pd.read_csv(floor_pcd_file_path, index_col=0)

    return item, floor


def remove_low_confidence(df_pcd, threshold_conf):
    return df_pcd[df_pcd["conf"] > threshold_conf]


def remove_starting_points_in_z_dir(df_pcd):
    return df_pcd.query("z > 5")


def convert_to_pcd(np_pcd):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np_pcd)
    return pcd


def remove_floor(item_pcd, floor_pcd, threshold_distance):

    dists = item_pcd.compute_point_cloud_distance(floor_pcd)
    dists = np.asarray(dists)

    ind = np.where(dists > threshold_distance)[0]
    only_item_pcd = item_pcd.select_by_index(ind)

    ind = np.where(dists <= threshold_distance)[0]
    floor_pcd = item_pcd.select_by_index(ind)

    return only_item_pcd, floor_pcd


def remove_outliers(pcd, nb_neighbors, std_ratio):
    cl, ind = pcd.remove_statistical_outlier(
        nb_neighbors=nb_neighbors, std_ratio=std_ratio
    )

    inlier_cloud = pcd.select_by_index(ind)

    return inlier_cloud


def remove_outliers_dbscan(pcd, eps, min_points):
    labels = np.array(
        pcd.cluster_dbscan(eps=eps, min_points=min_points, print_progress=True)
    )

    inlier_cloud = pcd.select_by_index(np.where(labels != -1)[0])

    return inlier_cloud


def run(item_pcd_file_path, floor_pcd_file_path, visualize):

    # Lidar data is assumed to have been saved in a csv format.
    # This might not be the case during production where data is directly fed to the algorithm.
    # Floor data is however going to remain constant and can be read from a csv file
    item, floor = read_data_from_csv(item_pcd_file_path, floor_pcd_file_path)

    # the lidar device assigns a confidence level to each point. The density of these values
    # are mostly greater than 200.
    item = remove_low_confidence(item, THRESHOLD_CONF)
    floor = remove_low_confidence(floor, THRESHOLD_CONF)

    item = remove_starting_points_in_z_dir(item)
    floor = remove_starting_points_in_z_dir(floor)

    # This convertion is required to use open3d library
    item_pcd = convert_to_pcd(item[["x", "y", "z"]].values)
    floor_pcd = convert_to_pcd(floor[["x", "y", "z"]].values)

    # The idea to is that a part of the item point cloud is actually the floor.
    # We find those points that are close to the point cloud of the floor, and separate them.
    # If the distance of a point in the item point cloud to floor point cloud is smaller than threshold_distance,
    # then the point is actually the floor.
    # Through data analysis we realized that this value is better to be 10.
    item_pcd, floor_pcd = remove_floor(item_pcd, floor_pcd, THRESHOLD_DISTANCE)

    # There are points that are clearly outliers. We remove them using a statistical method (using z-score).

    item_pcd = remove_outliers(item_pcd, NB_NEIGHBORS, STD_RATIO)
    floor_pcd = remove_outliers(floor_pcd, NB_NEIGHBORS, STD_RATIO)

    # item_pcd = remove_outliers_dbscan(item_pcd, 100, 20)
    # floor_pcd = remove_outliers_dbscan(floor_pcd, 100, 20)

    # There are two ways to compute a bounding box. The simpler one assumes that the item is aligned with the x-y-z axis.
    # This method along side ensuring that in practice the items are actually aligned with x-y-z axis seems to be the robust one.
    bbox_item = item_pcd.get_axis_aligned_bounding_box()
    bbox_floor = floor_pcd.get_axis_aligned_bounding_box()

    # The height item's bounding box can be underestimated. This is due to the fact that lidar laser cannot read the bottom of the item.
    # To fix this underestimation, we make sure that the bottom of the item's bounding box is located on top of the floor.
    bbox_item.min_bound = (
        bbox_floor.max_bound[0],
        bbox_item.min_bound[1],
        bbox_item.min_bound[2],
    )

    if visualize:
        bbox_item.color = (1, 0, 0)
        bbox_floor.color = (0, 1, 0)

        item_pcd.paint_uniform_color([1, 0.1, 0.1])
        floor_pcd.paint_uniform_color([0.1, 1, 0.1])

        origin = bbox_floor.max_bound
        origin = (0, 0, 0)

        axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=50, origin=origin)

        o3d.visualization.draw_geometries(
            [item_pcd, floor_pcd, bbox_item, bbox_floor, axis]
        )

    return bbox_item


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("-i", "--item-file-path", help="Path to item point cloud")
    parser.add_argument("-f", "--floor-file-path", help="Path to floor point cloud")

    args = parser.parse_args()

    bbox_item = run(args.item_file_path, args.floor_file_path, visualize=True)

    size = (
        int(abs(bbox_item.max_bound[0] - bbox_item.min_bound[0])),
        int(abs(bbox_item.max_bound[1] - bbox_item.min_bound[1])),
        int(
            abs(bbox_item.max_bound[2] - bbox_item.min_bound[2]),
        ),
    )

    print(f"Item size: {size[0]} x {size[1]} x {size[2]} [mm]")
