import open3d as o3d
import numpy as np
import os
import sys
import struct


def read_point_cloud(file):
    pcd = o3d.io.read_point_cloud(file)
    print(pcd)
    print(np.asarray(pcd.points))

    return pcd


def visualize_point_cloud(pcd):
    o3d.visualization.draw_geometries([pcd],
                                  zoom=0.6412,
                                  front=[0.4257, -0.2125, -0.8795],
                                  lookat=[2.6172, 2.0475, 1.532],
                                  up=[-0.0694, -0.9768, 0.2024])


def convert_bin_to_array(file):

    size_float = 4
    list_pcd = []
    with open (file, "rb") as f:
        byte = f.read(size_float*4)
        while byte:
            x,y,z,intensity = struct.unpack("ffff", byte)
            list_pcd.append([x, y, z])
            byte = f.read(size_float*4)
    np_pcd = np.asarray(list_pcd)
    pcd = o3d.geometry.PointCloud()
    v3d = o3d.utility.Vector3dVector
    pcd.points = v3d(np_pcd)

    #o3d.io.write_point_cloud("pc_kitti.pcd", pcd)

    return pcd


def draw_3d_bounding_box(pcd):
    aabb = pcd.get_axis_aligned_bounding_box()
    aabb.color = (1, 0, 0)
    obb = pcd.get_oriented_bounding_box()
    obb.color = (0, 1, 0)
    o3d.visualization.draw_geometries([pcd, aabb, obb],
                                  zoom=0.7,
                                  front=[0.5439, -0.2333, -0.8060],
                                  lookat=[2.4615, 2.1331, 1.338],
                                  up=[-0.1781, -0.9708, 0.1608])

    return aabb, obb

def find_center(pcd):
    center = pcd.get_center()
    print(center)
    return center

def dist_btw_2geometries(a, b):
    dist = np.linalg.norm(a - b)
    print(dist)
    return dist


if __name__ == "__main__":
    pcd = convert_bin_to_array("data/000000.bin")
    visualize_point_cloud(pcd)
    aabb, obb = draw_3d_bounding_box(pcd)

    print("Center of the Point Cloud:")
    pcd_c = find_center(pcd)


    print("Center of the Axis Oriented Bounding Box:")
    aabb_c = find_center(aabb)


    print("Center of the Oriented Bounding Box:")
    obb_c = find_center(obb)

    print("\n\nDistance between the centers of PointCloud and Axis Oriented Bounding Box:")
    dist1 = dist_btw_2geometries(pcd_c, aabb_c)

    print("Distance between the centers of PointCloud and Oriented Bounding Box:")
    dist1 = dist_btw_2geometries(pcd_c, obb_c)