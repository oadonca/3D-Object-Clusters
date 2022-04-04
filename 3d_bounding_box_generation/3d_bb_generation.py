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


def find_center_faces(bb, bb_c):

    """
    returns a numpy array with six coordinates of different cntres of the faces of the bounding box 
    """

    # returns the eight bounding box coordinates of the corners; not used anywhere
    bb_coord = bb.get_box_points()

    # returns the length, breadth and height divided by 2 of the bounding box 
    half_extent = bb.get_half_extent()

    # empty numpy array initializaion 
    arr = np.empty((0,3), int)


    # formula used to calculate the centres of the six faces is = centroid +/- (lenghth/2 or breadth/2 or height/2)
    arr = np.append(arr, np.array([[bb_c[0] + half_extent[0], bb_c[1], bb_c[2]]]), axis=0)
    arr = np.append(arr, np.array([[bb_c[0] - half_extent[0], bb_c[1], bb_c[2]]]), axis=0)

    arr = np.append(arr, np.array([[bb_c[0], bb_c[1] + half_extent[1], bb_c[2]]]), axis=0)
    arr = np.append(arr, np.array([[bb_c[0], bb_c[1] - half_extent[1], bb_c[2]]]), axis=0)

    arr = np.append(arr, np.array([[bb_c[0], bb_c[1], bb_c[2] + half_extent[2]]]), axis=0)
    arr = np.append(arr, np.array([[bb_c[0], bb_c[1], bb_c[2] - half_extent[2]]]), axis=0)

    print(arr)
    return arr



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


    print("Center of all the faces of the bounding box:")
    bb_faces_c = find_center_faces(aabb, aabb_c)