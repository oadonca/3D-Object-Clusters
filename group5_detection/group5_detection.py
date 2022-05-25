from cgi import test
from operator import truediv
import os
import time
import matplotlib.pyplot as plt
from numpy import true_divide
import open3d
import itertools
import collections
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon
from matplotlib import patches
import multiprocessing as mp
import json
import argparse
import csv


from utils import *
from iou3d import get_3d_box, box3d_iou

def render_image_with_boxes(img, objects, calib):
    """
    Show image with 3D boxes
    """
    # projection matrix
    P_rect2cam2 = calib['P2'].reshape((3, 4))

    img1 = np.copy(img)
    for obj in objects:
        if obj.type == 'DontCare':
            continue
        box3d_pixelcoord = map_box_to_image(obj, P_rect2cam2)
        img1 = draw_projected_box3d(img1, box3d_pixelcoord)

    plt.imshow(img1)
    plt.yticks([])
    plt.xticks([])
    plt.show()


def render_lidar_with_boxes(pc_velo, objects, calib, img_width, img_height, dbscan=False):
    # projection matrix (project from velo2cam2)
    proj_velo2cam2 = project_velo_to_cam2(calib)

    # apply projection
    pts_2d = project_to_image(pc_velo.transpose(), proj_velo2cam2)

    # Filter lidar points to be within image FOV
    inds = np.where((pts_2d[0, :] < img_width) & (pts_2d[0, :] >= 0) &
                    (pts_2d[1, :] < img_height) & (pts_2d[1, :] >= 0) &
                    (pc_velo[:, 0] > 0)
                    )[0]
    imgfov_pc_velo = pc_velo[inds, :]

    # create open3d point cloud and axis
    mesh_frame = open3d.geometry.TriangleMesh.create_coordinate_frame(size=2, origin=[0, 0, 0])
    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(imgfov_pc_velo)
    
    if dbscan:
        # Run DBSCAN on point cloud
        with open3d.utility.VerbosityContextManager(open3d.utility.VerbosityLevel.Debug) as cm:
            labels = np.array(pcd.cluster_dbscan(eps=1, min_points=20, print_progress=True))

        # Set colors of point cloud to see DBSCAN clusters
        max_label = labels.max()
        print(f"point cloud has {max_label + 1} clusters")
        colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
        colors[labels < 0] = 0
        pcd.colors = open3d.utility.Vector3dVector(colors[:, :3])
    
    entities_to_draw = [pcd, mesh_frame]

    # Projection matrix
    proj_cam2_2_velo = project_cam2_to_velo(calib)

    # Draw objects on lidar
    for obj in objects:
        if obj.type == 'DontCare':
            continue

        # Project boxes from camera to lidar coordinate
        boxes3d_pts = project_camera_to_lidar(obj.in_camera_coordinate(), proj_cam2_2_velo)

        # Open3d boxes
        boxes3d_pts = open3d.utility.Vector3dVector(boxes3d_pts.T)
        box = open3d.geometry.OrientedBoundingBox.create_from_points(boxes3d_pts)
        box.color = [1, 0, 0]
        entities_to_draw.append(box)

    # Draw
    open3d.visualization.draw_geometries([*entities_to_draw],
                                         front=[-0.9945, 0.03873, 0.0970],
                                         lookat=[38.4120, 0.6139, 0.48500],
                                         up=[0.095457, -0.0421, 0.99453],
                                         zoom=0.33799
                                         )


def render_lidar_on_image(pts_velo, orig_img, calib, img_width, img_height, depth_limit = 40, visualize=True):
    """
    Projects the given lidar points onto the given image using the projection/transformation matrices provided in calib
    
    Returns: Image and projected points
    """
    
    img = np.copy(orig_img)
    # projection matrix (project from velo2cam2)
    proj_velo2cam2 = project_velo_to_cam2(calib)

    # apply projection
    pts_2d = project_to_image(pts_velo.transpose(), proj_velo2cam2)
    # Can extrapole this process to remove lidar points that are outside of 2D bounding boxes
    # Filter lidar points to be within image FOV
    inds = np.where((pts_2d[0, :] < img_width) & (pts_2d[0, :] >= 0) &
                    (pts_2d[1, :] < img_height) & (pts_2d[1, :] >= 0) &
                    (pts_velo[:, 0] > 0)
                    )[0]

    # Filter out pixels points
    imgfov_pc_pixel = pts_2d[:, inds]
    
    # Retrieve depth from lidar
    imgfov_pc_velo = pts_velo[inds, :]
    
    # make homoegenous
    imgfov_pc_velo = np.hstack((imgfov_pc_velo, np.ones((imgfov_pc_velo.shape[0], 1))))

    # Project lidar points onto image
    imgfov_pc_cam2 = proj_velo2cam2 @ imgfov_pc_velo.transpose()
    
    # Turn lidar into 2D array
    projected_grid = np.zeros_like(img).astype(float)
    for point in np.transpose(imgfov_pc_cam2):
        if point[2] < depth_limit:
            x = point[0]/point[2]
            y = point[1]/point[2]
            projected_grid[int(y), int(x)] = point

    if visualize:
        cmap = plt.cm.get_cmap('hsv', 256)
        cmap = np.array([cmap(i) for i in range(256)])[:, :3] * 255
        # Display projected point cloud on the image
        for i in range(imgfov_pc_pixel.shape[1]):
            depth = imgfov_pc_cam2[2, i]
            color = cmap[min(int(640.0 / depth), 255), :]
            cv2.circle(img, (int(np.round(imgfov_pc_cam2[0, i]/imgfov_pc_cam2[2, i])),
                            int(np.round(imgfov_pc_cam2[1, i]/imgfov_pc_cam2[2, i]))),
                    2, color=tuple(color), thickness=-1)
            
        fig, ax = plt.subplots(2, 1, figsize=(5, 10))

        ax[0].set_title('Original Image')
        ax[0].imshow(orig_img)

        ax[1].set_title('Image w/ Projected Points')
        ax[1].imshow(img)

        plt.yticks([])
        plt.xticks([])
        plt.savefig('render_lidar_on_image.png')
        
    return imgfov_pc_cam2, projected_grid


def segment_bb_frustum_from_projected_pcd(detections, projected_points, projected_grid, calib, labels=[], visualize=False, orig_img = None):
    """
    Given a np array of projected 2D points and a list of bounding boxes, 
    Removes all points outside the frustums generated by the provided bounding boxes
    
    Returns: A list of pointclouds for each segmented frustum
    """
    # Get image to cam 2 frame transform
    image_transform = project_image_to_cam2(calib)
    
    # Get only the points within bounding boxes
    # Only need this for only bounding box segmentation, not required for mask propagation
    # inds = within_bb_indices(detections, projected_points/projected_points[2,:])
    bb_inds = within_bb_indices(detections, projected_points/projected_points[2,:])

    # Non parallel
    for i, detection in enumerate(detections): 
        # mask propagation
        if 'mask' in detection.keys() and detection['mask'] is not None:
            # Get indicies where detection mask is true and projected points is non zero
            mask_inds = np.where(((detection['mask'][:, :] == True) & ((projected_grid[:,:,0] > 0) | (projected_grid[:,:,1] > 0) | (projected_grid[:,:,2] > 0))))
            object_pts_2d = np.transpose(projected_grid[mask_inds])

        # bounding box
        elif 'bb' in detection.keys() and detection['bb'] is not None:
            object_pts_2d = projected_points[:, bb_inds[i]]

        else:
            print('ERROR: no bounding box or mask detected')
            raise NotImplementedError
        
        if object_pts_2d.shape[0] == 3 and object_pts_2d.shape[1] > 0:
            cam_coords = image_transform[:3, :3] @ object_pts_2d

            # Get projection from camera coordinates to velodyne coordinates
            proj_mat = project_cam2_to_velo(calib)
            
            # apply projection
            velo_points = project_camera_to_lidar(cam_coords, proj_mat)
        
            detections[i]['frustum_pcd'] = velo_points
        else:
            detections[i]['frustum_pcd'] = None
    
    if visualize:
        o3d_pcd_list = []
        for detection in detections:
            pcd = open3d.geometry.PointCloud()
            pcd.points = open3d.utility.Vector3dVector(np.transpose(detection['frustum_pcd']))
            o3d_pcd_list.append(pcd)

    if visualize:
        open3d.visualization.draw_geometries(o3d_pcd_list + labels,
                                         front=[-0.9945, 0.03873, 0.0970],  
                                         lookat=[38.4120, 0.6139, 0.48500],
                                         up=[0.095457, -0.0421, 0.99453],
                                         zoom=0.33799
                                         )
        
    return [detection['frustum_pcd'] for detection in detections]

def remove_ground(pointcloud, labels=[], removal_offset = 0, visualize=False):
    """
    Removes ground points from provided pointcloud
    
    Returns: Pointcloud
    """
    
    # Create Open3D pointcloud
    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(pointcloud)
    
    # Run RANSAC
    model, inliers = pcd.segment_plane(distance_threshold=0.12,ransac_n=3, num_iterations=100)

    # Get the average inlier coorindate values
    average_inlier = np.mean(pointcloud[inliers], axis=0)

    # Remove inliers
    segmented_pointcloud = np.delete(pointcloud, inliers, axis=0)
        
    # Remove points below average inlier z value
    mask = np.argwhere(segmented_pointcloud[:, 2] < average_inlier[2]+removal_offset)
    segmented_pointcloud = np.delete(segmented_pointcloud, mask, axis=0)
    
    if visualize:
        # Visualize
        inlier_cloud = pcd.select_by_index(inliers) # use select_by_index() depending on version of open3d
        inlier_cloud.paint_uniform_color([1.0, 0, 0])
        outlier_cloud = pcd.select_by_index(inliers, invert=True) # same as above
        outlier_cloud.paint_uniform_color([0, 0, 1.0])
        open3d.visualization.draw_geometries([inlier_cloud, outlier_cloud] + labels,
                                    zoom=0.8,
                                    front=[-0.4999, -0.1659, -0.8499],
                                    lookat=[2.1813, 2.0619, 2.0999],
                                    up=[0.1204, -0.9852, 0.1215])
    
    return np.reshape(segmented_pointcloud, (-1, 3))


def get_cluster_scores(cluster_list, detection, weights = [1.0, 1.0, 1.0, 1.0], use_autodrive_classes = False):
    
    proj_velo2cam2 = project_velo_to_cam2(detection['calib'])
    cluster_losses = []
    for i, cluster in enumerate(cluster_list):
        loss_list = []
        if use_autodrive_classes:
            obj_list = get_autodrive_classes()
        else:
            obj_list = get_used_coco_classes()
        # Compare 3D extent of cluster vs car, truck, pedestrian avg size
        avg_volume = []
        
        # Compare 3D proportions vs average proportions
        ############################################################
        # Average height
        ############################################################
        avg_height = [1.64592, .856, 2, 4.1148, 2, 4.1148, 4.572, 1.27, 1.8, 1.4, 1, 4]
        class_idx = obj_list.index(get_coco_class(detection['class']))
        
        cluster_3d_bb = cluster.get_axis_aligned_bounding_box()
        cluster_height = cluster_3d_bb.get_half_extent()[2]
        
        loss = weights[0]*(cluster_height-avg_height[class_idx])**2
        loss_list.append(loss)
        
        ############################################################
        # 2D extent
        ############################################################
        cluster_points = np.array(cluster.points)
        
        # apply projection
        pts_2d = project_to_image(cluster_points.transpose(), proj_velo2cam2)

        # Filter out pixels points
        imgfov_pc_pixel = pts_2d
        
        # Retrieve depth from lidar
        imgfov_pc_velo = cluster_points
        
        # make homoegenous
        imgfov_pc_velo = np.hstack((imgfov_pc_velo, np.ones((imgfov_pc_velo.shape[0], 1))))

        # Project lidar points onto image
        imgfov_pc_cam2 = proj_velo2cam2 @ imgfov_pc_velo.transpose()
        
        # Turn lidar into 2D array
        min_x = float('inf')
        max_x = -float('inf')
        min_y = float('inf')
        max_y = -float('inf')
        for point in np.transpose(imgfov_pc_cam2):
            x = point[0]/point[2]
            y = point[1]/point[2]
            if x < min_x: min_x = x
            elif x > max_x: max_x = x
            if y < min_y: min_y = y
            elif y > max_y: max_y = y 
            
        cluster_extent_x = max_x - min_x
        cluster_extent_y = max_y - min_y
        
        bb_extent_x = detection['bb'][1][0] - detection['bb'][0][0]
        bb_extent_y = detection['bb'][2][1] - detection['bb'][0][1]
        
        IoU = (cluster_extent_x*cluster_extent_y)/(bb_extent_x*bb_extent_y)
        
        cluster_extent_score=weights[1]*1/IoU**2
        loss_list.append(cluster_extent_score)
        
        # Cluster centroid distances relative to % of 2D extent
        
        # Cluster centroid distance from frustum centroid
        # Cluster centroid distance from car

        cluster_losses.append(np.sum(np.array(loss_list)))
    
    return cluster_losses


def apply_dbscan(pointcloud, detection, keep_n=5, visualize=True):
    """
    Applies DBSCAN on the provided point cloud and returns the Open3D point cloud
    """
    
    # Create Open3D point cloud
    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(np.transpose(pointcloud))
        
    # Run DBSCAN on point cloud
    # with open3d.utility.VerbosityContextManager(open3d.utility.VerbosityLevel.Debug) as cm:
    labels = np.array(pcd.cluster_dbscan(eps=1, min_points=10))

    # Set colors of point cloud to see DBSCAN clusters
    max_label = labels.max()
    # print(f"point cloud has {max_label + 1} clusters")
    colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
    colors[labels < 0] = 0
    pcd.colors = open3d.utility.Vector3dVector(colors[:, :3])
    counts = collections.Counter(labels)
        
    # Get top n clusters
    pcd_list = []
    for label, count in counts.most_common(keep_n):
        if not label < 0:
            pcd_list.append(pcd.select_by_index(np.argwhere(labels==label))) # select_by_index() instead

    if len(pcd_list) > 1:
        cluster_losses = get_cluster_scores(pcd_list, detection)
        object_candidate_cluster_idx = np.argmin(cluster_losses)
        pcd_list = [pcd_list[object_candidate_cluster_idx]]
        
    if visualize:
        # Visualize
        open3d.visualization.draw_geometries([pcd])
    
    return pcd_list

def generate_3d_bb(pcd_list, detections, oriented=False, visualize=False):
    """
    Generates bounding boxes around each point cloud in pcd_list
    """
    generated_bb_list = []
    for i, pcd in enumerate(pcd_list):
        if detections[i]['object_candidate_cluster'] is not None and np.max(np.array(pcd.colors)) > 0:
            bb = pcd.get_axis_aligned_bounding_box() if not oriented else pcd.get_oriented_bounding_box()
            bb.color = [1.0, 0, 0]
            generated_bb_list.append(bb)
            detections[i]['generated_3d_bb'] = bb
        else:
            detections[i]['generated_3d_bb'] = None
        
    display_list = pcd_list + generated_bb_list
        
    if visualize:
        open3d.visualization.draw_geometries(display_list)
        
    return generated_bb_list


def detection_analysis(detections, labels):

    analysis_metrics = dict()

    print('='*50)
    analysis_metrics['total_detections']  = 0
    analysis_metrics['total_correct_detections'] = 0
    for i, kitti_3d_bb in enumerate(labels['kitti_gt_3d_bb']):
        print('.'*100)
        analysis_metrics[f'box_{i}'] = dict()
        analysis_metrics[f'box_{i}']['kitti_bb_class'] = labels['kitti_gt_labels'][i].type
        analysis_metrics[f'box_{i}']['kitti_3d_bb_center'] = kitti_3d_bb.get_center()
        analysis_metrics[f'box_{i}']['kitti_3d_bb_face_centers'] = get_bb_centers(kitti_3d_bb)
        
        # Get the closest box
        analysis_metrics[f'box_{i}']['closest_generated_box'] = None
        analysis_metrics[f'box_{i}']['closest_generated_box_coco_class'] = None
        analysis_metrics[f'box_{i}']['closest_generated_center'] = None
        analysis_metrics[f'box_{i}']['closest_center_distance'] = float('inf')
        for j, detection in enumerate(detections):

            if 'generated_3d_bb' in detection.keys() and detection['generated_3d_bb'] is not None:
                detection_coco_class = get_coco_class(detection['class'])
                generated_3d_bb_center = detection['generated_3d_bb'].get_center()
                distance = np.linalg.norm(analysis_metrics[f'box_{i}']['kitti_3d_bb_center'] - generated_3d_bb_center)

                if detection_coco_class in kitti_coco_class_mapping(analysis_metrics[f'box_{i}']['kitti_bb_class']) and distance < analysis_metrics[f'box_{i}']['closest_center_distance']:
                    analysis_metrics[f'box_{i}']['closest_center_distance'] = distance
                    analysis_metrics[f'box_{i}']['closest_generated_box'] = detection['generated_3d_bb']
                    analysis_metrics[f'box_{i}']['closest_generated_box_coco_class'] = detection_coco_class
                    analysis_metrics[f'box_{i}']['closest_generated_center'] = generated_3d_bb_center
                
        # Calculate face centers for the KITTI GT box
        analysis_metrics[f'box_{i}']['kitti_3d_bb_closest_face_center'] = None
        analysis_metrics[f'box_{i}']['closest_gt_face_center_distance'] = float('inf')
        for face_center in analysis_metrics[f'box_{i}']['kitti_3d_bb_face_centers']:

            distance = np.linalg.norm(face_center - np.array([0, 0, 0]))

            if distance < analysis_metrics[f'box_{i}']['closest_gt_face_center_distance']:
                analysis_metrics[f'box_{i}']['closest_gt_face_center_distance'] = distance
                analysis_metrics[f'box_{i}']['kitti_3d_bb_closest_face_center'] = face_center
                
        # Calculate face centers for the closest generated box
        analysis_metrics[f'box_{i}']['closest_box_face_centers'] = None
        analysis_metrics[f'box_{i}']['closest_box_closest_face_center'] = None
        analysis_metrics[f'box_{i}']['closest_box_face_center_distance'] = float('inf')
        analysis_metrics[f'box_{i}']['closest_face_and_gt_distance'] = float('inf')
        analysis_metrics[f'box_{i}']['IOU_3d'] = 0.0
        analysis_metrics[f'box_{i}']['IOU_2d'] = 0.0
        if analysis_metrics[f'box_{i}']['closest_generated_box'] is not None:
            analysis_metrics[f'box_{i}']['closest_box_face_centers'] = get_bb_centers(analysis_metrics[f'box_{i}']['closest_generated_box'])
            for face_center in analysis_metrics[f'box_{i}']['closest_box_face_centers']:

                distance = np.linalg.norm(face_center - np.array([0, 0, 0]))

                if distance < analysis_metrics[f'box_{i}']['closest_box_face_center_distance']:
                    analysis_metrics[f'box_{i}']['closest_box_face_center_distance'] = distance
                    analysis_metrics[f'box_{i}']['closest_box_closest_face_center'] = face_center
                    
            analysis_metrics[f'box_{i}']['closest_face_and_gt_distance'] = np.linalg.norm(analysis_metrics[f'box_{i}']['closest_box_closest_face_center'] - analysis_metrics[f'box_{i}']['kitti_3d_bb_closest_face_center'])
            
            # Calculating 3D IOU
            kitti_gt_corners = np.asarray(kitti_3d_bb.get_box_points())
            generated_3d_corners = np.asarray(analysis_metrics[f'box_{i}']['closest_generated_box'].get_box_points())

            # Convert to correct order for IoU
            kitti_gt_corners = np.array([kitti_gt_corners[4], kitti_gt_corners[7], kitti_gt_corners[2], kitti_gt_corners[5], kitti_gt_corners[6], kitti_gt_corners[1], kitti_gt_corners[0], kitti_gt_corners[3]])
            generated_3d_corners = np.array([generated_3d_corners[4], generated_3d_corners[7], generated_3d_corners[2], generated_3d_corners[5], generated_3d_corners[6], generated_3d_corners[1], generated_3d_corners[0], generated_3d_corners[3]])

            (analysis_metrics[f'box_{i}']['IOU_3d'],analysis_metrics[f'box_{i}']['IOU_2d'])=box3d_iou(generated_3d_corners,kitti_gt_corners)

        # Determine if detection is correct
        analysis_metrics[f'box_{i}']['correct_detection'] = False
        if analysis_metrics[f'box_{i}']['closest_face_and_gt_distance'] is not None and analysis_metrics[f'box_{i}']['closest_face_and_gt_distance'] < 2 and analysis_metrics[f'box_{i}']['closest_generated_box_coco_class'] in kitti_coco_class_mapping(analysis_metrics[f'box_{i}']['kitti_bb_class']):
            analysis_metrics[f'box_{i}']['correct_detection'] = True

        if analysis_metrics[f'box_{i}']['correct_detection']:
            analysis_metrics['total_correct_detections'] += 1
        analysis_metrics['total_detections']  += 1

        print(f'{f"KITTI GT 3D BB #{i} class: ":<50} {analysis_metrics[f"box_{i}"]["kitti_bb_class"]}')        
        print(f'{f"CLOSEST GENERATED 3D BB coco class: ":<50} {analysis_metrics[f"box_{i}"]["closest_generated_box_coco_class"]}')
        print(f'{f"KITTI GT 3D BB #{i} center: ":<50} {analysis_metrics[f"box_{i}"]["kitti_3d_bb_center"]} m.')
        print(f'{f"CLOSEST GENERATED 3D BB center: ":<50} {analysis_metrics[f"box_{i}"]["closest_generated_center"]} m.\n')
        print(f'{"Smallest distance between bb centers: ":<50} {analysis_metrics[f"box_{i}"]["closest_center_distance"]:.4f} m.')
        print(f'{"Distance between closest face centers:":<50} {analysis_metrics[f"box_{i}"]["closest_face_and_gt_distance"]:.4f} m.')
        print(f'{"KITTI GT Closest face center: ":<50} {analysis_metrics[f"box_{i}"]["kitti_3d_bb_closest_face_center"]}')
        print(f'{"GENERATED Closest face center: ":<50} {analysis_metrics[f"box_{i}"]["closest_box_closest_face_center"]}\n')
        print(f'{"Closest Box 3D IoU: ":<50} {analysis_metrics[f"box_{i}"]["IOU_3d"]:.4f}')
        print(f'{"Closest Box 2D IoU: ":<50} {analysis_metrics[f"box_{i}"]["IOU_2d"]:.4f}\n')
        print(f'{"Detection Correct Class and Within +-2m? ":<50} {analysis_metrics[f"box_{i}"]["correct_detection"]}')
        print('.'*100)

    print(f'{"Total Correct Detections: ":<50} {analysis_metrics["total_correct_detections"]} Correct/{analysis_metrics["total_detections"]} Total\n')
    print('='*50)

    return analysis_metrics

def compare_generation_accuracy(comp_list, visualize=False):    
    # comp format: [point cloud, ground truth bb, generated bb]
    for i, comp in enumerate(comp_list):
        centers = [item.get_center() for item in comp]

        pointcloud_to_groundtruth_center_distance = np.linalg.norm(np.array(centers[1]) - np.array(centers[0]))
        generated_to_groundtruth_center_distance = np.linalg.norm(np.array(centers[1]) - np.array(centers[2]))
        groundtruth_to_lidar_distance = np.linalg.norm(np.array(centers[1]) - np.array([0, 0, 0]))

        print(f'Item {i+1} {"Object Pointcloud Center to Groundtruth BB Distance:":<50} {pointcloud_to_groundtruth_center_distance:>10.4f} m')
        print(f'Item {i+1} {"Generated BB Center to Groundtruth BB Distance:":<50} {generated_to_groundtruth_center_distance:>10.4f} m')
        print(f'Item {i+1} {"Groundtruth BB Center to LiDAR Distance:":<50} {groundtruth_to_lidar_distance:>10.4f} m')

        # Point cloud
        pointcloud_center = open3d.geometry.PointCloud()
        pointcloud_center.points = open3d.utility.Vector3dVector(np.array([centers[0]]))
        pointcloud_center.paint_uniform_color([0, 0, 1.0]) 
        
        # Generated BB
        generated_center = open3d.geometry.PointCloud()
        generated_center.points = open3d.utility.Vector3dVector(np.array([centers[1]]))
        generated_center.paint_uniform_color([1.0, 0, 0]) 
        
        # Groundtruth BB
        groundtruth_center = open3d.geometry.PointCloud()
        groundtruth_center.points = open3d.utility.Vector3dVector(np.array([centers[2]]))
        groundtruth_center.paint_uniform_color([0, 1.0, 0]) 
        
        if visualize:
            open3d.visualization.draw_geometries(comp + [pointcloud_center, generated_center, groundtruth_center])


def run_detection(calib, image, pcd, detection_info, labels, use_vis = False, use_mask = False):
    """
    Runs 3D object detection 

    calib: KITTI calibration file
    image: image corresponding to the scene
    pcd: pointcloud corresponding to the scene
    detection_info: list of dictionaries containing detection results, each dict contians a 2d bounding box and 2d mask, later will be modified to contain frustum pcd, object candidate cluster, and generated 3d bb
    labels: dictionary of label/groundtruth information
    use_vis: use visualizations
    """
    metrics = dict()
    metrics['total_time'] = 0

    #############################################################################
    # REMOVE GROUND 
    #############################################################################
    if not use_mask:
        ground_remove_pcd, metrics['ground_removal_time'] = time_function(remove_ground, (pcd,), {'labels': labels['kitti_gt_3d_bb'], 'removal_offset': .075, 'visualize': use_vis})
        
        metrics['total_time'] += metrics['ground_removal_time']
    else:
        ground_remove_pcd = pcd

    
    #############################################################################
    # PROJECT KITTI GT POINTCLOUD POINTS ONTO IMAGE
    #############################################################################
    (projected_pcd_points, projected_pcd_grid), metrics['3d_to_2d_projection_time'] = time_function(render_lidar_on_image, (ground_remove_pcd, image, calib, image.shape[1], image.shape[0]), {'visualize': use_vis})
    
    metrics['total_time'] += metrics['3d_to_2d_projection_time']
    
    
    #############################################################################
    # SEGMENT FRUSTUMS 
    #############################################################################
    segmented_pcds, metrics['frustum_segmentation_time'] = time_function(segment_bb_frustum_from_projected_pcd, (detection_info, projected_pcd_points, projected_pcd_grid, calib), {'labels': labels['kitti_gt_3d_bb'], 'visualize': use_vis, 'orig_img': image})
    
    metrics['total_time'] += metrics['frustum_segmentation_time']


    #############################################################################
    # APPLY DBSCAN CLUSTERING TO EACH FRUSTUM 
    #############################################################################
    object_candidate_clusters = []
    metrics['dbscan_clustering_time'] = 0

    for i, segmented_pcd in enumerate(segmented_pcds):
        if detection_info[i]['frustum_pcd'] is not None:
            keep_n = 1 if use_mask else 3
            object_candidate_cluster, execution_time = time_function(apply_dbscan, (segmented_pcd, detection_info[i]), {'keep_n': keep_n, 'visualize': False})
            object_candidate_clusters.extend(object_candidate_cluster)
            detection_info[i]['object_candidate_cluster'] = object_candidate_cluster

            metrics['dbscan_clustering_time'] += execution_time
        else:
            detection_info[i]['object_candidate_cluster'] = None
    
    metrics['total_time'] += metrics['dbscan_clustering_time']


    #############################################################################
    # GENERATE 3D BOUNDING BOXES 
    #############################################################################
    generated_3d_bb_list, metrics['3d_bounding_box_generation_time'] = time_function(generate_3d_bb, (object_candidate_clusters, detection_info), {'visualize': use_vis})
    
    metrics['total_time'] += metrics['3d_bounding_box_generation_time'] 

    if 'ground_removal_time' in metrics.keys():
        print(f'{"Ground Removal Time":<30}: {metrics["ground_removal_time"]:.>5.4f} s.')
    print(f'{"3D to 2D Projection Time":<30}: {metrics["3d_to_2d_projection_time"]:.>5.4f} s.')
    print(f'{"Frustum Segmentation Time":<30}: {metrics["frustum_segmentation_time"]:.>5.4f} s.')
    print(f'{"DBSCAN Clustering Time":<30}: {metrics["dbscan_clustering_time"]:.>5.4f} s.')
    print(f'{"Bounding Box Generation Time":<30}: {metrics["3d_bounding_box_generation_time"]:.>5.4f} s.')

    return generated_3d_bb_list, object_candidate_clusters, detection_info, metrics

def prepare_tracking_files(scene_list):
    scene_dict = {}
    for scene in scene_list:
        files = sorted(os.listdir(f'kitti_tracking/training/velodyne/{str(scene).zfill(4)}'))
        files = [int(os.path.splitext(file)[0]) for file in files]
        scene_dict[str(scene)] = files
    return scene_dict

def test_kitti_scenes(file_num = 0, use_vis = False, tracking = False, use_mask = False):
    if tracking:
        scenes = prepare_tracking_files(file_num)
    else:
        scenes = {"None": file_num}

    
    # initialize metrics dict
    test_metrics = dict()
    test_metrics['min_scene_time'] = float('inf')
    test_metrics['avg_scene_time'] = 0
    test_metrics['max_scene_time'] = -float('inf')
    test_metrics['total_detections'] = 0
    test_metrics['min_individual_detection_time'] = float('inf')
    test_metrics['avg_individual_detection_time'] = 0
    test_metrics['max_individual_detection_time'] = -float('inf')

    for key in sorted(scenes.keys()):
        if tracking:
            with open(f'detection/maskrcnn_Pedestrian_train/{str(key).zfill(4)}.txt', "w+", newline="") as f:
                f.write('')
            with open(f'detection/maskrcnn_Cyclist_train/{str(key).zfill(4)}.txt', "w+", newline="") as f:
                f.write('')
            with open(f'detection/maskrcnn_Car_train/{str(key).zfill(4)}.txt', "w+", newline="") as f:
                f.write('')
        for i, file in enumerate(scenes[key]):
            print('='*50)
            print(f'Running File {str(file).zfill(6)}')
            print(f'Starting File {str(file).zfill(6)} Object Detection')


            #############################################################################
            # LOAD KITTI GROUNDTRUTH
            #############################################################################
            kitti_gt_image, kitti_gt_pointcloud, kitti_gt_labels, kitti_gt_calib = load_kitti_groundtruth(file, key)
            

            #############################################################################
            # LOAD MASK RCNN INFERENCE
            #############################################################################
            mr_inf_images, mr_inf_bboxes, mr_inf_segmentations, mr_inf_labels = load_mask_rcnn_inference(file, key)
            print(f'{"# MASK RCNN DETECTIONS: ":<30}: {len(mr_inf_labels):>5}.')
            print(f'{"# KITTI GT LABELS: ":<30}: {len(kitti_gt_labels):>5}.\n\n')

            if use_vis:
                draw_masks(kitti_gt_image, mr_inf_segmentations)


            #############################################################################
            # GET KITTI GROUND TRUTH 3D BOUNDING BOXES
            #############################################################################
            kitti_gt_3d_bb = get_groundtruth_3d_bb(kitti_gt_labels, kitti_gt_calib)
            

            #############################################################################
            # GET DETECTOR 2D BB LIST 
            #############################################################################
            mr_inf_2d_bb_list = get_detector_2d_bb(mr_inf_bboxes, kitti_gt_image, visualize=use_vis)  
            
            
            #############################################################################
            # RUN DETECTION
            #############################################################################
            labels={'kitti_gt_3d_bb': kitti_gt_3d_bb, 'kitti_gt_labels': kitti_gt_labels}
            if use_mask:
                mr_detections = [{'calib': kitti_gt_calib, 'frame': file, 'class': cls, 'bb': bb, 'mask': mask} for cls, bb, mask in zip(mr_inf_labels, mr_inf_2d_bb_list, mr_inf_segmentations)]
            else:
                mr_detections = [{'calib': kitti_gt_calib, 'frame': file, 'class': cls, 'bb': bb} for cls, bb in zip(mr_inf_labels, mr_inf_2d_bb_list)]
            
            generated_3d_bb_list, clustered_kitti_gt_pcd_list, detection_info, detection_metrics = run_detection(kitti_gt_calib, kitti_gt_image, kitti_gt_pointcloud, mr_detections, labels, use_vis, use_mask)
            
            
            #############################################################################
            # CONVERT TO AB3DMOT FORMAT
            #############################################################################
            if tracking:
                frame_ab3dmot_format = get_ab3dmot_format(detection_info)
                peds = list(filter(lambda line: line[1] == 1, frame_ab3dmot_format))
                cyclists = list(filter(lambda line: line[1] == 3, frame_ab3dmot_format))
                cars = list(filter(lambda line: line[1] == 2, frame_ab3dmot_format))

                with open(f'detection/maskrcnn_Pedestrian_train/{str(key).zfill(4)}.txt', "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerows(peds)
                with open(f'detection/maskrcnn_Cyclist_train/{str(key).zfill(4)}.txt', "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerows(cyclists)
                with open(f'detection/maskrcnn_Car_train/{str(key).zfill(4)}.txt', "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerows(cars)
                            

            #############################################################################
            # VISUALIZE RESULTS FOR SCENE
            #############################################################################
            if use_vis:
                mesh_frame = open3d.geometry.TriangleMesh.create_coordinate_frame(size=2, origin=[0, 0, 0])
                open3d.visualization.draw_geometries(clustered_kitti_gt_pcd_list + kitti_gt_3d_bb + generated_3d_bb_list + [mesh_frame])

            #############################################################################
            # GET TEST INFERENCE METRICS
            #############################################################################
            if detection_metrics['total_time'] > test_metrics['max_scene_time']:
                test_metrics['max_scene_time'] = detection_metrics['total_time']
            if detection_metrics['total_time'] < test_metrics['min_scene_time']:
                test_metrics['min_scene_time'] = detection_metrics['total_time']
            test_metrics['avg_scene_time'] += detection_metrics['total_time']
            
            detection_metrics['avg_time_per_detection'] = detection_metrics['total_time']/len(mr_inf_labels)
            if detection_metrics['avg_time_per_detection'] > test_metrics['max_individual_detection_time']:
                test_metrics['max_individual_detection_time'] = detection_metrics['avg_time_per_detection']
            if detection_metrics['avg_time_per_detection'] < test_metrics['min_individual_detection_time']:
                test_metrics['min_individual_detection_time'] = detection_metrics['avg_time_per_detection']
            test_metrics['avg_individual_detection_time'] += detection_metrics['avg_time_per_detection']
            test_metrics['total_detections'] += len(mr_inf_labels)

            test_metrics[f'kitti_scene_{str(file).zfill(6)}_inference_metrics'] = detection_metrics

            print(f'\nFile {str(file).zfill(6)} Object Detection {"Total Execution Time":<40}: {detection_metrics["total_time"]:.>5.4f} s.')
            print(f'File {str(file).zfill(6)} Object Detection {"Avg Inference Time/Detection":<40}: {detection_metrics["avg_time_per_detection"]:.>5.4f} s.')
            print(f'Finished Running file {str(file).zfill(6)} Object Detection')


            # #############################################################################
            # # RUN ACCURACY ANALYSIS
            # #############################################################################
            # print(f'Starting Running file {str(file).zfill(6)} Detection Analysis')
            # analysis_metrics = detection_analysis(detection_info, labels)
            # print(f'Finished Running file {str(file).zfill(6)} Detection Analysis')

            # test_metrics[f'kitti_scene_{str(file).zfill(6)}_analysis_metrics'] = analysis_metrics

            # print('='*50)
        
    test_metrics["avg_individual_detection_time"] = float(test_metrics["avg_individual_detection_time"])/test_metrics["total_detections"]
    test_metrics["avg_scene_time"] = float(test_metrics["avg_scene_time"])/len(test_list)

    print('='*50)
    print(f'Overall Detection Inference Analysis')
    print('='*50)
    print(f'3D Object Detection {"Min Scene Inference Time":<40}: {test_metrics["min_scene_time"]:.>5.4f} s.')
    print(f'3D Object Detection {"Avg Scene Inference Time":<40}: {test_metrics["avg_scene_time"]:.>5.4f} s.')
    print(f'3D Object Detection {"Max Scene Inference Time":<40}: {test_metrics["max_scene_time"]:.>5.4f} s.')
    print(f'3D Object Detection {"Min Inference Time Per Detection":<40}: {test_metrics["min_individual_detection_time"]:.>5.4f} s.')
    print(f'3D Object Detection {"Avg Inference Time Per Detection":<40}: {float(test_metrics["avg_individual_detection_time"]):.>5.4f} s.')
    print(f'3D Object Detection {"Max Inference Time Per Detection":<40}: {test_metrics["max_individual_detection_time"]:.>5.4f} s.')
    
    json.dump(test_metrics, open('kitti_test_metrics.json', 'w'), cls=DetectionEncoder)

        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--tracking", default=False, action="store_true")
    parser.add_argument("--vis", default=False, action="store_true")
    parser.add_argument("--use-mask", default=False, action="store_true")
    arguments = parser.parse_args()
    test_list = [
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        16,
        17,
        18,
        19,
        20
    ]
    test_kitti_scenes(test_list, arguments.vis, arguments.tracking, arguments.use_mask)
