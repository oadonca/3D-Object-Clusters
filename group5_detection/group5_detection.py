import matplotlib.pyplot as plt
import open3d
import collections
import argparse
import numpy as np
import math

from utils import *
from iou3d import get_3d_box, box3d_iou
from test_scripts import *

def render_image_with_boxes(img, objects, calib, autodrive):
    """
    Show image with 3D boxes
    """
    # projection matrix
    P_rect2cam2 = calib['P2'].reshape((3, 4))

    img1 = np.copy(img)
    for obj in objects:
        if obj.type == 'DontCare':
            continue
        box3d_pixelcoord = map_box_to_image(obj, P_rect2cam2, autodrive)
        img1 = draw_projected_box3d(img1, box3d_pixelcoord)

    plt.imshow(img1)
    plt.yticks([])
    plt.xticks([])
    plt.show()


def render_lidar_with_boxes(pc_velo, objects, calib, img_width, img_height, dbscan=False, autodrive=True):
    # projection matrix (project from velo2cam2)
    proj_velo2cam2 = project_velo_to_cam2(calib)

    # apply projection
    pts_2d = project_to_image(pc_velo.transpose(), proj_velo2cam2, autodrive)

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


def render_lidar_on_image(pts_velo, orig_img, calib, img_width, img_height, depth_limit = 100, visualize=True, autodrive=True):
    """
    Projects the given lidar points onto the given image using the projection/transformation matrices provided in calib
    
    Returns: Image and projected points
    """
    img = np.copy(orig_img)
    # projection matrix (project from velo2cam2)
    if autodrive:        
        
        # [x, y, z, 1] -> [x, y, z, w]
        # transform poins from velo frame to camera frame
        # make homogenous
        # print('PointCloud XYZ: ', pts_velo.shape)
        
        pts_velo_homogeneous = np.hstack((pts_velo, np.ones((pts_velo.shape[0], 1))))
        # print('PointCloud XYZ pad One: ', pts_velo_homogeneous.shape)

        transform_mat = np.transpose(calib['ad_transform_mat'])
        # print('tform: \n', transform_mat)
       
        # Lidar points in camera frame
        pts_cam_frame = transform_mat @ np.transpose(pts_velo_homogeneous)
        # print('cameraXYZ before delete: ', pts_cam_frame.shape)
        pts_cam_frame = np.delete(pts_cam_frame, 3, 0)
        # print('cameraXYZ after delete: ', pts_cam_frame.shape)

        # [x, y, z, w] -> [x, y, w]
        # then project points from camera frame onto image plane
        # print('intrinsics: \n', np.transpose(calib['ad_projection_mat']))
        pts_image_plane = np.transpose(calib['ad_projection_mat']) @ pts_cam_frame
        
        # print('image_uv: ', pts_image_plane.shape)
        # print(pts_image_plane/pts_image_plane[2])
                
        projected_grid = np.zeros_like(img).astype(float)
        for point in np.transpose(pts_image_plane):
            if point[2] < depth_limit:
                x = point[0]/point[2]
                y = point[1]/point[2]
                projected_grid[int(y), int(x)] = point
        
        if visualize:   
            # cmap = plt.cm.get_cmap('hsv', 256)
            # cmap = np.array([cmap(i) for i in range(256)])[:, :3] * 255
            # # Display projected point cloud on the image
            # for i in range(pts_cam_frame.shape[1]):
            #     depth = pts_image_plane[2, i]
            #     color = cmap[min(int(640.0 / depth), 255), :]
            #     cv2.circle(img, (int(np.round(pts_image_plane[0, i]/pts_image_plane[2, i])),
            #                     int(np.round(pts_image_plane[1, i]/pts_image_plane[2, i]))),
            #             2, color=tuple(color), thickness=-1)
                
            plt.imshow(orig_img)
            plt.scatter(pts_image_plane[0]/pts_image_plane[2], pts_image_plane[1]/pts_image_plane[2], s=1, color='red')
            plt.show()
            
            plt.imshow(orig_img)
            plt.imshow(projected_grid)
            plt.show()
            
            points_list = []
            for y, row in enumerate(projected_grid):
                for x, col in enumerate(row):
                    if x > 503 and x < 817 and y > 589 and y < 690:
                        points_list.append(projected_grid[y][x])
                        
            pcd = open3d.geometry.PointCloud()
            points = np.array(points_list)
            pcd.points = open3d.utility.Vector3dVector(points)

            open3d.visualization.draw_geometries([pcd],
                                         front=[-0.9945, 0.03873, 0.0970],  
                                         lookat=[38.4120, 0.6139, 0.48500],
                                         up=[0.095457, -0.0421, 0.99453],
                                         zoom=0.33799
                                         )   
            
            # Image plane 2D to Camera frame 3D            
            inv_projection = np.linalg.inv(np.transpose(calib['ad_projection_mat']))
            cam_coords = inv_projection @ pts_image_plane
            
            # Camera frame to lidar sensor frame
            cam_coords = np.vstack((cam_coords, np.ones((1, cam_coords.shape[1]))))
            inv_transform = np.linalg.inv(np.transpose(calib['ad_transform_mat']))
            velo_points = inv_transform @ cam_coords
            velo_points = velo_points[:3, :]
            
            pcd = open3d.geometry.PointCloud()
            points = np.transpose(velo_points)
            pcd.points = open3d.utility.Vector3dVector(points)

            open3d.visualization.draw_geometries([pcd],
                                         front=[-0.9945, 0.03873, 0.0970],  
                                         lookat=[38.4120, 0.6139, 0.48500],
                                         up=[0.095457, -0.0421, 0.99453],
                                         zoom=0.33799
                                         )
                                
        return pts_image_plane, projected_grid
        
    else:    
        proj_velo2cam2 = project_velo_to_cam2(calib)

        # apply projection
        pts_2d = project_to_image(pts_velo.transpose(), proj_velo2cam2, autodrive)
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


def segment_bb_frustum_from_projected_pcd(detections, projected_points, projected_grid, calib, labels_info=None, visualize=False, orig_img = None, autodrive=True):
    """
    Given a np array of projected 2D points and a list of bounding boxes, 
    Removes all points outside the frustums generated by the provided bounding boxes
    
    Returns: A list of pointclouds for each segmented frustum
    """
    if labels_info is not None:
        labels = labels_info['kitti_gt_3d_bb']
    
    # Get image to cam 2 frame transform
    if not autodrive:
        image_transform = project_image_to_cam2(calib, autodrive)
    
    # Get only the points within bounding boxes
    # Only need this for only bounding box segmentation, not required for mask propagation
    bb_inds = within_bb_indices(detections, projected_points/projected_points[2], autodrive)
    
    # Non parallel
    mask_points_list = []
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
            if autodrive:
                # Image plane 2D to Camera frame 3D
                inv_projection = np.linalg.inv(np.transpose(calib['ad_projection_mat']))
                cam_coords = inv_projection @ object_pts_2d
                
                # Camera frame to lidar sensor frame
                cam_coords = np.vstack((cam_coords, np.ones((1, cam_coords.shape[1]))))
                inv_transform = np.linalg.inv(np.transpose(calib['ad_transform_mat']))
                velo_points = inv_transform @ cam_coords
                velo_points = velo_points[:3, :]
                
                if visualize:
                    mask_points_list.append(object_pts_2d)
                
            else:   
                cam_coords = image_transform[:3, :3] @ object_pts_2d

                # Get projection from camera coordinates to velodyne coordinates
                proj_mat = project_cam2_to_velo(calib, autodrive)
                
                # apply projection
                velo_points = project_camera_to_lidar(cam_coords, proj_mat)
        
            detections[i]['frustum_pcd'] = np.array(velo_points)
        else:
            detections[i]['frustum_pcd'] = None
        
    if autodrive and visualize:
        for pts in mask_points_list:
            fig, ax = plt.subplots(1, 1, figsize=(5, 5))
            ax.clear()
            ax.imshow(orig_img)
            ax.scatter(pts[0, :]/pts[2,:], pts[1, :]/pts[2,:], s=1, c='red')
            plt.show()
            
            # Image plane 2D to Camera frame 3D
            inv_projection = np.linalg.inv(np.transpose(calib['ad_projection_mat']))
            cam_coords = inv_projection @ pts
            
            pcd = open3d.geometry.PointCloud()
            points = np.transpose(cam_coords)
            pcd.points = open3d.utility.Vector3dVector(points)
            open3d.visualization.draw_geometries([pcd],
                                         front=[-0.9945, 0.03873, 0.0970],  
                                         lookat=[38.4120, 0.6139, 0.48500],
                                         up=[0.095457, -0.0421, 0.99453],
                                         zoom=0.33799
                                         )
            
            # Camera frame to lidar sensor frame
            cam_coords = np.vstack((cam_coords, np.ones((1, cam_coords.shape[1]))))
            inv_transform = np.linalg.inv(np.transpose(calib['ad_transform_mat']))
            velo_points = inv_transform @ cam_coords
            velo_points = velo_points[:3, :]
            
            pcd = open3d.geometry.PointCloud()
            points = np.transpose(velo_points)
            pcd.points = open3d.utility.Vector3dVector(points)
            open3d.visualization.draw_geometries([pcd],
                                         front=[-0.9945, 0.03873, 0.0970],  
                                         lookat=[38.4120, 0.6139, 0.48500],
                                         up=[0.095457, -0.0421, 0.99453],
                                         zoom=0.33799
                                         )
        
    if visualize:
        o3d_pcd_list = []
        for detection in detections:
            if detection['frustum_pcd'] is not None:
                pcd = open3d.geometry.PointCloud()
                points = np.transpose(detection['frustum_pcd'])
                pcd.points = open3d.utility.Vector3dVector(points)
                o3d_pcd_list.append(pcd)
                open3d.visualization.draw_geometries([pcd])

    if visualize:
        if autodrive:
            pcd_display_list = o3d_pcd_list
        else:
            pcd_display_list = o3d_pcd_list + labels
            
        open3d.visualization.draw_geometries(pcd_display_list)
        
    return [detection['frustum_pcd'] for detection in detections]

def remove_ground(pointcloud, labels=[], removal_offset = 0, visualize=False, autodrive=True, downsample=True):
    """
    Removes ground points from provided pointcloud
    
    Returns: Pointcloud
    """
    
    # Create Open3D pointcloud
    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(pointcloud)
    if downsample:
        pcd = pcd.voxel_down_sample(voxel_size=0.05)
    
    # Run RANSAC
    model, inliers = pcd.segment_plane(distance_threshold=0.15,ransac_n=3, num_iterations=500)
    # Get the average inlier coorindate values
    average_inlier = np.mean(pointcloud[inliers], axis=0)

    # Remove inliers
    segmented_pointcloud = pcd.select_by_index(inliers, invert=True)
    segmented_pointcloud_points = np.array(segmented_pointcloud.points)
    
    distance_to_plane = lambda x,y,z: (model[0]*x + model[1]*y + model[2]*z + model[3])/np.sqrt(np.sum(np.square(model[:3])))
    # Remove points below plane
    mask_inds = np.where(distance_to_plane(segmented_pointcloud_points[:, 0], segmented_pointcloud_points[:, 1], segmented_pointcloud_points[:, 2]) < removal_offset)
    segmented_pointcloud = segmented_pointcloud.select_by_index(mask_inds[0], invert=True)
    segmented_pointcloud_points = np.array(segmented_pointcloud.points)
    
    if visualize:
        # Visualize
        inlier_cloud = pcd.select_down_sample(inliers) # use select_by_index() depending on version of open3d
        inlier_cloud.paint_uniform_color([1.0, 0, 0])
        outlier_cloud = pcd.select_by_index(inliers, invert=True) # same as above
        outlier_cloud.paint_uniform_color([0, 1.0, 0.0])
        open3d.visualization.draw_geometries([inlier_cloud + outlier_cloud] + labels,
                                    zoom=0.8,
                                    front=[-0.4999, -0.1659, -0.8499],
                                    lookat=[2.1813, 2.0619, 2.0999],
                                    up=[0.1204, -0.9852, 0.1215])
        open3d.visualization.draw_geometries([segmented_pointcloud] + labels,
                                    zoom=0.8,
                                    front=[-0.4999, -0.1659, -0.8499],
                                    lookat=[2.1813, 2.0619, 2.0999],
                                    up=[0.1204, -0.9852, 0.1215])
    
    return np.reshape(segmented_pointcloud_points, (-1, 3))


def apply_dbscan(detection, keep_n=5, visualize=True, autodrive=True):
    """
    Applies DBSCAN on the provided point cloud and returns the Open3D point cloud
    """
    
    pointcloud = detection['frustum_pcd']
    
    if autodrive:
        keep_n = 1
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
            pcd_list.append(pcd.select_down_sample(np.argwhere(labels==label))) # select_by_index() instead

    if len(pcd_list) > 1:
        cluster_losses = get_cluster_scores(pcd_list, detection, use_autodrive_classes=autodrive)
        object_candidate_cluster_idx = np.argmin(cluster_losses)
        pcd_list = [pcd_list[object_candidate_cluster_idx]]
        
    if visualize:
        # Visualize
        open3d.visualization.draw_geometries([pcd])
    
    if pcd_list:
        detection['object_candidate_cluster'] = pcd_list[0]
        return pcd_list[0]
   
    detection['object_candidate_cluster'] = None
    return None

def generate_3d_bb(detections, oriented=False, visualize=False):
    """
    Generates bounding boxes around each point cloud in pcd_list
    """
    generated_bb_list = []
    for i, detection in enumerate(detections):
        if detection['object_candidate_cluster'] is not None and np.max(np.array(detection['object_candidate_cluster'].colors)) > 0:
            bb = detection['object_candidate_cluster'].get_axis_aligned_bounding_box() if not oriented else detection['object_candidate_cluster'].get_oriented_bounding_box()
            bb.color = [1.0, 0, 0]
            generated_bb_list.append(bb)
            detections[i]['generated_3d_bb'] = bb
        else:
            detections[i]['generated_3d_bb'] = None
                
    if visualize:
        open3d.visualization.draw_geometries(generated_bb_list)
        
    return generated_bb_list


def run_detection(calib, image, pcd, bb_list, labels=None, use_vis = False, use_mask = False, autodrive=True, depth_limit=100):
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
        
    if autodrive:
        detection_info = list()
        for bb in bb_list:
            detection = dict() 
            detection['bb'] = bb[:4]
            detection['class'] = bb[4]
            detection['confidence'] = bb[5]
            detection_info.append(detection)
    else:
        detection_info = bb_list
     
    #############################################################################
    # AUTODRIVE DEPTH LIMIT
    #############################################################################    
    if autodrive and depth_limit is not None: 
        pcd, metrics['depth_limit_time'] = time_function(limit_pcd_depth, args=(pcd, depth_limit))

    #############################################################################
    # REMOVE GROUND 
    #############################################################################
    if not use_mask:
        ground_remove_pcd, metrics['ground_removal_time'] = time_function(remove_ground, (pcd,), {'removal_offset': 0, 'visualize': use_vis, 'autodrive': autodrive})
        
        metrics['total_time'] += metrics['ground_removal_time']
    else:
        ground_remove_pcd = pcd

    
    #############################################################################
    # PROJECT KITTI GT POINTCLOUD POINTS ONTO IMAGE
    #############################################################################
    (projected_pcd_points, projected_pcd_grid), metrics['3d_to_2d_projection_time'] = time_function(render_lidar_on_image, (ground_remove_pcd, image, calib, image.shape[1], image.shape[0]), {'visualize': use_vis, 'autodrive': autodrive})
    
    metrics['total_time'] += metrics['3d_to_2d_projection_time']

    
    #############################################################################
    # SEGMENT FRUSTUMS 
    #############################################################################
    segmented_pcds, metrics['frustum_segmentation_time'] = time_function(segment_bb_frustum_from_projected_pcd, (detection_info, projected_pcd_points, projected_pcd_grid, calib), {'labels_info': labels, 'visualize': use_vis, 'orig_img': image, 'autodrive': autodrive})
    
    metrics['total_time'] += metrics['frustum_segmentation_time']


    #############################################################################
    # APPLY DBSCAN CLUSTERING TO EACH FRUSTUM 
    #############################################################################
    metrics['dbscan_clustering_time'] = 0

    for i, detection in enumerate(detection_info):
        if detection_info[i]['frustum_pcd'] is not None:
            keep_n = 1 if use_mask else 3
            object_candidate_cluster, execution_time = time_function(apply_dbscan, (detection_info[i],), {'keep_n': keep_n, 'visualize': False, 'autodrive': autodrive})
            metrics['dbscan_clustering_time'] += execution_time
            
        else:
            detection_info[i]['object_candidate_cluster'] = None
                
    metrics['total_time'] += metrics['dbscan_clustering_time']

    #############################################################################
    # GENERATE 3D BOUNDING BOXES 
    #############################################################################
    generated_3d_bb_list, metrics['3d_bounding_box_generation_time'] = time_function(generate_3d_bb, (detection_info,), {'visualize': use_vis})
    
    metrics['total_time'] += metrics['3d_bounding_box_generation_time'] 

    if 'depth_limit_time' in metrics.keys():
        print(f'{"Depth Limit Time":<30}: {metrics["depth_limit_time"]:.>5.4f} s.')
    if 'ground_removal_time' in metrics.keys():
        print(f'{"Ground Removal Time":<30}: {metrics["ground_removal_time"]:.>5.4f} s.')
    print(f'{"3D to 2D Projection Time":<30}: {metrics["3d_to_2d_projection_time"]:.>5.4f} s.')
    print(f'{"Frustum Segmentation Time":<30}: {metrics["frustum_segmentation_time"]:.>5.4f} s.')
    print(f'{"DBSCAN Clustering Time":<30}: {metrics["dbscan_clustering_time"]:.>5.4f} s.')
    print(f'{"Bounding Box Generation Time":<30}: {metrics["3d_bounding_box_generation_time"]:.>5.4f} s.')
        
    return generated_3d_bb_list, detection_info, metrics

def run_tracking(detection_info, classes, tracker_dict, frame):
    frame_ab3dmot_format = get_ab3dmot_format(detection_info)
    detect_dict = dict()
    detect_dict['Pedestrian'] = list(filter(lambda line: line[1] == 1, frame_ab3dmot_format))
    detect_dict['Car'] = list(filter(lambda line: line[1] == 2, frame_ab3dmot_format))
    detect_dict['Animal'] = list(filter(lambda line: line[1] == 3, frame_ab3dmot_format))

    for cat in classes:
        results = tracker_dict[cat].track(detect_dict[cat], frame, 'live')
        print(results)
    frame += 1
        
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
    test_kitti_scenes(test_list, arguments.vis, arguments.tracking, arguments.use_mask, autodrive=False)
    # test_autodrive_scenes(0, arguments.use_mask)
