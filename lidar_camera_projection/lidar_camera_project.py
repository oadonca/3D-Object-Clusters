from cgi import test
import os
import time
import matplotlib.pyplot as plt
import open3d
import itertools
import collections

from utils import *


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


def render_lidar_on_image(pts_velo, img, calib, img_width, img_height, visualize=True):
    """
    Projects the given lidar points onto the given image using the projection/transformation matrices provided in calib
    
    Returns: Image and projected points
    """
    
    orig_img = np.copy(img)
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

    cmap = plt.cm.get_cmap('hsv', 256)
    cmap = np.array([cmap(i) for i in range(256)])[:, :3] * 255

    # Display projected point cloud on the image
    for i in range(imgfov_pc_pixel.shape[1]):
        depth = imgfov_pc_cam2[2, i]
        color = cmap[int(640.0 / depth), :]
        cv2.circle(img, (int(np.round(imgfov_pc_cam2[0, i]/imgfov_pc_cam2[2, i])),
                         int(np.round(imgfov_pc_cam2[1, i]/imgfov_pc_cam2[2, i]))),
                   2, color=tuple(color), thickness=-1)
    
    if visualize:
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))

        ax[0].set_title('Original Image')
        ax[0].imshow(orig_img)

        ax[1].set_title('Image w/ Projected Points')
        ax[1].imshow(img)

        plt.yticks([])
        plt.xticks([])
        plt.show()
        
    return img, imgfov_pc_cam2

def render_pointcloud_from_projection(image, projected_points, calib, image_width, image_height):
    # Get image to cam 2 frame transform
    image_transform = project_image_to_cam2(calib)
    
    # Apply back-projection: K_inv @ pixels * depth
    # doing it this way because it is the same process if you just have the projected lidar point x,y and then depth value
    #   (projected_points/projected_points[2, :]) = homogenous coordinates -> N count of [x, y, 1] points
    #   projected_points[2, :] = depth value
    cam_coords = image_transform[:3, :3] @ (projected_points/projected_points[2, :]) * projected_points[2, :]

    # Get projection from camera coordinates to velodyne coordinates
    proj_mat = project_cam2_to_velo(calib)
    
    # apply projection
    velo_points = project_camera_to_lidar(cam_coords, proj_mat)
    print('2D to 3D VELO POINTS: ', velo_points.shape)
    
    # Visualize
    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(np.transpose(velo_points))
    open3d.visualization.draw_geometries([pcd],
                                         front=[-0.9945, 0.03873, 0.0970],
                                         lookat=[38.4120, 0.6139, 0.48500],
                                         up=[0.095457, -0.0421, 0.99453],
                                         zoom=0.33799
                                         )
    return velo_points
    
def render_pointcloud_from_bb_segmentation(image, projected_points, calib, image_width, image_height):
    # manually chosen bounding boxes in form [topleft x/y, topright x/y, bottomleft x/y, bottomright x/y]
    # bb 1 (leftmost minivan):                          
    bb1 = [(185, 183), (300, 183), (185, 240), (300, 240)]
    # bb 2 (person next to traffic light in median):    
    bb2 = [(442, 167), (474, 167), (442, 255), (474, 255)]
    # bb 3 (car in middle of image):                    
    bb3 = [(589, 189), (668, 189), (589, 252), (668, 252)]
    # bb 4 (pedestrian with bike):                      
    bb4 = [(905, 173), (996, 173), (905, 267), (996, 267)]
    bb_list = [bb1, bb2, bb3, bb4]    
    # bb_list = [bb3]

    # Get image to cam 2 frame transform
    image_transform = project_image_to_cam2(calib)
    
    pts_2d = projected_points/projected_points[2, :]
    
    # Get only the points within bounding boxes
    bb_inds = within_bb_indices(bb_list, pts_2d)
    inds = []
    for idx in bb_inds:
        inds.extend(idx)
    
    bb_pts_2d = pts_2d[:, inds]
    
    # Apply back-projection: K_inv @ pixels * depth
    # doing it this way because it is the same process if you just have the projected lidar point x,y and then depth value
    #   (projected_points/projected_points[2, :]) = homogenous coordinates -> N count of [x, y, 1] points
    #   projected_points[2, :] = depth value
    cam_coords = image_transform[:3, :3] @ bb_pts_2d * projected_points[2, inds]

    # Get projection from camera coordinates to velodyne coordinates
    proj_mat = project_cam2_to_velo(calib)
    
    # apply projection
    velo_points = project_camera_to_lidar(cam_coords, proj_mat)
    print('2D to 3D VELO POINTS: ', velo_points.shape)
    
    # Create Open3D point cloud
    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(np.transpose(velo_points))
    
    # Run DBSCAN on point cloud
    with open3d.utility.VerbosityContextManager(open3d.utility.VerbosityLevel.Debug) as cm:
        labels = np.array(pcd.cluster_dbscan(eps=1, min_points=20, print_progress=True))

    # Set colors of point cloud to see DBSCAN clusters
    max_label = labels.max()
    print(f"point cloud has {max_label + 1} clusters")
    colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
    colors[labels < 0] = 0
    pcd.colors = open3d.utility.Vector3dVector(colors[:, :3])
    
    # Visualize
    open3d.visualization.draw_geometries([pcd],
                                         front=[-0.9945, 0.03873, 0.0970],
                                         lookat=[38.4120, 0.6139, 0.48500],
                                         up=[0.095457, -0.0421, 0.99453],
                                         zoom=0.33799
                                         )
    return velo_points    

def segment_bb_frustum_from_projected_pcd(bb_list, projected_pointcloud, calib, labels=[], visualize=True):
    """
    Given a np array of projected 2D points and a list of bounding boxes, 
    Removes all points outside the frustums generated by the provided bounding boxes
    
    Returns: A list of pointclouds for each segmented frustum
    """
    # Get image to cam 2 frame transform
    image_transform = project_image_to_cam2(calib)
    
    # Get nonhomogenous points
    pts_2d = projected_pointcloud/projected_pointcloud[2, :]
    
    # Get only the points within bounding boxes
    inds = within_bb_indices(bb_list, pts_2d)
    
    pcd_list = []
    o3d_pcd_list = []
    
    for idx in inds:
        bb_pts_2d = pts_2d[:, idx]
        
        # Apply back-projection: K_inv @ pixels * depth
        # doing it this way because it is the same process if you just have the projected lidar point x,y and then depth value
        #   (projected_points/projected_points[2, :]) = homogenous coordinates -> N count of [x, y, 1] points
        #   projected_points[2, :] = depth value
        cam_coords = image_transform[:3, :3] @ bb_pts_2d * projected_pointcloud[2, idx]

        # Get projection from camera coordinates to velodyne coordinates
        proj_mat = project_cam2_to_velo(calib)
        
        # apply projection
        velo_points = project_camera_to_lidar(cam_coords, proj_mat)
    
        pcd_list.append(velo_points)
        if visualize:
            pcd = open3d.geometry.PointCloud()
            pcd.points = open3d.utility.Vector3dVector(np.transpose(velo_points))
            o3d_pcd_list.append(pcd)
        
    if visualize:
        open3d.visualization.draw_geometries(o3d_pcd_list + labels,
                                         front=[-0.9945, 0.03873, 0.0970],
                                         lookat=[38.4120, 0.6139, 0.48500],
                                         up=[0.095457, -0.0421, 0.99453],
                                         zoom=0.33799
                                         )
        
    return pcd_list

def remove_ground(pointcloud, labels=[], removal_offset = 0, visualize=True):
    """
    Removes ground points from provided pointcloud
    
    Returns: Pointcloud
    """
    
    # Create Open3D pointcloud
    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(pointcloud)
    
    # Run RANSAC
    model, inliers = pcd.segment_plane(distance_threshold=0.1,ransac_n=3, num_iterations=250)

    # Get the average inlier coorindate values
    average_inlier = np.mean(pointcloud[inliers], axis=0)

    # Remove inliers
    segmented_pointcloud = np.delete(pointcloud, inliers, axis=0)
        
    # Remove points below average inlier z value
    mask = np.argwhere(segmented_pointcloud[:, 2] < average_inlier[2]+removal_offset)
    segmented_pointcloud = np.delete(segmented_pointcloud, mask, axis=0)
    
    if visualize:
        # Visualize
        inlier_cloud = pcd.select_by_index(inliers)
        inlier_cloud.paint_uniform_color([1.0, 0, 0])
        outlier_cloud = pcd.select_by_index(inliers, invert=True)
        outlier_cloud.paint_uniform_color([0, 0, 1.0])            
        open3d.visualization.draw_geometries([inlier_cloud, outlier_cloud] + labels,
                                    zoom=0.8,
                                    front=[-0.4999, -0.1659, -0.8499],
                                    lookat=[2.1813, 2.0619, 2.0999],
                                    up=[0.1204, -0.9852, 0.1215])
    
    return np.reshape(segmented_pointcloud, (-1, 3))


def apply_dbscan(pointcloud, keep_n=None, visualize=True):
    """
    Applies DBSCAN on the provided point cloud and returns the Open3D point cloud
    """
    # Create Open3D point cloud
    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(np.transpose(pointcloud))
    
    # Run DBSCAN on point cloud
    # with open3d.utility.VerbosityContextManager(open3d.utility.VerbosityLevel.Debug) as cm:
    labels = np.array(pcd.cluster_dbscan(eps=1.5, min_points=10))

    # Set colors of point cloud to see DBSCAN clusters
    max_label = labels.max()
    # print(f"point cloud has {max_label + 1} clusters")
    colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
    colors[labels < 0] = 0
    pcd.colors = open3d.utility.Vector3dVector(colors[:, :3])
    
    counts = collections.Counter(labels)
        
    pcd_list = []
    if not keep_n:
        for label in range(labels.max()+1):
            pcd_list.append(pcd.select_by_index(np.argwhere(labels==label)))
    else:
        for label, count in counts.most_common(keep_n):
            pcd_list.append(pcd.select_by_index(np.argwhere(labels==label)))
        
    if visualize:
        # Visualize
        open3d.visualization.draw_geometries([pcd])
    
    return pcd_list

def generate_3d_bb(pcd_list, oriented=False, visualize=True):
    """
    Generates bounding boxes around each point cloud in pcd_list
    """
    generated_bb_list = []
    for pcd in pcd_list:
        bb = pcd.get_axis_aligned_bounding_box() if not oriented else pcd.get_oriented_bounding_box()
        bb.color = [1.0, 0, 0]
        generated_bb_list.append(bb)
        
    display_list = pcd_list + generated_bb_list
        
    if visualize:
        open3d.visualization.draw_geometries(display_list)
        
    return generated_bb_list

def get_groundtruth_3d_bb(objects, calib):
    """
    Gets the ground truth bounding boxes provided in objects
    """
    # Projection matrix
    proj_cam2_2_velo = project_cam2_to_velo(calib)

    bb_list = []

    # Draw objects on lidar
    for obj in objects:
        if obj.type == 'DontCare':
            continue

        # Project boxes from camera to lidar coordinate
        boxes3d_pts = project_camera_to_lidar(obj.in_camera_coordinate(), proj_cam2_2_velo)

        # Open3d boxes
        boxes3d_pts = open3d.utility.Vector3dVector(boxes3d_pts.T)
        box = open3d.geometry.OrientedBoundingBox.create_from_points(boxes3d_pts)
        box.color = [0, 1.0, 0]
        bb_list.append(box)
    
    return bb_list

def get_groundtruth_2d_bb():
    # manually chosen bounding boxes in form [topleft x/y, topright x/y, bottomleft x /y, bottomright x/y]
    # bb 1 (leftmost minivan):                          
    bb1 = [(185, 183), (300, 183), (185, 240), (300, 240)]
    # bb 2 (person next to traffic light in median):    
    bb2 = [(442, 167), (474, 167), (442, 255), (474, 255)]
    # bb 3 (car in middle of image):                    
    bb3 = [(589, 189), (668, 189), (589, 252), (668, 252)]
    # bb 4 (pedestrian with bike):                      
    bb4 = [(905, 173), (996, 173), (905, 267), (996, 267)]
    # bb5 farthest parked car
    bb5 = [(519, 179), (573, 179), (519, 200), (573, 200)]
    bb_list = [bb1, bb2, bb3, bb4, bb5]   
    
    return bb_list

def compare_generation_accuracy(comp_list, visualize=True):    
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
        
if __name__ == '__main__':
    # Load image, calibration file, label bbox
    rgb = cv2.cvtColor(cv2.imread(os.path.join('data/000114_image.png')), cv2.COLOR_BGR2RGB)
    img_height, img_width, img_channel = rgb.shape

    total_time = 0

    # Load calibration
    start = time.perf_counter()
    calib = read_calib_file('data/000114_calib.txt')
    stop = time.perf_counter()
    time_diff = stop-start
    total_time += time_diff
    print(f'{"Calibration File Loading Time":<30}: {time_diff:.>5.4f} s.')

    # Load labels
    start = time.perf_counter()
    labels = load_label('data/000114_label.txt')
    
    test_labels = load_label('data/000114_label_test.txt')
    stop = time.perf_counter()
    time_diff = stop-start
    total_time += time_diff
    print(f'{"Label Loading Time":<30}: {time_diff:.>5.4f} s.')
    
    # Load Lidar PC
    start = time.perf_counter()
    pc_velo = load_velo_scan('data/000114.bin')[:, :3]
    stop = time.perf_counter()
    time_diff = stop-start
    total_time += time_diff
    print(f'{"LiDAR Pointcloud Loading Time":<30}: {time_diff:.>5.4f} s.') 

    render_image_with_boxes(rgb, test_labels, calib)
    render_lidar_with_boxes(pc_velo, test_labels, calib, img_width=img_width, img_height=img_height)
    
    start = time.perf_counter()
    segmented_pc_velo = remove_ground(pc_velo, labels=get_groundtruth_3d_bb(test_labels, calib), removal_offset=.1, visualize=True)
    stop = time.perf_counter()
    time_diff = stop-start
    total_time += time_diff
    print(f'{"Ground Removal Time":<30}: {time_diff:.>5.4f} s.')
    
    render_lidar_with_boxes(segmented_pc_velo, test_labels, calib, img_width=img_width, img_height=img_height)
    
    start = time.perf_counter()
    im, points = render_lidar_on_image(segmented_pc_velo, rgb, calib, img_width, img_height, visualize=True)
    stop = time.perf_counter()
    time_diff = stop-start
    total_time += time_diff
    print(f'{"3D to 2D Projection Time":<30}: {time_diff:.>5.4f} s.')
    
    render_pointcloud_from_projection(im, points, calib, img_width, img_height)
    
    start = time.perf_counter()
    frustum_pcd_list = segment_bb_frustum_from_projected_pcd(get_groundtruth_2d_bb(), points, calib, labels=get_groundtruth_3d_bb(test_labels, calib), visualize=True)
    stop = time.perf_counter()
    time_diff = stop-start
    total_time += time_diff
    print(f'{"Frustum Segmentation Time":<30}: {time_diff:.>5.4f} s.')
    
    start = time.perf_counter()
    dbscan_pcd_list = []
    for pcd in frustum_pcd_list:
        dbscan_pcd_list.extend(apply_dbscan(pcd, keep_n=2, visualize=True))
    stop = time.perf_counter()
    time_diff = stop-start
    total_time += time_diff
    print(f'{"DBSCAN Clustering Time":<30}: {time_diff:.>5.4f} s.')
            
    groundtruth_bb_list = get_groundtruth_3d_bb(test_labels, calib)
    
    start = time.perf_counter()
    generated_bb_list = generate_3d_bb(dbscan_pcd_list, visualize=True)
    stop = time.perf_counter()
    time_diff = stop-start
    total_time += time_diff
    print(f'{"Bounding Box Generation Time":<30}: {time_diff:.>5.4f} s.')
    
    print(f'{"Total Execution Time":<30}: {total_time:.>5.4f} s.')
    
    # Visualize
    open3d.visualization.draw_geometries(dbscan_pcd_list + groundtruth_bb_list + generated_bb_list)
    
    # Bounding box generation analysis
    # Get ground truth bounding boxes, generated bounding boxes, and respective point cloud

    comp_list = []

    comp_list.append([dbscan_pcd_list[0], groundtruth_bb_list[1], generated_bb_list[0]])
    comp_list.append([dbscan_pcd_list[2], groundtruth_bb_list[3], generated_bb_list[2]])
    comp_list.append([dbscan_pcd_list[4], groundtruth_bb_list[0], generated_bb_list[4]])
    comp_list.append([dbscan_pcd_list[7], groundtruth_bb_list[2], generated_bb_list[7]])
    comp_list.append([dbscan_pcd_list[9], groundtruth_bb_list[4], generated_bb_list[9]])
    
    compare_generation_accuracy(comp_list)



