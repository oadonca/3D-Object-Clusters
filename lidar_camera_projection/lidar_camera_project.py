import os

import matplotlib.pyplot as plt
import open3d

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


def render_lidar_with_boxes(pc_velo, objects, calib, img_width, img_height):
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


def render_lidar_on_image(pts_velo, img, calib, img_width, img_height):
    
    orig_img = np.copy(img)
    # projection matrix (project from velo2cam2)
    proj_velo2cam2 = project_velo_to_cam2(calib)

    print('PTSVELO', pts_velo.shape)

    # apply projection
    pts_2d = project_to_image(pts_velo.transpose(), proj_velo2cam2)
    print('PTS2D:', pts_2d.shape)
    # Can extrapole this process to remove lidar points that are outside of 2D bounding boxes
    # Filter lidar points to be within image FOV
    inds = np.where((pts_2d[0, :] < img_width) & (pts_2d[0, :] >= 0) &
                    (pts_2d[1, :] < img_height) & (pts_2d[1, :] >= 0) &
                    (pc_velo[:, 0] > 0)
                    )[0]

    # Filter out pixels points
    imgfov_pc_pixel = pts_2d[:, inds]
    print('imgfov_pc_pixel:', imgfov_pc_pixel.shape)
    
    # Retrieve depth from lidar
    imgfov_pc_velo = pts_velo[inds, :]
    print('imgfov_pc_velo:', imgfov_pc_velo.shape)
    
    # make homoegenous
    imgfov_pc_velo = np.hstack((imgfov_pc_velo, np.ones((imgfov_pc_velo.shape[0], 1))))
    print('imgfov_pc_velo:', imgfov_pc_velo.shape)

    imgfov_pc_cam2 = proj_velo2cam2 @ imgfov_pc_velo.transpose()
    print('imgfov_pc_cam2:', imgfov_pc_cam2.shape)

    print(imgfov_pc_cam2.shape)

    cmap = plt.cm.get_cmap('hsv', 256)
    cmap = np.array([cmap(i) for i in range(256)])[:, :3] * 255

    for i in range(imgfov_pc_pixel.shape[1]):
        depth = imgfov_pc_cam2[2, i]
        color = cmap[int(640.0 / depth), :]
        cv2.circle(img, (int(np.round(imgfov_pc_cam2[0, i]/imgfov_pc_cam2[2, i])),
                         int(np.round(imgfov_pc_cam2[1, i]/imgfov_pc_cam2[2, i]))),
                   2, color=tuple(color), thickness=-1)
    
    # print(imgfov_pc_cam2/imgfov_pc_cam2[2, :])
    
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))

    ax[0].set_title('Original Image')
    ax[0].imshow(orig_img)

    ax[1].set_title('Image w/ Projected Points')
    ax[1].imshow(img)
    
    ax[2].set_title('Projected Points')
    ax[2].scatter(imgfov_pc_pixel[0, :], max(imgfov_pc_pixel[1, :])-imgfov_pc_pixel[1, :])

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
    
    # Get image to cam 2 frame transform
    image_transform = project_image_to_cam2(calib)
    
    pts_2d = projected_points/projected_points[2, :]
    
    # Get only the points within bounding boxes
    inds = within_bb_indices(bb_list, pts_2d)
    print(inds)
    bb_pts_2d = pts_2d[:, inds]
    
    plt.scatter(bb_pts_2d[0, :], bb_pts_2d[1, :])
    plt.show()
    
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

if __name__ == '__main__':
    # Load image, calibration file, label bbox
    rgb = cv2.cvtColor(cv2.imread(os.path.join('data/000114_image.png')), cv2.COLOR_BGR2RGB)
    img_height, img_width, img_channel = rgb.shape

    # Load calibration
    calib = read_calib_file('data/000114_calib.txt')

    # Load labels
    labels = load_label('data/000114_label.txt')

    # Load Lidar PC
    pc_velo = load_velo_scan('data/000114.bin')[:, :3]

    # render_image_with_boxes(rgb, labels, calib)
    # render_lidar_with_boxes(pc_velo, labels, calib, img_width=img_width, img_height=img_height)
    im, points = render_lidar_on_image(pc_velo, rgb, calib, img_width, img_height)
    # render_pointcloud_from_projection(im, points, calib, img_width, img_height)
    velo_points = render_pointcloud_from_bb_segmentation(im, points, calib, img_width, img_height)
    render_lidar_with_boxes(velo_points.transpose(), labels, calib, img_width=img_width, img_height=img_height)
