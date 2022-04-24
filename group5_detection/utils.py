import cv2
import mayavi.mlab as mlab
import numpy as np
import open3d
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon
from matplotlib import patches
from matplotlib import pyplot as plt
import multiprocessing as mp

class Box3D(object):
    """
    Represent a 3D box corresponding to data in label.txt
    """

    def __init__(self, label_file_line):
        data = label_file_line.split(' ')
        data[1:] = [float(x) for x in data[1:]]

        self.type = data[0]
        self.truncation = data[1]
        self.occlusion = int(data[2])  # 0=visible, 1=partly occluded, 2=fully occluded, 3=unknown
        self.alpha = data[3]  # object observation angle [-pi..pi]

        # extract 2d bounding box in 0-based coordinates
        self.xmin = data[4]  # left
        self.ymin = data[5]  # top
        self.xmax = data[6]  # right
        self.ymax = data[7]  # bottom
        self.box2d = np.array([self.xmin, self.ymin, self.xmax, self.ymax])

        # extract 3d bounding box information
        self.h = data[8]  # box height
        self.w = data[9]  # box width
        self.l = data[10]  # box length (in meters)
        self.t = (data[11], data[12], data[13])  # location (x,y,z) in camera coord.
        self.ry = data[14]  # yaw angle (around Y-axis in camera coordinates) [-pi..pi]

    def in_camera_coordinate(self, is_homogenous=False):
        # 3d bounding box dimensions
        l = self.l
        w = self.w
        h = self.h

        # 3D bounding box vertices [3, 8]
        x = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
        y = [0, 0, 0, 0, -h, -h, -h, -h]
        z = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]
        box_coord = np.vstack([x, y, z])

        # Rotation
        R = roty(self.ry)  # [3, 3]
        points_3d = R @ box_coord

        # Translation
        points_3d[0, :] = points_3d[0, :] + self.t[0]
        points_3d[1, :] = points_3d[1, :] + self.t[1]
        points_3d[2, :] = points_3d[2, :] + self.t[2]

        if is_homogenous:
            points_3d = np.vstack((points_3d, np.ones(points_3d.shape[1])))

        return points_3d


# =========================================================
# Projections
# =========================================================
def project_velo_to_cam2(calib):
    P_velo2cam_ref = np.vstack((calib['Tr_velo_to_cam'].reshape(3, 4), np.array([0., 0., 0., 1.])))  # velo2ref_cam
    R_ref2rect = np.eye(4)
    R0_rect = calib['R0_rect'].reshape(3, 3)  # ref_cam2rect
    R_ref2rect[:3, :3] = R0_rect
    P_rect2cam2 = calib['P2'].reshape((3, 4))
    proj_mat = P_rect2cam2 @ R_ref2rect @ P_velo2cam_ref
    return proj_mat

def project_image_to_cam2(calib):
    P_rect2cam2 = calib['P2'].reshape((3, 4))
    # Create square matrix
    P_rect2cam2 = np.vstack((P_rect2cam2, np.array([0., 0., 0., 1.])))
    # Get rectified frame to cam2 image frame inverse i.e. cam2 image to rectified transform
    P_rect2cam2_inv = np.linalg.inv(P_rect2cam2)
    
    return P_rect2cam2_inv

def project_cam2_to_velo(calib):
    R_ref2rect = np.eye(4)
    R0_rect = calib['R0_rect'].reshape(3, 3)  # ref_cam2rect
    R_ref2rect[:3, :3] = R0_rect
    R_ref2rect_inv = np.linalg.inv(R_ref2rect)  # rect2ref_cam

    # inverse rigid transformation
    velo2cam_ref = np.vstack((calib['Tr_velo_to_cam'].reshape(3, 4), np.array([0., 0., 0., 1.])))  # velo2ref_cam
    P_cam_ref2velo = np.linalg.inv(velo2cam_ref)

    proj_mat = P_cam_ref2velo @ R_ref2rect_inv
    return proj_mat


def project_to_image(points, proj_mat):
    """
    Apply the perspective projection
    Args:
        pts_3d:     3D points in camera coordinate [3, npoints]
        proj_mat:   Projection matrix [3, 4]
    """
    num_pts = points.shape[1]

    # Change to homogenous coordinate
    points = np.vstack((points, np.ones((1, num_pts))))

    # Apply projection
    points = proj_mat @ points

    # Convert back to nonhomogenous coordinates
    points[:2, :] /= points[2, :]
    return points[:2, :]


def project_camera_to_lidar(points, proj_mat):
    """
    Args:
        points:     3D points in camera coordinate [3, npoints]
        proj_mat:   Projection matrix [3, 4]

    Returns:
        points in lidar coordinate:     [3, npoints]
    """
    num_pts = points.shape[1]
    # Change to homogenous coordinate
    points = np.vstack((points, np.ones((1, num_pts))))
    points = proj_mat @ points
    return points[:3, :]


def map_box_to_image(box, proj_mat):
    """
    Projects 3D bounding box into the image plane.
    Args:
        box (Box3D)
        proj_mat: projection matrix
    """
    # box in camera coordinate
    points_3d = box.in_camera_coordinate()

    # project the 3d bounding box into the image plane
    points_2d = project_to_image(points_3d, proj_mat)

    return points_2d


# =========================================================
# Utils
# =========================================================
def load_label(label_filename):
    lines = [line.rstrip() for line in open(label_filename)]
    # load as list of Object3D
    objects = [Box3D(line) for line in lines]
    return objects


def load_image(img_filename):
    return cv2.imread(img_filename)


def load_velo_scan(velo_filename):
    scan = np.fromfile(velo_filename, dtype=np.float32)
    scan = scan.reshape((-1, 4))
    return scan


def read_calib_file(filepath):
    """
    Read in a calibration file and parse into a dictionary.
    Ref: https://github.com/utiasSTARS/pykitti/blob/master/pykitti/utils.py
    """
    data = {}
    with open(filepath, 'r') as f:
        for line in f.readlines():
            line = line.rstrip()
            if len(line) == 0: continue
            key, value = line.split(':', 1)
            # The only non-float values in these files are dates, which
            # we don't care about anyway
            try:
                data[key] = np.array([float(x) for x in value.split()])
            except ValueError:
                pass

    return data

def remove_extra_coco_detections(bbox, segmentation, label):
    CLASSES = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
               'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
               'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
               'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
               'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
               'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
               'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
               'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
               'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
               'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
               'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
               'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
               'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
               'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
    indices = np.argwhere((label == CLASSES.index('person')) | (label == CLASSES.index('car')) | (label == CLASSES.index('motorcycle')) | (label == CLASSES.index('bus')) | (label == CLASSES.index('truck'))).flatten()
    removed_bbox = bbox[indices]
    removed_segmentation = segmentation[indices]
    removed_label = label[indices]
    
    return removed_bbox, removed_segmentation, removed_label

def load_kitti_groundtruth(file_num = 0):
    calibs = []
    images = []
    labels = []
    pointclouds = []
    if isinstance(file_num, int):
        try:
            calibs = read_calib_file(f'kitti_groundtruth/calib/{str(file_num).zfill(6)}.txt')
            images = cv2.cvtColor(cv2.imread(f'kitti_groundtruth/image_2/{str(file_num).zfill(6)}.png'), cv2.COLOR_BGR2RGB)
            labels = load_label(f'kitti_groundtruth/label_2/{str(file_num).zfill(6)}.txt')
            pointclouds = load_velo_scan(f'kitti_groundtruth/velodyne/{str(file_num).zfill(6)}.bin')[:, :3]
        except:
            print(f'Error loading inference artifacts for scene {str(file_num).zfill(6)}')
    elif isinstance(file_num, list):
        for i, num in enumerate(file_num):
            try:
                calibs.append(read_calib_file(f'kitti_groundtruth/calib/{str(num).zfill(6)}.txt'))
                images.append(cv2.cvtColor(cv2.imread(f'kitti_groundtruth/image_2/{str(num).zfill(6)}.png'), cv2.COLOR_BGR2RGB))
                labels.append(load_label(f'kitti_groundtruth/label_2/{str(num).zfill(6)}.txt'))
                pointclouds.append(load_velo_scan(f'kitti_groundtruth/velodyne/{str(num).zfill(6)}.bin')[:, :3])
            except:
                print(f'Error loading inference artifacts for scene {str(num).zfill(6)}')

    return images, pointclouds, labels, calibs

def load_mask_rcnn_inference(file_num = 0):
    bboxes = []
    images = []
    labels = []
    segmentations = []
    if isinstance(file_num, int):
        try:
            bboxes = np.load(f'mask_rcnn_inference/bboxes/bboxes_kitti_{str(file_num).zfill(6)}.npy')
            images = cv2.cvtColor(cv2.imread(f'mask_rcnn_inference/image/inference_test-kitti_{str(file_num).zfill(6)}.png'), cv2.COLOR_BGR2RGB)
            labels = np.load(f'mask_rcnn_inference/labels/labels_kitti_{str(file_num).zfill(6)}.npy')
            segmentations = np.load(f'mask_rcnn_inference/segmentation/segmentation_kitti_{str(file_num).zfill(6)}.npy')
            
            bboxes, segmentations, labels = remove_extra_coco_detections(bboxes, segmentations, labels)
            
        except Exception as e:
            print(f'Error loading inference artifacts for scene {str(file_num).zfill(6)}')
            print(e)
            quit()
            
    elif isinstance(file_num, list):
        for i, num in enumerate(file_num):
            try:
                bbox = np.load(f'mask_rcnn_inference/bboxes/bboxes_kitti_{str(num).zfill(6)}.npy')
                image = cv2.cvtColor(cv2.imread(f'mask_rcnn_inference/image/inference_test-kitti_{str(num).zfill(6)}.png'), cv2.COLOR_BGR2RGB)
                label = np.load(f'mask_rcnn_inference/labels/labels_kitti_{str(num).zfill(6)}.npy')
                segmentation = np.load(f'mask_rcnn_inference/segmentation/segmentation_kitti_{str(num).zfill(6)}.npy')
                
                bbox, segmentation, label = remove_extra_coco_detections(bbox, segmentation, label)
                
                bboxes.append(bbox)
                images.append(image)
                labels.append(label)
                segmentations.append(segmentation)
                
            except Exception as e:
                print(f'Error loading inference artifacts for scene {str(num).zfill(6)}')
                print(e)
                quit()

    return images, bboxes, segmentations, labels

def roty(t):
    """
    Rotation about the y-axis.
    """
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, 0, s],
                     [0, 1, 0],
                     [-s, 0, c]])
    
def within_bb_indices(bb_list, points):
    """
    Returns an list of lists of indices for each bounding box
    """
    inds = []
    
    for bb in bb_list:
        x_range = [bb[0][0], bb[1][0]]
        y_range = [bb[0][1], bb[2][1]]
        inds.append(np.where((points[0, :] < x_range[1]) & (points[0, :] >= x_range[0]) &
                        (points[1, :] < y_range[1]) & (points[1, :] >= y_range[0])
                        )[0])
        
    return inds


# =========================================================
# Drawing tool
# =========================================================
def draw_projected_box3d(image, qs, color=(255, 255, 255), thickness=1):
    qs = qs.astype(np.int32).transpose()
    for k in range(0, 4):
        # http://docs.enthought.com/mayavi/mayavi/auto/mlab_helper_functions.html
        i, j = k, (k + 1) % 4
        cv2.line(image, (qs[i, 0], qs[i, 1]), (qs[j, 0], qs[j, 1]), color, thickness, cv2.LINE_AA)

        i, j = k + 4, (k + 1) % 4 + 4
        cv2.line(image, (qs[i, 0], qs[i, 1]), (qs[j, 0], qs[j, 1]), color, thickness, cv2.LINE_AA)

        i, j = k, k + 4
        cv2.line(image, (qs[i, 0], qs[i, 1]), (qs[j, 0], qs[j, 1]), color, thickness, cv2.LINE_AA)

    return image

def bitmap_to_polygon(bitmap):
    """Convert masks from the form of bitmaps to polygons.
    Args:
        bitmap (ndarray): masks in bitmap representation.
    Return:
        list[ndarray]: the converted mask in polygon representation.
        bool: whether the mask has holes.
    """
    bitmap = np.ascontiguousarray(bitmap).astype(np.uint8)
    # cv2.RETR_CCOMP: retrieves all of the contours and organizes them
    #   into a two-level hierarchy. At the top level, there are external
    #   boundaries of the components. At the second level, there are
    #   boundaries of the holes. If there is another contour inside a hole
    #   of a connected component, it is still put at the top level.
    # cv2.CHAIN_APPROX_NONE: stores absolutely all the contour points.
    outs = cv2.findContours(bitmap, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    contours = outs[-2]
    hierarchy = outs[-1]
    if hierarchy is None:
        return [], False
    # hierarchy[i]: 4 elements, for the indexes of next, previous,
    # parent, or nested contours. If there is no corresponding contour,
    # it will be -1.
    with_hole = (hierarchy.reshape(-1, 4)[:, 3] >= 0).any()
    contours = [c.reshape(-1, 2) for c in contours]
    return contours, with_hole

def get_bias_color(base, max_dist=30):
    """Get different colors for each masks.
    Get different colors for each masks by adding a bias
    color to the base category color.
    Args:
        base (ndarray): The base category color with the shape
            of (3, ).
        max_dist (int): The max distance of bias. Default: 30.
    Returns:
        ndarray: The new color for a mask with the shape of (3, ).
    """
    new_color = base + np.random.randint(
        low=-max_dist, high=max_dist + 1, size=3)
    return np.clip(new_color, 0, 255, new_color)


# =========================================================
# Mask 3D object detection
# =========================================================
def generate_3d_bb(pcd_list, oriented=False, visualize=True):
    """
    Generates bounding boxes around each point cloud in pcd_list
    """
    generated_bb_list = []
    for pcd in pcd_list:
        if np.max(np.array(pcd.colors)) > 0:
            bb = pcd.get_axis_aligned_bounding_box() if not oriented else pcd.get_oriented_bounding_box()
            bb.color = [1.0, 0, 0]
            generated_bb_list.append(bb)
        
    display_list = pcd_list + generated_bb_list
        
    if visualize:
        open3d.visualization.draw_geometries(display_list)
        
    return generated_bb_list

def get_groundtruth_3d_bb(objects, calib, oriented = False):
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
        box = open3d.geometry.OrientedBoundingBox.create_from_points(boxes3d_pts) if oriented else open3d.geometry.AxisAlignedBoundingBox.create_from_points(boxes3d_pts)
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

def get_detector_2d_bb(mask_bboxes, image = None, visualize=True):    
    # print('MASK BOXES\n', mask_bboxes)
    bb_list = []
    for bb in mask_bboxes:
        bb_list.append([(bb[0], bb[1]), (bb[2], bb[1]), (bb[0], bb[3]), (bb[2], bb[3])])
        
    if visualize:
        fig, ax = plt.subplots(1, 1, figsize=(5,5))
        ax.imshow(image) if image is not None else None
        for bb in bb_list:
            # print(bb)
            ax.add_patch(patches.Rectangle(bb[0], bb[1][0]-bb[0][0], bb[2][1]-bb[1][1], fill=None))
        
        plt.show()
        
    return bb_list

def get_bb_centers(bb):
    """
    bb: Open3D bounding box
    
    returns a numpy array with six coordinates of different cntres of the faces of the bounding box 
    """

    bb_c = bb.get_center()

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

    return arr

def get_closest_bb_center(bb_centers):
    pass

def draw_masks(img, masks, color = None, with_edge = True, alpha = 0.8):
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.imshow(img)
    
    taken_colors = set([0, 0, 0])
    if color is None:
        random_colors = np.random.randint(0, 255, (masks.shape[0], 3))
        color = [tuple(c) for c in random_colors]
        color = np.array(color, dtype=np.uint8)
    polygons = []
    for i, mask in enumerate(masks):
        if with_edge:
            contours, _ = bitmap_to_polygon(mask)
            polygons += [Polygon(c) for c in contours]

        color_mask = color[i]
        while tuple(color_mask) in taken_colors:
            color_mask = get_bias_color(color_mask)
        taken_colors.add(tuple(color_mask))

        mask = mask.astype(bool)
        img[mask] = img[mask] * (1 - alpha) + color_mask * alpha

    p = PatchCollection(
        polygons, edgecolors='w', linewidths=1, alpha=0.8)
    ax.add_collection(p)
        
    plt.show()