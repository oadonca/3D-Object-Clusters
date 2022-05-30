import cv2
# import mayavi.mlab as mlab
import numpy as np
import open3d
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon
from matplotlib import patches
from matplotlib import pyplot as plt
import multiprocessing as mp
import time
import json
import pickle
import re
import scipy.io as sio
import os

from iou3d import box3d_iou

class Box3D(object):
    """
    Represent a 3D box corresponding to data in label.txt
    """

    def __init__(self, label_file_line):
        data = label_file_line.split(' ')[-15:]
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


class DetectionEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, open3d.geometry.AxisAlignedBoundingBox) or isinstance(obj, open3d.geometry.OrientedBoundingBox):
            return np.asarray(obj.get_box_points()).tolist()
        return json.JSONEncoder.default(self, obj)

# =========================================================
# Projections
# =========================================================
def project_velo_to_cam2(calib):
    velo_tmp = calib['Tr_velo_cam'] if 'Tr_velo_cam' in calib else calib['Tr_velo_to_cam']
    P_velo2cam_ref = np.vstack((velo_tmp.reshape(3, 4), np.array([0., 0., 0., 1.])))  # velo2ref_cam
    R_ref2rect = np.eye(4)
    if 'R_rect' in calib:
        R0_rect = calib['R_rect'].reshape(3, 3)  # ref_cam2rect
    else:
        R0_rect = calib['R0_rect'].reshape(3, 3)  # ref_cam2rect
    R_ref2rect[:3, :3] = R0_rect
    P_rect2cam2 = calib['P2'].reshape((3, 4))
    proj_mat = P_rect2cam2 @ R_ref2rect @ P_velo2cam_ref
    return proj_mat


def project_image_to_cam2(calib, autodrive):
    if autodrive:
        P_rect2cam2 = np.transpose(calib['ad_transfrom_mat'])
    else:   
        P_rect2cam2 = calib['P2'].reshape((3, 4))
        # Create square matrix
        P_rect2cam2 = np.vstack((P_rect2cam2, np.array([0., 0., 0., 1.])))
    # Get rectified frame to cam2 image frame inverse i.e. cam2 image to rectified transform
    P_rect2cam2_inv = np.linalg.inv(P_rect2cam2)
    
    return P_rect2cam2_inv


# transform
def project_cam2_to_velo(calib, autodrive=True):
    R_ref2rect = np.eye(4)
    if autodrive:
        proj_mat = np.linalg.inv(np.transpose(calib['ad_transfrom_mat']))

    else:  
        if 'R_rect' in calib:
            R0_rect = calib['R_rect'].reshape(3, 3)  # ref_cam2rect
        else:
            R0_rect = calib['R0_rect'].reshape(3, 3)  # ref_cam2rect
        R_ref2rect[:3, :3] = R0_rect
        R_ref2rect_inv = np.linalg.inv(R_ref2rect)  # rect2ref_cam

        # inverse rigid transformation
        velo_tmp = calib['Tr_velo_cam'] if 'Tr_velo_cam' in calib else calib['Tr_velo_to_cam']
        velo2cam_ref = np.vstack((velo_tmp.reshape(3, 4), np.array([0., 0., 0., 1.])))  # velo2ref_cam
        P_cam_ref2velo = np.linalg.inv(velo2cam_ref)

        proj_mat = P_cam_ref2velo @ R_ref2rect_inv
    return proj_mat


def project_to_image(points, proj_mat, autodrive=True):
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
    if autodrive:
        points = np.transpose(proj_mat) @ points
    else:
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
    print('proj mat\n\n',proj_mat)
    # Change to homogenous coordinate
    points = np.vstack((points, np.ones((1, num_pts))))
    print(points.shape)
    points = proj_mat @ points
    return points[:3, :]


def map_box_to_image(box, proj_mat, autodrive):
    """
    Projects 3D bounding box into the image plane.
    Args:
        box (Box3D)
        proj_mat: projection matrix
    """
    # box in camera coordinate
    points_3d = box.in_camera_coordinate()

    # project the 3d bounding box into the image plane
    points_2d = project_to_image(points_3d, proj_mat, autodrive)

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
            key, value = re.split(':| ', line, 1)
            # The only non-float values in these files are dates, which
            # we don't care about anyway
            try:
                data[key] = np.array([float(x) for x in value.split()])
            except ValueError:
                pass

    return data

def get_autodrive_classes():
    return ['person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle', 'traffic light', 'traffic sign', 'animal']


def get_coco_classes():
    return ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
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

def get_used_coco_classes():
    return ['person', 'car', 'bus', 'truck', 'train']


def kitti_coco_class_mapping(cls, transform=0):
    coco_classes = get_coco_classes()
    kitti_classes = ['Car', 'Van', 'Truck',
                     'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram',
                     'Misc', 'DontCare']

    kitti_to_coco = {'Car': ['car', 'bus', 'truck'], 'Van': ['car', 'bus', 'truck', 'train'], 'Truck': ['car', 'bus', 'truck', 'train'], 'Pedestrian': ['person'], 'Person_sitting': ['person'], 'Cyclist': ['person'], 'Tram': ['car', 'bus', 'truck', 'train'], 'Misc': coco_classes}
    
    return kitti_to_coco[cls]


def get_coco_class(class_idx):
    classes = get_coco_classes()
    print(class_idx)
    return classes[class_idx]


def remove_extra_coco_detections(bbox, segmentation, label):
    classes = get_coco_classes()
    indices = np.argwhere((label == classes.index('person')) | (label == classes.index('car')) | (label == classes.index('bus')) | (label == classes.index('truck')) | (label == classes.index('train'))).flatten()
    removed_bbox = bbox[indices]
    removed_segmentation = segmentation[indices]
    removed_label = label[indices]
    
    return removed_bbox, removed_segmentation, removed_label


def load_kitti_groundtruth(file_num = 0, scene = "None"):
    calibs = []
    images = []
    labels = []
    pointclouds = []
    if scene == "None":
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
    else:
        try:
            calibs = read_calib_file(f'kitti_tracking/training/calib/{str(scene).zfill(4)}.txt')
            images = cv2.cvtColor(cv2.imread(f'kitti_tracking/training/image_02/{str(scene).zfill(4)}/{str(file_num).zfill(6)}.png'), cv2.COLOR_BGR2RGB)
            labels = load_label(f'kitti_tracking/training/label_02/{str(scene).zfill(4)}.txt')
            pointclouds = load_velo_scan(f'kitti_tracking/training/velodyne/{str(scene).zfill(4)}/{str(file_num).zfill(6)}.bin')[:, :3]
        except Exception as e:
            print(e)
            print(f'Error loading inference artifacts for scene {str(scene).zfill(4)} and file {str(file_num).zfill(6)}')

    return images, pointclouds, labels, calibs


def load_mask_rcnn_inference(file_num = 0, scene = "None"):
    bboxes = []
    images = []
    labels = []
    segmentations = []
    if scene != "None":
        with open(f'mask_rcnn_inference/training/{scene.zfill(4)}/{str(file_num).zfill(6)}.pkl', 'rb') as file:
            pickleFile = pickle.load(file)
        temp_bboxes = pickleFile[0]
        temp_segmentations = pickleFile[1]
        for label in range(len(temp_bboxes)):
            for i, bbox in enumerate(temp_bboxes[label]):
                bboxes.append(bbox)
                labels.append(label)
                segmentations.append(temp_segmentations[label][i])

        bboxes = np.array(bboxes)
        segmentations = np.array(segmentations)
        labels = np.array(labels)
        bboxes, segmentations, labels = remove_extra_coco_detections(bboxes, segmentations, labels)

    else:
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


def within_bb_indices(detections, points, autodrive=True):
    """
    Returns an list of lists of indices for each bounding box
    """
    inds = []
    if autodrive:
        for detection in detections:
            x_range = [int(detection['bb'][0]), int(detection['bb'][2])]
            y_range = [int(detection['bb'][1]), int(detection['bb'][3])]            
            ind = np.where((points[0, :] < x_range[1]) & (points[0, :] >= x_range[0]) &
                            (points[1, :] < y_range[1]) & (points[1, :] >= y_range[0])
                            )[0]
            inds.append(ind)
    else:   
        for detection in detections:
            x_range = [detection['bb'][0][0], detection['bb'][1][0]]
            y_range = [detection['bb'][0][1], detection['bb'][2][1]]
            inds.append(np.where((points[0, :] < x_range[1]) & (points[0, :] >= x_range[0]) &
                            (points[1, :] < y_range[1]) & (points[1, :] >= y_range[0])
                            )[0]
                        )
        
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
def get_groundtruth_3d_bb(objects, calib, oriented, autodrive):
    """
    Gets the ground truth bounding boxes provided in objects
    """
    # Projection matrix
    proj_cam2_2_velo = project_cam2_to_velo(calib, autodrive)

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

    bb_axis_aligned = bb.get_axis_aligned_bounding_box()

    bb_c = bb_axis_aligned.get_center()

    # returns the eight bounding box coordinates of the corners; not used anywhere
    bb_coord = bb_axis_aligned.get_box_points()

    # returns the length, breadth and height divided by 2 of the bounding box 
    half_extent = bb_axis_aligned.get_half_extent()

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


def time_function(function, args = (), kwargs = {}):
    start = time.perf_counter()
    func_return = function(*args, **kwargs)
    end = time.perf_counter() - start
    return func_return, end


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


def get_ab3dmot_format(detection_info, frame=0):
    # detection keys: class, bb, mask, frustum_pcd, object_candidate_cluster, generated_3d_bb
    frame_input = list()
    for i, detection in enumerate(detection_info):
        if 'generated_3d_bb' in detection.keys() and detection['generated_3d_bb'] is not None:
            detection_input = dict()
            # Frame number
            detection_input['frame'] = frame if 'frame' not in detection else detection['frame']
            
            # Detection type/class
            if get_coco_class(detection['class']) in ['car', 'bus', 'truck', 'train']:
                detection_input['type'] = 2
            elif get_coco_class(detection['class']) in ['bicycle']:
                detection_input['type'] = 3
            elif get_coco_class(detection['class']) in ['person']:
                detection_input['type'] = 1
                
            # 2D bounding box top left and bottom right coordinates
            detection_input['2d_bb'] = [detection['bb'][0][0], detection['bb'][0][1], detection['bb'][3][0], detection['bb'][3][1]]
            
            # Detection confidence
            detection_input['score'] = .5

            # 3D bounding box: height, width, length, x, y, z, rot_y
            bb_3d = detection['generated_3d_bb']
            if not isinstance(bb_3d, open3d.geometry.AxisAlignedBoundingBox):
                bb_3d = bb_3d.get_axis_aligned_bounding_box()
                
            bb_center = bb_3d.get_center()
            bb_half_extents = bb_3d.get_half_extent()
            
            front = np.array([bb_center[0] + bb_half_extents[0], bb_center[1], bb_center[2]])
            back = np.array([bb_center[0] - bb_half_extents[0], bb_center[1], bb_center[2]])
            
            direction_vector = front - back
            rot_z = np.arctan2(direction_vector[1], direction_vector[0])
            
            detection_input['3d_bb'] = [2*bb_half_extents[2], 2*bb_half_extents[1], 2*bb_half_extents[0], bb_center[0], bb_center[1], bb_center[2], rot_z]
            
            # Alpha, viewing angle
            detection_input['alpha'] = np.arctan2(bb_center[1], bb_center[0])
            
            final_input = []
            for val in detection_input.values():
                if isinstance(val, list):
                    final_input.extend(val)
                else:
                    final_input.append(val)
            
            detection_info[i]['ab3dmot_format'] = final_input
            frame_input.append(final_input)
        
    return frame_input


def load_ad_projection_mats(intrinsics_path, extrinsics_path):
    intrinsics_mat = sio.loadmat(intrinsics_path, squeeze_me=True)['intrinsics'].item()[6]
    try:
        extrinsics_mat = sio.loadmat(extrinsics_path, squeeze_me=True)['tform'].item()[1]
    except:
        print('warning: failed to read extrinsics matrix, trying again')
    
    try:
        extrinsics_mat = np.load(extrinsics_path)
    except:
        pass
    
    return extrinsics_mat, intrinsics_mat


def load_ad_files(image_path, bb_path, pcd_path):
    image = np.load(image_path)
    bb_list = np.load(bb_path)
    pcd = np.load(pcd_path)
    
    return image, bb_list, pcd


def prepare_tracking_files(scene_list):
    scene_dict = {}
    for scene in scene_list:
        files = sorted(os.listdir(f'kitti_tracking/training/velodyne/{str(scene).zfill(4)}'))
        files = [int(os.path.splitext(file)[0]) for file in files]
        scene_dict[str(scene)] = files
    return scene_dict
    

def limit_pcd_depth(pcd, depth_limit):
    inds = np.where(((pcd[:,0]**2)+(pcd[:,1]**2)+(pcd[:,2]**2))**.5 < depth_limit)
    return pcd[inds]

def get_autodrive_score_info(detection_info):
    for i, detection in enumerate(detection_info):
        score_info_det = dict()
        
        # Calculate and return closest face center
        face_centers = get_bb_centers(detection['generated_3d_bb'])
        score_info_det['closest_face_center'] = None
        score_info_det['closest_face_center_distance'] = float('inf')
        for face_center in face_centers:
            distance = np.linalg.norm(face_center - np.array([0, 0, 0]))
            if distance < detection_info['i']['closest_face_center_distance']:
                score_info_det['closest_face_center_distance'] = distance
                score_info_det['closest_face_center'] = face_center
                
        detection_info[i] = {**detection_info[i], **score_info_det}

        
