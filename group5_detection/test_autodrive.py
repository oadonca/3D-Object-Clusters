from group5_detection import run_detection
from utils import load_ad_files, load_ad_projection_mats
import open3d
import numpy as np

# Load project mat
intrinsics_path = 'autodrive/intrinsics_zed.mat'
extrinsics_path = 'autodrive/tform5.24.mat'

calib = dict()
calib['ad_transform_mat'], calib['ad_projection_mat'] = load_ad_projection_mats(intrinsics_path, extrinsics_path)

print(calib['ad_transform_mat'])
print(calib['ad_projection_mat'])

# Load AD files
image_path = 'autodrive/sensor_data/image/image231.npy'
bb_path = 'autodrive/sensor_data/bb/image231_obj.npy'
pcd_path = 'autodrive/sensor_data/pcd/pcd231.npy'
image, bb_list, pcd = load_ad_files(image_path, bb_path, pcd_path)
    
print(bb_list)
print(pcd.shape)

pcd = np.array(pcd)

generated_3d_bb_list, clustered_kitti_gt_pcd_list, detection_info, detection_metrics = run_detection(calib, image, pcd, bb_list, None, use_vis=False, use_mask=False)

print(detection_info[0])

if True:
    mesh_frame = open3d.geometry.TriangleMesh.create_coordinate_frame(size=2, origin=[0, 0, 0])
    open3d.visualization.draw_geometries(clustered_kitti_gt_pcd_list + generated_3d_bb_list + [mesh_frame])

pass