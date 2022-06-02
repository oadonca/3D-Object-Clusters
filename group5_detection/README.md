# AutoDrive-ObjectDetection
Running detection for AutoDrive

1. Import run_detection() function from group5_detection.py
```
from group5_detection import run_detection
```

## Call function
```
generated_3d_bb_list, detection_info, detection_metrics = run_detection(calib, image, pcd, bb_list, None, use_vis=False, use_mask=False)
```

##### run_detection inputs
- **calib**: Python dictionary containing the calibration matrices
  - ```calib['ad_transform_mat'] = <Loaded tformM.dd.mat file>```
    - Example:
        [[ 0.99949539  0.01640254  0.02720155  0.        ]
        [-0.02679296 -0.02463567  0.99933739  0.        ]
        [ 0.0170618  -0.99956192 -0.02418377  0.        ]
        [ 0.08124735  0.0277852  -0.21577977  1.        ]]
  - ```calib['ad_projection_mat'] = <Loaded intrinsics_zed.mat file>```
    - Example:
        [[1.01722412e+03 0.00000000e+00 0.00000000e+00]
        [0.00000000e+00 1.01930326e+03 0.00000000e+00]
        [9.64567972e+02 5.49720343e+02 1.00000000e+00]]

- **image**: Image of the scene as a NumPy array
- **pcd**: (Nx3) PointCloud of the scene as NumPy array
- **bb_list**: List of 2D detections from 2D detector backbone
  - Each detection in the list is of the format ```[x1, y1, x2, y2, class, confidence]```
  - Example of scene with 7 detections:
    - [['1494' '241' '1534' '328' 'traffic light' '0.59765625']
      ['691' '539' '715' '562' 'traffic sign' '0.4794921875']
      ['1012' '553' '1032' '577' 'traffic sign' '0.46044921875']
      ['816' '483' '831' '508' 'traffic light' '0.436279296875']
      ['503' '589' '818' '691' 'car' '0.41015625']
      ['1559' '531' '1586' '561' 'traffic sign' '0.409423828125']
      ['817' '422' '838' '449' 'traffic sign' '0.40283203125']]

##### run_detection outputs
- **generated_3d_bb_list**: List of Open3D Bounding Boxes (Can be AxisAlignedBoundingBox or OrientedBoundingBox)
- **detection_info**: List of Dictionaries containing comprehensive detection results for each detection in bb_list
  - Each dictionary within the list is of the following format:
    - ```detection_info['bb']: 2D bounding box```
    - ```detection_info['class']: Detection class```
    - ```detection_info['confidence']: 2D detection confidence```
    - ```detection_info['frustum_pcd']: PointCloud as np array, contains all points within the detections frustum```
    - ```detection_info['object_candidate_cluster']: PointCloud as np array, contains all points that belong to the 3D object```
    - ```detection_info['generated_3d_bb']: Open3D 3D bounding box```
    - ```detection_info['closest_face_center']: The closest face center of the 3D bounding box```
    - ```detection_info['closest_face_center_distance']: The distance to the closest face center of the 3D bounding box```
- **detection_metrics**: Dictionary containing inference speed metrics

##### Example of running detection
```
from group5_detection import run_detection
from utils import load_ad_files, load_ad_projection_mats
import open3d
import numpy as np

# Load project mat
intrinsics_path = 'autodrive/intrinsics_zed.mat'
extrinsics_path = 'autodrive/tform5.24.mat'

calib = dict()
calib['ad_transform_mat'], calib['ad_projection_mat'] = load_ad_projection_mats(intrinsics_path, extrinsics_path)

# Load AD files
image_path = 'autodrive/sensor_data/image/image231.npy'
bb_path = 'autodrive/sensor_data/bb/image231_obj.npy'
pcd_path = 'autodrive/sensor_data/pcd/pcd231.npy'
image, bb_list, pcd = load_ad_files(image_path, bb_path, pcd_path)

pcd = np.array(pcd)

start = time.perf_counter()
generated_3d_bb_list, detection_info, detection_metrics = run_detection(calib, image, pcd, bb_list, None, use_vis=False, use_mask=False)
print('TOTAL RUN_DETECTION TIME: ', time.perf_counter() - start)
    
object_candidate_clusters = [detection['object_candidate_cluster'] for detection in detection_info if detection['object_candidate_cluster'] is not None]
    
if True:
    mesh_frame = open3d.geometry.TriangleMesh.create_coordinate_frame(size=2, origin=[0, 0, 0])
    open3d.visualization.draw_geometries(object_candidate_clusters + generated_3d_bb_list + [mesh_frame])

pass

```

##### run_tracking inputs
- **detection_info**: List of dictionaries containing the detection results from run_detection (see above)
- **classes**: List of classes to perform tracking for
  - Example: ['Pedestrian', 'Car', 'Animal']
- **tracking_dict**: Dict of trackers for each class, must be initialized once per sequence per class
  - Trackers are of type AB3DMOT
    - Example: `tracker_dict[label] = AB3DMOT(cfg, label, calib=None, oxts=None, img_dir=None, vis_dir=None, hw=None, log=log, ID_init=ID_start)`
    - Only cfg, label, log, and ID_init are necessary for tracking without visualization
    - label is a string representation of a class
    - log is a writeable file
    - cfg must be a dict containing at minimum these values:
      - det_name: 3D detection method: `[pointrcnn, maskrcnn, bbox]`
      - dataset: Dataset being operated on, select 'cepton' for autodrive competition: `[KITTI, cepton, nuScenes]`
    - ID_init: Starting tracking ID number. Care should be taken when initializing trackers for different classes, as this number should
      be very different per tracker to avoid overlap of IDs
- **frame**: Frame number, should match the frame number of incoming detections
- **autodrive**: Boolean signaling to tracking to map classes for autodrive competition or not

##### run_tracking outputs
- **results**: List of Dictionaries containing detection and tracking results (Note: tracking filters out some inputted detections based on heuristics)
  - Each dictionary within the list is of the following format:
    - ```detection_info['bb']: 2D bounding box```
    - ```detection_info['class']: Detection class```
    - ```detection_info['confidence']: 2D detection confidence```
    - ```detection_info['frustum_pcd']: PointCloud as np array, contains all points within the detections frustum```
    - ```detection_info['object_candidate_cluster']: PointCloud as np array, contains all points that belong to the 3D object```
    - ```detection_info['generated_3d_bb']: Open3D 3D bounding box```
    - ```detection_info['closest_face_center']: The closest face center of the 3D bounding box```
    - ```detection_info['closest_face_center_distance']: The distance to the closest face center of the 3D bounding box```
    - ```detection_info['score']: 3D detection confidence```
    - ```detection_info['trk_id']: Tracking ID for 3D object```
    - ```detection_info['alpha']: Object observation angle, ranges from -pi to pi```
    - ```detection_info['velocity']: 1 x 3 array containing predicted velocity vector of object```
    - ```detection_info['tracking_3d_bb']: 3D bounding box in another format, 1 x 7 array containing h,w,l,x,y,z,rot_y```
    

##### Example of running tracking
```
generated_3d_bb_list, detection_info, detection_metrics = run_detection(calib, image, pcd, bb_list, None, use_vis=False, use_mask=False)

cfg = {
  'det_name': 'bbox',
  'dataset': 'cepton'
}
classes = ['Pedestrian', 'Car', 'Animal']
time_str = time.time()
log = 'log/log_%s.txt' % time_str
log = open(log, 'w')
tracker_dict = dict()
ID_start = 1
results = []
for label in classes:
    tracker_dict[label] = AB3DMOT(cfg, label, calib=None, oxts=None, img_dir=None, vis_dir=None, hw=None, log=log, ID_init=ID_start)
    ID_start += 1000
frame=0
results = run_tracking(detection_info, classes, tracker_dict, frame, True)
