import matplotlib.pyplot as plt
from AB3DMOT.AB3DMOT_libs.model import AB3DMOT
from AB3DMOT.AB3DMOT_libs.utils import Config
import open3d
import json
import csv
from matplotlib import patches

from utils import *
from group5_detection import run_detection, run_tracking

def test_kitti_scenes(file_num = 0, use_vis = False, tracking = False, use_mask = False, autodrive=False):
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
            
            generated_3d_bb_list, detection_info, detection_metrics = run_detection(kitti_gt_calib, kitti_gt_image, kitti_gt_pointcloud, mr_detections, labels, use_vis, use_mask, autodrive)
            
            
            #############################################################################
            # CONVERT TO AB3DMOT FORMAT
            #############################################################################
            if tracking:
                frame_ab3dmot_format = get_ab3dmot_format(detection_info, autodrive=autodrive)
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
            object_candidate_clusters = [detection['object_candidate_cluster'] for detection in detection_info if detection['object_candidate_cluster'] is not None]
            if use_vis:
                mesh_frame = open3d.geometry.TriangleMesh.create_coordinate_frame(size=2, origin=[0, 0, 0])
                open3d.visualization.draw_geometries(object_candidate_clusters + kitti_gt_3d_bb + generated_3d_bb_list + [mesh_frame])

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
    test_metrics["avg_scene_time"] = float(test_metrics["avg_scene_time"])/len(file_num)

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


def test_autodrive_scenes(file_num = 0, use_vis = False, tracking = False, use_mask = False):
    # Load project mat
    intrinsics_path = 'autodrive/intrinsics_zed.mat'
    extrinsics_path = 'autodrive/tform5.24.mat'
    
    calib = dict()
    calib['ad_transform_mat'], calib['ad_projection_mat'] = load_ad_projection_mats(intrinsics_path, extrinsics_path)

    vis = open3d.visualization.Visualizer()
    vis.create_window()

    geom_added = False
    

    for file_num in range(200,401):        
        # Load AD files
        image_path = f'autodrive/sensor_data/image/image{file_num}.npy'
        bb_path = f'autodrive/sensor_data/bb/{file_num}_obj_list.npy'
        pcd_path = f'autodrive/sensor_data/pcd/pcd{file_num}.npy'
        image, bb_list, pcd = load_ad_files(image_path, bb_path, pcd_path)

        # fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        # ax.imshow(image)
        # for bb in bb_list:
        #     ax.add_patch(patches.Rectangle((int(bb[0]), int(bb[1])), int(bb[2])-int(bb[0]), int(bb[3])-int(bb[1]), fill=False))
        # plt.show()

        pcd = np.array(pcd)

        start = time.perf_counter()
        generated_3d_bb_list, detection_info, detection_metrics = run_detection(calib, image, pcd, bb_list, None, use_vis=False, use_mask=False)
        print('TOTAL RUN_DETECTION TIME: ', time.perf_counter() - start)
        
        object_candidate_clusters = [detection['object_candidate_cluster'] for detection in detection_info if detection['object_candidate_cluster'] is not None]
            
        vis.clear_geometries()
        # ctr.camera_local_translate(-1, -1, 0)
        for obj_cluster, bb_3d in zip(object_candidate_clusters, generated_3d_bb_list):
            if not geom_added:
                vis.add_geometry(obj_cluster)
                vis.add_geometry(bb_3d)
            else:
                vis.update_geoemtry(obj_cluster)
                vis.update_geometry(bb_3d)

        time.sleep(.1)                
        vis.poll_events()
        vis.update_renderer()

        if False:
            mesh_frame = open3d.geometry.TriangleMesh.create_coordinate_frame(size=2, origin=[0, 0, 0])
            open3d.visualization.draw_geometries(object_candidate_clusters + generated_3d_bb_list + [mesh_frame])

    vis.destroy_window()

    if True:
        frame = 0
        cfg = Config('./AB3DMOT/configs/cepton.yml')[0]
        classes = cfg.cat_list
        time_str = time.time()
        log = 'log/log_%s.txt' % time_str
        log = open(log, 'w')

        # Trackers must only be initialized once per class, move to global scope if function will be called multiple times
        tracker_dict = dict()
        ID_start = 1
        for label in classes:
            tracker_dict[label] = AB3DMOT(cfg, label, calib=None, oxts=None, img_dir=None, vis_dir=None, hw=None, log=log, ID_init=ID_start)
            ID_start += 1000

        trk_results_dict = run_tracking(detection_info, classes, tracker_dict, frame, True)
        frame += 1
    pass
