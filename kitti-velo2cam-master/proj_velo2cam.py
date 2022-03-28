import sys
import matplotlib
from matplotlib import projections
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from mpl_toolkits import mplot3d
import numpy as np
from sklearn.cluster import DBSCAN
import open3d as o3d

def normalize(data):
	normalizedData = ((data - np.min(data)) / (np.max(data) - np.min(data)))
	normalizedData = np.reshape(normalizedData, (-1,1))
	print(np.repeat(normalizedData, 3, 1).shape)
	return np.repeat(normalizedData, 3, 1)

sn = int(sys.argv[1]) if len(sys.argv)>1 else 7 #default 0-7517
name = '%06d'%sn # 6 digit zeropadding
img = f'./data_object_image_2/testing/image_2/{name}.png'
binary = f'./data_object_velodyne/testing/velodyne/{name}.bin'
with open(f'./testing/calib/{name}.txt','r') as f:
    calib = f.readlines()

# P2 (3 x 4) for left eye
P2 = np.matrix([float(x) for x in calib[2].strip('\n').split(' ')[1:]]).reshape(3,4)
R0_rect = np.matrix([float(x) for x in calib[4].strip('\n').split(' ')[1:]]).reshape(3,3)
# Add a 1 in bottom-right, reshape to 4 x 4
R0_rect = np.insert(R0_rect,3,values=[0,0,0],axis=0)
R0_rect = np.insert(R0_rect,3,values=[0,0,0,1],axis=1)
Tr_velo_to_cam = np.matrix([float(x) for x in calib[5].strip('\n').split(' ')[1:]]).reshape(3,4)
Tr_velo_to_cam = np.insert(Tr_velo_to_cam,3,values=[0,0,0,1],axis=0)

# read raw data from binary
scan = np.fromfile(binary, dtype=np.float32).reshape((-1,4))
points = scan[:, 0:3] # lidar xyz (front, left, up)
# TODO: use fov filter? 
velo = np.insert(points,3,1,axis=1).T
velo = np.delete(velo,np.where(velo[0,:]<0),axis=1)
cam = P2 * R0_rect * Tr_velo_to_cam * velo
cam = np.delete(cam,np.where(cam[2,:]<0)[1],axis=1)
# get u,v,z
cam[:2] /= cam[2,:]
# do projection staff
plt.figure(figsize=(12,5),dpi=96,tight_layout=True)
png = mpimg.imread(img)
IMG_H,IMG_W,_ = png.shape
# restrict canvas in range
plt.axis([0,IMG_W,IMG_H,0])
plt.imshow(png)
# filter point out of canvas
u,v,z = cam
u_out = np.logical_or(u<0, u>IMG_W)
v_out = np.logical_or(v<0, v>IMG_H)
outlier = np.logical_or(u_out, v_out)
cam = np.delete(cam,np.where(outlier),axis=1)

eps=1.4
min_samples=40
# # labels = DBSCAN(eps=eps, min_samples=min_samples).fit(np.delete(velo,np.where(outlier),axis=1).T[:, 0:3]).labels_
# labels = DBSCAN(eps=eps, min_samples=min_samples).fit(velo.T[:, 0:3]).labels_

u,v,z = cam
# ax = plt.axes(projection='3d')
# plt.scatter([u],[v],c=labels,cmap='rainbow_r',alpha=0.5,s=2)
# ax.scatter3D(velo.T[:,0],velo.T[:,1],velo.T[:,2],c=labels)
pcd = o3d.geometry.PointCloud()
# pcd = o3d.io.read_point_cloud('./data_object_velodyne/testing/velodyne/10.pcd')
# vals, counts = np.unique(pcd.points, return_counts=True)
# points_no_ground = np.delete(pcd.points, np.where(pcd.points<vals[counts.argmax()]+.5), axis=0)
vals, counts = np.unique(points[:,2], return_counts=True)
# points_no_ground = np.delete(points, np.where(points[:,2]<vals[counts.argmax()]+.5), axis=0)
pcd.points = o3d.utility.Vector3dVector(points[:,0:3])
plane_model, inliers = pcd.segment_plane(distance_threshold=0.01,ransac_n=3, num_iterations=1000)
points_no_ground = np.delete(points[:,0:3], inliers, axis=0)
pcd.points = o3d.utility.Vector3dVector(points_no_ground)
# print(np.asarray(max(pcd.cluster_dbscan(eps=eps, min_points=min_samples, print_progress=False))))
pcd.colors = o3d.utility.Vector3dVector(plt.get_cmap("tab20")(np.asarray(pcd.cluster_dbscan(eps=eps, min_points=min_samples, print_progress=False)) + 1)[:, 0:3])
# pcd.colors = o3d.utility.Vector3dVector(normalize(pcd.cluster_dbscan(eps=eps, min_points=min_samples, print_progress=False)))
o3d.visualization.draw_geometries([pcd])
# plt.title(name)
# plt.savefig(f'./data_object_image_2/testing/projection/{name}.png',bbox_inches='tight')
# plt.show()

