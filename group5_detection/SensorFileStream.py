import cv2
import numpy as np
import scipy.io as sio
import open3d as o3d
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import time
from numpy.linalg import inv
import numpy as np
import math as m

from scipy.linalg import solve
from scipy.spatial.transform import Rotation


def Rx(theta):
    rx = np.eye(4)
    R = Rotation.from_euler("XYZ",[theta,0,0], degrees=True).as_matrix()
    rx[:3,:3] = R
    return rx
  
def Ry(theta):
    ry = np.eye(4)
    R = Rotation.from_euler("XYZ",[0,theta,0], degrees=True).as_matrix()
    ry[:3,:3] = R
    return ry
  
def Rz(theta):
    rz = np.eye(4)
    R = Rotation.from_euler("XYZ",[0,0,theta], degrees=True).as_matrix()
    rz[:3,:3] = R
    return rz

def Tx(trans_x):
    tx = np.eye(4)
    tx[0,3] = trans_x
    return tx

def Ty(trans_y):
    ty = np.eye(4)
    ty[1,3] = trans_y
    return ty

def Tz(trans_z):
    tz = np.eye(4)
    tz[2,3] = trans_z
    return tz


def generate_pcd_color_map(xyz):

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    # Make data.
    X = xyz[0,:]
    Y = xyz[1,:]
    Z = xyz[2,:]
    # Plot the surface.
    surf = ax.scatter(X,Y,Z,cmap=cm.hsv,c=Y)

    color_ndarray = surf.get_facecolors()[:,:3].tolist()
    color_ndarray = np.array(color_ndarray)*255

    
    return color_ndarray


def bboxTOpcd(bbox,uvxyz):

    x,y,w,h = bbox

    mask_W = uvxyz[0,:] < x+w 
    uvxyz = uvxyz[:,mask_W]
    
    mask_x = uvxyz[0,:] > x
    uvxyz = uvxyz[:,mask_x]
    
    mask_h = uvxyz[1,:] < y+h
    uvxyz = uvxyz[:,mask_h]

    mask_y = uvxyz[1,:] > y
    uvxyz = uvxyz[:,mask_y]

    return uvxyz

def load_camera_intrinsics_mat(path):

    intrinsics = sio.loadmat(path, squeeze_me=True) 
    intrinsics = intrinsics['intrinsics'].item()[6]
    return intrinsics

def load_tform_mat(path):

    trans = sio.loadmat(path, squeeze_me=True) 
    tform = trans['tform'].item()[1]
    return tform

def pointCloudTOuvxyz(pcd,intrinsics,tform):
    first = time.perf_counter()
    
    pointCloudXYZ = np.array(pcd.points)
    print('PointCloud XYZ: ', pointCloudXYZ.shape)

    #mask_W = pointCloudXYZ[:,1] < 50 
    #pointCloudXYZ = pointCloudXYZ[mask_W,:]

    ones = np.ones((pointCloudXYZ.shape[0],1))

    pointCloudXYZ_pad_one = np.hstack((pointCloudXYZ,ones))
    print('PointCloud XYZ pad One: ', pointCloudXYZ_pad_one.shape)


    print('tform: \n', tform.T)
    cameraXYZ = np.dot(tform.T,pointCloudXYZ_pad_one.T)

    print('cameraXYZ before delete: ', cameraXYZ.shape)
    cameraXYZ = np.delete(cameraXYZ, 3, 0)
    print('cameraXYZ after delete: ', cameraXYZ.shape)

    print('intrinsics: \n', intrinsics.T)
    image_UV = np.dot(intrinsics.T,cameraXYZ)

    print('image_uv: ', image_UV.shape)
    print(image_UV/image_UV[2])
    u = image_UV[0,:]/image_UV[2,:]
    v = image_UV[1,:]/image_UV[2,:]

    pointCloudXYZ = pointCloudXYZ.T

    px = pointCloudXYZ[0][np.logical_not(np.isnan(u))]
    py = pointCloudXYZ[1][np.logical_not(np.isnan(u))]
    pz = pointCloudXYZ[2][np.logical_not(np.isnan(u))]

    u = u[np.logical_not(np.isnan(u))]
    v = v[np.logical_not(np.isnan(v))]

    u = u.astype(int)
    v = v.astype(int)

    uv, xyz = np.array([u,v]), np.array([px,py,pz])
    print("mapping time:",time.perf_counter()-first)
    return uv, xyz

def uvxyzTOxyz(store_point,uv,xyz):


    mask = abs(uv[0,:] - store_point[0] ) < 2

    xyz = xyz[:,mask]
    uv = uv[:,mask]

    mask = abs(uv[1,:] - store_point[1] ) < 2

    xyz = xyz[:,mask]
    xyz = np.mean(xyz,axis=1)

    print("true",xyz)

    return xyz

def uvTOxyz(uv,intrinsics,tform,pcd):
    
    points = pcd.points
    points = np.array(points)

    #intrinsics = intrinsics.T

    fx = intrinsics[0][0]
    cx = intrinsics[2][0]
    fy = intrinsics[1][1]
    cy = intrinsics[2][1]

    zLim = 3.7

    xlim = ((uv[0] - cx)*zLim)/fx
    ylim = ((uv[1] - cy)*zLim)/fy

    worldCam = np.array([[xlim,ylim,zLim,1]])

   
    tform = np.linalg.inv(tform.T)
    xyz = np.dot(tform,worldCam.T)





    xyz = xyz[:3]

    mask = abs(points[:,0] - xyz[0] ) < 0.05 

    points = points[mask,:]
    mask = abs(points[:,2] - xyz[2] ) < 0.05 

    points = points[mask,:]


    print("xyz",xyz)
    #print("lidar points", points)

    

    return xyz

def mouse_click(event, x, y,flags, param):
    global store_point 
    # to check if left mouse 
    # button was clicked
    if event == cv2.EVENT_LBUTTONDOWN:
        print("px py:"+str(x)+" "+str(y))
        store_point.append([x,y])

        #uvTOxyz(store_point[-1],intrinsics,tform,pcd)
        uvxyzTOxyz(store_point[-1],uv,xyz)

        

    if event == cv2.EVENT_RBUTTONDOWN:
        store_point = []



def draw_point(image, u,v,col):

    for i in range(len(u)-1):
        cv2.circle(image, (int(u[i]),int(v[i])), radius=1, color=(int(col[i][0]), int(col[i][1]), int(col[i][2])), thickness=-1)
    
    return image

def convert_xyzTObirdseyeview(store_point):

    mapping_list = []
    for i in range(len(store_point)):
        mapping_list.append(uvxyzTOxyz(store_point[i],uv,xyz))


    return mapping_list


path = r'autodrive/tform5.24.mat'
trans = sio.loadmat(path, squeeze_me=True) 
tform = trans['tform'].item()[1]



# tform = np.load("tform_fine_tune.npy")
#tform = tform.T



path = r'autodrive/intrinsics_zed.mat'
intrinsics = sio.loadmat(path, squeeze_me=True) 
intrinsics = intrinsics['intrinsics'].item()[6]



pcd_file_path = r"autodrive/sensor_data/pcd/pcd"
img_file_path = r"autodrive/sensor_data/image/image"

pcd_npy = np.load(pcd_file_path + r"231.npy")


downsize = 0.08

pcd = o3d.geometry.PointCloud()

pcd.points = o3d.utility.Vector3dVector(pcd_npy)

o3d.visualization.draw_geometries([pcd])

# pcd = pcd.voxel_down_sample(voxel_size = downsize)


uv,xyz = pointCloudTOuvxyz(pcd,intrinsics,tform)
#bbox = [852,458,1073-852,606-458]
#bbox = [0,0,1920,1080]
#uvxyz = bboxTOpcd(bbox,uvxyz)




u = uv[0,:]
v = uv[1,:]


fig, ax = plt.subplots(subplot_kw={"projection": "3d"})


# Make data.
X = xyz[0,:]
Y = xyz[1,:]

Z = xyz[2,:]
# Plot the surface.
surf = ax.scatter(X,Y,Z,cmap=cm.hsv,c=Y)
col = surf.get_facecolors()[:,:3].tolist()
col = np.array(col)*255




'''
# Customize the z axis.
ax.set_zlim(-1.01, 1.01)
ax.zaxis.set_major_locator(LinearLocator(10))
# A StrMethodFormatter is used automatically
ax.zaxis.set_major_formatter('{x:.02f}')
# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()
'''







store_point = []
cv2.namedWindow("frame", cv2.WINDOW_AUTOSIZE)
cv2.setMouseCallback('frame', mouse_click)

path = img_file_path+r"231.npy"


image = np.load(path)


n = 1

while(True):
    start_time = time.time()

    image = np.load(path)
    frame = image.copy()

    #dim = (1280, 720)
    # resize image
    #frame = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)



    if len(store_point)>0:
        for i in range(len(store_point)):
            cv2.circle(frame, (store_point[i][0], store_point[i][1]), 1, (0, 255, 0), thickness = 4)


    for i in range(len(u)-1):
        cv2.circle(frame, (u[i],v[i]), radius=1, color=(col[i][0], col[i][1], col[i][2]), thickness=-1)
    

    cv2.imshow('frame', frame)

    k=cv2.waitKey(1)

    if k == 27:
       break
    if k == ord('w'):
        trans = Ty(0.01)
        tform = np.dot(tform.T, trans)
        tform = tform.T
        uv,xyz = pointCloudTOuvxyz(pcd,intrinsics,tform)

        u = uv[0,:]
        v = uv[1,:]

        col = generate_pcd_color_map(xyz)

    if k == ord('s'):
        trans = Ty(-0.01)
        tform = np.dot(tform.T, trans)
        tform = tform.T
        uv,xyz = pointCloudTOuvxyz(pcd,intrinsics,tform)

        u = uv[0,:]
        v = uv[1,:]

        col = generate_pcd_color_map(xyz)
        

    if k == ord('d'):
        trans = Tx(0.01)
        tform = np.dot(tform.T, trans)
        tform = tform.T
        uv,xyz = pointCloudTOuvxyz(pcd,intrinsics,tform)

        u = uv[0,:]
        v = uv[1,:]

        col = generate_pcd_color_map(xyz)
        

    if k == ord('a'):
        trans = Tx(-0.01)
        tform = np.dot(tform.T, trans)
        tform = tform.T
        uv,xyz = pointCloudTOuvxyz(pcd,intrinsics,tform)

        u = uv[0,:]
        v = uv[1,:]

        col = generate_pcd_color_map(xyz)
        

    if k == ord('q'):
        trans = Tz(-0.01)
        tform = np.dot(tform.T, trans)
        tform = tform.T
        uv,xyz = pointCloudTOuvxyz(pcd,intrinsics,tform)

        u = uv[0,:]
        v = uv[1,:]

        col = generate_pcd_color_map(xyz)
        

    if k == ord('e'):
        trans = Tz(0.01)
        tform = np.dot(tform.T, trans)
        tform = tform.T
        uv,xyz = pointCloudTOuvxyz(pcd,intrinsics,tform)

        u = uv[0,:]
        v = uv[1,:]

        col = generate_pcd_color_map(xyz)

    if k == ord('t'):
        trans = Rx(0.1)
        tform = np.dot(tform.T, trans)
        tform = tform.T
        uv,xyz = pointCloudTOuvxyz(pcd,intrinsics,tform)

        u = uv[0,:]
        v = uv[1,:]

        col = generate_pcd_color_map(xyz)

    if k == ord('g'):
        trans = Rx(-0.1)
        tform = np.dot(tform.T, trans)
        tform = tform.T
        uv,xyz = pointCloudTOuvxyz(pcd,intrinsics,tform)

        u = uv[0,:]
        v = uv[1,:]

        col = generate_pcd_color_map(xyz)


    if k == ord('h'):
        trans = Ry(0.1)
        tform = np.dot(tform.T, trans)
        tform = tform.T
        uv,xyz = pointCloudTOuvxyz(pcd,intrinsics,tform)

        u = uv[0,:]
        v = uv[1,:]

        col = generate_pcd_color_map(xyz)


    if k == ord('f'):
        trans = Ry(-0.1)
        tform = np.dot(tform.T, trans)
        tform = tform.T
        uv,xyz = pointCloudTOuvxyz(pcd,intrinsics,tform)

        u = uv[0,:]
        v = uv[1,:]

        col = generate_pcd_color_map(xyz)


    if k == ord('y'):
        trans = Rz(0.1)
        tform = np.dot(tform.T, trans)
        tform = tform.T
        uv,xyz = pointCloudTOuvxyz(pcd,intrinsics,tform)

        u = uv[0,:]
        v = uv[1,:]

        col = generate_pcd_color_map(xyz)


    if k == ord('r'):
        trans = Rz(-0.1)
        tform = np.dot(tform.T, trans)
        tform = tform.T
        uv,xyz = pointCloudTOuvxyz(pcd,intrinsics,tform)

        u = uv[0,:]
        v = uv[1,:]

        col = generate_pcd_color_map(xyz)


    if k == ord('n'):
        n = n + 1
        
        if n>1000:
            n = 1

        pcd_npy = np.load(pcd_file_path+"{}.npy".format(str(n)))

        pcd.points = o3d.utility.Vector3dVector(pcd_npy)

        pcd = pcd.voxel_down_sample(voxel_size = downsize)

        uv,xyz = pointCloudTOuvxyz(pcd,intrinsics,tform)

        u = uv[0,:]
        v = uv[1,:]

        col = generate_pcd_color_map(xyz)

        path = img_file_path+"{}.npy".format(str(n))

    if k == ord('b'):
        n = n - 1
        if n<0:
            n = 14
        
        pcd_npy = np.load(pcd_file_path+"{}.npy".format(str(n)))

        pcd.points = o3d.utility.Vector3dVector(pcd_npy)

        pcd = pcd.voxel_down_sample(voxel_size = downsize)


        uv,xyz = pointCloudTOuvxyz(pcd,intrinsics,tform)

        u = uv[0,:]
        v = uv[1,:]

        col = generate_pcd_color_map(xyz)

        path = img_file_path+"{}.npy".format(str(n))

    if k == ord('m'):


        #zeros = np.zeros((3,1))
        #intrinsics = np.hstack((intrinsics.T,zeros))
        #M = np.dot(intrinsics,tform.T)
        #np.save(r"intrin_extrin_Matrix.npy",M)
        #print("save intrin_extrin_Matrix.npy")

        tform_save = tform
        np.save(r"C:\workspace\lidar_camera\tform_fine_tune.npy", tform_save)
        print("save tform_fine_tune")

    if k == ord('l'):
        first = time.perf_counter()


        #convert_xyzTObirdseyeview(store_point)

        for i in range(len(store_point)):
            uvxyzTOxyz(store_point[i],uv,xyz)

        print(time.perf_counter() - first)

        
    #print("--- %s seconds ---" % (time.time() - start_time))





#cap.release()
cv2.destroyAllWindows()
