import os
os.environ['HOME'] = '/root'

SEED = 202


# std libs
import glob


# num libs
import math
import random
import numpy as np
random.seed(SEED)
np.random.seed(SEED)

import cv2
#import mayavi.mlab as mlab

#from pointcloud_utils.lidar import *
from lidar import *


## 360 side view from
## http://ronny.rest/blog/post_2017_04_03_point_cloud_panorama/
## See Bo li's paper:
##    http://prclibo.github.io/
##    [1] "Multi-View 3D Object Detection Network for Autonomous Driving" - Xiaozhi Chen, Huimin Ma, Ji Wan, Bo Li and Tian Xia , arXiv 2016
##    [2] "3D Fully Convolutional Network for Vehicle Detection in Point Cloud" - Bo Li, arXiv 2016
##    [3] "Vehicle Detection from 3D Lidar Using Fully Convolutional Network" - Bo Li and Tianlei Zhang and Tian Xia , arXiv 2016
##


##   cylindrial projection
SURROUND_U_STEP = 1.    #resolution
SURROUND_V_STEP = 1.33
SURROUND_U_MIN, SURROUND_U_MAX = np.array([0,    360])/SURROUND_U_STEP  # horizontal of cylindrial projection
SURROUND_V_MIN, SURROUND_V_MAX = np.array([-90,   90])/SURROUND_V_STEP  # vertical   of cylindrial projection


def lidar_to_surround_coords(x, y, z, d ):
    u =   np.arctan2(x, y)/np.pi*180 /SURROUND_U_STEP
    v = - np.arctan2(z, d)/np.pi*180 /SURROUND_V_STEP
    u = (u +90)%360  ##<todo> car will be spit into 2 at boundary  ...

    u = np.rint(u)
    v = np.rint(v)
    u = (u - SURROUND_U_MIN).astype(np.uint8)
    v = (v - SURROUND_V_MIN).astype(np.uint8)

    return u,v


def lidar_to_surround(lidar):
    def normalise_to_255(a):
        return (((a - min(a)) / float(max(a) - min(a))) * 255).astype(np.uint8)

    x = lidar['x']
    y = lidar['y']
    z = lidar['z']
    r = lidar['intensity']
    d = np.sqrt(x ** 2 + y ** 2)  # map distance relative to origin
    u,v = lidar_to_surround_coords(x,y,z,d)

    width  = int(SURROUND_U_MAX - SURROUND_U_MIN + 1)
    height = int(SURROUND_V_MAX - SURROUND_V_MIN + 1)
    surround     = np.zeros((height, width, 3), dtype=np.float32)
    surround_img = np.zeros((height, width, 3), dtype=np.uint8)

    surround[v, u, 0] = d
    surround[v, u, 1] = z
    surround[v, u, 2] = r
    surround_img[v, u, 0] = normalise_to_255(np.clip(d,     0, 30))
    surround_img[v, u, 1] = normalise_to_255(np.clip(z+1.8, 0, 100))
    surround_img[v, u, 2] = normalise_to_255(np.clip(r,     0, 30))

    return surround, surround_img


## drawing ####
def create_box3d_from_tracklet(obj_size, tracklet):  # TODO - Move to a utility function
    obj_center = tracklet['translation']
    obj_center_x = obj_center[0]
    obj_center_y = obj_center[1]
    obj_center_z = obj_center[2]

    x0, y0, z0 = obj_center_x + (obj_size[2] / 2.), obj_center_y + (obj_size[1] / 2.),  obj_center_z + (obj_size[0] / 2.)       # Top left front
    x1, y1, z1 = obj_center_x - (obj_size[2] / 2.), obj_center_y - (obj_size[1]),       obj_center_z + (obj_size[0] / 2.)       # Top right back
    x2, y2, z2 = obj_center_x + (obj_size[2] / 2.), obj_center_y + (obj_size[1] / 2.),  obj_center_z - (obj_size[0] / 2.)       # Bottom left front
    x3, y3, z3 = obj_center_x - (obj_size[2] / 2.), obj_center_y - (obj_size[1]),       obj_center_z - (obj_size[0] / 2.)       # Bottom right back
    # x0, y0, z0 = obj_center_x + (obj_size[2] / 2.), obj_center_y + (obj_size[1] / 2.), obj_center_z + (obj_size[0] / 2.)       # Top left front
    # x1, y1, z1 = obj_center_x - (obj_size[2] / 2.), obj_center_y - (obj_size[1] / 2.), obj_center_z + (obj_size[0] / 2.)       # Top right back
    # x2, y2, z2 = obj_center_x + (obj_size[2] / 2.), obj_center_y + (obj_size[1] / 2.), obj_center_z - (obj_size[0] / 2.)       # Bottom left front
    # x3, y3, z3 = obj_center_x - (obj_size[2] / 2.), obj_center_y - (obj_size[1] / 2.), obj_center_z - (obj_size[0] / 2.)       # Bottom right back

    # Other corners or completeness - needed for the box3d code later
    x4, y4, z4 = obj_center_x - (obj_size[2] / 2.), obj_center_y + (obj_size[1] / 2.),  obj_center_z + (obj_size[0] / 2.)
    x5, y5, z5 = obj_center_x + (obj_size[2] / 2.), obj_center_y - (obj_size[1]),       obj_center_z + (obj_size[0] / 2.)
    x6, y6, z6 = obj_center_x - (obj_size[2] / 2.), obj_center_y + (obj_size[1] / 2.),  obj_center_z - (obj_size[0] / 2.)
    x7, y7, z7 = obj_center_x + (obj_size[2] / 2.), obj_center_y - (obj_size[1]),       obj_center_z - (obj_size[0] / 2.)

    box3d = np.array([[x0, y0, z0], [x1, y1, z1], [x2, y2, z2], [x3, y3, z3],
                      [x4, y4, z4], [x5, y5, z5], [x6, y6, z6], [x7, y7, z7]])
    return box3d

## drawing ####
def track_to_surround_box(obj_size, track):
    boxes3d = create_box3d_from_tracklet(obj_size, track)
    is_reshape = boxes3d.shape==(8,3) #support for single box3d

    if is_reshape:
        boxes3d = boxes3d.reshape(1,8,3)

    num = len(boxes3d)
    surround_boxes = np.zeros((num,4),  dtype=np.float32)
    for n in range(num):
        b = boxes3d[n]

        x = b[:,0]
        y = b[:,1]
        z = b[:,2]
        d = np.sqrt(x ** 2 + y ** 2)
        u,v = lidar_to_surround_coords(x,y,z,d)
        umin,umax = np.min(u),np.max(u)
        vmin,vmax = np.min(v),np.max(v)
        surround_boxes[n] = np.array([umin,vmin,umax,vmax])

    if is_reshape:
        surround_boxes = surround_boxes.squeeze()

    return surround_boxes

def draw_box3d_on_surround(image, obj_size, track, color=(255,255,255), fill = 1):
    surround_boxes = track_to_surround_box(obj_size, track)
    is_reshape = surround_boxes.shape==(4)
    if is_reshape:
        surround_boxes = surround_boxes.reshape(1,4)

    # TODO: Handle when box splits across left and right of surround view
    # num = len(surround_boxes)
    # for n in range(num):
    b = surround_boxes   #[n]
    x1 = b[0]
    y1 = b[1]
    x2 = b[2]
    y2 = b[3]
    cv2.rectangle(image,(x1,y1),(x2,y2),color,fill,cv2.LINE_AA)

    print ('draw_track_on_top(): Track={}, Box=[{},{}]/[{},{}]'.format(track, x1, y1, x2, y2))
    return image


