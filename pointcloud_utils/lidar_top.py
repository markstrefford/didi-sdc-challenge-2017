#
"""Create top view from a pointcloud

Based on code from http://ronny.rest/blog/post_2017_03_26_lidar_birds_eye/
and https://github.com/hengck23/didi-udacity-2017/tree/master/baseline-04

See Bo li's paper:
   http://prclibo.github.io/
   [1] "Multi-View 3D Object Detection Network for Autonomous Driving" - Xiaozhi Chen, Huimin Ma, Ji Wan, Bo Li and Tian Xia , arXiv 2016
   [2] "3D Fully Convolutional Network for Vehicle Detection in Point Cloud" - Bo Li, arXiv 2016
   [3] "Vehicle Detection from 3D Lidar Using Fully Convolutional Network" - Bo Li and Tianlei Zhang and Tian Xia , arXiv 2016
"""


import os
os.environ['HOME'] = '/root'

import numpy as np
import cv2

from lidar import *

##

TOP_Y_MIN=-20     #40
TOP_Y_MAX=+20
TOP_X_MIN=-20
TOP_X_MAX=+20     #70.4
TOP_Z_MIN=-2.0    ###<todo> determine the correct values!
TOP_Z_MAX= 0.4

TOP_X_STEP=0.1  #0.1
TOP_Y_STEP=0.1
TOP_Z_STEP=0.4


def lidar_to_top_coords(x,y,z=None):
    X0, Xn = 0, int((TOP_X_MAX-TOP_X_MIN)//TOP_X_STEP)+1
    Y0, Yn = 0, int((TOP_Y_MAX-TOP_Y_MIN)//TOP_Y_STEP)+1
    xx = Yn-int((y-TOP_Y_MIN)//TOP_Y_STEP)
    yy = Xn-int((x-TOP_X_MIN)//TOP_X_STEP)

    return xx,yy


def top_to_lidar_coords(xx,yy):
    X0, Xn = 0, int((TOP_X_MAX-TOP_X_MIN)//TOP_X_STEP)+1
    Y0, Yn = 0, int((TOP_Y_MAX-TOP_Y_MIN)//TOP_Y_STEP)+1
    y = Xn*TOP_Y_STEP-(xx+0.5)*TOP_Y_STEP + TOP_Y_MIN
    x = Yn*TOP_X_STEP-(yy+0.5)*TOP_X_STEP + TOP_X_MIN

    return x,y



## lidar to top ##
def lidar_to_top(lidar):

    idx = np.where (lidar['x']>TOP_X_MIN)
    lidar = lidar[idx]
    idx = np.where (lidar['x']<TOP_X_MAX)
    lidar = lidar[idx]

    idx = np.where (lidar['y']>TOP_Y_MIN)
    lidar = lidar[idx]
    idx = np.where (lidar['y']<TOP_Y_MAX)
    lidar = lidar[idx]

    idx = np.where (lidar['z']>TOP_Z_MIN)
    lidar = lidar[idx]
    idx = np.where (lidar['z']<TOP_Z_MAX)
    lidar = lidar[idx]

    x = lidar['x']
    y = lidar['y']
    z = lidar['z']
    r = lidar['intensity']
    qxs=((x-TOP_X_MIN)//TOP_X_STEP).astype(np.int32)
    qys=((y-TOP_Y_MIN)//TOP_Y_STEP).astype(np.int32)
    qzs=((z-TOP_Z_MIN)//TOP_Z_STEP).astype(np.int32)
    quantized = np.dstack((qxs,qys,qzs,r)).squeeze()

    X0, Xn = 0, int((TOP_X_MAX-TOP_X_MIN)//TOP_X_STEP)+1
    Y0, Yn = 0, int((TOP_Y_MAX-TOP_Y_MIN)//TOP_Y_STEP)+1
    Z0, Zn = 0, int((TOP_Z_MAX-TOP_Z_MIN)//TOP_Z_STEP)+1
    height  = Yn - Y0
    width   = Xn - X0
    channel = Zn - Z0  + 2
    print('height,width,channel=%d,%d,%d'%(height,width,channel))
    top = np.zeros(shape=(width,height,channel), dtype=np.float32)


    # histogram = Bin(channel, 0, Zn, "z", Bin(height, 0, Yn, "y", Bin(width, 0, Xn, "x", Maximize("intensity"))))
    # histogram.fill.numpy({"x": qxs, "y": qys, "z": qzs, "intensity": prs})

    if 1:  #new method
        for z in range(Zn):
            iz = np.where (quantized[:,2]==z)
            quantized_z = quantized[iz]

            for y in range(Yn):
                iy  = np.where (quantized_z[:,1]==y)
                quantized_zy = quantized_z[iy]

                for x in range(Xn):
                    ix  = np.where (quantized_zy[:,0]==x)
                    quantized_zyx = quantized_zy[ix]
                    if len(quantized_zyx)>0:
                        yy,xx,zz = -x,-y, z

                        #height per slice
                        max_height = max(0,np.max(quantized_zyx[:,2])-TOP_Z_MIN)
                        top[yy,xx,zz]=max_height

                        #intensity
                        max_intensity = np.max(quantized_zyx[:,3])
                        top[yy,xx,Zn]=max_intensity

                        #density
                        count = len(idx)
                        top[yy,xx,Zn+1]+=count

                    pass
                pass
            pass

    top[:,:,Zn+1] = np.log(top[:,:,Zn+1]+1)/math.log(16)

    if 0:
        top_image = np.sum(top,axis=2)
        top_image = top_image-np.min(top_image)
        top_image = (top_image/np.max(top_image)*255)
        #top_image = np.clip(top_image,0,255)
        top_image = np.dstack((top_image, top_image, top_image)).astype(np.uint8)


    if 1: #unprocess
        top_image = np.zeros((height,width),dtype=np.float32)

        num = len(lidar)
        for n in range(num):
            x,y   = qxs[n],qys[n]
            yy,xx = -x,-y
            top_image[yy,xx] += 1

        max_value = np.max(np.log(top_image+0.001))
        top_image = top_image/max_value *255
        top_image = np.dstack((top_image, top_image, top_image)).astype(np.uint8)

    return top, top_image

## drawing ####
def create_box3d_from_tracklet(obj_size, tracklet):  # TODO - Move to a utility function
    obj_center = tracklet['translation']
    obj_center_x = obj_center[0]
    obj_center_y = obj_center[1]
    obj_center_z = obj_center[2]

    x0, y0, z0 = obj_center_x + (obj_size[2] / 2.), obj_center_y + (obj_size[1]), obj_center_z + (obj_size[0] / 2.)       # Top left front
    x1, y1, z1 = obj_center_x - (obj_size[2] / 2.), obj_center_y                , obj_center_z + (obj_size[0] / 2.)       # Top right back
    x2, y2, z2 = obj_center_x + (obj_size[2] / 2.), obj_center_y + (obj_size[1]), obj_center_z - (obj_size[0] / 2.)       # Bottom left front
    x3, y3, z3 = obj_center_x - (obj_size[2] / 2.), obj_center_y                , obj_center_z - (obj_size[0] / 2.)       # Bottom right back
    # x0, y0, z0 = obj_center_x + (obj_size[2] / 2.), obj_center_y + (obj_size[1] / 2.), obj_center_z + (obj_size[0] / 2.)       # Top left front
    # x1, y1, z1 = obj_center_x - (obj_size[2] / 2.), obj_center_y - (obj_size[1] / 2.), obj_center_z + (obj_size[0] / 2.)       # Top right back
    # x2, y2, z2 = obj_center_x + (obj_size[2] / 2.), obj_center_y + (obj_size[1] / 2.), obj_center_z - (obj_size[0] / 2.)       # Bottom left front
    # x3, y3, z3 = obj_center_x - (obj_size[2] / 2.), obj_center_y - (obj_size[1] / 2.), obj_center_z - (obj_size[0] / 2.)       # Bottom right back

    # Other corners or completeness - needed for the box3d code later
    x4, y4, z4 = obj_center_x - (obj_size[2] / 2.), obj_center_y + (obj_size[1] / 2.), obj_center_z + (obj_size[0] / 2.)
    x5, y5, z5 = obj_center_x + (obj_size[2] / 2.), obj_center_y - (obj_size[1] / 2.), obj_center_z + (obj_size[0] / 2.)
    x6, y6, z6 = obj_center_x - (obj_size[2] / 2.), obj_center_y + (obj_size[1] / 2.), obj_center_z - (obj_size[0] / 2.)
    x7, y7, z7 = obj_center_x + (obj_size[2] / 2.), obj_center_y - (obj_size[1] / 2.), obj_center_z - (obj_size[0] / 2.)

    box3d = np.array([[x0, y0, z0], [x1, y1, z1], [x2, y2, z2], [x3, y3, z3],
                      [x4, y4, z4], [x5, y5, z5], [x6, y6, z6], [x7, y7, z7]])
    return box3d

def track_to_top_box(obj_size, track):

    #FIXME: Add back in support for multiple objects!!
    boxes3d = create_box3d_from_tracklet(obj_size, track)

    is_reshape = boxes3d.shape==(8,3) #support for single box3d
    if is_reshape:
        boxes3d = boxes3d.reshape(1,8,3)

    num  = len(boxes3d)
    top_boxes = np.zeros((num,4),  dtype=np.float32)
    for n in range(num):
        b   = boxes3d[n]

        x0 = b[0,0]
        y0 = b[0,1]
        x1 = b[1,0]
        y1 = b[1,1]
        x2 = b[2,0]
        y2 = b[2,1]
        x3 = b[3,0]
        y3 = b[3,1]
        u0,v0=lidar_to_top_coords(x0,y0)
        u1,v1=lidar_to_top_coords(x1,y1)
        u2,v2=lidar_to_top_coords(x2,y2)
        u3,v3=lidar_to_top_coords(x3,y3)

        umin=min(u0,u1,u2,u3)
        umax=max(u0,u1,u2,u3)
        vmin=min(v0,v1,v2,v3)
        vmax=max(v0,v1,v2,v3)

        top_boxes[n]=np.array([umin,vmin,umax,vmax])

    if is_reshape:
        top_boxes = top_boxes.squeeze()

    return top_boxes


def draw_track_on_top(image, obj_size, track, color=(255,255,255), fill = 2):

    top_boxes = track_to_top_box(obj_size, track)
    is_reshape = top_boxes.shape==(4)
    if is_reshape:
        top_boxes = top_boxes.reshape(1,4)

    #FIXME: Add back in support for multiple objects!!
    #num = len(top_boxes)
    #for n in range(num):
    b = top_boxes  #[n]
    #x1,y1,x2,y2  = b
    x1 = b[0]
    y1 = b[1]
    x2 = b[2]
    y2 = b[3]
    cv2.rectangle(image, (x1,y1), (x2,y2),color, fill, cv2.LINE_AA)
    print ('draw_track_on_top(): Track={}, Box=[{},{}]/[{},{}]'.format(track, x1,y1, x2, y2))
    return image

