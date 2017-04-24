"""Create top view from a pointcloud

Based on code from http://ronny.rest/blog/post_2017_03_26_lidar_birds_eye/
and https://github.com/hengck23/didi-udacity-2017/tree/master/baseline-04

See Bo li's paper:
   http://prclibo.github.io/
   [1] "Multi-View 3D Object Detection Network for Autonomous Driving" - Xiaozhi Chen, Huimin Ma, Ji Wan, Bo Li and Tian Xia , arXiv 2016
   [2] "3D Fully Convolutional Network for Vehicle Detection in Point Cloud" - Bo Li, arXiv 2016
   [3] "Vehicle Detection from 3D Lidar Using Fully Convolutional Network" - Bo Li and Tianlei Zhang and Tian Xia , arXiv 2016

   This script will create surround and top view iamges for training
"""

import os

import sys; sys.path = [''] + sys.path  # Consistency for imports between modules and running as a script

# num libs
import math
import random
import numpy as np

SEED = 202

random.seed(SEED)
np.random.seed(SEED)

import cv2
import mayavi.mlab as mlab
import argparse

#from pointcloud_utils.lidar import *
#from pointcloud_utils.lidar_top import *
#from pointcloud_utils.lidar_surround import *
from pointcloud_utils.lidar_top import *
from pointcloud_utils.lidar_surround import *

# main #################################################################
# for demo data:  /root/share/project/didi/data/didi/didi-2/Out/1/15

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Convert point clouds to top and surround views for training.')
    parser.add_argument('-o', '--outdir', type=str, nargs='?', default='/output',
        help='Output folder')
    parser.add_argument('-i', '--indir', type=str, nargs='?', default='/data',
        help='Input folder where pointclouds are located')
    parser.add_argument('-p', '--pc-only', dest='pc_only', action='store_true', help='Pointclouds only, no images displayed')
    parser.set_defaults(msg_only=False)
    parser.set_defaults(debug=False)
    args = parser.parse_args()

    show_images = False if args.pc_only else True

    base_dir          = args.indir
    lidar_dir         = base_dir + '/pointcloud'
    radar_dir         = base_dir + '/radar_pointcloud'
    gt_boxes3d_dir    = base_dir + '/processed/gt_boxes3d'
    lidar_top_dir     = base_dir + '/processed/lidar_top'
    lidar_top_img_dir = base_dir + '/processed/lidar_top_img'
    mark_dir_top      = base_dir + '/processed/mark-top-box'
    avi_file_top      = base_dir + '/processed/mark-top-box.avi'
    lidar_surround_dir     = base_dir + '/processed/lidar_surround'
    lidar_surround_img_dir = base_dir + '/processed/lidar_surround_img'
    mark_dir_surround = base_dir + '/processed/mark-surround-box'
    avi_file_surround = base_dir + '/processed/mark-surround-box.avi'

    # TODO - Sort out error if directory exists!!
    os.makedirs(mark_dir) #, exist_ok=True)
    os.makedirs(lidar_top_dir) #, exist_ok=True)
    os.makedirs(lidar_top_img_dir) #, exist_ok=True)
    os.makedirs(lidar_surround_dir) #, exist_ok=True)
    os.makedirs(lidar_surround_img_dir) #, exist_ok=True)

    #fig   = mlab.figure(figure=None, bgcolor=(0,0,0), fgcolor=None, engine=None, size=(500, 500))
    for file in sorted(glob.glob(lidar_dir + '/*.npy')):
        name = os.path.basename(file).replace('.npy','')

        lidar_file    = lidar_dir +'/'+name+'.npy'
        top_file      = lidar_top_dir +'/'+name+'.npy'
        top_img_file  = lidar_top_img_dir +'/'+name+'.png'
        mark_file_top = mark_dir_top +'/'+name+'.png'
        surround_file = lidar_surround_dir + '/' + name + '.npy'
        surround_img_file  = lidar_surround_img_dir + '/' + name + '.png'
        mark_file_surround = mark_dir_top +'/'+name+'.png'
        boxes3d_file  = gt_boxes3d_dir+'/'+name+'.npy'

        # TODO - lidar_to_surround()

        lidar = np.load(lidar_file)
        top, top_img = lidar_to_top(lidar)
        surround, surround_img = lidar_to_surround(lidar)

        #boxes3d = np.load(boxes3d_file)

        #save
        cv2.imwrite(top_img_file,top_img)
        np.save(top_file,top)
        cv2.imwrite(surround_img_file, surround_img)
        np.save(surround_file,surround)

        #show
        #print ('main(): show mlab images')
        #mlab.clf(fig)
        #draw_didi_lidar(fig, lidar, is_grid=1, is_axis=1)

        #if len(boxes3d)!=0:
        #    draw_didi_boxes3d(fig, boxes3d)
        #    draw_box3d_on_top(top_img, boxes3d, color=(255,255,255))

        #azimuth,elevation,distance,focalpoint = MM_PER_VIEW1
        #mlab.view(azimuth,elevation,distance,focalpoint)
        #
        #mlab.show(1)
        if show_images:
            imshow('top_img',top_img,1)
            imshow('surrond_img',surround_img,3)
            cv2.waitKey(1)

        #save
        cv2.imwrite(mark_file_top,top_img)
        cv2.imwrite(mark_file_surround,surround_img)

    dir_to_avi(avi_file_top, mark_dir_top, show_images)
    dir_to_avi(avi_file_surround, mark_dir_surround, show_images)




