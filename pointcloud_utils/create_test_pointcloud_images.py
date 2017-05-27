"""Create top view from a pointcloud

Based on code from http://ronny.rest/blog/post_2017_03_26_lidar_birds_eye/
and https://github.com/hengck23/didi-udacity-2017/tree/master/baseline-04

See Bo li's paper:
   http://prclibo.github.io/
   [1] "Multi-View 3D Object Detection Network for Autonomous Driving" - Xiaozhi Chen, Huimin Ma, Ji Wan, Bo Li and Tian Xia , arXiv 2016
   [2] "3D Fully Convolutional Network for Vehicle Detection in Point Cloud" - Bo Li, arXiv 2016
   [3] "Vehicle Detection from 3D Lidar Using Fully Convolutional Network" - Bo Li and Tianlei Zhang and Tian Xia , arXiv 2016

   This script will create surround and top view iamges for testing
"""

import os

import sys; sys.path = [''] + sys.path  # Consistency for imports between modules and running as a script

# num libs
import numpy as np
import pandas as pd

import cv2
import argparse

from pointcloud_utils.lidar_top import *
from pointcloud_utils.lidar_surround import *

# TODO - Move these to a utils function later?
def makedirs(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


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

    # TODO - use os.path.join here instead!!
    base_dir          = args.indir
    lidar_dir         = base_dir + '/pointcloud'
    radar_dir         = base_dir + '/radar_pointcloud'

    lidar_top_dir     = base_dir + '/processed/lidar_top'
    lidar_top_img_dir = base_dir + '/processed/lidar_top_img'
    lidar_surround_dir     = base_dir + '/processed/lidar_surround'
    lidar_surround_img_dir = base_dir + '/processed/lidar_surround_img'

    # TODO - Radar
    makedirs(lidar_top_dir) #, exist_ok=True)
    makedirs(lidar_top_img_dir) #, exist_ok=True)
    makedirs(lidar_surround_dir) #, exist_ok=True)
    makedirs(lidar_surround_img_dir) #, exist_ok=True)

    for file in sorted(glob.glob(lidar_dir + '/*.npy')):
        name = os.path.basename(file).replace('.npy','')

        lidar_file    = lidar_dir +'/'+name+'.npy'
        top_file      = lidar_top_dir +'/'+name+'.npy'
        top_img_file  = lidar_top_img_dir +'/'+name+'.png'
        surround_file = lidar_surround_dir + '/' + name + '.npy'
        surround_img_file  = lidar_surround_img_dir + '/' + name + '.png'

        # TODO - radar

        lidar = np.load(lidar_file)
        top, top_img = lidar_to_top(lidar)
        surround, surround_img = lidar_to_surround(lidar)

        cv2.imwrite(top_img_file,top_img)
        np.save(top_file,top)
        cv2.imwrite(surround_img_file, surround_img)
        np.save(surround_file,surround)





