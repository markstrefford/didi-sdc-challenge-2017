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
import numpy as np
import pandas as pd

import cv2
import argparse

from pointcloud_utils.lidar_top import *
from pointcloud_utils.lidar_surround import *
from tracklets.parse_tracklet import Tracklet, parse_xml
from timestamp_utils import get_camera_timestamp_and_index

# TODO - Move these to a utils function later?
def makedirs(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

# Get camera timestamp and index closest to the pointcloud timestamp
#TODO Create a utility function
#TODO - Make this into a better coded function??? Perhaps a class?  timestamp.get_nearest() perhaps??
# def get_camera_timestamp_and_index(camera_data, pointcloud_timestamp, timestamp_offset):
#     camera_index = camera_data.ix[(camera_data.timestamp - pointcloud_timestamp).abs().argsort()[:1]].index[0]
#
#     proposed_index = camera_index + timestamp_offset
#     #print ('Proposed index: {} = camera_index: {} + timestamp_offset: {}'.format(proposed_index, camera_index, timestamp_offset))
#     if timestamp_offset > 0:
#         actual_index = min(proposed_index,len(camera_data))
#     elif timestamp_offset < 0:
#         actual_index = max(proposed_index, 0)
#     else:
#         actual_index = proposed_index
#
#     #print ('Actual index: {}'.format(actual_index))
#     #camera_timestamp = camera_data.ix[camera_index].timestamp
#     camera_timestamp = camera_data.ix[actual_index].timestamp
#     return camera_timestamp, actual_index

# Get info from tracklets
def get_obstacle_from_tracklet(tracklet_file):
    tracks = []
    tracklets = parse_xml(tracklet_file)
    for track in tracklets:
        obj_size = track.size    #FIXME: Single obstacle per tracklet file
        for translation, rotation, state, occlusion, truncation, amtOcclusion, amtBorders, absoluteFrameNumber in track:
            tracks.append({
                'frame': absoluteFrameNumber,
                'translation': translation,
                'rotation': rotation
            })          # TODO: Add in other tracklets info as required
    #print('get_obstacle_from_tracklet(): Tracks={}'.format(tracks))
    return obj_size, tracks

# main #################################################################
# for demo data:  /root/share/project/didi/data/didi/didi-2/Out/1/15

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Convert point clouds to top and surround views for training.')
    parser.add_argument('-o', '--outdir', type=str, nargs='?', default='/output',
        help='Output folder')
    parser.add_argument('-i', '--indir', type=str, nargs='?', default='/data',
        help='Input folder where pointclouds are located')
    parser.add_argument('-t', '--time_offset', type=int, default=0,
        help='Timestamp offset')
    parser.add_argument('-p', '--pc-only', dest='pc_only', action='store_true', help='Pointclouds only, no images displayed')
    parser.set_defaults(msg_only=False)
    parser.set_defaults(debug=False)
    args = parser.parse_args()

    show_images = False if args.pc_only else True

    # TODO - use os.path.join here instead!!
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

    tracklet_file      = base_dir + '/tracklet_labels.xml'
    camera_csv        = base_dir + '/capture_vehicle_camera.csv'

    timestamp_offset = args.time_offset

    # TODO - Radar

    makedirs(mark_dir_top) #, exist_ok=True)
    makedirs(mark_dir_surround) #, exist_ok=True)
    makedirs(lidar_top_dir) #, exist_ok=True)
    makedirs(lidar_top_img_dir) #, exist_ok=True)
    makedirs(lidar_surround_dir) #, exist_ok=True)
    makedirs(lidar_surround_img_dir) #, exist_ok=True)

    camera_data = pd.read_csv(camera_csv)   #['timestamp']
    obj_size, tracks = get_obstacle_from_tracklet(tracklet_file)    # FIXME Single obstacle per tracklet file

    for file in sorted(glob.glob(lidar_dir + '/*.npy')):
        name = os.path.basename(file).replace('.npy','')

        lidar_file    = lidar_dir +'/'+name+'.npy'
        top_file      = lidar_top_dir +'/'+name+'.npy'
        top_img_file  = lidar_top_img_dir +'/'+name+'.png'
        mark_file_top = mark_dir_top +'/'+name+'.png'
        surround_file = lidar_surround_dir + '/' + name + '.npy'
        surround_img_file  = lidar_surround_img_dir + '/' + name + '.png'
        mark_file_surround = mark_dir_surround +'/'+name+'.png'
        # boxes3d_file  = gt_boxes3d_dir+'/'+name+'.npy'

        # TODO - radar

        lidar = np.load(lidar_file)
        top, top_img = lidar_to_top(lidar)
        surround, surround_img = lidar_to_surround(lidar)

        # Draw box from tracklet file on the images
        pointcloud_timestamp = int(name)         # Assuming that the name is the timestamp!!
        camera_timestamp, index = get_camera_timestamp_and_index(camera_data, pointcloud_timestamp, timestamp_offset)
        #print ('Timestamps: pointcloud={}, camera={}, diff={}'.format(pointcloud_timestamp, camera_timestamp, abs(int(pointcloud_timestamp)-int(camera_timestamp))))
        #boxes3d = np.load(boxes3d_file)

        #save pointcloud as image
        cv2.imwrite(top_img_file,top_img)
        np.save(top_file,top)
        cv2.imwrite(surround_img_file, surround_img)
        np.save(surround_file,surround)

        # now add in object bounding box for display purposes
        top_img = draw_track_on_top(top_img, obj_size, tracks[index], color=(0,0, 255))
        surround_img = draw_box3d_on_surround(surround_img, obj_size, tracks[index], color=(255,255,255))

        if show_images:
            imshow('top_img',top_img,3)
            imshow('surround_img',surround_img,3)
            cv2.waitKey(1)

        #save
        cv2.imwrite(mark_file_top,top_img)
        cv2.imwrite(mark_file_surround,surround_img)

    dir_to_avi(avi_file_top, mark_dir_top, show_images)
    dir_to_avi(avi_file_surround, mark_dir_surround, show_images)




