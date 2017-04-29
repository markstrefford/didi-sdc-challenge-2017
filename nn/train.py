"""
train.py

Train the neural network to predict location and orientation of obstacles

TODO: This will evolve over time to handle more than 1 bag, to become a ROS subscriber, etc.

"""


import nn
import os
import tensorflow as tf
import numpy as np
import pandas as pd
from keras.models import Model, Sequential
import glob

# Get camera timestamp and index closest to the pointcloud timestamp
#TODO Create a utility function
def get_nearest_timestamp_and_index(data, timestamp):
    index = data.ix[(data.timestamp - timestamp).abs().argsort()[:1]].index[0]
    data_timestamp = data.ix[index].timestamp
    return data_timestamp, index


# Get info from tracklets
#TODO Create a utility function
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
    return obj_size, tracks


# Use file-based training for now, single bag file
class data_reader(object):
    def __init__(self, batch_size, start, base_dir):
        self.batch_size = batch_size
        self.start = start
        self.training_dir = base_dir
        self.lidar_top_dir = os.path.join(self.training_dir, '/processed/lidar_top')
        self.load()

    def load(self):
        xs=[]
        ys=[]

        self.train_batch_pointer = 0
        self.val_batch_pointer = 0
        total = 0

        #image_dir = os.path.join(self.training_dir, 'camera')
        camera_csv = os.path.join(self.training_dir, 'capture_vehicle_camera.csv')   # We're driven by camera messages
        pointcloud_csv = os.path.join(self.training_dir, 'capture_vehicle_pointcloud.csv')
        object_csv = os.path.join(self.training_dir, 'objects_obs1_rear_rtk_interpolated.csv')
        tracklet_file = os.path.join(self.training_dir, 'tracklet_labels.xml')

        camera_data = pd.read_csv(camera_csv)  # ['timestamp']
        obj_size, tracks = get_obstacle_from_tracklet(tracklet_file)

        for file in sorted(glob.glob(self.lidar_top_dir + '/*.npy')):
            lidar_file = os.path.join(self.lidar_top_dir, file)

            #lidar = np.load(lidar_file)
            xs.append(lidar_file)
            timestamp = int(os.path.basename(file).replace('.npy', ''))
            camera_timestamp, index = get_nearest_timestamp_and_index(camera_data, timestamp)
            t = tracks[index].translation
            r = tracks[index].rotation
            y = np.array([t[0], t[1], t[2], r[0], r[1], r[2]])
            ys.append(y)

            total +=1

        self.num_samples = len(xs)

        self.train_xs = xs[:int(len(xs) * 0.8)]
        self.train_ys = ys[:int(len(xs) * 0.8)]

        self.val_xs = xs[-int(len(xs) * 0.2)]
        self.val_ys = ys[-int(len(xs) * 0.2)]

        self.num_train_samples = len(self.train_xs)
        self.num_val_samples = len(self.val_xs)


    def load_train_batch(self, batch_size):
        x_out = []
        y_out = []
        for i in range(0, batch_size):
            index = (self.train_batch_pointer + i) % self.num_train_samples
            file = self.train_xs[index]
            pointcloud = np.load(file)
            x_out.append(pointcloud)
            y_out.append(self.train_ys[index])
        self.train_batch_pointer += batch_size
        return x_out, y_out


    def load_val_batch(self, batch_size):
        x_out = []
        y_out = []
        for i in range(0, batch_size):
            index = (self.val_batch_pointer + i) % self.num_val_samples
            file = self.val_xs[index]
            pointcloud = np.load(file)
            x_out.append(pointcloud)
            y_out.append(self.val_ys[index])
        self.val_batch_pointer += batch_size
        return x_out, y_out











