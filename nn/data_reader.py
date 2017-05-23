"""
data_reader.py

Read training and validation data from file

TODO: This will evolve over time to:
 1) handle more than 1 output folder
 2) get data as a ROS subscriber,
 3) etc.

"""

# Fix imports so we can load tracklets module! (FIXME: What's a better way?)
import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__) + '/' + '../'))
from tracklets.parse_tracklet import Tracklet, parse_xml
from pointcloud_utils.lidar_top import draw_track_on_top

import numpy as np
import pandas as pd
import glob

top_x, top_y, top_z = 400, 400, 8

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
# TODO: Make this work with multiple bags
# TODO: Make this work with multiple trackable objects
class DataReader(object):
    def __init__(self, base_dir):
        self.training_dir = base_dir
        self.lidar_top_dir = os.path.join(self.training_dir, 'processed/lidar_top')
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
        #print ('Loaded tracklets {}'.format(tracks))

        lidar_files = sorted(glob.glob(os.path.join(self.lidar_top_dir, '*.npy')))
        for file in lidar_files:
            lidar_file = os.path.join(self.lidar_top_dir, file)

            #lidar = np.load(lidar_file)
            xs.append(lidar_file)
            timestamp = int(os.path.basename(file).replace('.npy', ''))
            camera_timestamp, index = get_nearest_timestamp_and_index(camera_data, timestamp)
            # t = tracks[index]['translation']
            # r = tracks[index]['rotation']
            # FIXME: Size is the same for all tracks...?
            y = np.array([obj_size, tracks[index]])
            ys.append(y)

            total +=1

        self.num_samples = len(xs)

        # self.train_xs = xs[:int(len(xs) * 0.8)]
        # self.train_ys = ys[:int(len(xs) * 0.8)]
        #
        # self.val_xs = xs[-int(len(xs) * 0.2):]
        # self.val_ys = ys[-int(len(xs) * 0.2):]

        self.train_xs = xs  #[:int(len(xs) * 0.8)]
        self.train_ys = ys  #[:int(len(xs) * 0.8)]

        self.val_xs = xs    #[-int(len(xs) * 0.2):]
        self.val_ys = ys    #[-int(len(xs) * 0.2):]

        self.num_train_samples = len(self.train_xs)
        self.num_val_samples = len(self.val_xs)

    def _predict_obj_y(self, obj_size, track):
        obj_y = np.zeros((top_x, top_y))
        obj_y = draw_track_on_top(obj_y, obj_size, track, color=(255), fill=-1)
        return obj_y

    # TODO - This needs reworking!! Assume we'll predict bbox by start location in our NN (todo!!)
    # TODO - May be quicker to use numpy instead of cv2 to create a filled box
    def _predict_box_y(self, obj_size, track):
        box_y = np.zeros((top_x, top_y))
        box_y = draw_track_on_top(box_y, obj_size, track, color=(255))
        return box_y

    def convert_image_to_classes(self, image):
        classes=np.zeros((image.shape[0], image.shape[1], 2))
        classes[:,:,0] = image/255      # Assume 255 is the colour we've used to denote an object of class 1?
        classes[:,:,1] = 1-classes[:,:,1] # Swap 0s for 1s for class 0 (should be just the background!)
        return classes

    # TODO - These 2 functions are not DRY!!!
    def load_train_batch(self, batch_size=1):
        x_out = []
        y_out_obj = []
        y_out_box = []
        for i in range(0, batch_size):
            index = (self.train_batch_pointer + i) % self.num_train_samples
            file = self.train_xs[index]
            pointcloud = np.load(file)
            x_out.append(pointcloud)
            # FIXME: Prediction code is single object only at the moment
            obj_size = self.train_ys[index][0]
            obj_track = self.train_ys[index][1]
            y_out_obj.append(self.convert_image_to_classes(self._predict_obj_y(obj_size, obj_track)))    # Object prediction output (sphere??)
            y_out_box.append(self.convert_image_to_classes(self._predict_box_y(obj_size, obj_track)))    # Output of prediction bounding box
        self.train_batch_pointer += batch_size
        # return np.array(x_out), [np.array(y_out_obj), np.array(y_out_box)]
        return np.array(x_out), np.array(y_out_obj)[:,:,:,0].reshape(1,400,400,1)

    # TODO - These 2 functions are not DRY!!!
    def load_val_batch(self, batch_size):
        x_out = []
        y_out_obj = []
        y_out_box = []
        for i in range(0, batch_size):
            index = (self.val_batch_pointer + i) % self.num_val_samples
            file = self.val_xs[index]
            pointcloud = np.load(file)
            x_out.append(pointcloud)
            # FIXME: Prediction code is single object only at the moment
            obj_size = self.val_ys[index][0]
            obj_track = self.val_ys[index][1]
            y_out_obj.append(self.convert_image_to_classes(self._predict_obj_y(obj_size, obj_track)))    # Object prediction output (sphere??)
            y_out_box.append(self.convert_image_to_classes(self._predict_box_y(obj_size, obj_track)))    # Output of prediction bounding box
        self.val_batch_pointer += batch_size
        #return np.array(x_out), [np.array(y_out_obj), np.array(y_out_box)]
        return np.array(x_out), np.array(y_out_obj)[:,:,:,0].reshape(1,400,400,1)

