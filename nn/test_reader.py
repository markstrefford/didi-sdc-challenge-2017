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
from pointcloud_utils.timestamp_utils import get_camera_timestamp_and_index

import numpy as np
import pandas as pd
import glob
from sklearn.model_selection import train_test_split

top_x, top_y, top_z = 400, 400, 8
DATA_DIR = '/vol/dataset2/Didi-Release-2/Tracklets'    #GCP
#DATA_DIR = '/vol/didi/dataset2/Tracklets'               #Mac
RANDOM_STATE = 202
VALID_PERCENT = .25

# Get camera timestamp and index closest to the pointcloud timestamp
#TODO Create a utility function
# def get_nearest_timestamp_and_index(data, timestamp):
#     index = data.ix[(data.timestamp - timestamp).abs().argsort()[:1]].index[0]
#     data_timestamp = data.ix[index].timestamp
#     return data_timestamp, index

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
class TestReader(object):
    def __init__(self, test_dir):
        self.test_dir = test_dir
        self.load()

    def load(self, test_dir):
        xs=[]

        self.test_batch_pointer = 0
        print ('Processing test data: {}'.format(test_dir))
        lidar_top_dir = os.path.join(test_dir, 'processed/lidar_top')
        lidar_files = sorted(glob.glob(os.path.join(lidar_top_dir, '*.npy')))

        frame = 0
        for file in lidar_files:

            lidar_file = os.path.join(lidar_top_dir, file)
            xs.append(lidar_file)
            frame +=1

        self.num_samples = len(xs)
        self.num_test_samples = len(self.train_xs)
        print ('TestReader.load(): Found {} testing samples'.format(self.num_samples))


    def _predict_obj_y(self, obj_size, track):
        #print ('_predict_obj_y(): obj_size={}, track={}'.format(obj_size, track))
        obj_y = np.zeros((top_x, top_y))
        # TODO - May be quicker to use numpy instead of cv2 to create a filled box
        obj_y = draw_track_on_top(obj_y, obj_size, track, color=(1,1,1), fill=-1)
        return obj_y


    # TODO - This needs reworking!! Assume we'll predict bbox by start location in our NN (todo!!)
    def _predict_box_y(self, obj_size, track):
        box_y = np.zeros((top_x, top_y))
        box_y = draw_track_on_top(box_y, obj_size, track, color=(255))
        return box_y


    # TODO - These 2 functions are not DRY!!!

    def load_test_batch(self, batch_size=1):
        print ('load_train_batch(): batch = {}'.format(self.train_batch_pointer))
        x_out = []
        y_out_obj = []
        y_out_box = []
        for i in range(0, batch_size):
	    #print ('load_train_batch(): self.train_batch_pointer={}, i={}, self.num_train_samples={}'.format(self.train_batch_pointer,i,self.num_train_samples))
            index = (self.train_batch_pointer + i) % self.num_train_samples
            file = self.train_xs[index]
            pointcloud = np.load(file)
            x_out.append(pointcloud)
            # FIXME: Prediction code is single object only at the moment
        self.train_batch_pointer += batch_size
        x_out = np.array(x_out, dtype=np.uint8)
        return x_out



