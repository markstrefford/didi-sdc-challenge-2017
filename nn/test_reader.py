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
import cv2
sys.path.append(os.path.abspath(os.path.dirname(__file__) + '/' + '../'))
from tracklets.parse_tracklet import Tracklet, parse_xml
from pointcloud_utils.lidar_top import draw_track_on_top
from pointcloud_utils.timestamp_utils import get_camera_timestamp_and_index

import numpy as np
import glob

top_x, top_y, top_z = 400, 400, 8
DATA_DIR = '/vol/dataset2/Didi-Release-2/Tracklets'    #GCP


# Use file-based training for now, single bag file
# TODO: Make this work with multiple bags
# TODO: Make this work with multiple trackable objects
class TestReader(object):
    def __init__(self, test_dir):
        self.test_dir = test_dir
        self.timestamps = []
        self.load()

    def load(self):
        xs=[]
        timestamps=[]

        self.test_batch_pointer = 0
        print ('Processing test data: {}'.format(self.test_dir))
        lidar_top_dir = os.path.join(self.test_dir, 'processed/lidar_top')
        lidar_top_image_dir = os.path.join(self.test_dir, 'processed/lidar_top_img')
        lidar_files = sorted(glob.glob(os.path.join(lidar_top_dir, '*.npy')))

        frame = 0
        for file in lidar_files:

            lidar_file = os.path.join(lidar_top_dir, file)
            xs.append(lidar_file)

            timestamps.append(int(os.path.basename(file).replace('.npy', '')))
            frame +=1

        self.num_test_samples = len(xs)
        self.test_xs = xs
        self.test_timestamps = timestamps
        self.lidar_top_image_dir = lidar_top_image_dir

        print ('TestReader.load(): Found {} testing samples'.format(self.num_test_samples))


    # TODO - These 2 functions are not DRY!!!

    def load_test_batch(self, batch_size=1):
        x_out = []
        for i in range(0, batch_size):
            index = (self.test_batch_pointer + i) % self.num_test_samples
            file = self.test_xs[index]
            pointcloud = np.load(file)
            x_out.append(pointcloud)
            # FIXME: Prediction code is single object only at the moment
        self.test_batch_pointer += batch_size
        x_out = np.array(x_out, dtype=np.uint8)
        return x_out

    def get_timestamps(self):
        return self.test_timestamps

    def get_lidar_top_image(self, timestamp):
        filename = os.path.join(self.lidar_top_image_dir, str(timestamp) + '.png')
        image = cv2.imread(filename)
        return image
