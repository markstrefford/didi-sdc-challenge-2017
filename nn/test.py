"""
test.py

Predict obstacles based on data

Example:
    python train.py --data_dir=/vol/dataset2/Didi-Release-2/Tracklets/1pc/15pc/


"""

#TODO: This will evolve over time to handle more than 1 obstacle
#TODO: Subscribe to ROS messages


#TODO: Do this for the stage 1
#
# 1. Set correct default values
# 2. Render object proposals to files in PREDICT_OUTPUT/images as jpeg (code already written to render them in notebook)
# 3. Create a mpeg from it so that we can watch it later!!
# 4.

import os
import nn
import cv2
import numpy as np
import pandas as pd
import argparse
from test_reader import TestReader
#from tracklets.generate_tracklet import *
from pointcloud_utils import lidar_top

BATCH_SIZE = 8
DATA_DIR = '/vol/dataset2/Didi-Release-2/Round_1_Test_Tracklets/19_fpc2/'
WEIGHTS_PATH='/vol/training/logs/'
PREDICT_OUTPUT=os.path.join(DATA_DIR, 'predictions')
PREDICT_IMAGE_OUTPUT=os.path.join(DATA_DIR, 'images')

# Get the centre of a object proposal
# TODO: Really should do this in deep learning!!
# TODO: Need to do this differently for more than one obstacle
def calc_centroid(prediction):
    coords = np.transpose(np.nonzero(prediction > 0.7))
    if len(coords) > 0:
        min_y, max_y = min(coords[:, 0]), max(coords[:, 0])
        min_x, max_x = min(coords[:, 1]), max(coords[:, 1])
        centroid_x = (min_x + max_x) / 2
        centroid_y = (min_y + max_y) / 2
        lcent_x, lcent_y = lidar_top.top_to_lidar_coords(centroid_x, centroid_y)
        #  TODO - Need to determine z properly!
	lcent_z = -1
    else:
        lcent_x, lcent_y, lcent_z = 0, 0, 0
    print ('calc_centroids(): {}, {}, {}'.format(lcent_x, lcent_y, lcent_z))
    return [lcent_x, lcent_y, lcent_z]


def get_arguments():
    parser = argparse.ArgumentParser(description='Udacity Challenge Testing Script')
    parser.add_argument('--weights', type=str,
                        action='store', dest='weights_path', help='Path to a trained model')
    parser.add_argument('--data_dir', '--data', type=str, default=DATA_DIR,
                        action='store', dest='data_dir', help='The directory containing the testing data.')
    parser.add_argument('--predict_dir', '--predict', type=str, default=PREDICT_IMAGE_OUTPUT,
                        action='store', dest='predict_dir', help='The directory to write predicted images to.')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE,
                        action='store', dest='batch_size', help='Number of [pointcloud] samples in batch.')
    # parser.add_argument('--data_csv', '--csv', type=str, default=CSV,
    #                     action='store', dest='csv', help='The csv containing the training data.')
    return parser.parse_args()

def main():
    args=get_arguments()

    # TODO - Predict these in next iteration of code
    length = 4.241800
    width = 1.447800
    height = 1.574800
    tracklets = []

    LossHistory, model = nn.top_nn(weights_path=args.weights_path)
    summary = model.summary()
    print (summary)     # TODO: Write to disk together with diagram (see keras.model_to_dot)

    print('test.py: args.data_dir={}'.format(args.data_dir))
    data_reader = TestReader(args.data_dir)
    timestamps = data_reader.get_timestamps()

    frame = 0
    num_batches = data_reader.num_test_samples / args.batch_size
    for batch in range(num_batches):
        print ('Batch: {}/{}'.format(batch, num_batches))
        xs = data_reader.load_test_batch(batch_size=args.batch_size)   # Get all samples
        predictions = model.predict(xs, batch_size=args.batch_size)  # TODO - Move into the loop like training code

        for i in range(args.batch_size):
            timestamp = timestamps[frame]
            predict_output_file = os.path.join(PREDICT_OUTPUT, str(frame) + '.npy')
            np.save(predict_output_file, predictions[i])
            tracklets.append(calc_centroid(predictions[i]))

            im = np.array(data_reader.get_lidar_top_image(timestamp), dtype=np.uint8)
            im_pred = np.array(255 * predictions[i, :, :, 0], dtype=np.uint8)

            rgb_mask_pred = cv2.cvtColor(im_pred, cv2.COLOR_GRAY2RGB)
            rgb_mask_pred[:, :, 1:3] = 0 * rgb_mask_pred[:, :, 1:2]

            img_pred = cv2.addWeighted(rgb_mask_pred, 0.5, im, 0.5, 0)
            file = os.path.join(PREDICT_IMAGE_OUTPUT, str(frame) + '_' + str(timestamp) + '.png')
            cv2.imwrite(file, img_pred)

            frame += 1
            
    np.savetxt('data/tracklet_raw.txt', tracklets, delimiter=',')

if __name__ == '__main__':
    main()





