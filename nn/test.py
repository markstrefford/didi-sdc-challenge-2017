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
from tracklets.generate_tracklet import *

BATCH_SIZE = 1
DATA_DIR = '/vol/dataset2/Didi-Release-2/Round_1_Test_Tracklets/19_fpc2/'
WEIGHTS_PATH='/vol/training/logs/'
PREDICT_OUTPUT=os.path.join(DATA_DIR, 'images')

def get_arguments():
    parser = argparse.ArgumentParser(description='Udacity Challenge Testing Script')
    parser.add_argument('--weights', type=str,
                        action='store', dest='weights_path', help='Path to a trained model')
    parser.add_argument('--data_dir', '--data', type=str, default=DATA_DIR,
                        action='store', dest='data_dir', help='The directory containing the testing data.')
    parser.add_argument('--predict_dir', '--predict', type=str, default=PREDICT_OUTPUT,
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

    LossHistory, model = nn.top_nn(weights_path=args.weights_path)
    summary = model.summary()
    print (summary)     # TODO: Write to disk together with diagram (see keras.model_to_dot)

    print('test.py: args.data_dir={}'.format(args.data_dir))
    data_reader = TestReader(args.data_dir)

    for batch in range(data_reader.num_test_samples / args.batch_size):
        xs = data_reader.load_test_batch(batch_size=args.batch_size)   # Get all samples

        for i in range(args.batch_size):

            predictions = model.predict(xs, batch_size=args.batch_size)  # TODO - Move into the loop like training code
            predict_output_file = os.path.join(PREDICT_OUTPUT, str(data_reader.test_batch_pointer) + '_predictions.npy')
            np.save(predict_output_file, predictions)

if __name__ == '__main__':
    main()





