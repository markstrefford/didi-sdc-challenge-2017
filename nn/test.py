"""
test.py

Predict obstacles based on data

Example:
    python train.py --data_dir=/vol/dataset2/Didi-Release-2/Tracklets/1pc/15pc/


"""

#TODO: This will evolve over time to handle more than 1 obstacle
#TODO: Subscribe to ROS messages

import os
import nn
import cv2
import numpy as np
import pandas as pd
import argparse
from data_reader import DataReader
from tracklets.parse_tracklet import Tracklet, parse_xml
from tracklets.generate_tracklet import *

# TODO: Remove what's not needed here
BATCH_SIZE = 32
DATA_DIR = '/vol/dataset2/Didi-Release-2/Tracklets/1/2/'
WEIGHTS_PATH='/vol/training/logs/model-final-step-99-val-0.993674.ckpt'

# TODO: Only added in for initial code testing... remove asap!!
PCL_IMAGE_PATH='/vol/dataset2/Didi-Release-2/Tracklets/1/2/processed/lidar_top_img/'
PCL_CSV='/vol/dataset2/Didi-Release-2/Tracklets/1/2/capture_vehicle_pointcloud.csv'
PREDICT_OUTPUT='/vol/dataset2/Didi-Release-2/Predict/'

def get_arguments():
    parser = argparse.ArgumentParser(description='Udacity Challenge Testing Script')
    parser.add_argument('--weights', type=str, default=WEIGHTS_PATH,
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
    tracklet_file = '/vol/dataset2/Didi-Release-2/Tracklets/1/2/predicted_tracklets.xml'

    start_step = 0
    #LossHistory, model = nn.top_nn(weights_path=args.weights_path)
    #summary = model.summary()
    #print (summary)     # TODO: Write to disk together with diagram (see keras.model_to_dot)


    pcl_data = pd.read_csv(PCL_CSV)   # TODO: Needed?

    LossHistory, model = nn.top_nn(weights_path=WEIGHTS_PATH)
    summary = model.summary()
    print (summary)     # TODO: Write to disk together with diagram (see keras.model_to_dot)

    print('test.py: args.data_dir={}'.format(args.data_dir))
    data_reader = DataReader(args.data_dir)

    xs, ys = data_reader.load_val_batch(batch_size=args.batch_size)
    predictions = model.predict(xs, batch_size=args.batch_size)  # TODO - Move into the loop like training code

    # TODO - Predict these in next iteration of code
    length = 4.241800
    width = 1.447800
    height = 1.574800
    t = 0
    collection = TrackletCollection()
    for p in predictions:
        # print ('Predicted: {}, Actual: {}'.format(predictions[t], ys[t]))
        # tx = p[0]
        # ty = p[1]
        # tz = p[2]
        #obs_tracklet = Tracklet( object_type='Car', l=length, w=width, h=height, first_frame=t )
        # obs_tracklet.poses = [
        #     {'tx':tx,'ty':ty,'tz':tz,'rx':0,'ry':0,'rz':0}
        # ]
        # collection.tracklets.append(obs_tracklet)

        # Load relevant pcl image
        pcl_timestamp = pcl_data.ix[t].timestamp
        pcl_image_file = os.path.join(PCL_IMAGE_PATH,str(pcl_timestamp)+'.png')
        print ('Rendering prediction on image {}'.format(pcl_image_file))
        pcl_image = cv2.imread(pcl_image_file)
        img = np.zeros((400, 400, 3))   # TODO: Is this sufficient for generating the source pointcloud
        img = pcl_image

        # Merge with the prediction
        img[:,:,2] = predictions[t,:,:,0]

        # Render the combined image
        #cv2.imshow('predict[0]', img)
        #cv2.waitKey(1)
        predict_image_file = os.path.join(PREDICT_OUTPUT, str(t) + '.jpg')
        cv2.imwrite(predict_image_file, img)
        t += 1

    ## save
    #collection.write_xml(tracklet_file)

if __name__ == '__main__':
    main()





