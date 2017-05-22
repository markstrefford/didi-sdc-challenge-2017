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
import argparse
from data_reader import DataReader
from tracklets.parse_tracklet import Tracklet, parse_xml
from tracklets.generate_tracklet import *

# TODO: Remove what's not needed here
BATCH_SIZE = 32
DATA_DIR = '/vol/didi/dataset2/tracklets/1pc/10pc'
WEIGHTS_PATH='/vol/training/logs/model-final-step-999-val-0.025713.ckpt'
PREDICT_OUTPUT='/vol/dataset2/Didi-Release-2/Predict/'

def get_arguments():
    parser = argparse.ArgumentParser(description='Udacity Challenge Testing Script')
    parser.add_argument('--weights', type=str, default=WEIGHTS_PATH,
                        action='store', dest='weights_path', help='Path to a trained model')
    parser.add_argument('--data_dir', '--data', type=str, default=DATA_DIR,
                        action='store', dest='data_dir', help='The directory containing the testing data.')
    parser.add_argument('--predict_dir', '--predict', type=str, default=PREDICT_OUTPUT,
                        action='store', dest='predict_dir', help='The directory to write predicted images to.')
    # parser.add_argument('--data_csv', '--csv', type=str, default=CSV,
    #                     action='store', dest='csv', help='The csv containing the training data.')
    return parser.parse_args()

def main():
    args=get_arguments()
    tracklet_file = '/vol/dataset2/Didi-Release-2/Tracklets/1/2/predicted_tracklets.xml'

    start_step = 0
    LossHistory, model = nn.top_nn(weights_path=args.restore_from)
    #summary = model.summary()
    #print (summary)     # TODO: Write to disk together with diagram (see keras.model_to_dot)

    LossHistory, model = nn.top_nn(weights_path=WEIGHTS_PATH)

    print('test.py: args.data_dir={}'.format(args.data_dir))
    data_reader = DataReader(args.data_dir)

    xs, ys = data_reader.load_val_batch(batch_size=args.batch_size)
    predictions = model.predict(xs, batch_size=BATCH_SIZE)  # TODO - Move into the loop like training code

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
        img = np.zeros((400, 400, 3))
        img[:,:,0] = xs[t]
        img[:,:,1] = xs[t]
        #img[:,:,2] = xs[t]

        # Merge with the prediction
        img[:,:,2] = predictions[t]

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





