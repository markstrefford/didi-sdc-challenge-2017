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
import argparse
from data_reader import DataReader
from tracklets.parse_tracklet import Tracklet, parse_xml
from tracklets.generate_tracklet import *

# TODO: Remove what's not needed here
BATCH_SIZE = 32
DATA_DIR = '/vol/didi/dataset2/tracklets/1pc/10pc'
LOGDIR = '/vol/training/logs'
CSV='data.csv'
CHECKPOINT_EVERY = 100
NUM_STEPS = int(1e3)
CKPT_FILE = 'model.ckpt'
LEARNING_RATE = 1e-3
KEEP_PROB = 0.8
L2_REG = 0
EPSILON = 0.001
MOMENTUM = 0.9


def get_arguments():
    parser = argparse.ArgumentParser(description='Udacity Challenge Training')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE,
                        action='store', dest='batch_size', help='Number of [camera] samples in batch.')
    parser.add_argument('--data_dir', '--data', type=str, default=DATA_DIR,
                        action='store', dest='data_dir', help='The directory containing the training data.')
    # parser.add_argument('--data_csv', '--csv', type=str, default=CSV,
    #                     action='store', dest='csv', help='The csv containing the training data.')
    parser.add_argument('--logdir', type=str, default=LOGDIR,
                        help='Directory for log files.')
    parser.add_argument('--restore_from', type=str, default=None,
                        help='Checkpoint file to restore model weights from.')
    parser.add_argument('--checkpoint_every', type=int, default=CHECKPOINT_EVERY,
                        help='How many steps to save each checkpoint after')
    parser.add_argument('--num_steps', type=int, default=NUM_STEPS,
                        help='Number of training steps.')
    parser.add_argument('--learning_rate', type=float, default=LEARNING_RATE,
                        help='Learning rate for training.')
    parser.add_argument('--keep_prob', type=float, default=KEEP_PROB,
                        help='Dropout keep probability.')
    parser.add_argument('--l2_reg', type=float,
                        default=L2_REG)
    return parser.parse_args()

def main():
    args=get_arguments()
    tracklet_file = '/vol/didi/dataset2/tracklets/1pc/10pc/predicted_tracklets.xml'

    start_step = 0
    LossHistory, model = nn.top_nn(weights_path=args.restore_from)
    #summary = model.summary()
    #print (summary)     # TODO: Write to disk together with diagram (see keras.model_to_dot)

    print('test.py: args.data_dir={}'.format(args.data_dir))
    data_reader = DataReader(args.data_dir)

    xs, ys = data_reader.load_val_batch(batch_size=args.batch_size)
    predictions = model.predict(xs, batch_size=BATCH_SIZE)

    length = 4.241800
    width = 1.447800
    height = 1.574800
    t = 0
    collection = TrackletCollection()
    for p in predictions:
        print ('Predicted: {}, Actual: {}'.format(predictions[t], ys[t]))
        tx = p[0]
        ty = p[1]
        tz = p[2]
        obs_tracklet = Tracklet( object_type='Car', l=length, w=width, h=height, first_frame=t )
        obs_tracklet.poses = [
            {'tx':tx,'ty':ty,'tz':tz,'rx':0,'ry':0,'rz':0}
        ]
        collection.tracklets.append(obs_tracklet)
        t += 1

    ## save
    collection.write_xml(tracklet_file)

if __name__ == '__main__':
    main()





