"""
Create the NN for training

Takes input from 3 areas:

1) Camera input
2) Top view of pointcloud (includes LIDAR and RADAR)
3) Surround view of pointcloud (includes LIDAR and RADAR)

Training approach:

1) Train each individual part of the network to see if it gives a reasonable approximation (TODO - Render bounding box on camera and surround images)
2) Merge in turn (note this may mean playing about with the individual parts of the NN to ensure the merge works!!

Note we'll overfit on a single dataset (say 15.bag) to start with to assure there's no bugs in the code!!

"""

import tensorflow as tf
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Flatten, \
                         Activation, Conv2D, MaxPooling2D, UpSampling2D, \
                         Reshape, core, Dropout
from keras.layers.merge import add, concatenate
from keras.optimizers import Adam, SGD
from keras.callbacks import Callback
import keras.backend as K

#TODO - Just a starter!!!
num_filters = 32
filter_length = 2
num_pooling = 2
border_mode = "valid"
activation = "relu"

# Input shapes (note surround and top and the .npy files)
surround_x, surround_y, surround_z = 400, 8, 3 # TODO - Confirm surround_z, if z=1 then remove completely from the code!!
top_x, top_y, top_z = 400, 400, 8
camera_x, camera_y, camera_z = 1400, 800, 3    #TODO - Get correct values here!!


#TODO: Flesh this out as this is the measure from Udacity
def IoU(pred_y, act_y):
    return True


# TODO: Be DRY here... lots of code repetition... can this be a single function with different parameters calling it??
# TODO: Add in regularisers are they aren't used in this code yet!
def top_nn(weights_path=None, b_regularizer = None, w_regularizer=None):
    class LossHistory(Callback):
        def on_train_begin(self, logs={}):
            self.losses = []

        def on_batch_end(self, batch, logs={}):
            self.losses.append(logs.get('loss'))

    inputs = Input(shape=(top_x, top_y, top_z))
    print (inputs)

    conv1 = Conv2D(32, (2,2), activation='relu', padding='same')(inputs)
    conv1 = Dropout(0.2)(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(64, (2,2), activation='relu', padding='same')(pool1)
    conv2 = Dropout(0.2)(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(128, (2,2), activation='relu', padding='same')(pool2)
    conv3 = Dropout(0.2)(conv3)

    ## Now split into objectness and bounding box layers
    ## TODO: Is keras.layers.merge.add the best approach here???

    # Objectness (is this the center of an object or not
    up4obj = UpSampling2D(size=(2, 2))(conv3)
    conv4obj = Conv2D(64, (2, 2), activation='relu', padding='same')(up4obj)
    conv4obj = Dropout(0.2)(conv4obj)
    merge4obj = add([conv2, conv4obj])

    up5obj = UpSampling2D(size=(2,2))(merge4obj)
    conv5obj = Conv2D(32, (2, 2), activation='relu', padding='same')(up5obj)
    conv5obj = Dropout(0.2)(conv5obj)
    merge5obj = add([conv1, conv5obj])

    #FIXME: Currently only 2 classes (background and obstacle)
    #FIXME: Shouldn't this be a softmax!!
    prediction_obj = Conv2D(2, (2, 2), activation='relu', padding='same')(merge5obj)

    # Bounding box prediction
    # Objectness (is this the center of an object or not
    up4box = UpSampling2D(size=(2, 2))(conv3)
    conv4box = Conv2D(64, (2, 2), activation='relu', padding='same')(up4box)
    conv4box = Dropout(0.2)(conv4box)
    merge4box = add([conv2, conv4box])

    up5box = UpSampling2D(size=(2,2))(merge4box)
    conv5box = Conv2D(32, (2, 2), activation='relu', padding='same')(up5box)
    conv5box = Dropout(0.2)(conv5box)
    merge5box = add([conv1, conv5box])

    #FIXME: This is a regressor??? so what does it return...??
    prediction_box = Conv2D(2, (2, 2), activation='relu', padding='same')(merge5box)

    # Setup loss, etc. and
    model = Model(inputs=[inputs], outputs=[prediction_obj, prediction_box])
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.3, nesterov=False)

    #FIXME: Initial attempt at loss and metrics...
    model.compile(optimizer=sgd,
                  loss=['categorical_crossentropy', 'mean_squared_error'],
                  metrics=['accuracy', 'mean_squared_error'])

    if weights_path != None:
        print ('Loading weights from {}'.format(weights_path))
        model.load_weights(weights_path)
        print ('Loaded!')

    return LossHistory, model   # FIXME: Return both for now, handle LossHistory for merged NN later


def camera_nn(model, num_classes, weights_path=None, w_regularizer = None, b_regularizer = None):
    return model

