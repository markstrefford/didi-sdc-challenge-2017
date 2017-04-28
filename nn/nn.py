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
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers import Input, Dense, Flatten, Activation, Dropout, MaxPooling1D
from keras.optimizers import Adam, SGD
from keras.callbacks import Callback

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


# Create the seperate parts of the CNN
# Note that in all of this, x is a tensor (see https://keras.io/getting-started/functional-api-guide/)
# Create the surround CNN
def surround_nn(model, num_classes, weights_path=None, b_regularizer = None, w_regularizer=None):
    inputs = Input(shape=(surround_x, surround_y, surround_z))
    x = inputs # Initial tensor
    nf = num_filters
    for layer in range(3):
        x=Convolution2D(num_filters, filter_length, filter_length, border_mode=border_mode,
                        activation=activation,
                        W_regularizer=w_regularizer, b_regularizer=b_regularizer,
                        name='surround_layer' + str(layer))(x)
        nf *= 2     # Increment number of filters
    predictions=x    #FIXME - What's the output of this CNN??  Candidates for objects I expect?
    model = Model(inputs=inputs, outputs=predictions)
    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return inputs, predictions, model   # FIXME: Return both for now...


# TODO: Be DRY here... lots of code repetition... can this be a single function with different parameters calling it??
def top_nn(weights_path=None, b_regularizer = None, w_regularizer=None):
    inputs = Input(shape=(top_x, top_y, top_z))
    x = inputs # Initial tensor
    num_conv_layers = 3  # For now...
    nf = num_filters
    # Create CNN layers, each one has 2x the features of the previous one (FIXME: Not sure if this is the best approach, let's train something for now!!)
    for layer in range(num_conv_layers):
        print ('Layer {}, num_filters {}'.format(layer, nf))
        x=Convolution2D(nf, (filter_length, filter_length),
                        border_mode=border_mode,
                        W_regularizer = w_regularizer,
                        b_regularizer = b_regularizer,
                        name = 'cnn_layer_' + str(layer))(x)
        x=Activation(activation)(x)
        x=MaxPooling2D(pool_size=(num_pooling, num_pooling))(x)
        x=Dropout(0.25)(x)
        print ('Layer: {}, tensor: {}'.format(layer, x))
        nf *= 2
    x=Flatten()(x)
    x=Dense(1024, W_regularizer = w_regularizer, b_regularizer = b_regularizer)(x)
    x=Activation(activation)(x)
    x=Dropout(0.5)(x)
    x=Dense(6,W_regularizer = w_regularizer, b_regularizer = b_regularizer)(x)
    x=Activation(activation)(x)
    predictions=x   # tx, ty, tz, rx, ry, rz
                    # FIXME - What's the output of this CNN??  Candidates for objects I expect?
    model = Model(input=inputs, output=predictions)
    model.compile(optimizer='rmsprop',
                  loss='mean_squared_error',    #FIXME: Just for now...
                  metrics=['accuracy'])
    return x, model   # FIXME: Return both for now...


def camera_nn(model, num_classes, weights_path=None, w_regularizer = None, b_regularizer = None):
    return model



def nn(num_classes, weights_path=None, w_regularizer = None, b_regularizer = None):
    class LossHistory(Callback):
        def on_train_begin(self, logs={}):
            self.losses = []

        def on_batch_end(self, batch, logs={}):
            self.losses.append(logs.get('loss'))


    model = Sequential()

    x, model = top_nn(model, 6)


    if weights_path:
        print "Loading weights from {}".format(weights_path)
        model.load_weights(weights_path)

    return model, LossHistory



