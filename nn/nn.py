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
from keras.layers import Input, Dense, Flatten, merge, \
                         Activation, Conv2D, MaxPooling2D, UpSampling2D, \
                         Reshape, core, Dropout
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


# Create the seperate parts of the CNN
# Note that in all of this, x is a tensor (see https://keras.io/getting-started/functional-api-guide/)
# Create the surround CNN
# def surround_nn(model, num_classes, weights_path=None, b_regularizer = None, w_regularizer=None):
#     inputs = Input(shape=(surround_x, surround_y, surround_z))
#     x = inputs # Initial tensor
#     nf = num_filters
#     for layer in range(3):
#         x=Conv2D(num_filters, filter_length, filter_length, border_mode=border_mode,
#                         activation=activation,
#                         W_regularizer=w_regularizer, b_regularizer=b_regularizer,
#                         name='surround_layer' + str(layer))(x)
#         nf *= 2     # Increment number of filters
#     predictions=x    #FIXME - What's the output of this CNN??  Candidates for objects I expect?
#     model = Model(inputs=inputs, outputs=predictions)
#     model.compile(optimizer='rmsprop',
#                   loss='categorical_crossentropy',
#                   metrics=['accuracy'])
#     return inputs, predictions, model   # FIXME: Return both for now...


# TODO: Be DRY here... lots of code repetition... can this be a single function with different parameters calling it??
def top_nn(weights_path=None, b_regularizer = None, w_regularizer=None):
    class LossHistory(Callback):
        def on_train_begin(self, logs={}):
            self.losses = []

        def on_batch_end(self, batch, logs={}):
            self.losses.append(logs.get('loss'))

    # num_conv_layers = 3  # For now...
    # nf = num_filters

    # Create CNN layers, each one has 2x the features of the previous one (FIXME: Not sure if this is the best approach, let's train something for now!!)
    # for layer in range(num_conv_layers):
    #     print ('Layer {}, num_filters {}'.format(layer, nf))
    #     x=Conv2D(nf, (filter_length, filter_length),
    #                     border_mode=border_mode,
    #                     W_regularizer = w_regularizer,
    #                     b_regularizer = b_regularizer,
    #                     name = 'cnn_layer_' + str(layer))(x)
    #     x=Activation(activation)(x)
    #     x=MaxPooling2D(pool_size=(num_pooling, num_pooling))(x)
    #     x=Dropout(0.25)(x)
    #     print ('Layer: {}, tensor: {}'.format(layer, x))
    #     nf *= 2
    #
    # x=Flatten()(x)
    # x=Dense(1024, W_regularizer = w_regularizer, b_regularizer = b_regularizer)(x)
    # x=Activation(activation)(x)
    # x=Dropout(0.5)(x)
    # x=Dense(6,W_regularizer = w_regularizer, b_regularizer = b_regularizer)(x)
    # x=Activation(activation)(x)
    # predictions=x   # tx, ty, tz, rx, ry, rz
    #                 # FIXME - What's the output of this CNN??  Candidates for objects I expect?

    inputs = Input(shape=(top_x, top_y, top_z))
    print (inputs)

    conv1 = Conv2D(32, (2,2), activation='relu', padding='same')(inputs)
    conv1 = Dropout(0.2)(conv1)
    #conv1 = Conv2D(32, 3, 3, activation='relu', border_mode='same')(conv1)
    #print ('conv1 {}'.format(conv1))
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    #print ('pool1 {}'.format(pool1))

    conv2 = Conv2D(64, (2,2), activation='relu', padding='same')(pool1)
    conv2 = Dropout(0.2)(conv2)
    #conv2 = Conv2D(64, 3, 3, activation='relu', border_mode='same')(conv2)
    #print ('conv2 {}'.format(conv2))
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    #print ('pool2 {}'.format(pool2))

    conv3 = Conv2D(128, (2,2), activation='relu', padding='same')(pool2)
    conv3 = Dropout(0.2)(conv3)
    #conv3 = Conv2D(128, 3, 3, activation='relu', border_mode='same')(conv3)
    print ('conv3 {}'.format(conv3))
    #pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    ## Now split into objectness and bounding box layers

    # Objectness (is this the center of an object or not
    up4obj = UpSampling2D(size=(2, 2))(conv3)
    conv4obj = Conv2D(128, (2, 2), activation='relu', padding='same')(up4obj)
    conv4obj = Dropout(0.2)(conv4obj)
    merge4obj = merge([conv2, conv4obj])

    up5obj = UpSampling2D(size=(2,2))(merge4obj)
    conv5obj = Conv2D(32, (2, 2), activation='relu', padding='same')(up5obj)
    conv5obj = Dropout(0.2)(conv5obj)
    merge5obj = merge([conv1, conv5obj])

    #FIXME: Currently only 2 classes (background and obstacle)
    prediction_obj = Conv2D(2, (2, 2), activation='relu', padding='same')(merge5obj)


    # Bounding box prediction
    # Objectness (is this the center of an object or not
    up4box = UpSampling2D(size=(2, 2))(conv3)
    conv4box = Conv2D(128, (2, 2), activation='relu', padding='same')(up4box)
    conv4box = Dropout(0.2)(conv4box)
    merge4box = merge([conv2, conv4box])

    up5box = UpSampling2D(size=(2,2))(merge4box)
    conv5box = Conv2D(32, (2, 2), activation='relu', padding='same')(up5box)
    conv5box = Dropout(0.2)(conv5box)
    merge5box = merge([conv1, conv5box])

    #FIXME: This is a regressor??? so what does it return...??
    prediction_box = Conv2D(2, (2, 2), activation='relu', padding='same')(merge5box)







    model = Model(input=inputs, output=[object_predictions, bbox_predictions])
    #

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



# def nn(weights_path=None, w_regularizer = None, b_regularizer = None):
#
#
#
#     model = Sequential()
#
#     xmodel = top_nn(model, 6)
#     if weights_path:
#         print "Loading weights from {}".format(weights_path)
#         model.load_weights(weights_path)
#     return model, LossHistory


## Inspired by https://github.com/orobix/retina-unet/blob/master/src/retinaNN_training.py
def get_unet(top_z,top_y,top_x):
    inputs = Input((top_z, top_y, top_x))
    conv1 = Conv2D(32, 3, 3, activation='relu', border_mode='same')(inputs)
    conv1 = Dropout(0.2)(conv1)
    conv1 = Conv2D(32, 3, 3, activation='relu', border_mode='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    #
    conv2 = Conv2D(64, 3, 3, activation='relu', border_mode='same')(pool1)
    conv2 = Dropout(0.2)(conv2)
    conv2 = Conv2D(64, 3, 3, activation='relu', border_mode='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    #
    conv3 = Conv2D(128, 3, 3, activation='relu', border_mode='same')(pool2)
    conv3 = Dropout(0.2)(conv3)
    conv3 = Conv2D(128, 3, 3, activation='relu', border_mode='same')(conv3)

    up1 = merge([UpSampling2D(size=(2, 2))(conv3), conv2], mode='concat', concat_axis=1)
    conv4 = Conv2D(64, 3, 3, activation='relu', border_mode='same')(up1)
    conv4 = Dropout(0.2)(conv4)
    conv4 = Conv2D(64, 3, 3, activation='relu', border_mode='same')(conv4)
    #
    up2 = merge([UpSampling2D(size=(2, 2))(conv4), conv1], mode='concat', concat_axis=1)
    conv5 = Conv2D(32, 3, 3, activation='relu', border_mode='same')(up2)
    conv5 = Dropout(0.2)(conv5)
    conv5 = Conv2D(32, 3, 3, activation='relu', border_mode='same')(conv5)
    #
    conv6 = Conv2D(2, 1, 1, activation='relu',border_mode='same')(conv5)
    #conv6 = core.Reshape((2,patch_height*patch_width))(conv6)
    #conv6 = core.Permute((2,1))(conv6)
    ############
    conv7 = core.Activation('softmax')(conv6)

    model = Model(input=inputs, output=conv7)

    # sgd = SGD(lr=0.01, decay=1e-6, momentum=0.3, nesterov=False)
    model.compile(optimizer='sgd', loss='categorical_crossentropy',metrics=['accuracy'])

    return model
