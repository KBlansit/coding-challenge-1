"""Script for training data"""
import h5py
import logging
import numpy as np

from keras.models import Model, Input, load_model
from keras.layers import Convolution2D, MaxPooling2D, UpSampling2D, concatenate
from keras.optimizers import Adam, SGD
from keras.losses import binary_crossentropy

from src.data_generator import DataGenerator

# paths
DATA_PATH = "output/contour_annotations.hdf5"

# deep learning parameters
BATCH_SIZE = 8

KERNEL_SIZE = 3
EPOCHS = 10
LEARN_RATE = 10e-4
DECAY = DECAY = LEARN_RATE/(EPOCHS)
MOMENTUM = 0.99

INPUT_SHAPE = (256, 256, 1)

logging.basicConfig(
    filename = "logs/trainig.log",
    format = '%(asctime)-15s %(pathname)s %(message)-15s',
    level = logging.DEBUG,
)

# user defined functions
def get_2d_u_net_segmentation():

    # define inputs
    inputs = Input(INPUT_SHAPE, name = "input")

    # define first convolutional layer
    conv1 = Convolution2D(32, KERNEL_SIZE, padding="same", activation="relu", name = "conv1_1")(inputs)
    conv1 = Convolution2D(32, KERNEL_SIZE, padding="same", activation="relu", name = "conv1_2")(conv1)
    pool1 = MaxPooling2D(pool_size=2, name = "max_pool_1")(conv1)

    # define second convolutional layer
    conv2 = Convolution2D(64, KERNEL_SIZE, padding="same", activation="relu", name = "conv2_1")(pool1)
    conv2 = Convolution2D(64, KERNEL_SIZE, padding="same", activation="relu", name = "conv2_2")(conv2)
    pool2 = MaxPooling2D(pool_size=2, name = "max_pool_2")(conv2)

    # define third convolutional layer
    conv3 = Convolution2D(128, KERNEL_SIZE, padding="same", activation="relu", name = "conv3_1")(pool2)
    conv3 = Convolution2D(128, KERNEL_SIZE, padding="same", activation="relu", name = "conv3_2")(conv3)
    pool3 = MaxPooling2D(pool_size=2, name = "max_pool_3")(conv3)

    # define fourth convolutional layer
    conv4 = Convolution2D(256, KERNEL_SIZE, padding="same", activation="relu", name = "conv4_1")(pool3)
    conv4 = Convolution2D(256, KERNEL_SIZE, padding="same", activation="relu", name = "conv4_2")(conv4)
    conv4 = Convolution2D(256, KERNEL_SIZE, padding="same", activation="relu", name = "conv4_3")(conv4)
    conv4 = Convolution2D(256, KERNEL_SIZE, padding="same", activation="relu", name = "conv4_4")(conv4)
    pool4 = MaxPooling2D(pool_size=2, name = "max_pool_4")(conv4)

    # define fifth convolutional layer
    conv5 = Convolution2D(512, KERNEL_SIZE, padding="same", activation="relu", name = "conv5_1")(pool4)
    conv5 = Convolution2D(512, KERNEL_SIZE, padding="same", activation="relu", name = "conv5_2")(conv5)
    conv5 = Convolution2D(512, KERNEL_SIZE, padding="same", activation="relu", name = "conv5_3")(conv5)
    conv5 = Convolution2D(512, KERNEL_SIZE, padding="same", activation="relu", name = "conv5_4")(conv5)

    # define sixth convolutional layer
    up6 = UpSampling2D(size=2, name = "up_sample_6")(conv5)
    merged6 = concatenate([up6, conv4], axis=-1, name = "merged_6")
    conv6 = Convolution2D(256, KERNEL_SIZE, padding="same", activation="relu", name = "conv6_1")(merged6)
    conv6 = Convolution2D(256, KERNEL_SIZE, padding="same", activation="relu", name = "conv6_2")(conv6)

    # define seventh convolutional layer
    up7 = UpSampling2D(size=2, name = "up_sample_7")(conv6)
    merged7 = concatenate([up7, conv3], axis=-1, name = "merged_7")
    conv7 = Convolution2D(128,  KERNEL_SIZE, padding="same", activation="relu", name = "conv7_1")(merged7)
    conv7 = Convolution2D(128, KERNEL_SIZE, padding="same", activation="relu", name = "conv7_2")(conv7)

    # define eighth convolutional layer
    up8 = UpSampling2D(size=2, name = "up_sample_8")(conv7)
    merged8 = concatenate([up8, conv2], axis=-1, name = "merged_8")
    conv8 = Convolution2D(64, KERNEL_SIZE, padding="same", activation="relu", name = "conv8_1")(merged8)
    conv8 = Convolution2D(64, KERNEL_SIZE, padding="same", activation="relu", name = "conv8_2")(conv8)

    # define ninth convolutional layer
    up9 = UpSampling2D(size=2, name = "up_sample_9")(conv8)
    merged9 = concatenate([up9, conv1], axis=-1, name = "merged_9")
    conv9 = Convolution2D(32, KERNEL_SIZE, padding="same", activation="relu", name = "conv9_1")(merged9)
    conv9 = Convolution2D(32, KERNEL_SIZE, padding="same", activation="relu", name = "conv9_2")(conv9)

    # define final output layer
    conv10 = Convolution2D(1, 1, name = "segmentation")(conv9)

    # define model and compile
    model = Model(inputs=inputs, outputs=[conv10])

    # compile
    model.compile(
        loss=binary_crossentropy,
        optimizer=SGD(
            lr=float(LEARN_RATE),
            decay=float(DECAY),
            momentum=MOMENTUM,
        ),
    )

    return model

# load u net
u_net = get_2d_u_net_segmentation()

# get data gen object
data_gen = DataGenerator(DATA_PATH, BATCH_SIZE)

# fit model
u_net.fit_generator(
    generator = data_gen.data_generator(),
    steps_per_epoch = data_gen.get_num_steps(),
    epochs = EPOCHS,
)
