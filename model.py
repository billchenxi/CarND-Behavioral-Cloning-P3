import csv
import cv2
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from keras.models import Sequential, load_model
from keras.layers import Flatten, Dense, Lambda, Convolution2D, Dropout
from keras.regularizers import l2
from keras.optimizers import Adam
import random
import math
import os

samples = []

# Read the data from the csv file
def Read_Data():
    for i in range(1):
        with open('data/driving_log.csv') as csvfile:
            reader = csv.reader(csvfile)
            next(reader)
            for line in reader:
                samples.append(line)

# Augmentation Function for brightness adjustment
def brightness_adjustment(img, bval=None):
    # convert to HSV so that its easy to adjust brightness
    # new_img = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    # randomly generate the brightness reduction factor
    # Add a constant so that it prevents the image from being completely dark
    if bval:
        random_bright = bval
    else:
        random_bright = .25 + np.random.uniform()

    # Apply the brightness reduction to the V channel
    img[:, :, 2] = img[:, :, 2] * random_bright

    return img

# Augmentation Function for flipping the image and changing steering angle
def flip_image(image, steering_angle):
    image = cv2.flip(image,1)
    steering_angle = -1 * steering_angle
    return image, steering_angle

# Augmentation function for shifting image and steering angle
def shift_image(image, steer_ang, shift_range):
    # Translation
    rows, cols = image.shape[0:2]
    tr_x = shift_range * np.random.uniform() - shift_range / 2
    steer_ang = steer_ang + tr_x / shift_range * 2 * .2
    tr_y = 40 * np.random.uniform() - 40 / 2
    M = np.float32([[1, 0, tr_x], [0, 1, tr_y]])
    image_tr = cv2.warpAffine(image, M, (cols, rows))

    return image_tr, steer_ang

# Augmentation function to create random shadow
def random_shadow(image):

    shadow_factor = 0.3
    img_width = image.shape[1]
    img_height = image.shape[0]
    x = random.randint(0, img_width)
    y = random.randint(0, img_height)

    width = random.randint(int(img_width / 2), img_width)
    if (x + width > img_width):
        x = img_width - x

    height = random.randint(int(img_height / 2), img_height)
    if (y + height > img_height):
        y = img_height - y

    image[y:y + height, x:x + width, 2] = image[y:y + height, x:x + width, 2] * shadow_factor
    return image

# Pre-Processing Pipeline function for image data
def PreProcess_Data(image, steering_angle):
    #first change image from BGR to HSV
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # crop image
    image = image[50:(image.shape[0] - 25), 0:image.shape[1]]
    image = brightness_adjustment(image)
    image, steer_angle = shift_image(image, steering_angle, 100)

    if (random.random() <= 0.5):
        image = random_shadow(image)

    if (random.random() <= 0.5):
        image, steer_angle = flip_image(image, steer_angle)

    image = cv2.resize(image, (64, 64), interpolation=cv2.INTER_AREA)
    return image, steer_angle

# Generator function
def generator(samples, batch_size=32):
    num_samples = samples.shape[0]
    threshold = 1
    while 1:  # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            images = []
            steering_angles = []
            for batch_sample in batch_samples:
                correction = 0.25  # correction angle for steering left and right images
                steering_center = float(batch_sample[3])

                # Since the images are biased towards the center steering angle values
                # we will randonmly take the values for center images so that the network
                # doesn't become bias in learning

                take = 0
                if (abs(steering_center) < 0.1):
                    if (random.random() >= 0.7):
                        take = 1
                else:
                    take = 1

                if take == 1:
                    steering_left = steering_center + correction
                    steering_right = steering_center - correction

                    # img_center = cv2.imread(batch_sample[0].split('\\')[-2] + '/' + batch_sample[0].split('\\')[-1])
                    # img_left = cv2.imread(batch_sample[1].split('\\')[-2] + '/' + batch_sample[1].split('\\')[-1])
                    # img_right = cv2.imread(batch_sample[2].split('\\')[-2] + '/' + batch_sample[2].split('\\')[-1])

                    img_center = cv2.imread('data/IMG/' + batch_sample[0].split('/')[-1])
                    img_left = cv2.imread('data/IMG/' + batch_sample[1].split('/')[-1])
                    img_right = cv2.imread('data/IMG/' + batch_sample[2].split('/')[-1])

                    img_center, steering_center = PreProcess_Data(img_center, steering_center)
                    img_left, steering_left = PreProcess_Data(img_left, steering_left)
                    img_right, steering_right = PreProcess_Data(img_right, steering_right)

                    images.append(img_center)
                    steering_angles.append(steering_center)

                    images.append(img_left)
                    steering_angles.append(steering_left)

                    images.append(img_right)
                    steering_angles.append(steering_right)

            X_train = np.array(images)
            y_train = np.array(steering_angles)
            yield sklearn.utils.shuffle(X_train, y_train)

# Step 1 - Get the data
Read_Data()
print(len(samples))

#Step 2 - Split the data b/w training and validation in 80 and 20 ratio
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

# NVIDIA Based Model
def NVIDIA_Model():
    model = Sequential()
    #1st Layer - Add a flatten layer)
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(64,64,3)))
    model.add(Convolution2D(24,5,5,border_mode='valid', activation='elu', subsample=(2,2), W_regularizer=l2(0.0001), init='normal'))
    model.add(Dropout(0.5))
    model.add(Convolution2D(36,5,5,border_mode='valid', activation='elu', subsample=(2,2), W_regularizer=l2(0.0001), init='normal'))
    model.add(Dropout(0.5))
    model.add(Convolution2D(48,5,5,border_mode='valid', activation='elu', subsample=(2,2), W_regularizer=l2(0.0001), init='normal'))
    model.add(Dropout(0.5))
    model.add(Convolution2D(64,3,3,border_mode='valid', activation='elu', subsample=(1,1), W_regularizer=l2(0.0001), init='normal'))
    model.add(Dropout(0.5))
    model.add(Convolution2D(64,3,3,border_mode='valid', activation='elu', subsample=(1,1), W_regularizer=l2(0.0001), init='normal'))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(1164, activation='elu', W_regularizer=l2(0.0001), init='normal'))
    model.add(Dropout(.5))
    model.add(Dense(100, activation='elu', W_regularizer=l2(0.0001), init='normal'))
    model.add(Dropout(.4))
    model.add(Dense(50, activation='elu', W_regularizer=l2(0.0001), init='normal'))
    model.add(Dropout(.25))
    model.add(Dense(10, activation='elu', W_regularizer=l2(0.0001), init='normal'))
    model.add(Dense(1))
    return model

#Step 3
# If the model file doesn't exist call the NVIDIA model else use the saved model
if os.path.isfile('./model.h5'):
    nvidia_model = load_model('model.h5')
else:
    nvidia_model = NVIDIA_Model()

# Compile the model
nvidia_model.compile(loss='mse', optimizer=Adam(lr=0.0001))

# Convert the training and validation samples into array
train_samples = np.array(train_samples)
validation_samples = np.array(validation_samples)

print(train_samples.shape)
print(validation_samples.shape)

# Calculation for Samples per Epoch
def calc_samples_per_epoch(array_size, batch_size):
    num_batches = array_size / batch_size
    samples_per_epoch = math.ceil(num_batches)
    samples_per_epoch = samples_per_epoch * batch_size
    return samples_per_epoch

# Hyperparameter Batch Size
BATCH_SIZE = 128

# Get the Samples per epoch for training and validation
# Since we are adding also the left and right images for training and validating
# we multiplied the samples by 3
samples_per_epoch = calc_samples_per_epoch((len(train_samples)*3), BATCH_SIZE)
nb_val_samples = calc_samples_per_epoch((len(validation_samples)*3), BATCH_SIZE)

# Create the generator for training and validation
train_generator = generator(train_samples, BATCH_SIZE)
validation_generator = generator(validation_samples, BATCH_SIZE)

history = nvidia_model.fit_generator(train_generator, samples_per_epoch=samples_per_epoch,
                           validation_data=validation_generator,
                           nb_val_samples=nb_val_samples, nb_epoch=50)

# Save the model
nvidia_model.save('model.h5')

# Get the summary of the model
print(nvidia_model.summary())