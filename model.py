import os
import csv
import cv2

import matplotlib.image as mpimg
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

from keras.models import Sequential
from keras.layers import Cropping2D, Lambda, Flatten, Dense, Dropout
from keras.layers.convolutional import Conv2D
from keras.callbacks import ModelCheckpoint, TensorBoard

def sharp(img):
    gb  = cv2.GaussianBlur(img, (7,7), 15.0)
    shp = cv2.addWeighted(img, 2, gb, -1, 0)
    return shp.reshape(160,320,3)

def brightness(img):
    image = cv2.cvtColor(img,cv2.COLOR_RGB2HSV)
    image = np.array(image, dtype = np.float64)
    random_bright = .5+np.random.uniform()
    image[:,:,2] = image[:,:,2]*random_bright
    image[:,:,2][image[:,:,2]>255]  = 255
    image = np.array(image, dtype = np.uint8)
    image = cv2.cvtColor(image,cv2.COLOR_HSV2RGB)
    return image

def translation(image,steer,trans_range):
    # Translation
    rows,cols,channels = image.shape
    tr_x = trans_range*np.random.uniform()-trans_range/2
    steer_ang = steer + tr_x/trans_range*2*.2
    tr_y = 10*np.random.uniform()-10/2
    Trans_M = np.float32([[1,0,tr_x],[0,1,tr_y]])
    image_tr = cv2.warpAffine(image,Trans_M,(cols,rows))

    return image_tr,steer_ang,tr_x

csv_path  = '/home/workspace/data/driving_log.csv'
img_path  = '/home/workspace/data/IMG/'

def sample_generator(samples, batch_size=32):
    steering_correction = 0.2
    num_samples = len(samples)
    while 1:
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            images = []
            steering_angles = []
            for idx,batch_sample in batch_samples.iterrows():
                for i in range(0,3):
                    if i == 0:
                        steering_correction = 0
                    elif i == 1:
                        steering_correction = 0.2
                    else:
                        steering_correction = -0.2
                        
                    #image
                    image_name = img_path+batch_sample[i].split('/')[-1]      
                    image = mpimg.imread(image_name)
                    steering_angle = float(batch_sample[3])
                    images.append(image)
                    steering_angles.append(steering_angle + steering_correction)
                  
                    #brightness
                    bright_image = brightness(image)
                    
                    images.append(bright_image)
                    steering_angles.append(steering_angle + steering_correction)
                    
                    #translation 
                    translated = translation(image,float(batch_sample[3]),40)
                    images.append(translated[0])
                    steering_angles.append(translated[1] + steering_correction)
                                     
                    #Sharp
                    sharp_img = sharp(image)
                    images.append(sharp_img)
                    steering_angles.append(steering_angle + steering_correction)
          
            X_train = np.array(images)
            y_train = np.array(steering_angles)
            yield shuffle(X_train, y_train)
            
# Nvidia End-to-End Learning for Self-Driving Cars paper.
def NvidiaModel():
    # Reference Nvidia Model
    model = Sequential()
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
    model.add(Cropping2D(cropping=((70,25),(0,0))))  
    model.add(Conv2D(24, (5, 5), subsample=(2, 2), activation='relu'))
    model.add(Conv2D(36, (5, 5), subsample=(2, 2), activation='relu'))
    model.add(Conv2D(48, (5, 5), subsample=(2, 2), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Flatten())
    model.add(Dropout(0.2))
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(50, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(10, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1))

    return model

# main function
def main():
    drive_log = pd.read_csv(csv_path)
    drive_log = drive_log[drive_log['steering'] != 0].append(drive_log[drive_log['steering'] == 0].sample(frac=0.5))

    train_samples, validation_samples = train_test_split(drive_log, test_size=0.20)
    print('Train samples: {}'.format(len(train_samples)))
    print('Validation samples: {}'.format(len(validation_samples)))
    
    # model training hyperparameter batch_size
    batch_size = 128
    # model training hyperparameter epochs
    epochs = 15

    train_generator      = sample_generator(train_samples, batch_size=batch_size)
    validation_generator = sample_generator(validation_samples, batch_size=batch_size)
    
    # structure and compile the model
    model = NvidiaModel()
    model.compile(loss='mse', optimizer='adam', metrics=["accuracy"])

    #Train model
    history_object = model.fit_generator(train_generator, 
            steps_per_epoch=int(np.ceil(len(train_samples)/batch_size)), 
            validation_data=validation_generator, 
            validation_steps=int(np.ceil(len(validation_samples)/batch_size)), 
            epochs=epochs, verbose=1)
    #Save model
    model.save('model.h5')


if __name__ == "__main__":
    main()