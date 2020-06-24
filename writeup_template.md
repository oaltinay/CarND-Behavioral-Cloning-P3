# **Behavioral Cloning** 

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)
[image1]: ./examples/nvidiaModel.png "nvidiaModel"
[image2]: ./examples/original.png "Original Image"
[image3]: ./examples/sharped.png "Sharped Image"
[image4]: ./examples/translation.png "Translated Image"
[image5]: ./examples/bright.png "Bright Image"
---

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

The model is Nvidia's End to End Learning for Self-Driving Cars-2016. They trained a convolutional neural network (CNN) to map raw pixels from a single front-facing camera directly to steering commands. This end-to-end approach proved surprisingly powerful. With minimum training data from humans the system learns to drive in traffic on local roads with or without lane markings and on highways. It also operates in areas with unclear visual guidance such as in parking lots and on unpaved roads. The network has about 27 million connections and 250 thousand parameters and the system operates at 30 frames per second (FPS).

![alt text][image1]

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 103-105-107-109). 

The model was trained and validated on data which Udacity provides. I could not collect data using simulator. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 132). 
Batch size is 128 and epoch is 15.

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road.

#### 5. Data Preprocessing

To increase the amount of data in the data set I used some image processing techniques.I change the brightness of images, applied translation and sharping to left, right and center images.

To adjust model to see correct steering angle, I added '''steering_correction''' paramater and set to 0.2.

Original image:
![alt text][image2]

Sharped image:
![alt text][image3]

Translated image:
![alt text][image4]

Brightened image:
![alt text][image5]