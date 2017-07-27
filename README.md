# Behavioral Cloning

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* run1.mp4 showing a video of the car driving around the track
* README.md summarizing the results

To generate a model the following shell command can be executed
```sh
python model.py
```

To start a daemon used to control the simulator in autonomous mode the following command can be used
```sh
python drive.py model.h5
```

To record images of the a run the following command can be used (does not generate video, look at next command)
```sh
python drive.py model.h5 run1
```

After generating images of a run, you can use this command to generate an mp4 video with a given framerate
```sh
python video.py run1 --fps 48
```

The model.py file contains the code for training and saving the convolution neural network. 
The file shows the pipeline I used for training and validating the model, and it contains 
comments to explain how the code works.

# Model Architecture and Training Strategy

### NVIDIA Pipeline Model

For this project I used the NVIDIA pipeline model. It seemed like a logical choice as it's a fairly simple model and with
my initial trials it was very fast. On my new MacBook pro I could train the model in minutes.

To reduce overfitting I added a dropout layer to the model. The model actually worked without having to add a dropout
but because it's required in the rubric for the project I added it. I used a dropout probability of .5 as this seems to be a 
common dropout rate and it worked well.

I used an adam optimizer, no learning rate set manually.

# Model Architecture and Training Strategy

My approach to solving this problem was to get a baseline code working then to iterate on it. I didn't want to spend too
much time on one part only to find out later I didn't do something correctly. So I wrote my code with a very simple model
so the car would at least start driving then I began trying to tweak the model and get better data to train the model.
Since I started with the NVIDIA pipeline I quickly discovered that it was performing very well and my training data was what was
causing me issues getting around the track. I recorded multiple runs driving both directions, then I recorded short segments
of runs on particularly difficult turns. I wanted to have multiple full runs around the track but also focused on edge 
cases of tight turns or odd objects on the sides of the roads on particular turns. I also made sure to get good training data
that taught the car to correct when going off course, but made sure to not use training data that would teach bad behaviour like 
driving off the track.

Once I got plenty of training data the car was easily able to drive around the track without running off the road.

Here is a visualization of the architecture

* cropping2d_1 (Cropping2D)        (None, 70, 320, 3)    0           cropping2d_input_2[0][0]
* lambda_1 (Lambda)                (None, 70, 320, 3)    0           cropping2d_1[0][0]               
* convolution2d_1 (Convolution2D)  (None, 33, 158, 24)   1824        lambda_1[0][0]                   
* convolution2d_2 (Convolution2D)  (None, 15, 77, 36)    21636       convolution2d_1[0][0]            
* convolution2d_3 (Convolution2D)  (None, 6, 37, 48)     43248       convolution2d_2[0][0]            
* convolution2d_4 (Convolution2D)  (None, 4, 35, 64)     27712       convolution2d_3[0][0]            
* convolution2d_5 (Convolution2D)  (None, 2, 33, 64)     36928       convolution2d_4[0][0]            
* flatten_1 (Flatten)              (None, 4224)          0           convolution2d_5[0][0]            
* dense_1 (Dense)                  (None, 100)           422500      flatten_1[0][0]                  
* dropout_1 (Dropout)              (None, 100)           0           dense_1[0][0]                    
* dense_2 (Dense)                  (None, 50)            5050        dropout_1[0][0]                  
* dense_3 (Dense)                  (None, 10)            510         dense_2[0][0]                    
* dense_4 (Dense)                  (None, 1)             11          dense_3[0][0]                    

Total params: 559,419
Trainable params: 559,419
Non-trainable params: 0