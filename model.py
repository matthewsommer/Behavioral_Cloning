import os.path
import csv
import cv2
import numpy as np
import sklearn

from keras.models import load_model
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
from keras.layers.convolutional import Convolution2D
from sklearn.model_selection import train_test_split

# Set constants for local file paths
model_file_path = 'model.h5'
data_csv_file_path = 'data/driving_log.csv'

org_img_rows, org_img_cols, org_img_chnls = 160, 320, 3

if os.path.exists(model_file_path):
    model = load_model(model_file_path)
    model.save(model_file_path)
    del model

lines = []

with open(data_csv_file_path, newline='') as csvfile:
    csv_data = csv.reader(csvfile, delimiter=',', quotechar='|')
    for row in csv_data:
        lines.append(row)

train_samples, validation_samples = train_test_split(lines, test_size=0.2)

print(len(train_samples))

# Generator used to pull data as needed instead of loading all into memory (saves memory)
def generator(samples, batch_size=128):
    num_samples = len(samples)
    while 1:
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            gen_images = []
            gen_angles = []

            for batch_sample in batch_samples:
                name = batch_sample[0]
                center_image = cv2.imread(name)
                center_angle = float(batch_sample[3])
                gen_images.append(center_image)
                gen_angles.append(center_angle)
                center_image_flipped = np.fliplr(center_image)
                center_angle_flipped = -center_angle
                gen_images.append(center_image_flipped)
                gen_angles.append(center_angle_flipped)

            x_train = np.array(gen_images)
            y_train = np.array(gen_angles)
            yield sklearn.utils.shuffle(x_train, y_train)

images = []
measurements = []

#Use the generator to reduce peak memory usage
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

loss = 'mse'
optimizer = 'adam'
keep_prob = 0.5
epochs = 5

model = Sequential()

# crop input images to remove useless data
model.add(Cropping2D(cropping=((70, 20), (0, 0)), input_shape=(160, 320, 3)))

# normalize the image data
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(70, 320, 3)))

model.add(Convolution2D(24, 5, 5, activation='relu', subsample=(2, 2)))
model.add(Convolution2D(36, 5, 5, activation='relu', subsample=(2, 2)))
model.add(Convolution2D(48, 5, 5, activation='relu', subsample=(2, 2)))
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dropout(keep_prob))
model.add(Dense(50, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='tanh'))

model.summary()

model.compile(loss=loss, optimizer=optimizer)

model.fit_generator(train_generator, len(train_samples), validation_data=validation_generator, nb_val_samples=len(validation_samples), nb_epoch=epochs)

model.save(model_file_path)
