import os.path
import csv
import cv2
import numpy as np

from keras.models import load_model
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D
from keras.layers.convolutional import Convolution2D

model_file_path = 'model.h5'
data_csv_file_path = 'data/driving_log.csv'

if os.path.exists(model_file_path):
    model = load_model(model_file_path)
    model.save(model_file_path)
    del model

lines = []

with open(data_csv_file_path, newline='') as csvfile:
    csv_data = csv.reader(csvfile, delimiter=',', quotechar='|')
    for row in csv_data:
        lines.append(row)

images = []
measurements = []
for line in lines:
    img_source_path = line[0]
    image = cv2.imread(img_source_path)
    images.append(image)
    measurement = float(line[3])
    measurements.append(measurement)
    image_flipped = np.fliplr(image)
    measurement_flipped = -measurement
    images.append(image_flipped)
    measurements.append(measurement_flipped)

X_train = np.array(images)
y_train = np.array(measurements)

print('X_train length', len(X_train))
print('y_train length', len(y_train))

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
model.add(Dense(50, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='tanh'))

model.summary()

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=4)

model.save(model_file_path)
