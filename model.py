import os.path
import csv
import cv2
import numpy as np

from keras.models import load_model
from keras.models import Sequential
from keras.layers import Flatten, Dense

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
    image = cv2.imread('data/' + img_source_path)
    images.append(image)
    measurement = float(line[3])
    measurements.append(measurement)

X_train = np.array(images)
y_train = np.array(measurements)

model = Sequential()
model.add(Flatten(input_shape=(160, 320, 3)))
model.add(Dense(1))
model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=7)

model.save(model_file_path)
