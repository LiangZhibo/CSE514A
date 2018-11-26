# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 15:47:08 2018

@author: zhangzubin
"""
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import numpy as np
from tensorflow.keras import optimizers
import pandas as pd
from prepareImageData import *
from readData import *
from random import seed
from random import randrange
from csv import reader
from math import sqrt
from sklearn.ensemble import RandomForestRegressor

# Set the number of train, validation and test data
train_num = 500
validation_num = 100
test_num = 30
total_num = train_num + validation_num + test_num

# Prepare train, validation and test data
train, validation, test = prepare_image_data(train_num, validation_num, test_num)

# the number of train, validation and test data after perturbation
new_train_num = len(train)
new_validation_num = len(validation)
new_test_num = len(test)
new_total_num = new_train_num + new_validation_num + new_test_num

# Convert image data into numpy array
train = np.array(train)
validation = np.array(validation)
test = np.array(test)

# Adjust the dimension of the image data
train = train.reshape((new_train_num, 350, 350, 1))
validation = validation.reshape((new_validation_num, 350, 350, 1))
test = test.reshape((new_test_num, 350, 350, 1))

# Convert the type of the image data to float and normalize the image data
train = train.astype(np.float64)/255
validation = validation.astype(np.float64)/255
test = test.astype(np.float64)/255

# Prepare train, validation an test label
train_label = label(0, new_train_num)
validation_label = label(new_train_num, new_train_num + validation_num)
test_label = label(new_train_num + validation_num, new_total_num)

train_nsamples, train_nx, train_ny, train_nz = train.shape
train = train.reshape((train_nsamples,train_nx*train_ny*train_nz))
validation_nsamples, validation_nx, validation_ny, validation_nz = validation.shape
validation = validation.reshape((validation_nsamples,validation_nx*validation_ny*validation_nz))
test_nsamples, test_nx, test_ny, test_nz = test.shape
test = test.reshape((test_nsamples, test_nx*test_ny*test_nz))

new_train_label = np.array(train_label)
new_validation_label = np.array(validation_label)
new_test_label = np.array(test_label)


print(validation.shape)
print(test.shape)
print(new_test_label.shape)
print(new_validation_label.shape)
rf = RandomForestRegressor(n_estimators = 5, random_state = 42)
rf.fit(train, new_train_label)
predictions = rf.predict(validation)
predictions = predictions.astype(np.int0)
print(new_validation_label)
print(predictions)
count = 0
for i in range(0,validation_num):
    if predictions[i] == new_validation_label[i]:
        count += 1
accuracy=count/validation_num
print('Accuracy:', round(accuracy, 2)*100,'%')