import tensorflow
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import numpy as np
from tensorflow.keras import optimizers

from prepareImageData import *
from readData import *

# Set the number of train, validation and test data
train_num = 500
validation_num = 50
test_num = 30
total_num = train_num + validation_num + test_num

# Prepare train, validation and test data
train, validation, test = prepare_image_data(train_num, validation_num, test_num)

# Convert image data into numpy array
train = np.array(train)
validation = np.array(validation)
test = np.array(test)

# Adjust the dimension of the image data
train = train.reshape((train_num, 350, 350, 1))
validation = validation.reshape((validation_num, 350, 350, 1))
test = test.reshape((test_num, 350, 350, 1))

# Convert the type of the image data to float and normalize the image data
train = train.astype(np.float64)/255
validation = validation.astype(np.float64)/255
test = test.astype(np.float64)/255

# Prepare train, validation and test label
train_label = label(0, train_num)
validation_label = label(train_num, train_num + validation_num)
test_label = label(train_num + validation_num, total_num)

# Construct the CNN
model = Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(350, 350, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(256, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(256, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(1, activation='relu'))
# model.summary()

# Configure the model for training
model.compile(loss='binary_crossentropy', optimizer=optimizers.RMSprop(lr=1e-4), metrics=['acc'])

# Fit the model
print('Start training')
model.fit(train, train_label, epochs=5, batch_size=64)














