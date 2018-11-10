import tensorflow
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

from prepareImageData import *
from readData import *

# Set the number of train, validation and test data
train_num = 1
validation_num = 1
test_num = 3
total_num = train_num + validation_num + test_num

# Prepare train, validation and test data
train, validation, test = prepare_image_data(train_num, validation_num, test_num)

# Convert image data into range 0 to 1
train_tensor = tf.convert_to_tensor(train)

# Prepare train, validation and test label
train_label = label(0, train_num)
validation_label = label(train_num, train_num + validation_num)
test_label = label(train_num + validation_num, total_num)

# Construct the CNN
# model = Sequential()
# model.add(layers.Conv2D(32, (3, 3)), activation='relu', input_shape=())












