import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from readData import *
from create_dataset import *


def prepare_image_data(training_num, validation_num, test_num):
    # Construct the database for model training
    train_dir, validation_dir, test_dir = create_database(training_num, validation_num, test_num)

    # Construct the label of the training, validation and test data
    # total_num = training_num + validation_num + test_num
    # train_label = label(0, training_num)
    # validation_label = label(training_num, training_num + validation_num)
    # test_label = label(training_num + validation_num, total_num)

    # Construct the image data of training data
    train_file_name = []
    for a in os.walk(train_dir):
        train_file_name = a[2]

    train_image = []
    for name in train_file_name:
        path = os.path.join(train_dir, name)
        train_image.append(load_image(path))

    # Construct the image data of test data
    test_file_name = []
    for a in os.walk(test_dir):
        test_file_name = a[2]

    test_image = []
    for name in test_file_name:
        path = os.path.join(test_dir, name)
        test_image.append(load_image(path))

    # Construct the image data of validation data
    validation_file_name = []
    for a in os.walk(validation_dir):
        validation_file_name = a[2]

    validation_image = []
    for name in validation_file_name:
        path = os.path.join(validation_dir, name)
        validation_image.append(load_image(path))

    return train_image, validation_image, test_image

