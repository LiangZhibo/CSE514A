# -*- coding: utf-8 -*-
"""
Created on Sun Nov 11 12:59:26 2018

@author: zhangzubin
"""
import os
import tensorflow as tf
import cv2
import random
import math
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

    train_label = label(0, training_num)
    train_image = perturbation(train_image, train_label)
    print(len(train_image))

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

def perturbation(train_image, train_label):
    # store the number of images of an emotion for all emotions
    all_emotion_nums = []
    for emotion in range(7):
        all_emotion_nums.append(get_emotion_num(train_label, emotion))

    # trained images reorderd
    images_inorder = []

    # different emotions are stored into different lists,
    # and all of them are contained in a list (two dimensional list)
    for i in range(7):
        images_inorder.append([])
    for i in range(len(train_image)):
        current_image_label = train_label[i]
        images_inorder[current_image_label].append(train_image[i])

    # image data after perturbation
    images_after_perturb = train_image


    # get the emotion that has the most number
    most_emotion = get_most_emotion_num(all_emotion_nums)

    # lets say if we have 150 sadness, 100 happiness, and 40 fear
    # what I am doing is find the emotion with the greatest number of images
    # which is 150 in this case. Then, subtract 100 and 40 respectively,
    # and then we can get the number of additional images we expect
    # so that
    for emotion in range(7):
        current_emotion_num = all_emotion_nums[emotion]

        # the number of perturbation needed for this emotion
        num_perturb = most_emotion - current_emotion_num

        image_count = 0
        while (current_emotion_num < all_emotion_nums[most_emotion]):
            if (image_count >= len(images_inorder[emotion])-1):
                image_count = 0
            image = images_inorder[emotion][image_count]
            perturb_image = image_rotation(image)
            images_after_perturb.append(perturb_image)
            current_emotion_num += 1
            image_count += 1

    return images_after_perturb

# get the number of images with a specific emotion
def get_emotion_num(train_label, emotion):
    emotion_num = 0
    for e in train_label:
        if (e == emotion):
            emotion_num += 1
    return emotion_num

# get the emotion label that has the greatest number of images
def get_most_emotion_num(all_emotion_nums):
    max = 0
    for i in range(len(all_emotion_nums)):
        if (all_emotion_nums[i] > all_emotion_nums[max]):
            max = i
    return max

# rotate image
def image_rotation(image):
    # rotate image here
    rows = image.shape[0]
    cols = image.shape[1]
    degree = random.uniform(0, 2*math.pi)
    M = cv2.getRotationMatrix2D(((cols-1)/2.0, (rows-1)/2.0), degree, 1)
    return cv2.warpAffine(image, M, (cols, rows))
