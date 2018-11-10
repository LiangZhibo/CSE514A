import os
import shutil
import pandas as pd


def create_database(train_num, validation_num, test_num):
    # Read image name and label from the label.csv
    df = pd.read_csv('E:\Database\label.csv')
    image_names = df['image']

    # Create directory for train
    base_dir = 'E:\Database\Train'
    os.mkdir(base_dir)

    # Create directories for the training, validation and test splits
    train_dir = os.path.join(base_dir, 'Train')
    validation_dir = os.path.join(base_dir, 'Validation')
    test_dir = os.path.join(base_dir, "Test")
    os.mkdir(train_dir)
    os.mkdir(validation_dir)
    os.mkdir(test_dir)

    total_num = train_num + validation_num + test_num

    for x in range(0, train_num):
        src = os.path.join(r'E:\Database\images', image_names[x])
        shutil.copy(src, train_dir)

    for x in range(train_num, train_num + validation_num):
        src = os.path.join(r'E:\Database\images', image_names[x])
        shutil.copy(src, validation_dir)

    for x in range(train_num + validation_num, total_num):
        src = os.path.join(r'E:\Database\images', image_names[x])
        shutil.copy(src, test_dir)

    return train_dir, validation_dir, test_dir


def label(start_num, end_num):
    data = pd.read_csv('E:\Database\label.csv')
    total_label = data['emotion']
    sample_label = total_label[start_num: end_num]
    output = []
    for x in sample_label:
        temp = x.lower()
        if temp == 'happiness':
            output.append(0)

        if temp == 'neutral':
            output.append(1)

        if temp == 'sadness':
            output.append(2)

        if temp == 'fear':
            output.append(3)

        if temp == 'anger':
            output.append(4)

        if temp == 'surprise':
            output.append(5)

        if temp == 'disgust' or temp == 'contempt':
            output.append(6)

    return output

















