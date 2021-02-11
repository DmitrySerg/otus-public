import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def read_signals(filename):
    with open(filename, 'r') as fp:
        data = fp.read().splitlines()
        data = map(lambda x: x.rstrip().lstrip().split(), data)
        data = [list(map(float, line)) for line in data]
    return data

def read_labels(filename):        
    with open(filename, 'r') as fp:
        activities = fp.read().splitlines()
        activities = list(map(int, activities))
    return activities

def randomize(dataset, labels):
    permutation = np.random.permutation(labels.shape[0])
    shuffled_dataset = dataset[permutation, :, :]
    shuffled_labels = labels[permutation]
    return shuffled_dataset, shuffled_labels


def read_data(input_folder_train, input_folder_test, labelfile_train, labelfile_test):
    
    INPUT_FILES_TRAIN = [
        'body_acc_x_train.txt', 'body_acc_y_train.txt', 'body_acc_z_train.txt', 
        'body_gyro_x_train.txt', 'body_gyro_y_train.txt', 'body_gyro_z_train.txt',
        'total_acc_x_train.txt', 'total_acc_y_train.txt', 'total_acc_z_train.txt'
    ]

    INPUT_FILES_TEST = [
        'body_acc_x_test.txt', 'body_acc_y_test.txt', 'body_acc_z_test.txt', 
        'body_gyro_x_test.txt', 'body_gyro_y_test.txt', 'body_gyro_z_test.txt',
        'total_acc_x_test.txt', 'total_acc_y_test.txt', 'total_acc_z_test.txt'
    ]
    
    train_signals, test_signals = [], []

    for input_file in INPUT_FILES_TRAIN:
        signal = read_signals(input_folder_train + input_file)
        train_signals.append(signal)
    train_signals = np.transpose(np.array(train_signals), (1, 2, 0))

    for input_file in INPUT_FILES_TEST:
        signal = read_signals(input_folder_test + input_file)
        test_signals.append(signal)
    test_signals = np.transpose(np.array(test_signals), (1, 2, 0))

    train_labels = read_labels(labelfile_train)
    test_labels = read_labels(labelfile_test)
    
    return train_signals, test_signals, train_labels, test_labels


def plot_confusion(Y_test, Y_test_pred, labels):
    cm = confusion_matrix(Y_test, Y_test_pred)
    df_cm = pd.DataFrame(
        cm, 
        index=[i for i in labels], 
        columns=[i for i in labels])
    plt.figure(figsize=(6, 6))
    ax= sns.heatmap(df_cm,  cbar=False, cmap="BuGn", annot=True, fmt="d")
    plt.setp(ax.get_xticklabels(), rotation=45)

    plt.ylabel('True label', fontweight='bold', fontsize = 14)
    plt.xlabel('Predicted label', fontweight='bold', fontsize = 14)
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom + 0.5, top - 0.5)
    plt.show()