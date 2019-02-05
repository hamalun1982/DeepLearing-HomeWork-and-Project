import numpy as np
import tensorflow as tf
import pandas as pd
from random import randint

# load data
images=np.load('/work/cse496dl/shared/homework/01/fmnist_train_data.npy')
labels=np.load('/work/cse496dl/shared/homework/01/fmnist_train_labels.npy')

#one hot encode labels
labels_oh = np.zeros((labels.astype(int).size, labels.astype(int).max()+1))
labels_oh[np.arange(labels.size),labels.astype(int)] = 1

# split into 70% training, 10% validation and 20% test, with random seed 123
train_images, val_images, test_images = split_data(images, 0.7, 0.1, .2, 123)
train_labels, val_labels, test_labels = split_data(labels_oh, 0.7, 0.1, .2, 123)

architecture_search(train_images,train_labels,val_images,val_labels,'$WORK',3,10,784)
