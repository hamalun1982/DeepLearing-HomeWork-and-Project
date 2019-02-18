#import libaries
import numpy as np
import tensorflow as tf
import pandas as pd
from random import randint
from model import train_function
from util import split_data

# load data
# images=np.load('/work/cse496dl/shared/homework/01/fmnist_train_data.npy')
# labels=np.load('/work/cse496dl/shared/homework/01/fmnist_train_labels.npy')

# images=np.load('/work/cse496dl/shared/homework/01/fmnist_train_data.npy')
# labels=np.load('/work/cse496dl/shared/homework/01/fmnist_train_labels.npy')


images=np.load('/work/cse496dl/shared/homework/02/cifar_images.npy')
labels=np.load('/work/cse496dl/shared/homework/02/cifar_labels.npy')


#one hot encode labels
labels_oh = np.zeros((labels.astype(int).size, labels.astype(int).max()+1))
labels_oh[np.arange(labels.size),labels.astype(int)] = 1

# split into train and test
train_images, val_images, test_images = split_data(images, 0.7, 0.1, .2, 123)
train_labels, val_labels, test_labels = split_data(labels_oh, 0.7, 0.1, .2, 123)



#variables specification
c=0
hiddenlayers=[5]
batchsize=[128]
learningrate=[0.001]
regularization=[tf.contrib.layers.l2_regularizer(scale=0.01)]
filters = [16, 32, 64] 
# results=pd.DataFrame()
for h in hiddenlayers:
    for b in batchsize:
        for l in learningrate:
            for r in regularization:
                train_accuracy, validation_accuracy = train_function(train_images,train_labels,val_images,val_labels,test_images, test_labels, r, l, b, h)
#                 train_function(train_images,train_labels,val_images,val_labels,test_images, test_labels, r, l, b, h)

                #add hyperparameters and results to dataframe
                results.loc[c,'Hidden_layers']=h
                results.loc[c,'Regularization']=r
                results.loc[c,'Learning_rate']=l
                results.loc[c,'Batch_size']=b
                results.loc[c,'Train_accuracy']=train_accuracy
                results.loc[c,'Validation_accuracy']=validation_accuracy
                c=c+1

                #write results to csv -done in the loop so results are not lost if job fails or times out
                results.to_csv(filepath+'model_results.csv',index=False)
                print("Run",c,"Complete")
