# -*- coding: utf-8 -*-
"""
Created on Mon May  8 11:40:07 2023

@author: Yasmin K
"""

"""
Background: Hi-C data can be used to identify the 3D conformation of DNA in cells
by highlighting which regions of DNA are located close together. The 3D conformation
of DNA can be determined more accurately when higher resolution Hi-C data is available.

Data: Dataset was used by Zhang et al. for the development of HiCPlus.

Aim:

1. Implement a convolutional neural network (CNN) to transform patches of
   low-resolution Hi-C data into patches of high-resolution Hi-C data.


2. For several matrix patches in the training set, visualize and compare the
   training input, training label, and predicted label.


3. For several matrix patches in the test set, visualize and compare the
   training input and predicted label.

"""

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool2D
from sklearn.model_selection import KFold
import seaborn
import tensorflow_probability as tfp

'''
Load Dataset
'''
def get_data():
    train_x = np.load("data/GM12878_replicate_down16_chr19_22.npy").astype("float32")
    train_y = np.load("data/GM12878_replicate_original_chr19_22.npy").astype("float32")
    test_x = np.load("data/GM12878_replicate_down16_chr17_17.npy").astype("float32")
    train_x = np.reshape(train_x, (train_x.shape[0], train_x.shape[2], train_x.shape[3], 1))
    train_y = np.reshape(train_y, (train_y.shape[0], train_y.shape[2], train_y.shape[3], 1))
    test_x = np.reshape(test_x, (test_x.shape[0], test_x.shape[2], test_x.shape[3], 1))
    return train_x, train_y, test_x

"""
param train_y: a numpy array of training labels with dimensions (samples, 40, 40, 1)
param border_size: the size of the border to be removed
return: a numpy array of training labels with dimensions (samples, 40-(2*border_size), 40-(2*border_size), 1),
        where border_size is chosen based on the model architecture to ensure the training labels
        and model outputs have the same size
"""
def remove_borders(train_y, border_size):

    new_train = train_y[:,:40-2*border_size, :40-2*border_size,:]
    return new_train

"""
Implements and trains the model using a cross-validation scheme with MSE loss
param train_x: the training inputs
param train_y: the training labels
return: a trained model
"""
def train_model(train_x, train_y):
    kfold = KFold(n_splits=5, shuffle=True)

    opt = tf.keras.optimizers.Adam(learning_rate=0.005, name='Adam')
    best_loss = 10000
    for train, test in kfold.split(train_x, train_y):
        model = tf.keras.Sequential()
        model.add(Conv2D(8, kernel_size=(9, 9), activation='relu', padding='VALID'))
        model.add(Conv2D(8, kernel_size=(1, 1), activation='relu', padding='VALID'))
        model.add(Conv2D(1, kernel_size=(5, 5), activation='relu', padding='VALID'))

        model.compile(optimizer=opt, loss='mse')
        history = model.fit(train_x[train], train_y[train], epochs=100, batch_size=32, verbose=1)

        loss = model.evaluate(train_x[test], train_y[test])
        print("hi loss")
        print(loss)
        if loss< best_loss:
            best_model = model
            best_loss = loss
    return best_model, best_loss

def make_prediction(model, input_data):
    """
    param model: a trained model
    param input_data: model inputs
    return: the model's predictions for the provided input data
    """
    pred = model.predict(input_data)
    return pred

def main():
    train_x, train_y, test_x = get_data()
    border_size = 6
    new_train_y = remove_borders(train_y, border_size)
    trained_model, best_loss = train_model(train_x, new_train_y)
    print("best loss: {}".format(best_loss))
    prediction = make_prediction(trained_model, test_x)
    prediction_train = make_prediction(trained_model, train_x)

    loss = trained_model.evaluate(train_x, new_train_y)
    print("Entire model loss: {}".format(loss))
    pearson = tfp.stats.correlation(prediction_train, new_train_y)
    avg_pearson = tf.math.reduce_mean(pearson)
    print("Pearson Score: {}".format(avg_pearson))

    #plot of training for verification
    fig, (ax111, ax22) = plt.subplots(1,2)
    seaborn.heatmap(prediction_train[10,:,:,0], ax = ax22)
    seaborn.heatmap(new_train_y[10,:,:,0], ax = ax111)
    plt.show()

    #plot of testing

    fig, (ax1, ax2) = plt.subplots(1,2)
    seaborn.heatmap(prediction[10,:,:,0], ax = ax2)
    seaborn.heatmap(test_x[10,:,:,0], ax = ax1)
    plt.show()

    fig, (ax3, ax4) = plt.subplots(1,2)
    seaborn.heatmap(prediction[0,:,:,0], ax = ax4)
    seaborn.heatmap(test_x[0,:,:,0], ax = ax3)
    plt.show()

    fig, (ax9, ax10) = plt.subplots(1,2)
    seaborn.heatmap(prediction[60,:,:,0], ax = ax10)
    seaborn.heatmap(test_x[60,:,:,0], ax = ax9)
    plt.show()

if __name__ == '__main__':
    main()
