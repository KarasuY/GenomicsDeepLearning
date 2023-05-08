# -*- coding: utf-8 -*-
"""
Created on Mon May  8 11:48:09 2023

@author: Yasmin K
"""

'''
Dataset: synthetically generated for the task of "homotypic motif density localization".
Translation: Finding clusters of the same type of sequence motif


Background:
Transcription Factors (TFs) often bind to particular sequence patterns, known as motifs.
Regulatory regions of DNA often have more than one binding site/motif for a particular TF clustered in a small region.
The small dataset provided has sequences classified as either positive or negative depending on where there are clusters of a certain motif.

AIM:
1) Train a CNN to classify the sequences as positive or negative for homotypic motif clusters

2) Plot the training and validation loss after each epoch of training.

3) Implement a 2-layer fully connected neural network architecture (with non-linearity) to perform binary classification.

'''

import numpy as np

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Conv1D, Dense, MaxPool1D, Flatten, Dropout, LSTM

'''
Load Dataset
'''
def get_data(filename='hw1_data.npz'):
    all_data = np.load(filename)
    train_seq = all_data['train_seq']
    train_y = all_data['train_y']
    valid_seq = all_data['valid_seq']
    valid_y = all_data['valid_y']
    test_seq = all_data['test_seq']
    test_y = all_data['test_y']

    return train_seq, train_y, valid_seq, valid_y, test_seq, test_y

"""
param seq_array: np array of DNA sequences
return: np array of one-hot encodings of input DNA sequences
"""
def one_hot_encoding(seq_array: np.ndarray) -> np.ndarray:

    nuc2id = {'A' : 0, 'C' : 1, 'T' : 2, 'G' : 3}
    onehot_array = np.zeros((len(seq_array), 4, 1500))
    for seq_num, seq in enumerate(seq_array):
        for seq_idx, nucleotide in enumerate(seq):
            nuc_idx = nuc2id[nucleotide]
            onehot_array[seq_num, nuc_idx, seq_idx] = 1

    return onehot_array

"""
param seq_array: np array of DNA sequences
param k: length of k-mers
return: np array of k-mer counts per sequence
"""
def kmer_counts(seq_array: np.ndarray, k: int) -> np.ndarray:

    bagofkmer = {}
    count = 0
    kmer = np.zeros((len(seq_array),1500-k+1, 1))

    for seq_count, seq in enumerate(seq_array):
        for kmer_count in range(1500-k+1):
            a = seq[kmer_count:kmer_count+k]
            if a in bagofkmer:
                kmer[seq_count, kmer_count,0] = bagofkmer.get(a)
            else:
                kmer[seq_count, kmer_count,0] = count
                bagofkmer[a] = count
                count += 1
    return kmer

"""
Implements and trains a CNN with 1 convolution layer (including non-linearity and pooling)
followed by 1 dense output layer
param train_onehot_array: np array of one-hot encodings of input DNA sequences for training
param train_y: np array of training labels
param valid_onehot_array: np array of one-hot encodings of input DNA sequences for validation
param valid_y: np array of validation labels
return: the trained model
"""
def train_cnn(train_onehot_array, train_y, valid_onehot_array, valid_y):

    opt = tf.keras.optimizers.Adam(learning_rate=0.00015, name='Adam')
    model_cnn = tf.keras.Sequential()
    model_cnn.add(Conv1D(128, 3, activation='selu'))
    model_cnn.add(MaxPool1D(pool_size=4, strides=4, padding='valid'))
    model_cnn.add(Flatten())
    model_cnn.add(Dropout(0.2))
    model_cnn.add(Dense(1, activation='sigmoid'))
    model_cnn.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])

    history = model_cnn.fit(train_onehot_array, train_y, epochs=30, batch_size=32, verbose=1, validation_data = (valid_onehot_array, valid_y))

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('CNN Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validate'], loc='upper left')
    plt.show()

    return model_cnn

"""
param train_counts: np array of kmer counts in input DNA sequences for training
param train_y: np array of training labels
param valid_counts: np array of kmer counts in input DNA sequences for validation
param valid_y: np array of validation labels
return: a trained model using kmer count input data
"""
def train_kmer_counts_nn(train_counts, train_y, valid_counts, valid_y):

    opt2 = tf.keras.optimizers.Adam(learning_rate=0.00015, name='Adam')
    model_kmer = tf.keras.Sequential()

    model_kmer.add(Dense(64, activation='relu'))
    model_kmer.add(Dense(32, activation='relu'))
    model_kmer.add(Flatten())
    model_kmer.add(Dense(1, activation='sigmoid', name = "dense_last"))
    model_kmer.compile(optimizer=opt2, loss='binary_crossentropy', metrics=['accuracy'])

    history_kmer = model_kmer.fit(train_counts, train_y, epochs=50, batch_size=32, verbose=1, validation_data = (valid_counts, valid_y))

    plt.plot(history_kmer.history['loss'])
    plt.plot(history_kmer.history['val_loss'])
    plt.title('Kmer Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validate'], loc='upper left')
    plt.show()

    return model_kmer

"""
Prints the % accuracy of the model on the test data
param trained_model: a trained model
param test_inputs: np array of model inputs in the test set
param test_y: np array of test labels
"""
def evaluate_model(trained_model, test_inputs, test_y):

    loss, acc = trained_model.evaluate(test_inputs, test_y)
    print("Accuracy: {}".format(acc))

def main():

    train_seq, train_y, valid_seq, valid_y, test_seq, test_y = get_data()
    onehot_train_seq = np.transpose(one_hot_encoding(train_seq), axes = (0,2,1))
    onehot_valid_seq = np.transpose(one_hot_encoding(valid_seq), axes = (0,2,1))
    onehot_test_seq = np.transpose(one_hot_encoding(test_seq), axes = (0,2,1))
    trained_model_cnn = train_cnn(onehot_train_seq, train_y, onehot_valid_seq, valid_y)
    evaluate_model(trained_model_cnn, onehot_test_seq, test_y)
    k = 4
    kmer_train_seq = kmer_counts(train_seq,k)
    kmer_valid_seq = kmer_counts(valid_seq,k)
    kmer_test_seq = kmer_counts(test_seq,k)
    trained_model_kmer = train_kmer_counts_nn(kmer_train_seq, train_y, kmer_valid_seq, valid_y)
    evaluate_model(trained_model_kmer, kmer_test_seq, test_y)

if __name__ == '__main__':
    main()

