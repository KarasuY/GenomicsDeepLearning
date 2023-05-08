# -*- coding: utf-8 -*-
"""
Created on Mon May  8 11:55:24 2023

@author: Yasmin K
"""

'''
Aim:
1) Train an Autoencoder to learn the distribution of RNAseq counts

2) Analyze the learned encodings using PCA and t-SNE

'''
import os
import sys
import numpy as np
import argparse

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

# import torch
# from torch import nn
# from torch.utils.data import DataLoader

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Conv1D, Conv1DTranspose, Flatten, Dropout, MaxPooling1D, Reshape, BatchNormalization


parser = argparse.ArgumentParser()
parser.add_argument('-epochs', type=int, default=10)
parser.add_argument('-batch_size', type=int, default=128)
parser.add_argument('-lr', type=float, default=1e-4)
parser.add_argument('-latent_size', type=int, default=10)
args = parser.parse_args()

num_epochs = args.epochs
batch_size = args.batch_size
lr = args.lr
latent_size = args.latent_size

'''
Input: NxM np.array of counts data
Output: NxM np.array filled with 1 corresponding to non-zero inputs
        and 0 corresponding to dropped inputs
'''
def return_out_mask(batch_data: np.array) -> np.array:

    zero_indices = np.nonzero(batch_data == 0)
    mask = np.ones_like(batch_data)
    zeros = np.zeros_like(batch_data)
    mask[zero_indices] = zeros[zero_indices] # Fills in 0's into appropriate indices
    return mask


# Load Data
dataset = np.load('data/counts.npy')
labels = np.loadtxt("data/labels.txt")

# Define Model, Optimizer, and Loss
class AE(Model):
  def __init__(self):
    super(AE, self).__init__()
    self.encoder = Sequential([
      Dense(latent_size),
      Dropout(0.2, name = 'dropout1')])

    self.decoder = Sequential([
      Dense(1000),
      Dropout(0.2, name = 'dropout3')])

  def call(self, x):
    encoded_data = self.encoder(x)
    decoded_data = self.decoder(encoded_data)
    mask = return_out_mask(x)
    mask_decode = tf.math.multiply(decoded_data,tf.convert_to_tensor(mask))
    return mask_decode

opt = tf.keras.optimizers.Adam(learning_rate=lr, name='Adam')

# Training Loop
autoencoder = AE()
autoencoder.compile(optimizer=opt, loss='mse', metrics=['mse'], run_eagerly=True)
history = autoencoder.fit(dataset, dataset, epochs=200, batch_size=batch_size, verbose=1 )


# T-SNE & PCA Plot of Counts Data
tsne_data = TSNE(n_components=2).fit_transform(dataset)
plt.scatter(tsne_data[:,0],tsne_data[:,1],c=labels,s=3)
plt.savefig('data_tsne_original.png')
plt.close()

pca = PCA(n_components=2)
pca_data = pca.fit_transform(dataset)
plt.scatter(pca_data[:,0],pca_data[:,1],c=labels,s=3)
plt.savefig('data_pca_original.png')
plt.close() # Always remember to close the plot!

# run encoder without decoder
# T-SNE & PCA Plot of Encodings
encoded_data = autoencoder.encoder(dataset) # np.array

tsne_latent = TSNE(n_components=2).fit_transform(encoded_data)
plt.scatter(tsne_latent[:,0],tsne_latent[:,1],c=labels,s=3)
plt.savefig('encoded_tsne_' + str(latent_size)+'.png')
plt.close()

pca = PCA(n_components=2)
pca_data = pca.fit_transform(encoded_data)
plt.scatter(pca_data[:,0],pca_data[:,1],c=labels,s=3)
plt.savefig('encoded_pca_' + str(latent_size)+'.png')
plt.close()

# T-SNE & PCA Plot of Reconstructions
encode = autoencoder.encoder(dataset)
reconstructions = autoencoder.decoder(encode)

tsne_latent = TSNE(n_components=2).fit_transform(reconstructions)
plt.scatter(tsne_latent[:,0],tsne_latent[:,1],c=labels,s=3)
plt.savefig('reconstructed_tsne_' + str(latent_size)+ '.png')
plt.close()

pca = PCA(n_components=2)
pca_data = pca.fit_transform(reconstructions)
plt.scatter(pca_data[:,0],pca_data[:,1],c=labels,s=3)
plt.savefig('reconstructed_pca_' + str(latent_size) + '.png')
plt.close()
