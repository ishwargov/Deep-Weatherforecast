import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import sys
import csv
import tensorflow as tf
import os
import keras
import json
import requests
import seaborn as sns
from keras.models import Sequential
from keras import layers
from keras.layers.core import Dense, Activation, Dropout, Flatten
from keras.layers import Convolution2D
from keras.layers.recurrent import LSTM, SimpleRNN, GRU
from keras.utils.vis_utils import plot_model
from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import mean_absolute_error
from pandas.plotting import lag_plot
from pandas.plotting import autocorrelation_plot
from sklearn.metrics import mean_squared_error
np.random.seed(42)

class ConvLSTM2:
  def __init__(self,dat):
    self.id_to_latlon = {}
    self.latlon_to_id = {}
    self.xmx,self.xmn,self.ymx,self.ymn = dat.iloc[:,1].max()*4,dat.iloc[:,1].min()*4,dat.iloc[:,2].max()*4,dat.iloc[:,2].min()*4
    for i in range(len(dat)):
      x,y = dat.iloc[i,1]*4,dat.iloc[i,2]*4
      self.id_to_latlon[dat.iloc[i,0]] = [int(x-self.xmn),int(y-self.ymn)]
      self.latlon_to_id[(int(x-self.xmn),int(y-self.ymn))] = dat.iloc[i,0]
    self.mask = np.zeros((1,int(self.xmx-self.xmn)+1,int(self.ymx-self.ymn)+1))
    for a,b in self.id_to_latlon.items():
      self.mask[0,b[0],b[1]] = 1
    self.mask = tf.constant(self.mask,dtype=tf.float32)
    self.mae = tf.keras.losses.MeanAbsoluteError()
    self.mse = tf.keras.losses.MeanSquaredError()


  def make_dataset(self,grid,inp_horizon):
    sz = grid.shape[3]
    x = np.zeros((sz-inp_horizon,inp_horizon,grid.shape[0],grid.shape[1],grid.shape[2]))
    y = np.zeros((sz-inp_horizon,grid.shape[0],grid.shape[1]))
    for i in range(sz-inp_horizon):
      x[i,:,:,:,:] = np.rollaxis(grid[:,:,:,i:i+inp_horizon],3,0)
      y[i,:,:] = grid[:,:,0,i+inp_horizon]
    return(x,y)

  def buildModel(self,input_shape):
    inp = layers.Input(shape=input_shape)
    x = layers.ConvLSTM2D(
        filters=32,
        kernel_size=(5, 5),
        padding="same",
        return_sequences=True,
        activation="relu",
    )(inp)
    x = layers.BatchNormalization()(x)
    x = layers.ConvLSTM2D(
      filters=32,
      kernel_size=(3, 3),
      padding="same",
      return_sequences=True,
      activation="relu",
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.ConvLSTM2D(
        filters=1,
        kernel_size=(1, 1),
        padding="same",
        return_sequences=False,
        activation="relu",
    )(x)
    model = keras.models.Model(inp, x)
    model.compile(
        loss=self.grid_loss, optimizer='adam', metrics = [self.grid_loss,"mse"]
    )
    return model