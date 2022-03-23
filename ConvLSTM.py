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

class ConvLSTM:
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

  def make_2d(self,dat,grid_data,lsci,choice):
    x = dat.iloc[:,1].to_numpy()*4
    y = dat.iloc[:,2].to_numpy()*4
    x = x-min(x)
    x = x.astype(int)
    y = y-min(y)
    y = y.astype(int)
    grid = np.zeros((max(x)+1,max(y)+1,bin(choice).count('1')+1,grid_data.shape[1]))
    for i in range(len(dat)):
      grid[x[i],y[i],0,:] = grid_data[int(dat.iloc[i,0]),:]
      if(choice&1<<3):
        grid[x[i],y[i],bin(choice).count('1'),:] = np.full((1,grid_data.shape[1]),dat.iloc[i,3])
    for i in range(grid_data.shape[1]):
      cnt = 1
      if(choice&1<<0):
        grid[:,:,cnt,i] = np.full((max(x)+1,max(y)+1),lsci.iloc[i,1])
        cnt+=1
      if(choice&1<<1):
        grid[:,:,cnt,i] = np.full((max(x)+1,max(y)+1),lsci.iloc[i,2])
        cnt+=1
      if(choice&1<<2):
        grid[:,:,cnt,i] = np.full((max(x)+1,max(y)+1),lsci.iloc[i,3])
        cnt+=1
    return grid

  def make_dataset(self,grid,inp_horizon):
    sz = grid.shape[3]
    x = np.zeros((sz-inp_horizon,inp_horizon,grid.shape[0],grid.shape[1],grid.shape[2]))
    y = np.zeros((sz-inp_horizon,grid.shape[0],grid.shape[1]))
    for i in range(sz-inp_horizon):
      x[i,:,:,:,:] = np.rollaxis(grid[:,:,:,i:i+inp_horizon],3,0)
      y[i,:,:] = grid[:,:,0,i+inp_horizon]
    return(x,y)
  
  def make_generator(self,grid,inp_horizon,batch_size):
    sz = grid.shape[3]
    for i in range((sz-inp_horizon)//batch_size):
      x = []
      y = []
      for j in range(batch_size):
        x.append(np.rollaxis(grid[:,:,:,i:i+inp_horizon],3,0))
        y.append(grid[:,:,0,i+inp_horizon])
      yield(np.array(x),np.array(y))

  def Normalize(self,dat,clip):
    dat = np.clip(dat,0,clip)
    self.mx = dat.max()
    self.mn = dat.min()
    dat = (dat - dat.min())/dat.max()
    return dat

  def deNormalize(self,dat,mx,mn):
    dat = dat*self.mx+self.mn
    return dat
  
  def grid_loss(self,y_true,y_pred):
    return(self.mae(self.mask*y_true,self.mask*y_pred))

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
  def plot(self,id,model,x_test,y_test):
    p = self.id_to_latlon[id]
    plt.figure(figsize=(7,5))
    plt.plot(self.deNormalize(y_test[:,p[0],p[1]],self.mx,self.mn))
    plt.plot(self.deNormalize(model.predict(x_test)[:,p[0],p[1]],self.mx,self.mn),c='r')
  def evaluate(self,model,x_test,y_test):
    return(self.grid_loss(self.deNormalize(y_test,self.mx,self.mn),self.deNormalize(model.predict(x_test)[:,:,:,0],self.mx,self.mn)))
  def norm_evaluate(self,model,x_test,y_test):
    return(self.grid_loss(y_test,model.predict(x_test)[:,:,:,0]))