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

class DeepForecastLSTM:
  def __init__(self,inOutVecDim,inputHorizon,activation,dat):
    self.inOutVecDim = inOutVecDim #4964
    self.inputHorizon = inputHorizon #12
    self.activation = activation #"relu"
    self.id_to_loc = {}
    for i in range(len(dat)):
      self.id_to_loc[dat.iloc[i,0]] = i
  def Normalize(self,dat,clip):
    dat = np.clip(dat,0,clip)
    self.mx = dat.max()
    self.mn = dat.min()
    dat = (dat - dat.min())/dat.max()
    return dat

  def deNormalize(self,dat,mx,mn):
    dat = dat*self.mx+self.mn
    return dat

  def loadData(self,dat,grid_data,lsci,split,clip):
    data = []
    for i in range(len(dat)):
      data.append(grid_data[dat.iloc[i,0]])
    data.append(lsci.iloc[:,1])
    data.append(lsci.iloc[:,2])
    data.append(lsci.iloc[:,3])
    data = np.array(data)
    data[:-3] = self.Normalize(data[:-3],clip)
    data = data.T
    result = []
    for index in range(len(data) - self.inputHorizon):
        result.append(data[index:index + self.inputHorizon])
    result = np.array(result)
    x = result[:,:]
    y = data[self.inputHorizon:,:self.inOutVecDim]
    print(x.shape,y.shape)
    return tts(x, y, test_size=split, random_state=42,shuffle=False)

  def buildModel(self):
    model = Sequential()
    in_nodes = out_nodes = self.inOutVecDim
    layers = [in_nodes+3, 1024, 512, 1024 , out_nodes]
    model.add(LSTM(input_shape=(self.inputHorizon,layers[0]),units=layers[1],return_sequences=False))
    model.add(Dense(layers[4]))
    model.add(Activation(self.activation))
    optimizer = 'adam'
    model.compile(loss="mae", optimizer=optimizer, metrics = ["mae","mse"])
    return model

  def plot(self,id,model,x_test,y_test):
      plt.figure(figsize=(7,5))
      plt.plot(self.deNormalize(y_test[:,self.id_to_loc[id]],self.mx,self.mn))  
      plt.plot(self.deNormalize(model.predict(x_test)[:,self.id_to_loc[id]],self.mx,self.mn),c='red')
      plt.pause(0.001)
