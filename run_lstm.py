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
from DeepForecastLSTM import DeepForecastLSTM


GRID_PATH = './Monthly PPT(1901-2015)/PPT'
ENSO_PATH = './ENSO_PDO_AMO.csv'
grid_points = os.listdir(GRID_PATH)
lsci = pd.read_csv(ENSO_PATH)

def get_grid_id(s):
  return int(s[7:-4])

def load_grid_data(GRID_PATH,grid_points):
  grid_data = np.zeros((4965,1380))
  for i in range(len(grid_points)):
    if(grid_points[i][-3:]=='csv'):
      eg = pd.read_csv(os.path.join(GRID_PATH,grid_points[i]))
      pr_dat = eg.iloc[:,3].to_numpy()
      grid_data[get_grid_id(grid_points[i]),:] = pr_dat[:]
  return grid_data

india_data = pd.read_csv('./Monthly PPT(1901-2015)/ID_LAT_LON(4964)/ID_4964.csv')
grid_data = load_grid_data(GRID_PATH,grid_points)

def run(clip,epochs,horizon):
    Model1 = DeepForecastLSTM(len(india_data),horizon,"relu",india_data)
    x_train,x_test,y_train,y_test = Model1.loadData(india_data,grid_data,lsci,0.01,clip)
    lstmModel = Model1.buildModel()
    early_stopping = keras.callbacks.EarlyStopping(monitor="val_loss", patience=6)
    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=3)
    batch_size = 8

    hist = lstmModel.fit(
        x_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_split=0.1,
        callbacks=[early_stopping, reduce_lr],
    )
    filename = "lstm_" + str(clip) + '_' + str(epochs) + '_' + str(horizon) + '.h5'
    lstmModel.save(filename)
    os.system(f"mv ./{filename} ./saved_models")

    