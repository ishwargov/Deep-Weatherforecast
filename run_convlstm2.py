from distutils.command.build_clib import build_clib
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
from ConvLSTM2 import ConvLSTM2
np.random.seed(42)


GRID_PATH = './Monthly PPT(1901-2015)/PPT'
ENSO_PATH = './ENSO_PDO_AMO.csv'
grid_points = os.listdir(GRID_PATH)
lsci = pd.read_csv(ENSO_PATH)


def get_grid_id(s):
    return int(s[7:-4])

# storing all grid points


def load_grid_data(GRID_PATH, grid_points):
    grid_data = np.zeros((4965, 1380))
    for i in range(len(grid_points)):
        if(grid_points[i][-3:] == 'csv'):
            eg = pd.read_csv(os.path.join(GRID_PATH, grid_points[i]))
            pr_dat = eg.iloc[:, 3].to_numpy()
            grid_data[get_grid_id(grid_points[i]), :] = pr_dat[:]
    return grid_data

# get altitude of each latitude longitude using Openelevation API (not accurate /for testing)


def getElevation(lat, lon):
    url = "https://api.open-elevation.com/api/v1/lookup?locations=" + \
        str(lat)+","+str(lon)
    payload = {}
    headers = {}
    response = requests.request("GET", url, headers=headers, data=payload)
    a = json.loads(response.text)
    return(float(a["results"][0]['elevation']))


def get_bin(x, n): return format(x, 'b').zfill(n)


grid_data = load_grid_data(GRID_PATH, grid_points)
india_data = pd.read_csv(
    './Monthly PPT(1901-2015)/ID_LAT_LON(4964)/ID_4964.csv')

indiamodels = []
combinations = []

dat = pd.read_csv('./Monthly PPT(1901-2015)/ID_LAT_LON(4964)/ID_4964.csv')
id_to_latlon = {}
latlon_to_id = {}
xmx, xmn, ymx, ymn = dat.iloc[:, 1].max(
)*4, dat.iloc[:, 1].min()*4, dat.iloc[:, 2].max()*4, dat.iloc[:, 2].min()*4


def get_mae(y_pred, y_test, id):
    return mean_absolute_error(y_pred[:, id_to_latlon[id][0], id_to_latlon[id][1]], y_test[:, id_to_latlon[id][0], id_to_latlon[id][1]])


def make_2d(dat, grid_data, lsci, choice):
    x = dat.iloc[:, 1].to_numpy()*4
    y = dat.iloc[:, 2].to_numpy()*4
    x = x-min(x)
    x = x.astype(int)
    y = y-min(y)
    y = y.astype(int)
    grid = np.zeros(
        (max(x)+1, max(y)+1, bin(choice).count('1')+1, grid_data.shape[1]))
    for i in range(len(dat)):
        grid[x[i], y[i], 0, :] = grid_data[int(dat.iloc[i, 0]), :]
    return grid


grid = make_2d(india_data, grid_data, lsci, 0)

grid_pad = np.pad(grid, ((17, 17), (17, 17), (0, 0), (0, 0)),
                  mode='constant', constant_values=((0, 0), (0, 0), (0, 0), (0, 0)))

blocks = []
cnt = 0
block_to_latlon = {}
latlon_to_block = {}

for i in range(0, grid_pad.shape[0], 10):
    for j in range(0, grid_pad.shape[1], 10):
        cur_block = grid_pad[i:i+10, j:j+10, :, :]
        if(np.sum(cur_block) == 0):
            continue
        blocks.append(cur_block)
        block_to_latlon[cnt] = (i, j)
        latlon_to_block[(i, j)] = cnt
        cnt += 1


def make_data(blocks, grid_pad, block_num, horizon):
    x = []
    y = []
    block = blocks[block_num]
    p, q = block_to_latlon[block_num]
    for i in range(block.shape[3]-horizon):
        x.append(np.rollaxis(
            grid[(p-9):(p+19), (q-9):(q+19), :, i:i+12], 3, 0))
        y.append(block[:, :, :, i+12])
    return(np.array(x), np.array(y))


def train_ondat(model, epochs, x_train, y_train):
    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=6)
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", patience=3)
    epochs = 20
    batch_size = 5
    filename = f'convmodel_new_{block}.h5'
    hist = model.fit(
        x_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_split=0.3,
        callbacks=[early_stopping, reduce_lr],
    )
    return model


def build_model():
    input_shape = (12, 28, 28, 1)
    inp = layers.Input(shape=input_shape)
    x = layers.ConvLSTM2D(
        filters=32,
        kernel_size=(5, 5),
        padding="same",
        return_sequences=True
    )(inp)
    x = layers.BatchNormalization()(x)
    x = layers.ConvLSTM2D(
        filters=32,
        kernel_size=(3, 3),
        padding="same",
        return_sequences=True
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.ConvLSTM2D(
        filters=1,
        kernel_size=(1, 1),
        padding="same",
        return_sequences=False
    )(x)
    x = layers.Conv2D(
        filters=256,
        kernel_size=(3, 3),
        padding='same'
    )(x)
    x = layers.Conv2D(
        filters=128,
        kernel_size=(3, 3),
        padding='same'
    )(x)
    x = layers.Conv2D(
        filters=64,
        kernel_size=(3, 3),
        strides=(2, 2),
        padding='same'
    )(x)
    x = layers.Conv2D(
        filters=32,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding='valid'
    )(x)
    x = layers.Conv2D(
        filters=1,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding='valid',
        activation='relu'
    )(x)
    model = keras.models.Model(inp, x)
    return model


def run(epoch = 20):
    os.system(f'mkdir ./saved_models')
    os.system(f'mkdir ./model_output')
    f = open(f'model_output.txt', 'a')
    f.write(f'{time.strftime("%d/%m/%Y %H:%M:%S")}\n')
    for i in range(len(blocks)):
        model = build_model()
        model.compile(
            loss="mae", optimizer='adam', metrics=["mae", "mse"]
        )

        x, y = make_data(blocks, grid_pad, i, 12)
        x_train, x_test, y_train, y_test = tts(
            x, y, test_size=0.3, shuffle=False)

        early_stopping = keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=6)
        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", patience=3)
        batch_size = 5
        filename = 'convmodel_new'+str(i)+'.h5'
        hist = model.fit(
            x_train,
            y_train,
            batch_size=batch_size,
            epochs=epoch,
            validation_split=0.3,
            callbacks=[early_stopping, reduce_lr],
        )
        model.save(filename)
        f.write(f"Model{i} error = {model.evaluate(x_test,y_test)} \n")
        os.system(f"mv ./{filename} ./saved_models")
    f.close()
    os.system(f"mv ./model_output.txt ./model_output")
