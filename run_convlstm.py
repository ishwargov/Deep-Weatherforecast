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
from ConvLSTM import ConvLSTM


GRID_PATH = './Monthly PPT(1901-2015)/PPT'
ENSO_PATH = './ENSO_PDO_AMO.csv'
grid_points = os.listdir(GRID_PATH)
lsci = pd.read_csv(ENSO_PATH)


def get_grid_id(s):
  return int(s[7:-4])

#storing all grid points
def load_grid_data(GRID_PATH,grid_points):
  grid_data = np.zeros((4965,1380))
  for i in range(len(grid_points)):
    if(grid_points[i][-3:]=='csv'):
      eg = pd.read_csv(os.path.join(GRID_PATH,grid_points[i]))
      pr_dat = eg.iloc[:,3].to_numpy()
      grid_data[get_grid_id(grid_points[i]),:] = pr_dat[:]
  return grid_data

#get altitude of each latitude longitude using Openelevation API (not accurate /for testing)
def getElevation(lat,lon):
  url = "https://api.open-elevation.com/api/v1/lookup?locations="+str(lat)+","+str(lon)
  payload={}
  headers = {}
  response = requests.request("GET", url, headers=headers, data=payload)
  a = json.loads(response.text)
  return(float(a["results"][0]['elevation']))

get_bin = lambda x, n: format(x, 'b').zfill(n)
grid_data = load_grid_data(GRID_PATH,grid_points)
india_data = pd.read_csv('./Monthly PPT(1901-2015)/ID_LAT_LON(4964)/ID_4964.csv')

indiamodels = []
combinations = []

dat = pd.read_csv('./Monthly PPT(1901-2015)/ID_LAT_LON(4964)/ID_4964.csv')
id_to_latlon = {}
latlon_to_id = {}
xmx,xmn,ymx,ymn = dat.iloc[:,1].max()*4,dat.iloc[:,1].min()*4,dat.iloc[:,2].max()*4,dat.iloc[:,2].min()*4

def get_mae(y_pred,y_test,id):
  return mean_absolute_error(y_pred[:,id_to_latlon[id][0],id_to_latlon[id][1]],y_test[:,id_to_latlon[id][0],id_to_latlon[id][1]])

def heatmap(model,x_test,y_test):
  y_pred = model.predict(x_test)
  for i in range(len(dat)):
    x,y = dat.iloc[i,1]*4,dat.iloc[i,2]*4
    id_to_latlon[dat.iloc[i,0]] = [int(x-xmn),int(y-ymn)]
    latlon_to_id[(int(x-xmn),int(y-ymn))] = dat.iloc[i,0]
  hmap = np.zeros((int(xmx-xmn)+1,int(ymx-ymn)+1))
  for i in range(1,4965):
    hmap[id_to_latlon[i][0],id_to_latlon[i][1]] = get_mae(y_pred,y_test,i)  
  plt.figure(figsize=(15,15))
  ax = sns.heatmap(np.rot90((hmap.T), k=1))
  plt.savefig('convlstm_heatmap.png') 

def run(clip_value,epochs,combs,horizon):
  for i in range(combs,8):
    Model2 = ConvLSTM(india_data)
    choice = i
    print(bin(choice))
    grid = Model2.make_2d(india_data,grid_data,lsci,choice)
    if(choice&(1<<3)):
      grid[:,:,bin(choice).count('1'),:] = Model2.Normalize(grid[:,:,bin(choice).count('1'),:],1e9)
    grid[:,:,0,:] = Model2.Normalize(grid[:,:,0,:],clip_value)
    #print(grid.shape)
    x,y = Model2.make_dataset(grid,horizon)
    x_train,x_test,y_train,y_test = tts(x,y,test_size=0.1,shuffle=False)
    print(x_train.shape,x_test.shape,y_train.shape,y_test.shape)
    convlstmModel = Model2.buildModel(x_train.shape[1:])

    early_stopping = keras.callbacks.EarlyStopping(monitor="val_loss", patience=6)
    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=3)
    batch_size = 1
    filename = 'convlstm_'+str(i)+"_"+str(clip_value)+"_"+str(epochs)+'_'+str(horizon)+'.h5'
    if(not os.path.isfile("./saved_models/"+filename)):
      hist = convlstmModel.fit(
          x_train,
          y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_split=0.1,
          callbacks=[early_stopping, reduce_lr],
      )
    else:
      convlstmModel.load_weights("./saved_models/"+filename)
    print(Model2.evaluate(convlstmModel,x_test,y_test))
    indiamodels.append(convlstmModel)
    combinations.append((get_bin(choice,4),Model2.evaluate(convlstmModel,x_test,y_test).numpy(),Model2.norm_evaluate(convlstmModel,x_test,y_test).numpy()))
    convlstmModel.save(filename)
    os.system(f"mv ./{filename} ./saved_models")
    heatmap(convlstmModel,x_test,y_test)
    #files.download('convmodel'+str(i)+'.h5')

  # print(combinations)
  # os.system(f"rm -r {filename}_stats.txt")
  # with open(filename+'_stats.txt','x') as f:
  #   f.write(str(combinations))
  # os.system(f"mv ./{filename}_stats.txt ./stats")



