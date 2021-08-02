# -*- coding: utf-8 -*-
"""
Created on Fri May  7 11:41:30 2021

@author: Win10
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os, cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input, Concatenate, Flatten, Reshape
from keras.callbacks import TensorBoard
from keras import backend as K
from z_train_s1 import load_model_gcn
from spektral.layers import GCNConv
from spektral.utils import gcn_filter

img_path = 'training/'  # path for saved images from Unity
model_path = 'training/model_gcn_s1.h5'
pixel_dim = 84
img_size = (pixel_dim, pixel_dim*4, 3) # h w c

def load_input(num_sensors, select_case, select_env=1, path='training/'):
    # load path for sensors
    all_sensors = []
    for kk in range(num_sensors):
        all_sensors.append('sensor_{}'.format(kk+1))
    # load img name
    filePath = path+'env_{}/'.format(select_env) +'sensor_1/1'
    filelist = os.listdir(filePath)
    filelist.sort(key = lambda x: int(x[:-4]))
    
    all_input = []
    image_index = []
    for i in select_case:
    # obs for one batch
        all_sensor_input = np.zeros((num_sensors, pixel_dim, pixel_dim*4, 3)) # h,w, rgb
        for idx_sensor in range(num_sensors):
            sensor_path = path + 'env_{}/'.format(select_env) + all_sensors[idx_sensor]
            img_1 = image.load_img(sensor_path+'/1/'+filelist[i], target_size=(pixel_dim,pixel_dim))  #height-width
            img_array_1 = image.img_to_array(img_1)
            img_2 = image.load_img(sensor_path+'/2/'+filelist[i], target_size=(pixel_dim,pixel_dim))  #height-width
            img_array_2 = image.img_to_array(img_2)
            img_3 = image.load_img(sensor_path+'/3/'+filelist[i], target_size=(pixel_dim,pixel_dim))  #height-width
            img_array_3 = image.img_to_array(img_3)
            img_4 = image.load_img(sensor_path+'/4/'+filelist[i], target_size=(pixel_dim,pixel_dim))  #height-width
            img_array_4 = image.img_to_array(img_4)  
            all_sensor_input[idx_sensor,:, pixel_dim*3:pixel_dim*4,:] = img_array_1/255 
            all_sensor_input[idx_sensor,:, pixel_dim*2:pixel_dim*3,:] = img_array_2/255
            all_sensor_input[idx_sensor,:, pixel_dim*1:pixel_dim*2,:] = img_array_3/255
            all_sensor_input[idx_sensor,:, pixel_dim*0:pixel_dim*1,:] = img_array_4/255    
        all_input.append(all_sensor_input.copy())
        image_index.append(int(filelist[i][:-4]))
    return np.array(all_input), image_index
      
def test_model(num_sensors, batch_size=1, lr=3e-4, is_robot=0):
    z_input = load_input(num_sensors, batch_size)
    z_model = load_model_gcn(num_sensors) 
    z_model.load_weights(model_path)
    z_model.compile(optimizer=Adam(learning_rate=lr, clipnorm=1), loss='mse')    
    
    if is_robot:
        z_admatrix = np.ones((batch_size, num_sensors+1, num_sensors+1))
    else:
        z_admatrix = np.ones((batch_size, num_sensors, num_sensors))
    
    input_admatrix = gcn_filter(z_admatrix)
    input_data = []
    
    for i in range(num_sensors):
        input_data.append(np.expand_dims(z_input[0][i],axis=0))   
    input_data.append(input_admatrix)
    
    z_res = z_model.predict(input_data)
    return z_res
    
    