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
from resnet import sensor_cnn
from spektral.layers import GCNConv
from spektral.utils import gcn_filter

#from gnn_pathplanning.utils.graphUtils import graphML as gml

def load_input(num_sensors, batch_size=1, path='training/'):
    # load path for sensors
    all_sensors = []
    for kk in range(num_sensors):
        all_sensors.append('sensor_{}'.format(kk+1))
    # load img name
    filePath = 'training/sensor_1/1'
    filelist = os.listdir(filePath)
    filelist.sort(key = lambda x: int(x[:-4]))
    
    all_input = []
    for i in range(batch_size):
    # obs for one batch
        all_sensor_input = np.zeros((num_sensors, 84, 84*4, 3)) # h,w, rgb
        for idx_sensor in range(num_sensors):
            sensor_path = path + all_sensors[idx_sensor]
            img_1 = image.load_img(sensor_path+'/1/'+filelist[i], target_size=(84,84))  #height-width
            img_array_1 = image.img_to_array(img_1)
            img_2 = image.load_img(sensor_path+'/2/'+filelist[i], target_size=(84,84))  #height-width
            img_array_2 = image.img_to_array(img_2)
            img_3 = image.load_img(sensor_path+'/3/'+filelist[i], target_size=(84,84))  #height-width
            img_array_3 = image.img_to_array(img_3)
            img_4 = image.load_img(sensor_path+'/4/'+filelist[i], target_size=(84,84))  #height-width
            img_array_4 = image.img_to_array(img_4)  
            all_sensor_input[idx_sensor,:, 84*3:84*4,:] = img_array_1/255 
            all_sensor_input[idx_sensor,:, 84*2:84*3,:] = img_array_2/255
            all_sensor_input[idx_sensor,:, 84*1:84*2,:] = img_array_3/255
            all_sensor_input[idx_sensor,:, 84*0:84*1,:] = img_array_4/255    
        all_input.append(all_sensor_input.copy())
    return np.array(all_input)

def mlp_model(gnn_unit=128):
    input_data = Input(shape=gnn_unit)
    output1 = Dense(32, activation='relu',  name='mlp_1')(input_data)
    output1 = Dense(2, activation='linear', name='sensors')(output1)  
    model = Model(inputs=[input_data], outputs=[output1])
    return model

def load_model(num_sensors, input_shape=(84,84*4,3), gnn_layers=1, gnn_unit=128, is_robot=0):
    
    input_data, output_data = [], []   
    #tf.compat.v1.enable_eager_execution()
    s_cnn = sensor_cnn(input_shape, repetitions = [2,2,2,2])
 
    if is_robot:
        sensor_matrix = Input(shape=(num_sensors+1, num_sensors+1))
    else:
        sensor_matrix = Input(shape=(num_sensors, num_sensors))
        
    for i in range(num_sensors):
        exec('s_input{} = Input(shape=(input_shape[0], input_shape[1], input_shape[2]))'.format(i))
        exec('extract_cnn{} = s_cnn(s_input{})'.format(i, i))
        exec('input_data.append(s_input{})'.format(i))
    input_data.append(sensor_matrix)
    
    exec('extract_cnn = extract_cnn0')
    for i in range(1,num_sensors):
        exec('extract_cnn = Concatenate(axis=1)([extract_cnn, extract_cnn{}])'.format(i))
    
    for j in range(1, gnn_layers+1):
        if j == 1:
            exec("G_h{} = GCNConv(gnn_unit, activation='relu')([extract_cnn, sensor_matrix])".format(j))
        else:
            exec("G_h{} = GCNConv(gnn_unit, activation='relu')([G_h{}, sensor_matrix])".format(j, j-1))

    exec('gnn_output = tf.split(G_h{}, num_sensors, 1)'.format(gnn_layers))
    
    mlp_layer = mlp_model()
    for i in range(num_sensors):
        exec('output{} = mlp_layer(Flatten()(gnn_output[i]))'.format(i))
        exec('output_data.append(output{})'.format(i))
        
    model = Model(inputs=input_data, 
                  outputs= output_data)
    return model    
    
def test_model(num_sensors, batch_size=1, lr=3e-4, is_robot=0):
    z_input = load_input(num_sensors, batch_size)
    z_model = load_model(num_sensors)
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
    
    









