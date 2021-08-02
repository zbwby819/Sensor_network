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
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from keras.callbacks import TensorBoard
from keras import backend as K
from z_train_s1 import load_model_gcn, load_input
from spektral.layers import GCNConv
from spektral.utils import gcn_filter

img_path = 'training/'  # path for saved images from Unity
model_path = 'training/model_gcn_s1.h5'
pixel_dim = 84
input_shape = (pixel_dim,pixel_dim*4,3) # h w c
test_env = 1
test_case = np.arange(32)
num_sensor = 4

#####################################################################
 
def test_model(num_sensors, batch_size=1, lr=3e-4, is_robot=0):
    z_input = load_input(num_sensors=num_sensor, select_case=test_case, select_env=test_env)
    z_model = load_model_gcn(num_sensors=num_sensor) 
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
    
# test_model(4)    