# -*- coding: utf-8 -*-
"""
Created on Thu May 13 23:16:15 2021

@author: Win10
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os, cv2, math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from keras.callbacks import TensorBoard
from spektral.layers import GCNConv, GATConv
from spektral.utils import gcn_filter
from loc2dir import theta, angle2xy
from keras_lr_multiplier import LRMultiplier
from tensorflow.keras.layers import (
    Input,Conv2D, Dense, UpSampling2D, MaxPooling2D, AveragePooling2D, Concatenate, Flatten, Reshape )

#################################################################   hyper parameters
batch_size = 32
max_train_iter = 10000
max_num_sensors = 10
max_env_map = 10
pixel_dim = 84  # image size 
input_shape = (pixel_dim,pixel_dim*4,3)  # input size 
sensor_dis_threshold = 20  # distance for admatrix   20 == full connection
init_lr = 3e-4
   
#################################################################    load_input
def load_input(num_sensors=4, select_case=np.arange(2,33), select_env=1, path='training/'):
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

def read_target(select_case=np.arange(2,33), target_path='training/env_1/target_loc.txt'):
    target_loc = []    
    target_label = open(target_path,"r") 
    lines = target_label.readlines() 
    for i in range(len(select_case)):
        label_index = lines[select_case[i]-1].index(')')
        label_target = int(lines[select_case[i]-1][label_index+1:-1])
        x_index_1 = lines[select_case[i]-1].index('(')
        x_index_2 = lines[select_case[i]-1].index(',')
        label_x = float(lines[select_case[i]-1][x_index_1+1:x_index_2])
        z_index_1 = lines[select_case[i]-1].index(',', x_index_2+1)
        z_index_2 = lines[select_case[i]-1].index(')')  
        label_z = float(lines[select_case[i]-1][z_index_1+2:z_index_2])
        #t_x, t_y, t_z = change_axis(env, (label_x, 0, label_z))
        target_loc.append((label_x, label_z))
    return target_loc

def load_label(select_env, num_sensors, image_index):
    all_sensors = []
    for i in range(num_sensors):
        all_sensors.append('sensor_{}'.format(i+1))    
    all_target_label = []
    
    env_map = np.load('training/env_{}.npy'.format(select_env))
    sen_loc = np.load('training/env_{}_sensor.npy'.format(select_env))
    tar_loc = read_target('training/env_{}/target_loc.txt'.format(select_env), image_index)
    
    for i in range(len(image_index)):
        target_label = []
        for j in range(num_sensors):
            sensor_dir = []
            s_x, s_z = sen_loc[0][j], sen_loc[1][j]
            s_path = theta(env_map, (s_x, s_z), (math.ceil(tar_loc[i][0]), math.ceil(tar_loc[i][1])))
            s_angle = angle2xy(s_path[0], s_path[1])
            sensor_dir.append(s_angle)
            target_label.append(sensor_dir)
        all_target_label.append(target_label)
    return all_target_label

##################################################################################       
def cal_admatrix(env_index=1, num_sensors=4, sensor_dis = sensor_dis_threshold):
    sensor_loc = np.load('training/env_{}_sensor.npy'.format(env_index))
    ad_matrix = np.zeros((num_sensors, num_sensors))
    for s1 in range(num_sensors):
        for s2 in range(num_sensors):
            if s1 != s2:
                s_dis = np.sqrt((sensor_loc[0][s1]-sensor_loc[0][s2])**2+(sensor_loc[1][s1]-sensor_loc[1][s2])**2)
                if s_dis <= sensor_dis_threshold:
                    ad_matrix[s1, s2] = 1
    return ad_matrix
    
#################################################################    build model
def mlp_model(gnn_unit=256):
    input_data = Input(shape=gnn_unit)
    output1 = Dense(128, activation='relu',  name='mlp_1')(input_data)
    output1 = Dense(32, activation='relu',  name='mlp_2')(output1)
    output1 = Dense(2, activation='linear', name='sensors')(output1)  
    model = Model(inputs=[input_data], outputs=[output1])
    return model

def cnn_model(input_shape=(pixel_dim,pixel_dim*4,3)):
    act_func = 'relu'
    input_layer = Input(shape=input_shape)
    h = Conv2D(64,  (1, 2), strides=(1, 2), activation = act_func, padding='valid', name='conv_1')(input_layer)
    h = Conv2D(64,  (1, 2), strides=(1, 2), activation = act_func, padding='valid', name='conv_2')(h)
    h = Conv2D(128, (3, 3), strides=(2, 2) ,activation = act_func, padding='same', name='conv_3')(h) # pooling
    h = Conv2D(128, (3, 3), activation = act_func, padding='same', name='conv_4')(h)
    h = Conv2D(256, (3, 3), strides=(2, 2) ,activation = act_func, padding='same', name='conv_5')(h) # pooling
    h = Conv2D(256, (3, 3), activation = act_func, padding='same', name='conv_6')(h)
    h = Conv2D(256, (3, 3), strides=(2, 2) ,activation = act_func, padding='same', name='conv_7')(h) # pooling
    h = Conv2D(256, (3, 3), activation = act_func, padding='same', name='conv_8')(h)
    h = Conv2D(256, (3, 3), strides=(2, 2) ,activation = act_func, padding='same', name='conv_9')(h) # pooling
    h = Conv2D(256, (3, 3), activation = act_func, padding='same', name='conv_10')(h)
    h = Conv2D(256, (3, 3), strides=(2, 2) ,activation = act_func, padding='same', name='conv_11')(h) # pooling
    h = Conv2D(256, (3, 3), activation = 'relu', padding='valid', name='conv_12')(h)
    output_layer = Reshape((1,h.shape[-1]))(h)
    return Model(input_layer, output_layer)

def load_model_gcn(num_sensors, input_shape=(pixel_dim,pixel_dim*4,3), gnn_layers=2, gnn_unit=256, is_robot=0):
    
    input_data, output_data = [], []   
    #tf.compat.v1.enable_eager_execution()
    s_cnn = cnn_model(input_shape)
 
    if is_robot:
        num_sensors += 1
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
            exec("G_h{} = GCNConv(gnn_unit, activation='relu', dropout_rate=0, name='GNN_{}',)([extract_cnn, sensor_matrix])".format(j, j))
        else:
            exec("G_h{} = GCNConv(gnn_unit, activation='relu', dropout_rate=0, name='GNN_{}',)([G_h{}, sensor_matrix])".format(j, j, j-1))

    exec('gnn_output = tf.split(G_h{}, num_sensors, 1)'.format(gnn_layers))
    
    mlp_layer = mlp_model()
    for i in range(num_sensors):
        exec('output{} = mlp_layer(Flatten()(gnn_output[i]))'.format(i))
        exec('output_data.append(output{})'.format(i))
        
    model = Model(inputs=input_data, 
                  outputs= output_data)
    return model   

#################################################################    training

sensor_per_map = [4,5,7,6,8,9,9,10,10,10,10,10]
def train_model(num_epoch, num_batch, num_maps=10, lr_decay=True, is_robot=0, is_gcn=0):
    cur_lr = init_lr
    for ep in range(num_epoch):
        print('starting training epoch:', ep)
        if lr_decay:
            cur_lr = init_lr/np.sqrt(ep+1)
        
        select_env = np.random.randint(10)
        num_sensors = sensor_per_map[select_env]
        select_case = np.arange(2,33)
        np.random.shuffle(select_case)
        select_case = select_case[:32]
        input_image, image_index = load_input(num_sensors, select_case, select_env)
        
        if is_robot:
            z_admatrix = np.ones((len(select_case), num_sensors+1, num_sensors+1))
        else:
            z_admatrix = np.ones((len(select_case), num_sensors, num_sensors))   
        if is_gcn:
            z_admatrix = gcn_filter(z_admatrix)
        #input_admatrix = gcn_filter(z_admatrix)
        input_admatrix = z_admatrix.copy()
    
        input_data = []
        for i in range(num_sensors):
            input_data.append(input_image[:,i])   
        input_data.append(input_admatrix)
        
        input_label = load_label(select_env, num_sensors, image_index)
        
        model = load_model_gcn(num_sensors)
        model.load_weights('training/model_gcn_s1.h5')
        model.compile(optimizer=Adam(learning_rate=cur_lr), loss='mse')
        train_his = model.fit(input_data, np.asarray(input_label), batch_size=num_batch, epochs=int(len(input_label)/num_batch))
        hist_df = pd.DataFrame(train_his.history)
        hist_csv_file = 'training/history.csv'
        with open(hist_csv_file, mode='a') as f:
            hist_df.to_csv(f, index=False) 
        model.save('training/model_gcn_s1.h5')
        
        #z_res = model.predict(input_data)
                
##################################################################
#tf.config.run_functions_eagerly(True)
#train_model(num_epoch=max_train_iter, num_batch=batch_size, num_maps = max_env_map)

################################################################## plot history
#history_loss = pd.read_csv('training/history.csv')
#all_loss = history_loss['loss']
#s1_loss = history_loss['model_4_loss']
#z_his = np.load()

