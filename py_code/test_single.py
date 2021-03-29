# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 14:27:19 2020

@author: azrael
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
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input, Concatenate, Flatten, Reshape
from keras.callbacks import TensorBoard
from resnet import resnet_sensor_network
from spektral.layers import GraphConv
from loc2dir import s_label, sen_angle, sen_angle_single
#from gnn_pathplanning.utils.graphUtils import graphML as gml

def dis_angle(loc_self, loc_target):
    x_self = loc_self[0]
    z_self = loc_self[2]
    x_target = loc_target[0]    
    z_target = loc_target[2]
    dis = np.sqrt(abs(x_self-x_target)**2 + abs(z_self-z_target)**2)
    angle = np.arctan((x_target - x_self)/(z_target - z_self))
    return dis, angle

def process_img(x_train):
    mean_image = np.mean(x_train, axis=0)
    x_train -= mean_image
    x_train /= 64.
    return x_train

#################################################################   hyper parameters
batch_size = 32
train_iter = 20000
init_lr = 3e-4
all_sensors = ['sensor_1']
#all_sensors = ['sensor_1']
num_sensors = len(all_sensors)
input_shape = (84,84*4,3)
sensor_loc = [(0, 0, 0)]
#sensor_loc = [(0,0,0)]

env_1 = np.load('env_1.npy')    
target_label = open("target_loc_random.txt","r") 
lines = target_label.readlines() 
#lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0, patience=5, min_lr=0.5e-6)

#################################################################   get_label
filePath = 'training_random/sensor_1/1'
filelist = os.listdir(filePath)
filelist.sort(key = lambda x: int(x[:-4]))
#all_label = s_label(1)
all_label = sen_angle_single()
np.save('stage1_label_random.npy', all_label)
print('datasize:', len(all_label))

batch_matrix = np.ones((batch_size, num_sensors, num_sensors))
for i in range(batch_size):
    batch_matrix[i] = batch_matrix[i] - np.eye(num_sensors) 
    
#################################################################    build model
def origin_model():
    sensor_matrix = Input(shape=(num_sensors, num_sensors))
    
    s_input1, extract_cnn1 = resnet_sensor_network(input_shape, repetitions=[2,2,2,2])
    #s_input2, extract_cnn2 = resnet_sensor_network(input_shape, repetitions=[2,2,2,2])
    #s_input3, extract_cnn3 = resnet_sensor_network(input_shape, repetitions=[2,2,2,2])
    #s_input4, extract_cnn4 = resnet_sensor_network(input_shape, repetitions=[2,2,2,2])
    #extract_cnn = Concatenate(axis=1)([extract_cnn1, extract_cnn2, extract_cnn3, extract_cnn4])
        
    #extract_cnn = np.reshape(extract_cnn, (-1,))
    G_1 = GraphConv(256, 'relu')([extract_cnn1, sensor_matrix])
    G_2 = GraphConv(256, 'relu')([G_1, sensor_matrix])
    #gnn_output = tf.split(G_2, num_sensors, 1)
    
    output1 = Dense(32, activation='relu')(Flatten()(G_2))
    output1 = Dense(1, activation='linear', name='sensor_1')(output1)
    
    #output2 = Dense(32, activation='relu')(Flatten()(gnn_output[1]))
    #output2 = Dense(1, activation='linear', name='sensor_2')(output2)
    
    #output3 = Dense(32, activation='relu')(Flatten()(gnn_output[2]))
    #output3 = Dense(1, activation='linear', name='sensor_3')(output3)                                                         
                                                             
    #output4 = Dense(32, activation='relu')(Flatten()(gnn_output[3]))
    #output4 = Dense(1, activation='linear', name='sensor_4')(output4)
    
    model = Model(inputs=[s_input1, sensor_matrix], 
                  outputs= [output1])
    return model

def pose_model():
    sensor_matrix = Input(shape=(num_sensors, num_sensors))
    
    s_input1, extract_cnn1 = resnet_sensor_network(input_shape, repetitions=[2,2,2,2])
    s_input2, extract_cnn2 = resnet_sensor_network(input_shape, repetitions=[2,2,2,2])
    s_input3, extract_cnn3 = resnet_sensor_network(input_shape, repetitions=[2,2,2,2])
    s_input4, extract_cnn4 = resnet_sensor_network(input_shape, repetitions=[2,2,2,2])
    
    pose_input1 = Input(shape=(2))  
    s_pose1 = Dense(64, activation='relu')(pose_input1)
    pose_input2 = Input(shape=(2)) 
    s_pose2 = Dense(64, activation='relu')(pose_input2)
    pose_input3 = Input(shape=(2)) 
    s_pose3 = Dense(64, activation='relu')(pose_input3)
    pose_input4 = Input(shape=(2))
    s_pose4 = Dense(64, activation='relu')(pose_input4)
    
    extract_cnn1 = Concatenate(axis=-1)([extract_cnn1, Reshape((1,s_pose1.shape[-1]))(s_pose1)])
    extract_cnn2 = Concatenate(axis=-1)([extract_cnn2, Reshape((1,s_pose2.shape[-1]))(s_pose2)])
    extract_cnn3 = Concatenate(axis=-1)([extract_cnn3, Reshape((1,s_pose3.shape[-1]))(s_pose3)])
    extract_cnn4 = Concatenate(axis=-1)([extract_cnn4, Reshape((1,s_pose4.shape[-1]))(s_pose4)])
    extract_cnn = Concatenate(axis=1)([extract_cnn1, extract_cnn2, extract_cnn3, extract_cnn4])
      
    #extract_cnn = np.reshape(extract_cnn, (-1,))
    G_1 = GraphConv(256, 'relu')([extract_cnn, sensor_matrix])
    G_2 = GraphConv(256, 'relu')([G_1, sensor_matrix])
    gnn_output = tf.split(G_2, num_sensors, 1)
    
    output1 = Dense(32, activation='relu')(Flatten()(gnn_output[0]))
    output1 = Dense(1, activation='linear', name='sensor_1')(output1)
    
    output2 = Dense(32, activation='relu')(Flatten()(gnn_output[1]))
    output2 = Dense(1, activation='linear', name='sensor_2')(output2)
    
    output3 = Dense(32, activation='relu')(Flatten()(gnn_output[2]))
    output3 = Dense(1, activation='linear', name='sensor_3')(output3)                                                         
                                                             
    output4 = Dense(32, activation='relu')(Flatten()(gnn_output[3]))
    output4 = Dense(1, activation='linear', name='sensor_4')(output4)
    
    model = Model(inputs=[s_input1, s_input2, s_input3, s_input4, 
                          s_pose1, s_pose2, s_pose3, s_pose4,
                          sensor_matrix],  
                  outputs= [output1,output2,output3,output4])
    return model


model = origin_model()
from tensorflow.keras.optimizers import Adam
model.compile(optimizer=Adam(learning_rate=init_lr), loss='mse')    
    

#################################################################    training 

for num_iter in range(train_iter):
    print('start_training round:', num_iter)
    if num_iter == int(train_iter/3):
        init_lr = init_lr/10
        model.compile(optimizer=Adam(learning_rate=init_lr), loss='mse')
        print('new_learning:', init_lr)
    if num_iter == int(2*train_iter/3):
        init_lr = init_lr/10
        model.compile(optimizer=Adam(learning_rate=init_lr), loss='mse')
        print('new_learning:', init_lr)
    #randomly select a batch of sample
    select_case = [np.random.randint(1,len(all_label)) for _ in range(batch_size)] 
    batch_input = []
    batch_output = []
    for i in range(batch_size):
        all_sensor_input = np.zeros((num_sensors, 84, 84*4, 3))
        all_sensor_output = np.zeros((num_sensors, 1))
        for idx_sensor in range(num_sensors):
            sensor_path = 'training_random/' + all_sensors[idx_sensor]
            img_1 = image.load_img(sensor_path+'/1/'+filelist[select_case[i]], target_size=(84,84))  #height-width
            img_array_1 = image.img_to_array(img_1)
            img_2 = image.load_img(sensor_path+'/2/'+filelist[select_case[i]], target_size=(84,84))  #height-width
            img_array_2 = image.img_to_array(img_2)
            img_3 = image.load_img(sensor_path+'/3/'+filelist[select_case[i]], target_size=(84,84))  #height-width
            img_array_3 = image.img_to_array(img_3)
            img_4 = image.load_img(sensor_path+'/4/'+filelist[select_case[i]], target_size=(84,84))  #height-width
            img_array_4 = image.img_to_array(img_4)               
            all_sensor_input[idx_sensor,:, 84*3:84*4,:] = img_array_4/255 
            all_sensor_input[idx_sensor,:, 84*2:84*3,:] = img_array_3/255
            all_sensor_input[idx_sensor,:, 84*1:84*2,:] = img_array_2/255
            all_sensor_input[idx_sensor,:, 84*0:84*1,:] = img_array_1/255    
        batch_input.append(all_sensor_input.copy())
        #  get label data 
        img_index = int(filelist[select_case[i]][:-4])
        batch_output.append(all_label[select_case[i]]) 
    batch_input = np.array(batch_input) 
    batch_output = np.array(batch_output)  
    history = model.fit(x = [batch_input[:,0], batch_matrix], 
                        y = [batch_output[:,0]],
                        batch_size=batch_size, epochs=1 )
                    #callbacks=[TensorBoard(log_dir='mytensorboard')])
    hist_df = pd.DataFrame(history.history)
    hist_csv_file = 'cnn_history_single.csv'
    with open(hist_csv_file, mode='a') as f:
        hist_df.to_csv(f) 
    if num_iter % 500 == 0:
        print('save_model')
        model.save('stage_1_single.h5')
#################################################################    plot loss curve
'''
train_his = pd.read_csv('mytensorboard/cnn_history.csv')
train_loss = train_his['loss']
sensor_1_loss = train_his['sensor_1_loss']
sensor_2_loss = train_his['sensor_2_loss']
sensor_3_loss = train_his['sensor_3_loss']
sensor_4_loss = train_his['sensor_4_loss']
all_loss = []
s1_loss = []
s2_loss = []
s3_loss = []
s4_loss = []
for i in range(0, len(train_loss), 2):
    all_loss.append(float(train_loss[i][:6]))
    s1_loss.append(float(sensor_1_loss[i][:6]))
    s2_loss.append(float(sensor_2_loss[i][:6]))
    s3_loss.append(float(sensor_3_loss[i][:6]))
    s4_loss.append(float(sensor_4_loss[i][:6]))
plt.ylim(0, 2)
plt.xlabel('Training epochs')
plt.ylabel('Training Loss')
#plt.xticks([])
#plt.yticks([])
plt.plot(all_loss)
plt.plot(s1_loss)
plt.plot(s2_loss)
plt.plot(s3_loss)
plt.plot(s4_loss)

#plt.plot(history.history['loss'])
'''