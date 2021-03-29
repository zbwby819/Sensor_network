
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 16:21:34 2020

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
from resnet import resnet_sensor_network, sensor_cnn
from spektral.layers import GraphConv,GraphAttention
from spektral.utils import localpooling_filter
from loc2dir import s_label, sen_angle, s_label_batch, sample_batch
#from gnn_pathplanning.utils.graphUtils import graphML as gml

#################################################################   hyper parameters
batch_size = 128
train_iter = 20000
all_sensors = ['sensor_1', 'sensor_2', 'sensor_3', 'sensor_4',
               'sensor_5', 'sensor_6', 'sensor_7', 'sensor_8',
               'sensor_9']
#all_sensors = ['sensor_1']
num_sensors = len(all_sensors)
input_shape = (84,84*4,3)
#sensor_loc = [(5, 0, 5), (5, 0,-5), (-5, 0, -5), (-5, 0, 5)] #env-1
#sensor_loc = [(15, 0, 0), (5, 0, 0), (-5, 0, 0), (-15, 0, 0)] #env-2
#sensor_loc = [(0,0,0)]

#env_3 = np.load('env_3.npy')  
#env_4 = np.load('env_4.npy')   

#for k-hop   env_2
#spektral.utils.localpooling_filter(A, symmetric=True) 

#################################################################    build model
def mlp_model():
    input_data = Input(shape=1024)
    output1 = Dense(256, activation='selu',  name='mlp_1')(input_data)
    output1 = Dense(64, activation='selu',  name='mlp_2')(output1)
    output1 = Dense(2, activation='linear', name='sensors')(output1)
    model = Model(inputs=[input_data], outputs=[output1])
    return model

def khop_model_distribute(num_sensors=10): # input/output = num of sensors 
    gnn_unit = 128
    sensor_matrix1 = Input(shape=(num_sensors, num_sensors))
    sensor_matrix2 = Input(shape=(num_sensors, num_sensors))
    sensor_matrix3 = Input(shape=(num_sensors, num_sensors))
    sensor_matrix4 = Input(shape=(num_sensors, num_sensors))
    #sensor_matrix3 = Input(shape=(num_sensors, num_sensors))
    s_input1 = Input(shape=(input_shape[0], input_shape[1], input_shape[2]))
    s_input2 = Input(shape=(input_shape[0], input_shape[1], input_shape[2]))
    s_input3 = Input(shape=(input_shape[0], input_shape[1], input_shape[2]))
    s_input4 = Input(shape=(input_shape[0], input_shape[1], input_shape[2]))
    s_input5 = Input(shape=(input_shape[0], input_shape[1], input_shape[2]))
    s_input6 = Input(shape=(input_shape[0], input_shape[1], input_shape[2]))
    s_input7 = Input(shape=(input_shape[0], input_shape[1], input_shape[2]))
    s_input8 = Input(shape=(input_shape[0], input_shape[1], input_shape[2]))
    s_input9 = Input(shape=(input_shape[0], input_shape[1], input_shape[2]))
    s_input10 = Input(shape=(input_shape[0], input_shape[1], input_shape[2]))
    
    s_cnn = sensor_cnn(input_shape, repetitions = [2,2,2,2])
    extract_cnn1 = s_cnn(s_input1)
    extract_cnn2 = s_cnn(s_input2)
    extract_cnn3 = s_cnn(s_input3)
    extract_cnn4 = s_cnn(s_input4)
    extract_cnn5 = s_cnn(s_input5)
    extract_cnn6 = s_cnn(s_input6)
    extract_cnn7 = s_cnn(s_input7)
    extract_cnn8 = s_cnn(s_input8)
    extract_cnn9 = s_cnn(s_input9)
    extract_cnn10 = s_cnn(s_input10)
    
    extract_cnn = Concatenate(axis=1)([extract_cnn1, extract_cnn2, extract_cnn3, 
                                       extract_cnn4, extract_cnn5, extract_cnn6,
                                       extract_cnn7, extract_cnn8, extract_cnn9, extract_cnn10])
        
    #extract_cnn = np.reshape(extract_cnn, (-1,))
    G_h1 = GraphConv(gnn_unit, 'selu')([extract_cnn, sensor_matrix1])
    G_h2 = GraphConv(gnn_unit, 'selu')([extract_cnn, sensor_matrix2])
    G_h3 = GraphConv(gnn_unit, 'selu')([extract_cnn, sensor_matrix3])
    G_h4 = GraphConv(gnn_unit, 'selu')([extract_cnn, sensor_matrix4])
    G_1 = Concatenate(axis=-1)([G_h1, G_h2, G_h3, G_h4])
  
    G_2h1 = GraphConv(gnn_unit, 'selu')([G_1, sensor_matrix1])
    G_2h2 = GraphConv(gnn_unit, 'selu')([G_1, sensor_matrix2])
    G_2h3 = GraphConv(gnn_unit, 'selu')([G_1, sensor_matrix3])
    G_2h4 = GraphConv(gnn_unit, 'selu')([G_1, sensor_matrix4])
    G_2 = Concatenate(axis=-1)([G_2h1, G_2h2, G_2h3, G_2h4])

    #G_3h1 = GraphConv(gnn_unit, 'selu')([G_2, sensor_matrix1])
    #G_3h2 = GraphConv(gnn_unit, 'selu')([G_2, sensor_matrix2])
    #G_3h3 = GraphConv(gnn_unit, 'selu')([G_2, sensor_matrix3])
    #G_3h4 = GraphConv(gnn_unit, 'selu')([G_2, sensor_matrix4])
    #G_3 = Concatenate(axis=-1)([G_3h1, G_3h2, G_3h3, G_3h4])
    
    #G_4h1 = GraphConv(gnn_unit, 'selu')([G_3, sensor_matrix1])
    #G_4h2 = GraphConv(gnn_unit, 'selu')([G_3, sensor_matrix2])
    #G_4h3 = GraphConv(gnn_unit, 'selu')([G_3, sensor_matrix3])
    #G_4h4 = GraphConv(gnn_unit, 'selu')([G_3, sensor_matrix4])
    #G_4 = Concatenate(axis=-1)([G_4h1, G_4h2, G_4h3, G_4h4])

    gnn_output = tf.split(G_2, num_sensors, 1)
        
    mlp_layer = mlp_model()
    
    output1 = mlp_layer(Flatten()(gnn_output[0]))
    output2 = mlp_layer(Flatten()(gnn_output[1]))
    output3 = mlp_layer(Flatten()(gnn_output[2]))
    output4 = mlp_layer(Flatten()(gnn_output[3]))
    output5 = mlp_layer(Flatten()(gnn_output[4]))
    output6 = mlp_layer(Flatten()(gnn_output[5]))
    output7 = mlp_layer(Flatten()(gnn_output[6]))
    output8 = mlp_layer(Flatten()(gnn_output[7]))
    output9 = mlp_layer(Flatten()(gnn_output[8]))
    output10 = mlp_layer(Flatten()(gnn_output[9]))
    
    model = Model(inputs=[s_input1, s_input2, s_input3, s_input4,
                          s_input5, s_input6, s_input7, s_input8, s_input9, s_input10,
                          sensor_matrix1, sensor_matrix2, sensor_matrix3, sensor_matrix4], 
                  outputs= [output1,output2,output3,output4,
                            output5,output6,output7,output8,output9, output10])
    return model

def att_model_distribute(num_sensors=10): # input/output = num of sensors 
    gnn_unit = 128
    sensor_matrix1 = Input(shape=(num_sensors, num_sensors))
    sensor_matrix2 = Input(shape=(num_sensors, num_sensors))
    sensor_matrix3 = Input(shape=(num_sensors, num_sensors))
    sensor_matrix4 = Input(shape=(num_sensors, num_sensors))
    #sensor_matrix3 = Input(shape=(num_sensors, num_sensors))
    s_input1 = Input(shape=(input_shape[0], input_shape[1], input_shape[2]))
    s_input2 = Input(shape=(input_shape[0], input_shape[1], input_shape[2]))
    s_input3 = Input(shape=(input_shape[0], input_shape[1], input_shape[2]))
    s_input4 = Input(shape=(input_shape[0], input_shape[1], input_shape[2]))
    s_input5 = Input(shape=(input_shape[0], input_shape[1], input_shape[2]))
    s_input6 = Input(shape=(input_shape[0], input_shape[1], input_shape[2]))
    s_input7 = Input(shape=(input_shape[0], input_shape[1], input_shape[2]))
    s_input8 = Input(shape=(input_shape[0], input_shape[1], input_shape[2]))
    s_input9 = Input(shape=(input_shape[0], input_shape[1], input_shape[2]))
    s_input10 = Input(shape=(input_shape[0], input_shape[1], input_shape[2]))
    
    s_cnn = sensor_cnn(input_shape, repetitions = [2,2,2,2])
    extract_cnn1 = s_cnn(s_input1)
    extract_cnn2 = s_cnn(s_input2)
    extract_cnn3 = s_cnn(s_input3)
    extract_cnn4 = s_cnn(s_input4)
    extract_cnn5 = s_cnn(s_input5)
    extract_cnn6 = s_cnn(s_input6)
    extract_cnn7 = s_cnn(s_input7)
    extract_cnn8 = s_cnn(s_input8)
    extract_cnn9 = s_cnn(s_input9)
    extract_cnn10 = s_cnn(s_input10)
    
    extract_cnn = Concatenate(axis=1)([extract_cnn1, extract_cnn2, extract_cnn3, 
                                       extract_cnn4, extract_cnn5, extract_cnn6,
                                       extract_cnn7, extract_cnn8, extract_cnn9, extract_cnn10])
        
    #extract_cnn = np.reshape(extract_cnn, (-1,))
    G_h1 = GraphAttention(gnn_unit, activation='selu', dropout_rate=0)([extract_cnn, sensor_matrix1])
    G_h2 = GraphAttention(gnn_unit, activation='selu', dropout_rate=0)([extract_cnn, sensor_matrix2])
    G_h3 = GraphAttention(gnn_unit, activation='selu', dropout_rate=0)([extract_cnn, sensor_matrix3])
    G_h4 = GraphAttention(gnn_unit, activation='selu', dropout_rate=0)([extract_cnn, sensor_matrix4])
    G_1 = Concatenate(axis=-1)([G_h1, G_h2, G_h3, G_h4])
  
    G_2h1 = GraphAttention(gnn_unit, activation='selu', dropout_rate=0)([G_1, sensor_matrix1])
    G_2h2 = GraphAttention(gnn_unit, activation='selu', dropout_rate=0)([G_1, sensor_matrix2])
    G_2h3 = GraphAttention(gnn_unit, activation='selu', dropout_rate=0)([G_1, sensor_matrix3])
    G_2h4 = GraphAttention(gnn_unit, activation='selu', dropout_rate=0)([G_1, sensor_matrix4])
    G_2 = Concatenate(axis=-1)([G_2h1, G_2h2, G_2h3, G_2h4])

    #G_3h1 = GraphAttention(gnn_unit, activation='selu', dropout_rate=0)([G_2, sensor_matrix1])
    #G_3h2 = GraphAttention(gnn_unit, activation='selu', dropout_rate=0)([G_2, sensor_matrix2])
    #G_3h3 = GraphAttention(gnn_unit, activation='selu', dropout_rate=0)([G_2, sensor_matrix3])
    #G_3h4 = GraphAttention(gnn_unit, activation='selu', dropout_rate=0)([G_2, sensor_matrix4])
    #G_3 = Concatenate(axis=-1)([G_3h1, G_3h2, G_3h3, G_3h4])
    
    #G_4h1 = GraphAttention(gnn_unit, activation='selu', dropout_rate=0)([G_3, sensor_matrix1])
    #G_4h2 = GraphAttention(gnn_unit, activation='selu', dropout_rate=0)([G_3, sensor_matrix2])
    #G_4h3 = GraphAttention(gnn_unit, activation='selu', dropout_rate=0)([G_3, sensor_matrix3])
    #G_4h4 = GraphAttention(gnn_unit, activation='selu', dropout_rate=0)([G_3, sensor_matrix4])
    #G_4 = Concatenate(axis=-1)([G_4h1, G_4h2, G_4h3, G_4h4])

    gnn_output = tf.split(G_2, num_sensors, 1)
    
    mlp_input1 = Concatenate(axis = -1)([gnn_output[0], extract_cnn1])
    mlp_input2 = Concatenate(axis = -1)([gnn_output[1], extract_cnn2])
    mlp_input3 = Concatenate(axis = -1)([gnn_output[2], extract_cnn3])
    mlp_input4 = Concatenate(axis = -1)([gnn_output[3], extract_cnn4])
    mlp_input5 = Concatenate(axis = -1)([gnn_output[4], extract_cnn5])
    mlp_input6 = Concatenate(axis = -1)([gnn_output[5], extract_cnn6])
    mlp_input7 = Concatenate(axis = -1)([gnn_output[6], extract_cnn7])
    mlp_input8 = Concatenate(axis = -1)([gnn_output[7], extract_cnn8])
    mlp_input9 = Concatenate(axis = -1)([gnn_output[8], extract_cnn9])
    mlp_input10 = Concatenate(axis = -1)([gnn_output[9], extract_cnn10])
    
        
    mlp_layer = mlp_model()
    
    output1 = mlp_layer(Flatten()(mlp_input1))
    output2 = mlp_layer(Flatten()(mlp_input2))
    output3 = mlp_layer(Flatten()(mlp_input3))
    output4 = mlp_layer(Flatten()(mlp_input4))
    output5 = mlp_layer(Flatten()(mlp_input5))
    output6 = mlp_layer(Flatten()(mlp_input6))
    output7 = mlp_layer(Flatten()(mlp_input7))
    output8 = mlp_layer(Flatten()(mlp_input8))
    output9 = mlp_layer(Flatten()(mlp_input9))
    output10 = mlp_layer(Flatten()(mlp_input10))
    
    model = Model(inputs=[s_input1, s_input2, s_input3, s_input4,
                          s_input5, s_input6, s_input7, s_input8, s_input9, s_input10,
                          sensor_matrix1, sensor_matrix2, sensor_matrix3, sensor_matrix4], 
                  outputs= [output1,output2,output3,output4,
                            output5,output6,output7,output8,output9, output10])
    return model

def train_model(num_sensors=9, num_hop=4, input_shape=(84,84*4,3)): # input/output = num of sensors     
    input_data, output_data = [], []    
    
    s_cnn = sensor_cnn(input_shape, repetitions = [2,2,2,2])
    for i in range(num_sensors):
        exec('s_input{} = Input(shape=(input_shape[0], input_shape[1], input_shape[2]))'.format(i))
        exec('extract_cnn{} = s_cnn(s_input{})'.format(i, i))
        exec('input_data.append(s_input{})'.format(i))
    
    for j in range(num_hop):    
        exec('sensor_matrix{} = Input(shape=(num_sensors, num_sensors))'.format(j))
        exec('input_data.append(sensor_matrix{})'.format(j))
    
    exec('extract_cnn = extract_cnn0')
    for i in range(1,num_sensors):
        exec('extract_cnn = Concatenate(axis=1)([extract_cnn, extract_cnn{}])'.format(i))
    
    for j in range(num_hop):
        exec("G_h{} = GraphConv(256, 'relu')([extract_cnn, sensor_matrix{}])".format(j,j))
    exec('G_1 = G_h0')
    for j in range(1, num_hop):
        exec('G_1 = Concatenate(axis=-1)([G_1, G_h{}])'.format(j))
    
    for j in range(num_hop):
        exec("G_2h{} = GraphConv(256, 'relu')([G_1, sensor_matrix{}])".format(j,j))
    exec('G_2 = G_2h0')
    for j in range(1, num_hop):
        exec('G_2 = Concatenate(axis=-1)([G_2, G_2h{}])'.format(j))
    
    exec('gnn_output = tf.split(G_2, num_sensors, 1)')
    mlp_layer = mlp_model()
    
    for i in range(num_sensors):
        exec('output{} = mlp_layer(Flatten()(gnn_output[i]))'.format(i))
        exec('output_data.append(output{})'.format(i))
    
    model = Model(inputs=input_data, 
                  outputs= output_data)
    return model

##################################################################  train khop-share
total_group1 = 6
total_group2 = 6
#num_sensors = 9

def train_sample(if_att = False):
    init_lr = 3e-5
    for num_iter in range(train_iter):
        print('start_training round:', num_iter)        
        if num_iter == int(train_iter/4):
            new_lr = init_lr/10
            model.compile(optimizer=Adam(learning_rate=new_lr), loss='mse')
            print('new_learning:', init_lr)
        if num_iter == int(train_iter/2):
            new_lr = init_lr/100
            model.compile(optimizer=Adam(learning_rate=new_lr), loss='mse')
            print('new_learning:', init_lr)
        if num_iter == int(train_iter/1.3):
            new_lr = init_lr/1000
            model.compile(optimizer=Adam(learning_rate=new_lr), loss='mse')
            print('new_learning:', init_lr)
            
        input_data, output_data, ad_matrix1, ad_matrix2, ad_matrix3, ad_matrix4, num_sensors = sample_batch(batch_size)

        for i in range(1):
            #  num of different batches, currently is 1
            '''
            if num_sensors == 9:
                for z_i in range(len(input_data[i])):    
                    input_data[i][z_i] = np.concatenate((input_data[i][z_i], np.zeros((1,84,336,3))), axis=0 )
                    output_data[i][z_i] = np.concatenate((output_data[i][z_i], np.zeros((1,2))), axis=0)
                num_sensors = 10
                ad_matrix1[i] = np.concatenate((ad_matrix1[i], np.zeros((1,9))), axis=0)
                ad_matrix1[i] = np.concatenate((ad_matrix1[i], np.zeros((10,1))), axis=-1)
                ad_matrix2[i] = np.concatenate((ad_matrix2[i], np.zeros((1,9))), axis=0)
                ad_matrix2[i] = np.concatenate((ad_matrix2[i], np.zeros((10,1))), axis=-1)
                ad_matrix3[i] = np.concatenate((ad_matrix3[i], np.zeros((1,9))), axis=0)
                ad_matrix3[i] = np.concatenate((ad_matrix3[i], np.zeros((10,1))), axis=-1)
                ad_matrix4[i] = np.concatenate((ad_matrix4[i], np.zeros((1,9))), axis=0)
                ad_matrix4[i] = np.concatenate((ad_matrix4[i], np.zeros((10,1))), axis=-1)     
            '''
                
            batch_input = np.array(input_data[i])
            batch_output = np.array(output_data[i])
            batch_matrix1 = np.zeros((batch_size, num_sensors, num_sensors))
            batch_matrix2 = np.zeros((batch_size, num_sensors, num_sensors))
            batch_matrix3 = np.zeros((batch_size, num_sensors, num_sensors))
            batch_matrix4 = np.zeros((batch_size, num_sensors, num_sensors))
                
            for j in range(batch_size):
                if if_att == False:
                    batch_matrix1[j] = localpooling_filter(ad_matrix1[i])
                    batch_matrix2[j] = localpooling_filter(ad_matrix2[i])
                    batch_matrix3[j] = localpooling_filter(ad_matrix3[i])
                    batch_matrix4[j] = localpooling_filter(ad_matrix4[i])
                else:
                    batch_matrix1[j] = ad_matrix1[i]
                    batch_matrix2[j] = ad_matrix2[i]
                    batch_matrix3[j] = ad_matrix3[i]
                    batch_matrix4[j] = ad_matrix4[i]
                    
            history = model.fit(x = [batch_input[:,0], batch_input[:,1], batch_input[:,2], batch_input[:,3], 
                                     batch_input[:,4], batch_input[:,5], batch_input[:,6], batch_input[:,7], 
                                     batch_input[:,8], #batch_input[:,9], 
                                     batch_matrix1, batch_matrix2, batch_matrix3, batch_matrix4], 
                                y = [batch_output[:,0], batch_output[:,1], batch_output[:,2], batch_output[:,3],
                                     batch_output[:,4], batch_output[:,5], batch_output[:,6], batch_output[:,7],
                                     batch_output[:,8], #batch_output[:,9]
                                     ],
                                batch_size=batch_size, epochs=1, shuffle = True)
                        #callbacks=[TensorBoard(log_dir='mytensorboard')])
            hist_df = pd.DataFrame(history.history)
            hist_csv_file = 'history1002_att.csv'
            with open(hist_csv_file, mode='a') as f:
                hist_df.to_csv(f) 
        if num_iter % 500 == 100:
            print('save_model')
            model.save('gnn_1002_att.h5')
   
##################################################################
model = att_model_distribute()
from tensorflow.keras.optimizers import Adam
init_lr = 3e-5
model.compile(optimizer=Adam(learning_rate=init_lr, clipnorm=3), loss='mse')    
train_sample(if_att=True)
