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
from spektral.layers import GraphConv
from spektral.utils import localpooling_filter
from loc2dir import s_label, sen_angle, s_label_batch, sample_batch
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
batch_size = 64
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

env_3 = np.load('env_3.npy')  
env_4 = np.load('env_4.npy')   

#for k-hop   env_2
#spektral.utils.localpooling_filter(A, symmetric=True) 

#################################################################    build model
def khop_model_share(): # input/output = num of sensors 
    sensor_matrix1 = Input(shape=(num_sensors, num_sensors))
    sensor_matrix2 = Input(shape=(num_sensors, num_sensors))
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
    
    extract_cnn = Concatenate(axis=1)([extract_cnn1, extract_cnn2, extract_cnn3, 
                                       extract_cnn4, extract_cnn5, extract_cnn6,
                                       extract_cnn7, extract_cnn8, extract_cnn9])
        
    #extract_cnn = np.reshape(extract_cnn, (-1,))
    G_h1 = GraphConv(256, 'relu')([extract_cnn, sensor_matrix1])
    G_h2 = GraphConv(256, 'relu')([extract_cnn, sensor_matrix2])
    G_1 = Concatenate(axis=-1)([G_h1, G_h2])
  
    G_2h1 = GraphConv(256, 'relu')([G_1, sensor_matrix1])
    G_2h2 = GraphConv(256, 'relu')([G_1, sensor_matrix2])
    G_2 = Concatenate(axis=-1)([G_2h1, G_2h2])
    
    gnn_output = tf.split(G_2, num_sensors, 1)
    
    output1 = Dense(32, activation='relu')(Flatten()(gnn_output[0]))
    output1 = Dense(2, activation='linear', name='sensor_1')(output1)
    
    output2 = Dense(32, activation='relu')(Flatten()(gnn_output[1]))
    output2 = Dense(2, activation='linear', name='sensor_2')(output2)
    
    output3 = Dense(32, activation='relu')(Flatten()(gnn_output[2]))
    output3 = Dense(2, activation='linear', name='sensor_3')(output3)                                                         
                                                             
    output4 = Dense(32, activation='relu')(Flatten()(gnn_output[3]))
    output4 = Dense(2, activation='linear', name='sensor_4')(output4)
    
    output5 = Dense(32, activation='relu')(Flatten()(gnn_output[4]))
    output5 = Dense(2, activation='linear', name='sensor_5')(output5)

    output6 = Dense(32, activation='relu')(Flatten()(gnn_output[5]))
    output6 = Dense(2, activation='linear', name='sensor_6')(output6)
    
    output7 = Dense(32, activation='relu')(Flatten()(gnn_output[6]))
    output7 = Dense(2, activation='linear', name='sensor_7')(output7)

    output8 = Dense(32, activation='relu')(Flatten()(gnn_output[7]))
    output8 = Dense(2, activation='linear', name='sensor_8')(output8)
    
    output9 = Dense(32, activation='relu')(Flatten()(gnn_output[8]))
    output9 = Dense(2, activation='linear', name='sensor_9')(output9)
    
    model = Model(inputs=[s_input1, s_input2, s_input3, s_input4,
                          s_input5, s_input6, s_input7, s_input8, s_input9,
                          sensor_matrix1, sensor_matrix2], 
                  outputs= [output1,output2,output3,output4,
                            output5,output6,output7,output8,output9])
    return model

def mlp_model():
    input_data = Input(shape=512)
    output1 = Dense(128, activation='selu',  name='mlp_1')(input_data)
    output1 = Dense(32, activation='selu',  name='mlp_2')(output1)
    output1 = Dense(2, activation='linear', name='sensors')(output1)
    
    model = Model(inputs=[input_data], outputs=[output1])
    return model

def khop_model_distribute(): # input/output = num of sensors 
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
    
    extract_cnn = Concatenate(axis=1)([extract_cnn1, extract_cnn2, extract_cnn3, 
                                       extract_cnn4, extract_cnn5, extract_cnn6,
                                       extract_cnn7, extract_cnn8, extract_cnn9])
        
    #extract_cnn = np.reshape(extract_cnn, (-1,))
    G_h1 = GraphConv(128, 'relu')([extract_cnn, sensor_matrix1])
    G_h2 = GraphConv(128, 'relu')([extract_cnn, sensor_matrix2])
    G_h3 = GraphConv(128, 'relu')([extract_cnn, sensor_matrix3])
    G_h4 = GraphConv(128, 'relu')([extract_cnn, sensor_matrix4])
    G_1 = Concatenate(axis=-1)([G_h1, G_h2, G_h3, G_h4])
  
    G_2h1 = GraphConv(128, 'relu')([G_1, sensor_matrix1])
    G_2h2 = GraphConv(128, 'relu')([G_1, sensor_matrix2])
    G_2h3 = GraphConv(128, 'relu')([G_1, sensor_matrix3])
    G_2h4 = GraphConv(128, 'relu')([G_1, sensor_matrix4])
    G_2 = Concatenate(axis=-1)([G_2h1, G_2h2, G_2h3, G_2h4])
    
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
    
    model = Model(inputs=[s_input1, s_input2, s_input3, s_input4,
                          s_input5, s_input6, s_input7, s_input8, s_input9,
                          sensor_matrix1, sensor_matrix2, sensor_matrix3, sensor_matrix4], 
                  outputs= [output1,output2,output3,output4,
                            output5,output6,output7,output8,output9])
    return model

def khop_model_distribute10(): # input/output = num of sensors 
    sensor_matrix1 = Input(shape=(num_sensors+1, num_sensors+1))
    sensor_matrix2 = Input(shape=(num_sensors+1, num_sensors+1))
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
    G_h1 = GraphConv(256, 'relu')([extract_cnn, sensor_matrix1])
    G_h2 = GraphConv(256, 'relu')([extract_cnn, sensor_matrix2])
    G_1 = Concatenate(axis=-1)([G_h1, G_h2])
  
    G_2h1 = GraphConv(256, 'relu')([G_1, sensor_matrix1])
    G_2h2 = GraphConv(256, 'relu')([G_1, sensor_matrix2])
    G_2 = Concatenate(axis=-1)([G_2h1, G_2h2])
    
    gnn_output = tf.split(G_2, num_sensors+1, 1)
        
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
                          s_input5, s_input6, s_input7, s_input8, s_input9,s_input10,
                          sensor_matrix1, sensor_matrix2], 
                  outputs= [output1,output2,output3,output4,
                            output5,output6,output7,output8,output9,output10])
    return model

##################################################################  train khop-share
total_group1 = 6
total_group2 = 6
num_sensors = 9

def train_sample():
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

        input_data, output_data, ad_matrix1, ad_matrix2, ad_matrix3, ad_matrix4 = sample_batch(batch_size)

        for i in range(4):
            batch_input = np.array(input_data[i])
            batch_output = np.array(output_data[i])
            batch_matrix1 = np.zeros((batch_size, num_sensors, num_sensors))
            batch_matrix2 = np.zeros((batch_size, num_sensors, num_sensors))
            batch_matrix3 = np.zeros((batch_size, num_sensors, num_sensors))
            batch_matrix4 = np.zeros((batch_size, num_sensors, num_sensors))
            
            for j in range(batch_size):
                batch_matrix1[j] = localpooling_filter(ad_matrix1[i])
                batch_matrix2[j] = localpooling_filter(ad_matrix2[i])
                batch_matrix3[j] = localpooling_filter(ad_matrix3[i])
                batch_matrix4[j] = localpooling_filter(ad_matrix4[i])
            history = model.fit(x = [batch_input[:,0], batch_input[:,1], batch_input[:,2], batch_input[:,3], 
                                     batch_input[:,4], batch_input[:,5], batch_input[:,6], batch_input[:,7], batch_input[:,8],
                                     batch_matrix1, batch_matrix2, batch_matrix3, batch_matrix4], 
                                y = [batch_output[:,0], batch_output[:,1], batch_output[:,2], batch_output[:,3],
                                     batch_output[:,4], batch_output[:,5], batch_output[:,6], batch_output[:,7], batch_output[:,8]],
                                batch_size=batch_size, epochs=1, shuffle = True)
                        #callbacks=[TensorBoard(log_dir='mytensorboard')])
            hist_df = pd.DataFrame(history.history)
            hist_csv_file = 'history0910.csv'
            with open(hist_csv_file, mode='a') as f:
                hist_df.to_csv(f) 
        if num_iter % 500 == 100:
            print('save_model')
            model.save('gnn_0911.h5')
    

def train_khop_share():
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
        #randomly select a batch of sample
        select_group = np.random.randint(1,total_group1+1)
        select_group2 = np.random.randint(1,total_group2+1)
        print('select_env3_:', select_group,  '    select_env4_:', select_group2)
        if select_group == 1:    
            label_path_env3 = "target_env3_1.txt"
            ad_matrix_env3 = np.load('ad_matrix_env3_1.npy')
            ad_matrix2_env3 = np.load('ad_matrix2_env3_1.npy')
        if select_group == 2:     
            label_path_env3 = "target_env3_2.txt"
            ad_matrix_env3 = np.load('ad_matrix_env3_2.npy')
            ad_matrix2_env3 = np.load('ad_matrix2_env3_2.npy')
        if select_group == 3:     
            label_path_env3 = "target_env3_3.txt"
            ad_matrix_env3 = np.load('ad_matrix_env3_3.npy')
            ad_matrix2_env3 = np.load('ad_matrix2_env3_3.npy')
        if select_group == 4:     
            label_path_env3 = "target_env3_4.txt"
            ad_matrix_env3 = np.load('ad_matrix_env3_4.npy')
            ad_matrix2_env3 = np.load('ad_matrix2_env3_4.npy')
        if select_group == 5:     
            label_path_env3 = "target_env3_5.txt"
            ad_matrix_env3 = np.load('ad_matrix_env3_5.npy')
            ad_matrix2_env3 = np.load('ad_matrix2_env3_5.npy')
        if select_group == 6:     
            label_path_env3 = "target_env3_6.txt"
            ad_matrix_env3 = np.load('ad_matrix_env3_6.npy')
            ad_matrix2_env3 = np.load('ad_matrix2_env3_6.npy')
        
        if select_group2 == 1:    
            label_path_env4 = "target_env4_1.txt"
            ad_matrix_env4 = np.load('ad_matrix_env4_1.npy')
            ad_matrix2_env4 = np.load('ad_matrix2_env4_1.npy')
        if select_group2 == 2:     
            label_path_env4 = "target_env4_2.txt"
            ad_matrix_env4 = np.load('ad_matrix_env4_2.npy')
            ad_matrix2_env4 = np.load('ad_matrix2_env4_2.npy')
        if select_group2 == 3:     
            label_path_env4 = "target_env4_3.txt"
            ad_matrix_env4 = np.load('ad_matrix_env4_3.npy')
            ad_matrix2_env4 = np.load('ad_matrix2_env4_3.npy')
        if select_group2 == 4:     
            label_path_env4 = "target_env4_4.txt"
            ad_matrix_env4 = np.load('ad_matrix_env4_4.npy')
            ad_matrix2_env4 = np.load('ad_matrix2_env4_4.npy')
        if select_group2 == 5:     
            label_path_env4 = "target_env4_5.txt"
            ad_matrix_env4 = np.load('ad_matrix_env4_5.npy')
            ad_matrix2_env4 = np.load('ad_matrix2_env4_5.npy')
        if select_group2 == 6:     
            label_path_env4 = "target_env4_6.txt"
            ad_matrix_env4 = np.load('ad_matrix_env4_6.npy')
            ad_matrix2_env4 = np.load('ad_matrix2_env4_6.npy')
        
        filePath_env3 = 'training_env3_{}/sensor_1/1'.format(select_group)
        filelist_env3 = os.listdir(filePath_env3)
        filelist_env3.sort(key = lambda x: int(x[:-4]))
        
        filePath_env4 = 'training_env4_{}/sensor_1/1'.format(select_group2)
        filelist_env4 = os.listdir(filePath_env4)
        filelist_env4.sort(key = lambda x: int(x[:-4]))

        batch_matrix1_env3 = np.zeros((batch_size, num_sensors, num_sensors))
        batch_matrix2_env3 = np.zeros((batch_size, num_sensors, num_sensors))
        batch_matrix1_env4 = np.zeros((batch_size, num_sensors, num_sensors))
        batch_matrix2_env4 = np.zeros((batch_size, num_sensors, num_sensors))
        for i in range(batch_size):
            batch_matrix1_env3[i] = localpooling_filter(ad_matrix_env3)
            batch_matrix2_env3[i] = localpooling_filter(ad_matrix2_env3)
            batch_matrix1_env4[i] = localpooling_filter(ad_matrix_env4)
            batch_matrix2_env4[i] = localpooling_filter(ad_matrix2_env4)
        batch_matrix1 = np.concatenate((batch_matrix1_env3,batch_matrix1_env4), axis=0) 
        batch_matrix2 = np.concatenate((batch_matrix2_env3,batch_matrix2_env4), axis=0) 
        
        target_label_env3 = open(label_path_env3,"r") 
        target_label_env4 = open(label_path_env4,"r") 
        lines_env3 = target_label_env3.readlines() 
        lines_env4 = target_label_env4.readlines()
        #select_case_env3 = np.arange(batch_size)
        select_case_env3 = [np.random.randint(len(lines_env3)) for _ in range(batch_size)]
        select_case_env4 = [np.random.randint(len(lines_env4)) for _ in range(batch_size)]
        
        batch_input_env3 = []
        batch_input_env4 = []
        batch_output_env3 = s_label_batch(select_group, select_case_env3, 3)
        batch_output_env4 = s_label_batch(select_group2, select_case_env4, 4)
        
        for i in range(batch_size):
            ####### for env3
            all_sensor_input_env3 = np.zeros((num_sensors, 84, 84*4, 3))
            for idx_sensor in range(num_sensors):
                sensor_path = 'training_env3_{}/'.format(select_group) + all_sensors[idx_sensor]
                img_1 = image.load_img(sensor_path+'/1/'+filelist_env3[select_case_env3[i]], target_size=(84,84))  #height-width
                img_array_1 = image.img_to_array(img_1)
                img_2 = image.load_img(sensor_path+'/2/'+filelist_env3[select_case_env3[i]], target_size=(84,84))  #height-width
                img_array_2 = image.img_to_array(img_2)
                img_3 = image.load_img(sensor_path+'/3/'+filelist_env3[select_case_env3[i]], target_size=(84,84))  #height-width
                img_array_3 = image.img_to_array(img_3)
                img_4 = image.load_img(sensor_path+'/4/'+filelist_env3[select_case_env3[i]], target_size=(84,84))  #height-width
                img_array_4 = image.img_to_array(img_4)  
                all_sensor_input_env3[idx_sensor,:, 84*3:84*4,:] = img_array_1/255 
                all_sensor_input_env3[idx_sensor,:, 84*2:84*3,:] = img_array_2/255
                all_sensor_input_env3[idx_sensor,:, 84*1:84*2,:] = img_array_3/255
                all_sensor_input_env3[idx_sensor,:, 84*0:84*1,:] = img_array_4/255    
            batch_input_env3.append(all_sensor_input_env3.copy())
            
        for i in range(batch_size): 
            ####### for env4
            all_sensor_input_env4 = np.zeros((num_sensors, 84, 84*4, 3))
            for idx_sensor in range(num_sensors):
                sensor_path = 'training_env4_{}/'.format(select_group2) + all_sensors[idx_sensor]
                img_1 = image.load_img(sensor_path+'/1/'+filelist_env4[select_case_env4[i]], target_size=(84,84))  #height-width
                img_array_1 = image.img_to_array(img_1)
                img_2 = image.load_img(sensor_path+'/2/'+filelist_env4[select_case_env4[i]], target_size=(84,84))  #height-width
                img_array_2 = image.img_to_array(img_2)
                img_3 = image.load_img(sensor_path+'/3/'+filelist_env4[select_case_env4[i]], target_size=(84,84))  #height-width
                img_array_3 = image.img_to_array(img_3)
                img_4 = image.load_img(sensor_path+'/4/'+filelist_env4[select_case_env4[i]], target_size=(84,84))  #height-width
                img_array_4 = image.img_to_array(img_4)  
                all_sensor_input_env4[idx_sensor,:, 84*3:84*4,:] = img_array_1/255 
                all_sensor_input_env4[idx_sensor,:, 84*2:84*3,:] = img_array_2/255
                all_sensor_input_env4[idx_sensor,:, 84*1:84*2,:] = img_array_3/255
                all_sensor_input_env4[idx_sensor,:, 84*0:84*1,:] = img_array_4/255    
            batch_input_env4.append(all_sensor_input_env4.copy())
            
            #  get label data 
            #img_index = int(filelist[select_case[i]][:-4])

        batch_input = np.array(batch_input_env3+batch_input_env4) 
        batch_output = np.array(batch_output_env3+batch_output_env4)  
        history = model.fit(x = [batch_input[:,0], batch_input[:,1], batch_input[:,2], batch_input[:,3], 
                                 batch_input[:,4], batch_input[:,5], batch_input[:,6], batch_input[:,7], batch_input[:,8],
                                 batch_matrix1, batch_matrix2], 
                            y = [batch_output[:,0], batch_output[:,1], batch_output[:,2], batch_output[:,3],
                                 batch_output[:,4], batch_output[:,5], batch_output[:,6], batch_output[:,7], batch_output[:,8]],
                            batch_size=batch_size, epochs=1, shuffle = True)
                        #callbacks=[TensorBoard(log_dir='mytensorboard')])
        hist_df = pd.DataFrame(history.history)
        hist_csv_file = 'cnn_history_mix.csv'
        with open(hist_csv_file, mode='a') as f:
            hist_df.to_csv(f) 
        if num_iter % 500 == 100:
            print('save_model')
            model.save('gnn_khop_env34_share.h5')

##################################################################
model = khop_model_distribute()
from tensorflow.keras.optimizers import Adam
init_lr = 3e-5
model.compile(optimizer=Adam(learning_rate=init_lr, clipnorm=3), loss='mse')    
train_sample()