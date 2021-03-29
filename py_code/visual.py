# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 22:03:58 2020

@author: azrael
"""

import tensorflow as tf
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.preprocessing import image
from loc2dir import s_label, sen_angle, s_label_batch, read_target, theta, angle2xy
from resnet import resnet_sensor_network, sensor_cnn
from spektral.layers import GraphConv, GraphAttention
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input, Concatenate, Flatten, Reshape
from spektral.utils import localpooling_filter
#from stage_0_cnn import origin_model

input_shape = (84,84*4,3)

def khop_model_distribute10(num_sensors=10): # input/output = num of sensors 
    gnn_unit = 256
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
    G_h1 = GraphConv(gnn_unit, 'relu')([extract_cnn, sensor_matrix1])
    G_h2 = GraphConv(gnn_unit, 'relu')([extract_cnn, sensor_matrix2])
    G_h3 = GraphConv(gnn_unit, 'relu')([extract_cnn, sensor_matrix3])
    G_h4 = GraphConv(gnn_unit, 'relu')([extract_cnn, sensor_matrix4])
    G_1 = Concatenate(axis=-1)([G_h1, G_h2, G_h3, G_h4])
  
    G_2h1 = GraphConv(gnn_unit, 'relu')([G_1, sensor_matrix1])
    G_2h2 = GraphConv(gnn_unit, 'relu')([G_1, sensor_matrix2])
    G_2h3 = GraphConv(gnn_unit, 'relu')([G_1, sensor_matrix3])
    G_2h4 = GraphConv(gnn_unit, 'relu')([G_1, sensor_matrix4])
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
    output10 = mlp_layer(Flatten()(gnn_output[9]))
    
    model = Model(inputs=[s_input1, s_input2, s_input3, s_input4,
                          s_input5, s_input6, s_input7, s_input8, s_input9,s_input10,
                          sensor_matrix1, sensor_matrix2, sensor_matrix3, sensor_matrix4], 
                  outputs= [output1,output2,output3,output4,
                            output5,output6,output7,output8,output9,output10])
    return model
###############################################################
num_sensors=22
all_sensor_input = np.zeros((num_sensors, 84, 84*4, 3))

z_sensor_loc = np.load('stage2_envs/env_1_loc.npy')
z_ad1,z_ad2,z_ad3,z_ad4 = cal_admatrix_sensor(z_sensor_loc)
for idx_sensor in range(22):
    sensor_path = 'training/sensor_{}'.format(idx_sensor+1)
    img_1 = image.load_img(sensor_path+'/1/1.png', target_size=(84,84))  #height-width
    img_array_1 = image.img_to_array(img_1)
    img_2 = image.load_img(sensor_path+'/2/1.png', target_size=(84,84))  #height-width
    img_array_2 = image.img_to_array(img_2)
    img_3 = image.load_img(sensor_path+'/3/1.png', target_size=(84,84))  #height-width
    img_array_3 = image.img_to_array(img_3)
    img_4 = image.load_img(sensor_path+'/4/1.png', target_size=(84,84))  #height-width
    img_array_4 = image.img_to_array(img_4)               
    all_sensor_input[idx_sensor,:, 84*3:84*4,:] = img_array_1/255
    all_sensor_input[idx_sensor,:, 84*2:84*3,:] = img_array_2/255
    all_sensor_input[idx_sensor,:, 84*1:84*2,:] = img_array_3/255
    all_sensor_input[idx_sensor,:, 84*0:84*1,:] = img_array_4/255
    #sensor_pose_input[idx_sensor] = (sensor_loc[idx_sensor][0], sensor_loc[idx_sensor][2])
#sensor_pose.append(sensor_pose_input.copy()) 
#sensor_pose = np.array(sensor_pose)  
res = att_22.predict([np.expand_dims(all_sensor_input[0], axis=0),  np.expand_dims(all_sensor_input[1], axis=0), 
                      np.expand_dims(all_sensor_input[2], axis=0),  np.expand_dims(all_sensor_input[3], axis=0), 
                      np.expand_dims(all_sensor_input[4], axis=0),  np.expand_dims(all_sensor_input[5], axis=0), 
                      np.expand_dims(all_sensor_input[6], axis=0),  np.expand_dims(all_sensor_input[7], axis=0), 
                      np.expand_dims(all_sensor_input[8], axis=0),  np.expand_dims(all_sensor_input[9], axis=0), 
                      np.expand_dims(all_sensor_input[10], axis=0), np.expand_dims(all_sensor_input[11], axis=0), 
                      np.expand_dims(all_sensor_input[12], axis=0), np.expand_dims(all_sensor_input[13], axis=0), 
                      np.expand_dims(all_sensor_input[14], axis=0), np.expand_dims(all_sensor_input[15], axis=0), 
                      np.expand_dims(all_sensor_input[16], axis=0), np.expand_dims(all_sensor_input[17], axis=0), 
                      np.expand_dims(all_sensor_input[18], axis=0), np.expand_dims(all_sensor_input[19], axis=0), 
                      np.expand_dims(all_sensor_input[20], axis=0), np.expand_dims(all_sensor_input[21], axis=0),             
                      z_ad1, z_ad2, z_ad3, z_ad4])


def att_model_22(num_sensors=22): # input/output = num of sensors 
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
    s_input11 = Input(shape=(input_shape[0], input_shape[1], input_shape[2]))
    s_input12 = Input(shape=(input_shape[0], input_shape[1], input_shape[2]))
    s_input13 = Input(shape=(input_shape[0], input_shape[1], input_shape[2]))
    s_input14 = Input(shape=(input_shape[0], input_shape[1], input_shape[2]))
    s_input15 = Input(shape=(input_shape[0], input_shape[1], input_shape[2]))
    s_input16 = Input(shape=(input_shape[0], input_shape[1], input_shape[2]))
    s_input17 = Input(shape=(input_shape[0], input_shape[1], input_shape[2]))
    s_input18 = Input(shape=(input_shape[0], input_shape[1], input_shape[2]))
    s_input19 = Input(shape=(input_shape[0], input_shape[1], input_shape[2]))
    s_input20 = Input(shape=(input_shape[0], input_shape[1], input_shape[2]))
    s_input21 = Input(shape=(input_shape[0], input_shape[1], input_shape[2]))
    s_input22 = Input(shape=(input_shape[0], input_shape[1], input_shape[2]))
    
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
    extract_cnn11 = s_cnn(s_input11)
    extract_cnn12 = s_cnn(s_input12)
    extract_cnn13 = s_cnn(s_input13)
    extract_cnn14 = s_cnn(s_input14)
    extract_cnn15 = s_cnn(s_input15)
    extract_cnn16 = s_cnn(s_input16)
    extract_cnn17 = s_cnn(s_input17)
    extract_cnn18 = s_cnn(s_input18)
    extract_cnn19 = s_cnn(s_input19)
    extract_cnn20 = s_cnn(s_input20)
    extract_cnn21 = s_cnn(s_input21)
    extract_cnn22 = s_cnn(s_input22)
        
    extract_cnn = Concatenate(axis=1)([extract_cnn1, extract_cnn2, extract_cnn3, 
                                       extract_cnn4, extract_cnn5, extract_cnn6,
                                       extract_cnn7, extract_cnn8, extract_cnn9, 
                                       extract_cnn10,extract_cnn11,extract_cnn12,extract_cnn13,
                                       extract_cnn14,extract_cnn15,extract_cnn16,extract_cnn17,
                                       extract_cnn18,extract_cnn19,extract_cnn20,extract_cnn21,
                                       extract_cnn22
                                       ])
        
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

    G_3h1 = GraphAttention(gnn_unit, activation='selu', dropout_rate=0)([G_2, sensor_matrix1])
    G_3h2 = GraphAttention(gnn_unit, activation='selu', dropout_rate=0)([G_2, sensor_matrix2])
    G_3h3 = GraphAttention(gnn_unit, activation='selu', dropout_rate=0)([G_2, sensor_matrix3])
    G_3h4 = GraphAttention(gnn_unit, activation='selu', dropout_rate=0)([G_2, sensor_matrix4])
    G_3 = Concatenate(axis=-1)([G_3h1, G_3h2, G_3h3, G_3h4])
    
    gnn_output = tf.split(G_3, num_sensors, 1)
        
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
    output11 = mlp_layer(Flatten()(gnn_output[10]))
    output12 = mlp_layer(Flatten()(gnn_output[11]))
    output13 = mlp_layer(Flatten()(gnn_output[12]))
    output14 = mlp_layer(Flatten()(gnn_output[13]))
    output15 = mlp_layer(Flatten()(gnn_output[14]))
    output16 = mlp_layer(Flatten()(gnn_output[15]))
    output17 = mlp_layer(Flatten()(gnn_output[16]))
    output18 = mlp_layer(Flatten()(gnn_output[17]))
    output19 = mlp_layer(Flatten()(gnn_output[18]))
    output20 = mlp_layer(Flatten()(gnn_output[19]))
    output21 = mlp_layer(Flatten()(gnn_output[20]))
    output22 = mlp_layer(Flatten()(gnn_output[21]))
    
    model = Model(inputs=[s_input1, s_input2, s_input3, s_input4,
                          s_input5, s_input6, s_input7, s_input8, s_input9,s_input10, s_input11,
                          s_input12,s_input13,s_input14,s_input15,s_input16,s_input17,s_input18,
                          s_input19,s_input20,s_input21,s_input22,
                          sensor_matrix1, sensor_matrix2, sensor_matrix3, sensor_matrix4], 
                  outputs= [output1, output2, output3, output4,
                            output5, output6, output7, output8, output9, output10, output11,
                            output12,output13,output14,output15,output16,output17, output18,
                            output19,output20,output21,output22
                            ])
    return model
#################################################
def khop_model_distribute12(num_sensors=12): # input/output = num of sensors 
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
    s_input11 = Input(shape=(input_shape[0], input_shape[1], input_shape[2]))
    s_input12 = Input(shape=(input_shape[0], input_shape[1], input_shape[2]))
    
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
    extract_cnn11 = s_cnn(s_input11)
    extract_cnn12 = s_cnn(s_input12)
    
    extract_cnn = Concatenate(axis=1)([extract_cnn1, extract_cnn2, extract_cnn3, 
                                       extract_cnn4, extract_cnn5, extract_cnn6,
                                       extract_cnn7, extract_cnn8, extract_cnn9, 
                                       extract_cnn10,extract_cnn11,extract_cnn12])
        
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
    output10 = mlp_layer(Flatten()(gnn_output[9]))
    output11 = mlp_layer(Flatten()(gnn_output[10]))
    output12 = mlp_layer(Flatten()(gnn_output[11]))
    
    model = Model(inputs=[s_input1, s_input2, s_input3, s_input4, s_input5, 
                          s_input6, s_input7, s_input8, s_input9, s_input10, 
                          s_input11,s_input12,
                          sensor_matrix1, sensor_matrix2, sensor_matrix3, sensor_matrix4], 
                  outputs= [output1,output2,output3,output4,
                            output5,output6,output7,output8,output9,output10, output11, output12])
    return model



def mlp_model():
    input_data = Input(shape=512)
    output1 = Dense(256, activation='selu',  name='mlp_1')(input_data)
    output1 = Dense(64, activation='selu',  name='mlp_2')(output1)
    output1 = Dense(2, activation='linear', name='sensors')(output1)
    
    model = Model(inputs=[input_data], outputs=[output1])
    return model
'''
def change_axis(img, loc):
    env_x, env_z = img.shape
    loc_x = loc[0]
    loc_z = loc[2]
    return (env_z/2 + loc_x, 0, env_x/2 - loc_z)

def plot_angle(sensor_loc, res, label_angle):
    angle_true = []
    angle_pred = []
    for i in range(len(res)):     # i for sensor
        s_true = label_angle[i]
        s_pred = res[i][0]    
        angle_true.append((s_true[0], s_true[1]))
        angle_pred.append((s_pred[0], s_pred[1]))
    return angle_pred, angle_true
import matplotlib as mpl

def draw_arrow(img, all_pred, all_true, sensor_loc, img_num):
    plt.imshow(img, cmap=mpl.colors.ListedColormap('white'))
    for i in range(len(all_pred)):
        pred_x, pred_y = all_pred[i]
        true_x, true_y = all_true[i]
        s_x, s_s, s_y = sensor_loc[i]
        
        plt.annotate('',xy=(s_x+2*true_x, s_y-2*true_y),xytext=(s_x, s_y), 
                     arrowprops=dict(arrowstyle="->",connectionstyle="arc3"))
        plt.annotate('',xy=(s_x+3*pred_x, s_y-3*pred_y),xytext=(s_x, s_y), 
                     arrowprops=dict(arrowstyle="->",connectionstyle="arc3"))
    plt.xticks([])
    plt.yticks([])
    plt.savefig('test/arrow/arrow_{}.jpg'.format(img_num))
    plt.close()
    #return img
'''

def load_env(select_group, env_index):
    if env_index == 3:
        env = np.load('env_3.npy') 
        if select_group == 1:    
            sensor_loc = [(7, 0, 7),  (14, 0, 14), (27, 0, 20), (37, 0, 9),
                              (7, 0, 28), (15, 0, 39), (18, 0, 26), (28, 0, 37), (39, 0, 26)] #env-3-1
            label_path = "target_env3_1.txt"
            ad_matrix = np.load('ad_matrix_env3_1.npy')
            ad_matrix2 = np.load('ad_matrix2_env3_1.npy')
        if select_group == 2:     
            sensor_loc = [(19, 0, 9), (5, 0, 10),  (28, 0, 13), (36, 0, 4),
                              (9, 0, 24), (12, 0, 38), (22, 0, 23), (26, 0, 35), (35, 0, 29)] #env-3-2
            label_path = "target_env3_2.txt"
            ad_matrix = np.load('ad_matrix_env3_2.npy')
            ad_matrix2 = np.load('ad_matrix2_env3_2.npy')
        if select_group == 3:     
            sensor_loc = [(19, 0, 9), (5, 0, 7),  (24, 0, 18), (31, 0, 9),
                              (8, 0, 21), (6, 0, 35), (19, 0, 31), (29, 0, 39), (31, 0, 29)] #env-3-3
            label_path = "target_env3_3.txt"
            ad_matrix = np.load('ad_matrix_env3_3.npy')
            ad_matrix2 = np.load('ad_matrix2_env3_3.npy')
        if select_group == 4:     
            sensor_loc = [(15, 0, 4), (28, 0, 11), (39, 0, 19), (35, 0, 5),
                              (7, 0, 22), (12, 0, 34), (21, 0, 23), (29, 0, 34), (32, 0, 25)] #env-3-4
            label_path = "target_env3_4.txt"
            ad_matrix = np.load('ad_matrix_env3_4.npy')
            ad_matrix2 = np.load('ad_matrix2_env3_4.npy')
        if select_group == 5:     
            sensor_loc = [(16, 0, 0), (7, 0, 10),  (26, 0, 10), (34, 0, 7),
                              (6, 0, 24), (12, 0, 38), (18, 0, 19), (25, 0, 31), (32, 0, 21)] #env-3-5
            label_path = "target_env3_5.txt"
            ad_matrix = np.load('ad_matrix_env3_5.npy')
            ad_matrix2 = np.load('ad_matrix2_env3_5.npy')
        if select_group == 6:     
            sensor_loc = [(18, 0, 0),  (7, 0, 9),   (25, 0, 12), (32, 0, 0),
                              (10, 0, 23), (13, 0, 37), (21, 0, 32), (35, 0, 35), (38, 0, 19)] #env-3-6
            label_path = "target_env3_6.txt"
            ad_matrix = np.load('ad_matrix_env3_6.npy')
            ad_matrix2 = np.load('ad_matrix2_env3_6.npy')
        if select_group == 7:
            sensor_loc = [(16, 0, 3),  (14, 0, 14), (3, 0, 22), (12, 0, 31),
                  (21, 0, 39), (23, 0, 24), (26, 0, 12), (33, 0, 33), (31, 0, 9)] #env-3-7
            label_path = "test_target_3_7.txt"
            ad_matrix = np.load('test_admatrix_env3_7.npy')
            ad_matrix2 = np.load('test_admatrix2_env3_7.npy')

    if env_index == 4:
        env = np.load('env_4.npy')   
        if select_group == 1:     
            sensor_loc = [(12, 0, 1),  (6, 0, 13),  (7, 0, 26), (17, 0, 16),
                              (19, 0, 30), (27, 0, 20), (31, 0, 8), (33, 0, 31), (39, 0, 19)] #env-4-1
            label_path = "target_env4_1.txt"
            ad_matrix = np.load('ad_matrix_env4_1.npy')
            ad_matrix2 = np.load('ad_matrix2_env4_1.npy')
        if select_group == 2:     
            sensor_loc = [(5, 0, 4),   (0, 0, 17),  (4, 0, 31),   (13, 0, 13),
                              (17, 0, 26), (27, 0, 16), (28, 0, 3), (30, 0, 30), (39, 0, 12)] #env-4-2
            label_path = "target_env4_2.txt"
            ad_matrix = np.load('ad_matrix_env4_2.npy')
            ad_matrix2 = np.load('ad_matrix2_env4_2.npy')
        if select_group == 3:     
            sensor_loc = [(10, 0, 13), (0, 0, 24), (7, 0, 37), (22, 0, 19),
                              (21, 0, 33), (29, 0, 13), (26, 0, 7), (35, 0, 32), (39, 0, 3)] #env-4-3
            label_path = "target_env4_3.txt"
            ad_matrix = np.load('ad_matrix_env4_3.npy')
            ad_matrix2 = np.load('ad_matrix2_env4_3.npy')
        if select_group == 4:     
            sensor_loc = [(10, 0, 8),  (17, 0, 21), (3, 0, 39), (16, 0, 34),
                              (29, 0, 29), (29, 0, 19), (23, 0, 8), (39, 0, 39), (39, 0, 21)] #env-4-4
            label_path = "target_env4_4.txt"
            ad_matrix = np.load('ad_matrix_env4_4.npy')
            ad_matrix2 = np.load('ad_matrix2_env4_4.npy')
        if select_group == 5:     
            sensor_loc = [(10, 0, 3), (5, 0, 13), (0, 0, 26), (18, 0, 23),
                              (11, 0, 35), (29, 0, 26), (23, 0, 10), (25, 0, 39), (35, 0, 13)] #env-4-5
            label_path = "target_env4_5.txt"
            ad_matrix = np.load('ad_matrix_env4_5.npy')
            ad_matrix2 = np.load('ad_matrix2_env4_5.npy')
        if select_group == 6:     
            sensor_loc = [(10, 0, 5),  (8, 0, 19),  (2, 0, 31), (15, 0, 28),
                              (18, 0, 10), (25, 0, 17), (29, 0, 6), (34, 0, 28), (38, 0, 15)] #env-4-6
            label_path = "target_env4_6.txt"
            ad_matrix = np.load('ad_matrix_env4_6.npy')
            ad_matrix2 = np.load('ad_matrix2_env4_6.npy')
        if select_group == 7:
            sensor_loc = [(13, 0, 13),  (4, 0, 15), (7, 0, 26), (14, 0, 37),
                  (22, 0, 27), (26, 0, 7), (29, 0, 19), (29, 0, 21), (34, 0, 33)] #env-3-7
            label_path = "test_target_4_7.txt"
            ad_matrix = np.load('test_admatrix_env4_7.npy')
            ad_matrix2 = np.load('test_admatrix2_env4_7.npy')
    if env_index == 5:
        env = np.load('env_5.npy')
        if select_group == 1:     
            sensor_loc = [(11, 0, 8),  (7, 0, 21),  (12, 0, 34), (16, 0, 21),
                              (28, 0, 15), (26, 0, 34), (33, 0, 4), (39, 0, 14), (36, 0, 26)] #env-5-1
            label_path = "target_env5_1.txt"
            ad_matrix = np.load('test_admatrix_env5_1.npy')
            ad_matrix2 = np.load('test_admatrix2_env5_1.npy')
    if env_index == 6:
        env = np.load('env_6.npy')   
        if select_group == 1:     
            sensor_loc = [(4, 0, 0),  (3, 0, 14),  (8, 0, 28), (16, 0, 17),
                              (22, 0, 30), (22, 0, 5), (27, 0, 18), (36, 0, 8), (35, 0, 36)] #env-6-1
            label_path = "target_env6_1.txt"
            ad_matrix = np.load('test_admatrix_env6_1.npy')
            ad_matrix2 = np.load('test_admatrix2_env6_1.npy')
    return env, sensor_loc, label_path, ad_matrix, ad_matrix2
        
testmodel = khop_model_distribute()
testmodel.load_weights('gnn_khop_env34_share2.h5')

batch_size = 100
all_sensors = ['sensor_1', 'sensor_2', 'sensor_3', 'sensor_4',
               'sensor_5', 'sensor_6', 'sensor_7', 'sensor_8', 'sensor_9', 'sensor_10']
num_sensors = len(all_sensors)

select_group = np.random.randint(1,7)
env_index = 3
env, sensor_loc, label_path, ad_matrix, ad_matrix2 = load_env(select_group, env_index)
filePath = 'training_env{}_{}/sensor_1/1'.format(env_index, select_group)
filelist = os.listdir(filePath)
filelist.sort(key = lambda x: int(x[:-4]))
#all_label = s_label(1)
target_label = open(label_path,"r") 
lines = target_label.readlines() 
#select_case = [np.random.randint(len(lines)) for _ in range(batch_size)]
select_case = np.arange(batch_size)

batch_matrix1 = np.zeros((1, num_sensors, num_sensors)) #the first dimension should be 1 for test
batch_matrix2 = np.zeros((1, num_sensors, num_sensors))    
for i in range(1):
    batch_matrix1[i] = localpooling_filter(ad_matrix)
    batch_matrix2[i] = localpooling_filter(ad_matrix2)

#randomly select a batch of sample
#select_case = [np.random.randint(1,len(all_label)) for _ in range(batch_size)] 
batch_input = []
batch_output = s_label_batch(select_group, select_case, env_index)

all_result = []
all_angle_pred = []
all_angle_true = []
for i in range(batch_size):
    sensor_pose =[]
    all_sensor_input = np.zeros((num_sensors, 84, 84*4, 3))
    all_sensor_output = np.zeros((num_sensors, 1))
    #sensor_pose_input = np.zeros((num_sensors, 2))
    for idx_sensor in range(num_sensors):
        sensor_path = 'training_env{}_{}/'.format(env_index, select_group) + all_sensors[idx_sensor]
        img_1 = image.load_img(sensor_path+'/1/'+filelist[select_case[i]], target_size=(84,84))  #height-width
        img_array_1 = image.img_to_array(img_1)
        img_2 = image.load_img(sensor_path+'/2/'+filelist[select_case[i]], target_size=(84,84))  #height-width
        img_array_2 = image.img_to_array(img_2)
        img_3 = image.load_img(sensor_path+'/3/'+filelist[select_case[i]], target_size=(84,84))  #height-width
        img_array_3 = image.img_to_array(img_3)
        img_4 = image.load_img(sensor_path+'/4/'+filelist[select_case[i]], target_size=(84,84))  #height-width
        img_array_4 = image.img_to_array(img_4)               
        all_sensor_input[idx_sensor,:, 84*3:84*4,:] = img_array_1/255
        all_sensor_input[idx_sensor,:, 84*2:84*3,:] = img_array_2/255
        all_sensor_input[idx_sensor,:, 84*1:84*2,:] = img_array_3/255
        all_sensor_input[idx_sensor,:, 84*0:84*1,:] = img_array_4/255 
        #sensor_pose_input[idx_sensor] = (sensor_loc[idx_sensor][0], sensor_loc[idx_sensor][2])
    #sensor_pose.append(sensor_pose_input.copy()) 
    #sensor_pose = np.array(sensor_pose)  
    res = testmodel.predict([np.expand_dims(all_sensor_input[0], axis=0), np.expand_dims(all_sensor_input[1], axis=0), 
                             np.expand_dims(all_sensor_input[2], axis=0), np.expand_dims(all_sensor_input[3], axis=0), 
                             np.expand_dims(all_sensor_input[4], axis=0), np.expand_dims(all_sensor_input[5], axis=0), 
                             np.expand_dims(all_sensor_input[6], axis=0), np.expand_dims(all_sensor_input[7], axis=0), 
                             np.expand_dims(all_sensor_input[8], axis=0), 
                             batch_matrix1, batch_matrix2])
    all_result.append(res)
    batch_input.append(all_sensor_input)

z_env3_1_norm = []
for i in range(100):
    norm = []
    a_pred = all_result[i]
    for j in range(9):
        norm.append((a_pred[j][0][0]/np.sqrt(a_pred[j][0][0]**2+a_pred[j][0][1]**2), 
                    a_pred[j][0][1]/np.sqrt(a_pred[j][0][0]**2+a_pred[j][0][1]**2)))   
    z_env3_1_norm.append(norm)
    
np.save('pred_env{}_{}.npy'.format(env_index, select_group), z_env3_1_norm)
np.save('true_env{}_{}.npy'.format(env_index, select_group), batch_output)
    
    

'''
    angle_pred, angle_true = plot_angle(sensor_loc, res, all_label[i])
    all_angle_pred.append(angle_pred)
    all_angle_true.append(angle_true)
    img_num = select_case[i]
    draw_arrow(angle_pred, angle_true, sensor_loc, img_num)
    #np.save('test/arrow_{}.npy'.format(select_case[i]), bird_img)    

    for idx_sensor in range(num_sensors):
        t_img = all_sensor_input[idx_sensor]
        cv2.putText(t_img, str(res[idx_sensor][0][0]), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0.9, 0, 0), 2)
        img_index = int(filelist[select_case[i]][:-4])
        #batch_output.append(all_label[select_case[i]][0]) 
        cv2.putText(t_img, str(all_label[select_case[i]][idx_sensor]), (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0.9), 2)
        plt.imsave('test/{}_{}.jpg'.format(select_case[i], idx_sensor) ,t_img)
        batch_input.append(all_sensor_input.copy())
        #  get label data 
    '''
batch_input = np.array(batch_input) 

all_angle_loss = []
for i in range(len(all_result)):
    a_true = batch_output[i+1]
    a_pred = all_result[i]
    angle_loss = []
    for j in range(10):
        cur_loss = (a_pred[j][0][0]/np.sqrt(a_pred[j][0][0]**2+a_pred[j][0][1]**2), 
                    a_pred[j][0][1]/np.sqrt(a_pred[j][0][0]**2+a_pred[j][0][1]**2))
        
        a_loss = abs(a_true[j][0]- cur_loss[0])+ abs(a_true[j][1]- cur_loss[1])
        angle_loss.append(a_loss.copy())
    all_angle_loss.append(angle_loss.copy())

num_weird = 0
for i in range(len(all_angle_loss)):
    if all_angle_loss[i][-1] >=0.6:
        num_weird +=1
print(num_weird)

#################################################################  with robot
def att_model_distribute12(num_sensors=10): # input/output = num of sensors 
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
    s_input11 = Input(shape=(input_shape[0], input_shape[1], input_shape[2]))
    s_input12 = Input(shape=(input_shape[0], input_shape[1], input_shape[2]))
    
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
    extract_cnn11 = s_cnn(s_input11)
    extract_cnn12 = s_cnn(s_input12)
    
    extract_cnn = Concatenate(axis=1)([extract_cnn1, extract_cnn2, extract_cnn3, 
                                       extract_cnn4, extract_cnn5, extract_cnn6,
                                       extract_cnn7, extract_cnn8, extract_cnn9, 
                                       extract_cnn10,extract_cnn11,extract_cnn12])
        
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

    G_3h1 = GraphAttention(gnn_unit, activation='selu', dropout_rate=0)([G_2, sensor_matrix1])
    G_3h2 = GraphAttention(gnn_unit, activation='selu', dropout_rate=0)([G_2, sensor_matrix2])
    G_3h3 = GraphAttention(gnn_unit, activation='selu', dropout_rate=0)([G_2, sensor_matrix3])
    G_3h4 = GraphAttention(gnn_unit, activation='selu', dropout_rate=0)([G_2, sensor_matrix4])
    G_3 = Concatenate(axis=-1)([G_3h1, G_3h2, G_3h3, G_3h4])
    
    G_4h1 = GraphAttention(gnn_unit, activation='selu', dropout_rate=0)([G_3, sensor_matrix1])
    G_4h2 = GraphAttention(gnn_unit, activation='selu', dropout_rate=0)([G_3, sensor_matrix2])
    G_4h3 = GraphAttention(gnn_unit, activation='selu', dropout_rate=0)([G_3, sensor_matrix3])
    G_4h4 = GraphAttention(gnn_unit, activation='selu', dropout_rate=0)([G_3, sensor_matrix4])
    G_4 = Concatenate(axis=-1)([G_4h1, G_4h2, G_4h3, G_4h4])

    gnn_output = tf.split(G_4, num_sensors, 1)
        
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
    output11 = mlp_layer(Flatten()(gnn_output[10]))
    output12 = mlp_layer(Flatten()(gnn_output[11]))
    
    model = Model(inputs=[s_input1, s_input2, s_input3, s_input4,
                          s_input5, s_input6, s_input7, s_input8, s_input9, 
                          s_input10,s_input11,s_input12,
                          sensor_matrix1, sensor_matrix2, sensor_matrix3, sensor_matrix4], 
                  outputs= [output1,output2,output3,output4,
                            output5,output6,output7,output8,output9, 
                            output10, output11, output12])
    return model


def att_model_distribute11(num_sensors=11): # input/output = num of sensors 
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
    s_input11 = Input(shape=(input_shape[0], input_shape[1], input_shape[2]))
    
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
    extract_cnn11 = s_cnn(s_input11)
    
    extract_cnn = Concatenate(axis=1)([extract_cnn1, extract_cnn2, extract_cnn3, 
                                       extract_cnn4, extract_cnn5, extract_cnn6,
                                       extract_cnn7, extract_cnn8, extract_cnn9, 
                                       extract_cnn10,extract_cnn11])
        
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

    G_3h1 = GraphAttention(gnn_unit, activation='selu', dropout_rate=0)([G_2, sensor_matrix1])
    G_3h2 = GraphAttention(gnn_unit, activation='selu', dropout_rate=0)([G_2, sensor_matrix2])
    G_3h3 = GraphAttention(gnn_unit, activation='selu', dropout_rate=0)([G_2, sensor_matrix3])
    G_3h4 = GraphAttention(gnn_unit, activation='selu', dropout_rate=0)([G_2, sensor_matrix4])
    G_3 = Concatenate(axis=-1)([G_3h1, G_3h2, G_3h3, G_3h4])
    
    G_4h1 = GraphAttention(gnn_unit, activation='selu', dropout_rate=0)([G_3, sensor_matrix1])
    G_4h2 = GraphAttention(gnn_unit, activation='selu', dropout_rate=0)([G_3, sensor_matrix2])
    G_4h3 = GraphAttention(gnn_unit, activation='selu', dropout_rate=0)([G_3, sensor_matrix3])
    G_4h4 = GraphAttention(gnn_unit, activation='selu', dropout_rate=0)([G_3, sensor_matrix4])
    G_4 = Concatenate(axis=-1)([G_4h1, G_4h2, G_4h3, G_4h4])

    gnn_output = tf.split(G_4, num_sensors, 1)
        
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
    output11 = mlp_layer(Flatten()(gnn_output[10]))
    
    model = Model(inputs=[s_input1, s_input2, s_input3, s_input4,
                          s_input5, s_input6, s_input7, s_input8, s_input9, s_input10, s_input11,
                          sensor_matrix1, sensor_matrix2, sensor_matrix3, sensor_matrix4], 
                  outputs= [output1,output2,output3,output4,
                            output5,output6,output7,output8,output9, output10, output11])
    return model


def collect_sen_obs(env_index, select_group, i, filelist, num_sensors=10):
    all_sensor_input = np.zeros((num_sensors+1, 84, 84*4, 3))
    #all_sensor_output = np.zeros((num_sensors, 2))
    for idx_sensor in range(num_sensors+1):
        sensor_path = 'train_data0911/test_{}_{}_{}/'.format(env_index, num_sensors, select_group) + all_sensors[idx_sensor]   
        img_1 = image.load_img(sensor_path+'/1/'+filelist[i], target_size=(84,84))  #height-width
        img_array_1 = image.img_to_array(img_1)
        img_2 = image.load_img(sensor_path+'/2/'+filelist[i], target_size=(84,84))  #height-width
        img_array_2 = image.img_to_array(img_2)
        img_3 = image.load_img(sensor_path+'/3/'+filelist[i], target_size=(84,84)) #height-width
        img_array_3 = image.img_to_array(img_3)
        img_4 = image.load_img(sensor_path+'/4/'+filelist[i], target_size=(84,84))  #height-width
        img_array_4 = image.img_to_array(img_4)  
        all_sensor_input[idx_sensor,:, 84*3:84*4,:] = img_array_1/255 
        all_sensor_input[idx_sensor,:, 84*2:84*3,:] = img_array_2/255
        all_sensor_input[idx_sensor,:, 84*1:84*2,:] = img_array_3/255
        all_sensor_input[idx_sensor,:, 84*0:84*1,:] = img_array_4/255   
    return all_sensor_input

def cal_admatrix(pos, sensor_loc, mat1, mat2, mat3, mat4, num_sensors=10):
    robot_loc = pos
    ad_matrix1 = np.zeros((num_sensors+1,num_sensors+1))
    ad_matrix2 = np.zeros((num_sensors+1,num_sensors+1))
    ad_matrix3 = np.zeros((num_sensors+1,num_sensors+1))
    ad_matrix4 = np.zeros((num_sensors+1,num_sensors+1))
    #if mat1.shape[0] == 10:
    #    mat1 = mat1[:9, :9]
    #    mat2 = mat2[:9, :9]
    #    mat3 = mat3[:9, :9]
    #    mat4 = mat4[:9, :9]
    ad_matrix1[:num_sensors,:num_sensors] = mat1
    ad_matrix2[:num_sensors,:num_sensors] = mat2
    ad_matrix3[:num_sensors,:num_sensors] = mat3
    ad_matrix4[:num_sensors,:num_sensors] = mat4
    for j, sen in enumerate(sensor_loc):
        if np.sqrt((robot_loc[0]-sen[0])**2+(robot_loc[-1]-sen[-1])**2) <= 15:
            ad_matrix1[-1, j] = 1
            ad_matrix1[j, -1] = 1
    
    for i in range(num_sensors):
        if ad_matrix1[-1,i] == 1:
            index = i
            sensor_nei = ad_matrix1[index,:]
            for j in range(num_sensors-1):
                if sensor_nei[j] == 1 and ad_matrix1[j, -1] == 0:
                    ad_matrix2[-1, j] = 1
                    ad_matrix2[j, -1] = 1

    for i in range(num_sensors):
        if ad_matrix2[-1,i] == 1:
            index = i
            sensor_nei = ad_matrix1[index,:]
            for j in range(num_sensors):
                if sensor_nei[j] == 1 and ad_matrix1[j, -1] == 0 and ad_matrix2[j, -1] == 0:
                    ad_matrix3[-1, j] = 1
                    ad_matrix3[j, -1] = 1

    for i in range(num_sensors):
        if ad_matrix3[-1,i] == 1:
            index = i
            sensor_nei = ad_matrix1[index,:]
            for j in range(num_sensors-1):
                if sensor_nei[j] == 1 and ad_matrix1[j, -1] == 0 and ad_matrix2[j, -1] == 0 and ad_matrix3[j, -1] == 0:
                    ad_matrix4[-1, j] = 1
                    ad_matrix4[j, -1] = 1
                    
    return np.expand_dims(ad_matrix1, axis=0), np.expand_dims(ad_matrix2,axis=0), np.expand_dims(ad_matrix3,axis=0), np.expand_dims(ad_matrix4,axis=0)

def sample_robot0911(select_env, select_group, select_case, all_sensors, num_sensors=10):
    batch_size = len(select_case)

    #select_case = np.arange(1,100)
    sen_loc = np.load('train_data0911/ad_matrix0911/env_{}_{}_{}_loc.npy'.format(select_env, num_sensors, select_group))
    tar_loc = read_target('train_data0911/test_{}_{}_{}.txt'.format(select_env, num_sensors, select_group), select_case)
    robot_loc = read_target('train_data0911/test_robot_{}_{}_{}.txt'.format(select_env, num_sensors, select_group), select_case)
    ad1 = np.load('train_data0911/ad_matrix0911/env_{}_{}_{}_ad1.npy'.format(select_env, num_sensors, select_group))
    ad2 = np.load('train_data0911/ad_matrix0911/env_{}_{}_{}_ad2.npy'.format(select_env, num_sensors, select_group))
    ad3 = np.load('train_data0911/ad_matrix0911/env_{}_{}_{}_ad3.npy'.format(select_env, num_sensors, select_group))
    ad4 = np.load('train_data0911/ad_matrix0911/env_{}_{}_{}_ad4.npy'.format(select_env, num_sensors, select_group))
    env = np.load('train_data0911/env_map/env_{}.npy'.format(select_env))
                
    batch_label = []
    all_sensor_loc = []
    all_target_loc = []
    for i in range(batch_size):
        t_x, t_z = tar_loc[i]
        #t_x, t_y, t_z = change_axis(env, (label_x, 0, label_z))    
        sensor_loc2 = list(sen_loc)
        sensor_loc2.append(np.array((round(robot_loc[i][0]), round(robot_loc[i][1]))))
        all_sensor_loc.append(robot_loc[i])
        all_target_loc.append(tar_loc[i])
        sensor_direction = []
        for j in range(len(all_sensors)):
            #s_x, s_y, s_z = change_axis(env, sensor_loc[j])
            s_x, s_z = sensor_loc2[j]
            #s_path = AStarSearch(env, (round(s_x), round(s_z)), (round(t_x), round(t_z)))
            s_path = theta(env, (round(s_x), round(s_z)), (round(t_x), round(t_z)))
            if s_path[0] == s_path[1]:
                print('num_group:', select_group, '   num_case:', select_case[i], '   num_sensor:', j)
                #print('target_loc:', (label_x, 0, label_z))
                s_angle = angle2xy((s_x, s_z), (t_x, t_z))
            else:
                s_angle = angle2xy(s_path[0], s_path[1])
            #s_relative = (s_path[1][0]-s_path[0][0], s_path[1][1]-s_path[0][1])
            #s_angle = loc2angle(s_relative)
            sensor_direction.append(s_angle)
        batch_label.append(sensor_direction)
    return batch_label, all_sensor_loc, all_target_loc, ad1, ad2, ad3, ad4, sen_loc

all_sensors = ['sensor_1', 'sensor_2', 'sensor_3', 'sensor_4',
               'sensor_5', 'sensor_6', 'sensor_7', 'sensor_8', 
               'sensor_9', 'sensor_10', 'sensor_11', 'robot']
select_case = np.arange(100)
num_sensors = 10
select_group = 1
env_index = 1
filePath = 'train_data0911/test_{}_{}_{}/sensor_1/1'.format(env_index, num_sensors, select_group)
filelist = os.listdir(filePath)
filelist.sort(key = lambda x: int(x[:-4]))
#filelist = filelist[1:]

batch_output, sensor_locs, target_locs, ad_matrix1, ad_matrix2, ad_matrix3, ad_matrix4, sensor_loc = sample_robot0911(env_index, select_group, select_case, all_sensors, num_sensors)
batch_input = []
all_result = []
all_angle_pred = []
all_angle_true = []
if_att = True
for i in range(batch_size):
    sensor_pose =[]
    all_sensor_input = collect_sen_obs(env_index, select_group, i, filelist, num_sensors)
    all_sensor_output = np.zeros((num_sensors+1, 1))
    batch_matrix1 = np.zeros((1, num_sensors+1, num_sensors+1)) #the first dimension should be 1 for test
    batch_matrix2 = np.zeros((1, num_sensors+1, num_sensors+1))
    batch_matrix3 = np.zeros((1, num_sensors+1, num_sensors+1))
    batch_matrix4 = np.zeros((1, num_sensors+1, num_sensors+1))
    
    ad_mat1, ad_mat2, ad_mat3, ad_mat4 = cal_admatrix(sensor_locs[i], sensor_loc, ad_matrix1, ad_matrix2, ad_matrix3, ad_matrix4, num_sensors)
    
    if if_att:
        batch_matrix1[0] = ad_mat1
        batch_matrix2[0] = ad_mat2
        batch_matrix3[0] = ad_mat3
        batch_matrix4[0] = ad_mat4
    else:
        batch_matrix1[0] = localpooling_filter(ad_mat1)
        batch_matrix2[0] = localpooling_filter(ad_mat2)
        batch_matrix3[0] = localpooling_filter(ad_mat3)
        batch_matrix4[0] = localpooling_filter(ad_mat4)
       
    res = testmodel.predict([np.expand_dims(all_sensor_input[0], axis=0), np.expand_dims(all_sensor_input[1], axis=0), 
                             np.expand_dims(all_sensor_input[2], axis=0), np.expand_dims(all_sensor_input[3], axis=0), 
                             np.expand_dims(all_sensor_input[4], axis=0), np.expand_dims(all_sensor_input[5], axis=0), 
                             np.expand_dims(all_sensor_input[6], axis=0), np.expand_dims(all_sensor_input[7], axis=0), 
                             np.expand_dims(all_sensor_input[8], axis=0), np.expand_dims(all_sensor_input[9], axis=0),
                             np.expand_dims(all_sensor_input[10], axis=0), #np.expand_dims(all_sensor_input[11], axis=0),
                             batch_matrix1, batch_matrix2, batch_matrix3, batch_matrix4])
    all_result.append(res)
    batch_input.append(all_sensor_input)
 
np.save('pred_robot_1_1.npy', all_result)
np.save('true_robot_1_1.npy', batch_output)    

for kk in range(1,4):
    all_result = np.load('res1001_origin/new_data_layer3/pred_robot_1_{}.npy'.format(kk))
    batch_output = np.load('res1001_origin/new_data_layer3/true_robot_1_{}.npy'.format(kk))
    
    all_angle_loss = []
    for i in range(len(all_result)):
        a_true = batch_output[i]
        a_pred = all_result[i]
        angle_loss = []
        for j in range(num_sensors+1):
            cur_loss = (a_pred[j][0][0]/np.sqrt(a_pred[j][0][0]**2+a_pred[j][0][1]**2), 
                        a_pred[j][0][1]/np.sqrt(a_pred[j][0][0]**2+a_pred[j][0][1]**2))
            
            a_loss = abs(a_true[j][0]- cur_loss[0])+ abs(a_true[j][1]- cur_loss[1])
            angle_loss.append(a_loss.copy())
        all_angle_loss.append(angle_loss.copy())
    
    z_loss = np.zeros((1,10))
    for i in range(100):
        for j in range(10):    
            z_loss[0][j] += all_angle_loss[i][j]
    exec('z_loss_3_{} = z_loss.copy()'.format(kk))

num_weird = 0
z_weird_index = []
for i in range(len(all_angle_loss)):
    if all_angle_loss[i][-1] >=0.6:
        num_weird +=1
        z_weird_index.append(i)
print(num_weird)
    
bad_sensor = []
for i in range(num_sensors+1):
    num_bad = 0
    for j in range(len(all_angle_loss)):
        if all_angle_loss[j][i] >=0.6:
            num_bad += 1
    bad_sensor.append(num_bad)

loss_sum = []
for i in range(len(all_angle_loss)):
    loss_sum.append(np.sum(all_angle_loss[i][:9]))
np.sum(loss_sum)

z_p1= np.reshape(z_p1, (100,11,2))
z_p7= np.reshape(z_p7, (100,12,2))

z_p1 = np.load('pred_robot_1_1.npy')
z_t1 = np.load('true_robot_1_1.npy')
z_p2 = np.load('pred_robot_1_2.npy')
z_t2 = np.load('true_robot_1_2.npy')
z_p3 = np.load('pred_robot_1_3.npy')
z_t3 = np.load('true_robot_1_3.npy')
z_p4 = np.load('pred_robot_2_1.npy')
z_t4 = np.load('true_robot_2_1.npy')
z_p5 = np.load('pred_robot_2_2.npy')
z_t5 = np.load('true_robot_2_2.npy')
z_p6 = np.load('pred_robot_2_3.npy')
z_t6 = np.load('true_robot_2_3.npy')
z_p7 = np.load('pred_robot_3_1.npy')
z_t7 = np.load('true_robot_3_1.npy')
z_p8 = np.load('pred_robot_3_2.npy')
z_t8 = np.load('true_robot_3_2.npy')
z_p9 = np.load('pred_robot_3_3.npy')
z_t9 = np.load('true_robot_3_3.npy')

z_p_all = []
z_t_all = []
for i in range(1,10):
    exec('z_p_all.append(z_p{})'.format(i)) 
    exec('z_t_all.append(z_t{})'.format(i)) 
    
for kk in range(9):
    z_group2 = [np.random.randint(9) for _ in range(100)]
    z_index2 = [np.random.randint(100) for _ in range(100)]
    z_group3 = [np.random.randint(9) for _ in range(100)]
    z_index3 = [np.random.randint(100) for _ in range(100)]
    z_group4 = [np.random.randint(9) for _ in range(100)]
    z_index4 = [np.random.randint(100) for _ in range(100)]
    zz_pred = []
    zz_true = []
    for i in range(100):
        pred_1 = z_p_all[kk][i][:9]
        pred_2 = np.concatenate((pred_1, np.expand_dims(z_p_all[z_group2[i]][z_index2[i]][0],axis=0)))
        pred_3 = np.concatenate((pred_2, np.expand_dims(z_p_all[z_group3[i]][z_index3[i]][0],axis=0)))
        pred_4 = np.concatenate((pred_3, np.expand_dims(z_p_all[z_group4[i]][z_index4[i]][0],axis=0)))
    
        true_1 = z_t_all[kk][i][:9]
        true_2 = np.concatenate((true_1, np.expand_dims(z_t_all[z_group2[i]][z_index2[i]][0],axis=0)))
        true_3 = np.concatenate((true_2, np.expand_dims(z_t_all[z_group3[i]][z_index3[i]][0],axis=0)))
        true_4 = np.concatenate((true_3, np.expand_dims(z_t_all[z_group4[i]][z_index4[i]][0],axis=0)))
        
        for j in range(10):
            pp_1 = pred_2[j][0][0]
            pp_2 = pred_2[j][0][1]
            pp_all = np.sqrt(pp_1**2+pp_2**2)
            pred_2[j][0][0] = pp_1/pp_all
            pred_2[j][0][1] = pp_2/pp_all
        
        #true_9 = z_res_true[z_group[i]][z_index[i]]  
        if kk >= 6:
            zz_pred.append(pred_4.copy())
            zz_true.append(true_4.copy())
        else:    
            zz_pred.append(pred_3.copy())
            zz_true.append(true_3.copy())

    np.save('res1001/pred_robot_{}_{}.npy'.format(int(kk/3)+1, kk%3), zz_pred)
    np.save('res1001/true_robot_{}_{}.npy'.format(int(kk/3)+1, kk%3), zz_true)    

z_new1 = np.load('res1001/new_data_layer4/pred_robot_1_3.npy')
z_new1 = np.reshape(z_new1, (100,10,2))
z_ne1 = np.load('res1001/new_data_layer4/true_robot_1_3.npy')


import matplotlib as mpl
matplotlib.use('Agg')
from matplotlib.colors import ListedColormap,LinearSegmentedColormap
from matplotlib.pyplot import arrow
clist=['green','red', 'blue','black']
newcmp = LinearSegmentedColormap.from_list('chaos',clist)

env = np.load('new_env_1.npy')
for kk in range(100):
    t1=draw_arrow(env, z_p1[kk], z_t1[kk], batch_output[kk], sensor_loc, sensor_locs[kk], target_locs[kk],kk)

def draw_arrow(img, sen_pred, sen_true, all_true, sensor_loc, robot_loc, target_loc, img_num):
    img = env.astype(float)
    img[img==0] = None
    for sen in sensor_loc:
        img[sen[0]][sen[1]] = 0.7
    img[round(robot_loc[0])][round(robot_loc[-1])] = 0.3
    img[round(target_loc[0])][round(target_loc[-1])] = 0.1
    sen_error = []
    sensor_loc = list(sensor_loc)
    sensor_loc.append(robot_loc)
    all_p = []
    fig, axs = plt.subplots()
    for j in range(len(sen_pred)):
        arrow(round(sensor_loc[j][1]), round(sensor_loc[j][0]), all_true[j][1]*2, all_true[j][0]*2, head_width=0.2, color='g')
        sen_error.append((sen_true[j][0]-sen_pred[j][0], sen_true[j][1]-sen_pred[j][1]))
        z1, z2 = all_true[j][0]+sen_error[j][0], all_true[j][1]+sen_error[j][1]
        #print('ori:', z1,z2)
        z_sum = np.sqrt(z1**2+z2**2)
        z1, z2 = z1/z_sum, z2/z_sum
        #print(sensor_loc[j])
        #print('norm:', z1,z2)
        arrow(round(sensor_loc[j][1]), round(sensor_loc[j][0]), z2*2, z1*2, head_width=0.2, color='y')
        all_p.append((z1,z2))
    plt.imshow(img, cmap=newcmp)
    plt.savefig('images/arrow_{}.jpg'.format(img_num))
    return all_p
    
    
arrow(round(sensor_loc[j][1]), round(sensor_loc[j][0]), all_true[j][1]*2, all_true[j][0]*2, head_width=0.2, color='g')
plt.imshow(img, cmap=newcmp)            
    
select_group = 1
env_index = 1
batch_output, sensor_locs, target_locs, ad_matrix1, ad_matrix2, ad_matrix3, ad_matrix4, sensor_loc = sample_robot0911(env_index, select_group, select_case, all_sensors, num_sensors)
    


env1 = np.load('new_env_1.npy')
env2 = np.load('new_env_2.npy')
env3 = np.load('new_env_3.npy')
weird_t1 = np.load('res1001/dyna_data_layer3/true_robot_3_3.npy')
weird_p1 = np.load('res1001/dyna_data_layer3/pred_robot_3_3.npy')
batch_output, sensor_locs, target_locs, ad_matrix1, ad_matrix2, ad_matrix3, ad_matrix4, sensor_loc = sample_robot0911(env_index, select_group, select_case, all_sensors, num_sensors)


####################################################################
z_p1 = np.load('pred_new_1.npy')
z_p2 = np.load('pred_new_2.npy')
z_p3 = np.load('pred_new_3.npy')
z_p4 = np.load('pred_new_4.npy')
z_p5 = np.load('pred_new_5.npy')
z_p6 = np.load('pred_new_6.npy')
z_p7 = np.load('pred_new_7.npy')
z_p8 = np.load('pred_new_8.npy')
z_p9 = np.load('pred_new_9.npy')
z_p10 = np.load('pred_new_10.npy')
z_p11 = np.load('pred_new_11.npy')
z_p12 = np.load('pred_new_12.npy')

z_t12 = np.load('true_new_12.npy')
z_t11 = np.load('true_new_11.npy')
z_t10 = np.load('true_new_10.npy')
z_t9 = np.load('true_new_9.npy')
z_t8 = np.load('true_new_8.npy')
z_t7 = np.load('true_new_7.npy')
z_t6 = np.load('true_new_6.npy')
z_t5 = np.load('true_new_5.npy')
z_t4 = np.load('true_new_4.npy')
z_t3 = np.load('true_new_3.npy')
z_t2 = np.load('true_new_2.npy')
z_t1 = np.load('true_new_1.npy')

all_result = z_p1
batch_output = z_t1

z_p2[7,:,:,:] = z_p5[2,:,:,:]
z_t2[7,:,:] = z_t5[2,:,:]



sum_loss = []
s1_loss, s2_loss, s3_loss, s4_loss, s5_loss, s6_loss, s7_loss, s8_loss, s9_loss, s10_loss = 0,0,0,0,0,0,0,0,0,0
for i in range(len(all_angle_loss)):
    s1_loss += all_angle_loss[i][0]
    s2_loss += all_angle_loss[i][1]
    s3_loss += all_angle_loss[i][2]
    s4_loss += all_angle_loss[i][3]
    s5_loss += all_angle_loss[i][4]
    s6_loss += all_angle_loss[i][5]
    s7_loss += all_angle_loss[i][6]
    s8_loss += all_angle_loss[i][7]
    s9_loss += all_angle_loss[i][8]
    s10_loss += all_angle_loss[i][9]
    
###################################################################
z_p1 = np.load('res0911/pred_robot_1_1.npy')
z_p2 = np.load('res0911/pred_robot_2_1.npy')
z_p3 = np.load('res0911/pred_robot_3_1.npy')
z_p4 = np.load('res0911/pred_robot_4_1.npy')
z_p5 = np.load('res0911/pred_robot_5_1.npy')
z_p6 = np.load('res0911/pred_robot_6_1.npy')
z_p7 = np.load('res0911/pred_robot_7_9_1.npy')
z_p8 = np.load('res0911/pred_robot_8_1.npy')
z_p9 = np.load('res0911/pred_robot_9_1.npy')
z_p10 = np.load('res0911/pred_robot_10_1.npy')
z_p11 = np.load('res0911/pred_robot_11_1.npy')
z_p12 = np.load('res0911/pred_robot_12_1.npy')
z_p13 = np.load('res0911/pred_robot_13_1.npy')
z_p14 = np.load('res0911/pred_robot_14_1.npy')
z_p15 = np.load('res0911/pred_robot_15_1.npy')
z_p16 = np.load('res0911/pred_robot_16_1.npy')
z_p17 = np.load('res0911/pred_robot_17_1.npy')
z_p18 = np.load('res0911/pred_robot_18_1.npy')

z_p7 = np.reshape(z_p7, (100,10,2))
z_p14 = np.reshape(z_p14, (100,10,2))
z_p15 = np.reshape(z_p15, (100,10,2))
z_p18 = np.reshape(z_p18, (100,10,2))

z_t18 = np.load('res0911/true_robot_18_1.npy')
z_t17 = np.load('res0911/true_robot_17_1.npy')
z_t16 = np.load('res0911/true_robot_16_1.npy')
z_t15 = np.load('res0911/true_robot_15_1.npy')
z_t14 = np.load('res0911/true_robot_14_1.npy')
z_t13 = np.load('res0911/true_robot_13_1.npy')
z_t12 = np.load('res0911/true_robot_12_1.npy')
z_t11 = np.load('res0911/true_robot_11_1.npy')
z_t10 = np.load('res0911/true_robot_10_1.npy')
z_t9 = np.load('res0911/true_robot_9_1.npy')
z_t8 = np.load('res0911/true_robot_8_1.npy')
z_t7 = np.load('res0911/true_robot_7_9_1.npy')
z_t6 = np.load('res0911/true_robot_6_1.npy')
z_t5 = np.load('res0911/true_robot_5_1.npy')
z_t4 = np.load('res0911/true_robot_4_1.npy')
z_t3 = np.load('res0911/true_robot_3_1.npy')
z_t2 = np.load('res0911/true_robot_1_1.npy')
z_t1 = np.load('res0911/true_robot_1_1.npy')

z_res_pred = []
for i in range(1,19):
    exec('z_res_pred.append(z_p{})'.format(i))
    
z_res_true = []
for i in range(1,19):
    exec('z_res_true.append(z_t{})'.format(i))



z_p7_1 = np.load('pred_skip_robot_7_9_1.npy')
z_t7_1 = np.load('true_skip_robot_7_9_1.npy')
z_p7_2 = np.load('pred_skip_robot_7_9_2.npy')
z_t7_2 = np.load('true_skip_robot_7_9_2.npy')
z_p7_3 = np.load('pred_skip_robot_7_9_3.npy')
z_t7_3 = np.load('true_skip_robot_7_9_3.npy')
z_p7_4 = np.load('pred_skip_robot_7_9_4.npy')
z_t7_4 = np.load('true_skip_robot_7_9_4.npy')

z_p7_1 = np.reshape(z_p7_1, (100,10,2))
z_p7_2 = np.reshape(z_p7_2, (100,10,2))
z_p7_3 = np.reshape(z_p7_3, (100,10,2))
z_p7_4 = np.reshape(z_p7_4, (100,10,2))


z_res_pred7 = []
for i in range(1,5):
    exec('z_res_pred7.append(z_p7_{})'.format(i))
    
z_res_true7 = []
for i in range(1,5):
    exec('z_res_true7.append(z_t7_{})'.format(i))


z_env = 7

z_group = [np.random.randint(4, 18) for _ in range(100)]
z_index = [np.random.randint(100) for _ in range(100)]
z_app = [np.random.randint(9, 10) for _ in range(100)]
z_env2 = [np.random.randint(4, 18) for _ in range(100)]
z_index2 = [np.random.randint(100) for _ in range(100)]
z_app2 = [np.random.randint(9, 10) for _ in range(100)] 
z_env3 = [np.random.randint(4, 18) for _ in range(100)]
z_index3 = [np.random.randint(100) for _ in range(100)]
z_app3 = [np.random.randint(9, 10) for _ in range(100)] 
z_env4 = [np.random.randint(4, 18) for _ in range(100)]
z_index4 = [np.random.randint(100) for _ in range(100)]
z_app4 = [np.random.randint(9, 10) for _ in range(100)] 
z_env5 = [np.random.randint(4, 18) for _ in range(100)]
z_index5 = [np.random.randint(100) for _ in range(100)]
z_app5 = [np.random.randint(9, 10) for _ in range(100)] 
z_env6 = [np.random.randint(4, 18) for _ in range(100)]
z_index6 = [np.random.randint(100) for _ in range(100)]
z_app6 = [np.random.randint(9, 10) for _ in range(100)] 
z_env7 = [np.random.randint(4, 18) for _ in range(100)]
z_index7 = [np.random.randint(100) for _ in range(100)]
z_app7 = [np.random.randint(9, 10) for _ in range(100)] 
z_env8 = [np.random.randint(4, 18) for _ in range(100)]
z_index8 = [np.random.randint(100) for _ in range(100)]
z_app8 = [np.random.randint(9, 10) for _ in range(100)] 
z_env9 = [np.random.randint(4, 18) for _ in range(100)]
z_index9 = [np.random.randint(100) for _ in range(100)]
z_app9 = [np.random.randint(9, 10) for _ in range(100)] 
z_env10 = [np.random.randint(4, 18) for _ in range(100)]
z_index10 = [np.random.randint(100) for _ in range(100)]
z_app10 = [np.random.randint(9, 10) for _ in range(100)]
z_env10 = [np.random.randint(4, 18) for _ in range(100)]
z_index10 = [np.random.randint(100) for _ in range(100)]
z_app10 = [np.random.randint(9, 10) for _ in range(100)]
z_env11 = [np.random.randint(4, 18) for _ in range(100)]
z_index11 = [np.random.randint(100) for _ in range(100)]
z_app11 = [np.random.randint(9, 10) for _ in range(100)]
z_env12 = [np.random.randint(4, 18) for _ in range(100)]
z_index12 = [np.random.randint(100) for _ in range(100)]
z_app12 = [np.random.randint(9, 10) for _ in range(100)]

zz_pred = []
zz_true = []

for i in range(100):
    pred_1 = np.expand_dims(z_res_pred[z_env[i]][z_index[i]][z_app[i]], axis=0)
    pred_2 = np.concatenate((np.expand_dims(z_res_pred[z_env2[i]][z_index2[i]][z_app2[i]], axis=0), pred_1), axis=0 )
    pred_3 = np.concatenate((np.expand_dims(z_res_pred[z_env3[i]][z_index3[i]][z_app3[i]], axis=0), pred_2), axis=0 )
    pred_4 = np.concatenate((np.expand_dims(z_res_pred[z_env4[i]][z_index4[i]][z_app4[i]], axis=0), pred_3), axis=0 )
    pred_5 = np.concatenate((np.expand_dims(z_res_pred[z_env5[i]][z_index5[i]][z_app5[i]], axis=0), pred_4), axis=0 )
    pred_6 = np.concatenate((np.expand_dims(z_res_pred[z_env6[i]][z_index6[i]][z_app6[i]], axis=0), pred_5), axis=0 )
    pred_7 = np.concatenate((np.expand_dims(z_res_pred[z_env7[i]][z_index7[i]][z_app7[i]], axis=0), pred_6), axis=0 )
    pred_8 = np.concatenate((np.expand_dims(z_res_pred[z_env8[i]][z_index8[i]][z_app8[i]], axis=0), pred_7), axis=0 )
    pred_9 = np.concatenate((np.expand_dims(z_res_pred[z_env9[i]][z_index9[i]][z_app9[i]], axis=0), pred_8), axis=0 )
    pred_10 = np.concatenate((np.expand_dims(z_res_pred[z_env10[i]][z_index10[i]][z_app10[i]], axis=0), pred_9), axis=0 )
    pred_11 = np.concatenate((np.expand_dims(z_res_pred[z_env11[i]][z_index11[i]][z_app11[i]], axis=0), pred_10), axis=0 )
    pred_12 = np.concatenate((np.expand_dims(z_res_pred[z_env12[i]][z_index12[i]][z_app12[i]], axis=0), pred_11), axis=0 )
    
    true_1 = np.expand_dims(z_res_true[z_env[i]][z_index[i]][z_app[i]], axis=0)
    true_2 = np.concatenate((np.expand_dims(z_res_true[z_env2[i]][z_index2[i]][z_app2[i]], axis=0), true_1), axis=0 )
    true_3 = np.concatenate((np.expand_dims(z_res_true[z_env3[i]][z_index3[i]][z_app3[i]], axis=0), true_2), axis=0 )
    true_4 = np.concatenate((np.expand_dims(z_res_true[z_env4[i]][z_index4[i]][z_app4[i]], axis=0), true_3), axis=0 )
    true_5 = np.concatenate((np.expand_dims(z_res_true[z_env5[i]][z_index5[i]][z_app5[i]], axis=0), true_4), axis=0 )
    true_6 = np.concatenate((np.expand_dims(z_res_true[z_env6[i]][z_index6[i]][z_app6[i]], axis=0), true_5), axis=0 )
    true_7 = np.concatenate((np.expand_dims(z_res_true[z_env7[i]][z_index7[i]][z_app7[i]], axis=0), true_6), axis=0 )
    true_8 = np.concatenate((np.expand_dims(z_res_true[z_env8[i]][z_index8[i]][z_app8[i]], axis=0), true_7), axis=0 )
    true_9 = np.concatenate((np.expand_dims(z_res_true[z_env9[i]][z_index9[i]][z_app9[i]], axis=0), true_8), axis=0 )
    true_10 = np.concatenate((np.expand_dims(z_res_true[z_env10[i]][z_index10[i]][z_app10[i]], axis=0), true_9), axis=0 )
    true_11 = np.concatenate((np.expand_dims(z_res_true[z_env11[i]][z_index11[i]][z_app11[i]], axis=0), true_10), axis=0 )
    true_12 = np.concatenate((np.expand_dims(z_res_true[z_env12[i]][z_index12[i]][z_app12[i]], axis=0), true_11), axis=0 )
    
    #true_9 = z_res_true[z_group[i]][z_index[i]]    
    zz_pred.append(pred_11)
    zz_true.append(true_11)

np.save('env7_0914/pred_robot_7_10_1.npy', zz_pred)
np.save('env7_0914/true_robot_7_10_1.npy', zz_true)

zz_pred = []
zz_true = []
z_env = [np.random.randint(4, 18) for _ in range(100)]
z_index = [np.random.randint(100) for _ in range(100)]
z_env2 = [np.random.randint(4, 18) for _ in range(100)]
z_index2 = [np.random.randint(100) for _ in range(100)]
z_app2 = [np.random.randint(8) for _ in range(100)] 
z_env3 = [np.random.randint(4, 18) for _ in range(100)]
z_index3 = [np.random.randint(100) for _ in range(100)]
z_app3 = [np.random.randint(8) for _ in range(100)] 
z_env4 = [np.random.randint(4, 18) for _ in range(100)]
z_index4 = [np.random.randint(100) for _ in range(100)]
z_app4 = [np.random.randint(8) for _ in range(100)] 
z_env5 = [np.random.randint(4, 18) for _ in range(100)]
z_index5 = [np.random.randint(100) for _ in range(100)]
z_app5 = [np.random.randint(8) for _ in range(100)] 
for i in range(100):
    pred_9 = z_res_pred[z_env[i]][z_index[i]]
    pred_10 = np.concatenate((np.expand_dims(z_res_pred[z_env2[i]][z_index2[i]][z_app2[i]], axis=0), pred_9), axis=0 )
    pred_11 = np.concatenate((np.expand_dims(z_res_pred[z_env3[i]][z_index3[i]][z_app3[i]], axis=0), pred_10), axis=0 )
    pred_12 = np.concatenate((np.expand_dims(z_res_pred[z_env4[i]][z_index4[i]][z_app4[i]], axis=0), pred_11), axis=0 )
    pred_13 = np.concatenate((np.expand_dims(z_res_pred[z_env5[i]][z_index5[i]][z_app5[i]], axis=0), pred_12), axis=0 )

    true_9 = z_res_true[z_env[i]][z_index[i]]
    true_10 = np.concatenate((np.expand_dims(z_res_true[z_env2[i]][z_index2[i]][z_app2[i]], axis=0), true_9), axis=0 )
    true_11 = np.concatenate((np.expand_dims(z_res_true[z_env3[i]][z_index3[i]][z_app3[i]], axis=0), true_10), axis=0 )     
    true_12 = np.concatenate((np.expand_dims(z_res_true[z_env4[i]][z_index4[i]][z_app4[i]], axis=0), true_11), axis=0 ) 
    true_13 = np.concatenate((np.expand_dims(z_res_true[z_env5[i]][z_index5[i]][z_app5[i]], axis=0), true_12), axis=0 ) 
    
    zz_pred.append(pred_13)
    zz_true.append(true_13)

np.save('res0911/11/pred_env1_sensor11.npy', zz_pred)
np.save('res0911/11/true_env1_sensor11.npy', zz_true)


num_sensors = 10    
all_angle_loss = []
for i in range(100):
    a_true = zz_true[i]
    a_pred = zz_pred[i]
    angle_loss = []
    for j in range(num_sensors+1):
        cur_loss = (a_pred[j][0]/np.sqrt(a_pred[j][0]**2+a_pred[j][1]**2), 
                    a_pred[j][1]/np.sqrt(a_pred[j][0]**2+a_pred[j][1]**2))
        
        a_loss = abs(a_true[j][0]- cur_loss[0])+ abs(a_true[j][1]- cur_loss[1])
        angle_loss.append(a_loss.copy())
    all_angle_loss.append(angle_loss.copy())

num_weird = 0
z_weird_index = []
for i in range(len(all_angle_loss)):
    if all_angle_loss[i][-1] >=1.6:
        num_weird +=1
        z_weird_index.append(i)
print(num_weird)
