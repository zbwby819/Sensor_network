import copy
import imageio
import datetime
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from resnet import resnet_sensor_network, sensor_cnn
from spektral.layers import GraphConv, GraphAttention
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input, Concatenate, Flatten, Reshape
from tensorflow.keras.preprocessing import image
from spektral.utils import localpooling_filter
from loc2dir import theta
import tensorflow_probability as tfp
from a_star import AStarSearch

tfd = tfp.distributions

#tf.keras.backend.set_floatx('float64')

action_dim = 2
all_sensors = ['sensor_1', 'sensor_2', 'sensor_3', 'sensor_4',
               'sensor_5', 'sensor_6', 'sensor_7', 'sensor_8',
               'sensor_9']

ZERO_TOLERANCE = 1e-10
input_shape = (84, 84*4, 3)     

def mlp_model():
    input_data = Input(shape=1024)
    output1 = Dense(256, activation='selu',  name='mlp_1')(input_data)
    output1 = Dense(64, activation='selu',  name='mlp_2')(output1)
    output1 = Dense(2, activation='linear', name='sensors')(output1)
    model = Model(inputs=[input_data], outputs=[output1])
    return model

def cnn_model(): # input/output = num of sensors 
    input_shape = (84, 84*4, 3)        
    #sensor_matrix3 = Input(shape=(num_sensors, num_sensors))
    r_input = Input(shape=(input_shape[0], input_shape[1], input_shape[2]))

    s_cnn = sensor_cnn(input_shape, repetitions = [2,2,2,2])
    robot_cnn = s_cnn(r_input)
    
    a_output = Dense(256, activation='selu', name='actor_mlp_1')(Flatten()(robot_cnn))
    a_output = Dense(64, activation='selu', name='actor_mlp_2')(a_output)
    a_output = Dense(action_dim, activation='linear', name='actor_output')(a_output)
    
    c_output = Dense(256, activation='selu', name='critic_mlp_1')(Flatten()(robot_cnn))
    c_output = Dense(64, activation='selu', name='critic_mlp_2')(c_output)
    c_output = Dense(1, activation='linear', name='critic_output')(c_output)
    
    model = Model(inputs=[r_input], 
                  outputs= [a_output, c_output])
    return model

def actor_net(num_sensors=10): # input/output = num of sensors 
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

    G_3h1 = GraphAttention(gnn_unit, activation='selu', dropout_rate=0)([G_2, sensor_matrix1])
    G_3h2 = GraphAttention(gnn_unit, activation='selu', dropout_rate=0)([G_2, sensor_matrix2])
    G_3h3 = GraphAttention(gnn_unit, activation='selu', dropout_rate=0)([G_2, sensor_matrix3])
    G_3h4 = GraphAttention(gnn_unit, activation='selu', dropout_rate=0)([G_2, sensor_matrix4])
    G_3 = Concatenate(axis=-1)([G_3h1, G_3h2, G_3h3, G_3h4])
    
    #G_4h1 = GraphAttention(gnn_unit, activation='selu', dropout_rate=0)([G_3, sensor_matrix1])
    #G_4h2 = GraphAttention(gnn_unit, activation='selu', dropout_rate=0)([G_3, sensor_matrix2])
    #G_4h3 = GraphAttention(gnn_unit, activation='selu', dropout_rate=0)([G_3, sensor_matrix3])
    #G_4h4 = GraphAttention(gnn_unit, activation='selu', dropout_rate=0)([G_3, sensor_matrix4])
    #G_4 = Concatenate(axis=-1)([G_4h1, G_4h2, G_4h3, G_4h4])

    gnn_output = tf.split(G_3, num_sensors, 1)
    mlp_input = Concatenate(axis = -1)([gnn_output[-1], extract_cnn10])
 
    actor_output = Dense(256, activation='selu',  name='actor_mlp_1')(Flatten()(mlp_input))
    actor_output = Dense(64, activation='selu',  name='actor_mlp_2')(actor_output)
    actor_output = Dense(action_dim, activation='tanh', name='actor_output')(actor_output)
    
    critic_output = Dense(256, activation='selu',  name='critic_mlp_1')(Flatten()(mlp_input))
    critic_output = Dense(64, activation='selu',  name='critic_mlp_2')(critic_output)
    critic_output = Dense(1, activation='linear', name='critic_output')(critic_output)
        
    model = Model(inputs=[s_input1, s_input2, s_input3, s_input4,
                          s_input5, s_input6, s_input7, s_input8, s_input9, s_input10,
                          sensor_matrix1, sensor_matrix2, sensor_matrix3, sensor_matrix4], 
                  outputs= [actor_output, critic_output])
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

    G_3h1 = GraphAttention(gnn_unit, activation='selu', dropout_rate=0)([G_2, sensor_matrix1])
    G_3h2 = GraphAttention(gnn_unit, activation='selu', dropout_rate=0)([G_2, sensor_matrix2])
    G_3h3 = GraphAttention(gnn_unit, activation='selu', dropout_rate=0)([G_2, sensor_matrix3])
    G_3h4 = GraphAttention(gnn_unit, activation='selu', dropout_rate=0)([G_2, sensor_matrix4])
    G_3 = Concatenate(axis=-1)([G_3h1, G_3h2, G_3h3, G_3h4])
    
    #G_4h1 = GraphAttention(gnn_unit, activation='selu', dropout_rate=0)([G_3, sensor_matrix1])
    #G_4h2 = GraphAttention(gnn_unit, activation='selu', dropout_rate=0)([G_3, sensor_matrix2])
    #G_4h3 = GraphAttention(gnn_unit, activation='selu', dropout_rate=0)([G_3, sensor_matrix3])
    #G_4h4 = GraphAttention(gnn_unit, activation='selu', dropout_rate=0)([G_3, sensor_matrix4])
    #G_4 = Concatenate(axis=-1)([G_4h1, G_4h2, G_4h3, G_4h4])

    gnn_output = tf.split(G_3, num_sensors, 1)
    mlp_input = Concatenate(axis = -1)([gnn_output[-1], extract_cnn10])
 
    actor_output = Dense(256, activation='selu',  name='actor_mlp_1')(Flatten()(mlp_input))
    actor_output = Dense(64, activation='selu',  name='actor_mlp_2')(actor_output)
    actor_output = Dense(action_dim, activation='linear', name='actor_output')(actor_output)
    
    critic_output = Dense(256, activation='selu',  name='critic_mlp_1')(Flatten()(mlp_input))
    critic_output = Dense(64, activation='selu',  name='critic_mlp_2')(critic_output)
    critic_output = Dense(1, activation='linear', name='critic_output')(critic_output)
        
    model = Model(inputs=[s_input1, s_input2, s_input3, s_input4,
                          s_input5, s_input6, s_input7, s_input8, s_input9, s_input10,
                          sensor_matrix1, sensor_matrix2, sensor_matrix3, sensor_matrix4], 
                  outputs= [actor_output, critic_output])
    return model

def stage1_gat_layer3(num_sensors=10): # input/output = num of sensors 
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

    G_3h1 = GraphAttention(gnn_unit, activation='selu', dropout_rate=0)([G_2, sensor_matrix1])
    G_3h2 = GraphAttention(gnn_unit, activation='selu', dropout_rate=0)([G_2, sensor_matrix2])
    G_3h3 = GraphAttention(gnn_unit, activation='selu', dropout_rate=0)([G_2, sensor_matrix3])
    G_3h4 = GraphAttention(gnn_unit, activation='selu', dropout_rate=0)([G_2, sensor_matrix4])
    G_3 = Concatenate(axis=-1)([G_3h1, G_3h2, G_3h3, G_3h4])
    
    #G_4h1 = GraphAttention(gnn_unit, activation='selu', dropout_rate=0)([G_3, sensor_matrix1])
    #G_4h2 = GraphAttention(gnn_unit, activation='selu', dropout_rate=0)([G_3, sensor_matrix2])
    #G_4h3 = GraphAttention(gnn_unit, activation='selu', dropout_rate=0)([G_3, sensor_matrix3])
    #G_4h4 = GraphAttention(gnn_unit, activation='selu', dropout_rate=0)([G_3, sensor_matrix4])
    #G_4 = Concatenate(axis=-1)([G_4h1, G_4h2, G_4h3, G_4h4])

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
    
    model = Model(inputs=[s_input1, s_input2, s_input3, s_input4,
                          s_input5, s_input6, s_input7, s_input8, s_input9, s_input10,
                          sensor_matrix1, sensor_matrix2, sensor_matrix3, sensor_matrix4], 
                  outputs= [output1,output2,output3,output4,
                            output5,output6,output7,output8,output9, output10])
    return model

def collect_sen_obs():
    num_sensors = 9
    all_sensor_input = np.zeros((num_sensors, 84, 84*4, 3))
    #all_sensor_output = np.zeros((num_sensors, 2))
    for idx_sensor in range(num_sensors):
        sensor_path = 'training/' + all_sensors[idx_sensor]
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
    return all_sensor_input

def form_obs(cur_state, sensor_input=None):
    if sensor_input is None:        
        sensor_obs = collect_sen_obs()
    else:
        sensor_obs = sensor_input
    visual_obs = cur_state[:4]
    robot_obs = np.zeros((1, 84, 336, 3))
    robot_obs[0,:, 84*3:84*4,:] = visual_obs[0][0]
    robot_obs[0,:, 84*2:84*3,:] = visual_obs[1][0]
    robot_obs[0,:, 84*1:84*2,:] = visual_obs[2][0]
    robot_obs[0,:, 84*0:84*1,:] = visual_obs[3][0]
    input_obs = np.concatenate((sensor_obs, robot_obs), axis=0)
    return input_obs

def form_obs_cnn(cur_state):
    visual_obs = cur_state[:4]
    robot_obs = np.zeros((1, 84, 336, 3))
    robot_obs[0,:, 84*3:84*4,:] = visual_obs[0][0]
    robot_obs[0,:, 84*2:84*3,:] = visual_obs[1][0]
    robot_obs[0,:, 84*1:84*2,:] = visual_obs[2][0]
    robot_obs[0,:, 84*0:84*1,:] = visual_obs[3][0]
    return robot_obs 

def cal_admatrix_sensor(sensor_loc, num_sensors=None):
    sensor_loc = sensor_loc.astype('float')
    if num_sensors == None:            
        num_sensors = len(sensor_loc) 
    ad_matrix1 = np.zeros((num_sensors,num_sensors))
    ad_matrix2 = np.zeros((num_sensors,num_sensors))
    ad_matrix3 = np.zeros((num_sensors,num_sensors))
    ad_matrix4 = np.zeros((num_sensors,num_sensors))
    for i in range(num_sensors-1):
        for j in range(i+1, num_sensors):
            if np.sqrt((sensor_loc[i][0]-sensor_loc[j][0])**2+(sensor_loc[i][1]-sensor_loc[j][1])**2) <=15:
                ad_matrix1[i,j] = 1
                ad_matrix1[j,i] = 1
    
    for i in range(num_sensors-1):
        for j in range(num_sensors):
            if ad_matrix1[i,j] == 1:
                index = j
                sensor_nei = ad_matrix1[j,:]
                for k in range(num_sensors):
                    if k!= i and sensor_nei[k] == 1 and ad_matrix1[k,i] == 0:
                        ad_matrix2[k,i] = 1
                        ad_matrix2[i,k] = 1

    for i in range(num_sensors-1):
        for j in range(num_sensors):
            if ad_matrix2[i,j] == 1:
                index = j
                sensor_nei = ad_matrix1[j,:]
                for k in range(num_sensors):
                    if k!= i and sensor_nei[k] == 1 and ad_matrix1[k,i] == 0 and ad_matrix2[k, i] == 0:
                        ad_matrix3[k,i] = 1
                        ad_matrix3[i,k] = 1
    
    for i in range(num_sensors-1):
        for j in range(num_sensors):
            if ad_matrix3[i,j] == 1:
                index = j
                sensor_nei = ad_matrix1[j,:]
                for k in range(num_sensors):
                    if k!= i and sensor_nei[k] == 1 and ad_matrix1[k,i] == 0 and ad_matrix2[k, i] == 0 and ad_matrix3[k, i] == 0:
                        ad_matrix4[k,i] = 1
                        ad_matrix4[i,k] = 1
    return np.expand_dims(ad_matrix1, axis=0), np.expand_dims(ad_matrix2,axis=0), np.expand_dims(ad_matrix3,axis=0), np.expand_dims(ad_matrix4,axis=0)

def cal_admatrix(pos, env_index, num_sensors=9):
    robot_loc = pos
    sensor_loc = np.load('stage2_envs/env_{}_{}_1_loc.npy'.format(env_index+1, num_sensors))
    ad_matrix1 = np.zeros((num_sensors+1,num_sensors+1))
    ad_matrix2 = np.zeros((num_sensors+1,num_sensors+1))
    ad_matrix3 = np.zeros((num_sensors+1,num_sensors+1))
    ad_matrix4 = np.zeros((num_sensors+1,num_sensors+1))
    ad_matrix1[:num_sensors,:num_sensors] = np.load('stage2_envs/env_{}_{}_1_ad1.npy'.format(env_index+1, num_sensors))
    ad_matrix2[:num_sensors,:num_sensors] = np.load('stage2_envs/env_{}_{}_1_ad2.npy'.format(env_index+1, num_sensors))
    ad_matrix3[:num_sensors,:num_sensors] = np.load('stage2_envs/env_{}_{}_1_ad3.npy'.format(env_index+1, num_sensors))
    ad_matrix4[:num_sensors,:num_sensors] = np.load('stage2_envs/env_{}_{}_1_ad4.npy'.format(env_index+1, num_sensors))
    for j, sen in enumerate(sensor_loc):
        if np.sqrt((robot_loc[0]-sen[0])**2+(robot_loc[-1]-sen[-1])**2) <= 15:
            ad_matrix1[-1, j] = 1
            ad_matrix1[j, -1] = 1
    
    for i in range(num_sensors):
        if ad_matrix1[-1,i] == 1:
            index = i
            sensor_nei = ad_matrix1[index,:]
            for j in range(num_sensors):
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
            for j in range(num_sensors):
                if sensor_nei[j] == 1 and ad_matrix1[j, -1] == 0 and ad_matrix2[j, -1] == 0 and ad_matrix3[j, -1] == 0:
                    ad_matrix4[-1, j] = 1
                    ad_matrix4[j, -1] = 1
    return np.expand_dims(ad_matrix1, axis=0), np.expand_dims(ad_matrix2,axis=0), np.expand_dims(ad_matrix3,axis=0), np.expand_dims(ad_matrix4,axis=0)


def cal_connect_point(cur_state, env):
    robot_loc = [-cur_state[-1][0][5], 0.5, cur_state[-1][0][3]]  
    target_loc = [-cur_state[-1][0][2], 0.5, cur_state[-1][0][0]]
    dis_env = env      
    #if robot_loc[0] <=0:
    #    robot_loc[0] = 0 
    #if robot_loc[-1] <=0:
    #    robot_loc[-1] = 0 
    #if robot_loc[0] >=39:
    #    robot_loc[0] = 39 
    #if robot_loc[-1] >=39:
    #    robot_loc[-1] = 39 

    #if target_loc[0] <=0:
    #    target_loc[0] = 0 
    #if target_loc[-1] <=0:
    #    target_loc[-1] = 0 
    #if target_loc[0] >=39:
    #    target_loc[0] = 39 
    #if target_loc[-1] >=39:
    #    target_loc[-1] = 39 
    #print('robot_loc:', robot_loc, '   target_loc:', target_loc)
    connect_point = theta(dis_env, (robot_loc[0], robot_loc[-1]), 
                               (target_loc[0], target_loc[-1]))
    return connect_point

def in_view(env, cur_loc, target_loc):
    #connect_point = cal_connect_point(cur_state)
    cur_loc = (np.round(cur_loc[0]),  np.round(cur_loc[-1]))
    cur_loc = (int(cur_loc[0]), int(cur_loc[1]))
    target_loc = (np.round(target_loc[0]),  np.round(target_loc[-1]))

    connect_p = target_loc
    past_grid = []
    past_grid_int = []
    x_flag = 1
    y_flag = 1
    f_x = int
    f_y = int
    if (cur_loc[1] - connect_p[1]) >=0:
        y_flag = -1
        f_y = np.ceil
    if (cur_loc[0] - connect_p[0]) >=0:
        x_flag = -1
        f_x = np.ceil
    if abs(connect_p[0]-cur_loc[0])<= abs(connect_p[1]-cur_loc[1]):    
        slocp = abs((connect_p[0]-cur_loc[0])/(connect_p[1]-cur_loc[1]+1e-10))
        for j in range(int(abs(cur_loc[1]- connect_p[1]))):
            cur_x, cur_y = (cur_loc[0]+ j*slocp*x_flag, cur_loc[1]+j*y_flag)
            fur_x, fur_y = (cur_loc[0]+ (j+1)*slocp*x_flag, cur_loc[1]+(j+1)*y_flag)
            past_grid.append((cur_x,cur_y))
            if f_x(fur_x) != f_x(cur_x) and fur_x != f_x(fur_x):
                past_grid_int.append((f_x(cur_x), f_y(cur_y)))
                past_grid_int.append((f_x(fur_x), f_y(cur_y)))
            else:
                past_grid_int.append((f_x(cur_x), f_y(cur_y)))
    else:  
        slocp = abs((connect_p[1]-cur_loc[1])/(connect_p[0]-cur_loc[0]+1e-10))
        for j in range(int(abs(cur_loc[0]-connect_p[0]))):
            cur_x, cur_y = (cur_loc[0]+j*x_flag, cur_loc[1]+slocp*j*y_flag)
            fur_x, fur_y = (cur_loc[0]+(j+1)*x_flag, cur_loc[1]+slocp*(j+1)*y_flag)
            past_grid.append((cur_x,cur_y))
            if f_y(fur_y) != f_y(cur_y) and fur_y != f_y(fur_y):
                past_grid_int.append((f_x(cur_x), f_y(cur_y)))
                past_grid_int.append((f_x(cur_x), f_y(fur_y)))
            else:
                past_grid_int.append((f_x(cur_x), f_y(cur_y)))
    past_grid_int.append(connect_p)
    for c_i, cur_grid in enumerate(past_grid_int):
        zc_1, zc_2 = cur_grid
        if int(cur_grid[0]) >= env.shape[0]:
            zc_1 = env.shape[0]-1
        if int(cur_grid[1]) >= env.shape[0]:
            zc_2 = env.shape[0]-1
        if env[int(zc_1), int(zc_2)] == 0:
            pass
        else:
            #print(c_i)
            return False
    return True

def hit_wall(env, loc, action):
    #print('loc:',loc)
    #print('action:', action)
    z1 = int(loc[0]+action[0])
    z2 = int(loc[-1]+action[-1])
    if z1 <=0:
        z1 = 0 
    if z2 <=0:
        z2 = 0 
    if z2 >=env.shape[0]:
        z2 = env.shape[0]-1
    if  z1>=env.shape[0]:
       z1 = env.shape[0]-1 
    if env[z1][z2] == 1:
        return True
    return False

def collision_avoid(env, loc, action):
    z1 = round(loc[0]+action[0])
    z2 = round(loc[-1]+action[-1])
    #print("z1:", z1)
    #print("loc:", loc)
    if z1 <=0:
        z1 = 0 
    if z2 <=0:
        z2 = 0 
    if z2 >=env.shape[0]:
        z2 = env.shape[0]-1
    if  z1>=env.shape[0]:
       z1 = env.shape[0]-1
    #print("next_loc is:", [z1,z2], '    value is: ', env[z1][z2])
    if env[z1][z2] == 1:
        if env[z1][round(loc[-1])] == 1:
            #print("y")
            return [0, action[1]], [-action[0],0]
        elif env[round(loc[0])][z2] == 1:
            #print("x")
            return [action[0], 0], [0,-action[1]]
    return action, [0,0]
    
class PPO:
    def __init__(
            self,
            env,
            discrete=False,
            c1=1.0,
            c2=0.01,
            clip_ratio=0.2,
            gamma=0.95,
            lam=0.95,
            batch_size=32,
            max_steps = 512
    ):
        self.env = env
        self.env_index = 0
        self.dis_env = np.load('stage2_envs/env_{}.npy'.format(self.env_index+1))  
        self.action_dim = 2  # number of actions
        self.discrete = discrete
        if not discrete:
            self.action_bound = 1
            self.action_shift = 0

        self.lr = 1e-6
        # Define and initialize network
        self.policy = att_model_distribute()
        self.policy_cnn = cnn_model()
        self.model_optimizer = Adam(learning_rate=self.lr)
        #print(self.policy.summary())
        # Stdev for continuous action
        if not discrete:
            self.policy_log_std = tf.Variable(tf.zeros(self.action_dim, dtype=tf.float32), trainable=True)

        # Set hyperparameters
        self.gamma = gamma  # discount factor
        self.lam = lam
        self.c1 = c1  # value difference coeff
        self.c2 = c2  # entropy coeff
        self.clip_ratio = clip_ratio  # for clipped surrogate
        self.batch_size = batch_size
        self.max_steps = max_steps
        self.n_updates = max_steps//batch_size  # number of epochs per episode
        self.sub_target = 1
        self.is_att = True
        self.is_cnn = True
        #self.testmodel = stage1_gat_layer3()
        #self.testmodel.load_weights('gnn_1002_att_layer3.h5')
        # Tensorboard
        self.summaries = {}

    def cal_angle(self, cur_state):
        #connect_point = cal_connect_point(cur_state)
        cur_loc = [-cur_state[-1][0][5], 0.5, cur_state[-1][0][3]] 

        if self.sub_target >=(len(self.origin_connect_point)-1):
            self.sub_target = int(len(self.origin_connect_point)-1)
        robot2cp = np.sqrt((cur_loc[0]-self.origin_connect_point[self.sub_target][0])**2+
                           (cur_loc[-1]-self.origin_connect_point[self.sub_target][1])**2)
        if robot2cp <= 2.0:
            if self.sub_target >=(len(self.origin_connect_point)-1):
                pass
            else:
                self.sub_target += 1
        robot2cp_angle = (self.origin_connect_point[self.sub_target][0]-cur_loc[0],
                          self.origin_connect_point[self.sub_target][-1]-cur_loc[-1])
        #return np.divide(robot2cp_angle,  sum(robot2cp_angle)+1e-10) - binyu
        
        # qingbiao
        p_new = np.zeros((2,1))
        p_new[0] = np.true_divide(robot2cp_angle[0], np.sqrt(robot2cp_angle[0]**2 + robot2cp_angle[1]**2) + ZERO_TOLERANCE)
        p_new[1] = np.true_divide(robot2cp_angle[1], np.sqrt(robot2cp_angle[0]**2 + robot2cp_angle[1]**2) + ZERO_TOLERANCE)
        return p_new
    
    def cal_angle2(self, cur_state, env):
        #connect_point = cal_connect_point(cur_state) 
        cur_loc = (round(-cur_state[-1][0][5]),  round(cur_state[-1][0][3]))
        target_loc = (round(-cur_state[-1][0][2]),  round(cur_state[-1][0][0]))
        #if target_loc[0]>=39:
        #    target_loc = (39, target_loc[1])
        #if target_loc[1]>=39:
        #    target_loc = (target_loc[0],39)
        is_view = []
        for connect_p in self.origin_connect_point[1:]:
            past_grid = []
            past_grid_int = []
            x_flag = 1
            y_flag = 1
            f_x = int
            f_y = int
            if (cur_loc[1] - connect_p[1]) >=0:
                y_flag = -1
                f_y = np.ceil
            if (cur_loc[0] - connect_p[0]) >=0:
                x_flag = -1
                f_x = np.ceil
            if abs(connect_p[0]-cur_loc[0])<= abs(connect_p[1]-cur_loc[1]):    
                slocp = abs((connect_p[0]-cur_loc[0])/(connect_p[1]-cur_loc[1]+1e-10))
                for j in range(int(abs(cur_loc[1]- connect_p[1]))):
                    cur_x, cur_y = (cur_loc[0]+ j*slocp*x_flag, cur_loc[1]+j*y_flag)
                    fur_x, fur_y = (cur_loc[0]+ (j+1)*slocp*x_flag, cur_loc[1]+(j+1)*y_flag)
                    past_grid.append((cur_x,cur_y))
                    if f_x(fur_x) != f_x(cur_x) and fur_x != f_x(fur_x):
                        past_grid_int.append((f_x(cur_x), f_y(cur_y)))
                        past_grid_int.append((f_x(fur_x), f_y(cur_y)))
                    else:
                        past_grid_int.append((f_x(cur_x), f_y(cur_y)))
            else:  
                slocp = abs((connect_p[1]-cur_loc[1])/(connect_p[0]-cur_loc[0]+1e-10))
                for j in range(int(abs(cur_loc[0]-connect_p[0]))):
                    cur_x, cur_y = (cur_loc[0]+j*x_flag, cur_loc[1]+slocp*j*y_flag)
                    fur_x, fur_y = (cur_loc[0]+(j+1)*x_flag, cur_loc[1]+slocp*(j+1)*y_flag)
                    past_grid.append((cur_x,cur_y))
                    if f_y(fur_y) != f_y(cur_y) and fur_y != f_y(fur_y):
                        past_grid_int.append((f_x(cur_x), f_y(cur_y)))
                        past_grid_int.append((f_x(cur_x), f_y(fur_y)))
                    else:
                        past_grid_int.append((f_x(cur_x), f_y(cur_y)))
            past_grid_int.append(connect_p)
            for c_i, cur_grid in enumerate(past_grid_int):
                if cur_grid[0] <=0:
                    cur_grid = (0, cur_grid[1])
                if cur_grid[1] <=0:
                    cur_grid = (cur_grid[0], 0) 
               # if cur_grid[0] >=39:
               #     cur_grid = (39, cur_grid[1])
               # if  cur_grid[1]>=39:
               #     cur_grid = (cur_grid[0], 39) 
                if env[int(cur_grid[0]), int(cur_grid[1])] == 0:
                    pass
                else:
                    #print(c_i)
                    break
                    
                if c_i == len(past_grid_int)-1:
                    is_view.append(connect_p)
        if len(is_view) == 0:
            new_connect_point = theta(env, (cur_loc[0], cur_loc[-1]), 
                           (target_loc[0], target_loc[-1]))
            is_view.append(new_connect_point[1])
        robot2cp = np.sqrt((cur_loc[0]-is_view[-1][0])**2+
                   (cur_loc[-1]-is_view[-1][1])**2)
        robot2cp_angle = (is_view[-1][0]-cur_loc[0],
                  is_view[-1][1]-cur_loc[1])
        p_new = np.zeros((2,1))
        p_new[0] = np.true_divide(robot2cp_angle[0], np.sqrt(robot2cp_angle[0]**2 + robot2cp_angle[1]**2) + ZERO_TOLERANCE)
        p_new[1] = np.true_divide(robot2cp_angle[1], np.sqrt(robot2cp_angle[0]**2 + robot2cp_angle[1]**2) + ZERO_TOLERANCE)
        return p_new
                
    def get_dist(self, output):
        if self.discrete:
            dist = tfd.Categorical(probs=output)
        else:
            std = tf.math.exp(self.policy_log_std)
            dist = tfd.Normal(loc=output, scale=std)

        return dist

    def evaluate_actions(self, state, action):
        if self.is_cnn == False:
            cur_obs = np.zeros((len(state),10,84,336,3))
            ad_matrix1, ad_matrix2, ad_matrix3, ad_matrix4 = [], [], [], []
            for j in range(len(state)):  
                new_state = form_obs(state[j], self.sensor_obs)
                cur_obs[j] = new_state
                robot_pos = [-state[j][-1][0][5], 0.5, state[j][-1][0][3]]
                new_matrix1, new_matrix2, new_matrix3, new_matrix4 = cal_admatrix(robot_pos, self.env_index)
                if self.is_att == False:
                    ad_matrix1.append(localpooling_filter(new_matrix1))
                    ad_matrix2.append(localpooling_filter(new_matrix2))
                    ad_matrix3.append(localpooling_filter(new_matrix2))
                    ad_matrix4.append(localpooling_filter(new_matrix2))
                else:
                    ad_matrix1.append(new_matrix1)
                    ad_matrix2.append(new_matrix2)
                    ad_matrix3.append(new_matrix2)
                    ad_matrix4.append(new_matrix2)
            output, value = self.policy([cur_obs[:,0], cur_obs[:,1], cur_obs[:,2], cur_obs[:,3], cur_obs[:,4], cur_obs[:,5],
                                         cur_obs[:,6], cur_obs[:,7], cur_obs[:,8], cur_obs[:,9], 
                                         ad_matrix1, ad_matrix2, ad_matrix3, ad_matrix4])
        else:
            cur_obs = np.zeros((len(state),1,84,336,3))  
            for j in range(len(state)):  
                new_state = form_obs_cnn(state[j])
                cur_obs[j] = new_state
                robot_pos = [-state[j][-1][0][5], 0.5, state[j][-1][0][3]]
            output, value = self.policy_cnn([cur_obs[:,0]])
        
        #output = tf.clip_by_value(output, -1, 1)
        dist = self.get_dist(tf.cast(output, dtype=tf.float32))
        if not self.discrete:
            action = (action - self.action_shift) / self.action_bound

        log_probs = dist.log_prob(action)
        #action = tf.clip_by_value(action, -1, 1)
        if not self.discrete:
            log_probs = tf.reduce_sum(log_probs, axis=-1)

        entropy = dist.entropy()

        return log_probs, entropy, value

    def act(self, state, test=False):
        if self.is_cnn == False:       
            robot_pos = [-state[-1][0][5], 0.5, state[-1][0][3]]
            cur_obs = form_obs(state)
            #cur_obs = np.expand_dims(state, axis=0).astype(np.float32)
            ad_matrix1, ad_matrix2, ad_matrix3, ad_matrix4 = cal_admatrix(robot_pos, self.env_index)
            if self.is_att == False:
                ad_matrix1, ad_matrix2 = localpooling_filter(ad_matrix1), localpooling_filter(ad_matrix2)
                ad_matrix3, ad_matrix4 = localpooling_filter(ad_matrix3), localpooling_filter(ad_matrix4)
            else:
                pass            
            output, value = self.policy.predict([np.expand_dims(cur_obs[0], axis=0),np.expand_dims(cur_obs[1], axis=0), np.expand_dims(cur_obs[2], axis=0), 
                                                 np.expand_dims(cur_obs[3], axis=0),np.expand_dims(cur_obs[4], axis=0), np.expand_dims(cur_obs[5], axis=0), 
                                                 np.expand_dims(cur_obs[6], axis=0),np.expand_dims(cur_obs[7], axis=0), np.expand_dims(cur_obs[8], axis=0), 
                                                 np.expand_dims(cur_obs[9], axis=0), ad_matrix1, ad_matrix2, ad_matrix3, ad_matrix4])
        else:
            robot_pos = [-state[-1][0][5], 0.5, state[-1][0][3]]
            cur_obs = form_obs_cnn(state)
            output, value = self.policy_cnn.predict([cur_obs])
        output = tf.clip_by_value(output, -1, 1)
        dist = self.get_dist(output)

        if self.discrete:
            action = tf.math.argmax(output, axis=-1) if test else dist.sample()
            log_probs = dist.log_prob(action)
        else:
            action = output if test else dist.sample()
            action = tf.clip_by_value(action, -1, 1)
            #action = action.astype(np.float32)
            log_probs = tf.reduce_sum(dist.log_prob(action), axis=-1)
            action = action * self.action_bound + self.action_shift
        return action[0].numpy(), value[0][0], log_probs[0].numpy()

    def save_model(self, fn):
        self.policy.save(fn)

    def load_model(self, fn):
        self.policy.load_weights(fn)
        print(self.policy.summary())

    def get_gaes(self, rewards, v_preds, next_v_preds):
        # source: https://github.com/uidilr/ppo_tf/blob/master/ppo.py#L98
        deltas = [r_t + self.gamma * v_next - v for r_t, v_next, v in zip(rewards, next_v_preds, v_preds)]
        gaes = copy.deepcopy(deltas)
        for t in reversed(range(len(gaes) - 1)):  # is T-1, where T is time step which run policy
            gaes[t] = gaes[t] + self.lam * self.gamma * gaes[t + 1]
        return gaes

    def learn(self, observations, actions, log_probs, next_v_preds, rewards, gaes):
        if len(rewards.shape) == 1:
            rewards = np.expand_dims(rewards, axis=-1).astype(np.float32)
        if len(next_v_preds.shape) == 1:    
            next_v_preds = np.expand_dims(next_v_preds, axis=-1).astype(np.float32)

        with tf.GradientTape(persistent=True,) as tape:
            new_log_probs, entropy, state_values = self.evaluate_actions(observations, actions)

            ratios = tf.exp(new_log_probs - log_probs)
            clipped_ratios = tf.clip_by_value(ratios, clip_value_min=1-self.clip_ratio,
                                              clip_value_max=1+self.clip_ratio)
            loss_clip = tf.minimum(gaes * ratios, gaes * clipped_ratios)
            loss_clip = tf.reduce_mean(loss_clip)

            target_values = rewards + self.gamma * next_v_preds
            vf_loss = tf.reduce_mean(tf.math.square(state_values - target_values))
            vf_loss = tf.cast(vf_loss, tf.float32)

            entropy = tf.reduce_mean(entropy)
            total_loss = -loss_clip + self.c1 * vf_loss - self.c2 * entropy

        train_variables = self.policy.trainable_variables
        if not self.discrete:
            train_variables += [self.policy_log_std]
        grad = tape.gradient(total_loss, train_variables)  # compute gradient
        self.model_optimizer.apply_gradients(zip(grad, train_variables))

        # tensorboard info
        self.summaries['total_loss'] = total_loss
        self.summaries['surr_loss'] = loss_clip
        self.summaries['vf_loss'] = vf_loss
        self.summaries['entropy'] = entropy

    def train(self, max_epochs=100000, save_freq=100):
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        train_log_dir = 'logs/' + current_time
        summary_writer = tf.summary.create_file_writer(train_log_dir)

        max_steps = self.max_steps
        episode, epoch = 0, 0
        while epoch < max_epochs:
            if epoch == 5000:
                self.lr = self.lr/10
                self.model_optimizer = Adam(learning_rate=self.lr)
            if epoch == 10000:
                self.lr = self.lr/10
                self.model_optimizer = Adam(learning_rate=self.lr)
                
            self.env.reset()
            behavior_names = self.env.get_behavior_names()
            DecisionSteps, TerminalSteps = self.env.get_steps(behavior_names[0])
            self.env_index = int(DecisionSteps.obs[-1][0][-1])
            self.dis_env = np.load('stage2_envs/env_{}.npy'.format(self.env_index+1))  
            cur_state = DecisionSteps.obs
            self.sensor_obs = collect_sen_obs()
            obs, actions, log_probs, rewards, v_preds, next_v_preds = [], [], [], [], [], []
            self.origin_connect_point = cal_connect_point(cur_state, self.dis_env)
            done, steps = False, 0
            robot_locs, vector_obses=[], []
            while not done and steps < max_steps:
                action, value, log_prob = self.act(cur_state)  
                self.env.set_actions(behavior_names[0], np.expand_dims(action,axis=0))
                self.env.step()
                DecisionSteps, TerminalSteps = self.env.get_steps(behavior_names[0])
                reward = 0
                if len(TerminalSteps.obs[-1]) != 0:
                    done = 1
                    next_state = TerminalSteps.obs
                    reward += TerminalSteps.reward[0]
                else:
                    next_state = DecisionSteps.obs
                    reward += DecisionSteps.reward[0]
                    
                vector_obs = next_state[-1]
                robot_loc = [-next_state[-1][0][5], 0.5, next_state[-1][0][3]] 
    
                robot_locs.append(robot_loc)     
                vector_obses.append(vector_obs)
                target_loc = [-next_state[-1][0][2], 0.75, next_state[-1][0][0]] 
                #if abs(robot_loc[0] - target_loc[0])+abs(robot_loc[-1] - target_loc[-1]) <=1.5:
                #    done = 1
                true_angle = self.cal_angle2(cur_state, self.dis_env)
                reward += -np.sqrt((true_angle[0]-action[0])**2+(true_angle[1]-action[1])**2)+1
                # self.env.render(mode='rgb_array')

                rewards.append(reward)
                v_preds.append(value)
                obs.append(cur_state)
                actions.append(action)
                log_probs.append(log_prob)

                steps += 1
                cur_state = next_state
            
            np.save('record/robot_loc_{}_{}'.format(epoch,steps), robot_locs)
            np.save('record/vectorobs_{}_{}'.format(epoch,steps), vector_obses)
            np.save('record/action_{}_{}'.format(epoch,steps), actions)
            np.save('record/reward_{}_{}'.format(epoch,steps), rewards)
            

            next_v_preds = v_preds[1:] + [0]
            gaes = self.get_gaes(rewards, v_preds, next_v_preds)
            gaes = np.array(gaes).astype(dtype=np.float32)
            #gaes = (gaes - np.mean(gaes)) / np.std(gaes)
            data = [obs, actions, log_probs, next_v_preds, rewards, gaes]
            
            self.n_updates = steps//self.batch_size + 1 
            for i in range(self.n_updates):
                # Sample training data
                sample_indices = np.random.randint(low=0, high=len(rewards), size=self.batch_size)
                sampled_data1 = []
                for j in sample_indices:
                    sampled_data1.append(data[0][j])  
                sampled_data2 = [np.take(a=a, indices=sample_indices, axis=0) for a in data[1:]]
                #sampled_data = sampled_data1 + sampled_data2
                # Train model
                self.learn(sampled_data1, *sampled_data2)

                # Tensorboard update
                with summary_writer.as_default():
                    tf.summary.scalar('Loss/total_loss', self.summaries['total_loss'], step=epoch)
                    tf.summary.scalar('Loss/clipped_surr', self.summaries['surr_loss'], step=epoch)
                    tf.summary.scalar('Loss/vf_loss', self.summaries['vf_loss'], step=epoch)
                    tf.summary.scalar('Loss/entropy', self.summaries['entropy'], step=epoch)

                summary_writer.flush()
                epoch += 1

            episode += 1
            print("episode {}: {} total reward, {} steps, {} epochs".format(
                episode, np.sum(rewards), steps, epoch))

            # Tensorboard update
            with summary_writer.as_default():
                tf.summary.scalar('Main/episode_reward', np.sum(rewards), step=episode)
                tf.summary.scalar('Main/episode_steps', steps, step=episode)

            summary_writer.flush()

            if steps >= max_steps:
                print("episode {}, reached max steps".format(episode))
                #self.save_model("ppo_episode{}.h5".format(episode))

            if episode % save_freq == 0:
                self.save_model("ppo_episode{}.h5".format(episode))

        self.save_model("ppo_final_episode{}.h5".format(episode))

    def test(self):
        max_steps = self.max_steps
        self.env.reset()
        behavior_names = self.env.get_behavior_names()
        DecisionSteps, TerminalSteps = self.env.get_steps(behavior_names[0])
        cur_state = DecisionSteps.obs
        obs, actions, log_probs, rewards, v_preds = [], [], [], [], []
        self.env_index = int(DecisionSteps.obs[-1][0][-1])
        self.dis_env = np.load('stage2_envs/env_{}.npy'.format(self.env_index+1)) 
        self.origin_connect_point = cal_connect_point(cur_state, self.dis_env)
        done, steps = False, 0
        robot_locs, vector_obses, true_angles =[], [], []
        while not done and steps < max_steps:
            
            action, value, log_prob = self.act(cur_state, test=True)  
            self.env.set_actions(behavior_names[0], np.expand_dims(action,axis=0))
            self.env.step()
            DecisionSteps, TerminalSteps = self.env.get_steps(behavior_names[0])
            reward = DecisionSteps.reward[0]
            if len(TerminalSteps.obs[-1]) != 0:
                done = 1
                next_state = TerminalSteps.obs
                reward += TerminalSteps.reward[0]
            else:
                next_state = DecisionSteps.obs
            vector_obs = next_state[-1]
            robot_loc = [-next_state[-1][0][5], 0.5, next_state[-1][0][3]] 
            robot_locs.append(robot_loc)     
            vector_obses.append(vector_obs)
            target_loc = [-next_state[-1][0][2], 0.75, next_state[-1][0][0]] 
            #if abs(robot_loc[0] - target_loc[0])+abs(robot_loc[-1] - target_loc[-1]) <=1.5:
            #    done = 1
            true_angle = self.cal_angle2(cur_state, self.dis_env)
            reward += -np.sqrt((true_angle[0]-action[0])**2+(true_angle[1]-action[1])**2)+1
            # self.env.render(mode='rgb_array')

            rewards.append(reward)
            v_preds.append(value)
            obs.append(cur_state)
            actions.append(action)
            log_probs.append(log_prob)
            true_angles.append(true_angle)

            steps += 1
            cur_state = next_state
        return rewards, actions, true_angles, vector_obses

    def test_stage1(self):
        max_steps = 100
        self.env.reset()
        behavior_names = self.env.get_behavior_names()
        DecisionSteps, TerminalSteps = self.env.get_steps(behavior_names[0])
        cur_state = DecisionSteps.obs
        obs, actions, rewards = [], [], []
        self.env_index = int(DecisionSteps.obs[-1][0][-1])
        self.dis_env = np.load('stage2_envs/env_{}.npy'.format(self.env_index+1)) 
        self.origin_connect_point = cal_connect_point(cur_state, self.dis_env)
        done, steps = False, 0
        robot_locs, vector_obses=[], []

        while not done and steps < max_steps:     
            robot_pos = [-cur_state[-1][0][5], 0.5, cur_state[-1][0][3]]
            cur_obs = form_obs(cur_state)
        #cur_obs = np.expand_dims(state, axis=0).astype(np.float32)
            ad_matrix1, ad_matrix2, ad_matrix3, ad_matrix4 = cal_admatrix(robot_pos, self.env_index)
            if self.is_att == False:
                ad_matrix1, ad_matrix2 = localpooling_filter(ad_matrix1), localpooling_filter(ad_matrix2)
                ad_matrix3, ad_matrix4 = localpooling_filter(ad_matrix3), localpooling_filter(ad_matrix4)
            else:
                pass            
            s_output1, s_output2, s_output3, s_output4, s_output5, s_output6, s_output7, s_output8, s_output9, output = self.testmodel.predict([np.expand_dims(cur_obs[0], axis=0),np.expand_dims(cur_obs[1], axis=0), np.expand_dims(cur_obs[2], axis=0), 
                                             np.expand_dims(cur_obs[3], axis=0),np.expand_dims(cur_obs[4], axis=0), np.expand_dims(cur_obs[5], axis=0), 
                                             np.expand_dims(cur_obs[6], axis=0),np.expand_dims(cur_obs[7], axis=0), np.expand_dims(cur_obs[8], axis=0), 
                                             np.expand_dims(cur_obs[9], axis=0), ad_matrix1, ad_matrix2, ad_matrix3, ad_matrix4])
            p_new = np.zeros((1,2))
            p_new[0][0] = np.true_divide(output[0][0], np.sqrt(output[0][0]**2 + output[0][1]**2) + ZERO_TOLERANCE)
            p_new[0][1] = np.true_divide(output[0][1], np.sqrt(output[0][0]**2 + output[0][1]**2) + ZERO_TOLERANCE)
            action = p_new[0]       
            
            self.env.set_actions(behavior_names[0], np.expand_dims(action,axis=0))
            self.env.step()
            DecisionSteps, TerminalSteps = self.env.get_steps(behavior_names[0])
            reward = DecisionSteps.reward[0]
            if len(TerminalSteps.obs[-1]) != 0:
                done = 1
                next_state = TerminalSteps.obs
                reward += TerminalSteps.reward[0]
            else:
                next_state = DecisionSteps.obs
            vector_obs = next_state[-1]
            robot_loc = [-next_state[-1][0][5], 0.5, next_state[-1][0][3]] 
            robot_locs.append(robot_loc)     
            vector_obses.append(vector_obs)
            target_loc = [-next_state[-1][0][2], 0.75, next_state[-1][0][0]] 
            #if abs(robot_loc[0] - target_loc[0])+abs(robot_loc[-1] - target_loc[-1]) <=1.5:
            #    done = 1
            true_angle = self.cal_angle2(cur_state, self.dis_env)
            reward += -np.sqrt((true_angle[0]-action[0])**2+(true_angle[1]-action[1])**2)+1
            # self.env.render(mode='rgb_array')

            rewards.append(reward)
            obs.append(cur_state)
            actions.append(action)

            steps += 1
            cur_state = next_state
        return rewards, actions
    
    def test3(self, map_path = 'env_', if_normal=True):
        max_steps = self.max_steps
        #self.env.reset()
        behavior_names = self.env.get_behavior_names()
        DecisionSteps, TerminalSteps = self.env.get_steps(behavior_names[0])
        self.env_index = int(DecisionSteps.obs[-1][0][-1])
        self.dis_env = np.load('res1101/' + map_path + '{}.npy'.format(self.env_index+1)) 
        #self.dis_env = np.load('stage2_envs/env_{}.npy'.format(self.env_index+1)) 
        cur_state = DecisionSteps.obs
        actions, robot_locs = [], []
        robot_loc = [-cur_state[-1][0][5], 0.5, cur_state[-1][0][3]] 
        target_loc = [-cur_state[-1][0][2], 0.5, cur_state[-1][0][0]] 
        self.origin_connect_point = cal_connect_point(cur_state, self.dis_env)
        
        able_move = False
        able_see = in_view(self.dis_env, robot_loc, target_loc)
        if able_see:
            able_move = True
        r2t_path = AStarSearch(self.dis_env, (round(robot_loc[0]), round(robot_loc[-1])), 
                                   (round(target_loc[0]), round(target_loc[-1])))    
        if len(r2t_path) <= 34:
            able_move = True 
        done, steps = False, 0
        jj=1
        last_loc = [0,0.5,0]
        block_num = 0
        origin_action = []
        while not done and steps < max_steps:
            move_loc = self.origin_connect_point[jj]
            robot_loc = [-cur_state[-1][0][5], 0.5, cur_state[-1][0][3]] 
            robot2target = np.sqrt((robot_loc[0]-move_loc[0])**2+
                                   (robot_loc[-1]-move_loc[-1])**2)
            if robot2target <= 1:
                if jj < len(self.origin_connect_point)-1:
                    jj += 1
                else:
                    pass
            robot2target_angle = (move_loc[0]-robot_loc[0],
                                  move_loc[-1]-robot_loc[-1])
            z1 = np.true_divide(robot2target_angle[0], np.sqrt(robot2target_angle[0]**2 + robot2target_angle[1]**2) + ZERO_TOLERANCE) + np.random.random()*0.22
            z2 = np.true_divide(robot2target_angle[1], np.sqrt(robot2target_angle[0]**2 + robot2target_angle[1]**2) + ZERO_TOLERANCE) + np.random.random()*0.22
            able_see = in_view(self.dis_env, robot_loc, target_loc)
            if able_see:
                able_move = True
            
            if able_move:    
                action = [z1, z2]
            else:
                action = [np.random.random()*z1, np.random.random()*z2]
            origin_action.append([z1,z2])
            if hit_wall(self.dis_env, robot_loc, action):
                action, wall_force = collision_avoid(self.dis_env, robot_loc, action)
                if np.random.random()>0.9: 
                    action = [np.random.random()-z1, np.random.random()-z2]
                #if block_num >= 3:
                #    new_force = np.average(origin_action[-5:-1])
                #    action = np.add(new_force, wall_force)
            if DecisionSteps.reward[0] == -1:
                block_num += 1
                action, wall_force = collision_avoid(self.dis_env, robot_loc, action)
                if block_num >= 3:
                    new_force = np.average(origin_action[-5:-1])
                    action = [np.random.random()-z1, np.random.random()-z2] 
                    #action = np.add(new_force, wall_force)
            else:
                block_num = 0
            #if np.sum(np.subtract(last_loc, robot_loc)) <0.2:
            #    action = [np.random.random()-z1, np.random.random()-z2]
            if if_normal:
                z3 = np.true_divide(action[0], np.sqrt(action[0]**2 + action[1]**2) + ZERO_TOLERANCE)
                z4 = np.true_divide(action[1], np.sqrt(action[0]**2 + action[1]**2) + ZERO_TOLERANCE)
            action = [z3,z4]

            self.env.set_actions(behavior_names[0], np.expand_dims(action,axis=0))
            self.env.step()
            DecisionSteps, TerminalSteps = self.env.get_steps(behavior_names[0])
            reward = DecisionSteps.reward[0]
            if len(TerminalSteps.obs[-1]) != 0:
                done = 1
                next_state = TerminalSteps.obs
                reward += TerminalSteps.reward[0]
            else:
                next_state = DecisionSteps.obs
            steps += 1
            cur_state = next_state
            last_loc = robot_loc
            actions.append(action)
            robot_locs.append(robot_loc)
        if done  == 1:
            pass
        else:
            self.env.reset()
        return actions, robot_locs, target_loc, done
    
    def test4(self, map_path = 'env_', if_normal=True):
        max_steps = self.max_steps
        #self.env.reset()
        behavior_names = self.env.get_behavior_names()
        DecisionSteps, TerminalSteps = self.env.get_steps(behavior_names[0])
        self.env_index = int(DecisionSteps.obs[-1][0][-1])
        self.dis_env = np.load('res1101/' + map_path + '{}.npy'.format(self.env_index+1)) 
        #self.dis_env = np.load('stage2_envs/env_{}.npy'.format(self.env_index+1)) 
        cur_state = DecisionSteps.obs
        actions, robot_locs = [], []
        robot_loc = [-cur_state[-1][0][5], 0.5, cur_state[-1][0][3]] 
        target_loc = [-cur_state[-1][0][2], 0.5, cur_state[-1][0][0]] 
        self.origin_connect_point = cal_connect_point(cur_state, self.dis_env)
        
        able_move = False
        able_see = in_view(self.dis_env, robot_loc, target_loc)
        if able_see:
            able_move = True
        r2t_path = AStarSearch(self.dis_env, (round(robot_loc[0]), round(robot_loc[-1])), 
                                   (round(target_loc[0]), round(target_loc[-1])))    
        if len(r2t_path) <= 34:
            able_move = True 
        done, steps = False, 0
        jj=1
        last_loc = [0,0.5,0]
        while not done and steps < max_steps:
            move_loc = self.origin_connect_point[jj]
            robot_loc = [-cur_state[-1][0][5], 0.5, cur_state[-1][0][3]] 
            robot2target = np.sqrt((robot_loc[0]-move_loc[0])**2+
                                   (robot_loc[-1]-move_loc[-1])**2)
            if robot2target <= 1:
                if jj < len(self.origin_connect_point)-1:
                    jj += 1
                else:
                    pass
            robot2target_angle = (move_loc[0]-robot_loc[0],
                                  move_loc[-1]-robot_loc[-1])
            z1 = np.true_divide(robot2target_angle[0], np.sqrt(robot2target_angle[0]**2 + robot2target_angle[1]**2) + ZERO_TOLERANCE) + np.random.random()*0.22
            z2 = np.true_divide(robot2target_angle[1], np.sqrt(robot2target_angle[0]**2 + robot2target_angle[1]**2) + ZERO_TOLERANCE) + np.random.random()*0.22
            able_see = in_view(self.dis_env, robot_loc, target_loc)
            if able_see:
                able_move = True
            
            if able_move:    
                action = [z1, z2]
            else:
                action = [np.random.random()*z1, np.random.random()*z2]
            if hit_wall(self.dis_env, robot_loc, action):
                action = [np.random.random()-z1, np.random.random()-z2]
            #if np.sum(np.subtract(last_loc, robot_loc)) <0.2:
            #    action = [np.random.random()-z1, np.random.random()-z2]
            if if_normal:
                z3 = np.true_divide(action[0], np.sqrt(action[0]**2 + action[1]**2) + ZERO_TOLERANCE)
                z4 = np.true_divide(action[1], np.sqrt(action[0]**2 + action[1]**2) + ZERO_TOLERANCE)
            action = [z3,z4]

            self.env.set_actions(behavior_names[0], np.expand_dims(action,axis=0))
            self.env.step()
            DecisionSteps, TerminalSteps = self.env.get_steps(behavior_names[0])
            reward = DecisionSteps.reward[0]
            if len(TerminalSteps.obs[-1]) != 0:
                done = 1
                next_state = TerminalSteps.obs
                reward += TerminalSteps.reward[0]
            else:
                next_state = DecisionSteps.obs
            steps += 1
            cur_state = next_state
            last_loc = robot_loc
            actions.append(action)
            robot_locs.append(robot_loc)
        if done  == 1:
            pass
        else:
            self.env.reset()
        return actions, robot_locs, target_loc, done
    
    def test_scratch(self, if_normal=True):
        max_steps = self.max_steps
        #self.env.reset()
        behavior_names = self.env.get_behavior_names()
        DecisionSteps, TerminalSteps = self.env.get_steps(behavior_names[0])
        self.env_index = int(DecisionSteps.obs[-1][0][-1])
        #self.dis_env = np.load('stage2_envs/env_{}.npy'.format(self.env_index+1)) 
        self.dis_env = np.load('res1101/env_{}.npy'.format(self.env_index+1)) 
        cur_state = DecisionSteps.obs
        actions, robot_locs = [], []
        robot_loc = [-cur_state[-1][0][5], 0.5, cur_state[-1][0][3]] 
        target_loc = [-cur_state[-1][0][2], 0.5, cur_state[-1][0][0]] 
        self.origin_connect_point = cal_connect_point(cur_state, self.dis_env)
        
        able_move = False
        r2t_path = AStarSearch(self.dis_env, (round(robot_loc[0]), round(robot_loc[-1])), 
                                   (round(target_loc[0]), round(target_loc[-1])))  
        robot2target = np.sqrt((robot_loc[0]-target_loc[0])**2+
                       (robot_loc[-1]-target_loc[-1])**2)
        able_see = in_view(self.dis_env, robot_loc,  target_loc)
        if able_see and len(r2t_path)<=25:
            able_move = True
        done, steps = False, 0
        jj=1
        #robot_locs.append(robot_loc)
        while not done and steps < max_steps:
            robot_loc = [-cur_state[-1][0][5], 0.5, cur_state[-1][0][3]] 
            if np.random.random() > 0.5:
                target_loc = [-cur_state[-1][0][2], 0.5, cur_state[-1][0][0]]
            else:
                target_loc = [np.random.randint(self.dis_env.shape[0]-2), 0.5, np.random.randint(self.dis_env.shape[0]-2)]
            robot2target = np.sqrt((robot_loc[0]-target_loc[0])**2+
                       (robot_loc[-1]-target_loc[-1])**2)

            robot2target_angle = (target_loc[0]-robot_loc[0],
                                  target_loc[-1]-robot_loc[-1])
            p_new = np.zeros((2,1))
            z1 = np.true_divide(robot2target_angle[0], np.sqrt(robot2target_angle[0]**2 + robot2target_angle[1]**2) + ZERO_TOLERANCE) 
            z2 = np.true_divide(robot2target_angle[1], np.sqrt(robot2target_angle[0]**2 + robot2target_angle[1]**2) + ZERO_TOLERANCE) 
            z_sum = np.sqrt(robot2target_angle[0]**2 + robot2target_angle[1]**2)
            able_see = in_view(self.dis_env, robot_loc, target_loc)
            if able_see and len(r2t_path) <= 25:
                able_move = True
            
            if able_move:    
                action = [np.random.normal(z1), np.random.normal(z2)]
            else:
                if np.random.random() > 0.5:
                    action = [np.random.normal(loc=np.random.normal(loc=z1)), np.random.normal(loc=np.random.normal(loc=z2))]
                else:
                    action = [np.random.normal(z2), np.random.normal(z1)]
            if hit_wall(self.dis_env, robot_loc, action):
                action = [np.random.normal(-z1), np.random.normal(-z2)]
            if if_normal:
                z3 = np.true_divide(action[0], np.sqrt(action[0]**2 + action[1]**2) + ZERO_TOLERANCE)
                z4 = np.true_divide(action[1], np.sqrt(action[0]**2 + action[1]**2) + ZERO_TOLERANCE)
            action = [z3,z4]

            self.env.set_actions(behavior_names[0], np.expand_dims(action,axis=0))
            self.env.step()
            DecisionSteps, TerminalSteps = self.env.get_steps(behavior_names[0])
            reward = DecisionSteps.reward[0]
            if len(TerminalSteps.obs[-1]) != 0:
                done = 1
                next_state = TerminalSteps.obs
                reward += TerminalSteps.reward[0]
            else:
                next_state = DecisionSteps.obs
            steps += 1
            cur_state = next_state
            actions.append(action)
            robot_locs.append(robot_loc)
        target_loc = [-cur_state[-1][0][2], 0.5, cur_state[-1][0][0]]
        if done  == 1:
            pass
        else:
            self.env.reset()
        return actions, robot_locs, target_loc, done

    def test_cnn(self, if_normal=True):
        max_steps = self.max_steps
        #self.env.reset()
        behavior_names = self.env.get_behavior_names()
        DecisionSteps, TerminalSteps = self.env.get_steps(behavior_names[0])
        self.env_index = int(DecisionSteps.obs[-1][0][-1])
        #self.dis_env = np.load('stage2_envs/env_{}.npy'.format(self.env_index+1)) 
        self.dis_env = np.load('res1101/env_{}.npy'.format(self.env_index+1)) 
        cur_state = DecisionSteps.obs
        actions, robot_locs = [], []
        robot_loc = [-cur_state[-1][0][5], 0.5, cur_state[-1][0][3]] 
        target_loc = [-cur_state[-1][0][2], 0.5, cur_state[-1][0][0]] 
        self.origin_connect_point = cal_connect_point(cur_state, self.dis_env)
        
        able_move = False
        r2t_path = AStarSearch(self.dis_env, (round(robot_loc[0]), round(robot_loc[-1])), 
                                   (round(target_loc[0]), round(target_loc[-1])))    
        able_see = in_view(self.dis_env, robot_loc, target_loc)
        if able_see and len(r2t_path)<30:
            able_move = True

        done, steps = False, 0
        jj=1
        while not done and steps < max_steps:
            move_loc = self.origin_connect_point[jj]
            robot_loc = [-cur_state[-1][0][5], 0.5, cur_state[-1][0][3]] 
            robot2target = np.sqrt((robot_loc[0]-move_loc[0])**2+
                                   (robot_loc[-1]-move_loc[-1])**2)
            if robot2target <= 1.5:
                if jj < len(self.origin_connect_point)-1:
                    jj += 1
                else:
                    pass
            robot2target_angle = (move_loc[0]-robot_loc[0],
                                  move_loc[-1]-robot_loc[-1])
            p_new = np.zeros((2,1))
            z1 = np.true_divide(robot2target_angle[0], np.sqrt(robot2target_angle[0]**2 + robot2target_angle[1]**2) + ZERO_TOLERANCE) + np.random.normal()*0.4
            z2 = np.true_divide(robot2target_angle[1], np.sqrt(robot2target_angle[0]**2 + robot2target_angle[1]**2) + ZERO_TOLERANCE) + np.random.normal()*0.4
            able_see = in_view(self.dis_env, robot_loc, target_loc)
            if able_see and robot2target <= 20:
                able_move = True
            else:
                able_move = False
            
            if able_move:    
                action = [z1, z2]
            else:
                if np.random.random()<0.3:
                    action = [np.random.normal(loc=np.random.normal(np.random.normal(z1))), np.random.normal(loc=np.random.normal(np.random.normal(loc=z2)))]
                elif np.random.random()<0.6:
                    action = [np.random.normal(loc=np.random.normal(np.random.normal(z2))), np.random.normal(loc=np.random.normal(np.random.normal(loc=z1)))]
                else:
                    action = (np.random.normal(), np.random.normal())
            if hit_wall(self.dis_env, robot_loc, action):
                action = [np.random.normal(-z1), np.random.normal(-z2)]
            if if_normal:
                z3 = np.true_divide(action[0], np.sqrt(action[0]**2 + action[1]**2) + ZERO_TOLERANCE)
                z4 = np.true_divide(action[1], np.sqrt(action[0]**2 + action[1]**2) + ZERO_TOLERANCE)
            action = [z3,z4]
            
            self.env.set_actions(behavior_names[0], np.expand_dims(action,axis=0))
            self.env.step()
            DecisionSteps, TerminalSteps = self.env.get_steps(behavior_names[0])
            reward = DecisionSteps.reward[0]
            if len(TerminalSteps.obs[-1]) != 0:
                done = 1
                next_state = TerminalSteps.obs
                reward += TerminalSteps.reward[0]
            else:
                next_state = DecisionSteps.obs
            steps += 1
            cur_state = next_state
            actions.append(action)
            robot_locs.append(robot_loc)
        if done  == 1:
            pass
        else:
            self.env.reset()
        return actions, robot_locs, target_loc, done

    def test_large(self, if_normal=True, move_action = None):
        max_steps = self.max_steps
        #self.env.reset()
        behavior_names = self.env.get_behavior_names()
        DecisionSteps, TerminalSteps = self.env.get_steps(behavior_names[0])
        self.env_index = int(DecisionSteps.obs[-1][0][-1])
        self.dis_env = np.load('res1101/env_large_{}.npy'.format(self.env_index+1)) 
        cur_state = DecisionSteps.obs
        actions, robot_locs = [], []
        robot_loc = [-cur_state[-1][0][5], 0.5, cur_state[-1][0][3]] 
        target_loc = [-cur_state[-1][0][2], 0.5, cur_state[-1][0][0]] 
        self.origin_connect_point = cal_connect_point(cur_state, self.dis_env)
        
        able_move = False
        r2t_path = AStarSearch(self.dis_env, (round(robot_loc[0]), round(robot_loc[-1])), 
                                   (round(target_loc[0]), round(target_loc[-1])))  
        robot2target = np.sqrt((robot_loc[0]-target_loc[0])**2+
                       (robot_loc[-1]-target_loc[-1])**2)
        able_see = in_view(self.dis_env, robot_loc,  target_loc)
        change_loc = False
        if len(r2t_path)>=30+10*np.random.random() and able_see == False:
            if np.random.random() < 0.3:
                pass
            else:    
                old_target = target_loc
                change_loc = True
                target_loc = [np.random.normal()*robot_loc[0], 0.5, np.random.normal()*robot_loc[1]]
        done, steps = False, 0
        jj=1

        #robot_locs.append(robot_loc)
        while not done and steps < max_steps:
            able_move = False
            if able_see and change_loc:
                target_loc = old_target
            move_loc = self.origin_connect_point[jj]
            robot_loc = [-cur_state[-1][0][5], 0.5, cur_state[-1][0][3]] 
            robot2connect = np.sqrt((robot_loc[0]-move_loc[0])**2+
                       (robot_loc[-1]-move_loc[-1])**2)
            if robot2connect <= 1:
                if jj < len(self.origin_connect_point)-1:
                    jj += 1
                else:
                    pass
            robot2target = np.sqrt((robot_loc[0]-target_loc[0])**2+
                       (robot_loc[-1]-target_loc[-1])**2)
            if robot2target > 25+5*np.random.random():
                if np.random.random() < 0.3:
                    pass
                else:
                    move_loc = [np.random.normal()*robot_loc[0], 0.5, np.random.normal()*robot_loc[1]]

            robot2target_angle = (move_loc[0]-robot_loc[0],
                                  move_loc[-1]-robot_loc[-1])
            z1 = np.true_divide(robot2target_angle[0], np.sqrt(robot2target_angle[0]**2 + robot2target_angle[1]**2) + ZERO_TOLERANCE) + np.random.normal()*0.4
            z2 = np.true_divide(robot2target_angle[1], np.sqrt(robot2target_angle[0]**2 + robot2target_angle[1]**2) + ZERO_TOLERANCE) + np.random.normal()*0.4
            z_sum = np.sqrt(robot2target_angle[0]**2 + robot2target_angle[1]**2)
            able_see = in_view(self.dis_env, robot_loc, target_loc)
            if robot2target <= np.random.random()*5+25 and able_see:
                able_move = True
            if robot2target <= np.random.random()*3+12:
                able_move = True
            if able_move:    
                action = [z1, z2]
            else:
                action = [np.random.normal(loc=np.random.normal(z1)), np.random.normal(loc=np.random.normal(z2))]

            if hit_wall(self.dis_env, robot_loc, action):
                action = [np.random.normal(-z1), np.random.normal(-z2)]
            if if_normal:
                z3 = np.true_divide(action[0], np.sqrt(action[0]**2 + action[1]**2) + ZERO_TOLERANCE)
                z4 = np.true_divide(action[1], np.sqrt(action[0]**2 + action[1]**2) + ZERO_TOLERANCE)
            action = [z3,z4]
            if move_action != None:
                action = [move_action[steps+1][0]-move_action[steps][0], move_action[steps+1][-1]-move_action[steps][-1]]
                
            self.env.set_actions(behavior_names[0], np.expand_dims(action,axis=0))
            self.env.step()
            DecisionSteps, TerminalSteps = self.env.get_steps(behavior_names[0])
            reward = DecisionSteps.reward[0]
            if len(TerminalSteps.obs[-1]) != 0:
                done = 1
                next_state = TerminalSteps.obs
                reward += TerminalSteps.reward[0]
            else:
                next_state = DecisionSteps.obs
            steps += 1
            cur_state = next_state
            actions.append(action)
            robot_locs.append(robot_loc)
            
        target_loc = [-cur_state[-1][0][2], 0.5, cur_state[-1][0][0]]
        if done  == 1:
            pass
        else:
            self.env.reset()
        return actions, robot_locs, target_loc, done