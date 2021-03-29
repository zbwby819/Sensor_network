# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 21:49:53 2020

@author: azrael
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.keras.backend as K
from mlagents_envs.environment import UnityEnvironment
from stage_0_dyna import train_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing import image
from spektral.layers import GraphConv,GraphAttention
from spektral.utils import localpooling_filter
from loc2dir import angle2xy, theta
import os, gc

init_lr = 3e-5
batch_size = 16
data_per_epoch=128
train_iter = 10000
origin_sensors = ['sensor_1', 'sensor_2', 'sensor_3', 'sensor_4',
                  'sensor_5', 'sensor_6', 'sensor_7', 'sensor_8',
                  'sensor_9']

print('load_enviroment for stage-1, prepare to hit play button!')
env = UnityEnvironment(file_name=None, side_channels=[])
# Start interacting with the evironment.
env.reset()
env.step()
print('env load success!')
# spdyer first, then unity play
def train_dyna(env):
    behavior_names = env.get_behavior_names()
    
    for num_iter in range(1, train_iter):
        #print('start_training round:', num_iter)
        new_lr = init_lr
        if num_iter == int(train_iter/4):
            new_lr = init_lr/10
            print('new_learning:', init_lr)
        if num_iter == int(train_iter/2):
            new_lr = init_lr/100
            print('new_learning:', init_lr)
        if num_iter == int(train_iter/1.3):
            new_lr = init_lr/1000
            print('new_learning:', init_lr)
        
        env.set_actions(behavior_names[0], np.expand_dims([1],axis=0))  # 1--save img   2--new episode
        env.step()
        (DecisionSteps, TerminalSteps) = env.get_steps(behavior_names[0]) 
        cur_state = DecisionSteps.obs
        select_env = int(cur_state[-1][0][6])
        num_sensors = int(cur_state[-1][0][7])
        cur_step = int(cur_state[-1][0][8])
        robot_loc = []
        target_loc = []
        sensor_locs = []
        for i in range(num_sensors):
            sensor_locs.append((int(cur_state[-1][0][9+i*3]), int(cur_state[-1][0][11+i*3])))
        ad_mat1, ad_mat2, ad_mat3, ad_mat4 = cal_admatrix_sensor(sensor_locs)
        np.save('train_data1101/sensor_locs/epoch_{}_sensors.npy'.format(num_iter), sensor_locs)
        np.save('train_data1101/epoch_{}_env.npy'.format(num_iter), select_env)
        #np.save('train_data1101/ad_matrix/epoch_{}_ad1.npy', ad_mat1)
        #np.save('train_data1101/ad_matrix/epoch_{}_ad2.npy', ad_mat2)
        #np.save('train_data1101/ad_matrix/epoch_{}_ad3.npy', ad_mat3)
        #np.save('train_data1101/ad_matrix/epoch_{}_ad4.npy', ad_mat4)
        
        model = train_model(num_sensors, num_hop=4, input_shape=(84,84*4,3), gnn_unit = 128)
        model.compile(optimizer=Adam(learning_rate=new_lr, clipnorm=3), loss='mse') 
        model.load_weights('train_data1101/gnn_dyna_1102.h5')
        print('training epoch:{}'.format(num_iter))
        history = train_sample(model, select_env, sensor_locs, num_sensors, num_iter,
                               ad_mat1, ad_mat2, ad_mat3, ad_mat4, batch_size,
                               if_att=True)
            
        hist_df = pd.DataFrame(history.history)
        hist_csv_file = 'history1102_dyna.csv'
        with open(hist_csv_file, mode='a') as f:
            hist_df.to_csv(f) 
        del model
        del history
        #env.set_actions(behavior_names[0], np.expand_dims([2],axis=0))  # 1--save img   2--new episode
        #env.step()
        gc.collect()
        env.reset()
    
    env.close()

def train_sample(model, select_env, sensor_locs, num_sensors, num_iter,
                 ad_matrix1, ad_matrix2, ad_matrix3, ad_matrix4, batch_size,
                 if_att = True):
    input_data, output_data = load_data(num_iter, select_env, num_sensors, sensor_locs)
    input_data2 = np.array(input_data)
    output_data2 = np.array(output_data)

    batch_matrix1 = np.zeros((data_per_epoch, num_sensors, num_sensors))
    batch_matrix2 = np.zeros((data_per_epoch, num_sensors, num_sensors))
    batch_matrix3 = np.zeros((data_per_epoch, num_sensors, num_sensors))
    batch_matrix4 = np.zeros((data_per_epoch, num_sensors, num_sensors))
        
    for j in range(data_per_epoch):
        if if_att == False:
            batch_matrix1[j] = localpooling_filter(ad_matrix1)
            batch_matrix2[j] = localpooling_filter(ad_matrix2)
            batch_matrix3[j] = localpooling_filter(ad_matrix3)
            batch_matrix4[j] = localpooling_filter(ad_matrix4)
        else:
            batch_matrix1[j] = ad_matrix1
            batch_matrix2[j] = ad_matrix2
            batch_matrix3[j] = ad_matrix3
            batch_matrix4[j] = ad_matrix4

    batch_input = []
    batch_output = []
    for k in range(num_sensors):  
        batch_input.append(input_data2[:,k])
        batch_output.append(output_data2[:,k])
    for k in range(4):
        exec("batch_input.append(batch_matrix{})".format(k+1))
    
    history = model.fit(x = batch_input, 
                        y = batch_output,
                        batch_size=batch_size, epochs=1, shuffle = True)
                #callbacks=[TensorBoard(log_dir='mytensorboard')])
    model.save('train_data1101/gnn_dyna_1102.h5')
    del model
    K.clear_session()
    gc.collect()
    #history = 0
    return history

def cal_admatrix_sensor(sensor_loc, num_sensors=None):
    #sensor_loc = sensor_loc.astype('float')
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

def read_target(label_path, select_case):
    target_loc = []    
    target_label = open(label_path,"r") 
    lines = target_label.readlines() 
    for i in range(len(select_case)):
        label_index = lines[select_case[i]].index(')')
        label_target = int(lines[select_case[i]][label_index+1:-1])
        x_index_1 = lines[select_case[i]].index('(')
        x_index_2 = lines[select_case[i]].index(',')
        label_x = float(lines[select_case[i]][x_index_1+1:x_index_2])
        z_index_1 = lines[select_case[i]].index(',', x_index_2+1)
        z_index_2 = lines[select_case[i]].index(')')  
        label_z = float(lines[select_case[i]][z_index_1+2:z_index_2])
        #t_x, t_y, t_z = change_axis(env, (label_x, 0, label_z))
        target_loc.append((label_x, label_z))
    return target_loc

def load_data(num_iter, select_env, num_sensors, sen_locs, batch_size=data_per_epoch): 
    target_loc = []
    robot_loc = []

    select_case = np.arange(batch_size)
    #select_case = [np.random.randint(500) for _ in range(batch_size)]  
    tar_loc = read_target('train_data1101/epoch_{}_target.txt'.format(num_iter), select_case)
    rob_loc = read_target('train_data1101/epoch_{}_robot.txt'.format(num_iter), select_case)
    dis_env = np.load('train_data1101/env_map/env_{}.npy'.format(select_env+1))
    target_loc.append(tar_loc)
    robot_loc.append(rob_loc)

    target_label = []
    batch_input = []
    filePath = 'training/sensor_1/1'
    filelist = os.listdir(filePath)
    filelist.sort(key = lambda x: int(x[:-4]))
    for s_i in range(len(select_case)):
        sensor_dir = []
        image_input = np.zeros((num_sensors, 84, 84*4, 3))
        for j in range(num_sensors):
            s_x, s_z = sen_locs[j]
            s_path = theta(dis_env, (round(s_x), round(s_z)), (round(-tar_loc[s_i][1]), round(tar_loc[s_i][0])))
            if s_path[0] == s_path[1]:
                #print('wrong position!')
                s_angle = angle2xy((s_x, s_z), (tar_loc[s_i][0], tar_loc[s_i][1]))
            elif len(s_path) == 2:
                s_angle = angle2xy((s_x, s_z), (tar_loc[s_i][0], tar_loc[s_i][1]))
            else:
                s_angle = angle2xy(s_path[0], s_path[1])
            sensor_dir.append(s_angle)
            
            #######################################################image
            sensor_path = 'training/sensor_' + str(j+1)
            img_1 = image.load_img(sensor_path+'/1/'+filelist[select_case[s_i]], target_size=(84,84))  #height-width
            img_array_1 = image.img_to_array(img_1)
            img_2 = image.load_img(sensor_path+'/2/'+filelist[select_case[s_i]], target_size=(84,84))  #height-width
            img_array_2 = image.img_to_array(img_2)
            img_3 = image.load_img(sensor_path+'/3/'+filelist[select_case[s_i]], target_size=(84,84))  #height-width
            img_array_3 = image.img_to_array(img_3)
            img_4 = image.load_img(sensor_path+'/4/'+filelist[select_case[s_i]], target_size=(84,84))  #height-width
            img_array_4 = image.img_to_array(img_4)  
            image_input[j,:, 84*3:84*4,:] = img_array_1/255 
            image_input[j,:, 84*2:84*3,:] = img_array_2/255
            image_input[j,:, 84*1:84*2,:] = img_array_3/255
            image_input[j,:, 84*0:84*1,:] = img_array_4/255  
        
        target_label.append(sensor_dir)
        batch_input.append(image_input)
            
    return batch_input, target_label


train_dyna(env)