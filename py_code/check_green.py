# -*- coding: utf-8 -*-
"""
Created on Sun Aug 23 23:33:28 2020

@author: azrael
"""
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image

batch_size = 2000
num_sensors = 9
select_group2 = 1
select_case_env4 = np.arange(batch_size)
all_sensors = ['sensor_1', 'sensor_2', 'sensor_3', 'sensor_4',
               'sensor_5', 'sensor_6', 'sensor_7', 'sensor_8',
               'sensor_9']

filePath_env4 = 'training_env4_{}/sensor_1/1'.format(select_group2)
filelist_env4 = os.listdir(filePath_env4)
filelist_env4.sort(key = lambda x: int(x[:-4]))

def check_green(img):
    z_img0 = img[:,:,0]
    z_img1 = img[:,:,1]
    z_img2 = img[:,:,2]
    z0 = np.where(z_img0<=40)
    z2 = np.where(z_img2<=64)
    if len(z0[0])>50 and len(z2[0])>50:
        return 1
    return 0 

def check_green(img):
    z_img0 = img[:,:,0]
    z_img1 = img[:,:,1]
    z_img2 = img[:,:,2]
    z0 = np.where((z_img0<=99)&(z_img0>=90))
    z1 = np.where((z_img1<=180)&(z_img1>=150))
    z2 = np.where((z_img2<=125)&(z_img2>=110))
    if len(z0[0])>8 and len(z1[0])>8 and len(z2[0])>8:
        return 1
    return 0


all_sensor_view = []
for i in range(batch_size): 
    ####### for env4
    all_sensor_input_env4 = np.zeros((num_sensors, 84, 84*4, 3))
    is_view = []
    all_view = []
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
        is_view.append(check_green(img_array_1))
        is_view.append(check_green(img_array_2))
        is_view.append(check_green(img_array_3))
        is_view.append(check_green(img_array_4))
        if sum(is_view) >=1:
            all_view.append(1)
        else:
            all_view.append(0)
    all_sensor_view.append(all_view)
np.save('is_view_env4_{}'.format(select_group2), np.array(all_sensor_view))
    
#############################################################

from loc2dir import theta

select_group = 1
env_index = 4
if env_index == 3:
    env = np.load('env_3.npy') 
    if select_group == 1:    
        sensor_loc = [(7, 0, 7),  (14, 0, 14), (27, 0, 20), (37, 0, 9),
                      (7, 0, 28), (15, 0, 39), (18, 0, 26), (28, 0, 37), (39, 0, 26)] #env-3-1
        label_path = "target_env3_1.txt"
    if select_group == 2:     
        sensor_loc = [(19, 0, 9), (5, 0, 10),  (28, 0, 13), (36, 0, 4),
                      (9, 0, 24), (12, 0, 38), (22, 0, 23), (26, 0, 35), (35, 0, 29)] #env-3-2
        label_path = "target_env3_2.txt"
    if select_group == 3:     
        sensor_loc = [(19, 0, 9), (5, 0, 7),  (24, 0, 18), (31, 0, 9),
                      (8, 0, 21), (6, 0, 35), (19, 0, 31), (29, 0, 39), (31, 0, 29)] #env-3-3
        label_path = "target_env3_3.txt"
    if select_group == 4:     
        sensor_loc = [(15, 0, 4), (28, 0, 11), (39, 0, 19), (35, 0, 5),
                      (7, 0, 22), (12, 0, 34), (21, 0, 23), (29, 0, 34), (32, 0, 25)] #env-3-4
        label_path = "target_env3_4.txt"
    if select_group == 5:     
        sensor_loc = [(16, 0, 0), (7, 0, 10),  (26, 0, 10), (34, 0, 7),
                      (6, 0, 24), (12, 0, 38), (18, 0, 19), (25, 0, 31), (32, 0, 21)] #env-3-5
        label_path = "target_env3_5.txt"
    if select_group == 6:     
        sensor_loc = [(18, 0, 0),  (7, 0, 9),   (25, 0, 12), (32, 0, 0),
                      (10, 0, 23), (13, 0, 37), (21, 0, 32), (35, 0, 35), (38, 0, 19)] #env-3-6
        label_path = "target_env3_6.txt"
    if select_group == 7:
        sensor_loc = [(16, 0, 3),  (14, 0, 14), (3, 0, 22), (12, 0, 31),
              (21, 0, 39), (23, 0, 24), (26, 0, 12), (33, 0, 33), (31, 0, 9)] #env-3-7
        label_path = "target_env3_7.txt"
if env_index == 4:
    env = np.load('env_4.npy')  
    if select_group == 1:     
        sensor_loc = [(12, 0, 1),  (6, 0, 13),  (7, 0, 26), (17, 0, 16),
                      (19, 0, 30), (27, 0, 20), (31, 0, 8), (33, 0, 31), (39, 0, 19)] #env-4-1
        label_path = "target_env4_1.txt"
    if select_group == 2:     
        sensor_loc = [(5, 0, 4),   (0, 0, 17),  (4, 0, 31),   (13, 0, 13),
                      (17, 0, 26), (27, 0, 16), (28, 0, 3), (30, 0, 30), (39, 0, 12)] #env-4-2
        label_path = "target_env4_2.txt"
    if select_group == 3:     
        sensor_loc = [(10, 0, 13), (0, 0, 24), (7, 0, 37), (22, 0, 19),
                      (21, 0, 33), (29, 0, 13), (26, 0, 7), (35, 0, 32), (39, 0, 3)] #env-4-3
        label_path = "target_env4_3.txt"
    if select_group == 4:     
        sensor_loc = [(10, 0, 8),  (17, 0, 21), (3, 0, 39), (16, 0, 34),
                      (29, 0, 29), (29, 0, 19), (23, 0, 8), (39, 0, 39), (39, 0, 21)] #env-4-4
        label_path = "target_env4_4.txt"
    if select_group == 5:     
        sensor_loc = [(10, 0, 3), (5, 0, 13), (0, 0, 26), (18, 0, 23),
                      (11, 0, 35), (29, 0, 26), (23, 0, 10), (25, 0, 39), (35, 0, 13)] #env-4-5
        label_path = "target_env4_5.txt"
    if select_group == 6:     
        sensor_loc = [(10, 0, 5),  (8, 0, 19),  (2, 0, 31), (15, 0, 28),
                      (18, 0, 10), (25, 0, 17), (29, 0, 6), (34, 0, 28), (38, 0, 15)] #env-4-6
        label_path = "target_env4_6.txt"
    if select_group == 7:  
        sensor_loc = [(13, 0, 13),  (4, 0, 15), (7, 0, 26), (14, 0, 37),
              (22, 0, 27), (26, 0, 7), (29, 0, 19), (29, 0, 21), (34, 0, 33)] #env-3-7
        label_path = "target_env4_7.txt"
        
num_sensors = len(all_sensors)
target_label = open(label_path,"r") 
lines = target_label.readlines() 
batch_label = []

select_case = np.arange(1,len(lines))
all_sensor_view = []
for i in range(len(lines)):
    label_index = lines[select_case[i]].index(')')
    label_target = int(lines[select_case[i]][label_index+1:-1])
    x_index_1 = lines[select_case[i]].index('(')
    x_index_2 = lines[select_case[i]].index(',')
    label_x = float(lines[select_case[i]][x_index_1+1:x_index_2])
    z_index_1 = lines[select_case[i]].index(',', x_index_2+1)
    z_index_2 = lines[select_case[i]].index(')')  
    label_z = float(lines[select_case[i]][z_index_1+2:z_index_2])
    #t_x, t_y, t_z = change_axis(env, (label_x, 0, label_z))
    t_x, t_y, t_z = (label_x, 0, label_z)
    sensor_direction = []
    is_view = []
    all_view = []
    for j in range(num_sensors):
        #s_x, s_y, s_z = change_axis(env, sensor_loc[j])
        s_x, s_y, s_z = sensor_loc[j]
        #s_path = AStarSearch(env, (round(s_x), round(s_z)), (round(t_x), round(t_z)))
        s_path = theta(env, (round(s_x), round(s_z)), (round(t_x), round(t_z)))
        if len(s_path) == 2:
            is_view.append(1)
        else:
            is_view.append(0)
        if s_path[0] == s_path[1]:
            print('num_group:', select_group, '   num_case:', select_case[i], '   num_sensor:', j)
            print('target_loc:', (label_x, 0, label_z))
    all_sensor_view.append(is_view)
np.save('is_view_env4_{}'.format(select_group), np.array(all_sensor_view))
 

