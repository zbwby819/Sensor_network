# -*- coding: utf-8 -*-
"""
Created on Fri Sep  4 10:30:05 2020

@author: azrael
"""

import numpy as np

num_dot = 100
com_dis_near = 10
com_dis_far = 15

env = np.load('env_3.npy')
free_space = np.where(env==0)
anchor_index = np.arange(len(free_space[0]))
np.random.shuffle(anchor_index)
dot_loc = []
for j in range(num_dot):
    dot_loc.append((free_space[0][anchor_index[j]], free_space[1][anchor_index[j]]))
    
sensor_loc = []
sensor_index = np.random.choice(anchor_index, 1) 
sensor_loc.append((free_space[0][sensor_index[0]], free_space[1][sensor_index[0]]))
for s_i in range(30):
    all_ends = []
    for i in range(len(env[0])):
        for j in range(len(env[1])):
            if env[i,j] == 0 and (abs(loc_start[0]-i)+abs(loc_start[1]-j)) <= com_dis_far 
                             and (abs(loc_start[0]-i)+abs(loc_start[1]-j)) >= com_dis_near:
                all_ends.append((i,j))
    

    
z_env = np.zeros((40,40))
z1 = z_env.copy()

z_env = np.zeros((40,40))
for i in range(len(data)):
    z_env[data[i][0]-1][data[i][1]-1] = 1

################################################## 0911 mat2npy
import scipy.io as io
import numpy as np

select_map = 1
select_layout = 9
select_case = 5
i = 2

matr = io.loadmat('train_env/env_{}/{}/{}_sensor_location_mat.mat'.format(select_map, select_layout, select_case))
data_loc = matr['sensor'] 
matr1 = io.loadmat('train_env/env_{}/{}/{}_adjency_matrix1.mat'.format(select_map, select_layout, select_case))
data_ad1 = matr1['adjency_matrix']
matr2 = io.loadmat('train_env/env_{}/{}/{}_adjency_matrix2.mat'.format(select_map, select_layout, select_case))
data_ad2 = matr2['adjency_matrix2']
matr3 = io.loadmat('a_env0911/env_{}/{}-{}-{}_adjency_matrix3.mat'.format(select_map, select_map, select_layout, select_case))
data_ad3 = matr3['adjency_matrix3']
matr4 = io.loadmat('a_env0911/env_{}/{}-{}-{}_adjency_matrix4.mat'.format(select_map, select_map, select_layout, select_case))
data_ad4 = matr4['adjency_matrix4']

np.save('ad_matrix0911/env_{}_{}_{}_loc.npy'.format(select_map, select_layout, i), data_loc)
np.save('ad_matrix0911/env_{}_{}_{}_ad1.npy'.format(select_map, select_layout, i), data_ad1)
np.save('ad_matrix0911/env_{}_{}_{}_ad2.npy'.format(select_map, select_layout, i), data_ad2)
np.save('ad_matrix0911/env_{}_{}_{}_ad3.npy'.format(select_map, select_layout, i), data_ad3)
np.save('ad_matrix0911/env_{}_{}_{}_ad4.npy'.format(select_map, select_layout, i), data_ad4)

#################################################### laod npy 

z_1_1 = np.load('ad_matrix0911/env_1_9_1_loc.npy')
z_1_1_ad1 = np.load('ad_matrix0911/env_1_9_1_ad1.npy')
z_1_1_ad2 = np.load('ad_matrix0911/env_1_9_1_ad2.npy')
z_1_1_ad3 = np.load('ad_matrix0911/env_1_9_1_ad3.npy')
z_1_1_ad4 = np.load('ad_matrix0911/env_1_9_1_ad4.npy')


####################################################  normal-0911
for i in range(9):
    for j in range(100):
        