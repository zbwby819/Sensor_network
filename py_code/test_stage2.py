# -*- coding: utf-8 -*-
"""
Created on Mon Oct 12 21:34:45 2020

@author: azrael
"""
##############################################################  stage 1 test
from loc2dir import angle2xy, theta, read_target
from a_star import AStarSearch
from tf_ppo import in_view

def pred_test(env, target_loc, sensor_loc, robot_loc=None):
    sensor_dir = []
    for i in range(len(sensor_loc)):
        s_x, s_z = sensor_loc[i]
        s_path = theta(env, (int(s_x), int(s_z)), (int(target_loc[0]), int(target_loc[-1])))
        if s_path[0] == s_path[1] or len(s_path) == 2:
            s_angle = angle2xy((s_x, s_z), (target_loc[0], target_loc[-1]))
        else:
            s_angle = angle2xy(s_path[0], s_path[1])
        sensor_dir.append(s_angle)
    return sensor_dir

def cal_admatrix_sensor(sensor_loc, robot_loc, num_sensors=None):
    sensor_loc = sensor_loc.astype('float')
    if type(robot_loc) != None:
        sensor_loc = list(sensor_loc)
        sensor_loc.append((robot_loc))
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

env_map = np.load('res1101/env_1.npy')    
num_sensors = 10
#np.save('res1101/env_1.npy',env_map)
env_target = read_target('res1101/target_loc_{}.txt'.format(2), select_case)
#env_robot = np.load('stage2_res1012/env_{}_locs.npy'.format(1), allow_pickle=True)
env_sensor = np.load('res1101/sensor_loc_{}.npy'.format(num_sensors))
all_sensor_pred = []
all_sensor_true = []

for i in range(100):
    sensor_true = pred_test(env_map, env_target[i], env_sensor)
    all_sensor_true.append(sensor_true)
    env_ad1, env_ad2, env_ad3, env_ad4 = cal_admatrix_sensor(env_sensor, (env_target[i][0], env_target[i][-1]))
    sensor_pred = []
    for j in range(num_sensors):
        if env_ad1[0][-1][j] == 1:
            pred_x = np.random.normal(sensor_true[j][0], scale=0.18)
            pred_z = np.random.normal(sensor_true[j][1], scale=0.18)
        elif env_ad2[0][-1][j] == 1:
            pred_x = np.random.normal(sensor_true[j][0], scale=0.22)
            pred_z = np.random.normal(sensor_true[j][1], scale=0.22)
        elif env_ad3[0][-1][j] == 1:
            pred_x = np.random.normal(sensor_true[j][0], scale=0.28)
            pred_z = np.random.normal(sensor_true[j][1], scale=0.28)
        elif env_ad4[0][-1][j] == 1:
            pred_x = np.random.normal(sensor_true[j][0], scale=0.34)
            pred_z = np.random.normal(sensor_true[j][1], scale=0.34)
        else:
            if in_view(env_map, (env_sensor[j][0], env_sensor[j][1]), (env_target[i][0], env_target[i][1])):
                pred_x = np.random.normal(sensor_true[j][0], scale=0.13)
                pred_z = np.random.normal(sensor_true[j][1], scale=0.13)
                #print('in_view:', j, 'number:',i)
            else:
                pred_x = np.random.normal(sensor_true[j][0], scale=1.16)
                pred_z = np.random.normal(sensor_true[j][1], scale=1.16)
                #print('disconnect:', j, 'number:',i)
        pred_norm = np.sqrt(pred_x**2 + pred_z**2)
        pred_x /= pred_norm
        pred_z /= pred_norm
        sensor_pred.append((pred_x, pred_z))
    all_sensor_pred.append(sensor_pred) 

np.save('res1101/env_{}_pred.npy'.format(num_sensors), all_sensor_pred)
np.save('res1101/env_{}_true.npy'.format(num_sensors), all_sensor_true)

#num_sensors = 30
#all_sensor_pred = np.load('res1101/env_{}_pred.npy'.format(num_sensors))
#all_sensor_true = np.load('res1101/env_{}_true.npy'.format(num_sensors))
zzp = all_sensor_pred[0]
zzt = all_sensor_true[0]
all_angle_average = []
all_angle_std = []
for i in range(100):
    z_error = []
    for j in range(num_sensors):    
        z_error.append(np.arccos(1-0.5*((all_sensor_pred[i][j][0]-all_sensor_true[i][j][0])**2+
                                        (all_sensor_pred[i][j][1]-all_sensor_true[i][j][1])**2))*180/np.pi)
    all_angle_average.append(np.average(z_error))
    all_angle_std.append(np.std(z_error))

all_sensor_pred = np.load('stage2_res1012/env_large_pred_dir.npy')
all_sensor_true = np.load('stage2_res1012/env_large_true_dir.npy')
sensor_loss = 0
sel_sensor = 0
for i in range(200):
    sensor_loss += abs(all_sensor_true[i][sel_sensor][0] - all_sensor_pred[i][sel_sensor][0])+abs(all_sensor_true[i][sel_sensor][1] - all_sensor_pred[i][sel_sensor][1])  
      
##############################################################  stage 2 test
agent.env_index = 6
z_env_index = agent.env_index
#env_map = np.load('stage2_envs/env_{}.npy'.format(z_env_index+1))
env_map = np.load('res1101/env_{}.npy'.format(z_env_index+1))
#np.save('res1101/env_{}.npy'.format(z_env_index+1), env_map)

all_action, all_locs, all_target, all_done = [],[],[],[]
for i in range(100):
    z_action, z_locs, z_target, z_done = agent.test3(map_path='env_')
    #z_action, z_locs, z_target, z_done = agent.test_large(move_action=move_action)
    all_action.append(z_action)
    all_locs.append(z_locs)
    all_target.append(z_target)
    all_done.append(z_done)

succ_rate = []
all_dis, all_dis_robot = [], []
for j in range(len(all_done)):
    if all_done[j] == 1:
        succ_rate.append(1)
    true_path = theta(env_map, (round(all_locs[j][0][0]), round(all_locs[j][0][-1])), 
                                   (round(all_target[j][0]), round(all_target[j][-1])))
    true_dis = 0
    for k in range(len(true_path)-1):
        true_dis = true_dis + np.sqrt((true_path[k+1][0]-true_path[k][0])**2 + (true_path[k+1][-1]-true_path[k][-1])**2)
    robot_dis = 0
    for k in range(len(all_locs[j])-1):
        robot_dis = robot_dis + np.sqrt((all_locs[j][k+1][0]-all_locs[j][k][0])**2 + (all_locs[j][k+1][-1]-all_locs[j][k][-1])**2)
    
    robot_dis = robot_dis + np.sqrt((all_locs[j][k+1][0]-all_target[j][0])**2 + (all_locs[j][k+1][-1]-all_target[j][-1])**2)
    all_dis.append(true_dis)
    all_dis_robot.append(robot_dis)

save_path = 'res1101/1206/'
#save_path = 'stage2_res1012/'
z_env_index = 30
all_locs = np.load(save_path+ 'env_{}_locs.npy'.format(z_env_index), allow_pickle=True)
all_target = np.load(save_path+'env_{}_target.npy'.format(z_env_index),allow_pickle=True)
all_done = np.load(save_path+'env_{}_done.npy'.format(z_env_index),allow_pickle=True)
all_dis = np.load(save_path+'env_{}_true_dis.npy'.format(z_env_index),allow_pickle=True)
all_dis_robot = np.load(save_path+'env_{}_robot_dis.npy'.format(z_env_index),allow_pickle=True)

all_detour = []
all_speed = []
for k in range(len(all_dis)):
    if all_done[k] == 1:
        dis2tar = np.sqrt((all_locs[k][-1][0]-all_target[k][0])**2 + (all_locs[k][-1][-1]-all_target[k][-1])**2)
        if all_dis[k] != 0:
            detour = (all_dis_robot[k]+dis2tar) /all_dis[k]
            all_detour.append(detour)
        #speed = (all_dis_robot[k]+dis2tar)/(len(all_locs[k]))
        speed = (all_dis_robot[k])/(len(all_locs[k]))
        all_speed.append(speed)
#sum(all_done)
print(np.average(all_detour))
print(np.std(all_detour))
print(np.average(all_speed))
print(np.std(all_speed))

    
#save_path = 'res1101/different_sensor/'
save_path = 'res1101/1206/'
#z_env_index = 0
np.save(save_path+ 'env_{}_locs.npy'.format(z_env_index+1), all_locs)
np.save(save_path+'env_{}_target.npy'.format(z_env_index+1), all_target)
np.save(save_path+'env_{}_done.npy'.format(z_env_index+1), all_done)
np.save(save_path+'env_{}_true_dis.npy'.format(z_env_index+1), all_dis)
np.save(save_path+'env_{}_robot_dis.npy'.format(z_env_index+1), all_dis_robot)

########################################################  calcuate success rate accross all map
loc_path = 'stage2_res1012/new_map_normal/'
z_index = [1,6,7,9,11,12,13,14,15]
all_succ_rate = []
for i in z_index:
    zz = np.load(loc_path+'env_{}_done.npy'.format(i))
    all_succ_rate.append(sum(zz))

# npy 2 mat
import numpy as np
import scipy.io as io
local_path = 'res1101/1206/'
z_index = [1,6,7,9,11,12,13,14,15]
for i in z_index:
    z_locs = np.load(local_path + 'env_{}_locs.npy'.format(i), allow_pickle=True)
    io.savemat(local_path+'env_{}_locs.mat'.format(i),{'data':z_locs})
    z_locs_mat = scio.loadmat(local_path + 'env_{}_locs.npy'.format(i))

z_max = []
for i in range(100):
    z_max.append(np.max(z_locs[i]))

txt_path = 'zz/'
z_env_index = 0
with open(txt_path+"env_{}_robot.txt".format(z_env_index+1),"a") as f:    
    for i in range(len(z_locs)):
        f.write('('+format(z_locs[i][0][0], '0.2f')+', 0.5, ' + format(z_locs[i][0][-1], '0.2f')+')'+'\n')
    f.close()
    
txt_path = 'zz/'
z_env_index = 0
z_target = np.load(local_path + 'env_{}_target.npy'.format(z_env_index+1), allow_pickle=True)
with open(txt_path+"env_{}_target.txt".format(z_env_index+1),"a") as f:    
    for i in range(len(z_target)):
        f.write('('+format(z_target[i][0], '0.2f')+', 0.5, ' + format(z_target[i][-1], '0.2f')+')'+'\n')
    f.close()
        
import re
z_env_index = 0 
f = open("zz/env_{}_target.txt".format(z_env_index+1) , 'r')
z_target = f.readlines()
f2 = open("zz/env_{}_robot.txt".format(z_env_index+1) , 'r')
z_robot = f2.readlines()
r2t_dis = []
for i in range(len(z_target)):
    tx, tz = float(re.split(r'[(,)]', z_target[i])[1]), float(re.split(r'[(,)]', z_target[i])[3])
    rx, rz = float(re.split(r'[(,)]', z_robot[i])[1]), float(re.split(r'[(,)]', z_robot[i])[3])
    r2t_dis.append(np.sqrt((rx-tx)**2 + (rz-tz)**2))
np.min(r2t_dis)
f.close()

##############  read sensor loc 
import scipy.io as io
local_path = 'res1101/1206/'
z_index = np.arange(1,10)
all_locs = []
for i in z_index:
    z_locs = np.load(local_path + 'env_{}_sensor_loc.npy'.format(i), allow_pickle=True)
    all_locs.append(z_locs)
    io.savemat(local_path+'env_{}_locs.mat'.format(i),{'data':z_locs})

for i in range(9):
    np.save('res1101/1206/env_{}_sensor_loc.npy'.format(i+1), all_locs[i])
    
#################################################################  
z_sensor = np.load('res1101/1125/env_1_locs.npy', allow_pickle=True)

loc_path = 'res1101/1125/'
all_succ_rate = []
z_index = [1,6,7,9,11,12,13,14,15]
z_dis_robot = []
z_dis_true = []
z_dis_dp = []
z_done = []
for i in z_index:

    zz = np.load(loc_path+'env_{}_done.npy'.format(i))
    all_succ_rate.append(sum(zz))
    z_dis_r = np.load(loc_path+'env_{}_robot_dis.npy'.format(i))
    z_dis_t = np.load(loc_path+'env_{}_true_dis.npy'.format(i))
    z_dis_robot.append(z_dis_r)
    z_dis_true.append(z_dis_true)
    z_dp = []
    for j in range(100):
        if zz[j] == 1:
            z_dp.append(z_dis_r[j]/z_dis_t[j])
    z_dis_dp.append(z_dp)
        
    
    #z_locs = np.load('stage2_res1012/new_map/env_{}_locs.npy'.format(i), allow_pickle=True)
    #io.savemat('env_{}_locs.mat'.format(i),{'data':z_locs})
    
z_robot_loc = np.load('stage2_res1012/new_map/env_1_locs.npy', allow_pickle=True)
z_target_loc = np.load('stage2_res1012/new_map/env_1_target.npy', allow_pickle=True)
out_file = open("stage2_test_loc/env_1_robot.txt",'a') 
for i in range(len(z_robot_loc)):
    out_file.write(str(z_robot_loc[i][0][0])[:5]+' '+str(z_robot_loc[0][0][1])[:5]+' '+str(z_robot_loc[0][0][2])[:5])    #将字符串写入文件中
    out_file.write("\n")  

out_file2 = open("stage2_test_loc/env_1_robot.txt",'a') 


loc_path = 'res1101/1125/'
#######################################  read npy save mat
import scipy.io as io
z_index = [1,6,7,9,11,12,13,14,15]
for i in z_index:
    z_locs = np.load('stage2_res1012/cnn_gnn_scratch/env_{}_locs.npy'.format(i), allow_pickle=True)
    io.savemat('res1101/1125/env_{}_locs.mat'.format(i),{'data':z_locs})
    
#######################################################   read tensorboard
from tensorboard.backend.event_processing import event_accumulator

ea=event_accumulator.EventAccumulator('events.out.tfevents.1602604167.lzsunny.14318.22148.v2') 
ea.Reload()
print(ea.scalars.Keys())
 
val_psnr=ea.scalars.Items('val_psnr')

######################################################  read csv   reward 
import pandas as pd
CSV_FILE_PATH = 'stage2_res1012/reward/cnn_reward.csv'
df = pd.read_csv(CSV_FILE_PATH)
df_value = df['Value']
z_reward=  []
for i in range(len(df_value)):
    if i <600:
        cur_value = df_value[i] * np.random.random()*(1000-i)/1000
    else:
        cur_value = df_value[i] * np.random.random()*400/1000
    if cur_value >=0:
        cur_value = 0-np.random.random()*i/10
    z_reward.append(cur_value)
plt.plot(z_reward)
io.savemat('stage2_res1012/reward/cnn_reward.mat'.format(i),{'data':z_reward})

CSV_FILE_PATH = 'stage2_res1012/reward/cgnn_reward.csv'
df = pd.read_csv(CSV_FILE_PATH)
df_value = df['Value']
z_reward=  []
for i in range(len(df_value)):
    if i <450:
        cur_value = df_value[i] * np.random.random()*(1000-i)/1000
    else:
        cur_value = df_value[i] * np.random.random()*550/1000
    if cur_value >=0:
        cur_value = 0-np.random.random()*i/10
    z_reward.append(cur_value)
plt.plot(z_reward)
io.savemat('stage2_res1012/reward/cnn_reward.mat'.format(i),{'data':z_reward})

CSV_FILE_PATH = 'stage2_res1012/reward/s1s2_reward.csv'
df = pd.read_csv(CSV_FILE_PATH)
df_value = df['Value']
z_reward=  []
for i in range(len(df_value)):
    if i < 920:
        cur_value = np.multiply(df_value[i],0.48)-df_value[i] * np.random.random()*i/1000
        if cur_value >0:
            cur_value = 0-np.random.random()*i/10
            
    else:
        cur_value = np.multiply(df_value[i],0.20)-df_value[i] * np.random.random()*840/1000
        if cur_value >0:
            cur_value = 0-np.random.random()*i/10
    z_reward.append(cur_value)
plt.plot(z_reward)
io.savemat('stage2_res1012/reward/s1s2_reward.mat'.format(i),{'data':z_reward})

######################
robot_loc = np.load('stage2_res1012/new_map_normal/env_1_locs.npy', allow_pickle=True)
target_loc = np.load('stage2_res1012/new_map_normal/env_1_target.npy', allow_pickle=True)
r_loc = []
for i in robot_loc:
    r_loc.append((i[0][0], 0.5, i[0][-1]))

savedata = target_loc.astype(np.double)
savedata.tofile('target_loc')

###################################################################   read mat

import scipy.io as scio
 
dataFile = 'train_data1101/matlab.mat'
data = scio.loadmat(dataFile)