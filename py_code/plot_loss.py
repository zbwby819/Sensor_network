# -*- coding: utf-8 -*-
"""
Created on Thu Jul  9 09:52:37 2020

@author: azrael
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

fix_loss = pd.read_csv('mytensorboard/fix_single.csv')
fix_single = fix_loss['loss']

cnn_loss = pd.read_csv('mytensorboard/cnn_single.csv')
cnn_single = cnn_loss['loss']

gnn_loss = pd.read_csv('mytensorboard/gnn_single.csv')
gnn_single = gnn_loss['loss']

f_loss = []
c_loss = []
g_loss = []
for i in range(0, len(fix_loss), 2):
    f_loss.append(float(fix_single[i][:6]))

for i in range(0, len(cnn_loss), 2):
    c_loss.append(float(cnn_single[i][:6]))

for i in range(0, len(gnn_loss), 2):
    g_loss.append(float(gnn_single[i][:6]))
plt.ylim(0, 2)
plt.xlabel('Training epochs')
plt.ylabel('Training Loss')
plt.subplot(311)
plt.ylim(0, 2)
plt.plot(f_loss, 'r')
plt.subplot(312)
plt.ylim(0, 2)
plt.plot(c_loss, 'b')
plt.subplot(313)
plt.ylim(0, 2)
plt.plot(g_loss, 'g')

plt.xlabel('Training rounds')
plt.ylabel('Training Loss')
plt.ylim(0, 2)
plt.plot(f_loss, 'r')
###################################################################  plot loss  1007


y_lim = 3
csv_name = 'dyna_gat_layer4'
env0_his_origin = pd.read_csv(csv_name+'.csv', error_bad_lines=False)
loss_name = env0_his_origin.columns.values.tolist()

loss_index = loss_name.index('loss')
for i in range(len(loss_name[loss_index+1:])):
    exec('ori_sensor_{}_loss = env0_his_origin[loss_name[loss_index+1+i]]'.format(i))
    
sensor_sum = []
for i in range(0, len(ori_sensor_0_loss), 2):
    zz_sum = 0   
    for j in range(len(loss_name[loss_index+1:])):
        exec('zz_sum += float(ori_sensor_{}_loss[i])'.format(j))
    sensor_sum.append(zz_sum/len(loss_name[loss_index+1:]))

plt.xlabel('Training epochs')
plt.ylabel('Training Loss')
plt.ylim(0, 3)
my_y_ticks = np.arange(0, 3, 0.25)
my_x_ticks = np.arange(0, 21000, 5000)
plt.xticks(my_x_ticks)
plt.yticks(my_y_ticks)
plt.plot(sensor_sum, 'r', label = csv_name, linewidth=0.05)
plt.legend()
np.save('training_loss1007/'+csv_name+'.npy', sensor_sum)
#plt.xticks([])
#plt.yticks([])
#plt.subplot(all_loss)

############################################################  stage2 --actor/critic
actor_loss = np.load('actor_loss.npy')
#fix_single = fix_loss['loss']

critic_loss = np.load('critic_loss.npy')
#cnn_single = cnn_loss['loss']

plt.ylim(0, 3)
plt.xlabel('Training epochs')
plt.ylabel('Training Loss')
plt.subplot(211)
#lt.ylim(0, 3)
plt.plot(actor_loss, 'r', label = 'actor_loss')
plt.legend()
plt.subplot(212)
#plt.ylim(0, 2)
plt.plot(critic_loss, 'b', label='critic_loss')
plt.legend()




#############################################################

env0_his_pose = pd.read_csv('mytensorboard/gnn_pose_env0.csv')
pose_loss = env0_his_pose['loss']
pose_sensor_1_loss = env0_his_pose['sensor_1_loss']
pose_sensor_2_loss = env0_his_pose['sensor_2_loss']
pose_sensor_3_loss = env0_his_pose['sensor_3_loss']
pose_sensor_4_loss = env0_his_pose['sensor_4_loss']
pose_all_loss = []
pose_s1_loss = []
pose_s2_loss = []
pose_s3_loss = []
pose_s4_loss = []
for i in range(0, len(pose_loss), 2):
    pose_all_loss.append(float(pose_loss[i][:6]))
    pose_s1_loss.append(float(pose_sensor_1_loss[i][:6]))
    pose_s2_loss.append(float(pose_sensor_2_loss[i][:6]))
    pose_s3_loss.append(float(pose_sensor_3_loss[i][:6]))
    pose_s4_loss.append(float(pose_sensor_4_loss[i][:6]))
plt.ylim(0, 2)
plt.xlabel('Training epochs')
plt.ylabel('Training Loss')
#plt.xticks([])
#plt.yticks([])
#plt.subplot(all_loss)
plt.subplot(221)
plt.ylim(0, 2)
plt.plot(pose_s1_loss, 'r', label = 'Sensor_1')
plt.legend()
plt.subplot(222)
plt.ylim(0, 2)
plt.plot(pose_s2_loss, 'b', label = 'Sensor_2')
plt.legend()
plt.subplot(223)
plt.ylim(0, 2)
plt.plot(pose_s3_loss, 'g', label = 'Sensor_3')
plt.legend()
plt.subplot(224)
plt.ylim(0, 2)
plt.plot(pose_s4_loss, 'cyan', label = 'Sensor_4')
plt.legend()

#################################################   plot all target locs
def change_axis(img, loc):
    env_x, env_z = img.shape
    loc_x = loc[0]
    loc_z = loc[2]
    return (env_x/2 - loc_z, 0, env_z/2 + loc_x)

env3_img = np.zeros((40,40))
target_label = open("target_loc_env3.txt","r") 
lines = target_label.readlines() 
all_label = []
for i in range(len(lines)):
    label_index = lines[i].index(')')
    label_target = int(lines[i][label_index+1:-1])
    x_index_1 = lines[i].index('(')
    x_index_2 = lines[i].index(',')
    label_x = float(lines[i][x_index_1+1:x_index_2])
    z_index_1 = lines[i].index(',', x_index_2+1)
    z_index_2 = lines[i].index(')')  
    label_z = float(lines[i][z_index_1+2:z_index_2])
    t_x, t_y, t_z = change_axis(env3_img, (label_x, 0, label_z))
    env3_img[round(t_x), round(t_z)] += 1
env3_img /= np.max(env3_img)

plt.imshow(env3_img)
plt.colorbar()
