# -*- coding: utf-8 -*-
"""
Created on Thu Jul  2 10:48:27 2020

@author: azrael
"""
import numpy as np
from a_star import AStarSearch
import math
from tensorflow.keras.preprocessing import image
import os

def change_axis(img, loc):
    env_x, env_z = img.shape
    loc_x = loc[0]
    loc_z = loc[2]
    return (env_x/2 - loc_z, 0, env_z/2 + loc_x)
 
def sen_angle():
    env = np.load('env_0.npy')
    all_sensors = ['sensor_1', 'sensor_2', 'sensor_3', 'sensor_4']
    sensor_loc = [(5, 0, 5), (5, 0,-5), (-5, 0, -5), (-5, 0, 5)]
    num_sensors = len(all_sensors)
    target_label = open("target_loc.txt","r") 
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
        #t_x, t_y, t_z = change_axis(env, (label_x, 0, label_z))     
        x_target = label_x
        z_target = label_z
        sensor_angle = np.zeros((len(sensor_loc), 1))
        for i, loc_s in enumerate(sensor_loc):    
            x_sensor = loc_s[0]
            z_sensor = loc_s[2]
            ######################
            angle = 0.0;
            dx = abs(x_target - x_sensor)
            dz = abs(z_target - z_sensor)
            if  x_target == x_sensor:
                if z_target > z_sensor:
                    angle = np.pi/2
                elif z_sensor > z_target :
                    angle =  np.pi * 1.5
            elif x_target > x_sensor and z_target >= z_sensor:
                angle = np.pi - np.arctan(dz / dx)
            elif  x_target > x_sensor and  z_target < z_sensor:
                angle = np.arctan(dz / dx) + np.pi
            elif  x_target < x_sensor and z_target < z_sensor:
                angle = 2*np.pi - np.arctan(dz / dx)
            elif  x_target < x_sensor and z_target >= z_sensor:
                angle = np.arctan(dz / dx)
            sensor_angle[i] = angle / np.pi  
        all_label.append(sensor_angle)
    return all_label

def sen_angle_single():
    env = np.load('env_0.npy')
    all_sensors = ['sensor_1']
    sensor_loc = [(0, 0, 0), (5, 0,-5), (-5, 0, -5), (-5, 0, 5)]
    num_sensors = len(all_sensors)
    target_label = open("target_loc_random.txt","r") 
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
        #t_x, t_y, t_z = change_axis(env, (label_x, 0, label_z))     
        x_target = label_x
        z_target = label_z
        sensor_angle = np.zeros((len(sensor_loc), 1))
        for i, loc_s in enumerate(sensor_loc):    
            x_sensor = loc_s[0]
            z_sensor = loc_s[2]
            ######################
            angle = 0.0;
            dx = abs(x_target - x_sensor)
            dz = abs(z_target - z_sensor)
            if  x_target == x_sensor:
                if z_target > z_sensor:
                    angle = np.pi/2
                elif z_sensor > z_target :
                    angle =  np.pi * 1.5
            elif x_target > x_sensor and z_target >= z_sensor:
                angle = np.pi - np.arctan(dz / dx)
            elif  x_target > x_sensor and  z_target < z_sensor:
                angle = np.arctan(dz / dx) + np.pi
            elif  x_target < x_sensor and z_target < z_sensor:
                angle = 2*np.pi - np.arctan(dz / dx)
            elif  x_target < x_sensor and z_target >= z_sensor:
                angle = np.arctan(dz / dx)
            sensor_angle[i] = angle / np.pi  
        all_label.append(sensor_angle)
    return all_label

def loc2angle(loc):
    angle = -1
    if loc == (0,-1):
        angle = 0
    if loc == (-1,-1):
        angle = 0.25
    if loc == (-1, 0):
        angle = 0.5
    if loc == (-1,1):
        angle = 0.75
    if loc == (0,1):
        angle = 1
    if loc == (1,1):
        angle = 1.25
    if loc == (1,0):
        angle = 1.5
    if loc == (1,-1):
        angle = 1.75
    return angle

def theta_old(env, start_loc, end_loc):           #theta*   return  connection points
    start_loc = (round(start_loc[0]), round(start_loc[-1])) 
    end_loc = (round(end_loc[0]), round(end_loc[-1])) 
    s_path = AStarSearch(env, start_loc, end_loc)
    cur_start = s_path[0]
    record_grid = []
    set_grid = s_path[0]
    record_grid.append(s_path[0])
    in_view = True
    if in_view:
        cur_loc = s_path[-1]
        past_grid = []
        x_flag = 1
        y_flag = 1
        if (cur_start[1] - cur_loc[1]) >=0:
            y_flag = -1
        if (cur_start[0] - cur_loc[0]) >=0:
            x_flag = -1
        if abs(cur_loc[0]-cur_start[0])<= abs(cur_loc[1]-cur_start[1]):  
            if cur_loc[1] == cur_start[1]:
                print('cur_loc:', cur_loc, '   end_loc[1]:', end_loc)
                record_grid.append(s_path[-1])
                return record_grid
                
            slocp = abs((cur_loc[0]-cur_start[0])/(cur_loc[1]-cur_start[1]+1e-10))
            for j in range(abs(cur_start[1]- cur_loc[1])):
                past_grid.append((cur_start[0]+ j*slocp*x_flag, cur_start[1]+j*y_flag))
        else:  
            slocp = abs((cur_loc[1]-cur_start[1])/(cur_loc[0]-cur_start[0]+1e-10))
            for j in range(abs(cur_start[0]-cur_loc[0])):
                past_grid.append((cur_start[0]+j*x_flag, cur_start[1]+slocp*j*y_flag))
        for cur_grid in past_grid:
            cur_grid = (math.ceil(cur_grid[0]), math.ceil(cur_grid[1]))
            if env[cur_grid[0], cur_grid[1]] == 0:
                pass
            else:
                record_grid.append(set_grid)
                cur_start = cur_loc   
                in_view = False
                break
        set_grid = cur_loc 
    if in_view:
        record_grid.append(s_path[-1])
        return record_grid
    cur_start = s_path[0]
    record_grid = []
    set_grid = s_path[0]
    record_grid.append(s_path[0])
    for i in range(1, len(s_path)):      
        cur_loc = s_path[i]
        past_grid = []
        x_flag = 1
        y_flag = 1
        if (cur_start[1] - cur_loc[1]) >=0:
            y_flag = -1
        if (cur_start[0] - cur_loc[0]) >=0:
            x_flag = -1
        if abs(cur_loc[0]-cur_start[0])<= abs(cur_loc[1]-cur_start[1]):    
            slocp = abs((cur_loc[0]-cur_start[0])/(cur_loc[1]-cur_start[1]))
            for j in range(abs(cur_start[1]- cur_loc[1])):
                past_grid.append((cur_start[0]+ j*slocp*x_flag, cur_start[1]+j*y_flag))
        else:  
            slocp = abs((cur_loc[1]-cur_start[1])/(cur_loc[0]-cur_start[0]))
            for j in range(abs(cur_start[0]-cur_loc[0])):
                past_grid.append((cur_start[0]+j*x_flag, cur_start[1]+slocp*j*y_flag))
        for cur_grid in past_grid:
            cur_grid = (math.ceil(cur_grid[0]), math.ceil(cur_grid[1]))
            if env[cur_grid[0], cur_grid[1]] == 0:
                pass
            else:
                record_grid.append(set_grid)
                cur_start = cur_loc   
                break
        set_grid = cur_loc   
    record_grid.append(s_path[-1])
    return record_grid

def theta(env, start_loc, end_loc):           #theta*   return  connection points
    start_loc = (round(start_loc[0]), round(start_loc[-1])) 
    end_loc = (round(end_loc[0]), round(end_loc[-1]))
    #print('start_loc:', start_loc, '   end_loc:', end_loc)
    s_path = AStarSearch(env, (int(start_loc[0]), int(start_loc[1])), (int(end_loc[0]), int(end_loc[1])))
    cur_start = s_path[0]
    record_grid = []
    #record_grid.append(s_path[0])
    for i in range(len(s_path)-1,-1, -1):      
        cur_loc = s_path[i]
        past_grid = []
        past_grid_int = []
        x_flag = 1
        y_flag = 1
        f_x = int
        f_y = int
        if (cur_start[1] - cur_loc[1]) >=0:
            y_flag = -1
            f_y = math.ceil
        if (cur_start[0] - cur_loc[0]) >=0:
            x_flag = -1
            f_x = math.ceil
        if abs(cur_loc[0]-cur_start[0])<= abs(cur_loc[1]-cur_start[1]):    
            slocp = abs((cur_loc[0]-cur_start[0])/(cur_loc[1]-cur_start[1]+1e-10))
            for j in range(abs(cur_start[1]- cur_loc[1])):
                cur_x, cur_y = (cur_start[0]+ j*slocp*x_flag, cur_start[1]+j*y_flag)
                fur_x, fur_y = (cur_start[0]+ (j+1)*slocp*x_flag, cur_start[1]+(j+1)*y_flag)
                past_grid.append((cur_x,cur_y))
                if f_x(fur_x) != f_x(cur_x) and fur_x != f_x(fur_x):
                    past_grid_int.append((f_x(cur_x), f_y(cur_y)))
                    past_grid_int.append((f_x(fur_x), f_y(cur_y)))
                else:
                    past_grid_int.append((f_x(cur_x), f_y(cur_y)))
        else:  
            slocp = abs((cur_loc[1]-cur_start[1])/(cur_loc[0]-cur_start[0]+1e-10))
            for j in range(abs(cur_start[0]-cur_loc[0])):
                cur_x, cur_y = (cur_start[0]+j*x_flag, cur_start[1]+slocp*j*y_flag)
                fur_x, fur_y = (cur_start[0]+(j+1)*x_flag, cur_start[1]+slocp*(j+1)*y_flag)
                past_grid.append((cur_x,cur_y))
                if f_y(fur_y) != f_y(cur_y) and fur_y != f_y(fur_y):
                    past_grid_int.append((f_x(cur_x), f_y(cur_y)))
                    past_grid_int.append((f_x(cur_x), f_y(fur_y)))
                else:
                    past_grid_int.append((f_x(cur_x), f_y(cur_y)))
        past_grid_int.append(cur_loc)
        for cur_grid in past_grid_int:
            #if cur_grid[0]>=39:
            #    cur_grid = (39, cur_grid[1])
            #if cur_grid[1]>=39:
            #    cur_grid = (cur_grid[0], 39)
            if env[cur_grid[0], cur_grid[1]] == 0:
                pass
            else:
                if i == len(s_path)-1:
                    record_grid.append(cur_loc)
                else:
                    record_grid.append(s_path[i+1])
                cur_start = cur_loc   
                break  
        if len(record_grid) == 0:
            break
    #print(record_grid)
    if len(record_grid) == 0:
            record_grid.append(s_path[-1])
            record_grid.append(s_path[0])
    else:
        record_grid.append(s_path[0])  
    record_grid = record_grid[::-1]
    #record_grid.append(s_path[-1])  
    return record_grid

def angle2xy_old(loc_s, loc_e):
    rel_x = loc_e[1] - loc_s[1]
    rel_y = -(loc_e[0] - loc_s[0])
    return(rel_x/np.sqrt(rel_x**2+rel_y**2), rel_y/np.sqrt(rel_x**2+rel_y**2))

def angle2xy(loc_s, loc_e):
    rel_x = loc_e[0] - loc_s[0]
    rel_y = loc_e[1] - loc_s[1]
    #print('loc_s:', loc_s,  '    loc_e:', loc_e)
    return(rel_x/(np.sqrt(rel_x**2+rel_y**2)+1e-10), rel_y/(np.sqrt(rel_x**2+rel_y**2)+1e-10))
           
#img = np.zeros((20,20))
def s_label(select_env):
    if select_env == 0:
        env = np.load('env_0.npy')
        all_sensors = ['sensor_1']
        sensor_loc = [(-5, 0, 5)]
    if select_env == 1:
        env = np.load('env_1.npy')
        all_sensors = ['sensor_1', 'sensor_2', 'sensor_3', 'sensor_4']
        sensor_loc = [(-5, 0, 5), (-5, 0,-5), (5, 0, -5), (5, 0, 5)]
    if select_env == 2:    
        env = np.load('env_2.npy')
        all_sensors = ['sensor_1', 'sensor_2', 'sensor_3', 'sensor_4']
        sensor_loc = [(15, 0, 0), (5, 0, 0), (-5, 0, 0), (-15, 0, 0)]
    if select_env == 3:    
        env = np.load('env_3.npy')
        all_sensors = ['sensor_1', 'sensor_2', 'sensor_3', 'sensor_4',
                       'sensor_5', 'sensor_6', 'sensor_7', 'sensor_8']
        sensor_loc = [(-15, 0, 15), (-7.5, 0, 2.5), (-7.5, 0, -7.5), (-15, 0, -15),
                      (18, 0, 10), (7.5, 0, 2.5), (7.5, 0, -7.5), (18, 0, -10)] #env-3
        
    num_sensors = len(all_sensors)
    target_label = open("target_loc.txt","r") 
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
        t_x, t_y, t_z = change_axis(env, (label_x, 0, label_z))
        sensor_direction = []
        for j in range(num_sensors):
            s_x, s_y, s_z = change_axis(env, sensor_loc[j])
            #s_path = AStarSearch(env, (round(s_x), round(s_z)), (round(t_x), round(t_z)))
            s_path = theta(env, (round(s_x), round(s_z)), (round(t_x), round(t_z)))
            if s_path[0] == s_path[1]:
                s_angle = angle2xy((round(s_x), round(s_z)), (round(t_x), round(t_z)))
            else:    
                s_angle = angle2xy(s_path[0], s_path[1])
            #s_relative = (s_path[1][0]-s_path[0][0], s_path[1][1]-s_path[0][1])
            #s_angle = loc2angle(s_relative)
            sensor_direction.append(s_angle)
        all_label.append(sensor_direction)
    return all_label
    
def s_label_batch(select_group, select_case, env_index):
    all_sensors = ['sensor_1', 'sensor_2', 'sensor_3', 'sensor_4',
                   'sensor_5', 'sensor_6', 'sensor_7', 'sensor_8', 'sensor_9']
    batch_size = len(select_case)
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
    if env_index == 5:
        env = np.load('env_5.npy')  
        if select_group == 1:  
            sensor_loc = [(11, 0, 8),  (7, 0, 21),  (12, 0, 34), (16, 0, 21),
                              (28, 0, 15), (26, 0, 34), (33, 0, 4), (39, 0, 14), (36, 0, 26)] #env-5-1
            label_path = "target_env5_1.txt"
    if env_index == 6:
        env = np.load('env_6.npy')   
        if select_group == 1:     
            sensor_loc = [(4, 0, 0),  (3, 0, 14),  (8, 0, 28), (16, 0, 17),
                              (22, 0, 30), (22, 0, 5), (27, 0, 18), (36, 0, 8), (35, 0, 36)] #env-6-1
            label_path = "target_env6_1.txt"

    num_sensors = len(all_sensors)
    target_label = open(label_path,"r") 
    lines = target_label.readlines() 
    batch_label = []
    for i in range(batch_size):
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
        for j in range(num_sensors):
            #s_x, s_y, s_z = change_axis(env, sensor_loc[j])
            s_x, s_y, s_z = sensor_loc[j]
            #s_path = AStarSearch(env, (round(s_x), round(s_z)), (round(t_x), round(t_z)))
            s_path = theta(env, (round(s_x), round(s_z)), (round(t_x), round(t_z)))
            if s_path[0] == s_path[1]:
                print('num_group:', select_group, '   num_case:', select_case[i], '   num_sensor:', j)
                print('target_loc:', (label_x, 0, label_z))
                s_angle = angle2xy((s_x, s_z), (t_x, t_z))
            else:
                s_angle = angle2xy(s_path[0], s_path[1])
            #s_relative = (s_path[1][0]-s_path[0][0], s_path[1][1]-s_path[0][1])
            #s_angle = loc2angle(s_relative)
            sensor_direction.append(s_angle)
        batch_label.append(sensor_direction)
    return batch_label

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
        
def sample_batch(batch_size = 128, num_mix = 1):
    
    total_sensor = 9
    if total_sensor == 9:    
        total_env = 12
        select_group = [np.random.randint(1, 4+1) for _ in range(num_mix)]
    else:
        total_env = 6
        select_group = [np.random.randint(1, 3) for _ in range(num_mix)]
    select_env = [np.random.randint(1, total_env+1) for _ in range(num_mix)]
    
    all_sensors = []
    for i in range(total_sensor):
        all_sensors.append('sensor_{}'.format(i+1))
    sensor_loc = []
    target_loc = []
    all_target_label = []
    ad_mat1, ad_mat2, ad_mat3, ad_mat4 = [], [], [], []
    all_input = []
    
    for i in range(num_mix):
        #select_case = np.arange(batch_size)
        select_case = [np.random.randint(500) for _ in range(batch_size)]  
        sen_loc = np.load('train_data0911/ad_matrix0911/env_{}_{}_{}_loc.npy'.format(select_env[i], total_sensor, select_group[i]))
        tar_loc = read_target('train_data0911/env_{}_{}_{}.txt'.format(select_env[i], total_sensor, select_group[i]), select_case)
        ad1 = np.load('train_data0911/ad_matrix0911/env_{}_{}_{}_ad1.npy'.format(select_env[i], total_sensor, select_group[i]))
        ad2 = np.load('train_data0911/ad_matrix0911/env_{}_{}_{}_ad2.npy'.format(select_env[i], total_sensor, select_group[i]))
        ad3 = np.load('train_data0911/ad_matrix0911/env_{}_{}_{}_ad3.npy'.format(select_env[i], total_sensor, select_group[i]))
        ad4 = np.load('train_data0911/ad_matrix0911/env_{}_{}_{}_ad4.npy'.format(select_env[i], total_sensor, select_group[i]))
        env = np.load('train_data0911/env_map/env_{}.npy'.format(select_env[i]))
        
        sensor_loc.append(sen_loc)
        target_loc.append(tar_loc)
        ad_mat1.append(ad1)
        ad_mat2.append(ad2)
        ad_mat3.append(ad3)
        ad_mat4.append(ad4)
        
        target_label = []
        batch_input = []
        filePath = 'train_data0911/env_{}_{}_{}/sensor_1/1'.format(select_env[i], total_sensor, select_group[i])
        filelist = os.listdir(filePath)
        filelist.sort(key = lambda x: int(x[:-4]))
        for s_i in range(len(select_case)):
            sensor_dir = []
            image_input = np.zeros((len(sen_loc), 84, 84*4, 3))
            for j in range(len(all_sensors)):
                s_x, s_z = sen_loc[j]
                s_path = theta(env, (round(s_x), round(s_z)), (round(tar_loc[s_i][0]), round(tar_loc[s_i][1])))
                if s_path[0] == s_path[1]:
                    print('wrong position!')
                    s_angle = angle2xy((s_x, s_z), (tar_loc[s_i][0], tar_loc[s_i][1]))
                elif len(s_path) == 2:
                    s_angle = angle2xy((s_x, s_z), (tar_loc[s_i][0], tar_loc[s_i][1]))
                else:
                    s_angle = angle2xy(s_path[0], s_path[1])
                sensor_dir.append(s_angle)
                
                #######################################################image
                sensor_path = 'train_data0911/env_{}_{}_{}/'.format(select_env[i], total_sensor, select_group[i]) + all_sensors[j]
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
        all_target_label.append(target_label)
        all_input.append(batch_input)
    return all_input, all_target_label, ad_mat1, ad_mat2, ad_mat3, ad_mat4, total_sensor
                        
def sample_batch1101(num_iter, select_env, num_sensors, batch_size=1024):
    
    all_sensors = []
    for i in range(num_sensors):
        all_sensors.append('sensor_{}'.format(i))
    all_sensors.append('robot')
        
    sensor_loc = []
    target_loc = []
    all_target_label = []
    ad_mat1, ad_mat2, ad_mat3, ad_mat4 = [], [], [], []
    all_input = []
    
    select_case = np.arange(batch_size)
    #select_case = [np.random.randint(500) for _ in range(batch_size)]  
    sen_loc = np.load('train_data1101/ad_matrix/env_{}_loc.npy'.format(num_iter))
    tar_loc = read_target('train_data1101/env_{}.txt'.format(num_iter), select_case)
    ad1 = np.load('train_data1101/ad_matrix/env_{}_ad1.npy'.format(num_iter))
    ad2 = np.load('train_data1101/ad_matrix/env_{}_ad2.npy'.format(num_iter))
    ad3 = np.load('train_data1101/ad_matrix/env_{}_ad3.npy'.format(num_iter))
    ad4 = np.load('train_data1101/ad_matrix/env_{}_ad4.npy'.format(num_iter))
    env = np.load('train_data1101/env_map/env_{}.npy'.format(select_env))
    
    sensor_loc.append(sen_loc)
    target_loc.append(tar_loc)
    ad_mat1.append(ad1)
    ad_mat2.append(ad2)
    ad_mat3.append(ad3)
    ad_mat4.append(ad4)
    
    target_label = []
    batch_input = []
    filePath = 'training/sensor_1/1'
    filelist = os.listdir(filePath)
    filelist.sort(key = lambda x: int(x[:-4]))
    for s_i in range(len(select_case)):
        sensor_dir = []
        image_input = np.zeros((len(sen_loc), 84, 84*4, 3))
        for j in range(len(all_sensors)):
            s_x, s_z = sen_loc[j]
            s_path = theta(env, (round(s_x), round(s_z)), (round(tar_loc[s_i][0]), round(tar_loc[s_i][1])))
            if s_path[0] == s_path[1]:
                print('wrong position!')
                s_angle = angle2xy((s_x, s_z), (tar_loc[s_i][0], tar_loc[s_i][1]))
            elif len(s_path) == 2:
                s_angle = angle2xy((s_x, s_z), (tar_loc[s_i][0], tar_loc[s_i][1]))
            else:
                s_angle = angle2xy(s_path[0], s_path[1])
            sensor_dir.append(s_angle)
            
            #######################################################image
            sensor_path = 'train_data0911/env_{}_{}_{}/'.format(select_env[i], total_sensor, select_group[i]) + all_sensors[j]
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
            
        all_target_label.append(target_label)
        all_input.append(batch_input)
    return all_input, all_target_label, ad_mat1, ad_mat2, ad_mat3, ad_mat4, total_sensor
                        
    
    