# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 17:19:00 2020

@author: azrael
"""

z_robot_locs = []
z_targets = []
z_dis = []
z_dis_true = []
z_env = zz_env.copy()
z_env_r = zz_env.copy()
for i in range(100):
    z_target = np.load('stage2_0925/test_{}_target.npy'.format(i))
    z_loc = np.load('stage2_0925/test_{}_loc.npy'.format(i))
    z_robot_locs.append(z_loc)
    z_targets.append(z_target)
    robot_dis = 0
    for j in range(len(z_loc)-1):
        s1 = abs(z_loc[j+1][0] - z_loc[j][0]) + abs(z_loc[j+1][1] - z_loc[j][1])
        robot_dis += s1
    z_dis.append(robot_dis)
    r_path =  AStarSearch(zz_env, (round(z_loc[0][0]), round(z_loc[0][1])), 
                                  (round(z_target[0]), round(z_target[-1])))
    z_dis_true.append(len(r_path))
    
    z_env[round(z_target[0])][round(z_target[-1])] -= 0.3
    
    z_env_r[round(z_loc[0][0])][round(z_loc[0][1])] -= 0.3
    
plt.imshow(z_env_r, cmap=plt.get_cmap('gray_r'))
    
z_env = zz_env.copy()



z_num = 0
for kk in range(100):
    if len(z_robot_locs[kk]) == 100:
        z_num+=1
        
zz_r = []
zz_t = []
zz_div = []
for kk in range(100):
    if len(z_robot_locs[kk]) == 100:
        pass
    else:
        if z_dis[kk] < z_dis_true[kk]:
            zz_div.append(z_dis_true[kk]/z_dis[kk])
        else:
            zz_r.append(z_dis[kk])
            zz_t.append(z_dis_true[kk])
            zz_div.append(z_dis[kk]/z_dis_true[kk])
            
plt.plot(np.arange(len(zz_r)), zz_r, color='r')
plt.plot(np.arange(len(zz_r)), zz_t, color='g')


all_z1, all_z2, all_z3 = [],[],[]
for i in range(100):
    env.reset()
    z1,z2,z3 = agent.test2()
    all_z1.append(z1)
    all_z2.append(z2)
    all_z3.append(z3)
    
zz_r = []
zz_t = []
zz_div = []
for kk in range(100):
    if len(all_z3[kk]) == 60:
        pass
    else:
        r_dis = 0
        r_true = 0
        r_true =  AStarSearch(zz_env, (round(all_z2[kk][0][0]), round(all_z2[kk][0][1]))
                                      (round(all_z2[kk][-1][0]), round(all_z2[kk][-1][1])))
        for kj in range(len(all_z1)):
            

            zz_div.append(all_z2[kk]/all_z1[kk])
    