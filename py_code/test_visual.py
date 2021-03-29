# -*- coding: utf-8 -*-
"""
Created on Thu Aug 20 23:33:42 2020

@author: azrael
"""

import matplotlib.pyplot as plt
import numpy
import matplotlib.colors as colors
import matplotlib.cm as cmx


 _locations = [
    (4, 4), # depot
    (4, 4), # unload depot_prime
    (4, 4), # unload depot_second
    (4, 4), # unload depot_fourth
    (4, 4), # unload depot_fourth
    (4, 4), # unload depot_fifth
    (2, 0),
    (8, 0), # locations to visit
    (0, 1),
    (1, 1),
    (5, 2),
    (7, 2),
    (3, 3),
    (6, 3),
    (5, 5),
    (8, 5),
    (1, 6),
    (2, 6),
    (3, 7),
    (6, 7),
    (0, 8),
    (7, 8)
  ]
 
 
plt.figure(figsize=(10, 10))
#p1 = [l[0] for l in _locations]
#p2 = [l[1] for l in _locations]
#plt.plot(p1[:6], p2[:6], 'g*', ms=20, label='depot')
#plt.plot(p1[6:], p2[6:], 'ro', ms=15, label='customer')
plt.grid(True)
plt.legend(loc='lower left')

way = [[0, 12, 18, 17, 16, 4, 14, 10, 11, 13, 5], [0, 6, 9, 8, 20, 3], [0, 19, 21, 15, 7, 2]]  # 

cmap = plt.cm.jet
cNorm = colors.Normalize(vmin=0, vmax=len(way))
scalarMap = cmx.ScalarMappable(norm=cNorm,cmap=cmap)

for k in range(0, len(way)):
  way0 = way[k]
  colorVal = scalarMap.to_rgba(k)
  for i in range(0, len(way0)-1):
    start = _locations[way0[i]]
    end = _locations[way0[i+1]]
#     plt.arrow(start[0], start[1], end[0]-start[0], end[1]-start[1], length_includes_head=True,
#         head_width=0.2, head_length=0.3, fc='k', ec='k', lw=2, ls=lineStyle[k], color='red')
    plt.arrow(start[0], start[1], end[0]-start[0], end[1]-start[1], 
         length_includes_head=True, head_width=0.2, lw=2,
         color=colorVal)
plt.show()
cmap = plt.cm.jet
cNorm = colors.Normalize(vmin=0, vmax=len(way))
scalarMap = cmx.ScalarMappable(norm=cNorm,cmap=cmap)
#######################################################
z_env3_2_pred = np.load('result_0820/pred_env3_2.npy')
z_env3_2_true = np.load('result_0820/true_env3_2.npy')
z_env3_2_test = np.load('result_0820/test_env3_2.npy')
z_env3_2_sum = []
for k in range(100):
    z_env3_2_sum.append(np.sum(z_env3_2_test[k]))
    
np.average(z_env3_2_test[:,0])
np.std(z_env3_2_test[:,0])

sensor_loc = [(19, 0, 9), (5, 0, 10),  (28, 0, 13), (36, 0, 4),
              (9, 0, 24), (12, 0, 38), (22, 0, 23), (26, 0, 35), (35, 0, 29)] #env-3-2


sensor_loc = [(12, 0, 1),  (6, 0, 13),  (7, 0, 26), (17, 0, 16),
              (19, 0, 30), (27, 0, 20), (31, 0, 8), (33, 0, 31), (39, 0, 19)] #env-4-1
#######################################################
plt.arrow(2, 2, 3, 3, length_includes_head=True, head_width=0.2, lw=2,color='r')

sel_index = 3
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111)
ax.set_xlim(0, 40)
ax.set_ylim(0, 40)
ax.grid()
for j in range(9):
    start=(sensor_loc[j][0], sensor_loc[j][-1])
    end_true = z_env4_1_true[sel_index][j]
    end_pred = z_env4_1_pred[sel_index][j]
    end_pred = (end_pred[0]/np.sqrt(end_pred[0]**2+end_pred[1]**2), end_pred[1]/np.sqrt(end_pred[0]**2+end_pred[1]**2)) 
    ax.arrow(start[-1], 40-start[0],end_true[-1]*5, -(end_true[0]*5),  length_includes_head=True,
         head_width=0.25,
         head_length=1,
         color='r')
    ax.arrow(start[-1], 40-start[0],end_pred[-1]*5, -(end_pred[0]*5),  length_includes_head=True,
         head_width=0.25,
         head_length=1,
         color='g')

###############################################################################################
z_env3_1_norm = []
for i in range(100):
    norm = []
    a_pred = z_env3_1_pred[i]
    for j in range(9):
        norm.append((a_pred[j][0][0]/np.sqrt(a_pred[j][0][0]**2+a_pred[j][0][1]**2), 
                    a_pred[j][0][1]/np.sqrt(a_pred[j][0][0]**2+a_pred[j][0][1]**2)))   
    z_env3_1_norm.append(norm)
        

from scipy import io
mat_pred = np.array(z_env3_1_norm)
mat_true = z_env3_1_true
io.savemat('env3_4_pred.mat', {'res:'mat_pred})
io.savemat('env3_4_true.mat', {'res:'mat_true})


z_env4_6_sum = []
for k in range(100):
    z_env4_6_sum.append(np.sum(all_angle_loss[k]))