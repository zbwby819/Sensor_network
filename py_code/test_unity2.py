# -*- coding: utf-8 -*-
"""
Created on Fri Aug 21 14:05:22 2020

@author: azrael
"""


sensor_loc = [(16, 0, 3),  (14, 0, 14), (3, 0, 22), (12, 0, 31),
                  (21, 0, 39), (23, 0, 24), (26, 0, 12), (33, 0, 33), (31, 0, 9)] #env-3-7
label_path = "target_env3_7.txt"
ad_matrix = np.load('test_admatrix_env3_7.npy')
ad_matrix2 = np.load('test_admatrix2_env3_7.npy')

sensor_loc = [(13, 0, 13),  (4, 0, 15), (7, 0, 26), (14, 0, 37),
                  (22, 0, 27), (26, 0, 7), (29, 0, 19), (29, 0, 21), (34, 0, 33)] #env-4-7
label_path = "target_env4_7.txt"
ad_matrix = np.load('test_admatrix_env4_7.npy')
ad_matrix2 = np.load('test_admatrix2_env4_7.npy')



select_group=7
filePath = 'training_env4_{}/sensor_1/1'.format(select_group)
filelist = os.listdir(filePath)
filelist.sort(key = lambda x: int(x[:-4]))
#all_label = s_label(1)
target_label = open(label_path,"r") 
lines = target_label.readlines() 
#select_case = [np.random.randint(len(lines)) for _ in range(batch_size)]
select_case = np.arange(batch_size)

all_label = s_label_batch(select_group, select_case,4)
#np.save('stage1_label.npy', all_label)
#print('datasize:', len(all_label))

testmodel = khop_model_share()
testmodel.load_weights('result_0820/gnn_khop_env34_share_0820.h5')


batch_matrix1 = np.zeros((1, num_sensors, num_sensors)) #the first dimension should be 1 for test
batch_matrix2 = np.zeros((1, num_sensors, num_sensors))    
for i in range(1):
    batch_matrix1[i] = localpooling_filter(ad_matrix)
    batch_matrix2[i] = localpooling_filter(ad_matrix2)

#randomly select a batch of sample
#select_case = [np.random.randint(1,len(all_label)) for _ in range(batch_size)] 
batch_input = []
batch_output = all_label.copy()

all_result = []
all_angle_pred = []
all_angle_true = []
for i in range(batch_size):
    sensor_pose =[]
    all_sensor_input = np.zeros((num_sensors, 84, 84*4, 3))
    all_sensor_output = np.zeros((num_sensors, 1))

    #sensor_pose_input = np.zeros((num_sensors, 2))
    for idx_sensor in range(num_sensors):
        sensor_path = 'training_env4_{}/'.format(select_group) + all_sensors[idx_sensor]
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
    angle_pred, angle_true = plot_angle(sensor_loc, res, all_label[i])
    all_angle_pred.append(angle_pred)
    all_angle_true.append(angle_true)
    
all_angle_loss = []
for i in range(len(all_angle_pred)):
    a_true = all_angle_true[i]
    a_pred = all_angle_pred[i]
    angle_loss = []
    for j in range(9):
        cur_loss = (a_pred[j][0]/np.sqrt(a_pred[j][0]**2+a_pred[j][1]**2), 
                    a_pred[j][1]/np.sqrt(a_pred[j][0]**2+a_pred[j][1]**2))
        
        a_loss = abs(a_true[j][0]- cur_loss[0])+ abs(a_true[j][1]- cur_loss[1])
        angle_loss.append(a_loss.copy())
    all_angle_loss.append(angle_loss.copy())
    
z_env3_1_average = []
z_env3_1_std = []    
for i in range(len(z_env3_1_test)):
    z_env3_1_average.append(np.average(z_env3_1_test[i]))
    z_env3_1_std.append(np.average(z_env3_1_test[i]))
    
    

    