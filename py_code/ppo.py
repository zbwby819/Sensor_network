# -*- coding: utf-8 -*-
"""
Created on Sat Jul 25 21:00:24 2020

@author: azrael
"""


from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
import numba as nb
#from tensorboardX import SummaryWriter
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os, cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input, Concatenate, Flatten, Reshape
from keras.callbacks import TensorBoard
from resnet import resnet_sensor_network, sensor_cnn
from spektral.layers import GraphConv
from loc2dir import s_label, sen_angle, s_label_batch
from loc2dir import theta


CONTINUOUS = True

EPISODES = 10000
Learning_rate = 3e-5
LOSS_CLIPPING = 0.2 # Only implemented clipping for the surrogate loss, paper said it was best
EPOCHS = 1
NOISE = 1.0 # Exploration noise

GAMMA = 0.99
ZERO_TOLERANCE = 1e-10

BUFFER_SIZE = 512
BATCH_SIZE = 32
NUM_ACTIONS = 2
NUM_STATE = 40*40
ENTROPY_LOSS = 5e-3
num_each_round = 2000
train_steps = 100000

DUMMY_ACTION, DUMMY_VALUE = np.zeros((1, NUM_ACTIONS)), np.zeros((1, 1))
DUMMY_LABEL = np.zeros((1, NUM_ACTIONS))

all_sensors = ['sensor_1', 'sensor_2', 'sensor_3', 'sensor_4',
               'sensor_5', 'sensor_6', 'sensor_7', 'sensor_8',
               'sensor_9']

@nb.jit
def robot_khop_model(): # input/output = num of sensors 
    num_sensors = 9
    input_shape = (84, 84*4, 3)        
    sensor_matrix1 = Input(shape=(num_sensors+1, num_sensors+1))
    sensor_matrix2 = Input(shape=(num_sensors+1, num_sensors+1))
    #sensor_matrix3 = Input(shape=(num_sensors, num_sensors))
    r_input = Input(shape=(input_shape[0], input_shape[1], input_shape[2]))
    s_input1 = Input(shape=(input_shape[0], input_shape[1], input_shape[2]))
    s_input2 = Input(shape=(input_shape[0], input_shape[1], input_shape[2]))
    s_input3 = Input(shape=(input_shape[0], input_shape[1], input_shape[2]))
    s_input4 = Input(shape=(input_shape[0], input_shape[1], input_shape[2]))
    s_input5 = Input(shape=(input_shape[0], input_shape[1], input_shape[2]))
    s_input6 = Input(shape=(input_shape[0], input_shape[1], input_shape[2]))
    s_input7 = Input(shape=(input_shape[0], input_shape[1], input_shape[2]))
    s_input8 = Input(shape=(input_shape[0], input_shape[1], input_shape[2]))
    s_input9 = Input(shape=(input_shape[0], input_shape[1], input_shape[2]))

    
    s_cnn = sensor_cnn(input_shape, repetitions = [2,2,2,2])
    robot_cnn = s_cnn(r_input)
    extract_cnn1 = s_cnn(s_input1)
    extract_cnn2 = s_cnn(s_input2)
    extract_cnn3 = s_cnn(s_input3)
    extract_cnn4 = s_cnn(s_input4)
    extract_cnn5 = s_cnn(s_input5)
    extract_cnn6 = s_cnn(s_input6)
    extract_cnn7 = s_cnn(s_input7)
    extract_cnn8 = s_cnn(s_input8)
    extract_cnn9 = s_cnn(s_input9)
    
    extract_cnn = Concatenate(axis=1)([extract_cnn1, extract_cnn2, extract_cnn3, 
                                       extract_cnn4, extract_cnn5, extract_cnn6,
                                       extract_cnn7, extract_cnn8, extract_cnn9, robot_cnn])
        
    #extract_cnn = np.reshape(extract_cnn, (-1,))
    G_h1 = GraphConv(256, 'relu')([extract_cnn, sensor_matrix1])
    G_h2 = GraphConv(256, 'relu')([extract_cnn, sensor_matrix2])
    G_1 = Concatenate(axis=-1)([G_h1, G_h2])
  
    G_2h1 = GraphConv(256, 'relu')([G_1, sensor_matrix1])
    G_2h2 = GraphConv(256, 'relu')([G_1, sensor_matrix2])
    G_2 = Concatenate(axis=-1)([G_2h1, G_2h2])
    
    gnn_output = tf.split(G_2, num_sensors+1, 1)
    
    r_output = Dense(64, activation='relu', name='policy_mlp')(Flatten()(gnn_output[-1]))
    output1 = Dense(2, activation='linear', name='robot_loss')(r_output)
    
    
    model = Model(inputs=[s_input1, s_input2, s_input3, s_input4,
                          s_input5, s_input6, s_input7, s_input8, s_input9,
                          r_input,
                          sensor_matrix1, sensor_matrix2], 
                  outputs= [output1])
    return model

def exponential_average(old, new, b1):
    return old * b1 + (1-b1) * new


def proximal_policy_optimization_loss(advantage, old_prediction):
    def loss(y_true, y_pred):
        prob = K.sum(y_true * y_pred, axis=-1)
        old_prob = K.sum(y_true * old_prediction, axis=-1)
        r = prob/(old_prob + 1e-10)
        return -K.mean(K.minimum(r * advantage, K.clip(r, min_value=1 - LOSS_CLIPPING, max_value=1 + LOSS_CLIPPING) * advantage) + ENTROPY_LOSS * -(prob * K.log(prob + 1e-10)))
    return loss


def proximal_policy_optimization_loss_continuous(advantage, old_prediction):
    def loss(y_true, y_pred):
        var = tf.square(NOISE)
        pi = 3.1415926
        denom = tf.sqrt(2 * pi * var)
        prob_num = tf.exp(- tf.square(y_true - y_pred) / (2 * var))
        old_prob_num = tf.exp(- tf.square(y_true - old_prediction) / (2 * var))

        prob = prob_num/denom
        old_prob = old_prob_num/denom
        r = prob/(old_prob + 1e-10)

        return -tf.reduce_mean(tf.minimum(r * advantage, tf.clip_by_value(r, 1 - LOSS_CLIPPING, 1 + LOSS_CLIPPING) * advantage))
    return loss

def proximal_policy_optimization_loss_con(advantage, old_prediction, y_true, y_pred):
    var = tf.square(NOISE)
    pi = 3.1415926
    denom = tf.sqrt(2 * pi * var)
    prob_num = tf.exp(- tf.square(y_true - y_pred) / (2 * var))
    old_prob_num = tf.exp(- tf.square(y_true - old_prediction) / (2 * var))

    prob = prob_num/denom
    old_prob = old_prob_num/denom
    r = prob/(old_prob + 1e-10)

    return -tf.reduce_mean(tf.minimum(r * advantage, tf.clip_by_value(r, 1 - LOSS_CLIPPING, 1 + LOSS_CLIPPING) * advantage))

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
        all_sensor_input[idx_sensor,:, 84*3:84*4,:] = img_array_4/255 
        all_sensor_input[idx_sensor,:, 84*2:84*3,:] = img_array_3/255
        all_sensor_input[idx_sensor,:, 84*1:84*2,:] = img_array_2/255
        all_sensor_input[idx_sensor,:, 84*0:84*1,:] = img_array_1/255   
    return all_sensor_input

def change_axis(img, loc):
    env_x, env_z = img.shape
    loc_x = loc[0]
    loc_z = loc[-1]
    return (round(env_x/2 - loc_z), 0, round(env_z/2 + loc_x))

class Agent:
    def __init__(self, env):
        self.critic = self.build_critic()   
        self.actor = self.build_actor_con()
        
        self.episode = 0
        self.env = env
        #self.env.reset()
        self.name = self.get_name()
        self.gradient_steps = 0
        self.behavior_names = self.env.get_behavior_names()
        self.behavior_specs = self.env.get_behavior_spec(self.behavior_names[0])
        (self.DecisionSteps, self.TerminalSteps) = self.env.get_steps(self.behavior_names[0])
        
        self.observation = self.DecisionSteps.obs
        self.val = False
        self.reward = []
        self.reward_over_time = []
        self.input_shape = (84, 84*4, 3)
        self.vector_obs = self.DecisionSteps.obs[-1][0]
        self.robot_start = self.vector_obs[3:6]
        self.target_loc = self.vector_obs[:3]
        self.dis_env = np.load('env_3.npy')
        self.connect_point = self.cal_connect_point(env)
        self.origin_connect_point = self.cal_connect_point(env)
        self.sub_target = 1
 
    def cal_connect_point(self, env):
        (DecisionSteps, TerminalSteps) = env.get_steps(self.behavior_names[0])
        vector_obs = DecisionSteps.obs[-1][0]
        robot_start = vector_obs[3:6]
        target_loc = vector_obs[:3]
        dis_env = np.load('env_3.npy')
        robot_change = np.array(change_axis(dis_env, robot_start))
        target_change = np.array(change_axis(dis_env, target_loc))
        #print('start_loc:',robot_change, '  Target_loc:', target_change)
        if robot_change[0] <=0:
            robot_change[0] = 0 
        if robot_change[-1] <=0:
            robot_change[-1] = 0 
        if robot_change[0] >=39:
            robot_change[0] = 39 
        if robot_change[-1] >=39:
            robot_change[-1] = 39 
        
        if target_change[0] <=0:
            target_change[0] = 0 
        if target_change[-1] <=0:
            target_change[-1] = 0 
        if target_change[0] >=39:
            target_change[0] = 39 
        if target_change[-1] >=39:
            target_change[-1] = 39 
        #print('start_loc:',robot_change, '  Target_loc:', target_change)
        connect_point = theta(dis_env, (robot_change[0], robot_change[-1]), 
                                   (target_change[0], target_change[-1]))
        return connect_point
 
    def cal_admatrix(self, env):
        (cur_DecisionSteps, cur_TerminalSteps) = env.get_steps(self.behavior_names[0])
        robot_loc = cur_DecisionSteps.obs[-1][0][3:6]
        sensor_loc = [(-13, 0, 14), (-4, 0, 6), (0, 0, -7), (-11, 0, -16),
                      (8, 0, 13), (19, 0, 5), (6, 0, 2), (17, 0, -9), (6, 0, -19)]
        ad_matrix1 = np.zeros((10,10))
        ad_matrix2 = np.zeros((10,10))
        ad_matrix1[:9,:9] = np.load('ad_matrix_env3_1.npy')
        ad_matrix2[:9,:9] = np.load('ad_matrix_env3_2.npy')
        for j, sen in enumerate(sensor_loc):
            if np.sqrt((robot_loc[0]-sen[0])**2+(robot_loc[-1]-sen[-1])**2) <= 15:
                ad_matrix1[-1, j] = 1
                ad_matrix1[j, -1] = 1
            elif np.sqrt((robot_loc[0]-sen[0])**2+(robot_loc[-1]-sen[-1])**2) <= 30:
                ad_matrix2[-1, j] = 1
                ad_matrix2[j, -1] = 1
        return np.expand_dims(ad_matrix1, axis=0), np.expand_dims(ad_matrix2,axis=0)
                
    def cal_angle(self, env):
        (DecisionSteps, TerminalSteps) = env.get_steps(self.behavior_names[0])
        self.connect_point = self.cal_connect_point(env)
        if self.origin_connect_point[0] == self.origin_connect_point[-1]:
            self.reset_env(env)
        cur_loc = DecisionSteps.obs[-1][0][3:6]
        cur_loc = change_axis(self.dis_env, cur_loc)
        if self.sub_target >=(len(self.origin_connect_point)-1):
            self.sub_target = int(len(self.origin_connect_point)-1)
        robot2cp = np.sqrt((cur_loc[0]-self.origin_connect_point[self.sub_target][0])**2+(cur_loc[-1]-self.origin_connect_point[self.sub_target][1])**2)
        #start2cp = np.sqrt((self.connect_point[self.sub_target][0]-self.connect_point[self.sub_target-1][0])**2+
        #                   (self.connect_point[self.sub_target][1]-self.connect_point[self.sub_target-1][1])**2)
        if robot2cp <= 2:
            if self.sub_target >=(len(self.origin_connect_point)-1):
                pass
            else:
                self.sub_target =int(self.sub_target+1)
        robot2cp_angle = (self.origin_connect_point[self.sub_target][0]-cur_loc[0],
                          self.origin_connect_point[self.sub_target][-1]-cur_loc[-1])
        #return np.divide(robot2cp_angle,  sum(robot2cp_angle)+1e-10) - binyu
        
        # qingbiao
        p_new = np.zeros((2,1))
        p_new[0] = np.true_divide(robot2cp_angle[0], np.sqrt(robot2cp_angle[0]**2 + robot2cp_angle[1]**2) + ZERO_TOLERANCE)
        p_new[1] = np.true_divide(robot2cp_angle[1], np.sqrt(robot2cp_angle[0]**2 + robot2cp_angle[1]**2) + ZERO_TOLERANCE)
        return p_new


    def form_obs(self, env):
        sensor_obs = collect_sen_obs()
        (DecisionSteps, TerminalSteps) = env.get_steps(self.behavior_names[0])
        visual_obs = DecisionSteps.obs[:-1]
        robot_obs = np.zeros((1, 84, 336, 3))
        robot_obs[0,:, 84*3:84*4,:] = visual_obs[3]
        robot_obs[0,:, 84*2:84*3,:] = visual_obs[2]
        robot_obs[0,:, 84*1:84*2,:] = visual_obs[1]
        robot_obs[0,:, 84*0:84*1,:] = visual_obs[0]
        input_obs = np.concatenate((sensor_obs, robot_obs), axis=0)
        return input_obs
        
    def get_name(self):
        name = 'AllRuns/'
        name += 'continous/'
        return name

    def build_actor_continuous(self):
        b_model = robot_khop_model()
        #b_model.load_weights('gnn_khop_env3_share.h5', by_name=True)
        advantage = Input(shape=(1,))
        old_prediction = Input(shape=(NUM_ACTIONS,))
        model = Model(inputs=[b_model.input, advantage, old_prediction], outputs=[b_model.output])        
        
        model.compile(optimizer=Adam(learning_rate=Learning_rate),
                      loss=[proximal_policy_optimization_loss_continuous(
                          advantage=advantage,
                          old_prediction=old_prediction)])
        return model
    
    def build_actor_con(self):
        b_model = robot_khop_model()
        #b_model.load_weights('gnn_khop_env3_share.h5', by_name=True)
        advantage = Input(shape=(1,))
        old_prediction = Input(shape=(NUM_ACTIONS,))
        y_true = Input(shape=(NUM_ACTIONS,))
        model = Model(inputs=[b_model.input, advantage, old_prediction, y_true], outputs=[b_model.output])        
        model.add_loss(proximal_policy_optimization_loss_con(advantage, old_prediction, y_true, b_model.output))
        model.compile(optimizer=Adam(learning_rate=Learning_rate))
        return model

    def build_critic(self):
        b_model = robot_khop_model()
        #b_model.load_weights('gnn_khop_env3_share.h5', by_name=True)
        out_value = Dense(1)(b_model.layers[-2].output)
        model = Model(inputs=b_model.input, outputs=[out_value])
        model.compile(optimizer=Adam(lr=Learning_rate), loss='mse')

        return model

    def reset_env(self, env):
        self.episode += 1
        env.reset()
        (self.DecisionSteps, self.TerminalSteps) = env.get_steps(self.behavior_names[0])
        self.observation = self.DecisionSteps.obs
        self.reward = []
        self.origin_connect_point = self.cal_connect_point(env)

    def get_action_continuous(self, env):
        (DecisionSteps, TerminalSteps) = env.get_steps(self.behavior_names[0])
        cur_obs = self.form_obs(env)
        #cur_obs = np.expand_dims(cur_obs, axis=1)
        ad_matrix1, ad_matrix2 = self.cal_admatrix(env)
        p = self.actor.predict([np.expand_dims(cur_obs[1], axis=0),np.expand_dims(cur_obs[2], axis=0), np.expand_dims(cur_obs[3], axis=0), 
                                np.expand_dims(cur_obs[4], axis=0),np.expand_dims(cur_obs[5], axis=0), np.expand_dims(cur_obs[6], axis=0), 
                                np.expand_dims(cur_obs[7], axis=0),np.expand_dims(cur_obs[8], axis=0), np.expand_dims(cur_obs[9], axis=0), 
                                np.expand_dims(cur_obs[0], axis=0), ad_matrix1, ad_matrix2, 
                                DUMMY_VALUE, DUMMY_ACTION, DUMMY_LABEL])
        # method 1:  p /= abs(sum(p[0]))
        p_new = np.zeros((1,2))
        p_new[0][0] = np.true_divide(p[0][0], np.sqrt(p[0][0]**2 + p[0][1]**2) + ZERO_TOLERANCE)
        p_new[0][1] = np.true_divide(p[0][1], np.sqrt(p[0][0]**2 + p[0][1]**2) + ZERO_TOLERANCE)
        action = action_matrix = p_new[0] + np.random.normal(loc=0, scale=NOISE, size=p_new[0].shape)
        return action, action_matrix, p_new

    def transform_reward(self, reward_list):
        cal_reward = reward_list.copy()
        for j in range(len(reward_list) - 2, -1, -1):
            cal_reward[j] += cal_reward[j + 1] * GAMMA
        return cal_reward

    def get_batch(self, env):
        batch = [[], [], [], []]

        tmp_batch = [[], [], [], []]
        done = 0
        while len(batch[0]) < BUFFER_SIZE:
            env_steps = 0
            action, action_matrix, predicted_action = self.get_action_continuous(env)
            env.step()
            env.set_actions(self.behavior_names[0], np.expand_dims(action,axis=0))
            env_steps += 1
            env.step()
            if env_steps >= num_each_round:
                done = 1
                env_steps = 0
            #self.env.step()
            (DecisionSteps, TerminalSteps) = env.get_steps(self.behavior_names[0])
            observation = self.form_obs(env)
            true_angle = self.cal_angle(env)
            #action = action/sum(action)
            
            # print("target:",true_angle, "predict:",predicted_action )

            # r = -sqrt((a_x - a'_x)**2 +  (a_y - a'_y)**2) + np.sqrt(2)
            reward = -np.true_divide(np.sqrt((true_angle[0]-predicted_action[0][0])**2+(true_angle[1]-predicted_action[0][1])**2), np.sqrt(2)) + 1
            
            vector_obs = DecisionSteps.obs[-1][0]
            self.robot_loc = vector_obs[3:6]
            self.target_loc = vector_obs[:3]
            #if sum(self.robot_loc - self.target_loc) <=2.5:
             #   done = 1
              #  reward +=1
            if sum(self.robot_loc - self.target_loc) <=2.5:
                done = 1
                # reach goal reward
                reward =10
            if self.robot_loc[0] <=-20 or self.robot_loc[0] >=20:
                done = 1
            if self.robot_loc[1] <=-20 or self.robot_loc[1] >=20:
                done = 1
            
            tmp_batch[0].append(np.expand_dims(observation[-1],axis=0))
            tmp_batch[1].append(action_matrix)
            tmp_batch[2].append(predicted_action)
            tmp_batch[3].append(reward)
            self.reward.append(reward)
            self.observation = observation
            if done:

                tmp_batch[3] = self.transform_reward(tmp_batch[3])
                #print("\t\t Temp_batch size: {} - reward size: {} ".format(len(tmp_batch[0]), len(self.reward)))    
                if self.val is False:
                    for i in range(len(tmp_batch[0])):
                        
                        #print("\t\t Reward: ", self.reward[i])
                        obs, action, pred, r = tmp_batch[0][i], tmp_batch[1][i], tmp_batch[2][i], tmp_batch[3][i]
                        #r = self.reward[i]
                        batch[0].append(obs)
                        batch[1].append(action)
                        batch[2].append(pred)
                        batch[3].append(r)
                tmp_batch = [[], [], [], []]
                self.reset_env(env)
                done = 0

        pred = np.array(batch[2]) 
        reward = np.reshape(np.array(batch[3]), (len(batch[3]), 1))
        pred = np.reshape(pred, (pred.shape[0], pred.shape[2]))
        
        print("Episode: {} - Current memory buff:{}/{} ".format(self.episode, len(batch[0]), BUFFER_SIZE))
        print("\t Current reward:{}, pred,{}".format(reward.shape, pred.shape))

        return np.array(batch[0]) , np.array(batch[1]), pred, reward

    def run(self, env):
        train_actor_loss = []
        train_critic_loss = []
        sensor_obs = collect_sen_obs() 
        while self.episode < EPISODES:
            obs, action, pred, reward = self.get_batch(env)
            obs, action, pred, reward = obs[:BUFFER_SIZE], action[:BUFFER_SIZE], pred[:BUFFER_SIZE], reward[:BUFFER_SIZE]
            
            new_obs = np.zeros([len(obs), 10, 84, 336, 3])
            for o_i in range(len(obs)):
                robot_obs = obs[o_i]
                new_obs[o_i] = np.concatenate((sensor_obs, robot_obs), axis=0) 
            obs = new_obs.copy()     
            del new_obs
            old_prediction = pred
            
            ad_matrix1, ad_matrix2 = self.cal_admatrix(env)
            batch_matrix1 = np.zeros((len(obs), 10, 10))
            batch_matrix2 = np.zeros((len(obs), 10, 10))
            for b_i in range(len(obs)):
                batch_matrix1[b_i] = ad_matrix1
                batch_matrix2[b_i] = ad_matrix2
                
            pred_values = self.critic.predict([obs[:,1],obs[:,2], obs[:,3], obs[:,4], obs[:,5], obs[:,6], 
                                 obs[:,7], obs[:,8], obs[:,9], obs[:,0], batch_matrix1, batch_matrix2])

            advantage = reward - pred_values
            advantage.astype('float64')
            advantage = K.cast_to_floatx(advantage)
            
            reward.astype('float64')
            reward = K.cast_to_floatx(reward)

            actor_loss = self.actor.fit([obs[:,1],obs[:,2], obs[:,3], obs[:,4], obs[:,5], obs[:,6], 
                                 obs[:,7], obs[:,8], obs[:,9], obs[:,0], batch_matrix1, batch_matrix2, advantage, old_prediction,action],   batch_size=BATCH_SIZE, shuffle=True, epochs=EPOCHS)
            critic_loss = self.critic.fit([obs[:,1],obs[:,2], obs[:,3], obs[:,4], obs[:,5], obs[:,6], 
                                 obs[:,7], obs[:,8], obs[:,9], obs[:,0], batch_matrix1, batch_matrix2], [reward], batch_size=BATCH_SIZE, shuffle=True, epochs=EPOCHS)
            train_actor_loss.append(actor_loss.history['loss'])
            train_critic_loss.append(critic_loss.history['loss'])
            self.gradient_steps += 1
            NOISE = 1-self.gradient_steps/train_steps
            if NOISE <=0:
                NOISE = 0
            if self.episode % 100 == 10:
            #if self.episode % 10 == 2:    
                print('save history and model {}'.format(self.episode))
                print('Data to Save: \n Loss_actor:{}, Loss_critic:{}'.format(train_actor_loss,train_critic_loss))
                #with open('actor_loss.npy', 'wb') as f:
                 #   np.save(f, np.asarray(train_actor_loss))
                #with open('critic_loss.npy', 'wb') as f2:
                 #   np.save(f2, np.asarray(train_critic_loss))
                
                np.save('actor_loss.npy', np.asarray(train_actor_loss))
                np.save('critic_loss.npy', np.asarray(train_critic_loss))
 
                self.actor.save('actor_model.h5')
                self.critic.save('critic_model.h5')
                
            #NOISE = 1* (EPISODES-self.gradient_steps)  



