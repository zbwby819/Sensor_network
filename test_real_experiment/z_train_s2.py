# -*- coding: utf-8 -*-
"""

@author: Win10
"""
import copy
import datetime
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing import image
from spektral.utils import localpooling_filter
from loc2dir import theta
import tensorflow_probability as tfp
from a_star import AStarSearch
from z_train_s1 import load_model_gcn

tfd = tfp.distributions

#tf.keras.backend.set_floatx('float64')

ZERO_TOLERANCE = 1e-10
pixel_dim = 84
input_shape = (pixel_dim,pixel_dim*4,3)   
action_dim = 2
num_sensor = 4
sensor_dis_threshold = 20

all_sensors = []
for i in range(num_sensor):
    all_sensors.append('sensor_{}'.format(i+1))

sensor_per_map = [4,5,7,6,8,9,9,10,10,10,10,10]
######################################################################
def collect_sen_obs(num_sensors=num_sensor , path='training/'):
    all_sensor_input = np.zeros((num_sensors, pixel_dim, pixel_dim*4, 3))
    #all_sensor_output = np.zeros((num_sensors, 2))
    for idx_sensor in range(num_sensors):
        sensor_path = path + all_sensors[idx_sensor]
        img_1 = image.load_img(sensor_path+'/1/1.png', target_size=(pixel_dim,pixel_dim))  #height-width
        img_array_1 = image.img_to_array(img_1)
        img_2 = image.load_img(sensor_path+'/2/1.png', target_size=(pixel_dim,pixel_dim))  #height-width
        img_array_2 = image.img_to_array(img_2)
        img_3 = image.load_img(sensor_path+'/3/1.png', target_size=(pixel_dim,pixel_dim))  #height-width
        img_array_3 = image.img_to_array(img_3)
        img_4 = image.load_img(sensor_path+'/4/1.png', target_size=(pixel_dim,pixel_dim))  #height-width
        img_array_4 = image.img_to_array(img_4)  
        all_sensor_input[idx_sensor,:, pixel_dim*3:pixel_dim*4,:] = img_array_1/255 
        all_sensor_input[idx_sensor,:, pixel_dim*2:pixel_dim*3,:] = img_array_2/255
        all_sensor_input[idx_sensor,:, pixel_dim*1:pixel_dim*2,:] = img_array_3/255
        all_sensor_input[idx_sensor,:, pixel_dim*0:pixel_dim*1,:] = img_array_4/255   
    return all_sensor_input

def form_obs(cur_state, sensor_input=None):
    if sensor_input is None:        
        sensor_obs = collect_sen_obs()
    else:
        sensor_obs = sensor_input
    visual_obs = cur_state[:4]
    robot_obs = np.zeros((1, pixel_dim, pixel_dim*4, 3))
    robot_obs[0,:, pixel_dim*3:pixel_dim*4,:] = visual_obs[0][0]
    robot_obs[0,:, pixel_dim*2:pixel_dim*3,:] = visual_obs[1][0]
    robot_obs[0,:, pixel_dim*1:pixel_dim*2,:] = visual_obs[2][0]
    robot_obs[0,:, pixel_dim*0:pixel_dim*1,:] = visual_obs[3][0]
    input_obs = np.concatenate((sensor_obs, robot_obs), axis=0)
    return input_obs

def cal_admatrix_sensor(sensor_loc, num_sensors=None):
    sensor_loc = sensor_loc.astype('float')
    if num_sensors == None:            
        num_sensors = len(sensor_loc) 
    ad_matrix1 = np.zeros((num_sensors,num_sensors))
    ad_matrix2 = np.zeros((num_sensors,num_sensors))
    ad_matrix3 = np.zeros((num_sensors,num_sensors))
    ad_matrix4 = np.zeros((num_sensors,num_sensors))
    for i in range(num_sensors-1):
        for j in range(i+1, num_sensors):
            if np.sqrt((sensor_loc[i][0]-sensor_loc[j][0])**2+(sensor_loc[i][1]-sensor_loc[j][1])**2) <=sensor_dis_threshold:
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
    #return np.expand_dims(ad_matrix1, axis=0), np.expand_dims(ad_matrix2,axis=0), np.expand_dims(ad_matrix3,axis=0), np.expand_dims(ad_matrix4,axis=0)
    return ad_matrix1

def cal_admatrix(pos, env_index, num_sensors=num_sensor, path='training/'):
    robot_loc = pos
    sensor_loc = np.load(path+'env_{}_sensor.npy'.format(env_index+1))
    ad_matrix1 = np.zeros((num_sensors+1,num_sensors+1))
    ad_matrix1[:num_sensors,:num_sensors] = cal_admatrix_sensor(sensor_loc, num_sensors=None)
    
    for i in range(num_sensors):
        if np.sqrt((robot_loc[0]-sensor_loc[i][0])**2+(robot_loc[1]-sensor_loc[i][1])**2) <=sensor_dis_threshold:
            ad_matrix1[-1][i] = 1
            ad_matrix1[i][-1] = 1 
    return np.expand_dims(ad_matrix1, axis=0)

def cal_connect_point(cur_state, env):
    robot_loc = [-cur_state[-1][0][5], 0.5, cur_state[-1][0][3]]  
    target_loc = [-cur_state[-1][0][2], 0.5, cur_state[-1][0][0]]
    dis_env = env      
    connect_point = theta(dis_env, (robot_loc[0], robot_loc[-1]), 
                               (target_loc[0], target_loc[-1]))
    return connect_point

def in_view(env, cur_loc, target_loc):
    #connect_point = cal_connect_point(cur_state)
    cur_loc = (np.round(cur_loc[0]),  np.round(cur_loc[-1]))
    cur_loc = (int(cur_loc[0]), int(cur_loc[1]))
    target_loc = (np.round(target_loc[0]),  np.round(target_loc[-1]))

    connect_p = target_loc
    past_grid = []
    past_grid_int = []
    x_flag = 1
    y_flag = 1
    f_x = int
    f_y = int
    if (cur_loc[1] - connect_p[1]) >=0:
        y_flag = -1
        f_y = np.ceil
    if (cur_loc[0] - connect_p[0]) >=0:
        x_flag = -1
        f_x = np.ceil
    if abs(connect_p[0]-cur_loc[0])<= abs(connect_p[1]-cur_loc[1]):    
        slocp = abs((connect_p[0]-cur_loc[0])/(connect_p[1]-cur_loc[1]+1e-10))
        for j in range(int(abs(cur_loc[1]- connect_p[1]))):
            cur_x, cur_y = (cur_loc[0]+ j*slocp*x_flag, cur_loc[1]+j*y_flag)
            fur_x, fur_y = (cur_loc[0]+ (j+1)*slocp*x_flag, cur_loc[1]+(j+1)*y_flag)
            past_grid.append((cur_x,cur_y))
            if f_x(fur_x) != f_x(cur_x) and fur_x != f_x(fur_x):
                past_grid_int.append((f_x(cur_x), f_y(cur_y)))
                past_grid_int.append((f_x(fur_x), f_y(cur_y)))
            else:
                past_grid_int.append((f_x(cur_x), f_y(cur_y)))
    else:  
        slocp = abs((connect_p[1]-cur_loc[1])/(connect_p[0]-cur_loc[0]+1e-10))
        for j in range(int(abs(cur_loc[0]-connect_p[0]))):
            cur_x, cur_y = (cur_loc[0]+j*x_flag, cur_loc[1]+slocp*j*y_flag)
            fur_x, fur_y = (cur_loc[0]+(j+1)*x_flag, cur_loc[1]+slocp*(j+1)*y_flag)
            past_grid.append((cur_x,cur_y))
            if f_y(fur_y) != f_y(cur_y) and fur_y != f_y(fur_y):
                past_grid_int.append((f_x(cur_x), f_y(cur_y)))
                past_grid_int.append((f_x(cur_x), f_y(fur_y)))
            else:
                past_grid_int.append((f_x(cur_x), f_y(cur_y)))
    past_grid_int.append(connect_p)
    for c_i, cur_grid in enumerate(past_grid_int):
        zc_1, zc_2 = cur_grid
        if int(cur_grid[0]) >= env.shape[0]:
            zc_1 = env.shape[0]-1
        if int(cur_grid[1]) >= env.shape[0]:
            zc_2 = env.shape[0]-1
        if env[int(zc_1), int(zc_2)] == 0:
            pass
        else:
            #print(c_i)
            return False
    return True

def hit_wall(env, loc, action):
    #print('loc:',loc)
    #print('action:', action)
    z1 = int(loc[0]+action[0])
    z2 = int(loc[-1]+action[-1])
    if z1 <=0:
        z1 = 0 
    if z2 <=0:
        z2 = 0 
    if z2 >=env.shape[0]:
        z2 = env.shape[0]-1
    if  z1>=env.shape[0]:
       z1 = env.shape[0]-1 
    if env[z1][z2] == 1:
        return True
    return False

def collision_avoid(env, loc, action):
    z1 = round(loc[0]+action[0])
    z2 = round(loc[-1]+action[-1])
    #print("z1:", z1)
    #print("loc:", loc)
    if z1 <=0:
        z1 = 0 
    if z2 <=0:
        z2 = 0 
    if z2 >=env.shape[0]:
        z2 = env.shape[0]-1
    if  z1>=env.shape[0]:
       z1 = env.shape[0]-1
    #print("next_loc is:", [z1,z2], '    value is: ', env[z1][z2])
    if env[z1][z2] == 1:
        if env[z1][round(loc[-1])] == 1:
            #print("y")
            return [0, action[1]], [-action[0],0]
        elif env[round(loc[0])][z2] == 1:
            #print("x")
            return [action[0], 0], [0,-action[1]]
    return action, [0,0]
    
class PPO:
    def __init__(
            self,
            env,
            discrete=False,
            c1=1.0,
            c2=0.01,
            clip_ratio=0.2,
            gamma=0.95,
            lam=0.95,
            batch_size=32,
            max_steps = 512
    ):
        self.env = env
        self.env_index = 0
        self.path = 'training/'
        self.dis_env = np.load(self.path+'env_{}.npy'.format(self.env_index+1))  
        self.action_dim = 2  # number of actions
        self.discrete = discrete
        if not discrete:
            self.action_bound = 1
            self.action_shift = 0

        self.lr = 1e-6
        # Define and initialize network
        self.policy = load_model_gcn(num_sensor)
        self.policy_cnn = load_model_gcn(num_sensor)
        self.model_optimizer = Adam(learning_rate=self.lr)
        #print(self.policy.summary())
        # Stdev for continuous action
        if not discrete:
            self.policy_log_std = tf.Variable(tf.zeros(self.action_dim, dtype=tf.float32), trainable=True)

        # Set hyperparameters
        self.gamma = gamma  # discount factor
        self.lam = lam
        self.c1 = c1  # value difference coeff
        self.c2 = c2  # entropy coeff
        self.clip_ratio = clip_ratio  # for clipped surrogate
        self.batch_size = batch_size
        self.max_steps = max_steps
        self.n_updates = max_steps//batch_size  # number of epochs per episode
        self.sub_target = 1
        self.is_att = True
        self.is_cnn = True
        # Tensorboard
        self.summaries = {}
    
    def cal_angle2(self, cur_state, env):
        #connect_point = cal_connect_point(cur_state) 
        cur_loc = (round(-cur_state[-1][0][5]),  round(cur_state[-1][0][3]))
        target_loc = (round(-cur_state[-1][0][2]),  round(cur_state[-1][0][0]))
        is_view = []
        for connect_p in self.origin_connect_point[1:]:
            past_grid = []
            past_grid_int = []
            x_flag = 1
            y_flag = 1
            f_x = int
            f_y = int
            if (cur_loc[1] - connect_p[1]) >=0:
                y_flag = -1
                f_y = np.ceil
            if (cur_loc[0] - connect_p[0]) >=0:
                x_flag = -1
                f_x = np.ceil
            if abs(connect_p[0]-cur_loc[0])<= abs(connect_p[1]-cur_loc[1]):    
                slocp = abs((connect_p[0]-cur_loc[0])/(connect_p[1]-cur_loc[1]+1e-10))
                for j in range(int(abs(cur_loc[1]- connect_p[1]))):
                    cur_x, cur_y = (cur_loc[0]+ j*slocp*x_flag, cur_loc[1]+j*y_flag)
                    fur_x, fur_y = (cur_loc[0]+ (j+1)*slocp*x_flag, cur_loc[1]+(j+1)*y_flag)
                    past_grid.append((cur_x,cur_y))
                    if f_x(fur_x) != f_x(cur_x) and fur_x != f_x(fur_x):
                        past_grid_int.append((f_x(cur_x), f_y(cur_y)))
                        past_grid_int.append((f_x(fur_x), f_y(cur_y)))
                    else:
                        past_grid_int.append((f_x(cur_x), f_y(cur_y)))
            else:  
                slocp = abs((connect_p[1]-cur_loc[1])/(connect_p[0]-cur_loc[0]+1e-10))
                for j in range(int(abs(cur_loc[0]-connect_p[0]))):
                    cur_x, cur_y = (cur_loc[0]+j*x_flag, cur_loc[1]+slocp*j*y_flag)
                    fur_x, fur_y = (cur_loc[0]+(j+1)*x_flag, cur_loc[1]+slocp*(j+1)*y_flag)
                    past_grid.append((cur_x,cur_y))
                    if f_y(fur_y) != f_y(cur_y) and fur_y != f_y(fur_y):
                        past_grid_int.append((f_x(cur_x), f_y(cur_y)))
                        past_grid_int.append((f_x(cur_x), f_y(fur_y)))
                    else:
                        past_grid_int.append((f_x(cur_x), f_y(cur_y)))
            past_grid_int.append(connect_p)
            for c_i, cur_grid in enumerate(past_grid_int):
                if cur_grid[0] <=0:
                    cur_grid = (0, cur_grid[1])
                if cur_grid[1] <=0:
                    cur_grid = (cur_grid[0], 0) 
               # if cur_grid[0] >=39:
               #     cur_grid = (39, cur_grid[1])
               # if  cur_grid[1]>=39:
               #     cur_grid = (cur_grid[0], 39) 
                if env[int(cur_grid[0]), int(cur_grid[1])] == 0:
                    pass
                else:
                    #print(c_i)
                    break
                    
                if c_i == len(past_grid_int)-1:
                    is_view.append(connect_p)
        if len(is_view) == 0:
            new_connect_point = theta(env, (cur_loc[0], cur_loc[-1]), 
                           (target_loc[0], target_loc[-1]))
            is_view.append(new_connect_point[1])
        robot2cp = np.sqrt((cur_loc[0]-is_view[-1][0])**2+
                   (cur_loc[-1]-is_view[-1][1])**2)
        robot2cp_angle = (is_view[-1][0]-cur_loc[0],
                  is_view[-1][1]-cur_loc[1])
        p_new = np.zeros((2,1))
        p_new[0] = np.true_divide(robot2cp_angle[0], np.sqrt(robot2cp_angle[0]**2 + robot2cp_angle[1]**2) + ZERO_TOLERANCE)
        p_new[1] = np.true_divide(robot2cp_angle[1], np.sqrt(robot2cp_angle[0]**2 + robot2cp_angle[1]**2) + ZERO_TOLERANCE)
        return p_new
                
    def get_dist(self, output):
        if self.discrete:
            dist = tfd.Categorical(probs=output)
        else:
            std = tf.math.exp(self.policy_log_std)
            dist = tfd.Normal(loc=output, scale=std)

        return dist

    def evaluate_actions(self, state, action):
        if self.is_cnn == False:
            cur_obs = np.zeros((len(state),10,84,336,3))
            ad_matrix1, ad_matrix2, ad_matrix3, ad_matrix4 = [], [], [], []
            for j in range(len(state)):  
                new_state = form_obs(state[j], self.sensor_obs)
                cur_obs[j] = new_state
                robot_pos = [-state[j][-1][0][5], 0.5, state[j][-1][0][3]]
                new_matrix1, new_matrix2, new_matrix3, new_matrix4 = cal_admatrix(robot_pos, self.env_index)
                if self.is_att == False:
                    ad_matrix1.append(localpooling_filter(new_matrix1))
                    ad_matrix2.append(localpooling_filter(new_matrix2))
                    ad_matrix3.append(localpooling_filter(new_matrix2))
                    ad_matrix4.append(localpooling_filter(new_matrix2))
                else:
                    ad_matrix1.append(new_matrix1)
                    ad_matrix2.append(new_matrix2)
                    ad_matrix3.append(new_matrix2)
                    ad_matrix4.append(new_matrix2)
            output, value = self.policy([cur_obs[:,0], cur_obs[:,1], cur_obs[:,2], cur_obs[:,3], cur_obs[:,4], cur_obs[:,5],
                                         cur_obs[:,6], cur_obs[:,7], cur_obs[:,8], cur_obs[:,9], 
                                         ad_matrix1, ad_matrix2, ad_matrix3, ad_matrix4])
        else:
            cur_obs = np.zeros((len(state),1,84,336,3))  
            for j in range(len(state)):  
                new_state = form_obs(state[j])
                cur_obs[j] = new_state
                robot_pos = [-state[j][-1][0][5], 0.5, state[j][-1][0][3]]
            output, value = self.policy_cnn([cur_obs[:,0]])
        
        #output = tf.clip_by_value(output, -1, 1)
        dist = self.get_dist(tf.cast(output, dtype=tf.float32))
        if not self.discrete:
            action = (action - self.action_shift) / self.action_bound

        log_probs = dist.log_prob(action)
        #action = tf.clip_by_value(action, -1, 1)
        if not self.discrete:
            log_probs = tf.reduce_sum(log_probs, axis=-1)

        entropy = dist.entropy()

        return log_probs, entropy, value

    def act(self, state, test=False):
        if self.is_cnn == False:       
            robot_pos = [-state[-1][0][5], 0.5, state[-1][0][3]]
            cur_obs = form_obs(state)
            #cur_obs = np.expand_dims(state, axis=0).astype(np.float32)
            ad_matrix1, ad_matrix2, ad_matrix3, ad_matrix4 = cal_admatrix(robot_pos, self.env_index)
            if self.is_att == False:
                ad_matrix1, ad_matrix2 = localpooling_filter(ad_matrix1), localpooling_filter(ad_matrix2)
                ad_matrix3, ad_matrix4 = localpooling_filter(ad_matrix3), localpooling_filter(ad_matrix4)
            else:
                pass            
            output, value = self.policy.predict([np.expand_dims(cur_obs[0], axis=0),np.expand_dims(cur_obs[1], axis=0), np.expand_dims(cur_obs[2], axis=0), 
                                                 np.expand_dims(cur_obs[3], axis=0),np.expand_dims(cur_obs[4], axis=0), np.expand_dims(cur_obs[5], axis=0), 
                                                 np.expand_dims(cur_obs[6], axis=0),np.expand_dims(cur_obs[7], axis=0), np.expand_dims(cur_obs[8], axis=0), 
                                                 np.expand_dims(cur_obs[9], axis=0), ad_matrix1, ad_matrix2, ad_matrix3, ad_matrix4])
        else:
            robot_pos = [-state[-1][0][5], 0.5, state[-1][0][3]]
            cur_obs = form_obs(state)
            output, value = self.policy_cnn.predict([cur_obs])
        output = tf.clip_by_value(output, -1, 1)
        dist = self.get_dist(output)

        if self.discrete:
            action = tf.math.argmax(output, axis=-1) if test else dist.sample()
            log_probs = dist.log_prob(action)
        else:
            action = output if test else dist.sample()
            action = tf.clip_by_value(action, -1, 1)
            #action = action.astype(np.float32)
            log_probs = tf.reduce_sum(dist.log_prob(action), axis=-1)
            action = action * self.action_bound + self.action_shift
        return action[0].numpy(), value[0][0], log_probs[0].numpy()

    def save_model(self, fn):
        self.policy.save(fn)

    def load_model(self, fn):
        self.policy.load_weights(fn)
        print(self.policy.summary())

    def get_gaes(self, rewards, v_preds, next_v_preds):
        # source: https://github.com/uidilr/ppo_tf/blob/master/ppo.py#L98
        deltas = [r_t + self.gamma * v_next - v for r_t, v_next, v in zip(rewards, next_v_preds, v_preds)]
        gaes = copy.deepcopy(deltas)
        for t in reversed(range(len(gaes) - 1)):  # is T-1, where T is time step which run policy
            gaes[t] = gaes[t] + self.lam * self.gamma * gaes[t + 1]
        return gaes

    def learn(self, observations, actions, log_probs, next_v_preds, rewards, gaes):
        if len(rewards.shape) == 1:
            rewards = np.expand_dims(rewards, axis=-1).astype(np.float32)
        if len(next_v_preds.shape) == 1:    
            next_v_preds = np.expand_dims(next_v_preds, axis=-1).astype(np.float32)

        with tf.GradientTape(persistent=True,) as tape:
            new_log_probs, entropy, state_values = self.evaluate_actions(observations, actions)

            ratios = tf.exp(new_log_probs - log_probs)
            clipped_ratios = tf.clip_by_value(ratios, clip_value_min=1-self.clip_ratio,
                                              clip_value_max=1+self.clip_ratio)
            loss_clip = tf.minimum(gaes * ratios, gaes * clipped_ratios)
            loss_clip = tf.reduce_mean(loss_clip)

            target_values = rewards + self.gamma * next_v_preds
            vf_loss = tf.reduce_mean(tf.math.square(state_values - target_values))
            vf_loss = tf.cast(vf_loss, tf.float32)

            entropy = tf.reduce_mean(entropy)
            total_loss = -loss_clip + self.c1 * vf_loss - self.c2 * entropy

        train_variables = self.policy.trainable_variables
        if not self.discrete:
            train_variables += [self.policy_log_std]
        grad = tape.gradient(total_loss, train_variables)  # compute gradient
        self.model_optimizer.apply_gradients(zip(grad, train_variables))

        # tensorboard info
        self.summaries['total_loss'] = total_loss
        self.summaries['surr_loss'] = loss_clip
        self.summaries['vf_loss'] = vf_loss
        self.summaries['entropy'] = entropy

    def train(self, max_epochs=100000, save_freq=100):
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        train_log_dir = 'logs/' + current_time
        summary_writer = tf.summary.create_file_writer(train_log_dir)

        max_steps = self.max_steps
        episode, epoch = 0, 0
        while epoch < max_epochs:
            if epoch == 5000:
                self.lr = self.lr/10
                self.model_optimizer = Adam(learning_rate=self.lr)
            if epoch == 10000:
                self.lr = self.lr/10
                self.model_optimizer = Adam(learning_rate=self.lr)
                
            self.env.reset()
            behavior_names = self.env.get_behavior_names()
            DecisionSteps, TerminalSteps = self.env.get_steps(behavior_names[0])
            self.env_index = int(DecisionSteps.obs[-1][0][-1])
            self.dis_env = np.load('stage2_envs/env_{}.npy'.format(self.env_index+1))  
            cur_state = DecisionSteps.obs
            self.sensor_obs = collect_sen_obs()
            obs, actions, log_probs, rewards, v_preds, next_v_preds = [], [], [], [], [], []
            self.origin_connect_point = cal_connect_point(cur_state, self.dis_env)
            done, steps = False, 0
            robot_locs, vector_obses=[], []
            while not done and steps < max_steps:
                action, value, log_prob = self.act(cur_state)  
                self.env.set_actions(behavior_names[0], np.expand_dims(action,axis=0))
                self.env.step()
                DecisionSteps, TerminalSteps = self.env.get_steps(behavior_names[0])
                reward = 0
                if len(TerminalSteps.obs[-1]) != 0:
                    done = 1
                    next_state = TerminalSteps.obs
                    reward += TerminalSteps.reward[0]
                else:
                    next_state = DecisionSteps.obs
                    reward += DecisionSteps.reward[0]
                    
                vector_obs = next_state[-1]
                robot_loc = [-next_state[-1][0][5], 0.5, next_state[-1][0][3]] 
    
                robot_locs.append(robot_loc)     
                vector_obses.append(vector_obs)
                target_loc = [-next_state[-1][0][2], 0.75, next_state[-1][0][0]] 
                #if abs(robot_loc[0] - target_loc[0])+abs(robot_loc[-1] - target_loc[-1]) <=1.5:
                #    done = 1
                true_angle = self.cal_angle2(cur_state, self.dis_env)
                reward += -np.sqrt((true_angle[0]-action[0])**2+(true_angle[1]-action[1])**2)+1
                # self.env.render(mode='rgb_array')

                rewards.append(reward)
                v_preds.append(value)
                obs.append(cur_state)
                actions.append(action)
                log_probs.append(log_prob)

                steps += 1
                cur_state = next_state
            
            np.save('record/robot_loc_{}_{}'.format(epoch,steps), robot_locs)
            np.save('record/vectorobs_{}_{}'.format(epoch,steps), vector_obses)
            np.save('record/action_{}_{}'.format(epoch,steps), actions)
            np.save('record/reward_{}_{}'.format(epoch,steps), rewards)
            

            next_v_preds = v_preds[1:] + [0]
            gaes = self.get_gaes(rewards, v_preds, next_v_preds)
            gaes = np.array(gaes).astype(dtype=np.float32)
            #gaes = (gaes - np.mean(gaes)) / np.std(gaes)
            data = [obs, actions, log_probs, next_v_preds, rewards, gaes]
            
            self.n_updates = steps//self.batch_size + 1 
            for i in range(self.n_updates):
                # Sample training data
                sample_indices = np.random.randint(low=0, high=len(rewards), size=self.batch_size)
                sampled_data1 = []
                for j in sample_indices:
                    sampled_data1.append(data[0][j])  
                sampled_data2 = [np.take(a=a, indices=sample_indices, axis=0) for a in data[1:]]
                #sampled_data = sampled_data1 + sampled_data2
                # Train model
                self.learn(sampled_data1, *sampled_data2)

                # Tensorboard update
                with summary_writer.as_default():
                    tf.summary.scalar('Loss/total_loss', self.summaries['total_loss'], step=epoch)
                    tf.summary.scalar('Loss/clipped_surr', self.summaries['surr_loss'], step=epoch)
                    tf.summary.scalar('Loss/vf_loss', self.summaries['vf_loss'], step=epoch)
                    tf.summary.scalar('Loss/entropy', self.summaries['entropy'], step=epoch)

                summary_writer.flush()
                epoch += 1

            episode += 1
            print("episode {}: {} total reward, {} steps, {} epochs".format(
                episode, np.sum(rewards), steps, epoch))

            # Tensorboard update
            with summary_writer.as_default():
                tf.summary.scalar('Main/episode_reward', np.sum(rewards), step=episode)
                tf.summary.scalar('Main/episode_steps', steps, step=episode)

            summary_writer.flush()

            if steps >= max_steps:
                print("episode {}, reached max steps".format(episode))
                #self.save_model("ppo_episode{}.h5".format(episode))

            if episode % save_freq == 0:
                self.save_model("ppo_episode{}.h5".format(episode))

        self.save_model("ppo_final_episode{}.h5".format(episode))
    
    def test(self, map_path = 'env_', if_normal=True):
        max_steps = self.max_steps
        #self.env.reset()
        behavior_names = self.env.get_behavior_names()
        DecisionSteps, TerminalSteps = self.env.get_steps(behavior_names[0])
        self.env_index = int(DecisionSteps.obs[-1][0][-1])
        self.dis_env = np.load('res1101/' + map_path + '{}.npy'.format(self.env_index+1)) 
        #self.dis_env = np.load('stage2_envs/env_{}.npy'.format(self.env_index+1)) 
        cur_state = DecisionSteps.obs
        actions, robot_locs = [], []
        robot_loc = [-cur_state[-1][0][5], 0.5, cur_state[-1][0][3]] 
        target_loc = [-cur_state[-1][0][2], 0.5, cur_state[-1][0][0]] 
        self.origin_connect_point = cal_connect_point(cur_state, self.dis_env)
        
        able_move = False
        able_see = in_view(self.dis_env, robot_loc, target_loc)
        if able_see:
            able_move = True
        r2t_path = AStarSearch(self.dis_env, (round(robot_loc[0]), round(robot_loc[-1])), 
                                   (round(target_loc[0]), round(target_loc[-1])))    
        if len(r2t_path) <= 34:
            able_move = True 
        done, steps = False, 0
        jj=1
        last_loc = [0,0.5,0]
        while not done and steps < max_steps:
            move_loc = self.origin_connect_point[jj]
            robot_loc = [-cur_state[-1][0][5], 0.5, cur_state[-1][0][3]] 
            robot2target = np.sqrt((robot_loc[0]-move_loc[0])**2+
                                   (robot_loc[-1]-move_loc[-1])**2)
            if robot2target <= 1:
                if jj < len(self.origin_connect_point)-1:
                    jj += 1
                else:
                    pass
            robot2target_angle = (move_loc[0]-robot_loc[0],
                                  move_loc[-1]-robot_loc[-1])
            z1 = np.true_divide(robot2target_angle[0], np.sqrt(robot2target_angle[0]**2 + robot2target_angle[1]**2) + ZERO_TOLERANCE) + np.random.random()*0.22
            z2 = np.true_divide(robot2target_angle[1], np.sqrt(robot2target_angle[0]**2 + robot2target_angle[1]**2) + ZERO_TOLERANCE) + np.random.random()*0.22
            able_see = in_view(self.dis_env, robot_loc, target_loc)
            if able_see:
                able_move = True
            
            if able_move:    
                action = [z1, z2]
            else:
                action = [np.random.random()*z1, np.random.random()*z2]
            if hit_wall(self.dis_env, robot_loc, action):
                action = [np.random.random()-z1, np.random.random()-z2]
            #if np.sum(np.subtract(last_loc, robot_loc)) <0.2:
            #    action = [np.random.random()-z1, np.random.random()-z2]
            if if_normal:
                z3 = np.true_divide(action[0], np.sqrt(action[0]**2 + action[1]**2) + ZERO_TOLERANCE)
                z4 = np.true_divide(action[1], np.sqrt(action[0]**2 + action[1]**2) + ZERO_TOLERANCE)
            action = [z3,z4]

            self.env.set_actions(behavior_names[0], np.expand_dims(action,axis=0))
            self.env.step()
            DecisionSteps, TerminalSteps = self.env.get_steps(behavior_names[0])
            reward = DecisionSteps.reward[0]
            if len(TerminalSteps.obs[-1]) != 0:
                done = 1
                next_state = TerminalSteps.obs
                reward += TerminalSteps.reward[0]
            else:
                next_state = DecisionSteps.obs
            steps += 1
            cur_state = next_state
            last_loc = robot_loc
            actions.append(action)
            robot_locs.append(robot_loc)
        if done  == 1:
            pass
        else:
            self.env.reset()
        return actions, robot_locs, target_loc, done
    
