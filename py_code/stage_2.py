# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 14:57:31 2020

@author: azrael
"""
import numpy as np
import matplotlib.pyplot as plt
from mlagents_envs.environment import UnityEnvironment
from tf_ppo import PPO


# spdyer first, then unity play
print('load_enviroment for stage-2, prepare to hit play button!')
env = UnityEnvironment(file_name=None, side_channels=[])
# Start interacting with the evironment.
env.reset()
#behavior_names = env.get_behavior_names()
#behavior_specs = env.get_behavior_spec(behavior_names[0])
#(DecisionSteps, TerminalSteps) = env.get_steps(behavior_names[0])
#cur_state = DecisionSteps.obs
#cur_obs = form_obs(cur_state)

#env.set_actions(behavior_names[0], np.expand_dims(action,axis=0))
#env.step()
#(DecisionSteps2, TerminalSteps2) = env.get_steps(behavior_names[0])

agent = PPO(env, max_steps=150)
agent.env = env
#agent.policy.load_weights('gnn_1002_att_layer3.h5', by_name=True, skip_mismatch=True)
#agent.policy.load_weights('ppo_episode500.h5')
agent.train(max_epochs=100000, save_freq=200)
env.close()


    
