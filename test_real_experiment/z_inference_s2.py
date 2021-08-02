# -*- coding: utf-8 -*-
"""
Created on Tue May 11 10:52:55 2021

@author: Win10
"""

import numpy as np
import matplotlib.pyplot as plt
from mlagents_envs.environment import UnityEnvironment
from tf_ppo import PPO, form_obs


# spdyer first, then unity play
print('load_enviroment for stage-2, prepare to hit play button!')
env = UnityEnvironment(file_name=None, side_channels=[])
# Start interacting with the evironment.
env.reset()
#behavior_name = list(env.behavior_specs)[0] 
#spec = env.behavior_specs[behavior_name]
#decision_steps, terminal_steps = env.get_steps(behavior_name)
#cur_state = decision_steps.obs
#cur_obs = form_obs(cur_state)

#action = ActionTuple(np.array([[1.0,1.0]], dtype=np.float32))
#env.set_actions(behavior_name, action)
#env.step()

agent = PPO(env, max_steps=150)
agent.env = env
agent.test()

env.close()