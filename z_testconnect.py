# -*- coding: utf-8 -*-
"""
Created on Fri May  7 11:06:32 2021

@author: Win10
"""

import numpy as np
import matplotlib.pyplot as plt
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.base_env import ActionTuple


# spdyer first, then unity play
print('load_enviroment, prepare to hit play button!')
env = UnityEnvironment(file_name=None, side_channels=[])
# Start interacting with the evironment.
env.reset()

behavior_name = list(env.behavior_specs)[0] 
spec = env.behavior_specs[behavior_name]

#vis_obs = any(len(spec.shape) == 3 for spec in spec.observation_specs)
decision_steps, terminal_steps = env.get_steps(behavior_name)
action = ActionTuple(np.array([[1.0,1.0]], dtype=np.float32))

action = spec.action_spec.random_action(len(decision_steps))
env.set_actions(behavior_name, action)

env.step()

decision_steps2, terminal_steps2 = env.get_steps(behavior_name)
env.close()
