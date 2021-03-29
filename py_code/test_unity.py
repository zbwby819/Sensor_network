# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 14:57:31 2020

@author: azrael
"""
import numpy as np
import matplotlib.pyplot as plt
from mlagents_envs.environment import UnityEnvironment

# This is a non-blocking call that only loads the environment.
env = UnityEnvironment(file_name='PushBlock', worker_id=0, seed=1, side_channels=[])
env = UnityEnvironment()

#env = UnityEnvironment(file_name="TutoBananaCollector.x86_64", worker_id=1)

brain_name = env.brain_names[0]
brain = env.brains[brain_name]

# reset the environment
env_info = env.reset(train_mode=True)[brain_name]

# number of agents in the environment
print("Number of agents:", len(env_info.agents))

# number of actions
action_size = brain.vector_action_space_size[0]
print("Number of actions:", action_size)

# examine the state space
state = env_info.vector_observations
print("States look like:", state)

state_size = len(state)
print("States have length:", state_size)


env_info = env.reset(train_mode=False)[brain_name]
state = env_info.vector_observations[0]
score = 0
for idx in range(100):
    action = np.random.randint(action_size)
    env_info = env.step(action)[brain_name]
    next_state = env_info.vector_observations[0]
    reward = env_info.rewards[0]
    done = env_info.local_done[0]
    score += reward
    state = next_state
    if done:
        break

print(f"Score {score}")
env.close()