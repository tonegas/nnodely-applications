from __future__ import annotations
import random
from tqdm import tqdm
import numpy as np
import os
import pandas as pd
import gymnasium as gym

def norm_angle(angle):
    """
    Normalize the angle.
    """
    #return angle - (2 * np.pi) * np.floor(angle / (2 * np.pi)) # [0, 2*pi] without using the modulo operator
    return angle
    #return ((angle + np.pi) % (2 * np.pi)) - np.pi # to the range [-pi, pi]

def random_smooth(x, action_space_dims, delta=0.5):
    min = x-delta if x-delta >= -3 else -3
    max = x+delta if x+delta <= 3 else 3
    return np.random.uniform(min, max, action_space_dims)

def save_one_simulation(epochs, dt, action_freq, zero_action_prob, action_mode, agent, env, i):
    df = pd.DataFrame(columns=['time', 'action', 'Xpos', 'Xth1', 'Xth2', 'Xvelocity', 'Xth1_dot', 'Xth2_dot', 'Xddx', 'Xddth1', 'Xddth2'])
    obs, _ = env.reset()
    env.unwrapped.skip_frame = 1
    env.unwrapped.frame_skip = 1
    dt = 0.01
    action = 0
    cont = 0
    for t in range(epochs):
        if t % action_freq == 0:
            if zero_action_prob > 0:
                action = agent.sample_action(obs) if random.random() > zero_action_prob else np.zeros(action_space_dims)
            elif action_mode == 'random':
                action = np.random.uniform(-3, 3, action_space_dims)
            elif action_mode == 'smooth':
                action = random_smooth(action, action_space_dims, delta=0.3)
            elif action_mode == 'minmax':
                action = np.random.uniform(2.5, 3, action_space_dims) if t%2==0 else np.random.uniform(-3, -2.5, action_space_dims)
            elif action_mode == 'continuous':
                if t % 10 == 0:
                    action = np.random.uniform(-3, 3, action_space_dims)
            else:
                action = np.zeros(action_space_dims)
        else:
            action = np.zeros(action_space_dims)
            
        old_obs = obs
        qpose = env.unwrapped.data.qpos
        qvel = env.unwrapped.data.qvel
        obs, _, _, _, _ = env.step(action)
        qacc = env.unwrapped.data.qacc
        
        df.loc[t] = [
            round(t*dt,4), round(float(action[0]),6), 
            round(qpose[0],6), round(qpose[1],6), round(qpose[2],6), 
            round(qvel[0],6), round(qvel[1],6), round(qvel[2],6), 
            round(qacc[0],6), round(qacc[1],6), round(qacc[2],6)
            ]
        
        """if abs(obs[1]) >= np.pi/2:
            cont += 1
        if cont > 10:
            break"""
    df.to_csv(f'data/data_DP/data_{env_name}_{i}.csv', index=False)

# Parameters
num_sim = 500  # number of simulation
T       = 3   # simulation time (s)
dt      = 0.01 # sampling time (s)
env_name = 'InvertedDoublePendulum-v5'
zero_action_prob = 0.0
action_freq = 1 # action frequency (in number of steps)
epochs = int(T/dt)

env = gym.make(env_name, frame_skip = 1, disable_env_checker=True)
env.unwrapped.skip_frame = 1
env.unwrapped.frame_skip = 1
print(f"Environment dt: {env.unwrapped.dt}")

# Observation space of InvertedDoublePendulum-v5 (4)
obs_space_dims = env.observation_space.shape[0]
# Action space of InvertedDoublePendulum-v5 (1)
action_space_dims = env.action_space.shape[0]

for i in tqdm(range(num_sim)):
    if i < 600:
        save_one_simulation(epochs, dt, action_freq+(i%80+1), zero_action_prob, 'random', None, env, i)
    
    else:
        save_one_simulation(epochs, dt, action_freq, zero_action_prob, 'zero', None, env, i)
env.close()