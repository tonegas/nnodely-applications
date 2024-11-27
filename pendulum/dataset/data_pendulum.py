# Pendulum system
# Alessandro Antonucci @AlexRookie
# University of Trento

# 1 DOF: angle theta
# Input: torque T

# Alessandro Antonucci @AlexRookie 
# University of Trento

# Import
import gym
from gym.wrappers import Monitor
import gym_neu4mes

import os
import numpy as np
import cv2 as cv2
import time

#import matplotlib.pyplot as plt

# Check available environments
#from gym import envs
#print(envs.registry.all())

#=================================================================================================#

# Parameters

num_sim = 100  # number of simulation
T       = 20   # simulation time (s)
dt      = 0.05 # sampling time (s) @20Hz

model = 'gym_neu4mes:Pendulum-v2'

mass    = 1.0  # pendulum mass (kg)
length  = 1.0  # pendulum lenght (m)
gravity = 9.81 # gravitational acceleration

theta0 = [-np.pi, np.pi] # initial x1
omega0 = [-1, 1]         # initial x'1

torque = [-2.0, 2.0] # torque (N°m)
u_type = 1 # input type: pulse

out_folder = './data/data-pendulum-a/';

# Options
options_show       = False
options_save_data  = True
options_save_video = False

#=================================================================================================#

# Create folder
if not os.path.exists(out_folder):
    os.makedirs(out_folder)

#env = gym.make('Pendulum-v1')
#
#env.reset()
#for _ in range(1000):
#    env.render()
#    observation, reward, done, info = env.step(env.action_space.sample()) # take a random action
#    theta = ( np.arctan2(observation[1], observation[0])*180/np.pi + 360 ) % 360
#env.close()

# Number of samples per simulation
epochs = int(T/dt)

# Open parameter file
if options_save_data:
    paramfile = open(out_folder+'params.txt', 'w')

samples = 0
tic = time.time()

for i_episode in range(num_sim):
    # Open data file
    if options_save_data:
        simfile = open(out_folder+str(i_episode+1)+'.txt', 'w')
    
    # Get variable parameters

    #mass_i = mass[0] + (mass[1]-mass[0])*np.random.rand(1)[0] # 1.0 
    #length_i = length[0] + (length[1]-length[0])*np.random.rand(1)[0] # 0.1
    mass_i = mass
    length_i = length

    tu_start = round(0 + (epochs/4 - 0)*np.random.rand(1)[0])
    tu_end = round(0 + (epochs/4 - 0)*np.random.rand(1)[0])
    if tu_start > tu_end:
        tu_start, tu_end = tu_end, tu_start

    # Load the environment
    env = gym.make(model, m=mass_i,
                          l=length_i,
                          dt=dt,
                          g=gravity,
                          torque=torque,
                          t0=theta0,
                          o0=omega0)

    if options_save_video and i_episode == num_sim-2:
        env = Monitor(env, out_folder+'video', force=True)
        #img = env.render(mode="rgb_array")
        #cv2.imwrite(out_folder+'plot.png', img)

    # Reset the environment (with random initial values)
    observation = env.reset()

    cos_theta0_i, sin_theta0_i, omega0_i = observation[0], observation[1], observation[2]
    theta0_i = ( np.arctan2(sin_theta0_i, cos_theta0_i) + 2*np.pi ) % (2*np.pi)

    for t in range(0,epochs+1):
        # Render the environment at each step
        if options_show:
            img = env.render()

        #print("t: "+str(t)+ " observation: "+str(observation))

        # Take a random action
        if t >= tu_start and t <= tu_end:
            action = env.action_space.sample()[0]
        else:
            action = 0.0

        # Simulate step
        observation, reward, done, info = env.step(action)

        cos_theta, sin_theta, omega = observation[0], observation[1], observation[2]
        theta = ( np.arctan2(sin_theta, cos_theta) + 2*np.pi ) % (2*np.pi)

        # Save results
        if options_save_data:            
            simfile.write("{:.3f};{:f};{:f};{:f};{:f};{:f}\n".format(t*dt, theta, observation[2], observation[0], observation[1], action))
        
    print("Episode finished after {} timesteps".format(t+1))
    samples += t+1
            
    env.close()

    # Close data file
    if options_save_data:
        simfile.close()

    # Save params
    if options_save_data:
        paramfile.write("{:d};{:d};{:f};{:f};{:f};{:f};{:f};{:d};{:f};{:f}\n".format(i_episode+1, t+1, mass_i, length_i, gravity, theta0_i, omega0_i, u_type, tu_start*dt, tu_end*dt))

toc = time.time() - tic
print("Elapsed time: "+str(toc))

# Close params file
if options_save_data:
    paramfile.close()

# Save statistics
if options_save_data:
    statsfile = open(out_folder+'stats.txt', 'w')
    statsfile.write("simulations;{:d}\nelapsed_time;{:4.3f}\ntotal_samples;{:d}\nparam_time;{:4.3f}\nparam_sampling_time;{:4.3f}\n".format(num_sim, toc, samples, T, dt))
    statsfile.write("m;{:4.3f}\nl;{:4.3f}\ntheta_0;{:4.3f};{:4.3f}\nomega_0;{:4.3f};{:4.3f}\ntorque;{:4.3f};{:4.3f}".format(mass, length, theta0[0], theta0[1], omega0[0], omega0[1], torque[0], torque[1]))
    #statsfile.write("m_c;{:4.3f};{:4.3f}\nm_p;{:4.3f};{:4.3f}\nx_0;{:4.3f};{:4.3f}\nv_0;{:4.3f};{:4.3f}\ntheta_0;{:4.3f};{:4.3f}\nomega_0;{:4.3f};{:4.3f}".format(mass_cart[0], mass_cart[1], mass_pole[0], mass_pole[1], x0[0], x0[1], theta0[0], theta0[1], v0[0], v0[1], omega0[0], omega0[1]))
    statsfile.close()
