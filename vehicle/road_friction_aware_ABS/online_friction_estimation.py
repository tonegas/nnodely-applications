'''
  ____   ___    _    ____      _____ ____  ___ ____ _____ ___ ___  _   _     _____ ____ _____ ___ __  __    _  _____ ___ ___  _   _ 
 |  _ \ / _ \  / \  |  _ \    |  ___|  _ \|_ _/ ___|_   _|_ _/ _ \| \ | |   | ____/ ___|_   _|_ _|  \/  |  / \|_   _|_ _/ _ \| \ | |
 | |_) | | | |/ _ \ | | | |   | |_  | |_) || | |     | |  | | | | |  \| |   |  _| \___ \ | |  | || |\/| | / _ \ | |  | | | | |  \| |
 |  _ <| |_| / ___ \| |_| |   |  _| |  _ < | | |___  | |  | | |_| | |\  |   | |___ ___) || |  | || |  | |/ ___ \| |  | | |_| | |\  |
 |_| \_\\___/_/   \_\____/    |_|   |_| \_\___\____| |_| |___\___/|_| \_|   |_____|____/ |_| |___|_|  |_/_/   \_\_| |___\___/|_| \_|
                                                                                                                                    
This script performs road friction estimation through robust model selection, using the methodology described in the paper titled:
"A Road Friction-Aware Anti-Lock Braking System Based on Model-Structured Neural Networks"

27/01/2025 -- Mattia Piccinini, Matteo Zumerle, Johannes Betz, Gastone Pietro Rosati Papini
'''

# ----------------------------------------------------------
# initial setup
# ----------------------------------------------------------

# import the necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=18)

# import the nnodely library
from nnodely import *

# set a random seed for reproducibility
random_seed = 10

# ----------------------------------------------------------
# load the trained models
# ----------------------------------------------------------

# directory in which the trained neural models are saved
trained_models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'trained_models')

# create the models with the nnodely library
MS_NN_dry_road  = nnodely(visualizer=MPLVisualizer(), seed=random_seed, workspace=os.path.join(trained_models_dir, 'dry_road'))
MS_NN_wet_road  = nnodely(visualizer=MPLVisualizer(), seed=random_seed, workspace=os.path.join(trained_models_dir, 'wet_road'))
MS_NN_snow_road = nnodely(visualizer=MPLVisualizer(), seed=random_seed, workspace=os.path.join(trained_models_dir, 'snow_road'))
MS_NN_ice_road  = nnodely(visualizer=MPLVisualizer(), seed=random_seed, workspace=os.path.join(trained_models_dir, 'ice_road'))

# load the parameters of the trained model from the corresponding JSON files
MS_NN_dry_road.loadModel()
MS_NN_wet_road.loadModel()
MS_NN_snow_road.loadModel()
MS_NN_ice_road.loadModel()

# hyperparameters:
# sampling time for the input windows (this sampling time can be different from the one of the dataset: an interpolation will be automatically performed)
sample_time = 0.05  # [s]
# number of samples in the input windows (for the past engine torque/brake pedal values, up to current time step)
n = 10   

# neuralize the models
MS_NN_dry_road.neuralizeModel(sample_time)
MS_NN_wet_road.neuralizeModel(sample_time)
MS_NN_snow_road.neuralizeModel(sample_time)
MS_NN_ice_road.neuralizeModel(sample_time)

# ----------------------------------------------------------
# load the test dataset
# ----------------------------------------------------------

# folder containing the test dataset
test_set_folder_name = 'data_test'
test_set_folder      = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'datasets', test_set_folder_name)

# structure of the data in the dataset
data_struct = ['time','brake_pedal','','speed','acc','','','','','motor_torque']
MS_NN_dry_road.loadData(name=test_set_folder_name,  source=test_set_folder, format=data_struct, skiplines=1)
MS_NN_wet_road.loadData(name=test_set_folder_name,  source=test_set_folder, format=data_struct, skiplines=1)
MS_NN_snow_road.loadData(name=test_set_folder_name, source=test_set_folder, format=data_struct, skiplines=1)
MS_NN_ice_road.loadData(name=test_set_folder_name,  source=test_set_folder, format=data_struct, skiplines=1)

# find all the csv files in the root
csv_files_path = os.path.join(test_set_folder, '*.csv')
csv_files = glob.glob(csv_files_path)

# store the recorded longitudinal acceleration values
acc_lengths = {}
acc_value = []
for file in csv_files:
  df = pd.read_csv(file)
  if 'acc' in df.columns:
    length = df['acc'].count()
    acc_lengths[file] = length
    acc_value.extend(df['acc'].dropna().tolist())
  else:
    acc_lengths[file] = None
    print(f"Column 'acc'not found in file: {file}")

# find the number of samples in the dataset
n_file = 0
num_el = 0
for file, length in acc_lengths.items():
  n_file = n_file+1
  num_el = num_el + acc_lengths[file]
num_el = num_el - n_file*(n - 1)
# time vector
time = np.arange(0, num_el * sample_time, sample_time)[:num_el] 

# ----------------------------------------------------------
# run the models on the test dataset
# ----------------------------------------------------------

# dry road
sample_test_set_dry = MS_NN_dry_road.getSamples(test_set_folder_name, index=0, window=num_el)
out_nn_test_set_dry = MS_NN_dry_road(sample_test_set_dry, sampled=True)
out_nn_test_set_extract_dry = out_nn_test_set_dry['acceleration']
samples_test_set_extract_dry = np.zeros((len(sample_test_set_dry['acc']),1))
for i in range(0,len(samples_test_set_extract_dry)):
  samples_test_set_extract_dry[i] = sample_test_set_dry['acc'][i]

# wet road
sample_test_set_wet = MS_NN_wet_road.getSamples(test_set_folder_name, index=0, window=num_el)
out_nn_test_set_wet = MS_NN_wet_road(sample_test_set_wet, sampled=True)
out_nn_test_set_extract_wet = out_nn_test_set_wet['acceleration']
samples_test_set_extract_wet = np.zeros((len(sample_test_set_wet['acc']),1))
for i in range(0,len(samples_test_set_extract_wet)):
  samples_test_set_extract_wet[i] = sample_test_set_wet['acc'][i]

# snowy road
sample_test_set_snow = MS_NN_snow_road.getSamples(test_set_folder_name, index=0, window=num_el)
out_nn_test_set_snow = MS_NN_snow_road(sample_test_set_snow, sampled=True)
out_nn_test_set_extract_snow = out_nn_test_set_snow['acceleration']
samples_test_set_extract_snow = np.zeros((len(sample_test_set_snow['acc']),1))
for i in range(0,len(samples_test_set_extract_snow)):
  samples_test_set_extract_snow[i] = sample_test_set_snow['acc'][i]

# icy road
sample_test_set_ice = MS_NN_ice_road.getSamples(test_set_folder_name, index=0, window=num_el)
out_nn_test_set_ice = MS_NN_ice_road(sample_test_set_ice, sampled=True)
out_nn_test_set_extract_ice = out_nn_test_set_ice['acceleration']
samples_test_set_extract_ice = np.zeros((len(sample_test_set_ice['acc']),1))
for i in range(0,len(samples_test_set_extract_ice)):
  samples_test_set_extract_ice[i] = sample_test_set_ice['acc'][i]

# ----------------------------------------------------------
# perform robust model selection for road friction estimation
# ----------------------------------------------------------

# number of channels (i.e. MS-NN models trained on different road conditions)
n_channels = 4  # channel 1-> dry, channel 2-> wet, channel 3-> snow, channel 3-> ice

# compute the predicted acceleration values for each channel
predictions       = np.zeros((n_channels, num_el))
predictions[0, :] = out_nn_test_set_extract_dry
predictions[1, :] = out_nn_test_set_extract_wet
predictions[2, :] = out_nn_test_set_extract_snow
predictions[3, :] = out_nn_test_set_extract_ice

# custom function to compute the evidence of the predictions
def evidence_fun(predict_error, sigma_evidence):
  # gaussian evidence function, where sigma_evidence is a tunable parameter
  return np.exp(-predict_error**2 / 2 * sigma_evidence**2)

# -------------------------------
# Multi-Hypothesis Sequential Probability Ratio Test (MSPRT) 
# algorithm for model selection
# -------------------------------

# MSPRT parameters
max_window_size   = 10      # window size used by the MSPRT algorithm to compute the salience and likelihood of each channel
thresh_likelihood = 0.2545  # threshold for the likelihood of each channel to select the best channel
sigma_evidence    = 0.1758  # parameter for the evidence function

# initialize the variables
past_evidence         = np.zeros((n_channels,max_window_size))  # evidence of each channel, from a past window
salience              = np.zeros(n_channels)                    # salience of each channel
salience_store        = np.zeros((n_channels, num_el))          # salience of all records of each channel
likelihood            = np.zeros(n_channels)                    # likelihood of each channel
likelihood_store      = np.zeros((n_channels, num_el))          # store the likelihood of each channel
best_channel_msprt    = np.zeros(num_el)                        # index of the best channel
best_channel_msprt[0] = 0                                       # initialize the index of the best channel
window                = 1                                       # initialize the window length

# run the MSPRT algorithm
for i in range(num_el):
  for chan in range(n_channels):
    predict_error = abs(acc_value[i] - predictions[chan, i])
    evidence = evidence_fun(predict_error, sigma_evidence)  
    # store the evidence in a window, whose max size is max_window_size
    past_evidence[chan, np.mod(window-1, max_window_size)] = evidence
    # compute the salience of each channel
    salience[chan] = np.mean(past_evidence[chan, :window])
  # compute the likelihood of each channel
  likelihood = np.exp(salience - np.log(np.sum(np.exp(salience))))
  likelihood_store[:, i] = likelihood
  if np.max(likelihood) >= thresh_likelihood:
    # select the channel with the highest likelihood
    best_channel_msprt[i] = np.argmax(likelihood)
    # reset the past evidence
    past_evidence = np.zeros((n_channels, max_window_size))
    # reset the window
    window = 1
  else:
    # keep the previous best channel
    best_channel_msprt[i] = best_channel_msprt[i-1]
    # increase the window
    window = np.min([max_window_size, window + 1])

# -------------------------------
# Winner Takes All (WTA) algorithm for model selection
# -------------------------------

# initialize the variables
best_channel_wta    = np.zeros(num_el)   # index of the best channel
best_channel_wta[0] = 0                  # initialize the index of the best channel
predict_error       = 0                  # prediction error
salience_store      = np.zeros((n_channels, num_el))  # salience of all records of each channel
for i in range(num_el):
  for chan in range(n_channels):
    predict_error = abs(acc_value[i] - predictions[chan, i])
    evidence = evidence_fun(predict_error, sigma_evidence)  
    salience_store[chan, i] = evidence
    # select the channel with the highest likelihood
    best_channel_wta[i] = np.argmax(salience_store[:, i])

# ---------------------------------------------------
# plots
# ---------------------------------------------------

# load the dataset containing the (ground truth) grip information 
file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'datasets', 'data_grip', 'grip_condition_variation.csv')
abs_file = os.path.abspath(file)
df = pd.read_csv(abs_file)
grip_time  = df.columns[0]   # time values
grip_value = df.columns[1]   # friction coefficient values

# plot the acceleration signals and the best channel determined by the MSPRT
plt.figure(figsize=(14, 8))

# plot the best channel
plt.subplot(3, 1, 1)
plt.ylabel(r'Friction coeff [-]')
plt.grid()
plt.gca().yaxis.set_major_locator(plt.MaxNLocator(integer=True))
plt.xlim(0, 100)
plt.ylim(0, 1)
plt.axhline(y=1, color='#1f77b4', linestyle='--', linewidth=1.5, label=r'Dry')  
plt.axhline(y=0.75, color='#ff7f0e', linestyle='--',linewidth=1.5,label=r'Wet')  
plt.axhline(y=0.5, color='#2ca02c', linestyle='--', linewidth=1.5,label=r'Snow')  
plt.axhline(y=0.25, color='#d62728', linestyle='--', linewidth=1.5,label=r'Ice') 
plt.plot(df[grip_time], df[grip_value],  color='black', linewidth=3, label=r'Friction coeff')
plt.legend(fontsize=11, loc='best', ncol=2, facecolor='white', framealpha=1.0)
yticks_values = np.arange(0, 1.4, 0.2) 
plt.yticks(yticks_values)
plt.tick_params(axis='x', labelbottom=False)

plt.subplot(3, 1, 2) 
road_strings = ['Dry', 'Wet', 'Snow', 'Ice']
plt.plot(time, acc_value[:num_el], linewidth=2, color='black', label=r'$a_x$ meas.')
for i in range(n_channels):
  plt.plot(time, predictions[i], linewidth=3, label=r'model ' + str(i+1) + ' (' + road_strings[i] + ')')
# plt.xlabel(r'Time [s]')
plt.ylabel(r'$a_x$ [m/s$^2$]')
plt.legend(fontsize=11, loc='lower left', ncol=2, facecolor='white', framealpha=1.0)
plt.grid()
plt.xlim(0, 100)
plt.ylim(-13, 3)
yticks_values = np.arange(-13, 3, 3)  
plt.yticks(yticks_values)
plt.tick_params(axis='x', labelbottom=False) 

# plot the best channel
plt.subplot(3, 1, 3)
plt.plot(time, best_channel_wta+1, ':', linewidth=0.5,  color='red', label=r'WTA')
plt.plot(time, best_channel_msprt+1, linewidth=3, color='black', label=r'MSPRT')
plt.xlabel(r'Time [s]')
plt.ylabel(r'Index selected model $i$')
plt.legend(fontsize=11, facecolor='white', framealpha=1.0, markerscale = 5)
plt.grid()
plt.gca().yaxis.set_major_locator(plt.MaxNLocator(integer=True))
plt.xlim(0, 100)

plt.show()