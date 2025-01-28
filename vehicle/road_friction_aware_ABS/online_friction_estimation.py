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

import csv
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=18)

import os
import glob

# import the nnodely library
from nnodely import *

# Create nnodely structure
vehicle_dry  = nnodely(visualizer=MPLVisualizer(), seed=10)
vehicle_wet  = nnodely(visualizer=MPLVisualizer(), seed=10)
vehicle_snow = nnodely(visualizer=MPLVisualizer(), seed=10)
vehicle_ice  = nnodely(visualizer=MPLVisualizer(), seed=10)

# Find the root where the model parameters are saved
current_dir = os.path.dirname(os.path.abspath(__file__))
json_dir = os.path.join(current_dir, "JSON")

# Decide if take the NN models trained without (0) or with (1) noise

json_dry = os.path.join(json_dir, "model_parameters_dry_noise_MS.json")
json_wet = os.path.join(json_dir, "model_parameters_wet_noise_MS.json")
json_snow = os.path.join(json_dir, "model_parameters_snow_noise_MS.json")
json_ice = os.path.join(json_dir, "model_parameters_ice_noise_MS.json")

# Dataset selection

name_data = 'test_noise'
file_csv = 'test_noise.csv'
csv_folder_name = 'data_test'

# Print dataset selection
print("It has been selected all json files (dry/wet/snow/ice) generating data in [", name_data, "] conditions")

# Acquire the sample rate value
file = os.path.join('./nn_abs/datasets/', csv_folder_name, file_csv)
abs_file = os.path.abspath(file)
with open(abs_file, mode='r') as file:
    lettore_csv = csv.reader(file)
    next(lettore_csv)  # Skip header
    first = next(lettore_csv)
    second = next(lettore_csv)
    sample_ratio = float(second[0]) - float(first[0])
print(f"The sample rate of the provided datasets is: {sample_ratio}")

# ______________ SECTION 1: LOAD THE TRAINED PARAMETERS ______________ #

# Load the model parameters from the JSON file
with open(json_dry, 'r') as infile:
    model_parameters_dry = json.load(infile)
with open(json_wet, 'r') as infile:
    model_parameters_wet = json.load(infile)
with open(json_snow, 'r') as infile:
    model_parameters_snow = json.load(infile)
with open(json_ice, 'r') as infile:
    model_parameters_ice = json.load(infile)

# Assign the loaded values from JSON to a model
vehicle_dry.model_def = model_parameters_dry
vehicle_wet.model_def = model_parameters_wet
vehicle_snow.model_def = model_parameters_snow
vehicle_ice.model_def = model_parameters_ice

# Visualize the model and check that all is correctly loaded
vehicle_dry.neuralizeModel()
vehicle_wet.neuralizeModel()
vehicle_snow.neuralizeModel()
vehicle_ice.neuralizeModel()

# ______________ SECTION 2: LOAD THE DATASET ______________ #

data_struct = ['time', 'brake_pedal', 'gas_pedal', 'speed', 'acc', 'in_tw_rr_brake', 'in_tw_rl_brake', 'in_tw_fr_brake', 'in_tw_fl_brake', 'wheel_torque_rear', 'f_rot_speed', 'r_rot_speed', 'ABS_FRONT_ON_OFF', 'ABS_REAR_ON_OFF']
data_folder = os.path.join('./nn_abs/datasets/', csv_folder_name)
vehicle_dry.loadData(name=csv_folder_name, source=data_folder, format=data_struct, skiplines=1)
vehicle_wet.loadData(name=csv_folder_name, source=data_folder, format=data_struct, skiplines=1)
vehicle_snow.loadData(name=csv_folder_name, source=data_folder, format=data_struct, skiplines=1)
vehicle_ice.loadData(name=csv_folder_name, source=data_folder, format=data_struct, skiplines=1)

# Find all the csv files in the root
csv_files_path = os.path.join('./nn_abs/datasets/', csv_folder_name, '*.csv')
csv_files = glob.glob(csv_files_path)

# Save what concerns with 'acc' values
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

n_file = 0
num_el = 0
initial_num_el = 0

# Define the sample window
sw = 10

# Select the correct data
for file, length in acc_lengths.items():
    print(f"File: {file}, lenght 'acc': {length}")
    n_file = n_file+1
    num_el = num_el + acc_lengths[file]

initial_num_el = num_el
num_el = num_el -n_file*(sw - 1)
print("Number of file in ",csv_folder_name, "are: ", n_file)
print("The correspondig total values usable as unique dataset are: ", num_el)
time = np.arange(0, num_el * sample_ratio, sample_ratio)[:num_el] 
print(time)


# ______________ SECTION 3: RUN THE TRAINED MODEL WITH THE LOADED DATASET ______________ #

#Dry
sample_test_set_dry = vehicle_dry.getSamples(csv_folder_name, index=0, window=num_el)
out_nn_test_set_dry = vehicle_dry(sample_test_set_dry, sampled=True)
out_nn_test_set_extract_dry = out_nn_test_set_dry['acceleration']
# extract the samples
samples_test_set_extract_dry = np.zeros((len(sample_test_set_dry['acc']),1))
for i in range(0,len(samples_test_set_extract_dry)):
  samples_test_set_extract_dry[i] = sample_test_set_dry['acc'][i]

#Wet
sample_test_set_wet = vehicle_wet.getSamples(csv_folder_name, index=0, window=num_el)
out_nn_test_set_wet = vehicle_wet(sample_test_set_wet, sampled=True)
out_nn_test_set_extract_wet = out_nn_test_set_wet['acceleration']
# extract the samples
samples_test_set_extract_wet = np.zeros((len(sample_test_set_wet['acc']),1))
for i in range(0,len(samples_test_set_extract_wet)):
  samples_test_set_extract_wet[i] = sample_test_set_wet['acc'][i]

#Snow
sample_test_set_snow = vehicle_snow.getSamples(csv_folder_name, index=0, window=num_el)
out_nn_test_set_snow = vehicle_snow(sample_test_set_snow, sampled=True)
out_nn_test_set_extract_snow = out_nn_test_set_snow['acceleration']
# extract the samples
samples_test_set_extract_snow = np.zeros((len(sample_test_set_snow['acc']),1))
for i in range(0,len(samples_test_set_extract_snow)):
  samples_test_set_extract_snow[i] = sample_test_set_snow['acc'][i]

#Ice
sample_test_set_ice = vehicle_ice.getSamples(csv_folder_name, index=0, window=num_el)
out_nn_test_set_ice = vehicle_ice(sample_test_set_ice, sampled=True)
out_nn_test_set_extract_ice = out_nn_test_set_ice['acceleration']
# extract the samples
samples_test_set_extract_ice = np.zeros((len(sample_test_set_ice['acc']),1))
for i in range(0,len(samples_test_set_extract_ice)):
  samples_test_set_extract_ice[i] = sample_test_set_ice['acc'][i]


# plot the results
plt.figure(figsize=(10, 5))
plt.subplot(4, 1, 1)
plt.plot(time, samples_test_set_extract_dry,label='Target')
plt.plot(time, out_nn_test_set_extract_dry,label='NN dry trained',linestyle='--')
#plt.title('Dry trained NN')
plt.ylabel(r'acc [$m/s^2$]')
plt.legend(fontsize=11, facecolor='white', framealpha=1.0)
plt.grid()
plt.xticks(np.arange(0, max(time) + 1, 10))  
plt.xlim(0, 100)
plt.tick_params(axis='x', labelbottom=False) 

plt.subplot(4, 1, 2)
plt.plot(time,samples_test_set_extract_wet,label=r'Target')
plt.plot(time,out_nn_test_set_extract_wet,label=r'NN wet trained',linestyle='--')
#plt.title('Wet trained NN')
plt.ylabel(r'$a_x$ [$m/s^2$]')
plt.legend(fontsize=11, facecolor='white', framealpha=1.0)
plt.grid()
plt.xticks(np.arange(0, max(time) + 1, 10))  
plt.xlim(0, 100)
plt.tick_params(axis='x', labelbottom=False)  

plt.subplot(4, 1, 3)
plt.plot(time,samples_test_set_extract_snow,label=r'Target')
plt.plot(time,out_nn_test_set_extract_snow,label=r'NN snow trained',linestyle='--')
#plt.title('Snow trained NN')
plt.ylabel(r'$a_x$ [$m/s^2$]')
plt.legend(fontsize=11, facecolor='white', framealpha=1.0)
plt.grid()
plt.xticks(np.arange(0, max(time) + 1, 10))  
plt.xlim(0, 100)
plt.tick_params(axis='x', labelbottom=False)  

plt.subplot(4, 1, 4)
plt.plot(time,samples_test_set_extract_ice,label=r'Target')
plt.plot(time,out_nn_test_set_extract_ice,label=r'NN ice trained',linestyle='--')
#plt.title('Ice trained NN')
plt.xlabel(r'Time [$s$]')
plt.ylabel(r'$a_x$ [$m/s^2$]')
plt.legend(fontsize=11, facecolor='white', framealpha=1.0)
plt.grid()
plt.xticks(np.arange(0, max(time) +1, 10))  
plt.xlim(0, 100)

plt.show()

# COMPARISON DRY-ICE
plt.figure(figsize=(10, 5))
plt.subplot(2, 1, 1)
plt.plot(time, samples_test_set_extract_dry, color='black',  linewidth=1.5, label=r'Measured' )
plt.plot(time, out_nn_test_set_extract_dry,color='#1f77b4', label=r'Our MS-NN')
plt.title('Road condition: Dry')
plt.ylabel(r'acc [$m/s^2$]')
plt.legend(fontsize=11, facecolor='white', framealpha=1.0)
plt.grid()
plt.xticks(np.arange(0, max(time) + 1, 10))  
plt.xlim(0, 100)
plt.tick_params(axis='x', labelbottom=False)  # Asse X

plt.subplot(2, 1, 2)
plt.plot(time,samples_test_set_extract_ice, color ='black',  linewidth=1.5, label=r'Measured')
plt.plot(time,out_nn_test_set_extract_ice, color='#d62728', label=r'Our MS-NN')
plt.title('Road condition: Ice')
plt.xlabel(r'Time [$s$]')
plt.ylabel(r'$a_x$ [$m/s^2$]')
plt.legend(fontsize=11, facecolor='white', framealpha=1.0)
plt.grid()
plt.xticks(np.arange(0, max(time) +1, 10))  
plt.xlim(0, 100)


# ______________SECTION 4: generate a dataset with vehicle acc values  ______________

# generate the predictions of the speed from the different channels
n_channels       = 4
# ch 1-> dry, ch 2-> wet, ch 3-> snow, ch 3-> ice
predictions      = np.zeros((n_channels, num_el))
predictions[0, :] = out_nn_test_set_extract_dry
predictions[1, :] = out_nn_test_set_extract_wet
predictions[2, :] = out_nn_test_set_extract_snow
predictions[3, :] = out_nn_test_set_extract_ice

# custom function to compute the evidence
def evidence_fun(predict_error, sigma_evidence):
  # gaussian evidence
  return np.exp(-predict_error**2 / 2 * sigma_evidence**2)

# calculate the MSPRT and WTA
max_window_size = 10   # size of the window for the MSPRT     -> average out noise, brings in more roboustness st = 0.05 -> 25*0.05 = 1.25s
threshold       = 0.2545  # threshold for the MSPRT              -> Slows down the switch to a new channel
sigma_evidence  = 0.1758   # parameter for the evidence function

past_evidence    = np.zeros((n_channels,max_window_size))  # evidence of each channel, from a past window
salience         = np.zeros(n_channels)  # salience of each channel
salience_store   = np.zeros((n_channels, num_el))  # salience all records of each channel
likelihood       = np.zeros(n_channels)  # likelihood of each channel
likelihood_store = np.zeros((n_channels, num_el))  # store the likelihood of each channel
best_channel_msprt=np.zeros(num_el)   # index of the best channel
best_channel_msprt[0]  = 0  # initialize the index of the best channel
best_channel_wta=np.zeros(num_el)   # index of the best channel
best_channel_wta[0]  = 0  # initialize the index of the best channel
window = 1  # initialize the window length

#_______________ MSRT algorithm ________________ 

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
  if np.max(likelihood) >= threshold:
    # select the channel with the highest likelihood
    best_channel_msprt[i] = np.argmax(likelihood)
    # reset the past evidence
    past_evidence        = np.zeros((n_channels, max_window_size))
    # reset the window
    window = 1
  else:
    # keep the previous best channel
    best_channel_msprt[i] = best_channel_msprt[i-1]
    # increase the window
    window = np.min([max_window_size, window + 1])


#_______________ WTA algorithm ________________

predict_error=0
salience_store   = np.zeros((n_channels, num_el))  # salience all records of each channel
for i in range(num_el):
  for chan in range(n_channels):
    predict_error = abs(acc_value[i] - predictions[chan, i])
    evidence = evidence_fun(predict_error, sigma_evidence)  
    salience_store[chan, i] = evidence
    # select the channel with the highest likelihood
    best_channel_wta[i] = np.argmax(salience_store[:, i])
print(best_channel_msprt)
print(best_channel_wta)

# ---------------------------------------------------
# plots
# ---------------------------------------------------
# plot the speed signals and the best channel determined by the MSPRT
plt.figure(figsize=(14, 8))
# plot the speed signal and the predictions
name_data = 'grip_condition_variation'
file_csv = 'grip_condition_variation.csv'
csv_folder_name = 'data_grip'
# Acquire the sample rate value
file = os.path.join('./nn_abs/datasets/', csv_folder_name, file_csv)
abs_file = os.path.abspath(file)
df = pd.read_csv(abs_file)
grip_time = df.columns[0] 
grip_value = df.columns[1]

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
plt.plot(time, acc_value[:num_el], linewidth=2, color='black', label=r'Acc signal')
for i in range(n_channels):
  plt.plot(time, predictions[i], linewidth=3, label=r'Chan ' + str(i))
# plt.xlabel(r'Time [s]')
plt.ylabel(r'$a_x$ [$m/s^2$]')
plt.legend(fontsize=11, loc='lower left', ncol=2, facecolor='white', framealpha=1.0)
plt.grid()
plt.xlim(0, 100)
plt.ylim(-13, 3)
yticks_values = np.arange(-13, 3, 3)  
plt.yticks(yticks_values)
plt.tick_params(axis='x', labelbottom=False) 

# plot the best channel
plt.subplot(3, 1, 3)
plt.plot(time, best_channel_wta, ':', linewidth=0.5,  color='red', label=r'Wta')
plt.plot(time, best_channel_msprt, linewidth=3, color='black', label=r'Msprt')
plt.xlabel(r'Time [s]')
plt.ylabel(r'Index chan [-]')
plt.legend(fontsize=11, facecolor='white', framealpha=1.0, markerscale = 5)
plt.grid()
plt.gca().yaxis.set_major_locator(plt.MaxNLocator(integer=True))
plt.xlim(0, 100)


# plot the likelihood of each channel
plt.figure(figsize=(14, 8))
for i in range(n_channels):
  plt.plot(time, likelihood_store[i], linewidth=3, label='Channel ' + str(i))
plt.xlabel(r'Time [s]')
plt.ylabel(r'Likelihood')
plt.legend(fontsize=11, facecolor='white', framealpha=1.0)
plt.grid()
plt.xlim(0, 100)

# plot the channel selection comparison
plt.figure(figsize=(14, 8))
plt.plot(time, best_channel_wta, ':', linewidth=0.5,  color='red', label=r'Wta')
plt.plot(time, best_channel_msprt, linewidth=3, color='black', label=r'Msprt')
plt.xlabel(r'Time [s]')
plt.ylabel(r'Best channel index [-]')
plt.legend(fontsize=11, facecolor='white', framealpha=1.0, markerscale = 5)
plt.grid()
plt.gca().yaxis.set_major_locator(plt.MaxNLocator(integer=True))
plt.xlim(0, 100)

plt.show()

# ---------------------------------------------------
# Generate the history of the model selection: when abs is activated, stop the model selection
# ---------------------------------------------------

abs_rear_lengths = {}
abs_rear_value = []
abs_front_lengths = {}
abs_front_value = []

# Itera su ciascun file CSV
for file in csv_files:
    # Carica il file CSV in un DataFrame
    df = pd.read_csv(file)
    
    # Verifica se la colonna 'acc' esiste nel file
    if 'ABS_REAR_ON_OFF' in df.columns:
        # Calcola la lunghezza della colonna 'acc' (numero di valori non nulli)
        length = df['ABS_REAR_ON_OFF'].count()
        # Salva la lunghezza associata al nome del file
        abs_rear_lengths[file] = length
        # Aggiungi i valori della colonna 'acc' alla lista
        abs_rear_value.extend(df['ABS_REAR_ON_OFF'].dropna().tolist())
        
    else:
        # Se il file non contiene la colonna 'acc', salva la lunghezza come 0 o None
        abs_rear_lengths[file] = None
        print(f"Colonna 'ABS_REAR_ON_OFF' non trovata nel file: {file}")



for file in csv_files:
  # Carica il file CSV in un DataFrame
  df = pd.read_csv(file) 
  # Verifica se la colonna 'acc' esiste nel file
  if 'ABS_FRONT_ON_OFF' in df.columns:
      # Calcola la lunghezza della colonna 'acc' (numero di valori non nulli)
      length = df['ABS_FRONT_ON_OFF'].count()
      # Salva la lunghezza associata al nome del file
      abs_front_lengths[file] = length
      # Aggiungi i valori della colonna 'acc' alla lista
      abs_front_value.extend(df['ABS_FRONT_ON_OFF'].dropna().tolist())
      
  else:
      # Se il file non contiene la colonna 'acc', salva la lunghezza come 0 o None
      abs_front_lengths[file] = None
      print(f"Colonna 'ABS_FRONT_ON_OFF' non trovata nel file: {file}")

num_el_abs_rear = 0
num_el_abs_front = 0
n_file = 0
# Stampa il risultato
for file, length in abs_rear_lengths.items():
    print(f"File: {file}, lenght 'ABS_REAR_ON_OFF': {length}")
    n_file = n_file+1
    num_el_abs_rear = num_el_abs_rear + abs_rear_lengths[file]
num_el_abs_rear = num_el_abs_rear -n_file*(sw - 1)
n_file = 0
for file, length in abs_front_lengths.items():
    print(f"File: {file}, lenght 'ABS_FRONT_ON_OFF': {length}")
    n_file = n_file+1
    num_el_abs_front = num_el_abs_front + abs_front_lengths[file]
num_el_abs_front = num_el_abs_front -n_file*(sw - 1)

print("The correspondig total ABS state values  are: ", num_el_abs_front , " for front and ", num_el_abs_rear, " for rear")
num_el_abs = min(num_el_abs_front, num_el_abs_rear, num_el)
print("The number of the usable wrt the syntetic values are: ", num_el_abs)

#Generate a matrix where are saved all the abs act/deact states
num_axles = 2
abs_value       = np.zeros((num_axles, num_el_abs))
abs_value[0, :] = abs_front_value[:num_el_abs]
abs_value[1, :] = abs_rear_value[:num_el_abs]

#Create the new matrix where save the blocked selection when braking
final_ABS_best_channel = np.zeros((num_axles, num_el_abs))
switch_brake = np.zeros((num_axles))
value = np.zeros((num_axles))

for i in range(num_el_abs):
  for axle in range(num_axles):
    if abs_value[axle, i] == 1 and switch_brake[axle] == 0:
      value[axle] = best_channel_msprt[i]
      switch_brake[axle] = 1
      final_ABS_best_channel[axle, i] = value[axle]
    elif abs_value[axle, i] == 1 and switch_brake[axle] == 1:
       final_ABS_best_channel[axle, i] = value[axle]
    else:
        value[axle] = best_channel_msprt[i]
        switch_brake[axle] = 0
        final_ABS_best_channel[axle, i] = value[axle]   

# Plot the comparison between the simple algorimt and the signal to send to ABS system
plt.figure(figsize=(14, 8))
plt.plot(time, best_channel_msprt, color='black', linewidth=3, label=r'Channel selection')
plt.plot(time, final_ABS_best_channel[0, :], '-',  linewidth=2, color='blue', label=r'Channel to ABS control for front')
plt.plot(time, final_ABS_best_channel[1, :], '--', linewidth=1, color='red', label=r'Channel to ABS control for rear')
plt.xlabel(r'Time [s]')
plt.ylabel(r'Final channel selection [-]')
plt.legend(fontsize=11, facecolor='white', framealpha=1.0)
plt.grid()
plt.xlim(0, 100)
plt.gca().yaxis.set_major_locator(plt.MaxNLocator(integer=True))
plt.show()

plt.figure(figsize=(25/2.54, 15/2.54))

# Graphs
plt.subplot(2, 1, 1)
plt.plot(time, best_channel_msprt, color='black', linewidth=1, label=r'Channel selection')
plt.plot(time, final_ABS_best_channel[0, :], linewidth=3, color='blue', label=r'Channel to ABS ctrl for front')
#plt.xlabel(r'Time [s]')
plt.ylabel(r'Final chan slctn [-]')
plt.legend(fontsize=11, facecolor='white', framealpha=1.0)
plt.grid()
plt.xlim(0, 100)
plt.gca().yaxis.set_major_locator(plt.MaxNLocator(integer=True))
plt.tick_params(axis='x', labelbottom=False)

plt.subplot(2, 1, 2)
plt.plot(time, best_channel_msprt, color='black', linewidth=1, label=r'Channel selection')
plt.plot(time, final_ABS_best_channel[1, :], linewidth=3, color='red', label=r'Channel to ABS ctrl rear')
plt.xlabel(r'Time [s]')
plt.ylabel(r'Final chan slctn [-]')
plt.legend(fontsize=11, facecolor='white', framealpha=1.0)
plt.grid()
plt.xlim(0, 100)
plt.gca().yaxis.set_major_locator(plt.MaxNLocator(integer=True))
#plt.savefig("graph_ch_slctn.pdf", bbox_inches='tight')

plt.show()


# ---------------------------------------------------
# Data padding for the last value in order to come back to the original dataset dimention (must to do due to windowing process in new data synthesis)
# ---------------------------------------------------

# Get the last value of each row
ultimo_valore = final_ABS_best_channel[:, -1].reshape(-1, 1)
print("Last value ", ultimo_valore)
# Calculate how many values are missing
mancanti = initial_num_el - num_el_abs
print("Missing value ", mancanti)
# Create a padding matrix with the last value repeated
padding = np.tile(ultimo_valore, (1, mancanti))

# Concatenate the padding to the original matrix
final_ABS_best_channel = np.hstack((final_ABS_best_channel, padding))

# Verify the new shape
print("Final matrix shape:", final_ABS_best_channel.shape)
print(final_ABS_best_channel)

