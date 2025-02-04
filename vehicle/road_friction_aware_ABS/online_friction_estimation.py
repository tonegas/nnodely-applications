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
import csv
import json
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
MS_NN_dry_road  = nnodely(visualizer=MPLVisualizer(), seed=random_seed)
MS_NN_wet_road  = nnodely(visualizer=MPLVisualizer(), seed=random_seed)
MS_NN_snow_road = nnodely(visualizer=MPLVisualizer(), seed=random_seed)
MS_NN_ice_road  = nnodely(visualizer=MPLVisualizer(), seed=random_seed)

# load the parameters of the trained model from the corresponding JSON files
with open(os.path.join(trained_models_dir, 'dry_road', 'net.json'), 'r') as json_file:
    MS_NN_dry_road.model_def = json.load(json_file)
with open(os.path.join(trained_models_dir, 'wet_road', 'net.json'), 'r') as json_file:
    MS_NN_wet_road.model_def = json.load(json_file)
with open(os.path.join(trained_models_dir, 'snow_road', 'net.json'), 'r') as json_file:
    MS_NN_snow_road.model_def = json.load(json_file)
with open(os.path.join(trained_models_dir, 'ice_road', 'net.json'), 'r') as json_file:
    MS_NN_ice_road.model_def = json.load(json_file)

# sampling time for the input windows (this sampling time can be different from the one of the dataset: an interpolation will be automatically performed)
sample_time = 0.05  # [s]

# number of samples in the input windows (for the past engine torque/brake pedal values, up to current time step)
n = 10   

# visualize the models and check that they have been correctly loaded
MS_NN_dry_road.neuralizeModel()
MS_NN_wet_road.neuralizeModel()
MS_NN_snow_road.neuralizeModel()
MS_NN_ice_road.neuralizeModel()

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
initial_num_el = 0
for file, length in acc_lengths.items():
    n_file = n_file+1
    num_el = num_el + acc_lengths[file]

initial_num_el = num_el
num_el = num_el -n_file*(n - 1)
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

# plot the results
plt.figure(figsize=(10, 5))
plt.subplot(4, 1, 1)
plt.plot(time, samples_test_set_extract_dry,label='Target')
plt.plot(time, out_nn_test_set_extract_dry,label='NN dry trained',linestyle='--')
# plt.title('Dry trained NN')
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
'''
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
num_el_abs_rear = num_el_abs_rear -n_file*(n - 1)
n_file = 0
for file, length in abs_front_lengths.items():
    print(f"File: {file}, lenght 'ABS_FRONT_ON_OFF': {length}")
    n_file = n_file+1
    num_el_abs_front = num_el_abs_front + abs_front_lengths[file]
num_el_abs_front = num_el_abs_front -n_file*(n - 1)

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

'''