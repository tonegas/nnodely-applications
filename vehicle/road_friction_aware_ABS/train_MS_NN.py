'''
  __  __ ____        _   _ _   _    _____ ____      _    ___ _   _ ___ _   _  ____ 
 |  \/  / ___|      | \ | | \ | |  |_   _|  _ \    / \  |_ _| \ | |_ _| \ | |/ ___|
 | |\/| \___ \ _____|  \| |  \| |    | | | |_) |  / _ \  | ||  \| || ||  \| | |  _ 
 | |  | |___) |_____| |\  | |\  |    | | |  _ <  / ___ \ | || |\  || || |\  | |_| |
 |_|  |_|____/      |_| \_|_| \_|    |_| |_| \_\/_/   \_\___|_| \_|___|_| \_|\____|
                                                                                  

This script implements and trains the model-structured neural networks (MS-NNs) proposed in the paper titled:
"A Road Friction-Aware Anti-Lock Braking System Based on Model-Structured Neural Networks"

27/01/2025 -- Mattia Piccinini, Matteo Zumerle, Johannes Betz, Gastone Pietro Rosati Papini
'''

# ----------------------------------------------------------
# initial setup
# ----------------------------------------------------------

# import the necessary libraries
import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=18)

# import the nnodely library
from nnodely import *

# set a random seed for reproducibility
random_seed = 10

# ----------------------------------------------------------
# training dataset selection
# ----------------------------------------------------------

# define the road condition to be used for training, and the corresponding dataset:
# possible options are: 'dry', 'wet', 'snow', 'ice'
dataset_select = 'dry'

# check if the dataset_select is valid
if dataset_select not in ['dry', 'wet', 'snow', 'ice']:
    print(f"Invalid dataset selection: {dataset_select}")
    sys.exit()

# folder containing the datasets
dataset_folder = os.path.join(os.path.dirname(os.path.realpath(__file__)),'datasets',f'data_{dataset_select}_noise')

# load the training datasets:
# dataset obtained by controlling the vehicle speed with a PID controller
train_data_speed_control = pd.read_csv(os.path.join(dataset_folder,f'{dataset_select}_noise.csv'))
# dataset obtained by controlling the throttle/brake pedals directly
train_data_pedal_control = pd.read_csv(os.path.join(dataset_folder,f'{dataset_select}_pedal_noise.csv'))

# folder to save the model
path_folder_save = os.path.join(os.path.dirname(os.path.realpath(__file__)),'trained_models',f'{dataset_select}_road')

# create a model with the nnodely library
MS_NN = nnodely(visualizer=MPLVisualizer(), seed=random_seed, workspace=path_folder_save)

# ----------------------------------------------------------
# build the MS-NN's internal architecture
# ----------------------------------------------------------

# define the needed signals
speed        = Input('speed')           # vehicle speed (input)
brake_pedal  = Input('brake_pedal')     # brake pedal (input)
motor_torque = Input('motor_torque')    # motor torque (input)
accel        = Input('acc')             # longitudinal acceleration (output)

# -------------------------------
# hyperparameters
# -------------------------------

# sampling time for the input windows (this sampling time can be different from the one of the dataset: an interpolation will be automatically performed)
sample_time = 0.05  # [s]

# number of samples in the input windows (for the past engine torque/brake pedal values, up to current time step)
n = 10   

# number of neurons in the fully connected layers of the data-driven part of the MS-NN
w = int(n/2)

# -------------------------------
# model-based part of the MS-NN
# -------------------------------

# create the function computing the longitudinal acceleration with the model-based part
def acc_model_based(Ty1,Ty2,v, mass,Kd,Cv,Cr):
    # function inputs:
    # Ty1,Ty2,v --> front and rear wheel torques and vehicle speed

    # learnable parameters:
    # ma,,Kd,Cv,Cr --> vehicle mass, aero drag, linear drag and rolling resistance coefficients
    
    # non-trainable parameters
    r1 = r2   = 0.2286     # [m] front and rear wheel radii 
    Iw1 = Iw2 = 0.5225796  # [kg*m^2] front and rear wheel inertias
    g_acc     = 9.81       # [m/s^2] gravity acceleration
    
    # function output: longitudinal acceleration, computed using the Newton's vehicle dynamics laws 
    return ((1.0/mass)*((Ty1/r1) + (Ty2/r2) - Kd * v**2 - Cv * v) - Cr*g_acc)/(1.0 + (2.0/mass)*((Iw1/r1**2.0)+(Iw2/r2**2.0)))

# -------------------------------
# neuro-fuzzy part of the MS-NN
# -------------------------------

# local neuro-fuzzy models to estimate the wheel torques induced by the motor torque
center_vector_torque = [0, 50, 100]        # centers of the activation (or membership) functions, in % of the maximum torque
max_torque = max(np.max(train_data_speed_control['motor_torque']).item(), np.max(train_data_pedal_control['motor_torque']).item())  # maximum torque value
scaling_factor_torque = (max_torque/100)   # scaling factor for the torque values
# fuzzyfication of the motor torque, with triangular activation (or membership) functions. The activation function's input is a window of n past samples of the motor torque
fuz_tor = Fuzzify(centers=([x * scaling_factor_torque for x in center_vector_torque]), functions='Triangular')(motor_torque.sw(n))  
# FIR filter (i.e., fully-connected layer) used as a local model to estimate the wheel torque induced by the motor torque. The FIR's input is a window of n past samples of the motor torque
loc_tor = LocalModel(output_function=Fir)(motor_torque.sw(n),fuz_tor)

# local neuro-fuzzy models to estimate the wheel torques induced by the brake pedal
center_vector_brake  = [0, 10, 50, 100]  # centers of the activation (or membership) functions, in % of the maximum brake pedal value
max_brake = max(np.max(train_data_speed_control['brake_pedal']).item(), np.max(train_data_pedal_control['brake_pedal']).item())  # maximum brake pedal value
scaling_factor_brake = (max_brake/100)   # scaling factor for the brake pedal values
# fuzzyfication of the brake pedal, with triangular activation (or membership) functions. The activation function's input is a window of n past samples of the brake pedal
fuz_brake = Fuzzify(centers=([x * scaling_factor_brake for x in center_vector_brake]))(brake_pedal.sw(n))
# FIR filter (i.e., fully-connected layer) used as a local model to estimate the wheel torque induced by the brake pedal. The FIR's input is a window of n past samples of the brake pedal
loc_brake = LocalModel(output_function=Fir)(brake_pedal.sw(n),fuz_brake) # mettere brake_pedal

# compute the resulting wheel torques
# front wheel torques (braking torque only)
Ty1 = loc_brake
# rear wheel torques (braking torque + motor torque)
Ty2 = loc_brake + loc_tor

# Build a parametric function for the model-based part of the MS-NN
model_based_out = ParamFun(acc_model_based)(Ty1,Ty2,speed.last())

# -------------------------------
# data-driven part of the MS-NN
# -------------------------------
# data-driven layers to estimate the effects of the aero drag and linear drag forces
air_drag_data_driven     = Linear(b=True)(speed.last()**2)  # simple linear (aka fully-connected) layer to estimate the aero drag force
linear_speed_data_driven = Linear(b=True)(speed.last())     # simple linear (aka fully-connected) layer to estimate the linear drag force

# data-driven layers to estimate the effects of the braking forces, by processing a past window of brake pedal values
braking_force_data_driven_layer_1 = -Relu(Fir(output_dimension=w, parameter_init = init_negexp, parameter_init_params={'size_index':0, 'first_value':0.01, 'lambda':3})(brake_pedal.sw(n)))
braking_force_data_driven_layer_2 = Tanh(braking_force_data_driven_layer_1)
braking_force_data_driven = Linear(b=True)(braking_force_data_driven_layer_2)

# data-driven layers to estimate the effects of the engine force, by processing a past window of motor torque values
engine_force_step_data_driven_layer_1 = Fir(output_dimension=w, parameter_init = init_negexp, parameter_init_params={'size_index':0, 'first_value':1, 'lambda':3})(motor_torque.sw(n))
engine_force_step_data_driven_layer_2 = Tanh(engine_force_step_data_driven_layer_1)
motor_force_data_driven = Linear(engine_force_step_data_driven_layer_2)

# the output of the data-driven part is the sum of the outputs of the single layers, to comply with the additivity of forces in the Newton's vehicle dynamics laws
data_driven_out = (air_drag_data_driven + linear_speed_data_driven + braking_force_data_driven + motor_force_data_driven)

# -------------------------------
# overall output of the MS-NN
# -------------------------------

# the overall output combines the model-based and data-driven parts
out = Output('acceleration',  (model_based_out + data_driven_out)/2)

# add this neural model to nnodely
MS_NN.addModel('acc',[out])

# add a loss function to minimize the error between the output of the MS-NN and the actual recorded longitudinal acceleration
MS_NN.addMinimize('acc_error', accel.last(), out, loss_function='rmse')

# neuralize the model with the chosen sampling time for the input windows
MS_NN.neuralizeModel(sample_time)

# -------------------------------
# load the training dataset
# -------------------------------

# define the structure of the dataset (columns of the csv file to be loaded)
data_struct = ['time','brake_pedal','','speed','acc','','','','','motor_torque']

# load the dataset
dataset_name = f'dataset_{dataset_select}'
MS_NN.loadData(name=dataset_name, source=dataset_folder, format=data_struct, skiplines=1)

# -------------------------------
# train the MS-NN
# ------------------------------

MS_NN.trainModel(splits=[100, 0, 0], shuffle_data=True, training_params={'num_of_epochs':800, 'val_batch_size':128, 'train_batch_size':128, 'lr':0.005})

# save the trained model
MS_NN.saveModel()

