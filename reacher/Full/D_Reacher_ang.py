""""
input: torques output: angles under nnodely platform
import sys
nnodely_path = "/Users/zahra/Documents/Mojtaba/Python_libs/nnodely"

if nnodely_path not in sys.path:
    sys.path.insert(0, nnodely_path)

import nnodely  # o qualunque modulo ti serve da lì
"""


# dont remove this: 
# export PYTHONPATH="/Users/zahra/Documents/Mojtaba/Python_libs/nnodely:$PYTHONPATH"

import sys
import os
sys.path.append(os.getcwd())
#sys.path.append("/Users/zahra/Documents/Mojtaba/Python_libs/nnodely")

from nnodely import *


#  Neural model to estimate l1 and l2 

# Inputs from dataset
theta1_data = Input('theta1_data')
theta2_data = Input('theta2_data')
x_tip_data  = Input('x_tip_data')
y_tip_data  = Input('y_tip_data')
T1_data     = Input('T1_data')
T2_data     = Input('T2_data')
thetadot1_data  = Input('thetadot1_data')
thetadot2_data  = Input('thetadot2_data')
#thetaddot1_data = Input('thetaddot1_data')
#thetaddot2_data = Input('thetaddot2_data')


m  = Parameter('m', values=2.8)  
#m= 1.6
#l = Parameter('l', values=.01)

#m = 1.555
l = .01
ml2= m * l**2   # m * l^2

theta1_d = theta1_data.last()
theta2_d = theta2_data.last()
#theta2_d =    Fir(W_init='init_constant', W_init_params={'value': 1})(theta2_data.last())
#-
thetadot1_d = thetadot1_data.last()
#thetadot1_d = Fir(W_init='init_constant', W_init_params={'value': 1})(thetadot1_data.last())
#-
thetadot2_d = thetadot2_data.last()
#thetadot2_d = Fir(W_init='init_constant', W_init_params={'value': 1})(thetadot2_data.last())
#-
#thetaddot1_d = thetaddot1_data.last()
#thetaddot2_d = thetaddot2_data.last()

T1_d = T1_data.last() * .055 #/ 200 # *200
#T1_d = Fir(W_init='init_constant', W_init_params={'value': .055})(T1_data.last())
T2_d = T2_data.last() * .064 #/ 200 # *200
#T2_d = Fir(W_init='init_constant', W_init_params={'value': .055})(T2_data.last())


thetaddot1 = (   (
   12 * (-7 * T1_d + 7 * T2_d + 6 * T2_d * Cos(theta2_d))
 + 6 * ml2 * (-3 * Sin(2 * theta2_d) * (thetadot1_d * thetadot1_d)
 + - 7 * Sin(theta2_d) * ((thetadot1_d + thetadot2_d) * (thetadot1_d + thetadot2_d)))
) / (ml2 * (-115 + 18 * Cos(2 * theta2_d))))


thetaddot2 = ( (
   6 * (2 * (7 * T1_d - 26 * T2_d + 6 * (T1_d - 2 * T2_d) * Cos(theta2_d))
 + ml2 * ((26 * Sin(theta2_d) + 6 * Sin(2 * theta2_d)) * (thetadot1_d * thetadot1_d)
 + 2 * (7 * Sin(theta2_d) + 3 * Sin(2 * theta2_d)) * thetadot1_d * thetadot2_d
 + (7 * Sin(theta2_d) + 3 * Sin(2 * theta2_d)) * (thetadot2_d * thetadot2_d)))
) / (ml2 * (-115 + 18 * Cos(2 * theta2_d))))


Cs0  = 0 * theta1_d # Parameter('Cs0', values=0)  
Cs1  = 1# Parameter('Cs1', values=1)  
Cs2  = 1# Parameter('Cs2', values=1)  
Cs3  = 0# Parameter('Cs3', values=0)  


thetadot1_bef = Cs0 + ( Integrate(thetaddot1, int_name ="dtheta1_init", method="euler")) #trapezoidal
thetadot2_bef =   Integrate(thetaddot2, int_name ="dtheta2_init", method="euler") # euler 

thetadot1_bef.closedLoop(thetadot1_data)
thetadot2_bef.closedLoop(thetadot2_data)

theta1_bef =   Integrate(thetadot1_bef, int_name ="theta1_init", method="euler") #trapezoidal
theta2_bef =   Integrate(thetadot2_bef, int_name ="theta2_init", method="euler") # euler 

theta1_bef.closedLoop(theta1_data)
theta2_bef.closedLoop(theta2_data)


#K1 = Parameter('K1', values=1) 
#thetadot2_bef = K1 * thetadot2_bef1

theta1 = Output('theta1', theta1_bef)
theta2 = Output('theta2', theta2_bef)

# Model container
model = Modely(seed=0)

model.addModel('theta1', theta1)
model.addModel('theta2', theta2)


# Objectives

model.addMinimize('theta1-error', theta1_d, theta1, 'mse')
model.addMinimize('theta2-error', theta2_d, theta2, 'mse')

#  sample_time consistent with CSV 
model.neuralizeModel(sample_time=0.02)

#  Data loading (CSV in current folder) 


data_struct =  ['step', 'T1_data','T2_data',('theta1_data', "theta1_init"), ('theta2_data', "theta2_init"),
 'x_tip_data', 'y_tip_data', ('thetadot1_data',"dtheta1_init"), ('thetadot2_data',"dtheta2_init"),
  'thetaddot1_data', 'thetaddot2_data']





data_folder = os.path.join(os.getcwd(), 'dataset', 'data')

# Load the CSV file
model.loadData(
    name='reacher_data',
    source=data_folder,
    format=data_struct,
    delimiter=';')
#skeeplines = 1

# Training 
train_params = {'num_of_epochs': 1, 'train_batch_size': 128, 'lr': 0.01}
# model.trainModel(splits=[70, 20, 10], training_params=train_params, prediction_samples=20, step=18) 

model.trainModel(dataset='reacher_data', splits=[70, 20, 10], training_params=train_params, prediction_samples=10, step=None) 
# step 18:it restarts 18 steps later so faster

model.neuralizeModel(sample_time=0.02) ########## check thissssssssss


##########################
########################################################################

##########################
#   PREDICTION + PLOTS   #
##########################

import numpy as np
import matplotlib
matplotlib.use("TkAgg")   # oppure "QtAgg" se hai Qt installato
import matplotlib.pyplot as plt


# 1) Choose how many time-steps you want to see in the plot
WINDOW = 2000  # or 500, 1000, etc.

# 2) Take a window of samples from dataset (same "trick" as pendulum, but with bigger window)
sample = model.getSamples(dataset='reacher_data', window=(WINDOW//1))

# 3) Run model to get predictions on that window
result = model(sample, sampled=True, prediction_samples=WINDOW)
# result = model(sample, sampled=True, num_of_samples=WINDOW, prediction_samples=100)

# 4) Convert to numpy arrays and flatten

y1_true = np.array(sample['theta1_data']).reshape(-1)
y1_pred = np.array(result['theta1']).reshape(-1)
y2_true = np.array(sample['theta2_data']).reshape(-1)
y2_pred = np.array(result['theta2']).reshape(-1)

# 5) Make sure all series have the same length
N = min(len(y2_true), len(y2_pred), len(y1_pred), len(y1_true))
N = N // 4

y1_true = y1_true[:N]
y1_pred = y1_pred[:N]
y2_true = y2_true[:N]
y2_pred = y2_pred[:N]

x = np.arange(N)


# ---- Plot for theta2 ----
plt.figure()
plt.plot(x, y2_true, label='theta2_data (true)')
plt.plot(x, y2_pred, '--', label='theta2 (pred)')
plt.xlabel('sample index')
plt.ylabel('theta2')
plt.legend()
plt.title('theta2: true vs predicted')
plt.grid(True)


# ---- Plot for theta1 ----
plt.figure()
plt.plot(x, y1_true, label='theta1_data (true)')
plt.plot(x, y1_pred, '--', label='theta1 (pred)')
plt.xlabel('sample index')
plt.ylabel('theta1')
plt.legend()
plt.title('theta1: true vs predicted')
plt.grid(True)

plt.show()




