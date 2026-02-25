""""
Complete one input: torques output: positions under nnodely platform
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

#def function(x, y):
#    import pandas as pd
#    return pd.derivate(x, y)
#ParamFun(fun=function)



m  = Parameter('m', values=.03)  
#m= 1.6
#l = Parameter('l', values=.01)

#m = 1.555
l = .1
ml2= m * l**2   # m * l^2
l1= l
l2 = .11

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

x_out = Output('x_out',(l1 * Cos(theta1_bef)) + (l2 * Cos(theta1_bef + theta2_bef)))
y_out = Output('y_out',(l1 * Sin(theta1_bef)) + (l2 * Sin(theta1_bef + theta2_bef)))



# Model container
model = Modely(seed=0)
model.addModel('theta1', theta1)
model.addModel('theta2', theta2)
model.addModel('x_out', x_out)
model.addModel('y_out', y_out)



# Objectives
model.addMinimize('x-error', x_tip_data.last(), x_out, 'mse')
model.addMinimize('y-error', y_tip_data.last(), y_out, 'mse')


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
train_params = {'num_of_epochs': 0, 'train_batch_size': 128, 'lr': 0.01}
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

x_ft_true = np.array(sample['x_tip_data']).reshape(-1)
x_ft_pred = np.array(result['x_out']).reshape(-1)
y_ft_true = np.array(sample['y_tip_data']).reshape(-1)
y_ft_pred = np.array(result['y_out']).reshape(-1)

# 5) Make sure all series have the same length
N = min(len(y2_true), len(y2_pred), len(y1_pred), len(y1_true), len(x_ft_pred), len(y_ft_true))
N = N // 4

y1_true = y1_true[:N]
y1_pred = y1_pred[:N]
y2_true = y2_true[:N]
y2_pred = y2_pred[:N]
y_ft_pred = y_ft_pred[:N]
x_ft_pred =  x_ft_pred[:N]
y_ft_true = y_ft_true[:N]
x_ft_true =  x_ft_true[:N]

x = np.arange(N)

'''
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


# ---- Plot for pos x ----
plt.figure()
plt.plot(x, x_ft_true, label='True fingertip x')
plt.plot(x, x_ft_pred, '--', label='Predicted fingertip x')
plt.xlabel('sample index')
plt.ylabel('x')
plt.legend()
plt.title('x: true vs predicted')
plt.grid(True)


# ---- Plot for pos y ----
plt.figure()
plt.plot(x, y_ft_true, label='True fingertip y')
plt.plot(x, y_ft_pred, '--', label='Predicted fingertip y')
plt.xlabel('sample index')
plt.ylabel('y')
plt.legend()
plt.title('y: true vs predicted')
plt.grid(True)

plt.show()
'''

'''
# ---- Plot for theta2 & theta1 (subplots) ----
plt.figure()
plt.subplot(2, 1, 1)
plt.plot(x, y2_true, color='k', label='theta2_data (true)')
plt.plot(x, y2_pred, '-.', color='r', label='theta2 (pred)')
plt.xlabel('sample index', fontsize=14)
plt.ylabel('theta2', fontsize=14)
plt.legend(fontsize=12)
plt.title('theta2: true vs predicted', fontsize=16)
plt.grid(True)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

plt.subplot(2, 1, 2)
plt.plot(x, y1_true, color='k', label='theta1_data (true)')
plt.plot(x, y1_pred, '-.', color='r', label='theta1 (pred)')
plt.xlabel('sample index', fontsize=14)
plt.ylabel('theta1', fontsize=14)
plt.legend(fontsize=12)
plt.title('theta1: true vs predicted', fontsize=16)
plt.grid(True)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

plt.tight_layout()

# ---- Plot for pos x & pos y (subplots) ----
plt.figure()
plt.subplot(2, 1, 1)
plt.plot(x, x_ft_true, color='k', label='True fingertip x')
plt.plot(x, x_ft_pred, '-.', color='r', label='Predicted fingertip x')
plt.xlabel('sample index', fontsize=14)
plt.ylabel('x', fontsize=14)
plt.legend(fontsize=12)
plt.title('x: true vs predicted', fontsize=16)
plt.grid(True)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

plt.subplot(2, 1, 2)
plt.plot(x, y_ft_true, color='k', label='True fingertip y')
plt.plot(x, y_ft_pred, '-.', color='r', label='Predicted fingertip y')
plt.xlabel('sample index', fontsize=14)
plt.ylabel('y', fontsize=14)
plt.legend(fontsize=12)
plt.title('y: true vs predicted', fontsize=16)
plt.grid(True)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

plt.tight_layout()
plt.show()
'''


# ---- One figure with 4 subplots: theta2, theta1, pos x, pos y ----
fig_all = plt.figure(figsize=(10, 5))

# 1) theta2
plt.subplot(2, 2, 1)
plt.plot(x, y2_true, color='k', linewidth=2, label=r'$\theta_2$ (Reference)')
plt.plot(x, y2_pred, '-.', color='r', label=r'$\theta_2$ (pred)')
plt.xlabel('Time step')
plt.ylabel(r'$\theta_2$')
plt.legend()
plt.title(r'$\theta_2$: Reference vs Predicted')
plt.grid(True)

# 2) theta1
plt.subplot(2, 2, 2)
plt.plot(x, y1_true, color='k', linewidth=2, label=r'$\theta_1$ (Reference)')
plt.plot(x, y1_pred, '-.', color='r', label=r'$\theta_1$ (pred)')
plt.xlabel('Time step')
plt.ylabel(r'$\theta_1$')
plt.legend()
plt.title(r'$\theta_1$: Reference vs Predicted')
plt.grid(True)

# 3) fingertip x
plt.subplot(2, 2, 3)
plt.plot(x, x_ft_true, color='k', linewidth=2, label='Reference fingertip x')
plt.plot(x, x_ft_pred, '-.', color='r', label='Predicted fingertip x')
plt.xlabel('Time step')
plt.ylabel('x')
plt.legend()
plt.title('x: Reference vs Predicted')
plt.grid(True)

# 4) fingertip y
plt.subplot(2, 2, 4)
plt.plot(x, y_ft_true, color='k', linewidth=2, label='Reference fingertip y')
plt.plot(x, y_ft_pred, '-.', color='r', label='Predicted fingertip y')
plt.xlabel('Time step')
plt.ylabel('y')
plt.legend()
plt.title('y: Reference vs Predicted')
plt.grid(True)

plt.tight_layout()

# save the combined figure
fig_all.savefig('Reacher_all.pdf', bbox_inches='tight')
fig_all.savefig('reacher_all_true_vs_pred.png', dpi=300, bbox_inches='tight')

plt.show()










