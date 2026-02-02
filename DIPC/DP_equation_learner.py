from nnodely import *
import sys
import os
import pandas as pd
import numpy as np
import glob
import torch

sys.path.append(os.getcwd())

workspace = os.path.join(os.getcwd(), "results")
torch.set_num_threads(5)

# ---- PREPROCESS CSVs ----
data_dir = "data/data_DP"
file_list = glob.glob(os.path.join(data_dir, "*.csv"))

def init_random_range(indexes, params_size, dict_param={'min_value': 0.0, 'max_value': 1.0}):
    import numpy as np
    min_val = dict_param.get('min_value', 0.0)
    max_val = dict_param.get('max_value', 1.0)
    return np.random.uniform(low=min_val, high=max_val)

# ---- DEFINE INPUTS ----
# Cart inputs
action = Input('action')
p = Input('Xpos')
v = Input('Xvelocity')
a = Input('Xddx')

# First pendulum inputs
theta1 = Input('Xth1')
omega1 = Input('Xth1_dot')
alpha1 = Input('Xddth1')

# Second pendulum inputs
theta2 = Input('Xth2')
omega2 = Input('Xth2_dot')
alpha2 = Input('Xddth2')

# ---- DEFINE EQUATION LEARNER MODEL ----
init_value = 0.001
linear_in = Linear(output_dimension=16, W_init=init_random_range, W_init_params={'min_value': -init_value, 'max_value': init_value})
linear_hidden1 = Linear(output_dimension=16,  W_init=init_random_range, W_init_params={'min_value': -init_value, 'max_value': init_value})
linear_hidden2 = Linear(output_dimension=16,W_init=init_random_range, W_init_params={'min_value': -init_value, 'max_value': init_value})
linear_hidden3 = Linear(output_dimension=16,W_init=init_random_range, W_init_params={'min_value': -init_value, 'max_value': init_value})

linear_out_num = Linear(output_dimension=3, W_init=init_random_range, W_init_params={'min_value': -init_value, 'max_value': init_value})
linear_out_den = Linear(output_dimension=1, W_init=init_random_range, W_init_params={'min_value': -init_value, 'max_value': init_value})

def Pow2(x):
    return x **2
Pow2_fun = ParamFun(param_fun=Pow2)

# Learn the denominaor 
eq_learner_den = EquationLearner(functions=[Sin, Cos, Identity, Mul, Add, Sub, Pow2_fun], linear_in=linear_in)
eq_learner2_den = EquationLearner(functions=[Sin, Cos, Identity, Mul, Add, Sub, Pow2_fun], linear_in=linear_hidden1)
eq_learner3_den = EquationLearner(functions=[Sin, Cos, Identity, Mul, Add, Sub, Pow2_fun], linear_in=linear_hidden2, linear_out=linear_out_den)

eq1_den = eq_learner_den(inputs=(action.last(), p.last(), v.last(), theta1.last(), omega1.last(), theta2.last(), omega2.last()))
eq2_den = eq_learner2_den(eq1_den)
eq3_den = eq_learner3_den(eq2_den)

# Learn the numerator
eq_learner_num = EquationLearner(functions=[Sin, Cos, Identity, Mul, Add, Sub, Pow2_fun], linear_in=linear_in)
eq_learner2_num = EquationLearner(functions=[Sin, Cos, Identity, Mul, Add, Sub, Pow2_fun], linear_in=linear_hidden1)
eq_learner3_num = EquationLearner(functions=[Sin, Cos, Identity, Mul, Add, Sub, Pow2_fun], linear_in=linear_hidden2)
eq_learner4_num = EquationLearner(functions=[Sin, Cos, Identity, Mul, Add, Sub, Pow2_fun], linear_in=linear_hidden3, linear_out=linear_out_num)

eq1_num = eq_learner_num(inputs=(action.last(), p.last(), v.last(), theta1.last(), omega1.last(), theta2.last(), omega2.last()))
eq2_num = eq_learner2_num(eq1_num)
eq3_num = eq_learner3_num(eq2_num)
eq4_num = eq_learner4_num(eq3_num)

acc_cart_est = Select(eq4_num, 0) / (eq3_den + 1e-10)
th1_dotdot_est = Select(eq4_num, 1) / (eq3_den + 1e-10)
th2_dotdot_est = Select(eq4_num, 2) / (eq3_den + 1e-10)

# ----------- INTEGRATIONS -----------
xdot_est = Integrate(acc_cart_est, int_name='int_xdot')
xdot_est.closedLoop(v)

omega1_est = Integrate(th1_dotdot_est, int_name='int_th1_dot')
omega1_est.closedLoop(omega1)

omega2_est = Integrate(th2_dotdot_est, int_name='int_th2_dot') 
omega2_est.closedLoop(omega2)

x_est = Integrate(xdot_est, int_name='int_x') 
x_est.closedLoop(p)

theta1_est = Integrate(omega1_est, int_name='int_th1') 
theta1_est.closedLoop(theta1)

theta2_est = Integrate(omega2_est, int_name='int_th2') 
theta2_est.closedLoop(theta2)

# ---- OUTPUTS ----
acc_cart_z_est = Output('acc_cart_est', acc_cart_est)
th1_ddot_z_est = Output('th1_ddot_est', th1_dotdot_est)
th2_ddot_z_est = Output('th2_ddot_est', th2_dotdot_est)

xdot_z_est = Output('xdot_est', xdot_est)
x_z_est = Output('x_est', x_est)
omega1_z_est = Output('omega1_est', omega1_est)
th1_z_est = Output('th1_est', theta1_est)
omega2_z_est = Output('omega2_est', omega2_est)
th2_z_est = Output('th2_est', theta2_est)

eqL = Modely(visualizer=TextVisualizer())
# eqL.addModel('double_pend_learner', [acc_cart_z_est, th1_ddot_z_est, th2_ddot_z_est, xdot_z_est, x_z_est, omega1_z_est, th1_z_est, omega2_z_est, th2_z_est])
eqL.addModel('double_pend_learner', [acc_cart_z_est, th1_ddot_z_est, th2_ddot_z_est])

# Train on the dataset accelerations
eqL.addMinimize('mse_x_dotdot', a.last(), acc_cart_z_est, loss_function='mse')
eqL.addMinimize('mse_th1_dotdot', alpha1.last(), th1_ddot_z_est, loss_function='mse')
eqL.addMinimize('mse_th2_dotdot', alpha2.last(), th2_ddot_z_est, loss_function='mse')

# Add loss on the states for recursive learning
# eqL.addMinimize('mse_x_vel', vx.next(), xdot_z_est, loss_function='mse')
# eqL.addMinimize('mse_x_pos', px.next(), x_z_est, loss_function='mse')
# eqL.addMinimize('mse_th1_vel', th1x_dot.next(), omega1_z_est, loss_function='mse')
# eqL.addMinimize('mse_th1_pos', th1x.next(), th1_z_est, loss_function='mse')
# eqL.addMinimize('mse_th2_vel', th2x_dot.next(), omega2_z_est, loss_function='mse')
# eqL.addMinimize('mse_th2_pos', th2x.next(), th2_z_est, loss_function='mse')

# Neuralize the model
eqL.neuralizeModel(0.01)

# ---- LOAD DATA ----
cols = ['time','action','Xpos','Xth1','Xth2','Xvelocity','Xth1_dot','Xth2_dot','Xddx','Xddth1','Xddth2']

# Dataset cols for recursive learning
# cols = ['time','action',('Xpos', 'int_x'),('Xth1', 'int_th1'),('Xth2', 'int_th2'),('Xvelocity', 'int_xdot'),('Xth1_dot', 'int_th1_dot'),('Xth2_dot', 'int_th2_dot'),'Xddx','Xddth1','Xddth2']

data_folder = os.path.join(os.path.abspath(''),'data','data_DP')

eqL.loadData(name='double_pendulum_data', source=data_folder, format=cols, delimiter=',', header=0)

# ---- NN TRAININg ----
# Normalize losses
max_x_ddot = 373.905342
max_th1_ddot = 1116.799946
max_th2_ddot = 2616.783088

max_x = 12.972762
max_th1 = 22.856282
max_th2 = 7.875139
max_x_dot = 10.807747
max_th1_dot = 13.006098
max_th2_dot = 23.350907

# max_x_ddot = 130.326977
# max_th1_ddot = 317.853023
# max_th2_ddot = 497.20553

max_x_ddot_norm = 1 / max_x_ddot**2
max_th1_ddot_norm = 1 / max_th1_ddot**2
max_th2_ddot_norm = 1 / max_th2_ddot**2

max_x_norm = 1 / max_x**2
max_th1_norm = 1 / max_th1**2
max_th2_norm = 1 / max_th2**2
max_x_dot_norm = 1 / max_x_dot**2
max_th1_dot_norm = 1 / max_th1_dot**2
max_th2_dot_norm = 1 / max_th2_dot**2

gain = {
        'mse_x_pos': max_x_norm,
        'mse_th1_pos': max_th1_norm,
        'mse_th2_pos': max_th2_norm,
        'mse_x_vel': max_x_dot_norm,
        'mse_th1_vel': max_th1_dot_norm,
        'mse_th2_vel': max_th2_dot_norm,
        'mse_x_dotdot': max_x_ddot_norm,
        'mse_th1_dotdot': max_th1_ddot_norm,
        'mse_th2_dotdot': max_th2_ddot_norm
        }

eqL.trainModel(splits=[100, 0, 0], lr = 0.001, num_of_epochs=300, optimizer='Adam', train_batch_size=256, minimize_gain=gain)
# eqL.trainModel(splits=[100, 0, 0], lr = 0.0001, num_of_epochs=200, prediction_samples=15, optimizer='Adam', train_batch_size=256, minimize_gain=gain)

eqL.addMinimize('mse_x_vel', v.next(), xdot_z_est, loss_function='mse')
eqL.addMinimize('mse_x_pos', p.next(), x_z_est, loss_function='mse')
eqL.addMinimize('mse_th1_vel', omega1.next(), omega1_z_est, loss_function='mse')
eqL.addMinimize('mse_th1_pos', theta1.next(), th1_z_est, loss_function='mse')
eqL.addMinimize('mse_th2_vel', omega2.next(), omega2_z_est, loss_function='mse')
eqL.addMinimize('mse_th2_pos', theta2.next(), th2_z_est, loss_function='mse')

eqL.neuralizeModel(0.01)

cols = ['time','action',('Xpos', 'int_x'),('Xth1', 'int_th1'),('Xth2', 'int_th2'),('Xvelocity', 'int_xdot'),('Xth1_dot', 'int_th1_dot'),('Xth2_dot', 'int_th2_dot'),'Xddx','Xddth1','Xddth2']
eqL.trainModel(splits=[100, 0, 0], lr = 0.00001, num_of_epochs=20, prediction_samples=50, optimizer='Adam', train_batch_size=256, minimize_gain=gain)
# ---- PREDICTION & EXPORT ----
sample = eqL.getSamples(dataset='double_pendulum_data', window=1)

eqL.exportPythonModel(name='eq_learner_recursive')
# eqL.exportONNX()
# eqL.exportReport()
newNN = Modely(workspace=workspace)
newNN.importPythonModel(name='eq_learner_recursive')
result_old = eqL(sample, sampled=True)
result_new = newNN(sample, sampled=True)