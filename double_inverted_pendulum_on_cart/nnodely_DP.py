from nnodely import *
import sys
import os
import pandas as pd
import numpy as np
import glob
import torch

sys.path.append(os.getcwd())

workspace = os.path.join(os.getcwd(), "results")
double_pendulum = Modely(workspace=workspace)
torch.set_num_threads(10)

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
w1 = Input('Xth1_dot')
a1 = Input('Xddth1')

# Second pendulum inputs
theta2 = Input('Xth2')
w2 = Input('Xth2_dot')
a2 = Input('Xddth2')

# ---- PARAMETERS ----
g = Constant('gravity', values=9.81)                                        # gravity acceleration
gear = Parameter('gear', values=500.0)                         # gear ratio
m = Parameter('m', values=10.47)                                # mass of the cart
m1 = Parameter('m1', values=4.19)                               # mass of the first pendulum
m2 = Parameter('m2', values=4.19)                               # mass of the second pendulum
Inertia1 = Parameter('Inertia1', values=0.12)                    # inertia of the first pendulum
Inertia2 = Parameter('Inertia2', values=0.12)                    # inertia of the second pendulum
bc = Parameter('bc', values=1.0)                                 # cart damping coefficient
b1 = Parameter('b1', values=1.0)                                 # first pendulum damping coefficient
b2 = Parameter('b2', values=1.0)                                 # second pendulum damping coefficient
l1 = 0.6                                                                    # length of the first pendulum
l2 = 0.6                                                                    # length of the second pendulum


# ---- DYNAMICS EQUATIONS ----
def inv_double_pend(x, v, th1, omega1, th2, omega2, u, eps=1e-8):
    """
    Restituisce (xdd, th1dd, th2dd) come Stream nnodely.
    ATTENZIONE: m1,m2,m2 nel codice sono:
      - m1: massa carrello (era 'm' nelle equazioni originali)
      - m2: massa pendolo 1 (era 'm1')
      - m2: massa pendolo 2 (era 'm2')
    """
    d1 = m + m1 + m2
    d2 = (1/2 * m1 + m2) * l1
    d3 = 1/2 * m2 * l2
    d4 = Inertia1 + m1 * l1**2 / 4 + m2 * l1**2
    d5 = m2 * l1 * l2 / 2
    d6 = Inertia2 + m2 * l2**2 / 4
    
    f1 = (1/2 * m1 + m2)*g*l1
    f2 = 1/2 * m2 * g * l2

    # forza effettiva
    F = u * gear

    # per leggibilità costruisco i termini grandi come espressioni multilinea
    # ---------------- denominator ----------------
    denom = (
        d3**2*d4*Cos(th1 + th2)**2 - d1*d4*d6 + d1*d5**2*Cos(th2)**2 + d2**2*d6*Cos(th1)**2 - 2*d2*d3*d5*Cos(th1 + th2)*Cos(th1)*Cos(th2)
    )

    # EPS-protected denominator
    denom_safe = denom + eps

    # ---------------- num_xdd ----------------
    num_xdd = F*d5**2*Cos(th2)**2 - F*d4*d6 + bc*d4*d6*v + (d3*d4*f2*Sin(2*th1 + 2*th2))/2 - bc*d5**2*v*Cos(th2)**2 + (d2*d6*f1*Sin(2*th1))/2 - d3*d4*d6*omega2**2*Sin(th1 + th2) - d2*d4*d6*omega1**2*Sin(th1) + d3*d5**2*omega2**2*Sin(th1 + th2)*Cos(th2)**2 + d2*d5**2*omega1**2*Cos(th2)**2*Sin(th1) + b2*d3*d4*omega1*Cos(th1 + th2) - b2*d3*d4*omega2*Cos(th1 + th2) - b1*d2*d6*omega1*Cos(th1) - b2*d2*d6*omega1*Cos(th1) + b2*d2*d6*omega2*Cos(th1) + d2*d5*d6*omega2**2*Cos(th1)*Sin(th2) - d3*d4*d6*omega1*omega2*Sin(th1 + th2) + d3*d5**2*omega1*omega2*Sin(th1 + th2)*Cos(th2)**2 + b1*d3*d5*omega1*Cos(th1 + th2)*Cos(th2) + b2*d3*d5*omega1*Cos(th1 + th2)*Cos(th2) - b2*d3*d5*omega2*Cos(th1 + th2)*Cos(th2) - b2*d2*d5*omega1*Cos(th1)*Cos(th2) + b2*d2*d5*omega2*Cos(th1)*Cos(th2) - d3*d5**2*omega2**2*Cos(th1 + th2)*Cos(th2)*Sin(th2) + d2*d5**2*omega1**2*Cos(th1)*Cos(th2)*Sin(th2) - d3*d4*d5*omega1**2*Cos(th1 + th2)*Sin(th2) - d2*d5*f2*Sin(th1 + th2)*Cos(th1)*Cos(th2) - d3*d5*f1*Cos(th1 + th2)*Cos(th2)*Sin(th1) + d2*d5*d6*omega1*omega2*Cos(th1)*Sin(th2) - d3*d5**2*omega1*omega2*Cos(th1 + th2)*Cos(th2)*Sin(th2)

    # ---------------- num_th1dd ----------------
    num_th1dd = (
          F*d2*d6*Cos(th1) - d1*d6*f1*Sin(th1) + d3**2*f1*Cos(th1 + th2)**2*Sin(th1) + b1*d1*d6*omega1 + b2*d1*d6*omega1 - b2*d1*d6*omega2 - b1*d3**2*omega1*Cos(th1 + th2)**2 - b2*d3**2*omega1*Cos(th1 + th2)**2 + b2*d3**2*omega2*Cos(th1 + th2)**2 - d1*d5**2*omega1**2*Cos(th2)*Sin(th2) + d2**2*d6*omega1**2*Cos(th1)*Sin(th1) + d1*d5*f2*Sin(th1 + th2)*Cos(th2) - d1*d5*d6*omega2**2*Sin(th2) + d3**2*d5*omega2**2*Cos(th1 + th2)**2*Sin(th2) + b2*d1*d5*omega1*Cos(th2) - b2*d1*d5*omega2*Cos(th2) - bc*d2*d6*v*Cos(th1) - F*d3*d5*Cos(th1 + th2)*Cos(th2) - d1*d5*d6*omega1*omega2*Sin(th2) + d3**2*d5*omega1*omega2*Cos(th1 + th2)**2*Sin(th2) - b2*d2*d3*omega1*Cos(th1 + th2)*Cos(th1) + b2*d2*d3*omega2*Cos(th1 + th2)*Cos(th1) + bc*d3*d5*v*Cos(th1 + th2)*Cos(th2) - d3**2*d5*omega2**2*Cos(th1 + th2)*Sin(th1 + th2)*Cos(th2) - d2*d3*f2*Cos(th1 + th2)*Sin(th1 + th2)*Cos(th1) + d2*d3*d6*omega2**2*Sin(th1 + th2)*Cos(th1) - d3**2*d5*omega1*omega2*Cos(th1 + th2)*Sin(th1 + th2)*Cos(th2) + d2*d3*d5*omega1**2*Cos(th1 + th2)*Cos(th1)*Sin(th2) - d2*d3*d5*omega1**2*Cos(th1 + th2)*Cos(th2)*Sin(th1) + d2*d3*d6*omega1*omega2*Sin(th1 + th2)*Cos(th1)
    )

    # ---------------- num_th2dd ----------------
    num_th2dd = (
        F*d3*d4*Cos(th1 + th2) + b2*d2**2*omega1*Cos(th1)**2 - b2*d2**2*omega2*Cos(th1)**2 - d1*d4*f2*Sin(th1 + th2) + d2**2*f2*Sin(th1 + th2)*Cos(th1)**2 + (d1*d5**2*omega2**2*Sin(2*th2))/2 + (d3**2*d4*omega2**2*Sin(2*th1 + 2*th2))/2 - b2*d1*d4*omega1 + b2*d1*d4*omega2 - F*d2*d5*Cos(th1)*Cos(th2) + d1*d4*d5*omega1**2*Sin(th2) + d1*d5*f1*Cos(th2)*Sin(th1) - d2**2*d5*omega1**2*Cos(th1)**2*Sin(th2) + (d1*d5**2*omega1*omega2*Sin(2*th2))/2 - bc*d3*d4*v*Cos(th1 + th2) - b1*d1*d5*omega1*Cos(th2) - b2*d1*d5*omega1*Cos(th2) + b2*d1*d5*omega2*Cos(th2) + (d3**2*d4*omega1*omega2*Sin(2*th1 + 2*th2))/2 + b1*d2*d3*omega1*Cos(th1 + th2)*Cos(th1) + b2*d2*d3*omega1*Cos(th1 + th2)*Cos(th1) - b2*d2*d3*omega2*Cos(th1 + th2)*Cos(th1) + bc*d2*d5*v*Cos(th1)*Cos(th2) - d2**2*d5*omega1**2*Cos(th1)*Cos(th2)*Sin(th1) + d2*d3*d4*omega1**2*Cos(th1 + th2)*Sin(th1) - d2*d3*f1*Cos(th1 + th2)*Cos(th1)*Sin(th1) - d2*d3*d5*omega2**2*Cos(th1 + th2)*Cos(th1)*Sin(th2) - d2*d3*d5*omega2**2*Sin(th1 + th2)*Cos(th1)*Cos(th2) - d2*d3*d5*omega1*omega2*Cos(th1 + th2)*Cos(th1)*Sin(th2) - d2*d3*d5*omega1*omega2*Sin(th1 + th2)*Cos(th1)*Cos(th2)
    )

    # final results (divide numeratori per denom_safe)
    xdd = num_xdd / denom_safe 
    th1dd = num_th1dd / denom_safe 
    th2dd = num_th2dd / denom_safe 
    
    xd = v
    th1d = omega1
    th2d = omega2

    return [xd, xdd, th1d, th1dd, th2d, th2dd]


# ---- INTEGRATE ----
acc_est = inv_double_pend(p.last(), v.last(), theta1.last(), w1.last(), theta2.last(), w2.last(), action.last())

acc_cart_est = acc_est[0]
th1_dotdot_est = acc_est[1]
th2_dotdot_est = acc_est[2]

dt = 0.01

k1 = inv_double_pend(p.last(), v.last(), theta1.last(), w1.last(), theta2.last(), w2.last(), action.last())
k2 = inv_double_pend(p.last() + k1[0]*dt/2, v.last() + k1[1]*dt/2, theta1.last() + k1[2]*dt/2, w1.last() + k1[3]*dt/2, theta2.last() + k1[4]*dt/2, w2.last() + k1[5]*dt/2, action.last())
k3 = inv_double_pend(p.last() + k2[0]*dt/2, v.last() + k2[1]*dt/2, theta1.last() + k2[2]*dt/2, w1.last() + k2[3]*dt/2, theta2.last() + k2[4]*dt/2, w2.last() + k2[5]*dt/2, action.last())
k4 = inv_double_pend(p.last() + k3[0]*dt, v.last() + k3[1]*dt, theta1.last() + k3[2]*dt, w1.last() + k3[3]*dt, theta2.last() + k3[4]*dt, w2.last() + k3[5]*dt, action.last())

dt_xdot_est = (k1[1] + 2*k2[1] + 2*k3[1] + k4[1]) * dt / 6
xdot_est = v.last() + dt_xdot_est
xdot_est.closedLoop(v)

dt_omega1_est = (k1[3] + 2*k2[3] + 2*k3[3] + k4[3]) * dt / 6
omega1_est = w1.last() + dt_omega1_est
omega1_est.closedLoop(w1)

dt_omega2_est = (k1[5] + 2*k2[5] + 2*k3[5] + k4[5]) * dt / 6
omega2_est = w2.last() + dt_omega2_est
omega2_est.closedLoop(w2)

dt_x_est = (k1[0] + 2*k2[0] + 2*k3[0] + k4[0]) * dt / 6
x_est = p.last() + dt_x_est
x_est.closedLoop(p)

dt_th1_est = (k1[2] + 2*k2[2] + 2*k3[2] + k4[2]) * dt / 6
theta1_est = theta1.last() + dt_th1_est
theta1_est.closedLoop(theta1)

dt_th2_est = (k1[4] + 2*k2[4] + 2*k3[4] + k4[4]) * dt / 6
theta2_est = theta2.last() + dt_th2_est
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

# Physical Constraints

# ---- MODEL + MINIMIZERS ----
double_pendulum.addModel('DP_modely', [acc_cart_z_est, th1_ddot_z_est, th2_ddot_z_est, xdot_z_est, x_z_est, omega1_z_est, th1_z_est, omega2_z_est, th2_z_est])

# Each output has its own minimizer (MSE)
double_pendulum.addMinimize('mse_x_dotdot', a.last(), acc_cart_z_est, loss_function='mse')
double_pendulum.addMinimize('mse_th1_dotdot', a1.last(), th1_ddot_z_est, loss_function='mse')
double_pendulum.addMinimize('mse_th2_dotdot', a2.last(), th2_ddot_z_est, loss_function='mse')

# Add loss on the states for recursive learning
double_pendulum.addMinimize('mse_x_vel', v.next(), xdot_z_est, loss_function='mse')
double_pendulum.addMinimize('mse_x_pos', p.next(), x_z_est, loss_function='mse')
double_pendulum.addMinimize('mse_th1_vel', w1.next(), omega1_z_est, loss_function='mse')
double_pendulum.addMinimize('mse_th1_pos', theta1.next(), th1_z_est, loss_function='mse')
double_pendulum.addMinimize('mse_th2_vel', w2.next(), omega2_z_est, loss_function='mse')
double_pendulum.addMinimize('mse_th2_pos', theta2.next(), th2_z_est, loss_function='mse')  

# Neuralize the model
double_pendulum.neuralizeModel(0.01)

cols = ['time','action','Xpos','Xth1','Xth2','Xvelocity','Xth1_dot','Xth2_dot','Xddx','Xddth1','Xddth2']

# Dataset cols for recursive learning
# cols = ['action',('Xpos', 'int_x'),('Xth1', 'int_th1'),('Xth2', 'int_th2'),('Xvelocity', 'int_xdot'),('Xth1_dot', 'int_th1_dot'),('Xth2_dot', 'int_th2_dot'),'Xddx','Xddth1','Xddth2']
data_folder = os.path.join(os.path.abspath(''),'data','data_DP')

double_pendulum.loadData(name='double_pendulum_data', source=data_folder, format=cols, delimiter=',', header=0)

# ---- NN TRAININg ----
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

double_pendulum.trainModel(splits=[100, 0, 0], lr = 0.001, num_of_epochs=300, prediction_samples=15, optimizer='Adam', train_batch_size=256, minimize_gain=gain)

# ---- PREDICTION & EXPORT ----
sample = double_pendulum.getSamples(dataset='double_pendulum_data', window=1)

double_pendulum.exportPythonModel(name='DP_dynamic_model_net_recursive')
# double_pendulum.exportONNX()
# double_pendulum.exportReport()

newNN = Modely(workspace=workspace)
newNN.importPythonModel(name='DP_dynamic_model_net_recursive')

result_old = double_pendulum(sample, sampled=True)
result_new = newNN(sample, sampled=True)