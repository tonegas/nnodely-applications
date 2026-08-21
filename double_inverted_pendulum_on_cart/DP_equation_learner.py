from nnodely import *
import sys
import os
import pandas as pd
import numpy as np
import glob
import torch
from nnodely.support import earlystopping
from sklearn.model_selection import train_test_split

sys.path.append(os.getcwd())

workspace = os.path.join(os.getcwd(), "saved_nets")
torch.set_num_threads(5)

# # ---- PREPROCESS CSVs ----
# data_dir = "data/data_DP"
# file_list = glob.glob(os.path.join(data_dir, "*.csv"))

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

linear_out = Linear(output_dimension=3, W_init=init_random_range, W_init_params={'min_value': -init_value, 'max_value': init_value})

def Pow2(x):
    return x **2
Pow2_fun = ParamFun(param_fun=Pow2)

eq_learner1 = EquationLearner(functions=[Sin, Sin, Cos, Cos, Identity, Mul, Add, Sub, Pow2_fun], linear_in=linear_in)
eq_learner2 = EquationLearner(functions=[Sin, Sin, Cos, Cos, Identity, Mul, Add, Sub, Pow2_fun], linear_in=linear_hidden1)
eq_learner3 = EquationLearner(functions=[Sin, Sin, Cos, Cos, Identity, Mul, Add, Sub, Pow2_fun], linear_in=linear_hidden2)
eq_learner4 = EquationLearner(functions=[Sin, Sin, Cos, Cos, Identity, Mul, Add, Sub, Pow2_fun], linear_in=linear_hidden3, linear_out=linear_out)

eq_in = (p.last(), v.last(), theta1.last(), omega1.last(), theta2.last(), omega2.last(), action.last())
eq1 = eq_learner1(inputs=eq_in)
eq2 = eq_learner2(eq1)
eq3 = eq_learner3(eq2)
eq4 = eq_learner4(eq3)

acc_cart_est = Select(eq4, 0)
th1_dotdot_est = Select(eq4, 1)
th2_dotdot_est = Select(eq4, 2)

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

# Load test data
test_data_folder = os.path.join(os.path.abspath(''),'data/data_DP_test')
test_data = pd.DataFrame()
for file in os.listdir(test_data_folder):
    if file.endswith(".csv"):
        data_path = os.path.join(test_data_folder, file)
        data = pd.read_csv(data_path)
        test_data = pd.concat([test_data, data], ignore_index=True)
test_data = test_data.astype(np.float64)

# Multiple seed and multiple fraction training
data_folder = os.path.join(os.path.abspath(''),'data/data_DP')
data_train = pd.DataFrame()
for file in os.listdir(data_folder):
    if file.endswith(".csv"):
        data_path = os.path.join(data_folder, file)
        data = pd.read_csv(data_path)
        data_train = pd.concat([data_train, data], ignore_index=True)

# Converti tutte le colonne a float64 per evitare problemi con i tipi object
data_train = data_train.astype(np.float64)

cols = ['time','action',('Xpos', 'int_x'),('Xth1', 'int_th1'),('Xth2', 'int_th2'),('Xvelocity', 'int_xdot'),('Xth1_dot', 'int_th1_dot'),('Xth2_dot', 'int_th2_dot'),'Xddx','Xddth1','Xddth2']
# cols = ['time','action','Xpos','Xth1','Xth2','Xvelocity','Xth1_dot','Xth2_dot','Xddx','Xddth1','Xddth2']
fractions = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.25, 0.2, 0.15, 0.1]
seeds = [19, 45, 79, 181, 331]
df = pd.DataFrame(columns=['model_name', 'fraction', 'seed', 'train_samples', 'val_samples', 'train_loss', 'val_loss', 'test_loss'])

for fraction in fractions:
    for seed in seeds:
        if fraction < 1.0:
            X_sub, _ = train_test_split(data_train, train_size=fraction, random_state=seed)
        else:
            X_sub = data_train.copy()
        X_train, X_val = train_test_split(X_sub, test_size=0.2, random_state=seed)
        
        # Converti a float64 per evitare problemi di tipo object
        X_train = X_train.astype(np.float64)
        X_val = X_val.astype(np.float64)
        
        for col in cols:
            if isinstance(col, tuple):
                col, new_col = col
                X_train[new_col] = X_train[col]
                X_val[new_col] = X_val[col]
                
        model_name = f'eq_learner_fraction_{fraction * 100:.0f}_seed_{seed}'
        eqL = Modely(visualizer=TextVisualizer(), workspace=workspace)
        
        eqL.addModel('double_pend_learner', [acc_cart_z_est, th1_ddot_z_est, th2_ddot_z_est, xdot_z_est, x_z_est, omega1_z_est, th1_z_est, omega2_z_est, th2_z_est])

        # Train on the dataset accelerations
        eqL.addMinimize('mse_x_dotdot', a.last(), acc_cart_z_est, loss_function='mse')
        eqL.addMinimize('mse_th1_dotdot', alpha1.last(), th1_ddot_z_est, loss_function='mse')
        eqL.addMinimize('mse_th2_dotdot', alpha2.last(), th2_ddot_z_est, loss_function='mse')

        # Neuralize the model
        eqL.neuralizeModel(0.01)
        
        # Data loading
        eqL.loadData(name='data', source = X_train, format=cols, delimiter=',', header=0)
        eqL.loadData(name='val_data', source = X_val, format=cols, delimiter=',', header=0)
        eqL.loadData(name='test_data', source=test_data, format=cols, delimiter=',', header=0)
        
        # Train the model
        prediction_samples = None
        step = None
        early_stop_patience = 20
        params = {'train_batch_size': 256, 'num_of_epochs': 1000}
        
        eqL.trainModel(train_dataset='data', validation_dataset='val_data', optimizer='Adam', prediction_samples=prediction_samples, step=step, training_params=params, early_stopping=earlystopping.early_stop_patience,  
                                                                    early_stopping_params={'patience':early_stop_patience}, select_model=earlystopping.select_best_model, lr=1e-3)
        
        eqL.exportPythonModel(name=model_name)
        # analyzeModel stampa i risultati ma in nnodely non restituisce un dict (ritorna None); le metriche sono in performance.
        eqL.analyzeModel('test_data')
        perf = eqL.performance['test_data']
        test_loss = perf['total']['mean_error']
        train_loss = np.mean([metrics['train'][-1] for metrics in eqL._training.values()])
        val_loss = np.mean([metrics['val'][-1] for metrics in eqL._training.values()])
        df = pd.concat([df, pd.DataFrame({'model_name': [model_name], 'fraction': [fraction*100], 'seed': [seed], 'train_samples': [len(X_train)], 'val_samples': [len(X_val)], 'train_loss': [train_loss], 'val_loss': [val_loss], 'test_loss': [test_loss]})], ignore_index=True)
        df.to_csv(os.path.join(workspace, 'data_eqL.csv'), index=False)
        
# Analyze results
cols = ['time','action',('Xpos', 'int_x'),('Xth1', 'int_th1'),('Xth2', 'int_th2'),('Xvelocity', 'int_xdot'),('Xth1_dot', 'int_th1_dot'),('Xth2_dot', 'int_th2_dot'),'Xddx','Xddth1','Xddth2']
train_data = [100, 80, 70, 50, 40, 30, 25, 20, 15, 10]

df = pd.DataFrame(columns=['model_name', 'fraction', 'seed', 'total_mse', 'acc_cart_mse', 'th1_ddot_mse', 'th2_ddot_mse'])
for fraction in train_data:
    for seed in seeds:
        
        model_name = f'eq_learner_fraction_{fraction}_seed_{seed}'
        eqL = Modely(workspace=workspace, seed =seed)
        eqL.importPythonModel(name=model_name)
        eqL.neuralizeModel(0.01)
        eqL.loadData(name='test_data', source=test_data, format=cols, delimiter=',', header=0)
        
        eqL.analyzeModel('test_data', batch_size=128)
        perf = eqL.performance['test_data']
        df = pd.concat([df, pd.DataFrame({'model_name': [model_name], 'fraction': [fraction], 'seed': [seed], 'total_mse': [perf['total']['mean_error']], 'acc_cart_mse': [perf['mse_x_dotdot']['mse']], 'th1_ddot_mse': [perf['mse_th1_dotdot']['mse']], 'th2_ddot_mse': [perf['mse_th2_dotdot']['mse']]})], ignore_index=True)
        