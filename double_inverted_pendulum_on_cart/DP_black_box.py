from nnodely import *
import numpy as np
import os
import pandas as pd
import sys
import torch
from nnodely.support import earlystopping
from sklearn.model_selection import train_test_split

sys.path.append(os.getcwd())

workspace = os.path.join(os.getcwd(), "saved_nets")
torch.set_num_threads(5)

def init_random_range(indexes, params_size, dict_param={'min_value': 0.0, 'max_value': 1.0}):
    min_val = dict_param.get('min_value', 0.0)
    max_val = dict_param.get('max_value', 1.0)
    return np.random.uniform(low=min_val, high=max_val)


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


hideden_size_1 = 10
hideden_size_2 = 10
dropout = 0.0

init_value = float(np.sqrt(1/4))

inputs = Concatenate(
    Concatenate(
        Concatenate(p.last(), theta1.last()),
        theta2.last()
    ),
    Concatenate(
        Concatenate(v.last(), omega1.last()),
        Concatenate(
            omega2.last(),
            Concatenate(
                a.last(),
                Concatenate(alpha1.last(), alpha2.last())
            )
        )
    )
)

l1 = Linear(output_dimension=hideden_size_1, W_init=init_random_range, W_init_params={'min_value': -init_value, 'max_value': init_value})(inputs)
x1 = Tanh(l1)
# l2 = Linear(output_dimension=hideden_size_2, W_init=init_random_range, W_init_params={'min_value': -init_value, 'max_value': init_value})(x1)
# x2 = Tanh(l2)
nn_out = Linear(output_dimension=3, W_init=init_random_range, W_init_params={'min_value': -init_value, 'max_value': init_value})(x1)

theta1_dot_est = Integrate(Select(nn_out, 1), int_name='int_th1_dot')
theta1_dot_est.closedLoop(omega1)

theta2_dot_est = Integrate(Select(nn_out, 2), int_name='int_th2_dot')
theta2_dot_est.closedLoop(omega2)

x_dot_est = Integrate(Select(nn_out, 0), int_name='int_xdot')
x_dot_est.closedLoop(v)

theta1_est = Integrate(theta1_dot_est, int_name='int_th1')
theta1_est.closedLoop(theta1)

theta2_est = Integrate(theta2_dot_est, int_name='int_th2')
theta2_est.closedLoop(theta2)

x_est = Integrate(x_dot_est, int_name='int_x')
x_est.closedLoop(p)

est_theta1_dotdot = Output('theta1_dotdot_est', Select(nn_out, 1))
est_theta2_dotdot = Output('theta2_dotdot_est', Select(nn_out, 2))
est_x_dotdot = Output('x_dotdot_est', Select(nn_out, 0))

est_theta1dot = Output('theta1_dot_est', theta1_dot_est)
est_theta2dot = Output('theta2_dot_est', theta2_dot_est)
est_xdot = Output('x_dot_est', x_dot_est)
est_theta1 = Output('theta1_est', theta1_est)
est_theta2 = Output('theta2_est', theta2_est)
est_x = Output('x_est', x_est)

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
                
        model_name = f'bb_fraction_{fraction * 100:.0f}_seed_{seed}'
        bb = Modely(visualizer=TextVisualizer(), workspace=workspace)
        
        bb.addModel('double_pend_learner', [est_theta1_dotdot, est_theta2_dotdot, est_x_dotdot, est_theta1dot, est_theta2dot, est_xdot, est_theta1, est_theta2, est_x])

        # Train on the dataset accelerations
        bb.addMinimize('mse_x_dotdot', a.last(), est_x_dotdot, loss_function='mse')
        bb.addMinimize('mse_th1_dotdot', alpha1.last(), est_theta1_dotdot, loss_function='mse')
        bb.addMinimize('mse_th2_dotdot', alpha2.last(), est_theta2_dotdot, loss_function='mse')

        # Neuralize the model
        bb.neuralizeModel(0.01)
        
        # Data loading
        bb.loadData(name='data', source = X_train, format=cols, delimiter=',', header=0)
        bb.loadData(name='val_data', source = X_val, format=cols, delimiter=',', header=0)
        bb.loadData(name='test_data', source=test_data, format=cols, delimiter=',', header=0)
        
        # Train the model
        prediction_samples = None
        step = None
        early_stop_patience = 20
        lr = 0.0010466265184194325
        params = {'train_batch_size': 256, 'num_of_epochs': 1000}
        
        bb.trainModel(train_dataset='data', validation_dataset='val_data', optimizer='Adam', prediction_samples=prediction_samples, step=step, training_params=params, early_stopping=earlystopping.early_stop_patience,  
                                                                    early_stopping_params={'patience':early_stop_patience}, select_model=earlystopping.select_best_model, lr=lr)
        
        bb.exportPythonModel(name=model_name)
        # analyzeModel stampa i risultati ma in nnodely non restituisce un dict (ritorna None); le metriche sono in performance.
        bb.analyzeModel('test_data')
        perf = bb.performance['test_data']
        test_loss = perf['total']['mean_error']
        train_loss = np.mean([metrics['train'][-1] for metrics in bb._training.values()])
        val_loss = np.mean([metrics['val'][-1] for metrics in bb._training.values()])
        df = pd.concat([df, pd.DataFrame({'model_name': [model_name], 'fraction': [fraction*100], 'seed': [seed], 'train_samples': [len(X_train)], 'val_samples': [len(X_val)], 'train_loss': [train_loss], 'val_loss': [val_loss], 'test_loss': [test_loss]})], ignore_index=True)
        df.to_csv(os.path.join(workspace, 'data_bb1.csv'), index=False)
        
# Analyze results
cols = ['time','action',('Xpos', 'int_x'),('Xth1', 'int_th1'),('Xth2', 'int_th2'),('Xvelocity', 'int_xdot'),('Xth1_dot', 'int_th1_dot'),('Xth2_dot', 'int_th2_dot'),'Xddx','Xddth1','Xddth2']
train_data = [100, 80, 70, 50, 40, 30, 25, 20, 15, 10, 5, 1]

df = pd.DataFrame(columns=['model_name', 'fraction', 'seed', 'total_mse', 'acc_cart_mse', 'th1_ddot_mse', 'th2_ddot_mse'])
for fraction in train_data:
    for seed in seeds:
        
        model_name = f'bb_fraction_{fraction}_seed_{seed}'
        bb = Modely(workspace=workspace, seed =seed)
        bb.importPythonModel(name=model_name)
        bb.neuralizeModel(0.01)
        bb.loadData(name='test_data', source=test_data, format=cols, delimiter=',', header=0)
        
        bb.analyzeModel('test_data', batch_size=128)
        perf = bb.performance['test_data']
        df = pd.concat([df, pd.DataFrame({'model_name': [model_name], 'fraction': [fraction], 'seed': [seed], 'total_mse': [perf['total']['mean_error']], 'acc_cart_mse': [perf['mse_x_dotdot']['mse']], 'th1_ddot_mse': [perf['mse_th1_dotdot']['mse']], 'th2_ddot_mse': [perf['mse_th2_dotdot']['mse']]})], ignore_index=True)
        