from nnodely import nnodely, Input, Output, ParamFun, LocalModel, Fuzzify, Concatenate, Select, Integrate, TextVisualizer, Parameter, Linear, Tanh, Relu
from nnodely.support.earlystopping import select_best_model
from nnodely.support import earlystopping
from nnodely.support.jsonutils import plot_graphviz_structure
import torch
import os

torch.set_default_device('cpu')
torch.set_default_dtype(torch.float32)


DATASET_NAME              = "dataset"
PLOT_VIZ                  = False
SEED                      = 345

ROOT                      = os.path.dirname(os.path.abspath(__file__))
WS                        = os.path.join(ROOT, 'model')
TRAIN_DATASET_FOLDER      = os.path.join(ROOT, DATASET_NAME, "train")
VALIDATION_DATASET_FOLDER = os.path.join(ROOT, DATASET_NAME, "val")
TEST_DATASET_FOLDER       = os.path.join(ROOT, DATASET_NAME, "test")

if not os.path.exists(WS):
    os.makedirs(WS)

z1_model = nnodely(visualizer=TextVisualizer(), seed=SEED, workspace=WS)

def friction(x, K_v, K_c, K_s, q_s):
    import torch 
    sign = torch.tanh(50 * x)   

    friction = (
        K_v * x
        + sign * (
            K_c
            + (K_s - K_c) * torch.exp(-(torch.abs(x) / q_s)**2)
        )
    )

    return friction

def ddq(q, dq, tau):
    import torch
    from adam.pytorch.computation_batch import KinDynComputationsBatch
    torch.set_default_device('cpu')
    torch.set_default_dtype(torch.float32)
    robot = '/home/dema/projects/friction_z1/z1_description.urdf'  

    joints_name_list = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']
    kinDyn = KinDynComputationsBatch(robot, joints_name_list, device='cpu', dtype=torch.float32)
    q = q.squeeze(1)
    dq = dq.squeeze(1)
    tau = tau.squeeze(1)
    tau_full = torch.concat((torch.zeros(tau.shape[0], 6), tau), dim=1)
    
    B = q.shape[0]

    H_b = torch.eye(4).expand(B, 4, 4)   
    v_b = torch.zeros(B, 6)       
    M = kinDyn.mass_matrix(H_b, q)
    h = kinDyn.bias_force(H_b, q, v_b, dq)
    t = tau_full - h

    ddq_full = (torch.linalg.inv(M[:,6:,6:]) @ t[:,6:].unsqueeze(-1)).squeeze(-1)
    ddq_full = ddq_full.unsqueeze(1)

    return ddq_full

nq = 6

# Define inputs
q1          = Input('q1')
q2          = Input('q2')
q3          = Input('q3')
q4          = Input('q4')
q5          = Input('q5')
q6          = Input('q6')
dq1         = Input('dq1')
dq2         = Input('dq2')
dq3         = Input('dq3')
dq4         = Input('dq4')
dq5         = Input('dq5')
dq6         = Input('dq6')
tau1        = Input('tau1')
tau2        = Input('tau2')
tau3        = Input('tau3')
tau4        = Input('tau4')
tau5        = Input('tau5')
tau6        = Input('tau6')
ddq1_target = Input('ddq1')
ddq2_target = Input('ddq2')
ddq3_target = Input('ddq3')
ddq4_target = Input('ddq4')
ddq5_target = Input('ddq5')
ddq6_target = Input('ddq6')

coul_visc_fric1 = ParamFun(friction)
coul_visc_fric2 = ParamFun(friction)
coul_visc_fric3 = ParamFun(friction)
coul_visc_fric4 = ParamFun(friction)
coul_visc_fric5 = ParamFun(friction)
coul_visc_fric6 = ParamFun(friction)

q = Concatenate(q1.last(), q2.last())
q = Concatenate(q,         q3.last())
q = Concatenate(q,         q4.last())
q = Concatenate(q,         q5.last())
q = Concatenate(q,         q6.last())

tau1_in = tau1.last() - coul_visc_fric1(dq1.last())
tau2_in = tau2.last() - coul_visc_fric2(dq2.last())
tau3_in = tau3.last() - coul_visc_fric3(dq3.last())
tau4_in = tau4.last() - coul_visc_fric4(dq4.last())
tau5_in = tau5.last() - coul_visc_fric5(dq5.last())
tau6_in = tau6.last() - coul_visc_fric6(dq6.last()) 

tau_final = Concatenate(tau1_in,     tau2_in)
tau_final = Concatenate(tau_final,   tau3_in)
tau_final = Concatenate(tau_final,   tau4_in)
tau_final = Concatenate(tau_final,   tau5_in)
tau_final = Concatenate(tau_final,   tau6_in)


dq = Concatenate(dq1.last(), dq2.last())
dq = Concatenate(dq,         dq3.last())   
dq = Concatenate(dq,         dq4.last())
dq = Concatenate(dq,         dq5.last())
dq = Concatenate(dq,         dq6.last())

ddq_full = ParamFun(ddq)(q, dq, tau_final)

ddq_out1 = Select(ddq_full, 0)
ddq_out2 = Select(ddq_full, 1)
ddq_out3 = Select(ddq_full, 2)
ddq_out4 = Select(ddq_full, 3)
ddq_out5 = Select(ddq_full, 4)
ddq_out6 = Select(ddq_full, 5)

dq_int_1 = Integrate(ddq_out1, int_name='dq1_int', method='euler')
dq_int_2 = Integrate(ddq_out2, int_name='dq2_int', method='euler')
dq_int_3 = Integrate(ddq_out3, int_name='dq3_int', method='euler')
dq_int_4 = Integrate(ddq_out4, int_name='dq4_int', method='euler')
dq_int_5 = Integrate(ddq_out5, int_name='dq5_int', method='euler')
dq_int_6 = Integrate(ddq_out6, int_name='dq6_int', method='euler')

q_int_1 = Integrate(dq_int_1, int_name='q1_int', method='euler')
q_int_2 = Integrate(dq_int_2, int_name='q2_int', method='euler')
q_int_3 = Integrate(dq_int_3, int_name='q3_int', method='euler')
q_int_4 = Integrate(dq_int_4, int_name='q4_int', method='euler')
q_int_5 = Integrate(dq_int_5, int_name='q5_int', method='euler')
q_int_6 = Integrate(dq_int_6, int_name='q6_int', method='euler')

full_dq = Concatenate(dq_int_1, dq_int_2)
full_dq = Concatenate(full_dq,   dq_int_3)
full_dq = Concatenate(full_dq,   dq_int_4)
full_dq = Concatenate(full_dq,   dq_int_5)
full_dq = Concatenate(full_dq,   dq_int_6)

full_q = Concatenate(q_int_1, q_int_2)
full_q = Concatenate(full_q,   q_int_3)
full_q = Concatenate(full_q,   q_int_4)
full_q = Concatenate(full_q,   q_int_5)
full_q = Concatenate(full_q,   q_int_6)

ddq_out = Output('ddq_out', ddq_full)
dq_out  = Output('dq_out', full_dq)
q_out   = Output('q_out', full_q)

q_target = Concatenate(q1.next(), q2.next())
q_target = Concatenate(q_target,  q3.next())
q_target = Concatenate(q_target,  q4.next())
q_target = Concatenate(q_target,  q5.next())
q_target = Concatenate(q_target,  q6.next())

dq_target = Concatenate(dq1.next(), dq2.next())
dq_target = Concatenate(dq_target,  dq3.next())
dq_target = Concatenate(dq_target,  dq4.next())
dq_target = Concatenate(dq_target,  dq5.next())
dq_target = Concatenate(dq_target,  dq6.next())

ddq_target = Concatenate(ddq1_target.last(), ddq2_target.last())
ddq_target = Concatenate(ddq_target, ddq3_target.last())
ddq_target = Concatenate(ddq_target, ddq4_target.last())
ddq_target = Concatenate(ddq_target, ddq5_target.last())
ddq_target = Concatenate(ddq_target, ddq6_target.last())

# Build the model
z1_model.addModel('friction', [q_out, dq_out, ddq_out])

z1_model.addMinimize('ddq_error', ddq_out, ddq_target, loss_function='mse')
z1_model.addMinimize('dq_error', dq_out, dq_target, loss_function='mse')
z1_model.addMinimize('q_error', q_out, q_target, loss_function='mse')

z1_model.neuralizeModel(0.01)

# Visualize the model structure
if PLOT_VIZ:
    plot_graphviz_structure(z1_model.json, 'z1_model_structure')

## LOAD DATA
data_struct = ['time', ('q1', 'q1_int'), ('q2', 'q2_int'), ('q3', 'q3_int'), ('q4', 'q4_int'), ('q5', 'q5_int'), ('q6', 'q6_int'),
               ('dq1', 'dq1_int'), ('dq2', 'dq2_int'), ('dq3', 'dq3_int'), ('dq4', 'dq4_int'), ('dq5', 'dq5_int'), ('dq6', 'dq6_int'),
               'ddq1', 'ddq2', 'ddq3', 'ddq4', 'ddq5', 'ddq6', 
               'tau1', 'tau2', 'tau3', 'tau4', 'tau5', 'tau6']

train_dataset_name = "train_data"
validation_dataset_name = "validation_data"
test_dataset_name  = "test_data"

z1_model.loadData(name=train_dataset_name, source=TRAIN_DATASET_FOLDER, format=data_struct, skiplines=1)
z1_model.loadData(name=validation_dataset_name, source=VALIDATION_DATASET_FOLDER, format=data_struct, skiplines=1)
z1_model.loadData(name=test_dataset_name, source=TEST_DATASET_FOLDER, format=data_struct, skiplines=1)

# TRAIN THE MODEL

params = {
    'num_of_epochs': 200,
    'train_batch_size': 256,
    'val_batch_size': 256,
    'test_batch_size': 256,
    'train_dataset': train_dataset_name,
    # 'validation_dataset': validation_dataset_name,
    # 'test_dataset': test_dataset_name,
    'lr': 0.1,
    'shuffle_data': True,
    # 'splits': [80, 20, 0],
    'select_model': select_best_model,
    'prediction_samples': -1,
    'step': -1,
    'early_stopping': earlystopping.early_stop_patience,
    'early_stopping_params': {
        'patience': 5,
        'error': 'ddq_error',
    },
    'minimize_gain' : {
        # 'ddq_error': 1.0,
        # 'dq_error': 0.5,
        'q_error': 1
    }
}

z1_model.trainModel(training_params=params)

params = {
    'num_of_epochs': 200,
    'train_batch_size': 256,
    'val_batch_size': 256,
    'test_batch_size': 256,
    'train_dataset': train_dataset_name,
    # 'validation_dataset': validation_dataset_name,
    # 'test_dataset': test_dataset_name,
    'lr': 0.001,
    'shuffle_data': True,
    # 'splits': [80, 20, 0],
    'select_model': select_best_model,
    'prediction_samples': -1,
    'step': -1,
    'early_stopping': earlystopping.early_stop_patience,
    'early_stopping_params': {
        'patience': 5,
        'error': 'q_error',
    },
    'minimize_gain' : {
        # 'ddq_error': 1.0,
        # 'dq_error': 0.5,
        'q_error': 1
    }
}

z1_model.trainModel(training_params=params)

params = {
    'num_of_epochs': 100,
    'train_batch_size': 256,
    'val_batch_size': 256,
    'test_batch_size': 256,
    'train_dataset': train_dataset_name,
    # 'validation_dataset': validation_dataset_name,
    # 'test_dataset': test_dataset_name,
    'lr': 0.001,
    'shuffle_data': True,
    # 'splits': [80, 20, 0],
    'select_model': select_best_model,
    'prediction_samples': 100,
    'step': 100,
    'early_stopping': earlystopping.early_stop_patience,
    'early_stopping_params': {
        'patience': 5,
        'error': 'q_error',
    },
    'minimize_gain' : {
        # 'ddq_error': 1.0,
        # 'dq_error': 0.5,
        'q_error': 1.0
    }
}

z1_model.trainModel(training_params=params)

z1_model.saveModel("friction_comp_sin_regressionv2")