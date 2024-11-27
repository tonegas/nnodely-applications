
import sys
import os
# append a new directory to sys.path
sys.path.append(os.getcwd())

from torch.fx import symbolic_trace

from neu4mes import *
from neu4mes.visualizer import MPLVisualizer

# Create neu4mes structure
workspace = os.path.join(os.getcwd(), "results")
pendolum = Neu4mes(visualizer=MPLVisualizer(), workspace=workspace)

# Create neural model
# Input of the neural model
theta = Input('theta')
T     = Input('torque')
omega = Input('omega')

# Relations of the neural model
gravity_force = Fir(Sin(theta.tw(0.5)))
friction = Fir(theta.tw(0.5))
torque = Fir(T.last())
out = Output('omega_pred', gravity_force+friction+torque)

# Add the neural model to the neu4mes structure and neuralization of the model
pendolum.addMinimize('omega error', omega.next(), out)
pendolum.addModel('pendulum',out)
pendolum.neuralizeModel(0.05)
#pendolum.exportJSON()

# Data load
data_struct = ['time','theta','omega','cos(theta)','sin(theta)','torque']
data_folder = './tutorials/datasets/pendulum/data/'
pendolum.loadData(name='pendulum_dataset', source=data_folder, format=data_struct, delimiter=';')

# Neural network train
params = {'train_batch_size':32, 'val_batch_size':32, 'num_of_epochs':100}
pendolum.trainModel(splits=[70,20,10], lr=0.001, training_params=params)

## Neural network Predict
sample = pendolum.getSamples(dataset='pendulum_dataset', window=1)
result = pendolum(sample, sampled=True)

print('Predicted omega: ', result['omega_pred'])
print('True omega: ', sample['omega'])

#file_name, _, _ = pendolum.exportTracer()
#pendolum.importTracer(file_path=file_name)
#result = pendolum(sample, sampled=True)

print('Predicted omega: ', result['omega_pred'])
print('True omega: ', sample['omega'])

## Test import 

#test = Neu4mes()

#sample = {'theta':[0,1,2,3,4,5,6,7,8,9], 'torque':[1.0], 'omega_target':[1.0]}
#onnx_path = os.path.join(os.getcwd(), 'results','neu4mes_2024_09_11_11_00','tracer_model.onnx')
#test.import_onnx(onnx_path, data=sample)
