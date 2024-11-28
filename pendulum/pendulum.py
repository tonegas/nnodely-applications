import sys
import os
# append a new directory to sys.path
sys.path.append(os.getcwd())

from nnodely import *

# Create nnodely structure
workspace = os.path.join(os.getcwd(), "results")
pendolum = Modely(visualizer=MPLVisualizer(), workspace=workspace)

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

# Add the neural model to the nnodely structure and neuralization of the model
pendolum.addMinimize('omega error', omega.next(), out)
pendolum.addModel('pendulum',out)
pendolum.neuralizeModel(0.05)
#pendolum.exportJSON()

# Data load
data_struct = ['time','theta','omega','cos(theta)','sin(theta)','torque']
data_folder = os.path.join(os.path.dirname(os.path.realpath(__file__)),'dataset','data')
pendolum.loadData(name='data', source=data_folder, format=data_struct, delimiter=';')

# Neural network train
params = {'train_batch_size':32, 'num_of_epochs':50}
pendolum.trainModel(splits=[70,20,10], lr=0.001, training_params=params)

## Neural network Predict
sample = pendolum.getSamples(dataset='data', window=1)

pendolum.exportPythonModel()
pendolum.exportONNX(['theta','torque'],['omega_pred'])
pendolum.exportReport()

newNN = Modely(workspace=workspace)
newNN.importPythonModel()

result_old = pendolum(sample, sampled=True)
result = newNN(sample, sampled=True)
print(f"Predicted omega: {result['omega_pred']}")
print(f"Predicted omega from loaded network: {result['omega_pred']}")
print('True omega: ', sample['omega'])
