import sys
import os
# append a new directory to sys.path
sys.path.append(os.getcwd())

from nnodely import *

# This example shows how to fit a simple linear model.
# The model chosen is a mass spring damper.
# The data was created previously and loaded from file.
# The data are the position/velocity of the mass and the force applied.
# The neural model mirrors the structure of the physical model.
# The network build estimate the future position of the mass and the velocity.

# Create neural model
# List the input of the model
x = Input('x')
F = Input('F')

# List the output of the model
x_z_est = Output('x_z_est', Fir(x.tw(0.25)) + Fir(F.last()))

# Add the neural models to the nnodely structure
mass_spring_damper = Modely(seed=0)
mass_spring_damper.addModel('x_z_est', x_z_est)

# These functions are used to impose the minimization objectives.
# Here it is minimized the error between the future position of x get from the dataset x_r.z(-1)
# and the estimator designed useing the neural network. The miniminzation is imposed via MSE error.
x_r = Input('x_r')
mass_spring_damper.addMinimize('next-pos', x_r.z(-1), x_z_est, 'mse')

# Nauralize the model and gatting the neural network. The sampling time depends on the datasets.
mass_spring_damper.neuralizeModel(sample_time = 0.05) # The sampling time depends on the dataset

# Data load
data_struct = ['time', ('x','x_r'), 'dx', 'F']
data_folder = os.path.join(os.path.dirname(os.path.realpath(__file__)),'dataset','data')
mass_spring_damper.loadData(name='mass_spring_dataset', source=data_folder, format=data_struct, delimiter=';')

#Neural network train not reccurent training
params = {'num_of_epochs': 50,
          'train_batch_size': 128,
          'lr':0.001}
mass_spring_damper.trainAndAnalyze(splits=[70,20,10], training_params = params)

# Inference
sample = {'F':[0.5], 'x':[0.25, 0.26, 0.27, 0.28, 0.29]}
results = mass_spring_damper(sample)
print(results)

# Add visualizer and show the results on the loaded dataset
vis = MPLVisualizer()
vis.setModely(mass_spring_damper)
vis.showResult("mass_spring_dataset_val")

