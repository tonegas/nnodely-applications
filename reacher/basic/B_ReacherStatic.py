

import sys
import os
sys.path.append(os.getcwd())
from nnodely import *

#  Neural model to estimate l1 and l2 

# Inputs from dataset
theta1 = Input('theta1')
theta2 = Input('theta2')
x_tip  = Input('x_tip')
y_tip  = Input('y_tip')

l1 = Parameter('l1')
l2 = Parameter('l2')


x_out = Output('x_out',(l1 * Cos(theta1.last())) + (l2 * Cos(theta1.last() + theta2.last())))
y_out = Output('y_out',(l1 * Sin(theta1.last())) + (l2 * Sin(theta1.last() + theta2.last())))



# Model container
model = Modely(seed=0)
model.addModel('x_out', x_out)
model.addModel('y_out', y_out)

# Objectives
model.addMinimize('x-error', x_tip.last(), x_out, 'mse')
model.addMinimize('y-error', y_tip.last(), y_out, 'mse')


#  sample_time consistent with CSV 
model.neuralizeModel(sample_time=0.02)

#  Data loading (CSV in current folder) 

#data_struct =  ['step', 'T1','T2','theta1', 'theta2', 'x_tip', 'y_tip']
data_struct =  ['step', 'T1','T2','theta1', 'theta2', 'x_tip', 'y_tip',
                'thetadot1', 'thetadot2', 'thetaddot1', 'thetaddot2']

data_folder = os.path.join(os.getcwd(), 'dataset', 'data')

# Load the CSV file
model.loadData(
    name='reacher_data',
    source=data_folder,
    format=data_struct,
    delimiter=';')
#skeeplines = 1

# Training 
train_params = {'num_of_epochs': 500, 'train_batch_size': 128, 'lr': 0.001}
model.trainModel(splits=[70, 20, 10], training_params=train_params)

model.neuralizeModel()


