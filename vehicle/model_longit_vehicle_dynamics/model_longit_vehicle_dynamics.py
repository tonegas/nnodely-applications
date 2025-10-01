import os
import numpy as np

from nnodely import *
from nnodely.support.earlystopping import select_best_model
from nnodely.support.jsonutils import plot_graphviz_structure

# Create nnodely structure
vehicle = nnodely(visualizer=MPLVisualizer())

# Dimensions of the layers
n  = 25
na = 21

#Create neural model inputs
velocity = Input('vel')
brake = Input('brk')
gear = Input('gear')
torque = Input('trq')
altitude = Input('alt',dimensions=na)
acc = Input('acc')

# Create neural network relations
air_drag_force = Linear(b=True)(velocity.last()**2)
breaking_force = -Relu(Fir(W_init = 'init_negexp', W_init_params={'size_index':0, 'first_value':0.002, 'lambda':3})(brake.sw(n)))
gravity_force = Linear(W_init='init_constant', W_init_params={'value':0}, dropout=0.1, W='gravity')(altitude.last())
fuzzi_gear = Fuzzify(6, range=[2,7], functions='Rectangular')(gear.last())
local_model = LocalModel(input_function=lambda: Fir(W_init = 'init_negexp', W_init_params={'size_index':0, 'first_value':0.002, 'lambda':3}))
engine_force = local_model(torque.sw(n), fuzzi_gear)

# Create neural network output
out = Output('accelleration', air_drag_force+breaking_force+gravity_force+engine_force)

# Add the neural model to the nnodely structure and neuralization of the model
vehicle.addModel('acc',[out])
vehicle.addMinimize('acc_error', acc.last(), out, loss_function='rmse')
vehicle.neuralizeModel(0.05)

# Load the training and the validation dataset
data_struct = ['vel','trq','brk','gear','alt','acc']
data_folder = os.path.join(os.path.dirname(os.path.realpath(__file__)),'dataset','trainingset')
vehicle.loadData(name='trainingset', source=data_folder, format=data_struct, skiplines=1)
data_folder = os.path.join(os.path.dirname(os.path.realpath(__file__)),'dataset','validationset')
vehicle.loadData(name='validationset', source=data_folder, format=data_struct, skiplines=1)

# Filter the data
def filter_function(sample):
   return np.all(sample['vel'] >= 1.).tolist()
vehicle.filterData(filter_function = filter_function, dataset_name = 'trainingset')

# Neural network train
optimizer_params = [{'params':'gravity','weight_decay': 0.1}]
optimizer_defaults = {'weight_decay': 0.00001}
training_params = {'num_of_epochs':150, 'val_batch_size':128, 'train_batch_size':128, 'lr':0.00003}
vehicle.trainModel(train_dataset='trainingset', validation_dataset='validationset', shuffle_data=True,
                   add_optimizer_params=optimizer_params, add_optimizer_defaults=optimizer_defaults, training_params=training_params,
                   select_model=select_best_model)

