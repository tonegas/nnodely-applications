import os
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
import time
import json

# ------------------------------------------------------------
# Optimize the hyperparameters of the proposed MS-NN-steer 
# steering controller with a grid search
# ------------------------------------------------------------

# Set the environment variable to avoid the Tensorflow warning message about the CPU instructions
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

# Change the current working directory to the one of this script
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Type of training to be performed 

from config_sequential_training import num_samples_future_feedfw
from config_sequential_training import learning_rate
from config_sequential_training import seed
from config_sequential_training import num_channels_vx_list
from config_sequential_training import num_channels_ax_list

# Names of the NN models to be trained
NN_model_names = ['control_steer_dynamics_A2RL_sequential']    

# Start the timer
start_time = time.time()

# Train the inverse neural models sequentially
for idx_num_channel_ay in range(len(num_samples_future_feedfw)):
    num_samples_future_feedfw = num_samples_future_feedfw[idx_num_channel_ay]
    for idx_num_seed in range(len(seed)):
        num_seed = seed[idx_num_seed]
        for idx_num_channel_vx in range(len(num_channels_vx_list)):
            num_channels_vx = num_channels_vx_list[idx_num_channel_vx]
            for idx_learning_rate in range(len(learning_rate)):
                num_learning_rate = learning_rate[idx_learning_rate]
                for idx_num_channel_ax in range(len(num_channels_ax_list)):
                    num_channels_ax = num_channels_ax_list[idx_num_channel_ax]
                    with open('./NN_parameters.json', 'w') as f:
                        f.write('{"num_samples_future_feedfw": ' + str(num_samples_future_feedfw) + ', "num_channels_vx": ' + str(num_channels_vx) + ', "num_channels_ax": ' + str(num_channels_ax) + ', "seed": ' + str(num_seed) + ', "learning_rate": ' + str(num_learning_rate) + '}')
                    for NN_model_name in NN_model_names:
                        print('\nTraining the inverse neural model ' + NN_model_name + ' with ' + str(num_seed) + \
                            ' seed and ' + str(num_samples_future_feedfw) + ' channels for ay '  + str(num_channels_vx) + ' channels for vx '+ str(num_channels_ax) + ' channels for ax ' + str(num_learning_rate) + ' learning rate ... \n')                   
                        filename = './' + NN_model_name + '.ipynb'
                        with open(filename) as ff:
                            nb_in = nbformat.read(ff, nbformat.NO_CONVERT)
                    
                        ep = ExecutePreprocessor(kernel_name='python3')
                        nb_out = ep.preprocess(nb_in, {'metadata': {'path': './'}})
                        print('Neural model trained correctly\n')

# Print the total training time
print('Total training time: %s seconds' % (time.time() - start_time))

print('All done folks!\n')
