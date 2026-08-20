import os
import pandas as pd

import numpy as np
import matplotlib.pyplot as plt
from nnodely import *
from nnodely.support.earlystopping import select_best_model
from pathlib import Path

# FLAGS:
train = False
analyse = True

# Models folder
path_folder = "trained_models"

# telemetries folder
path_train = "telemetries/training"
path_val   = "telemetries/validation"
path_test  = "telemetries/test"

# Import parameters:
vehicle_data = pd.read_csv("parameters/params.csv")
data_stats   = pd.read_csv("parameters/data_info.csv")
data_stats   = data_stats.iloc[0]

N_MODELS = 10

# Inport custom functions
from MSNN import analyse_model, initialize_model, train_model

if train == True:
    for i in range(N_MODELS):   

        # Initialize model
        model = initialize_model(path_folder, data_stats)   

        # Train model
        model = train_model(model, path_train, path_val, i)

if analyse == True:

    # Instance:
    model = nnodely(visualizer=None,seed=0,workspace=path_folder,save_history=False)

    rmse_yawrate_list = []
    rmse_yaw_list = []

    for i in range(N_MODELS):   

        # Load model
        clearNames()
        model.loadModel("MS_NN_"+str(i))

        model.neuralizeModel()
        if i == 0:
            model.loadData(name='test_telem', source=path_test,  format=['','handwheelAngle','vxCG','axCG','ayCG',('yawAngle','yawAngle_int'),('yawRate','yawRate_int'),''],skiplines=1)
            samples = 4000
            data_sampled = model.getSamples('test_telem', window = samples)


            # Analyse model
            yaw_angle_rmse, yaw_rate_rmse, output = analyse_model(model, data_sampled)
            plt.figure(figsize=(12, 6))
            plt.subplot(2, 1, 1)
            plt.plot(output['yaw_angle'], label='Predicted Yaw Angle')
            plt.plot(np.array(data_sampled['yawAngle'])[:,0,0], label='True Yaw Angle')
            plt.title('Yaw Angle Prediction vs True Values')
            plt.xlabel('Sample Index')
            plt.ylabel('Yaw Angle (rad)')
            plt.legend()
            plt.subplot(2, 1, 2)
            plt.plot(output['yaw_rate'], label='Predicted Yaw Rate')
            plt.plot(np.array(data_sampled['yawRate'])[:,0,0], label='True Yaw Rate')
            plt.title('Yaw Rate Prediction vs True Values')
            plt.xlabel('Sample Index')
            plt.ylabel('Yaw Rate (rad/s)')
            plt.legend()
            plt.tight_layout()
            plt.savefig("Fig/MSNN_prediction.pdf")

        else:
            # Analyse model
            yaw_angle_rmse, yaw_rate_rmse, _ = analyse_model(model, data_sampled)
        rmse_yaw_list.append(yaw_angle_rmse)
        rmse_yawrate_list.append(yaw_rate_rmse)

        # print(rmse_yaw_list)

    yaw_rmse = np.mean(rmse_yaw_list)
    yaw_rmse_std = np.std(rmse_yaw_list) / np.sqrt(N_MODELS) # Standard error of the mean
    yawrate_rmse = np.mean(rmse_yawrate_list)
    yawrate_rmse_std = np.std(rmse_yawrate_list) / np.sqrt(N_MODELS) # Standard error of the mean

    total_rmse_list = 0.1*np.array(rmse_yaw_list) + np.array(rmse_yawrate_list)
    total_rmse = np.mean(total_rmse_list)
    total_rmse_std = np.std(total_rmse_list) / np.sqrt(N_MODELS) # Standard error of the mean

    print("Yaw angle RMSE: ")
    print(f"{yaw_rmse:.3e} ± {yaw_rmse_std:.3e}")
    print("Yaw rate RMSE: ")
    print(f"{yawrate_rmse:.3e} ± {yawrate_rmse_std:.3e}")
    print("Total RMSE:")
    print(f"{total_rmse:.3e}")
    print("Total RMSE standard deviation:")
    print(f"{total_rmse_std:.3e}")