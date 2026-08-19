import numpy as np
import random
from nnodely import *
from nnodely.support.earlystopping import select_best_model
import torch

def initialize_model(path_folder, data_stats):

    seed = random.randint(0, 10000)
    print("Seed: ", seed)

    clearNames()

    # NNodely instantiation
    lateral_model = nnodely(visualizer=MPLNotebookVisualizer(),seed=seed,workspace=path_folder,save_history=False)

    # ------------------------------------------- #
    ###  BLACK-BOX NEURAL NETWORK MODEL  ###
    # ------------------------------------------- #

    # -----------------------------------------
    # Inputs
    # -----------------------------------------
    yaw_rate  = Input('yawRate')         # [rad/s] yaw rate
    yaw_angle = Input('yawAngle')        # [rad] yaw angle
    vx        = Input('vxCG')            # [m/s] longitudinal velocity
    steer     = Input('handwheelAngle')  # [rad] steering wheel angle
    ax        = Input('axCG')            # [m/s^2] longitudinal acceleration

    # -----------------------------------------
    # Hyperparameters
    # -----------------------------------------
    past_window = 10   # past samples 0.5s
    act_func    = ['Relu','Relu']  # activation function ['Relu', 'Tanh', 'Sigmoid']: applied to all hidden layers
    H = [8,5,1]   # n° of neurons per layer

    if len(H) != len(act_func)+1:
        raise ValueError("The number of activation functions and hidden layers does not match. Please check the hyperparameters.")

    # -------------------------------------------
    # Parametric Function: concatenate inputs
    # -------------------------------------------
    def concat(in1,in2,in3):
        out = torch.cat((in1,in2,in3),1)
        out = out.unsqueeze(1).squeeze(-1)
        return out

    # -----------------------------------------
    # Deep NN
    # -----------------------------------------
    vx_dyn =  vx.sw(past_window) 
    ax_dyn = ax.sw(past_window)
    steer_dyn = steer.sw(past_window) 

    nn_in = ParamFun(concat)(vx_dyn,ax_dyn,steer_dyn)

    layers = []
    layers.append(nn_in)

    for i,k in enumerate(H):
        if i < len(H)-1:
            if act_func[i] == 'Relu':
                layers.append(Relu(Linear(W=f"Linear{i}",b=True,output_dimension=k)(layers[i])))
            elif act_func[i] == 'Tanh':
                layers.append(Tanh(Linear(W=f"Linear{i}",b=True,output_dimension=k)(layers[i])))
            elif act_func[i] == 'Sigmoid':
                layers.append(Sigmoid(Linear(W=f"Linear{i}",b=True,output_dimension=k)(layers[i])))
        else:
            layers.append(Linear(W=f"Linear{i}",b=True,output_dimension=k)(layers[i]))

    nn_out = layers[len(H)]

    # -----------------------------------------
    # Outputs
    # -----------------------------------------
    yaw_rate_out = Output('yaw_rate',nn_out)                                                               # yaw rate [deg/s]    
    yaw_out      = Output('yaw_angle',Integrate(nn_out,int_name='yawAngle_int',
                                                der_name='yawRate_int',method='trapezoidal'))              # yaw angle [deg]

    # -----------------------------------------
    # Minimization and neuralization
    # -----------------------------------------
    lateral_model.addModel('lateral_model',[yaw_rate_out,yaw_out])

    lateral_model.addMinimize('yaw_rate_error',
                        yaw_rate.next(),
                        yaw_rate_out,
                        loss_function='mse')

    lateral_model.addMinimize('yaw_angle_error',
                        yaw_angle.next(),
                        yaw_out,
                        loss_function='mse')

    lateral_model.neuralizeModel(sample_time=0.05)

    return lateral_model


def train_model(lateral_model, path_train, path_val, i):
    
    # Load data
    lateral_model.loadData(name='training_telem',source=path_train,format=['','handwheelAngle','vxCG','axCG','ayCG',('yawAngle','yawAngle_int'),('yawRate','yawRate_int'),''],skiplines=1)
    lateral_model.loadData(name='valid_telem',   source=path_val,  format=['','handwheelAngle','vxCG','axCG','ayCG',('yawAngle','yawAngle_int'),('yawRate','yawRate_int'),''],skiplines=1)

    # -----------------------------------------
    # Training
    # -----------------------------------------
    num_epochs = 20
    batch_size = 256
    learn_rate = 1e-3
    early_stop_patience = 5 
    n_prediction = 500   # 25s of prediction
    training_pars = {'train_dataset':'training_telem',
                     'validation_dataset':'valid_telem',
                     'num_of_epochs':num_epochs, 
                     'val_batch_size':batch_size, 
                     'train_batch_size':batch_size, 
                     'lr':learn_rate,
                     'shuffle_data':True,
                     'optimizer':'Adam',
                     'select_model':select_best_model,
                     'prediction_samples':n_prediction,   # force to avoid prediction: intergal evaluated only during inference
                     'minimize_gain':{'yaw_rate_error':1.0, 'yaw_angle_error':0.1},
                     'early_stopping_params':{'patience':early_stop_patience,'error':'yaw_rate_error', 'threshold':0.05}}

    # First training
    lateral_model.trainModel(training_params=training_pars,lr=1e-3,num_of_epochs=500)
    # Refining
    lateral_model.trainModel(training_params=training_pars)

    lateral_model.saveModel('BB_NN_{:d}'.format(i))

    return lateral_model

def analyse_model(lateral_model, data_sampled):
    
    prediction = 500

    # Prepare data
    output = lateral_model(inputs=data_sampled, sampled=True, prediction_samples=prediction)

    RMSE_yaw_angle = np.sqrt(np.mean((np.array(output['yaw_angle'])- np.array(data_sampled['yawAngle'])[:,0,0])**2))
    RMSE_yaw_rate = np.sqrt(np.mean((np.array(output['yaw_rate']) - np.array(data_sampled['yawRate'])[:,0,0])**2))

    return RMSE_yaw_angle, RMSE_yaw_rate, output