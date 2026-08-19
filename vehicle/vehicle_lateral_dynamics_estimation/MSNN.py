import numpy as np
import random
from nnodely import *
from nnodely.support.earlystopping import select_best_model

def initialize_model(path_folder, data_stats):

    seed = random.randint(0, 10000)
    print("Seed: ", seed)

    clearNames()

    # Create model
    model = nnodely(visualizer='Standard',seed=seed,workspace=path_folder,save_history=False)

    # Inputs
    yaw_rate = Input('yawRate')         # [rad/s] yaw rate
    yaw_angle = Input('yawAngle')       # [deg] yaw angle
    vx       = Input('vxCG')            # [m/s] longitudinal velocity
    steer    = Input('handwheelAngle')  # [rad] steering wheel angle
    ax       = Input('axCG')            # [m/s^2] longitudinal acceleration

    # Hyperparameters
    past_window = 10   # past samples 0.5s

    # channels definition
    n_channels_vx    = 2
    n_channels_delta = 5
    n_channels_ax    = 2

    # -----------------------------------------
    # Fuzzy: 
    # -----------------------------------------
    # channels centers
    chan_centers_vx  = list(np.linspace(data_stats['vx_min'], data_stats['vx_max'], n_channels_vx))
    chan_centers_ax  = list(np.linspace(-np.fmin(-data_stats['ax_min'],data_stats['ax_max']), np.fmin(-data_stats['ax_min'],data_stats['ax_max']), n_channels_ax))
    chan_centers_delta = list(np.linspace(-data_stats['delta_max'], data_stats['delta_max'], n_channels_delta))

    # Define the equilibrium points values for linearisation
    vx_list, ax_list, delta_list = [],[],[]
    for i in range(n_channels_vx):
        vx_list.append(Constant('vx_'+str(i),values=chan_centers_vx[i]))
    for i in range(n_channels_ax):
        ax_list.append(Constant('ax_'+str(i),values=chan_centers_ax[i]))
    for i in range(n_channels_delta):
        delta_list.append(Constant('delta_'+str(i),values=chan_centers_delta[i]))

    # fuzzification
    fuzz_ss_delta   = Fuzzify(centers=chan_centers_delta,functions='Triangular')(steer.last())
    fuzz_ss_ax      = Fuzzify(centers=chan_centers_ax,functions='Triangular')(ax.last())
    fuzz_ss_vx      = Fuzzify(centers=chan_centers_vx,functions='Triangular')(vx.last())

    # Parametric Function: understeer correction
    def understeer_corr_local(delta,vx,ax,        # model inputs
                        delta_0,vx_0,ax_0,  # equilibrium points
                        k1,k2,k3,k4         # learnable parameters
                        ):
        return (k1 + k2*(delta-delta_0) + k3*(vx-vx_0) + k4*(ax-ax_0))

    understeer_corr = ParamFun(understeer_corr_local)

    def Generator_understeer_corr(idx_list):
        def understeer_corr_map(delta,vx,ax):
            idx_delta = idx_list[0]
            idx_speed = idx_list[1]
            idx_ax = idx_list[2]
            k1 = Parameter('k1_' + str(idx_delta) + str(idx_speed) + str(idx_ax),values=[[1e-5]])
            k2 = Parameter('k2_' + str(idx_delta) + str(idx_speed) + str(idx_ax), values=[[1e-5]])
            k3 = Parameter('k3_' + str(idx_delta) + str(idx_speed) + str(idx_ax), values=[[1e-5]])
            k4 = Parameter('k4_' + str(idx_delta) + str(idx_speed) + str(idx_ax), values=[[1e-5]])
            delta_0 = delta_list[idx_delta]
            vx_0    = vx_list[idx_speed]
            ax_0    = ax_list[idx_ax]
            return understeer_corr(delta,vx,ax,delta_0,vx_0,ax_0,k1,k2,k3,k4)
        return understeer_corr_map

    # FIR filter
    steer_dyn = Fir(W='Fir_steer', W_init='init_negexp')(steer.sw(past_window))
    vx_dyn    = Fir(W='Fir_vx', W_init='init_negexp')(vx.sw(past_window))
    ax_dyn    = Fir(W='Fir_ax', W_init='init_negexp')(ax.sw(past_window))

    # Local Model
    yaw_rate_est = LocalModel(input_function = Generator_understeer_corr,pass_indexes=True)((steer_dyn,vx_dyn,ax_dyn),
                                                                                          (fuzz_ss_delta,fuzz_ss_vx,fuzz_ss_ax))
    
    # Outputs
    yaw_rate_out = Output('yaw_rate',yaw_rate_est)                                              # yaw rate [deg/s]    
    yaw_out      = Output('yaw_angle',Integrate(yaw_rate_est,int_name='yawAngle_int',
                                                der_name='yawRate_int',method='trapezoidal'))       # yaw angle [deg]

    # Minimization and neuralization
    model.addModel('model',[yaw_rate_out,yaw_out])

    model.addMinimize('yaw_rate_error',
                        yaw_rate.next(),
                        yaw_rate_out,
                        loss_function='mse')
    model.addMinimize('yaw_angle_error',
                        yaw_angle.next(),
                        yaw_out,
                        loss_function='mse')

    model.neuralizeModel(sample_time=0.05)

    return model

def train_model(model, path_train, path_val, i):
    
    # Load data
    model.loadData(name='training_telem',source=path_train,format=['','handwheelAngle','vxCG','axCG','ayCG',('yawAngle','yawAngle_int'),('yawRate','yawRate_int'),''],skiplines=1)
    model.loadData(name='valid_telem',   source=path_val,  format=['','handwheelAngle','vxCG','axCG','ayCG',('yawAngle','yawAngle_int'),('yawRate','yawRate_int'),''],skiplines=1)

    # Training
    # -----------------------------------------
    num_epochs = 10
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
                     'prediction_samples':n_prediction, 
                     'step':int(0.5*n_prediction),
                     'minimize_gain':{'yaw_rate_error':1.0, 'yaw_angle_error':0.1},
                     'early_stopping_params':{'patience':early_stop_patience,'error':'yaw_rate_error', 'threshold':0.05}}

    # High-learning rate pre-training
    model.trainModel(training_params=training_pars,num_of_epochs=50,lr=1e-1)
    # Fine-tuning train
    model.trainModel(training_params=training_pars)

    model.saveModel('MS_NN_{:d}'.format(i))

    return model

def analyse_model(model, data_sampled):
    
    prediction = 500

    # Prepare data
    output = model(inputs=data_sampled, sampled=True, prediction_samples=prediction)

    RMSE_yaw_angle = np.sqrt(np.mean((np.array(output['yaw_angle'])- np.array(data_sampled['yawAngle'])[:,0,0])**2))
    RMSE_yaw_rate = np.sqrt(np.mean((np.array(output['yaw_rate']) - np.array(data_sampled['yawRate'])[:,0,0])**2))

    return RMSE_yaw_angle, RMSE_yaw_rate, output