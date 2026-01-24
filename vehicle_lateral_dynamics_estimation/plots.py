from nnodely import *
import numpy as np
import matplotlib.pyplot as plt

# Set up plot fonts
plt.rcParams.update({
    "text.usetex": False,
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times"],
    "font.size": 10
})


# Two nnodely instantiation for comparison
MSNN = nnodely(workspace="trained_models")
BBNN = nnodely(workspace="trained_models")

# Load trained models
MSNN.loadModel("MS_NN")
BBNN.loadModel("BB_NN")

MSNN.neuralizeModel()
BBNN.neuralizeModel()

# Load telemetries
MSNN.loadData(name='test_telem', source="telemetries/test", format=['','handwheelAngle','vxCG','axCG','ayCG',('yawAngle','yawAngle_int'),('yawRate','yawRate_int')],skiplines=1)

# Inference
prediction = 500   # intergal over 25s
window     = 2000
data_sampled = MSNN.getSamples('test_telem', window=window)
# Define the yaw angle as relative to initial condition
data_sampled['yawAngle_int'] = data_sampled['yawAngle'] - data_sampled['yawAngle'][0]

MSNN_out = MSNN(inputs=data_sampled,sampled=True,prediction_samples=prediction)
BBNN_out = BBNN(inputs=data_sampled,sampled=True,prediction_samples=prediction)
time = 0.05 * np.arange(window)

# Plotting
width = 7.14         # inches for IEEE column
aspect_ratio = 0.62  # height/width ratio

fig, ax = plt.subplots(2,1,figsize=(width, width * aspect_ratio))

ax[0].plot(time,np.array(data_sampled['yawAngle_int']).squeeze(),'k', label='Measured')
ax[0].plot(time,MSNN_out['yaw_angle'], '--', label='MS-NN')  
ax[0].plot(time,BBNN_out['yaw_angle'], '-.', label='NN')  
ax[0].set_ylabel(r"$\psi$ [deg]")
ax[0].tick_params(axis='x', labelbottom=False)
ax[0].grid()
ax[0].legend()
ax[0].set_xlim(0, 100)
ax[0].set_xticks([0, 25, 50, 75, 100])

# Yaw rate prediction plot
ax[1].plot(time,np.array(data_sampled['yawRate']).squeeze(),'k', label='measured')
ax[1].plot(time,MSNN_out['yaw_rate'], '--', label='MS-NN')  
ax[1].plot(time,BBNN_out['yaw_rate'], '-.', label='BB-NN')  
ax[1].set_xlabel(r"Time [s]")
ax[1].set_ylabel(r"$\Omega$ [deg/s]")
ax[1].grid()
ax[1].sharex(ax[0])


# Savefig
fig.savefig('Fig/results.pdf', bbox_inches='tight')