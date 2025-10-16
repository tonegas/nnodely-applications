import torch

def nnodely_basic_model_connect(data_in, rel):
    virtual = torch.cat((data_in[:, 1:, :], data_in[:, :1, :]), dim=1)
    max_dim = min(rel.size(1), data_in.size(1))
    virtual[:, -max_dim:, :] = rel[:, -max_dim:, :]
    return virtual

def nnodely_layers_fuzzify_slicing(res, i, x):
    res[:, :, i:i+1] = x

def nnodely_layers_parametricfunction_long_corr_lat_local(accy,accy_0,  # inputs
                    k1,k2          # learnable parameter 
                  ):
  return k1+ k2*(accy - accy_0)

def nnodely_layers_parametricfunction_acc_pos(T):
    mask_pos = torch.gt(T,0)
    return torch.mul(T,mask_pos)

def nnodely_layers_parametricfunction_acc_neg(T):
    mask_pos = torch.gt(T,0)
    return torch.mul(T,~mask_pos)

def nnodely_layers_parametricfunction_acc_model_based(Ty,v,F_y,delta,
                    r1,mass,Kd,Cv,Cr,Iw1):  # learnable parameter
    # function inputs:
    # Ty,v --> wheel torques and vehicle speed

    # learnable parameters:
    # r1,mass,Kd,Cv,Cr,Iw1 --> wheel radius,vehicle mass, aero drag, linear drag and rolling resistance coefficients, wheel inertia

    # non-trainable parameters
    g_acc     = 9.81       # [m/s^2] gravity acceleration

    # function output: longitudinal acceleration, computed using the Newton's vehicle dynamics laws 
    return ((1.0/mass)*( 2*(Ty/r1) - Kd * v**2 - Cv * v - F_y*torch.sin(delta)) - Cr*g_acc)/(1.0 + (2.0/mass)*(2*(Iw1/r1**2.0)))

def nnodely_layers_parametricfunction_understeer_corr_local(input,vx,input_0, vx_0,ax,ax_0,  # inputs
                    k1,k2,k3,k4          # learnable parameter 
                  ):
  return vx*(k1+ k2*(input - input_0) + k3* (vx-vx_0)+k4*(ax-ax_0)) 

class TracerModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.all_parameters = {}
        self.all_constants = {}
        self._tensor_constant100 = torch.tensor(0.0)
        self._tensor_constant101 = torch.tensor(1)
        self._tensor_constant102 = torch.tensor(0.0)
        self._tensor_constant103 = torch.tensor(1.0)
        self._tensor_constant104 = torch.tensor(2)
        self._tensor_constant105 = torch.tensor(0.0)
        self._tensor_constant106 = torch.tensor(1.0)
        self._tensor_constant107 = torch.tensor(0)
        self._tensor_constant108 = torch.tensor(0.0)
        self._tensor_constant109 = torch.tensor(0.0)
        self._tensor_constant110 = torch.tensor(1)
        self._tensor_constant111 = torch.tensor(0.0)
        self._tensor_constant112 = torch.tensor(1.0)
        self._tensor_constant113 = torch.tensor(2)
        self._tensor_constant114 = torch.tensor(0.0)
        self._tensor_constant115 = torch.tensor(1.0)
        self._tensor_constant116 = torch.tensor(0)
        self._tensor_constant117 = torch.tensor(0.0)
        self._tensor_constant118 = torch.tensor(0.0)
        self._tensor_constant119 = torch.tensor(1)
        self._tensor_constant120 = torch.tensor(0.0)
        self._tensor_constant121 = torch.tensor(0.0)
        self._tensor_constant122 = torch.tensor(2)
        self._tensor_constant123 = torch.tensor(0.0)
        self._tensor_constant124 = torch.tensor(0.0)
        self._tensor_constant125 = torch.tensor(3)
        self._tensor_constant126 = torch.tensor(0.0)
        self._tensor_constant127 = torch.tensor(1.0)
        self._tensor_constant128 = torch.tensor(4)
        self._tensor_constant129 = torch.tensor(0.0)
        self._tensor_constant130 = torch.tensor(1.0)
        self._tensor_constant131 = torch.tensor(0)
        self._tensor_constant132 = torch.tensor(0.0)
        self._tensor_constant133 = torch.tensor(0.0)
        self._tensor_constant134 = torch.tensor(1)
        self._tensor_constant135 = torch.tensor(0.0)
        self._tensor_constant136 = torch.tensor(1.0)
        self._tensor_constant137 = torch.tensor(2)
        self._tensor_constant69 = torch.tensor(0.0)
        self._tensor_constant70 = torch.tensor(1.0)
        self._tensor_constant71 = torch.tensor(0)
        self._tensor_constant72 = torch.tensor(0.0)
        self._tensor_constant73 = torch.tensor(1.0)
        self._tensor_constant74 = torch.tensor(1)
        self._tensor_constant75 = torch.tensor(0.0)
        self._tensor_constant76 = torch.tensor(1.0)
        self._tensor_constant77 = torch.tensor(0)
        self._tensor_constant78 = torch.tensor(0.0)
        self._tensor_constant79 = torch.tensor(0.0)
        self._tensor_constant80 = torch.tensor(1)
        self._tensor_constant81 = torch.tensor(0.0)
        self._tensor_constant82 = torch.tensor(0.0)
        self._tensor_constant83 = torch.tensor(2)
        self._tensor_constant84 = torch.tensor(0.0)
        self._tensor_constant85 = torch.tensor(1.0)
        self._tensor_constant86 = torch.tensor(3)
        self._tensor_constant87 = torch.tensor(0.0)
        self._tensor_constant88 = torch.tensor(1.0)
        self._tensor_constant89 = torch.tensor(0)
        self._tensor_constant90 = torch.tensor(0.0)
        self._tensor_constant91 = torch.tensor(0.0)
        self._tensor_constant92 = torch.tensor(1)
        self._tensor_constant93 = torch.tensor(0.0)
        self._tensor_constant94 = torch.tensor(1.0)
        self._tensor_constant95 = torch.tensor(2)
        self._tensor_constant96 = torch.tensor(0.0)
        self._tensor_constant97 = torch.tensor(1.0)
        self._tensor_constant98 = torch.tensor(0)
        self._tensor_constant99 = torch.tensor(0.0)
        self.all_constants["ax_center_0"] = torch.tensor([-4.772998809814453], requires_grad=False)
        self.all_constants["ax_center_1"] = torch.tensor([0.49301645159721375], requires_grad=False)
        self.all_constants["ax_center_2"] = torch.tensor([5.759031772613525], requires_grad=False)
        self.all_constants["ay_center_0"] = torch.tensor([-3.747269868850708], requires_grad=False)
        self.all_constants["ay_center_1"] = torch.tensor([-0.28388628363609314], requires_grad=False)
        self.all_constants["ay_center_2"] = torch.tensor([3.179497241973877], requires_grad=False)
        self.all_constants["ay_center_3"] = torch.tensor([6.642880916595459], requires_grad=False)
        self.all_constants["steer_center_0"] = torch.tensor([-0.26957762241363525], requires_grad=False)
        self.all_constants["steer_center_1"] = torch.tensor([-0.09745820611715317], requires_grad=False)
        self.all_constants["steer_center_2"] = torch.tensor([0.07466120272874832], requires_grad=False)
        self.all_constants["steer_center_3"] = torch.tensor([0.2467806041240692], requires_grad=False)
        self.all_constants["steer_center_4"] = torch.tensor([0.4189000129699707], requires_grad=False)
        self.all_constants["vx_center_0"] = torch.tensor([-0.3245462477207184], requires_grad=False)
        self.all_constants["vx_center_1"] = torch.tensor([1.6398653984069824], requires_grad=False)
        self.all_constants["vx_center_2"] = torch.tensor([3.6042771339416504], requires_grad=False)
        self.all_parameters["Cr"] = torch.nn.Parameter(torch.tensor([[9.999999747378752e-05]]), requires_grad=True)
        self.all_parameters["Cv"] = torch.nn.Parameter(torch.tensor([[9.999999747378752e-05]]), requires_grad=True)
        self.all_parameters["Iw1"] = torch.nn.Parameter(torch.tensor([[9.999999747378752e-05]]), requires_grad=True)
        self.all_parameters["Kd"] = torch.nn.Parameter(torch.tensor([[9.999999747378752e-05]]), requires_grad=True)
        self.all_parameters["k1_0"] = torch.nn.Parameter(torch.tensor([[9.999999747378752e-06]]), requires_grad=True)
        self.all_parameters["k1_1"] = torch.nn.Parameter(torch.tensor([[9.999999747378752e-06]]), requires_grad=True)
        self.all_parameters["k1_2"] = torch.nn.Parameter(torch.tensor([[9.999999747378752e-06]]), requires_grad=True)
        self.all_parameters["k1_3"] = torch.nn.Parameter(torch.tensor([[9.999999747378752e-06]]), requires_grad=True)
        self.all_parameters["k2_0"] = torch.nn.Parameter(torch.tensor([[9.999999747378752e-06]]), requires_grad=True)
        self.all_parameters["k2_1"] = torch.nn.Parameter(torch.tensor([[9.999999747378752e-06]]), requires_grad=True)
        self.all_parameters["k2_2"] = torch.nn.Parameter(torch.tensor([[9.999999747378752e-06]]), requires_grad=True)
        self.all_parameters["k2_3"] = torch.nn.Parameter(torch.tensor([[9.999999747378752e-06]]), requires_grad=True)
        self.all_parameters["lat_k1_0_0_0"] = torch.nn.Parameter(torch.tensor([[9.999999747378752e-06]]), requires_grad=True)
        self.all_parameters["lat_k1_0_0_1"] = torch.nn.Parameter(torch.tensor([[9.999999747378752e-06]]), requires_grad=True)
        self.all_parameters["lat_k1_0_0_2"] = torch.nn.Parameter(torch.tensor([[9.999999747378752e-06]]), requires_grad=True)
        self.all_parameters["lat_k1_0_1_0"] = torch.nn.Parameter(torch.tensor([[9.999999747378752e-06]]), requires_grad=True)
        self.all_parameters["lat_k1_0_1_1"] = torch.nn.Parameter(torch.tensor([[9.999999747378752e-06]]), requires_grad=True)
        self.all_parameters["lat_k1_0_1_2"] = torch.nn.Parameter(torch.tensor([[9.999999747378752e-06]]), requires_grad=True)
        self.all_parameters["lat_k1_0_2_0"] = torch.nn.Parameter(torch.tensor([[9.999999747378752e-06]]), requires_grad=True)
        self.all_parameters["lat_k1_0_2_1"] = torch.nn.Parameter(torch.tensor([[9.999999747378752e-06]]), requires_grad=True)
        self.all_parameters["lat_k1_0_2_2"] = torch.nn.Parameter(torch.tensor([[9.999999747378752e-06]]), requires_grad=True)
        self.all_parameters["lat_k1_0_3_0"] = torch.nn.Parameter(torch.tensor([[9.999999747378752e-06]]), requires_grad=True)
        self.all_parameters["lat_k1_0_3_1"] = torch.nn.Parameter(torch.tensor([[9.999999747378752e-06]]), requires_grad=True)
        self.all_parameters["lat_k1_0_3_2"] = torch.nn.Parameter(torch.tensor([[9.999999747378752e-06]]), requires_grad=True)
        self.all_parameters["lat_k1_0_4_0"] = torch.nn.Parameter(torch.tensor([[9.999999747378752e-06]]), requires_grad=True)
        self.all_parameters["lat_k1_0_4_1"] = torch.nn.Parameter(torch.tensor([[9.999999747378752e-06]]), requires_grad=True)
        self.all_parameters["lat_k1_0_4_2"] = torch.nn.Parameter(torch.tensor([[9.999999747378752e-06]]), requires_grad=True)
        self.all_parameters["lat_k1_1_0_0"] = torch.nn.Parameter(torch.tensor([[9.999999747378752e-06]]), requires_grad=True)
        self.all_parameters["lat_k1_1_0_1"] = torch.nn.Parameter(torch.tensor([[9.999999747378752e-06]]), requires_grad=True)
        self.all_parameters["lat_k1_1_0_2"] = torch.nn.Parameter(torch.tensor([[9.999999747378752e-06]]), requires_grad=True)
        self.all_parameters["lat_k1_1_1_0"] = torch.nn.Parameter(torch.tensor([[9.999999747378752e-06]]), requires_grad=True)
        self.all_parameters["lat_k1_1_1_1"] = torch.nn.Parameter(torch.tensor([[9.999999747378752e-06]]), requires_grad=True)
        self.all_parameters["lat_k1_1_1_2"] = torch.nn.Parameter(torch.tensor([[9.999999747378752e-06]]), requires_grad=True)
        self.all_parameters["lat_k1_1_2_0"] = torch.nn.Parameter(torch.tensor([[9.999999747378752e-06]]), requires_grad=True)
        self.all_parameters["lat_k1_1_2_1"] = torch.nn.Parameter(torch.tensor([[9.999999747378752e-06]]), requires_grad=True)
        self.all_parameters["lat_k1_1_2_2"] = torch.nn.Parameter(torch.tensor([[9.999999747378752e-06]]), requires_grad=True)
        self.all_parameters["lat_k1_1_3_0"] = torch.nn.Parameter(torch.tensor([[9.999999747378752e-06]]), requires_grad=True)
        self.all_parameters["lat_k1_1_3_1"] = torch.nn.Parameter(torch.tensor([[9.999999747378752e-06]]), requires_grad=True)
        self.all_parameters["lat_k1_1_3_2"] = torch.nn.Parameter(torch.tensor([[9.999999747378752e-06]]), requires_grad=True)
        self.all_parameters["lat_k1_1_4_0"] = torch.nn.Parameter(torch.tensor([[9.999999747378752e-06]]), requires_grad=True)
        self.all_parameters["lat_k1_1_4_1"] = torch.nn.Parameter(torch.tensor([[9.999999747378752e-06]]), requires_grad=True)
        self.all_parameters["lat_k1_1_4_2"] = torch.nn.Parameter(torch.tensor([[9.999999747378752e-06]]), requires_grad=True)
        self.all_parameters["lat_k1_2_0_0"] = torch.nn.Parameter(torch.tensor([[9.999999747378752e-06]]), requires_grad=True)
        self.all_parameters["lat_k1_2_0_1"] = torch.nn.Parameter(torch.tensor([[9.999999747378752e-06]]), requires_grad=True)
        self.all_parameters["lat_k1_2_0_2"] = torch.nn.Parameter(torch.tensor([[9.999999747378752e-06]]), requires_grad=True)
        self.all_parameters["lat_k1_2_1_0"] = torch.nn.Parameter(torch.tensor([[9.999999747378752e-06]]), requires_grad=True)
        self.all_parameters["lat_k1_2_1_1"] = torch.nn.Parameter(torch.tensor([[9.999999747378752e-06]]), requires_grad=True)
        self.all_parameters["lat_k1_2_1_2"] = torch.nn.Parameter(torch.tensor([[9.999999747378752e-06]]), requires_grad=True)
        self.all_parameters["lat_k1_2_2_0"] = torch.nn.Parameter(torch.tensor([[9.999999747378752e-06]]), requires_grad=True)
        self.all_parameters["lat_k1_2_2_1"] = torch.nn.Parameter(torch.tensor([[9.999999747378752e-06]]), requires_grad=True)
        self.all_parameters["lat_k1_2_2_2"] = torch.nn.Parameter(torch.tensor([[9.999999747378752e-06]]), requires_grad=True)
        self.all_parameters["lat_k1_2_3_0"] = torch.nn.Parameter(torch.tensor([[9.999999747378752e-06]]), requires_grad=True)
        self.all_parameters["lat_k1_2_3_1"] = torch.nn.Parameter(torch.tensor([[9.999999747378752e-06]]), requires_grad=True)
        self.all_parameters["lat_k1_2_3_2"] = torch.nn.Parameter(torch.tensor([[9.999999747378752e-06]]), requires_grad=True)
        self.all_parameters["lat_k1_2_4_0"] = torch.nn.Parameter(torch.tensor([[9.999999747378752e-06]]), requires_grad=True)
        self.all_parameters["lat_k1_2_4_1"] = torch.nn.Parameter(torch.tensor([[9.999999747378752e-06]]), requires_grad=True)
        self.all_parameters["lat_k1_2_4_2"] = torch.nn.Parameter(torch.tensor([[9.999999747378752e-06]]), requires_grad=True)
        self.all_parameters["lat_k2_0_0_0"] = torch.nn.Parameter(torch.tensor([[9.999999747378752e-06]]), requires_grad=True)
        self.all_parameters["lat_k2_0_0_1"] = torch.nn.Parameter(torch.tensor([[9.999999747378752e-06]]), requires_grad=True)
        self.all_parameters["lat_k2_0_0_2"] = torch.nn.Parameter(torch.tensor([[9.999999747378752e-06]]), requires_grad=True)
        self.all_parameters["lat_k2_0_1_0"] = torch.nn.Parameter(torch.tensor([[9.999999747378752e-06]]), requires_grad=True)
        self.all_parameters["lat_k2_0_1_1"] = torch.nn.Parameter(torch.tensor([[9.999999747378752e-06]]), requires_grad=True)
        self.all_parameters["lat_k2_0_1_2"] = torch.nn.Parameter(torch.tensor([[9.999999747378752e-06]]), requires_grad=True)
        self.all_parameters["lat_k2_0_2_0"] = torch.nn.Parameter(torch.tensor([[9.999999747378752e-06]]), requires_grad=True)
        self.all_parameters["lat_k2_0_2_1"] = torch.nn.Parameter(torch.tensor([[9.999999747378752e-06]]), requires_grad=True)
        self.all_parameters["lat_k2_0_2_2"] = torch.nn.Parameter(torch.tensor([[9.999999747378752e-06]]), requires_grad=True)
        self.all_parameters["lat_k2_0_3_0"] = torch.nn.Parameter(torch.tensor([[9.999999747378752e-06]]), requires_grad=True)
        self.all_parameters["lat_k2_0_3_1"] = torch.nn.Parameter(torch.tensor([[9.999999747378752e-06]]), requires_grad=True)
        self.all_parameters["lat_k2_0_3_2"] = torch.nn.Parameter(torch.tensor([[9.999999747378752e-06]]), requires_grad=True)
        self.all_parameters["lat_k2_0_4_0"] = torch.nn.Parameter(torch.tensor([[9.999999747378752e-06]]), requires_grad=True)
        self.all_parameters["lat_k2_0_4_1"] = torch.nn.Parameter(torch.tensor([[9.999999747378752e-06]]), requires_grad=True)
        self.all_parameters["lat_k2_0_4_2"] = torch.nn.Parameter(torch.tensor([[9.999999747378752e-06]]), requires_grad=True)
        self.all_parameters["lat_k2_1_0_0"] = torch.nn.Parameter(torch.tensor([[9.999999747378752e-06]]), requires_grad=True)
        self.all_parameters["lat_k2_1_0_1"] = torch.nn.Parameter(torch.tensor([[9.999999747378752e-06]]), requires_grad=True)
        self.all_parameters["lat_k2_1_0_2"] = torch.nn.Parameter(torch.tensor([[9.999999747378752e-06]]), requires_grad=True)
        self.all_parameters["lat_k2_1_1_0"] = torch.nn.Parameter(torch.tensor([[9.999999747378752e-06]]), requires_grad=True)
        self.all_parameters["lat_k2_1_1_1"] = torch.nn.Parameter(torch.tensor([[9.999999747378752e-06]]), requires_grad=True)
        self.all_parameters["lat_k2_1_1_2"] = torch.nn.Parameter(torch.tensor([[9.999999747378752e-06]]), requires_grad=True)
        self.all_parameters["lat_k2_1_2_0"] = torch.nn.Parameter(torch.tensor([[9.999999747378752e-06]]), requires_grad=True)
        self.all_parameters["lat_k2_1_2_1"] = torch.nn.Parameter(torch.tensor([[9.999999747378752e-06]]), requires_grad=True)
        self.all_parameters["lat_k2_1_2_2"] = torch.nn.Parameter(torch.tensor([[9.999999747378752e-06]]), requires_grad=True)
        self.all_parameters["lat_k2_1_3_0"] = torch.nn.Parameter(torch.tensor([[9.999999747378752e-06]]), requires_grad=True)
        self.all_parameters["lat_k2_1_3_1"] = torch.nn.Parameter(torch.tensor([[9.999999747378752e-06]]), requires_grad=True)
        self.all_parameters["lat_k2_1_3_2"] = torch.nn.Parameter(torch.tensor([[9.999999747378752e-06]]), requires_grad=True)
        self.all_parameters["lat_k2_1_4_0"] = torch.nn.Parameter(torch.tensor([[9.999999747378752e-06]]), requires_grad=True)
        self.all_parameters["lat_k2_1_4_1"] = torch.nn.Parameter(torch.tensor([[9.999999747378752e-06]]), requires_grad=True)
        self.all_parameters["lat_k2_1_4_2"] = torch.nn.Parameter(torch.tensor([[9.999999747378752e-06]]), requires_grad=True)
        self.all_parameters["lat_k2_2_0_0"] = torch.nn.Parameter(torch.tensor([[9.999999747378752e-06]]), requires_grad=True)
        self.all_parameters["lat_k2_2_0_1"] = torch.nn.Parameter(torch.tensor([[9.999999747378752e-06]]), requires_grad=True)
        self.all_parameters["lat_k2_2_0_2"] = torch.nn.Parameter(torch.tensor([[9.999999747378752e-06]]), requires_grad=True)
        self.all_parameters["lat_k2_2_1_0"] = torch.nn.Parameter(torch.tensor([[9.999999747378752e-06]]), requires_grad=True)
        self.all_parameters["lat_k2_2_1_1"] = torch.nn.Parameter(torch.tensor([[9.999999747378752e-06]]), requires_grad=True)
        self.all_parameters["lat_k2_2_1_2"] = torch.nn.Parameter(torch.tensor([[9.999999747378752e-06]]), requires_grad=True)
        self.all_parameters["lat_k2_2_2_0"] = torch.nn.Parameter(torch.tensor([[9.999999747378752e-06]]), requires_grad=True)
        self.all_parameters["lat_k2_2_2_1"] = torch.nn.Parameter(torch.tensor([[9.999999747378752e-06]]), requires_grad=True)
        self.all_parameters["lat_k2_2_2_2"] = torch.nn.Parameter(torch.tensor([[9.999999747378752e-06]]), requires_grad=True)
        self.all_parameters["lat_k2_2_3_0"] = torch.nn.Parameter(torch.tensor([[9.999999747378752e-06]]), requires_grad=True)
        self.all_parameters["lat_k2_2_3_1"] = torch.nn.Parameter(torch.tensor([[9.999999747378752e-06]]), requires_grad=True)
        self.all_parameters["lat_k2_2_3_2"] = torch.nn.Parameter(torch.tensor([[9.999999747378752e-06]]), requires_grad=True)
        self.all_parameters["lat_k2_2_4_0"] = torch.nn.Parameter(torch.tensor([[9.999999747378752e-06]]), requires_grad=True)
        self.all_parameters["lat_k2_2_4_1"] = torch.nn.Parameter(torch.tensor([[9.999999747378752e-06]]), requires_grad=True)
        self.all_parameters["lat_k2_2_4_2"] = torch.nn.Parameter(torch.tensor([[9.999999747378752e-06]]), requires_grad=True)
        self.all_parameters["lat_k3_0_0_0"] = torch.nn.Parameter(torch.tensor([[9.999999747378752e-06]]), requires_grad=True)
        self.all_parameters["lat_k3_0_0_1"] = torch.nn.Parameter(torch.tensor([[9.999999747378752e-06]]), requires_grad=True)
        self.all_parameters["lat_k3_0_0_2"] = torch.nn.Parameter(torch.tensor([[9.999999747378752e-06]]), requires_grad=True)
        self.all_parameters["lat_k3_0_1_0"] = torch.nn.Parameter(torch.tensor([[9.999999747378752e-06]]), requires_grad=True)
        self.all_parameters["lat_k3_0_1_1"] = torch.nn.Parameter(torch.tensor([[9.999999747378752e-06]]), requires_grad=True)
        self.all_parameters["lat_k3_0_1_2"] = torch.nn.Parameter(torch.tensor([[9.999999747378752e-06]]), requires_grad=True)
        self.all_parameters["lat_k3_0_2_0"] = torch.nn.Parameter(torch.tensor([[9.999999747378752e-06]]), requires_grad=True)
        self.all_parameters["lat_k3_0_2_1"] = torch.nn.Parameter(torch.tensor([[9.999999747378752e-06]]), requires_grad=True)
        self.all_parameters["lat_k3_0_2_2"] = torch.nn.Parameter(torch.tensor([[9.999999747378752e-06]]), requires_grad=True)
        self.all_parameters["lat_k3_0_3_0"] = torch.nn.Parameter(torch.tensor([[9.999999747378752e-06]]), requires_grad=True)
        self.all_parameters["lat_k3_0_3_1"] = torch.nn.Parameter(torch.tensor([[9.999999747378752e-06]]), requires_grad=True)
        self.all_parameters["lat_k3_0_3_2"] = torch.nn.Parameter(torch.tensor([[9.999999747378752e-06]]), requires_grad=True)
        self.all_parameters["lat_k3_0_4_0"] = torch.nn.Parameter(torch.tensor([[9.999999747378752e-06]]), requires_grad=True)
        self.all_parameters["lat_k3_0_4_1"] = torch.nn.Parameter(torch.tensor([[9.999999747378752e-06]]), requires_grad=True)
        self.all_parameters["lat_k3_0_4_2"] = torch.nn.Parameter(torch.tensor([[9.999999747378752e-06]]), requires_grad=True)
        self.all_parameters["lat_k3_1_0_0"] = torch.nn.Parameter(torch.tensor([[9.999999747378752e-06]]), requires_grad=True)
        self.all_parameters["lat_k3_1_0_1"] = torch.nn.Parameter(torch.tensor([[9.999999747378752e-06]]), requires_grad=True)
        self.all_parameters["lat_k3_1_0_2"] = torch.nn.Parameter(torch.tensor([[9.999999747378752e-06]]), requires_grad=True)
        self.all_parameters["lat_k3_1_1_0"] = torch.nn.Parameter(torch.tensor([[9.999999747378752e-06]]), requires_grad=True)
        self.all_parameters["lat_k3_1_1_1"] = torch.nn.Parameter(torch.tensor([[9.999999747378752e-06]]), requires_grad=True)
        self.all_parameters["lat_k3_1_1_2"] = torch.nn.Parameter(torch.tensor([[9.999999747378752e-06]]), requires_grad=True)
        self.all_parameters["lat_k3_1_2_0"] = torch.nn.Parameter(torch.tensor([[9.999999747378752e-06]]), requires_grad=True)
        self.all_parameters["lat_k3_1_2_1"] = torch.nn.Parameter(torch.tensor([[9.999999747378752e-06]]), requires_grad=True)
        self.all_parameters["lat_k3_1_2_2"] = torch.nn.Parameter(torch.tensor([[9.999999747378752e-06]]), requires_grad=True)
        self.all_parameters["lat_k3_1_3_0"] = torch.nn.Parameter(torch.tensor([[9.999999747378752e-06]]), requires_grad=True)
        self.all_parameters["lat_k3_1_3_1"] = torch.nn.Parameter(torch.tensor([[9.999999747378752e-06]]), requires_grad=True)
        self.all_parameters["lat_k3_1_3_2"] = torch.nn.Parameter(torch.tensor([[9.999999747378752e-06]]), requires_grad=True)
        self.all_parameters["lat_k3_1_4_0"] = torch.nn.Parameter(torch.tensor([[9.999999747378752e-06]]), requires_grad=True)
        self.all_parameters["lat_k3_1_4_1"] = torch.nn.Parameter(torch.tensor([[9.999999747378752e-06]]), requires_grad=True)
        self.all_parameters["lat_k3_1_4_2"] = torch.nn.Parameter(torch.tensor([[9.999999747378752e-06]]), requires_grad=True)
        self.all_parameters["lat_k3_2_0_0"] = torch.nn.Parameter(torch.tensor([[9.999999747378752e-06]]), requires_grad=True)
        self.all_parameters["lat_k3_2_0_1"] = torch.nn.Parameter(torch.tensor([[9.999999747378752e-06]]), requires_grad=True)
        self.all_parameters["lat_k3_2_0_2"] = torch.nn.Parameter(torch.tensor([[9.999999747378752e-06]]), requires_grad=True)
        self.all_parameters["lat_k3_2_1_0"] = torch.nn.Parameter(torch.tensor([[9.999999747378752e-06]]), requires_grad=True)
        self.all_parameters["lat_k3_2_1_1"] = torch.nn.Parameter(torch.tensor([[9.999999747378752e-06]]), requires_grad=True)
        self.all_parameters["lat_k3_2_1_2"] = torch.nn.Parameter(torch.tensor([[9.999999747378752e-06]]), requires_grad=True)
        self.all_parameters["lat_k3_2_2_0"] = torch.nn.Parameter(torch.tensor([[9.999999747378752e-06]]), requires_grad=True)
        self.all_parameters["lat_k3_2_2_1"] = torch.nn.Parameter(torch.tensor([[9.999999747378752e-06]]), requires_grad=True)
        self.all_parameters["lat_k3_2_2_2"] = torch.nn.Parameter(torch.tensor([[9.999999747378752e-06]]), requires_grad=True)
        self.all_parameters["lat_k3_2_3_0"] = torch.nn.Parameter(torch.tensor([[9.999999747378752e-06]]), requires_grad=True)
        self.all_parameters["lat_k3_2_3_1"] = torch.nn.Parameter(torch.tensor([[9.999999747378752e-06]]), requires_grad=True)
        self.all_parameters["lat_k3_2_3_2"] = torch.nn.Parameter(torch.tensor([[9.999999747378752e-06]]), requires_grad=True)
        self.all_parameters["lat_k3_2_4_0"] = torch.nn.Parameter(torch.tensor([[9.999999747378752e-06]]), requires_grad=True)
        self.all_parameters["lat_k3_2_4_1"] = torch.nn.Parameter(torch.tensor([[9.999999747378752e-06]]), requires_grad=True)
        self.all_parameters["lat_k3_2_4_2"] = torch.nn.Parameter(torch.tensor([[9.999999747378752e-06]]), requires_grad=True)
        self.all_parameters["lat_k4_0_0_0"] = torch.nn.Parameter(torch.tensor([[9.999999747378752e-06]]), requires_grad=True)
        self.all_parameters["lat_k4_0_0_1"] = torch.nn.Parameter(torch.tensor([[9.999999747378752e-06]]), requires_grad=True)
        self.all_parameters["lat_k4_0_0_2"] = torch.nn.Parameter(torch.tensor([[9.999999747378752e-06]]), requires_grad=True)
        self.all_parameters["lat_k4_0_1_0"] = torch.nn.Parameter(torch.tensor([[9.999999747378752e-06]]), requires_grad=True)
        self.all_parameters["lat_k4_0_1_1"] = torch.nn.Parameter(torch.tensor([[9.999999747378752e-06]]), requires_grad=True)
        self.all_parameters["lat_k4_0_1_2"] = torch.nn.Parameter(torch.tensor([[9.999999747378752e-06]]), requires_grad=True)
        self.all_parameters["lat_k4_0_2_0"] = torch.nn.Parameter(torch.tensor([[9.999999747378752e-06]]), requires_grad=True)
        self.all_parameters["lat_k4_0_2_1"] = torch.nn.Parameter(torch.tensor([[9.999999747378752e-06]]), requires_grad=True)
        self.all_parameters["lat_k4_0_2_2"] = torch.nn.Parameter(torch.tensor([[9.999999747378752e-06]]), requires_grad=True)
        self.all_parameters["lat_k4_0_3_0"] = torch.nn.Parameter(torch.tensor([[9.999999747378752e-06]]), requires_grad=True)
        self.all_parameters["lat_k4_0_3_1"] = torch.nn.Parameter(torch.tensor([[9.999999747378752e-06]]), requires_grad=True)
        self.all_parameters["lat_k4_0_3_2"] = torch.nn.Parameter(torch.tensor([[9.999999747378752e-06]]), requires_grad=True)
        self.all_parameters["lat_k4_0_4_0"] = torch.nn.Parameter(torch.tensor([[9.999999747378752e-06]]), requires_grad=True)
        self.all_parameters["lat_k4_0_4_1"] = torch.nn.Parameter(torch.tensor([[9.999999747378752e-06]]), requires_grad=True)
        self.all_parameters["lat_k4_0_4_2"] = torch.nn.Parameter(torch.tensor([[9.999999747378752e-06]]), requires_grad=True)
        self.all_parameters["lat_k4_1_0_0"] = torch.nn.Parameter(torch.tensor([[9.999999747378752e-06]]), requires_grad=True)
        self.all_parameters["lat_k4_1_0_1"] = torch.nn.Parameter(torch.tensor([[9.999999747378752e-06]]), requires_grad=True)
        self.all_parameters["lat_k4_1_0_2"] = torch.nn.Parameter(torch.tensor([[9.999999747378752e-06]]), requires_grad=True)
        self.all_parameters["lat_k4_1_1_0"] = torch.nn.Parameter(torch.tensor([[9.999999747378752e-06]]), requires_grad=True)
        self.all_parameters["lat_k4_1_1_1"] = torch.nn.Parameter(torch.tensor([[9.999999747378752e-06]]), requires_grad=True)
        self.all_parameters["lat_k4_1_1_2"] = torch.nn.Parameter(torch.tensor([[9.999999747378752e-06]]), requires_grad=True)
        self.all_parameters["lat_k4_1_2_0"] = torch.nn.Parameter(torch.tensor([[9.999999747378752e-06]]), requires_grad=True)
        self.all_parameters["lat_k4_1_2_1"] = torch.nn.Parameter(torch.tensor([[9.999999747378752e-06]]), requires_grad=True)
        self.all_parameters["lat_k4_1_2_2"] = torch.nn.Parameter(torch.tensor([[9.999999747378752e-06]]), requires_grad=True)
        self.all_parameters["lat_k4_1_3_0"] = torch.nn.Parameter(torch.tensor([[9.999999747378752e-06]]), requires_grad=True)
        self.all_parameters["lat_k4_1_3_1"] = torch.nn.Parameter(torch.tensor([[9.999999747378752e-06]]), requires_grad=True)
        self.all_parameters["lat_k4_1_3_2"] = torch.nn.Parameter(torch.tensor([[9.999999747378752e-06]]), requires_grad=True)
        self.all_parameters["lat_k4_1_4_0"] = torch.nn.Parameter(torch.tensor([[9.999999747378752e-06]]), requires_grad=True)
        self.all_parameters["lat_k4_1_4_1"] = torch.nn.Parameter(torch.tensor([[9.999999747378752e-06]]), requires_grad=True)
        self.all_parameters["lat_k4_1_4_2"] = torch.nn.Parameter(torch.tensor([[9.999999747378752e-06]]), requires_grad=True)
        self.all_parameters["lat_k4_2_0_0"] = torch.nn.Parameter(torch.tensor([[9.999999747378752e-06]]), requires_grad=True)
        self.all_parameters["lat_k4_2_0_1"] = torch.nn.Parameter(torch.tensor([[9.999999747378752e-06]]), requires_grad=True)
        self.all_parameters["lat_k4_2_0_2"] = torch.nn.Parameter(torch.tensor([[9.999999747378752e-06]]), requires_grad=True)
        self.all_parameters["lat_k4_2_1_0"] = torch.nn.Parameter(torch.tensor([[9.999999747378752e-06]]), requires_grad=True)
        self.all_parameters["lat_k4_2_1_1"] = torch.nn.Parameter(torch.tensor([[9.999999747378752e-06]]), requires_grad=True)
        self.all_parameters["lat_k4_2_1_2"] = torch.nn.Parameter(torch.tensor([[9.999999747378752e-06]]), requires_grad=True)
        self.all_parameters["lat_k4_2_2_0"] = torch.nn.Parameter(torch.tensor([[9.999999747378752e-06]]), requires_grad=True)
        self.all_parameters["lat_k4_2_2_1"] = torch.nn.Parameter(torch.tensor([[9.999999747378752e-06]]), requires_grad=True)
        self.all_parameters["lat_k4_2_2_2"] = torch.nn.Parameter(torch.tensor([[9.999999747378752e-06]]), requires_grad=True)
        self.all_parameters["lat_k4_2_3_0"] = torch.nn.Parameter(torch.tensor([[9.999999747378752e-06]]), requires_grad=True)
        self.all_parameters["lat_k4_2_3_1"] = torch.nn.Parameter(torch.tensor([[9.999999747378752e-06]]), requires_grad=True)
        self.all_parameters["lat_k4_2_3_2"] = torch.nn.Parameter(torch.tensor([[9.999999747378752e-06]]), requires_grad=True)
        self.all_parameters["lat_k4_2_4_0"] = torch.nn.Parameter(torch.tensor([[9.999999747378752e-06]]), requires_grad=True)
        self.all_parameters["lat_k4_2_4_1"] = torch.nn.Parameter(torch.tensor([[9.999999747378752e-06]]), requires_grad=True)
        self.all_parameters["lat_k4_2_4_2"] = torch.nn.Parameter(torch.tensor([[9.999999747378752e-06]]), requires_grad=True)
        self.all_parameters["mass"] = torch.nn.Parameter(torch.tensor([[3.5]]), requires_grad=True)
        self.all_parameters["r1"] = torch.nn.Parameter(torch.tensor([[0.004999999888241291]]), requires_grad=True)
        self.all_parameters["PFir407b"] = torch.nn.Parameter(torch.tensor([4.978706783731468e-06, 6.948345344426343e-06, 9.697197128843982e-06, 1.3533528544940054e-05, 1.8887560145230964e-05, 2.6359713956480846e-05, 3.678794382722117e-05, 5.134171078680083e-05, 7.1653128543403e-05, 9.999999747378752e-05]), requires_grad=True)
        self.Fir1300 = torch.nn.Dropout(p=0.05)
        self.all_parameters["PFir407W"] = torch.nn.Parameter(torch.tensor([[4.978706783731468e-06, 4.978706783731468e-06, 4.978706783731468e-06, 4.978706783731468e-06, 4.978706783731468e-06, 4.978706783731468e-06, 4.978706783731468e-06, 4.978706783731468e-06, 4.978706783731468e-06, 4.978706783731468e-06], [6.1685013861279e-06, 6.1685013861279e-06, 6.1685013861279e-06, 6.1685013861279e-06, 6.1685013861279e-06, 6.1685013861279e-06, 6.1685013861279e-06, 6.1685013861279e-06, 6.1685013861279e-06, 6.1685013861279e-06], [7.642628588655498e-06, 7.642628588655498e-06, 7.642628588655498e-06, 7.642628588655498e-06, 7.642628588655498e-06, 7.642628588655498e-06, 7.642628588655498e-06, 7.642628588655498e-06, 7.642628588655498e-06, 7.642628588655498e-06], [9.469037649978418e-06, 9.469037649978418e-06, 9.469037649978418e-06, 9.469037649978418e-06, 9.469037649978418e-06, 9.469037649978418e-06, 9.469037649978418e-06, 9.469037649978418e-06, 9.469037649978418e-06, 9.469037649978418e-06], [1.1731916856660973e-05, 1.1731916856660973e-05, 1.1731916856660973e-05, 1.1731916856660973e-05, 1.1731916856660973e-05, 1.1731916856660973e-05, 1.1731916856660973e-05, 1.1731916856660973e-05, 1.1731916856660973e-05, 1.1731916856660973e-05], [1.4535569789586589e-05, 1.4535569789586589e-05, 1.4535569789586589e-05, 1.4535569789586589e-05, 1.4535569789586589e-05, 1.4535569789586589e-05, 1.4535569789586589e-05, 1.4535569789586589e-05, 1.4535569789586589e-05, 1.4535569789586589e-05], [1.800923200789839e-05, 1.800923200789839e-05, 1.800923200789839e-05, 1.800923200789839e-05, 1.800923200789839e-05, 1.800923200789839e-05, 1.800923200789839e-05, 1.800923200789839e-05, 1.800923200789839e-05, 1.800923200789839e-05], [2.2313015506369993e-05, 2.2313015506369993e-05, 2.2313015506369993e-05, 2.2313015506369993e-05, 2.2313015506369993e-05, 2.2313015506369993e-05, 2.2313015506369993e-05, 2.2313015506369993e-05, 2.2313015506369993e-05, 2.2313015506369993e-05], [2.764530472632032e-05, 2.764530472632032e-05, 2.764530472632032e-05, 2.764530472632032e-05, 2.764530472632032e-05, 2.764530472632032e-05, 2.764530472632032e-05, 2.764530472632032e-05, 2.764530472632032e-05, 2.764530472632032e-05], [3.4251886972924694e-05, 3.4251886972924694e-05, 3.4251886972924694e-05, 3.4251886972924694e-05, 3.4251886972924694e-05, 3.4251886972924694e-05, 3.4251886972924694e-05, 3.4251886972924694e-05, 3.4251886972924694e-05, 3.4251886972924694e-05], [4.243728471919894e-05, 4.243728471919894e-05, 4.243728471919894e-05, 4.243728471919894e-05, 4.243728471919894e-05, 4.243728471919894e-05, 4.243728471919894e-05, 4.243728471919894e-05, 4.243728471919894e-05, 4.243728471919894e-05], [5.257880184217356e-05, 5.257880184217356e-05, 5.257880184217356e-05, 5.257880184217356e-05, 5.257880184217356e-05, 5.257880184217356e-05, 5.257880184217356e-05, 5.257880184217356e-05, 5.257880184217356e-05, 5.257880184217356e-05], [6.514390406664461e-05, 6.514390406664461e-05, 6.514390406664461e-05, 6.514390406664461e-05, 6.514390406664461e-05, 6.514390406664461e-05, 6.514390406664461e-05, 6.514390406664461e-05, 6.514390406664461e-05, 6.514390406664461e-05], [8.071177580859512e-05, 8.071177580859512e-05, 8.071177580859512e-05, 8.071177580859512e-05, 8.071177580859512e-05, 8.071177580859512e-05, 8.071177580859512e-05, 8.071177580859512e-05, 8.071177580859512e-05, 8.071177580859512e-05], [9.999999747378752e-05, 9.999999747378752e-05, 9.999999747378752e-05, 9.999999747378752e-05, 9.999999747378752e-05, 9.999999747378752e-05, 9.999999747378752e-05, 9.999999747378752e-05, 9.999999747378752e-05, 9.999999747378752e-05]]), requires_grad=True)
        self.all_parameters["PFir412b"] = torch.nn.Parameter(torch.tensor([4.978706783731468e-06, 6.948345344426343e-06, 9.697197128843982e-06, 1.3533528544940054e-05, 1.8887560145230964e-05, 2.6359713956480846e-05, 3.678794382722117e-05, 5.134171078680083e-05, 7.1653128543403e-05, 9.999999747378752e-05]), requires_grad=True)
        self.Fir1303 = torch.nn.Dropout(p=0.05)
        self.all_parameters["PFir412W"] = torch.nn.Parameter(torch.tensor([[4.978706783731468e-06, 4.978706783731468e-06, 4.978706783731468e-06, 4.978706783731468e-06, 4.978706783731468e-06, 4.978706783731468e-06, 4.978706783731468e-06, 4.978706783731468e-06, 4.978706783731468e-06, 4.978706783731468e-06], [6.1685013861279e-06, 6.1685013861279e-06, 6.1685013861279e-06, 6.1685013861279e-06, 6.1685013861279e-06, 6.1685013861279e-06, 6.1685013861279e-06, 6.1685013861279e-06, 6.1685013861279e-06, 6.1685013861279e-06], [7.642628588655498e-06, 7.642628588655498e-06, 7.642628588655498e-06, 7.642628588655498e-06, 7.642628588655498e-06, 7.642628588655498e-06, 7.642628588655498e-06, 7.642628588655498e-06, 7.642628588655498e-06, 7.642628588655498e-06], [9.469037649978418e-06, 9.469037649978418e-06, 9.469037649978418e-06, 9.469037649978418e-06, 9.469037649978418e-06, 9.469037649978418e-06, 9.469037649978418e-06, 9.469037649978418e-06, 9.469037649978418e-06, 9.469037649978418e-06], [1.1731916856660973e-05, 1.1731916856660973e-05, 1.1731916856660973e-05, 1.1731916856660973e-05, 1.1731916856660973e-05, 1.1731916856660973e-05, 1.1731916856660973e-05, 1.1731916856660973e-05, 1.1731916856660973e-05, 1.1731916856660973e-05], [1.4535569789586589e-05, 1.4535569789586589e-05, 1.4535569789586589e-05, 1.4535569789586589e-05, 1.4535569789586589e-05, 1.4535569789586589e-05, 1.4535569789586589e-05, 1.4535569789586589e-05, 1.4535569789586589e-05, 1.4535569789586589e-05], [1.800923200789839e-05, 1.800923200789839e-05, 1.800923200789839e-05, 1.800923200789839e-05, 1.800923200789839e-05, 1.800923200789839e-05, 1.800923200789839e-05, 1.800923200789839e-05, 1.800923200789839e-05, 1.800923200789839e-05], [2.2313015506369993e-05, 2.2313015506369993e-05, 2.2313015506369993e-05, 2.2313015506369993e-05, 2.2313015506369993e-05, 2.2313015506369993e-05, 2.2313015506369993e-05, 2.2313015506369993e-05, 2.2313015506369993e-05, 2.2313015506369993e-05], [2.764530472632032e-05, 2.764530472632032e-05, 2.764530472632032e-05, 2.764530472632032e-05, 2.764530472632032e-05, 2.764530472632032e-05, 2.764530472632032e-05, 2.764530472632032e-05, 2.764530472632032e-05, 2.764530472632032e-05], [3.4251886972924694e-05, 3.4251886972924694e-05, 3.4251886972924694e-05, 3.4251886972924694e-05, 3.4251886972924694e-05, 3.4251886972924694e-05, 3.4251886972924694e-05, 3.4251886972924694e-05, 3.4251886972924694e-05, 3.4251886972924694e-05], [4.243728471919894e-05, 4.243728471919894e-05, 4.243728471919894e-05, 4.243728471919894e-05, 4.243728471919894e-05, 4.243728471919894e-05, 4.243728471919894e-05, 4.243728471919894e-05, 4.243728471919894e-05, 4.243728471919894e-05], [5.257880184217356e-05, 5.257880184217356e-05, 5.257880184217356e-05, 5.257880184217356e-05, 5.257880184217356e-05, 5.257880184217356e-05, 5.257880184217356e-05, 5.257880184217356e-05, 5.257880184217356e-05, 5.257880184217356e-05], [6.514390406664461e-05, 6.514390406664461e-05, 6.514390406664461e-05, 6.514390406664461e-05, 6.514390406664461e-05, 6.514390406664461e-05, 6.514390406664461e-05, 6.514390406664461e-05, 6.514390406664461e-05, 6.514390406664461e-05], [8.071177580859512e-05, 8.071177580859512e-05, 8.071177580859512e-05, 8.071177580859512e-05, 8.071177580859512e-05, 8.071177580859512e-05, 8.071177580859512e-05, 8.071177580859512e-05, 8.071177580859512e-05, 8.071177580859512e-05], [9.999999747378752e-05, 9.999999747378752e-05, 9.999999747378752e-05, 9.999999747378752e-05, 9.999999747378752e-05, 9.999999747378752e-05, 9.999999747378752e-05, 9.999999747378752e-05, 9.999999747378752e-05, 9.999999747378752e-05]]), requires_grad=True)
        self.all_parameters["PFir434W"] = torch.nn.Parameter(torch.tensor([[6.737947160218027e-07], [9.63014372246107e-07], [1.3763786910203635e-06], [1.9671754216687987e-06], [2.811565991578391e-06], [4.018402705696644e-06], [5.743262136093108e-06], [8.208499821193982e-06], [1.1731916856660973e-05], [1.676772444625385e-05], [2.39651035371935e-05], [3.4251886972924694e-05], [4.895416714134626e-05], [6.996725278440863e-05], [9.999999747378752e-05]]), requires_grad=True)
        self.all_parameters["PFir436W"] = torch.nn.Parameter(torch.tensor([[6.737947160218027e-07], [9.63014372246107e-07], [1.3763786910203635e-06], [1.9671754216687987e-06], [2.811565991578391e-06], [4.018402705696644e-06], [5.743262136093108e-06], [8.208499821193982e-06], [1.1731916856660973e-05], [1.676772444625385e-05], [2.39651035371935e-05], [3.4251886972924694e-05], [4.895416714134626e-05], [6.996725278440863e-05], [9.999999747378752e-05]]), requires_grad=True)
        self.all_parameters["PFir637W"] = torch.nn.Parameter(torch.tensor([[6.737947160218027e-07], [9.63014372246107e-07], [1.3763786910203635e-06], [1.9671754216687987e-06], [2.811565991578391e-06], [4.018402705696644e-06], [5.743262136093108e-06], [8.208499821193982e-06], [1.1731916856660973e-05], [1.676772444625385e-05], [2.39651035371935e-05], [3.4251886972924694e-05], [4.895416714134626e-05], [6.996725278440863e-05], [9.999999747378752e-05]]), requires_grad=True)
        self.all_parameters["PFir639W"] = torch.nn.Parameter(torch.tensor([[6.737947160218027e-07], [9.63014372246107e-07], [1.3763786910203635e-06], [1.9671754216687987e-06], [2.811565991578391e-06], [4.018402705696644e-06], [5.743262136093108e-06], [8.208499821193982e-06], [1.1731916856660973e-05], [1.676772444625385e-05], [2.39651035371935e-05], [3.4251886972924694e-05], [4.895416714134626e-05], [6.996725278440863e-05], [9.999999747378752e-05]]), requires_grad=True)
        self.all_parameters["PFir641W"] = torch.nn.Parameter(torch.tensor([[6.737947160218027e-07], [9.63014372246107e-07], [1.3763786910203635e-06], [1.9671754216687987e-06], [2.811565991578391e-06], [4.018402705696644e-06], [5.743262136093108e-06], [8.208499821193982e-06], [1.1731916856660973e-05], [1.676772444625385e-05], [2.39651035371935e-05], [3.4251886972924694e-05], [4.895416714134626e-05], [6.996725278440863e-05], [9.999999747378752e-05]]), requires_grad=True)
        self.all_parameters["PFir643W"] = torch.nn.Parameter(torch.tensor([[6.737947160218027e-07], [9.63014372246107e-07], [1.3763786910203635e-06], [1.9671754216687987e-06], [2.811565991578391e-06], [4.018402705696644e-06], [5.743262136093108e-06], [8.208499821193982e-06], [1.1731916856660973e-05], [1.676772444625385e-05], [2.39651035371935e-05], [3.4251886972924694e-05], [4.895416714134626e-05], [6.996725278440863e-05], [9.999999747378752e-05]]), requires_grad=True)
        self.all_parameters["PFir645W"] = torch.nn.Parameter(torch.tensor([[6.737947160218027e-07], [9.63014372246107e-07], [1.3763786910203635e-06], [1.9671754216687987e-06], [2.811565991578391e-06], [4.018402705696644e-06], [5.743262136093108e-06], [8.208499821193982e-06], [1.1731916856660973e-05], [1.676772444625385e-05], [2.39651035371935e-05], [3.4251886972924694e-05], [4.895416714134626e-05], [6.996725278440863e-05], [9.999999747378752e-05]]), requires_grad=True)
        self.all_parameters["PFir647W"] = torch.nn.Parameter(torch.tensor([[6.737947160218027e-07], [9.63014372246107e-07], [1.3763786910203635e-06], [1.9671754216687987e-06], [2.811565991578391e-06], [4.018402705696644e-06], [5.743262136093108e-06], [8.208499821193982e-06], [1.1731916856660973e-05], [1.676772444625385e-05], [2.39651035371935e-05], [3.4251886972924694e-05], [4.895416714134626e-05], [6.996725278440863e-05], [9.999999747378752e-05]]), requires_grad=True)
        self.all_parameters["PFir649W"] = torch.nn.Parameter(torch.tensor([[6.737947160218027e-07], [9.63014372246107e-07], [1.3763786910203635e-06], [1.9671754216687987e-06], [2.811565991578391e-06], [4.018402705696644e-06], [5.743262136093108e-06], [8.208499821193982e-06], [1.1731916856660973e-05], [1.676772444625385e-05], [2.39651035371935e-05], [3.4251886972924694e-05], [4.895416714134626e-05], [6.996725278440863e-05], [9.999999747378752e-05]]), requires_grad=True)
        self.all_parameters["PFir651W"] = torch.nn.Parameter(torch.tensor([[6.737947160218027e-07], [9.63014372246107e-07], [1.3763786910203635e-06], [1.9671754216687987e-06], [2.811565991578391e-06], [4.018402705696644e-06], [5.743262136093108e-06], [8.208499821193982e-06], [1.1731916856660973e-05], [1.676772444625385e-05], [2.39651035371935e-05], [3.4251886972924694e-05], [4.895416714134626e-05], [6.996725278440863e-05], [9.999999747378752e-05]]), requires_grad=True)
        self.all_parameters["PFir653W"] = torch.nn.Parameter(torch.tensor([[6.737947160218027e-07], [9.63014372246107e-07], [1.3763786910203635e-06], [1.9671754216687987e-06], [2.811565991578391e-06], [4.018402705696644e-06], [5.743262136093108e-06], [8.208499821193982e-06], [1.1731916856660973e-05], [1.676772444625385e-05], [2.39651035371935e-05], [3.4251886972924694e-05], [4.895416714134626e-05], [6.996725278440863e-05], [9.999999747378752e-05]]), requires_grad=True)
        self.all_parameters["PLinear410W"] = torch.nn.Parameter(torch.tensor([[0.46568745374679565], [0.23276680707931519], [0.4527209997177124], [0.5871122479438782], [0.40864473581314087], [0.1271744966506958], [0.6372835040092468], [0.2420617938041687], [0.7311904430389404], [0.722437858581543]]), requires_grad=True)
        self.all_parameters["PLinear415W"] = torch.nn.Parameter(torch.tensor([[0.19923532009124756], [0.694827139377594], [0.5830032825469971], [0.6318286061286926], [0.5558860898017883], [0.12624889612197876], [0.9790288209915161], [0.8442656397819519], [0.1255868673324585], [0.4456220865249634]]), requires_grad=True)
        self.all_constants["SamplePart1295"] = torch.tensor([[1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.]], requires_grad=False)
        self.all_constants["SamplePart1298"] = torch.tensor([[1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.]], requires_grad=False)
        self.all_constants["SamplePart1307"] = torch.tensor([[1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.]], requires_grad=False)
        self.all_constants["SamplePart1310"] = torch.tensor([[1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.]], requires_grad=False)
        self.all_constants["SamplePart1313"] = torch.tensor([[1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.]], requires_grad=False)
        self.all_constants["SamplePart1361"] = torch.tensor([[1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.]], requires_grad=False)
        self.all_constants["SamplePart1364"] = torch.tensor([[1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.]], requires_grad=False)
        self.all_constants["SamplePart1367"] = torch.tensor([[1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.]], requires_grad=False)
        self.all_constants["SamplePart1370"] = torch.tensor([[1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.]], requires_grad=False)
        self.all_constants["SamplePart1373"] = torch.tensor([[1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.]], requires_grad=False)
        self.all_constants["SamplePart1376"] = torch.tensor([[1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.]], requires_grad=False)
        self.all_constants["SamplePart1378"] = torch.tensor([[1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.]], requires_grad=False)
        self.all_constants["SamplePart1380"] = torch.tensor([[1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.]], requires_grad=False)
        self.all_constants["SamplePart2424"] = torch.tensor([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.]], requires_grad=False)
        self.all_constants["SamplePart2428"] = torch.tensor([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.]], requires_grad=False)
        self.all_constants["SamplePart2430"] = torch.tensor([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.]], requires_grad=False)
        self.all_constants["SamplePart2433"] = torch.tensor([[1.]], requires_grad=False)
        self.all_constants["SamplePart2435"] = torch.tensor([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.]], requires_grad=False)
        self.all_constants["Select1321"] = torch.tensor([1., 0., 0., 0.], requires_grad=False)
        self.all_constants["Select1330"] = torch.tensor([0., 1., 0., 0.], requires_grad=False)
        self.all_constants["Select1339"] = torch.tensor([0., 0., 1., 0.], requires_grad=False)
        self.all_constants["Select1348"] = torch.tensor([0., 0., 0., 1.], requires_grad=False)
        self.all_constants["Select1353"] = torch.tensor([1., 0.], requires_grad=False)
        self.all_constants["Select1356"] = torch.tensor([0., 1.], requires_grad=False)
        self.all_constants["Select1396"] = torch.tensor([1., 0., 0.], requires_grad=False)
        self.all_constants["Select1397"] = torch.tensor([1., 0., 0., 0., 0.], requires_grad=False)
        self.all_constants["Select1399"] = torch.tensor([1., 0., 0.], requires_grad=False)
        self.all_constants["Select1417"] = torch.tensor([1., 0., 0.], requires_grad=False)
        self.all_constants["Select1418"] = torch.tensor([1., 0., 0., 0., 0.], requires_grad=False)
        self.all_constants["Select1420"] = torch.tensor([0., 1., 0.], requires_grad=False)
        self.all_constants["Select1438"] = torch.tensor([1., 0., 0.], requires_grad=False)
        self.all_constants["Select1439"] = torch.tensor([1., 0., 0., 0., 0.], requires_grad=False)
        self.all_constants["Select1441"] = torch.tensor([0., 0., 1.], requires_grad=False)
        self.all_constants["Select1459"] = torch.tensor([1., 0., 0.], requires_grad=False)
        self.all_constants["Select1460"] = torch.tensor([0., 1., 0., 0., 0.], requires_grad=False)
        self.all_constants["Select1462"] = torch.tensor([1., 0., 0.], requires_grad=False)
        self.all_constants["Select1480"] = torch.tensor([1., 0., 0.], requires_grad=False)
        self.all_constants["Select1481"] = torch.tensor([0., 1., 0., 0., 0.], requires_grad=False)
        self.all_constants["Select1483"] = torch.tensor([0., 1., 0.], requires_grad=False)
        self.all_constants["Select1501"] = torch.tensor([1., 0., 0.], requires_grad=False)
        self.all_constants["Select1502"] = torch.tensor([0., 1., 0., 0., 0.], requires_grad=False)
        self.all_constants["Select1504"] = torch.tensor([0., 0., 1.], requires_grad=False)
        self.all_constants["Select1522"] = torch.tensor([1., 0., 0.], requires_grad=False)
        self.all_constants["Select1523"] = torch.tensor([0., 0., 1., 0., 0.], requires_grad=False)
        self.all_constants["Select1525"] = torch.tensor([1., 0., 0.], requires_grad=False)
        self.all_constants["Select1543"] = torch.tensor([1., 0., 0.], requires_grad=False)
        self.all_constants["Select1544"] = torch.tensor([0., 0., 1., 0., 0.], requires_grad=False)
        self.all_constants["Select1546"] = torch.tensor([0., 1., 0.], requires_grad=False)
        self.all_constants["Select1564"] = torch.tensor([1., 0., 0.], requires_grad=False)
        self.all_constants["Select1565"] = torch.tensor([0., 0., 1., 0., 0.], requires_grad=False)
        self.all_constants["Select1567"] = torch.tensor([0., 0., 1.], requires_grad=False)
        self.all_constants["Select1585"] = torch.tensor([1., 0., 0.], requires_grad=False)
        self.all_constants["Select1586"] = torch.tensor([0., 0., 0., 1., 0.], requires_grad=False)
        self.all_constants["Select1588"] = torch.tensor([1., 0., 0.], requires_grad=False)
        self.all_constants["Select1606"] = torch.tensor([1., 0., 0.], requires_grad=False)
        self.all_constants["Select1607"] = torch.tensor([0., 0., 0., 1., 0.], requires_grad=False)
        self.all_constants["Select1609"] = torch.tensor([0., 1., 0.], requires_grad=False)
        self.all_constants["Select1627"] = torch.tensor([1., 0., 0.], requires_grad=False)
        self.all_constants["Select1628"] = torch.tensor([0., 0., 0., 1., 0.], requires_grad=False)
        self.all_constants["Select1630"] = torch.tensor([0., 0., 1.], requires_grad=False)
        self.all_constants["Select1648"] = torch.tensor([1., 0., 0.], requires_grad=False)
        self.all_constants["Select1649"] = torch.tensor([0., 0., 0., 0., 1.], requires_grad=False)
        self.all_constants["Select1651"] = torch.tensor([1., 0., 0.], requires_grad=False)
        self.all_constants["Select1669"] = torch.tensor([1., 0., 0.], requires_grad=False)
        self.all_constants["Select1670"] = torch.tensor([0., 0., 0., 0., 1.], requires_grad=False)
        self.all_constants["Select1672"] = torch.tensor([0., 1., 0.], requires_grad=False)
        self.all_constants["Select1690"] = torch.tensor([1., 0., 0.], requires_grad=False)
        self.all_constants["Select1691"] = torch.tensor([0., 0., 0., 0., 1.], requires_grad=False)
        self.all_constants["Select1693"] = torch.tensor([0., 0., 1.], requires_grad=False)
        self.all_constants["Select1711"] = torch.tensor([0., 1., 0.], requires_grad=False)
        self.all_constants["Select1712"] = torch.tensor([1., 0., 0., 0., 0.], requires_grad=False)
        self.all_constants["Select1714"] = torch.tensor([1., 0., 0.], requires_grad=False)
        self.all_constants["Select1732"] = torch.tensor([0., 1., 0.], requires_grad=False)
        self.all_constants["Select1733"] = torch.tensor([1., 0., 0., 0., 0.], requires_grad=False)
        self.all_constants["Select1735"] = torch.tensor([0., 1., 0.], requires_grad=False)
        self.all_constants["Select1753"] = torch.tensor([0., 1., 0.], requires_grad=False)
        self.all_constants["Select1754"] = torch.tensor([1., 0., 0., 0., 0.], requires_grad=False)
        self.all_constants["Select1756"] = torch.tensor([0., 0., 1.], requires_grad=False)
        self.all_constants["Select1774"] = torch.tensor([0., 1., 0.], requires_grad=False)
        self.all_constants["Select1775"] = torch.tensor([0., 1., 0., 0., 0.], requires_grad=False)
        self.all_constants["Select1777"] = torch.tensor([1., 0., 0.], requires_grad=False)
        self.all_constants["Select1795"] = torch.tensor([0., 1., 0.], requires_grad=False)
        self.all_constants["Select1796"] = torch.tensor([0., 1., 0., 0., 0.], requires_grad=False)
        self.all_constants["Select1798"] = torch.tensor([0., 1., 0.], requires_grad=False)
        self.all_constants["Select1816"] = torch.tensor([0., 1., 0.], requires_grad=False)
        self.all_constants["Select1817"] = torch.tensor([0., 1., 0., 0., 0.], requires_grad=False)
        self.all_constants["Select1819"] = torch.tensor([0., 0., 1.], requires_grad=False)
        self.all_constants["Select1837"] = torch.tensor([0., 1., 0.], requires_grad=False)
        self.all_constants["Select1838"] = torch.tensor([0., 0., 1., 0., 0.], requires_grad=False)
        self.all_constants["Select1840"] = torch.tensor([1., 0., 0.], requires_grad=False)
        self.all_constants["Select1858"] = torch.tensor([0., 1., 0.], requires_grad=False)
        self.all_constants["Select1859"] = torch.tensor([0., 0., 1., 0., 0.], requires_grad=False)
        self.all_constants["Select1861"] = torch.tensor([0., 1., 0.], requires_grad=False)
        self.all_constants["Select1879"] = torch.tensor([0., 1., 0.], requires_grad=False)
        self.all_constants["Select1880"] = torch.tensor([0., 0., 1., 0., 0.], requires_grad=False)
        self.all_constants["Select1882"] = torch.tensor([0., 0., 1.], requires_grad=False)
        self.all_constants["Select1900"] = torch.tensor([0., 1., 0.], requires_grad=False)
        self.all_constants["Select1901"] = torch.tensor([0., 0., 0., 1., 0.], requires_grad=False)
        self.all_constants["Select1903"] = torch.tensor([1., 0., 0.], requires_grad=False)
        self.all_constants["Select1921"] = torch.tensor([0., 1., 0.], requires_grad=False)
        self.all_constants["Select1922"] = torch.tensor([0., 0., 0., 1., 0.], requires_grad=False)
        self.all_constants["Select1924"] = torch.tensor([0., 1., 0.], requires_grad=False)
        self.all_constants["Select1942"] = torch.tensor([0., 1., 0.], requires_grad=False)
        self.all_constants["Select1943"] = torch.tensor([0., 0., 0., 1., 0.], requires_grad=False)
        self.all_constants["Select1945"] = torch.tensor([0., 0., 1.], requires_grad=False)
        self.all_constants["Select1963"] = torch.tensor([0., 1., 0.], requires_grad=False)
        self.all_constants["Select1964"] = torch.tensor([0., 0., 0., 0., 1.], requires_grad=False)
        self.all_constants["Select1966"] = torch.tensor([1., 0., 0.], requires_grad=False)
        self.all_constants["Select1984"] = torch.tensor([0., 1., 0.], requires_grad=False)
        self.all_constants["Select1985"] = torch.tensor([0., 0., 0., 0., 1.], requires_grad=False)
        self.all_constants["Select1987"] = torch.tensor([0., 1., 0.], requires_grad=False)
        self.all_constants["Select2005"] = torch.tensor([0., 1., 0.], requires_grad=False)
        self.all_constants["Select2006"] = torch.tensor([0., 0., 0., 0., 1.], requires_grad=False)
        self.all_constants["Select2008"] = torch.tensor([0., 0., 1.], requires_grad=False)
        self.all_constants["Select2026"] = torch.tensor([0., 0., 1.], requires_grad=False)
        self.all_constants["Select2027"] = torch.tensor([1., 0., 0., 0., 0.], requires_grad=False)
        self.all_constants["Select2029"] = torch.tensor([1., 0., 0.], requires_grad=False)
        self.all_constants["Select2047"] = torch.tensor([0., 0., 1.], requires_grad=False)
        self.all_constants["Select2048"] = torch.tensor([1., 0., 0., 0., 0.], requires_grad=False)
        self.all_constants["Select2050"] = torch.tensor([0., 1., 0.], requires_grad=False)
        self.all_constants["Select2068"] = torch.tensor([0., 0., 1.], requires_grad=False)
        self.all_constants["Select2069"] = torch.tensor([1., 0., 0., 0., 0.], requires_grad=False)
        self.all_constants["Select2071"] = torch.tensor([0., 0., 1.], requires_grad=False)
        self.all_constants["Select2089"] = torch.tensor([0., 0., 1.], requires_grad=False)
        self.all_constants["Select2090"] = torch.tensor([0., 1., 0., 0., 0.], requires_grad=False)
        self.all_constants["Select2092"] = torch.tensor([1., 0., 0.], requires_grad=False)
        self.all_constants["Select2110"] = torch.tensor([0., 0., 1.], requires_grad=False)
        self.all_constants["Select2111"] = torch.tensor([0., 1., 0., 0., 0.], requires_grad=False)
        self.all_constants["Select2113"] = torch.tensor([0., 1., 0.], requires_grad=False)
        self.all_constants["Select2131"] = torch.tensor([0., 0., 1.], requires_grad=False)
        self.all_constants["Select2132"] = torch.tensor([0., 1., 0., 0., 0.], requires_grad=False)
        self.all_constants["Select2134"] = torch.tensor([0., 0., 1.], requires_grad=False)
        self.all_constants["Select2152"] = torch.tensor([0., 0., 1.], requires_grad=False)
        self.all_constants["Select2153"] = torch.tensor([0., 0., 1., 0., 0.], requires_grad=False)
        self.all_constants["Select2155"] = torch.tensor([1., 0., 0.], requires_grad=False)
        self.all_constants["Select2173"] = torch.tensor([0., 0., 1.], requires_grad=False)
        self.all_constants["Select2174"] = torch.tensor([0., 0., 1., 0., 0.], requires_grad=False)
        self.all_constants["Select2176"] = torch.tensor([0., 1., 0.], requires_grad=False)
        self.all_constants["Select2194"] = torch.tensor([0., 0., 1.], requires_grad=False)
        self.all_constants["Select2195"] = torch.tensor([0., 0., 1., 0., 0.], requires_grad=False)
        self.all_constants["Select2197"] = torch.tensor([0., 0., 1.], requires_grad=False)
        self.all_constants["Select2215"] = torch.tensor([0., 0., 1.], requires_grad=False)
        self.all_constants["Select2216"] = torch.tensor([0., 0., 0., 1., 0.], requires_grad=False)
        self.all_constants["Select2218"] = torch.tensor([1., 0., 0.], requires_grad=False)
        self.all_constants["Select2236"] = torch.tensor([0., 0., 1.], requires_grad=False)
        self.all_constants["Select2237"] = torch.tensor([0., 0., 0., 1., 0.], requires_grad=False)
        self.all_constants["Select2239"] = torch.tensor([0., 1., 0.], requires_grad=False)
        self.all_constants["Select2257"] = torch.tensor([0., 0., 1.], requires_grad=False)
        self.all_constants["Select2258"] = torch.tensor([0., 0., 0., 1., 0.], requires_grad=False)
        self.all_constants["Select2260"] = torch.tensor([0., 0., 1.], requires_grad=False)
        self.all_constants["Select2278"] = torch.tensor([0., 0., 1.], requires_grad=False)
        self.all_constants["Select2279"] = torch.tensor([0., 0., 0., 0., 1.], requires_grad=False)
        self.all_constants["Select2281"] = torch.tensor([1., 0., 0.], requires_grad=False)
        self.all_constants["Select2299"] = torch.tensor([0., 0., 1.], requires_grad=False)
        self.all_constants["Select2300"] = torch.tensor([0., 0., 0., 0., 1.], requires_grad=False)
        self.all_constants["Select2302"] = torch.tensor([0., 1., 0.], requires_grad=False)
        self.all_constants["Select2320"] = torch.tensor([0., 0., 1.], requires_grad=False)
        self.all_constants["Select2321"] = torch.tensor([0., 0., 0., 0., 1.], requires_grad=False)
        self.all_constants["Select2323"] = torch.tensor([0., 0., 1.], requires_grad=False)
        self.all_constants["Select2370"] = torch.tensor([1., 0., 0.], requires_grad=False)
        self.all_constants["Select2371"] = torch.tensor([1., 0., 0.], requires_grad=False)
        self.all_constants["Select2375"] = torch.tensor([1., 0., 0.], requires_grad=False)
        self.all_constants["Select2376"] = torch.tensor([0., 1., 0.], requires_grad=False)
        self.all_constants["Select2380"] = torch.tensor([1., 0., 0.], requires_grad=False)
        self.all_constants["Select2381"] = torch.tensor([0., 0., 1.], requires_grad=False)
        self.all_constants["Select2385"] = torch.tensor([0., 1., 0.], requires_grad=False)
        self.all_constants["Select2386"] = torch.tensor([1., 0., 0.], requires_grad=False)
        self.all_constants["Select2390"] = torch.tensor([0., 1., 0.], requires_grad=False)
        self.all_constants["Select2391"] = torch.tensor([0., 1., 0.], requires_grad=False)
        self.all_constants["Select2395"] = torch.tensor([0., 1., 0.], requires_grad=False)
        self.all_constants["Select2396"] = torch.tensor([0., 0., 1.], requires_grad=False)
        self.all_constants["Select2400"] = torch.tensor([0., 0., 1.], requires_grad=False)
        self.all_constants["Select2401"] = torch.tensor([1., 0., 0.], requires_grad=False)
        self.all_constants["Select2405"] = torch.tensor([0., 0., 1.], requires_grad=False)
        self.all_constants["Select2406"] = torch.tensor([0., 1., 0.], requires_grad=False)
        self.all_constants["Select2410"] = torch.tensor([0., 0., 1.], requires_grad=False)
        self.all_constants["Select2411"] = torch.tensor([0., 0., 1.], requires_grad=False)
        self.all_parameters = torch.nn.ParameterDict(self.all_parameters)
        self.all_constants = torch.nn.ParameterDict(self.all_constants)

    def update(self, closed_loop={}, connect={}, disconnect=False):
        pass
    def forward(self, delta, vel, accy, crt, acc, yaw_rate):
        getitem = delta
        relation_forward_sample_part2430_w = self.all_constants.SamplePart2430
        einsum = torch.functional.einsum('bij,ki->bkj', getitem, relation_forward_sample_part2430_w);  getitem = relation_forward_sample_part2430_w = None
        getitem_1 = vel
        relation_forward_sample_part1307_w = self.all_constants.SamplePart1307
        einsum_1 = torch.functional.einsum('bij,ki->bkj', getitem_1, relation_forward_sample_part1307_w);  getitem_1 = relation_forward_sample_part1307_w = None
        zeros_like = torch.zeros_like(einsum_1)
        repeat = zeros_like.repeat(1, 1, 2);  zeros_like = None
        sub = einsum_1 - -0.3245462443934415
        neg = -sub;  sub = None
        truediv = neg / 3.928823281751481;  neg = None
        add = truediv + 1;  truediv = None
        _tensor_constant69 = self._tensor_constant69
        maximum = torch.maximum(add, _tensor_constant69);  add = _tensor_constant69 = None
        _tensor_constant70 = self._tensor_constant70
        minimum = torch.minimum(maximum, _tensor_constant70);  maximum = _tensor_constant70 = None
        _tensor_constant71 = self._tensor_constant71
        slicing = nnodely_layers_fuzzify_slicing(repeat, _tensor_constant71, minimum);  _tensor_constant71 = minimum = slicing = None
        sub_1 = einsum_1 - -0.3245462443934415;  einsum_1 = None
        truediv_1 = sub_1 / 3.928823281751481;  sub_1 = None
        _tensor_constant72 = self._tensor_constant72
        maximum_1 = torch.maximum(truediv_1, _tensor_constant72);  truediv_1 = _tensor_constant72 = None
        _tensor_constant73 = self._tensor_constant73
        minimum_1 = torch.minimum(maximum_1, _tensor_constant73);  maximum_1 = _tensor_constant73 = None
        _tensor_constant74 = self._tensor_constant74
        slicing_1 = nnodely_layers_fuzzify_slicing(repeat, _tensor_constant74, minimum_1);  _tensor_constant74 = minimum_1 = slicing_1 = None
        relation_forward_select1356_w = self.all_constants.Select1356
        einsum_2 = torch.functional.einsum('ijk,k->ij', repeat, relation_forward_select1356_w);  relation_forward_select1356_w = None
        unsqueeze = einsum_2.unsqueeze(2);  einsum_2 = None
        getitem_2 = accy
        relation_forward_sample_part1310_w = self.all_constants.SamplePart1310
        einsum_3 = torch.functional.einsum('bij,ki->bkj', getitem_2, relation_forward_sample_part1310_w);  getitem_2 = relation_forward_sample_part1310_w = None
        zeros_like_1 = torch.zeros_like(einsum_3)
        repeat_1 = zeros_like_1.repeat(1, 1, 4);  zeros_like_1 = None
        sub_2 = einsum_3 - -3.7472697844131804
        neg_1 = -sub_2;  sub_2 = None
        truediv_2 = neg_1 / 3.463383508469832;  neg_1 = None
        add_1 = truediv_2 + 1;  truediv_2 = None
        _tensor_constant75 = self._tensor_constant75
        maximum_2 = torch.maximum(add_1, _tensor_constant75);  add_1 = _tensor_constant75 = None
        _tensor_constant76 = self._tensor_constant76
        minimum_2 = torch.minimum(maximum_2, _tensor_constant76);  maximum_2 = _tensor_constant76 = None
        _tensor_constant77 = self._tensor_constant77
        slicing_2 = nnodely_layers_fuzzify_slicing(repeat_1, _tensor_constant77, minimum_2);  _tensor_constant77 = minimum_2 = slicing_2 = None
        sub_3 = einsum_3 - -3.7472697844131804
        truediv_3 = sub_3 / 3.463383508469832;  sub_3 = None
        _tensor_constant78 = self._tensor_constant78
        maximum_3 = torch.maximum(truediv_3, _tensor_constant78);  truediv_3 = _tensor_constant78 = None
        sub_4 = einsum_3 - -0.2838862759433485
        neg_2 = -sub_4;  sub_4 = None
        truediv_4 = neg_2 / 3.463383508469832;  neg_2 = None
        add_2 = truediv_4 + 1;  truediv_4 = None
        _tensor_constant79 = self._tensor_constant79
        maximum_4 = torch.maximum(add_2, _tensor_constant79);  add_2 = _tensor_constant79 = None
        minimum_3 = torch.minimum(maximum_3, maximum_4);  maximum_3 = maximum_4 = None
        _tensor_constant80 = self._tensor_constant80
        slicing_3 = nnodely_layers_fuzzify_slicing(repeat_1, _tensor_constant80, minimum_3);  _tensor_constant80 = minimum_3 = slicing_3 = None
        sub_5 = einsum_3 - -0.2838862759433485
        truediv_5 = sub_5 / 3.463383508469832;  sub_5 = None
        _tensor_constant81 = self._tensor_constant81
        maximum_5 = torch.maximum(truediv_5, _tensor_constant81);  truediv_5 = _tensor_constant81 = None
        sub_6 = einsum_3 - 3.1794972325264834
        neg_3 = -sub_6;  sub_6 = None
        truediv_6 = neg_3 / 3.4633835084698323;  neg_3 = None
        add_3 = truediv_6 + 1;  truediv_6 = None
        _tensor_constant82 = self._tensor_constant82
        maximum_6 = torch.maximum(add_3, _tensor_constant82);  add_3 = _tensor_constant82 = None
        minimum_4 = torch.minimum(maximum_5, maximum_6);  maximum_5 = maximum_6 = None
        _tensor_constant83 = self._tensor_constant83
        slicing_4 = nnodely_layers_fuzzify_slicing(repeat_1, _tensor_constant83, minimum_4);  _tensor_constant83 = minimum_4 = slicing_4 = None
        sub_7 = einsum_3 - 3.1794972325264834;  einsum_3 = None
        truediv_7 = sub_7 / 3.4633835084698323;  sub_7 = None
        _tensor_constant84 = self._tensor_constant84
        maximum_7 = torch.maximum(truediv_7, _tensor_constant84);  truediv_7 = _tensor_constant84 = None
        _tensor_constant85 = self._tensor_constant85
        minimum_5 = torch.minimum(maximum_7, _tensor_constant85);  maximum_7 = _tensor_constant85 = None
        _tensor_constant86 = self._tensor_constant86
        slicing_5 = nnodely_layers_fuzzify_slicing(repeat_1, _tensor_constant86, minimum_5);  _tensor_constant86 = minimum_5 = slicing_5 = None
        relation_forward_select1348_w = self.all_constants.Select1348
        einsum_4 = torch.functional.einsum('ijk,k->ij', repeat_1, relation_forward_select1348_w);  relation_forward_select1348_w = None
        unsqueeze_1 = einsum_4.unsqueeze(2);  einsum_4 = None
        getitem_3 = accy
        relation_forward_sample_part1313_w = self.all_constants.SamplePart1313
        einsum_5 = torch.functional.einsum('bij,ki->bkj', getitem_3, relation_forward_sample_part1313_w);  getitem_3 = relation_forward_sample_part1313_w = None
        all_constants_ay_center_3 = self.all_constants.ay_center_3
        all_parameters_k1_3 = self.all_parameters.k1_3
        all_parameters_k2_3 = self.all_parameters.k2_3
        long_corr_lat_local = nnodely_layers_parametricfunction_long_corr_lat_local(einsum_5, all_constants_ay_center_3, all_parameters_k1_3, all_parameters_k2_3);  all_constants_ay_center_3 = all_parameters_k1_3 = all_parameters_k2_3 = None
        mul = long_corr_lat_local * unsqueeze_1;  long_corr_lat_local = unsqueeze_1 = None
        relation_forward_select1339_w = self.all_constants.Select1339
        einsum_6 = torch.functional.einsum('ijk,k->ij', repeat_1, relation_forward_select1339_w);  relation_forward_select1339_w = None
        unsqueeze_2 = einsum_6.unsqueeze(2);  einsum_6 = None
        all_constants_ay_center_2 = self.all_constants.ay_center_2
        all_parameters_k1_2 = self.all_parameters.k1_2
        all_parameters_k2_2 = self.all_parameters.k2_2
        long_corr_lat_local_1 = nnodely_layers_parametricfunction_long_corr_lat_local(einsum_5, all_constants_ay_center_2, all_parameters_k1_2, all_parameters_k2_2);  all_constants_ay_center_2 = all_parameters_k1_2 = all_parameters_k2_2 = None
        mul_1 = long_corr_lat_local_1 * unsqueeze_2;  long_corr_lat_local_1 = unsqueeze_2 = None
        relation_forward_select1330_w = self.all_constants.Select1330
        einsum_7 = torch.functional.einsum('ijk,k->ij', repeat_1, relation_forward_select1330_w);  relation_forward_select1330_w = None
        unsqueeze_3 = einsum_7.unsqueeze(2);  einsum_7 = None
        all_constants_ay_center_1 = self.all_constants.ay_center_1
        all_parameters_k1_1 = self.all_parameters.k1_1
        all_parameters_k2_1 = self.all_parameters.k2_1
        long_corr_lat_local_2 = nnodely_layers_parametricfunction_long_corr_lat_local(einsum_5, all_constants_ay_center_1, all_parameters_k1_1, all_parameters_k2_1);  all_constants_ay_center_1 = all_parameters_k1_1 = all_parameters_k2_1 = None
        mul_2 = long_corr_lat_local_2 * unsqueeze_3;  long_corr_lat_local_2 = unsqueeze_3 = None
        relation_forward_select1321_w = self.all_constants.Select1321
        einsum_8 = torch.functional.einsum('ijk,k->ij', repeat_1, relation_forward_select1321_w);  repeat_1 = relation_forward_select1321_w = None
        unsqueeze_4 = einsum_8.unsqueeze(2);  einsum_8 = None
        all_constants_ay_center_0 = self.all_constants.ay_center_0
        all_parameters_k1_0 = self.all_parameters.k1_0
        all_parameters_k2_0 = self.all_parameters.k2_0
        long_corr_lat_local_3 = nnodely_layers_parametricfunction_long_corr_lat_local(einsum_5, all_constants_ay_center_0, all_parameters_k1_0, all_parameters_k2_0);  einsum_5 = all_constants_ay_center_0 = all_parameters_k1_0 = all_parameters_k2_0 = None
        mul_3 = long_corr_lat_local_3 * unsqueeze_4;  long_corr_lat_local_3 = unsqueeze_4 = None
        add_4 = mul_3 + mul_2;  mul_3 = mul_2 = None
        add_5 = add_4 + mul_1;  add_4 = mul_1 = None
        add_6 = add_5 + mul;  add_5 = mul = None
        mul_4 = add_6 * unsqueeze;  unsqueeze = None
        size = mul_4.size(0)
        relation_forward_fir1358_weights = self.all_parameters.PFir436W
        size_1 = relation_forward_fir1358_weights.size(1)
        squeeze = mul_4.squeeze(-1);  mul_4 = None
        matmul = torch.matmul(squeeze, relation_forward_fir1358_weights);  squeeze = relation_forward_fir1358_weights = None
        to = matmul.to(dtype = torch.float32);  matmul = None
        view = to.view(size, 1, size_1);  to = size = size_1 = None
        relation_forward_select1353_w = self.all_constants.Select1353
        einsum_9 = torch.functional.einsum('ijk,k->ij', repeat, relation_forward_select1353_w);  repeat = relation_forward_select1353_w = None
        unsqueeze_5 = einsum_9.unsqueeze(2);  einsum_9 = None
        mul_5 = add_6 * unsqueeze_5;  add_6 = unsqueeze_5 = None
        size_2 = mul_5.size(0)
        relation_forward_fir1355_weights = self.all_parameters.PFir434W
        size_3 = relation_forward_fir1355_weights.size(1)
        squeeze_1 = mul_5.squeeze(-1);  mul_5 = None
        matmul_1 = torch.matmul(squeeze_1, relation_forward_fir1355_weights);  squeeze_1 = relation_forward_fir1355_weights = None
        to_1 = matmul_1.to(dtype = torch.float32);  matmul_1 = None
        view_1 = to_1.view(size_2, 1, size_3);  to_1 = size_2 = size_3 = None
        add_7 = view_1 + view;  view_1 = view = None
        getitem_4 = vel
        relation_forward_sample_part2428_w = self.all_constants.SamplePart2428
        einsum_10 = torch.functional.einsum('bij,ki->bkj', getitem_4, relation_forward_sample_part2428_w);  getitem_4 = relation_forward_sample_part2428_w = None
        getitem_5 = crt
        relation_forward_sample_part1295_w = self.all_constants.SamplePart1295
        einsum_11 = torch.functional.einsum('bij,ki->bkj', getitem_5, relation_forward_sample_part1295_w);  getitem_5 = relation_forward_sample_part1295_w = None
        acc_pos = nnodely_layers_parametricfunction_acc_pos(einsum_11);  einsum_11 = None
        size_4 = acc_pos.size(0)
        relation_forward_fir1300_weights = self.all_parameters.PFir407W
        size_5 = relation_forward_fir1300_weights.size(1)
        squeeze_2 = acc_pos.squeeze(-1);  acc_pos = None
        matmul_2 = torch.matmul(squeeze_2, relation_forward_fir1300_weights);  squeeze_2 = relation_forward_fir1300_weights = None
        to_2 = matmul_2.to(dtype = torch.float32);  matmul_2 = None
        view_2 = to_2.view(size_4, 1, size_5);  to_2 = size_4 = size_5 = None
        relation_forward_fir1300_bias = self.all_parameters.PFir407b
        add_8 = view_2 + relation_forward_fir1300_bias;  view_2 = relation_forward_fir1300_bias = None
        relation_forward_fir1300_dropout = self.Fir1300(add_8);  add_8 = None
        tanh = torch.tanh(relation_forward_fir1300_dropout);  relation_forward_fir1300_dropout = None
        relation_forward_linear1302_weights = self.all_parameters.PLinear410W
        einsum_12 = torch.functional.einsum('bwi,io->bwo', tanh, relation_forward_linear1302_weights);  tanh = relation_forward_linear1302_weights = None
        getitem_6 = crt
        relation_forward_sample_part1298_w = self.all_constants.SamplePart1298
        einsum_13 = torch.functional.einsum('bij,ki->bkj', getitem_6, relation_forward_sample_part1298_w);  getitem_6 = relation_forward_sample_part1298_w = None
        acc_neg = nnodely_layers_parametricfunction_acc_neg(einsum_13);  einsum_13 = None
        size_6 = acc_neg.size(0)
        relation_forward_fir1303_weights = self.all_parameters.PFir412W
        size_7 = relation_forward_fir1303_weights.size(1)
        squeeze_3 = acc_neg.squeeze(-1);  acc_neg = None
        matmul_3 = torch.matmul(squeeze_3, relation_forward_fir1303_weights);  squeeze_3 = relation_forward_fir1303_weights = None
        to_3 = matmul_3.to(dtype = torch.float32);  matmul_3 = None
        view_3 = to_3.view(size_6, 1, size_7);  to_3 = size_6 = size_7 = None
        relation_forward_fir1303_bias = self.all_parameters.PFir412b
        add_9 = view_3 + relation_forward_fir1303_bias;  view_3 = relation_forward_fir1303_bias = None
        relation_forward_fir1303_dropout = self.Fir1303(add_9);  add_9 = None
        tanh_1 = torch.tanh(relation_forward_fir1303_dropout);  relation_forward_fir1303_dropout = None
        relation_forward_linear1305_weights = self.all_parameters.PLinear415W
        einsum_14 = torch.functional.einsum('bwi,io->bwo', tanh_1, relation_forward_linear1305_weights);  tanh_1 = relation_forward_linear1305_weights = None
        add_10 = einsum_14 + einsum_12;  einsum_14 = einsum_12 = None
        all_parameters_r1 = self.all_parameters.r1
        all_parameters_mass = self.all_parameters.mass
        all_parameters_kd = self.all_parameters.Kd
        all_parameters_cv = self.all_parameters.Cv
        all_parameters_cr = self.all_parameters.Cr
        all_parameters_iw1 = self.all_parameters.Iw1
        acc_model_based = nnodely_layers_parametricfunction_acc_model_based(add_10, einsum_10, add_7, einsum, all_parameters_r1, all_parameters_mass, all_parameters_kd, all_parameters_cv, all_parameters_cr, all_parameters_iw1);  add_10 = einsum_10 = add_7 = einsum = all_parameters_r1 = all_parameters_mass = all_parameters_kd = all_parameters_cv = all_parameters_cr = all_parameters_iw1 = None
        getitem_7 = acc
        relation_forward_sample_part2435_w = self.all_constants.SamplePart2435
        einsum_15 = torch.functional.einsum('bij,ki->bkj', getitem_7, relation_forward_sample_part2435_w);  getitem_7 = relation_forward_sample_part2435_w = None
        getitem_8 = acc
        relation_forward_sample_part1364_w = self.all_constants.SamplePart1364
        einsum_16 = torch.functional.einsum('bij,ki->bkj', getitem_8, relation_forward_sample_part1364_w);  getitem_8 = relation_forward_sample_part1364_w = None
        zeros_like_2 = torch.zeros_like(einsum_16)
        repeat_2 = zeros_like_2.repeat(1, 1, 3);  zeros_like_2 = None
        sub_8 = einsum_16 - -4.772998923371973
        neg_4 = -sub_8;  sub_8 = None
        truediv_8 = neg_4 / 5.266015368825087;  neg_4 = None
        add_11 = truediv_8 + 1;  truediv_8 = None
        _tensor_constant87 = self._tensor_constant87
        maximum_8 = torch.maximum(add_11, _tensor_constant87);  add_11 = _tensor_constant87 = None
        _tensor_constant88 = self._tensor_constant88
        minimum_6 = torch.minimum(maximum_8, _tensor_constant88);  maximum_8 = _tensor_constant88 = None
        _tensor_constant89 = self._tensor_constant89
        slicing_6 = nnodely_layers_fuzzify_slicing(repeat_2, _tensor_constant89, minimum_6);  _tensor_constant89 = minimum_6 = slicing_6 = None
        sub_9 = einsum_16 - -4.772998923371973
        truediv_9 = sub_9 / 5.266015368825087;  sub_9 = None
        _tensor_constant90 = self._tensor_constant90
        maximum_9 = torch.maximum(truediv_9, _tensor_constant90);  truediv_9 = _tensor_constant90 = None
        sub_10 = einsum_16 - 0.4930164454531143
        neg_5 = -sub_10;  sub_10 = None
        truediv_10 = neg_5 / 5.266015368825088;  neg_5 = None
        add_12 = truediv_10 + 1;  truediv_10 = None
        _tensor_constant91 = self._tensor_constant91
        maximum_10 = torch.maximum(add_12, _tensor_constant91);  add_12 = _tensor_constant91 = None
        minimum_7 = torch.minimum(maximum_9, maximum_10);  maximum_9 = maximum_10 = None
        _tensor_constant92 = self._tensor_constant92
        slicing_7 = nnodely_layers_fuzzify_slicing(repeat_2, _tensor_constant92, minimum_7);  _tensor_constant92 = minimum_7 = slicing_7 = None
        sub_11 = einsum_16 - 0.4930164454531143;  einsum_16 = None
        truediv_11 = sub_11 / 5.266015368825088;  sub_11 = None
        _tensor_constant93 = self._tensor_constant93
        maximum_11 = torch.maximum(truediv_11, _tensor_constant93);  truediv_11 = _tensor_constant93 = None
        _tensor_constant94 = self._tensor_constant94
        minimum_8 = torch.minimum(maximum_11, _tensor_constant94);  maximum_11 = _tensor_constant94 = None
        _tensor_constant95 = self._tensor_constant95
        slicing_8 = nnodely_layers_fuzzify_slicing(repeat_2, _tensor_constant95, minimum_8);  _tensor_constant95 = minimum_8 = slicing_8 = None
        relation_forward_select2411_w = self.all_constants.Select2411
        einsum_17 = torch.functional.einsum('ijk,k->ij', repeat_2, relation_forward_select2411_w);  relation_forward_select2411_w = None
        unsqueeze_6 = einsum_17.unsqueeze(2);  einsum_17 = None
        getitem_9 = vel
        relation_forward_sample_part1361_w = self.all_constants.SamplePart1361
        einsum_18 = torch.functional.einsum('bij,ki->bkj', getitem_9, relation_forward_sample_part1361_w);  getitem_9 = relation_forward_sample_part1361_w = None
        zeros_like_3 = torch.zeros_like(einsum_18)
        repeat_3 = zeros_like_3.repeat(1, 1, 3);  zeros_like_3 = None
        sub_12 = einsum_18 - -0.3245462443934415
        neg_6 = -sub_12;  sub_12 = None
        truediv_12 = neg_6 / 1.9644116408757406;  neg_6 = None
        add_13 = truediv_12 + 1;  truediv_12 = None
        _tensor_constant96 = self._tensor_constant96
        maximum_12 = torch.maximum(add_13, _tensor_constant96);  add_13 = _tensor_constant96 = None
        _tensor_constant97 = self._tensor_constant97
        minimum_9 = torch.minimum(maximum_12, _tensor_constant97);  maximum_12 = _tensor_constant97 = None
        _tensor_constant98 = self._tensor_constant98
        slicing_9 = nnodely_layers_fuzzify_slicing(repeat_3, _tensor_constant98, minimum_9);  _tensor_constant98 = minimum_9 = slicing_9 = None
        sub_13 = einsum_18 - -0.3245462443934415
        truediv_13 = sub_13 / 1.9644116408757406;  sub_13 = None
        _tensor_constant99 = self._tensor_constant99
        maximum_13 = torch.maximum(truediv_13, _tensor_constant99);  truediv_13 = _tensor_constant99 = None
        sub_14 = einsum_18 - 1.6398653964822991
        neg_7 = -sub_14;  sub_14 = None
        truediv_14 = neg_7 / 1.9644116408757404;  neg_7 = None
        add_14 = truediv_14 + 1;  truediv_14 = None
        _tensor_constant100 = self._tensor_constant100
        maximum_14 = torch.maximum(add_14, _tensor_constant100);  add_14 = _tensor_constant100 = None
        minimum_10 = torch.minimum(maximum_13, maximum_14);  maximum_13 = maximum_14 = None
        _tensor_constant101 = self._tensor_constant101
        slicing_10 = nnodely_layers_fuzzify_slicing(repeat_3, _tensor_constant101, minimum_10);  _tensor_constant101 = minimum_10 = slicing_10 = None
        sub_15 = einsum_18 - 1.6398653964822991;  einsum_18 = None
        truediv_15 = sub_15 / 1.9644116408757404;  sub_15 = None
        _tensor_constant102 = self._tensor_constant102
        maximum_15 = torch.maximum(truediv_15, _tensor_constant102);  truediv_15 = _tensor_constant102 = None
        _tensor_constant103 = self._tensor_constant103
        minimum_11 = torch.minimum(maximum_15, _tensor_constant103);  maximum_15 = _tensor_constant103 = None
        _tensor_constant104 = self._tensor_constant104
        slicing_11 = nnodely_layers_fuzzify_slicing(repeat_3, _tensor_constant104, minimum_11);  _tensor_constant104 = minimum_11 = slicing_11 = None
        relation_forward_select2410_w = self.all_constants.Select2410
        einsum_19 = torch.functional.einsum('ijk,k->ij', repeat_3, relation_forward_select2410_w);  relation_forward_select2410_w = None
        unsqueeze_7 = einsum_19.unsqueeze(2);  einsum_19 = None
        mul_6 = unsqueeze_7 * unsqueeze_6;  unsqueeze_7 = unsqueeze_6 = None
        getitem_10 = acc
        relation_forward_sample_part1373_w = self.all_constants.SamplePart1373
        einsum_20 = torch.functional.einsum('bij,ki->bkj', getitem_10, relation_forward_sample_part1373_w);  getitem_10 = relation_forward_sample_part1373_w = None
        zeros_like_4 = torch.zeros_like(einsum_20)
        repeat_4 = zeros_like_4.repeat(1, 1, 3);  zeros_like_4 = None
        sub_16 = einsum_20 - -4.772998923371973
        neg_8 = -sub_16;  sub_16 = None
        truediv_16 = neg_8 / 5.266015368825087;  neg_8 = None
        add_15 = truediv_16 + 1;  truediv_16 = None
        _tensor_constant105 = self._tensor_constant105
        maximum_16 = torch.maximum(add_15, _tensor_constant105);  add_15 = _tensor_constant105 = None
        _tensor_constant106 = self._tensor_constant106
        minimum_12 = torch.minimum(maximum_16, _tensor_constant106);  maximum_16 = _tensor_constant106 = None
        _tensor_constant107 = self._tensor_constant107
        slicing_12 = nnodely_layers_fuzzify_slicing(repeat_4, _tensor_constant107, minimum_12);  _tensor_constant107 = minimum_12 = slicing_12 = None
        sub_17 = einsum_20 - -4.772998923371973
        truediv_17 = sub_17 / 5.266015368825087;  sub_17 = None
        _tensor_constant108 = self._tensor_constant108
        maximum_17 = torch.maximum(truediv_17, _tensor_constant108);  truediv_17 = _tensor_constant108 = None
        sub_18 = einsum_20 - 0.4930164454531143
        neg_9 = -sub_18;  sub_18 = None
        truediv_18 = neg_9 / 5.266015368825088;  neg_9 = None
        add_16 = truediv_18 + 1;  truediv_18 = None
        _tensor_constant109 = self._tensor_constant109
        maximum_18 = torch.maximum(add_16, _tensor_constant109);  add_16 = _tensor_constant109 = None
        minimum_13 = torch.minimum(maximum_17, maximum_18);  maximum_17 = maximum_18 = None
        _tensor_constant110 = self._tensor_constant110
        slicing_13 = nnodely_layers_fuzzify_slicing(repeat_4, _tensor_constant110, minimum_13);  _tensor_constant110 = minimum_13 = slicing_13 = None
        sub_19 = einsum_20 - 0.4930164454531143;  einsum_20 = None
        truediv_19 = sub_19 / 5.266015368825088;  sub_19 = None
        _tensor_constant111 = self._tensor_constant111
        maximum_19 = torch.maximum(truediv_19, _tensor_constant111);  truediv_19 = _tensor_constant111 = None
        _tensor_constant112 = self._tensor_constant112
        minimum_14 = torch.minimum(maximum_19, _tensor_constant112);  maximum_19 = _tensor_constant112 = None
        _tensor_constant113 = self._tensor_constant113
        slicing_14 = nnodely_layers_fuzzify_slicing(repeat_4, _tensor_constant113, minimum_14);  _tensor_constant113 = minimum_14 = slicing_14 = None
        relation_forward_select2323_w = self.all_constants.Select2323
        einsum_21 = torch.functional.einsum('ijk,k->ij', repeat_4, relation_forward_select2323_w);  relation_forward_select2323_w = None
        unsqueeze_8 = einsum_21.unsqueeze(2);  einsum_21 = None
        getitem_11 = delta
        relation_forward_sample_part1367_w = self.all_constants.SamplePart1367
        einsum_22 = torch.functional.einsum('bij,ki->bkj', getitem_11, relation_forward_sample_part1367_w);  getitem_11 = relation_forward_sample_part1367_w = None
        zeros_like_5 = torch.zeros_like(einsum_22)
        repeat_5 = zeros_like_5.repeat(1, 1, 5);  zeros_like_5 = None
        sub_20 = einsum_22 - -0.2695776118175756
        neg_10 = -sub_20;  sub_20 = None
        truediv_20 = neg_10 / 0.1721194061968866;  neg_10 = None
        add_17 = truediv_20 + 1;  truediv_20 = None
        _tensor_constant114 = self._tensor_constant114
        maximum_20 = torch.maximum(add_17, _tensor_constant114);  add_17 = _tensor_constant114 = None
        _tensor_constant115 = self._tensor_constant115
        minimum_15 = torch.minimum(maximum_20, _tensor_constant115);  maximum_20 = _tensor_constant115 = None
        _tensor_constant116 = self._tensor_constant116
        slicing_15 = nnodely_layers_fuzzify_slicing(repeat_5, _tensor_constant116, minimum_15);  _tensor_constant116 = minimum_15 = slicing_15 = None
        sub_21 = einsum_22 - -0.2695776118175756
        truediv_21 = sub_21 / 0.1721194061968866;  sub_21 = None
        _tensor_constant117 = self._tensor_constant117
        maximum_21 = torch.maximum(truediv_21, _tensor_constant117);  truediv_21 = _tensor_constant117 = None
        sub_22 = einsum_22 - -0.09745820562068902
        neg_11 = -sub_22;  sub_22 = None
        truediv_22 = neg_11 / 0.1721194061968866;  neg_11 = None
        add_18 = truediv_22 + 1;  truediv_22 = None
        _tensor_constant118 = self._tensor_constant118
        maximum_22 = torch.maximum(add_18, _tensor_constant118);  add_18 = _tensor_constant118 = None
        minimum_16 = torch.minimum(maximum_21, maximum_22);  maximum_21 = maximum_22 = None
        _tensor_constant119 = self._tensor_constant119
        slicing_16 = nnodely_layers_fuzzify_slicing(repeat_5, _tensor_constant119, minimum_16);  _tensor_constant119 = minimum_16 = slicing_16 = None
        sub_23 = einsum_22 - -0.09745820562068902
        truediv_23 = sub_23 / 0.1721194061968866;  sub_23 = None
        _tensor_constant120 = self._tensor_constant120
        maximum_23 = torch.maximum(truediv_23, _tensor_constant120);  truediv_23 = _tensor_constant120 = None
        sub_24 = einsum_22 - 0.07466120057619757
        neg_12 = -sub_24;  sub_24 = None
        truediv_24 = neg_12 / 0.1721194061968866;  neg_12 = None
        add_19 = truediv_24 + 1;  truediv_24 = None
        _tensor_constant121 = self._tensor_constant121
        maximum_24 = torch.maximum(add_19, _tensor_constant121);  add_19 = _tensor_constant121 = None
        minimum_17 = torch.minimum(maximum_23, maximum_24);  maximum_23 = maximum_24 = None
        _tensor_constant122 = self._tensor_constant122
        slicing_17 = nnodely_layers_fuzzify_slicing(repeat_5, _tensor_constant122, minimum_17);  _tensor_constant122 = minimum_17 = slicing_17 = None
        sub_25 = einsum_22 - 0.07466120057619757
        truediv_25 = sub_25 / 0.1721194061968866;  sub_25 = None
        _tensor_constant123 = self._tensor_constant123
        maximum_25 = torch.maximum(truediv_25, _tensor_constant123);  truediv_25 = _tensor_constant123 = None
        sub_26 = einsum_22 - 0.24678060677308417
        neg_13 = -sub_26;  sub_26 = None
        truediv_26 = neg_13 / 0.17211940619688654;  neg_13 = None
        add_20 = truediv_26 + 1;  truediv_26 = None
        _tensor_constant124 = self._tensor_constant124
        maximum_26 = torch.maximum(add_20, _tensor_constant124);  add_20 = _tensor_constant124 = None
        minimum_18 = torch.minimum(maximum_25, maximum_26);  maximum_25 = maximum_26 = None
        _tensor_constant125 = self._tensor_constant125
        slicing_18 = nnodely_layers_fuzzify_slicing(repeat_5, _tensor_constant125, minimum_18);  _tensor_constant125 = minimum_18 = slicing_18 = None
        sub_27 = einsum_22 - 0.24678060677308417;  einsum_22 = None
        truediv_27 = sub_27 / 0.17211940619688654;  sub_27 = None
        _tensor_constant126 = self._tensor_constant126
        maximum_27 = torch.maximum(truediv_27, _tensor_constant126);  truediv_27 = _tensor_constant126 = None
        _tensor_constant127 = self._tensor_constant127
        minimum_19 = torch.minimum(maximum_27, _tensor_constant127);  maximum_27 = _tensor_constant127 = None
        _tensor_constant128 = self._tensor_constant128
        slicing_19 = nnodely_layers_fuzzify_slicing(repeat_5, _tensor_constant128, minimum_19);  _tensor_constant128 = minimum_19 = slicing_19 = None
        relation_forward_select2321_w = self.all_constants.Select2321
        einsum_23 = torch.functional.einsum('ijk,k->ij', repeat_5, relation_forward_select2321_w);  relation_forward_select2321_w = None
        unsqueeze_9 = einsum_23.unsqueeze(2);  einsum_23 = None
        getitem_12 = vel
        relation_forward_sample_part1370_w = self.all_constants.SamplePart1370
        einsum_24 = torch.functional.einsum('bij,ki->bkj', getitem_12, relation_forward_sample_part1370_w);  getitem_12 = relation_forward_sample_part1370_w = None
        zeros_like_6 = torch.zeros_like(einsum_24)
        repeat_6 = zeros_like_6.repeat(1, 1, 3);  zeros_like_6 = None
        sub_28 = einsum_24 - -0.3245462443934415
        neg_14 = -sub_28;  sub_28 = None
        truediv_28 = neg_14 / 1.9644116408757406;  neg_14 = None
        add_21 = truediv_28 + 1;  truediv_28 = None
        _tensor_constant129 = self._tensor_constant129
        maximum_28 = torch.maximum(add_21, _tensor_constant129);  add_21 = _tensor_constant129 = None
        _tensor_constant130 = self._tensor_constant130
        minimum_20 = torch.minimum(maximum_28, _tensor_constant130);  maximum_28 = _tensor_constant130 = None
        _tensor_constant131 = self._tensor_constant131
        slicing_20 = nnodely_layers_fuzzify_slicing(repeat_6, _tensor_constant131, minimum_20);  _tensor_constant131 = minimum_20 = slicing_20 = None
        sub_29 = einsum_24 - -0.3245462443934415
        truediv_29 = sub_29 / 1.9644116408757406;  sub_29 = None
        _tensor_constant132 = self._tensor_constant132
        maximum_29 = torch.maximum(truediv_29, _tensor_constant132);  truediv_29 = _tensor_constant132 = None
        sub_30 = einsum_24 - 1.6398653964822991
        neg_15 = -sub_30;  sub_30 = None
        truediv_30 = neg_15 / 1.9644116408757404;  neg_15 = None
        add_22 = truediv_30 + 1;  truediv_30 = None
        _tensor_constant133 = self._tensor_constant133
        maximum_30 = torch.maximum(add_22, _tensor_constant133);  add_22 = _tensor_constant133 = None
        minimum_21 = torch.minimum(maximum_29, maximum_30);  maximum_29 = maximum_30 = None
        _tensor_constant134 = self._tensor_constant134
        slicing_21 = nnodely_layers_fuzzify_slicing(repeat_6, _tensor_constant134, minimum_21);  _tensor_constant134 = minimum_21 = slicing_21 = None
        sub_31 = einsum_24 - 1.6398653964822991;  einsum_24 = None
        truediv_31 = sub_31 / 1.9644116408757404;  sub_31 = None
        _tensor_constant135 = self._tensor_constant135
        maximum_31 = torch.maximum(truediv_31, _tensor_constant135);  truediv_31 = _tensor_constant135 = None
        _tensor_constant136 = self._tensor_constant136
        minimum_22 = torch.minimum(maximum_31, _tensor_constant136);  maximum_31 = _tensor_constant136 = None
        _tensor_constant137 = self._tensor_constant137
        slicing_22 = nnodely_layers_fuzzify_slicing(repeat_6, _tensor_constant137, minimum_22);  _tensor_constant137 = minimum_22 = slicing_22 = None
        relation_forward_select2320_w = self.all_constants.Select2320
        einsum_25 = torch.functional.einsum('ijk,k->ij', repeat_6, relation_forward_select2320_w);  relation_forward_select2320_w = None
        unsqueeze_10 = einsum_25.unsqueeze(2);  einsum_25 = None
        mul_7 = unsqueeze_10 * unsqueeze_9;  unsqueeze_10 = unsqueeze_9 = None
        mul_8 = mul_7 * unsqueeze_8;  mul_7 = unsqueeze_8 = None
        getitem_13 = acc
        relation_forward_sample_part1380_w = self.all_constants.SamplePart1380
        einsum_26 = torch.functional.einsum('bij,ki->bkj', getitem_13, relation_forward_sample_part1380_w);  getitem_13 = relation_forward_sample_part1380_w = None
        getitem_14 = vel
        relation_forward_sample_part1378_w = self.all_constants.SamplePart1378
        einsum_27 = torch.functional.einsum('bij,ki->bkj', getitem_14, relation_forward_sample_part1378_w);  getitem_14 = relation_forward_sample_part1378_w = None
        getitem_15 = delta
        relation_forward_sample_part1376_w = self.all_constants.SamplePart1376
        einsum_28 = torch.functional.einsum('bij,ki->bkj', getitem_15, relation_forward_sample_part1376_w);  getitem_15 = relation_forward_sample_part1376_w = None
        all_constants_steer_center_4 = self.all_constants.steer_center_4
        all_constants_vx_center_2 = self.all_constants.vx_center_2
        all_constants_ax_center_2 = self.all_constants.ax_center_2
        all_parameters_lat_k1_2_4_2 = self.all_parameters.lat_k1_2_4_2
        all_parameters_lat_k2_2_4_2 = self.all_parameters.lat_k2_2_4_2
        all_parameters_lat_k3_2_4_2 = self.all_parameters.lat_k3_2_4_2
        all_parameters_lat_k4_2_4_2 = self.all_parameters.lat_k4_2_4_2
        understeer_corr_local = nnodely_layers_parametricfunction_understeer_corr_local(einsum_28, einsum_27, all_constants_steer_center_4, all_constants_vx_center_2, einsum_26, all_constants_ax_center_2, all_parameters_lat_k1_2_4_2, all_parameters_lat_k2_2_4_2, all_parameters_lat_k3_2_4_2, all_parameters_lat_k4_2_4_2);  all_parameters_lat_k1_2_4_2 = all_parameters_lat_k2_2_4_2 = all_parameters_lat_k3_2_4_2 = all_parameters_lat_k4_2_4_2 = None
        mul_9 = understeer_corr_local * mul_8;  understeer_corr_local = mul_8 = None
        relation_forward_select2302_w = self.all_constants.Select2302
        einsum_29 = torch.functional.einsum('ijk,k->ij', repeat_4, relation_forward_select2302_w);  relation_forward_select2302_w = None
        unsqueeze_11 = einsum_29.unsqueeze(2);  einsum_29 = None
        relation_forward_select2300_w = self.all_constants.Select2300
        einsum_30 = torch.functional.einsum('ijk,k->ij', repeat_5, relation_forward_select2300_w);  relation_forward_select2300_w = None
        unsqueeze_12 = einsum_30.unsqueeze(2);  einsum_30 = None
        relation_forward_select2299_w = self.all_constants.Select2299
        einsum_31 = torch.functional.einsum('ijk,k->ij', repeat_6, relation_forward_select2299_w);  relation_forward_select2299_w = None
        unsqueeze_13 = einsum_31.unsqueeze(2);  einsum_31 = None
        mul_10 = unsqueeze_13 * unsqueeze_12;  unsqueeze_13 = unsqueeze_12 = None
        mul_11 = mul_10 * unsqueeze_11;  mul_10 = unsqueeze_11 = None
        all_constants_ax_center_1 = self.all_constants.ax_center_1
        all_parameters_lat_k1_2_4_1 = self.all_parameters.lat_k1_2_4_1
        all_parameters_lat_k2_2_4_1 = self.all_parameters.lat_k2_2_4_1
        all_parameters_lat_k3_2_4_1 = self.all_parameters.lat_k3_2_4_1
        all_parameters_lat_k4_2_4_1 = self.all_parameters.lat_k4_2_4_1
        understeer_corr_local_1 = nnodely_layers_parametricfunction_understeer_corr_local(einsum_28, einsum_27, all_constants_steer_center_4, all_constants_vx_center_2, einsum_26, all_constants_ax_center_1, all_parameters_lat_k1_2_4_1, all_parameters_lat_k2_2_4_1, all_parameters_lat_k3_2_4_1, all_parameters_lat_k4_2_4_1);  all_parameters_lat_k1_2_4_1 = all_parameters_lat_k2_2_4_1 = all_parameters_lat_k3_2_4_1 = all_parameters_lat_k4_2_4_1 = None
        mul_12 = understeer_corr_local_1 * mul_11;  understeer_corr_local_1 = mul_11 = None
        relation_forward_select2281_w = self.all_constants.Select2281
        einsum_32 = torch.functional.einsum('ijk,k->ij', repeat_4, relation_forward_select2281_w);  relation_forward_select2281_w = None
        unsqueeze_14 = einsum_32.unsqueeze(2);  einsum_32 = None
        relation_forward_select2279_w = self.all_constants.Select2279
        einsum_33 = torch.functional.einsum('ijk,k->ij', repeat_5, relation_forward_select2279_w);  relation_forward_select2279_w = None
        unsqueeze_15 = einsum_33.unsqueeze(2);  einsum_33 = None
        relation_forward_select2278_w = self.all_constants.Select2278
        einsum_34 = torch.functional.einsum('ijk,k->ij', repeat_6, relation_forward_select2278_w);  relation_forward_select2278_w = None
        unsqueeze_16 = einsum_34.unsqueeze(2);  einsum_34 = None
        mul_13 = unsqueeze_16 * unsqueeze_15;  unsqueeze_16 = unsqueeze_15 = None
        mul_14 = mul_13 * unsqueeze_14;  mul_13 = unsqueeze_14 = None
        all_constants_ax_center_0 = self.all_constants.ax_center_0
        all_parameters_lat_k1_2_4_0 = self.all_parameters.lat_k1_2_4_0
        all_parameters_lat_k2_2_4_0 = self.all_parameters.lat_k2_2_4_0
        all_parameters_lat_k3_2_4_0 = self.all_parameters.lat_k3_2_4_0
        all_parameters_lat_k4_2_4_0 = self.all_parameters.lat_k4_2_4_0
        understeer_corr_local_2 = nnodely_layers_parametricfunction_understeer_corr_local(einsum_28, einsum_27, all_constants_steer_center_4, all_constants_vx_center_2, einsum_26, all_constants_ax_center_0, all_parameters_lat_k1_2_4_0, all_parameters_lat_k2_2_4_0, all_parameters_lat_k3_2_4_0, all_parameters_lat_k4_2_4_0);  all_parameters_lat_k1_2_4_0 = all_parameters_lat_k2_2_4_0 = all_parameters_lat_k3_2_4_0 = all_parameters_lat_k4_2_4_0 = None
        mul_15 = understeer_corr_local_2 * mul_14;  understeer_corr_local_2 = mul_14 = None
        relation_forward_select2260_w = self.all_constants.Select2260
        einsum_35 = torch.functional.einsum('ijk,k->ij', repeat_4, relation_forward_select2260_w);  relation_forward_select2260_w = None
        unsqueeze_17 = einsum_35.unsqueeze(2);  einsum_35 = None
        relation_forward_select2258_w = self.all_constants.Select2258
        einsum_36 = torch.functional.einsum('ijk,k->ij', repeat_5, relation_forward_select2258_w);  relation_forward_select2258_w = None
        unsqueeze_18 = einsum_36.unsqueeze(2);  einsum_36 = None
        relation_forward_select2257_w = self.all_constants.Select2257
        einsum_37 = torch.functional.einsum('ijk,k->ij', repeat_6, relation_forward_select2257_w);  relation_forward_select2257_w = None
        unsqueeze_19 = einsum_37.unsqueeze(2);  einsum_37 = None
        mul_16 = unsqueeze_19 * unsqueeze_18;  unsqueeze_19 = unsqueeze_18 = None
        mul_17 = mul_16 * unsqueeze_17;  mul_16 = unsqueeze_17 = None
        all_constants_steer_center_3 = self.all_constants.steer_center_3
        all_parameters_lat_k1_2_3_2 = self.all_parameters.lat_k1_2_3_2
        all_parameters_lat_k2_2_3_2 = self.all_parameters.lat_k2_2_3_2
        all_parameters_lat_k3_2_3_2 = self.all_parameters.lat_k3_2_3_2
        all_parameters_lat_k4_2_3_2 = self.all_parameters.lat_k4_2_3_2
        understeer_corr_local_3 = nnodely_layers_parametricfunction_understeer_corr_local(einsum_28, einsum_27, all_constants_steer_center_3, all_constants_vx_center_2, einsum_26, all_constants_ax_center_2, all_parameters_lat_k1_2_3_2, all_parameters_lat_k2_2_3_2, all_parameters_lat_k3_2_3_2, all_parameters_lat_k4_2_3_2);  all_parameters_lat_k1_2_3_2 = all_parameters_lat_k2_2_3_2 = all_parameters_lat_k3_2_3_2 = all_parameters_lat_k4_2_3_2 = None
        mul_18 = understeer_corr_local_3 * mul_17;  understeer_corr_local_3 = mul_17 = None
        relation_forward_select2239_w = self.all_constants.Select2239
        einsum_38 = torch.functional.einsum('ijk,k->ij', repeat_4, relation_forward_select2239_w);  relation_forward_select2239_w = None
        unsqueeze_20 = einsum_38.unsqueeze(2);  einsum_38 = None
        relation_forward_select2237_w = self.all_constants.Select2237
        einsum_39 = torch.functional.einsum('ijk,k->ij', repeat_5, relation_forward_select2237_w);  relation_forward_select2237_w = None
        unsqueeze_21 = einsum_39.unsqueeze(2);  einsum_39 = None
        relation_forward_select2236_w = self.all_constants.Select2236
        einsum_40 = torch.functional.einsum('ijk,k->ij', repeat_6, relation_forward_select2236_w);  relation_forward_select2236_w = None
        unsqueeze_22 = einsum_40.unsqueeze(2);  einsum_40 = None
        mul_19 = unsqueeze_22 * unsqueeze_21;  unsqueeze_22 = unsqueeze_21 = None
        mul_20 = mul_19 * unsqueeze_20;  mul_19 = unsqueeze_20 = None
        all_parameters_lat_k1_2_3_1 = self.all_parameters.lat_k1_2_3_1
        all_parameters_lat_k2_2_3_1 = self.all_parameters.lat_k2_2_3_1
        all_parameters_lat_k3_2_3_1 = self.all_parameters.lat_k3_2_3_1
        all_parameters_lat_k4_2_3_1 = self.all_parameters.lat_k4_2_3_1
        understeer_corr_local_4 = nnodely_layers_parametricfunction_understeer_corr_local(einsum_28, einsum_27, all_constants_steer_center_3, all_constants_vx_center_2, einsum_26, all_constants_ax_center_1, all_parameters_lat_k1_2_3_1, all_parameters_lat_k2_2_3_1, all_parameters_lat_k3_2_3_1, all_parameters_lat_k4_2_3_1);  all_parameters_lat_k1_2_3_1 = all_parameters_lat_k2_2_3_1 = all_parameters_lat_k3_2_3_1 = all_parameters_lat_k4_2_3_1 = None
        mul_21 = understeer_corr_local_4 * mul_20;  understeer_corr_local_4 = mul_20 = None
        relation_forward_select2218_w = self.all_constants.Select2218
        einsum_41 = torch.functional.einsum('ijk,k->ij', repeat_4, relation_forward_select2218_w);  relation_forward_select2218_w = None
        unsqueeze_23 = einsum_41.unsqueeze(2);  einsum_41 = None
        relation_forward_select2216_w = self.all_constants.Select2216
        einsum_42 = torch.functional.einsum('ijk,k->ij', repeat_5, relation_forward_select2216_w);  relation_forward_select2216_w = None
        unsqueeze_24 = einsum_42.unsqueeze(2);  einsum_42 = None
        relation_forward_select2215_w = self.all_constants.Select2215
        einsum_43 = torch.functional.einsum('ijk,k->ij', repeat_6, relation_forward_select2215_w);  relation_forward_select2215_w = None
        unsqueeze_25 = einsum_43.unsqueeze(2);  einsum_43 = None
        mul_22 = unsqueeze_25 * unsqueeze_24;  unsqueeze_25 = unsqueeze_24 = None
        mul_23 = mul_22 * unsqueeze_23;  mul_22 = unsqueeze_23 = None
        all_parameters_lat_k1_2_3_0 = self.all_parameters.lat_k1_2_3_0
        all_parameters_lat_k2_2_3_0 = self.all_parameters.lat_k2_2_3_0
        all_parameters_lat_k3_2_3_0 = self.all_parameters.lat_k3_2_3_0
        all_parameters_lat_k4_2_3_0 = self.all_parameters.lat_k4_2_3_0
        understeer_corr_local_5 = nnodely_layers_parametricfunction_understeer_corr_local(einsum_28, einsum_27, all_constants_steer_center_3, all_constants_vx_center_2, einsum_26, all_constants_ax_center_0, all_parameters_lat_k1_2_3_0, all_parameters_lat_k2_2_3_0, all_parameters_lat_k3_2_3_0, all_parameters_lat_k4_2_3_0);  all_parameters_lat_k1_2_3_0 = all_parameters_lat_k2_2_3_0 = all_parameters_lat_k3_2_3_0 = all_parameters_lat_k4_2_3_0 = None
        mul_24 = understeer_corr_local_5 * mul_23;  understeer_corr_local_5 = mul_23 = None
        relation_forward_select2197_w = self.all_constants.Select2197
        einsum_44 = torch.functional.einsum('ijk,k->ij', repeat_4, relation_forward_select2197_w);  relation_forward_select2197_w = None
        unsqueeze_26 = einsum_44.unsqueeze(2);  einsum_44 = None
        relation_forward_select2195_w = self.all_constants.Select2195
        einsum_45 = torch.functional.einsum('ijk,k->ij', repeat_5, relation_forward_select2195_w);  relation_forward_select2195_w = None
        unsqueeze_27 = einsum_45.unsqueeze(2);  einsum_45 = None
        relation_forward_select2194_w = self.all_constants.Select2194
        einsum_46 = torch.functional.einsum('ijk,k->ij', repeat_6, relation_forward_select2194_w);  relation_forward_select2194_w = None
        unsqueeze_28 = einsum_46.unsqueeze(2);  einsum_46 = None
        mul_25 = unsqueeze_28 * unsqueeze_27;  unsqueeze_28 = unsqueeze_27 = None
        mul_26 = mul_25 * unsqueeze_26;  mul_25 = unsqueeze_26 = None
        all_constants_steer_center_2 = self.all_constants.steer_center_2
        all_parameters_lat_k1_2_2_2 = self.all_parameters.lat_k1_2_2_2
        all_parameters_lat_k2_2_2_2 = self.all_parameters.lat_k2_2_2_2
        all_parameters_lat_k3_2_2_2 = self.all_parameters.lat_k3_2_2_2
        all_parameters_lat_k4_2_2_2 = self.all_parameters.lat_k4_2_2_2
        understeer_corr_local_6 = nnodely_layers_parametricfunction_understeer_corr_local(einsum_28, einsum_27, all_constants_steer_center_2, all_constants_vx_center_2, einsum_26, all_constants_ax_center_2, all_parameters_lat_k1_2_2_2, all_parameters_lat_k2_2_2_2, all_parameters_lat_k3_2_2_2, all_parameters_lat_k4_2_2_2);  all_parameters_lat_k1_2_2_2 = all_parameters_lat_k2_2_2_2 = all_parameters_lat_k3_2_2_2 = all_parameters_lat_k4_2_2_2 = None
        mul_27 = understeer_corr_local_6 * mul_26;  understeer_corr_local_6 = mul_26 = None
        relation_forward_select2176_w = self.all_constants.Select2176
        einsum_47 = torch.functional.einsum('ijk,k->ij', repeat_4, relation_forward_select2176_w);  relation_forward_select2176_w = None
        unsqueeze_29 = einsum_47.unsqueeze(2);  einsum_47 = None
        relation_forward_select2174_w = self.all_constants.Select2174
        einsum_48 = torch.functional.einsum('ijk,k->ij', repeat_5, relation_forward_select2174_w);  relation_forward_select2174_w = None
        unsqueeze_30 = einsum_48.unsqueeze(2);  einsum_48 = None
        relation_forward_select2173_w = self.all_constants.Select2173
        einsum_49 = torch.functional.einsum('ijk,k->ij', repeat_6, relation_forward_select2173_w);  relation_forward_select2173_w = None
        unsqueeze_31 = einsum_49.unsqueeze(2);  einsum_49 = None
        mul_28 = unsqueeze_31 * unsqueeze_30;  unsqueeze_31 = unsqueeze_30 = None
        mul_29 = mul_28 * unsqueeze_29;  mul_28 = unsqueeze_29 = None
        all_parameters_lat_k1_2_2_1 = self.all_parameters.lat_k1_2_2_1
        all_parameters_lat_k2_2_2_1 = self.all_parameters.lat_k2_2_2_1
        all_parameters_lat_k3_2_2_1 = self.all_parameters.lat_k3_2_2_1
        all_parameters_lat_k4_2_2_1 = self.all_parameters.lat_k4_2_2_1
        understeer_corr_local_7 = nnodely_layers_parametricfunction_understeer_corr_local(einsum_28, einsum_27, all_constants_steer_center_2, all_constants_vx_center_2, einsum_26, all_constants_ax_center_1, all_parameters_lat_k1_2_2_1, all_parameters_lat_k2_2_2_1, all_parameters_lat_k3_2_2_1, all_parameters_lat_k4_2_2_1);  all_parameters_lat_k1_2_2_1 = all_parameters_lat_k2_2_2_1 = all_parameters_lat_k3_2_2_1 = all_parameters_lat_k4_2_2_1 = None
        mul_30 = understeer_corr_local_7 * mul_29;  understeer_corr_local_7 = mul_29 = None
        relation_forward_select2155_w = self.all_constants.Select2155
        einsum_50 = torch.functional.einsum('ijk,k->ij', repeat_4, relation_forward_select2155_w);  relation_forward_select2155_w = None
        unsqueeze_32 = einsum_50.unsqueeze(2);  einsum_50 = None
        relation_forward_select2153_w = self.all_constants.Select2153
        einsum_51 = torch.functional.einsum('ijk,k->ij', repeat_5, relation_forward_select2153_w);  relation_forward_select2153_w = None
        unsqueeze_33 = einsum_51.unsqueeze(2);  einsum_51 = None
        relation_forward_select2152_w = self.all_constants.Select2152
        einsum_52 = torch.functional.einsum('ijk,k->ij', repeat_6, relation_forward_select2152_w);  relation_forward_select2152_w = None
        unsqueeze_34 = einsum_52.unsqueeze(2);  einsum_52 = None
        mul_31 = unsqueeze_34 * unsqueeze_33;  unsqueeze_34 = unsqueeze_33 = None
        mul_32 = mul_31 * unsqueeze_32;  mul_31 = unsqueeze_32 = None
        all_parameters_lat_k1_2_2_0 = self.all_parameters.lat_k1_2_2_0
        all_parameters_lat_k2_2_2_0 = self.all_parameters.lat_k2_2_2_0
        all_parameters_lat_k3_2_2_0 = self.all_parameters.lat_k3_2_2_0
        all_parameters_lat_k4_2_2_0 = self.all_parameters.lat_k4_2_2_0
        understeer_corr_local_8 = nnodely_layers_parametricfunction_understeer_corr_local(einsum_28, einsum_27, all_constants_steer_center_2, all_constants_vx_center_2, einsum_26, all_constants_ax_center_0, all_parameters_lat_k1_2_2_0, all_parameters_lat_k2_2_2_0, all_parameters_lat_k3_2_2_0, all_parameters_lat_k4_2_2_0);  all_parameters_lat_k1_2_2_0 = all_parameters_lat_k2_2_2_0 = all_parameters_lat_k3_2_2_0 = all_parameters_lat_k4_2_2_0 = None
        mul_33 = understeer_corr_local_8 * mul_32;  understeer_corr_local_8 = mul_32 = None
        relation_forward_select2134_w = self.all_constants.Select2134
        einsum_53 = torch.functional.einsum('ijk,k->ij', repeat_4, relation_forward_select2134_w);  relation_forward_select2134_w = None
        unsqueeze_35 = einsum_53.unsqueeze(2);  einsum_53 = None
        relation_forward_select2132_w = self.all_constants.Select2132
        einsum_54 = torch.functional.einsum('ijk,k->ij', repeat_5, relation_forward_select2132_w);  relation_forward_select2132_w = None
        unsqueeze_36 = einsum_54.unsqueeze(2);  einsum_54 = None
        relation_forward_select2131_w = self.all_constants.Select2131
        einsum_55 = torch.functional.einsum('ijk,k->ij', repeat_6, relation_forward_select2131_w);  relation_forward_select2131_w = None
        unsqueeze_37 = einsum_55.unsqueeze(2);  einsum_55 = None
        mul_34 = unsqueeze_37 * unsqueeze_36;  unsqueeze_37 = unsqueeze_36 = None
        mul_35 = mul_34 * unsqueeze_35;  mul_34 = unsqueeze_35 = None
        all_constants_steer_center_1 = self.all_constants.steer_center_1
        all_parameters_lat_k1_2_1_2 = self.all_parameters.lat_k1_2_1_2
        all_parameters_lat_k2_2_1_2 = self.all_parameters.lat_k2_2_1_2
        all_parameters_lat_k3_2_1_2 = self.all_parameters.lat_k3_2_1_2
        all_parameters_lat_k4_2_1_2 = self.all_parameters.lat_k4_2_1_2
        understeer_corr_local_9 = nnodely_layers_parametricfunction_understeer_corr_local(einsum_28, einsum_27, all_constants_steer_center_1, all_constants_vx_center_2, einsum_26, all_constants_ax_center_2, all_parameters_lat_k1_2_1_2, all_parameters_lat_k2_2_1_2, all_parameters_lat_k3_2_1_2, all_parameters_lat_k4_2_1_2);  all_parameters_lat_k1_2_1_2 = all_parameters_lat_k2_2_1_2 = all_parameters_lat_k3_2_1_2 = all_parameters_lat_k4_2_1_2 = None
        mul_36 = understeer_corr_local_9 * mul_35;  understeer_corr_local_9 = mul_35 = None
        relation_forward_select2113_w = self.all_constants.Select2113
        einsum_56 = torch.functional.einsum('ijk,k->ij', repeat_4, relation_forward_select2113_w);  relation_forward_select2113_w = None
        unsqueeze_38 = einsum_56.unsqueeze(2);  einsum_56 = None
        relation_forward_select2111_w = self.all_constants.Select2111
        einsum_57 = torch.functional.einsum('ijk,k->ij', repeat_5, relation_forward_select2111_w);  relation_forward_select2111_w = None
        unsqueeze_39 = einsum_57.unsqueeze(2);  einsum_57 = None
        relation_forward_select2110_w = self.all_constants.Select2110
        einsum_58 = torch.functional.einsum('ijk,k->ij', repeat_6, relation_forward_select2110_w);  relation_forward_select2110_w = None
        unsqueeze_40 = einsum_58.unsqueeze(2);  einsum_58 = None
        mul_37 = unsqueeze_40 * unsqueeze_39;  unsqueeze_40 = unsqueeze_39 = None
        mul_38 = mul_37 * unsqueeze_38;  mul_37 = unsqueeze_38 = None
        all_parameters_lat_k1_2_1_1 = self.all_parameters.lat_k1_2_1_1
        all_parameters_lat_k2_2_1_1 = self.all_parameters.lat_k2_2_1_1
        all_parameters_lat_k3_2_1_1 = self.all_parameters.lat_k3_2_1_1
        all_parameters_lat_k4_2_1_1 = self.all_parameters.lat_k4_2_1_1
        understeer_corr_local_10 = nnodely_layers_parametricfunction_understeer_corr_local(einsum_28, einsum_27, all_constants_steer_center_1, all_constants_vx_center_2, einsum_26, all_constants_ax_center_1, all_parameters_lat_k1_2_1_1, all_parameters_lat_k2_2_1_1, all_parameters_lat_k3_2_1_1, all_parameters_lat_k4_2_1_1);  all_parameters_lat_k1_2_1_1 = all_parameters_lat_k2_2_1_1 = all_parameters_lat_k3_2_1_1 = all_parameters_lat_k4_2_1_1 = None
        mul_39 = understeer_corr_local_10 * mul_38;  understeer_corr_local_10 = mul_38 = None
        relation_forward_select2092_w = self.all_constants.Select2092
        einsum_59 = torch.functional.einsum('ijk,k->ij', repeat_4, relation_forward_select2092_w);  relation_forward_select2092_w = None
        unsqueeze_41 = einsum_59.unsqueeze(2);  einsum_59 = None
        relation_forward_select2090_w = self.all_constants.Select2090
        einsum_60 = torch.functional.einsum('ijk,k->ij', repeat_5, relation_forward_select2090_w);  relation_forward_select2090_w = None
        unsqueeze_42 = einsum_60.unsqueeze(2);  einsum_60 = None
        relation_forward_select2089_w = self.all_constants.Select2089
        einsum_61 = torch.functional.einsum('ijk,k->ij', repeat_6, relation_forward_select2089_w);  relation_forward_select2089_w = None
        unsqueeze_43 = einsum_61.unsqueeze(2);  einsum_61 = None
        mul_40 = unsqueeze_43 * unsqueeze_42;  unsqueeze_43 = unsqueeze_42 = None
        mul_41 = mul_40 * unsqueeze_41;  mul_40 = unsqueeze_41 = None
        all_parameters_lat_k1_2_1_0 = self.all_parameters.lat_k1_2_1_0
        all_parameters_lat_k2_2_1_0 = self.all_parameters.lat_k2_2_1_0
        all_parameters_lat_k3_2_1_0 = self.all_parameters.lat_k3_2_1_0
        all_parameters_lat_k4_2_1_0 = self.all_parameters.lat_k4_2_1_0
        understeer_corr_local_11 = nnodely_layers_parametricfunction_understeer_corr_local(einsum_28, einsum_27, all_constants_steer_center_1, all_constants_vx_center_2, einsum_26, all_constants_ax_center_0, all_parameters_lat_k1_2_1_0, all_parameters_lat_k2_2_1_0, all_parameters_lat_k3_2_1_0, all_parameters_lat_k4_2_1_0);  all_parameters_lat_k1_2_1_0 = all_parameters_lat_k2_2_1_0 = all_parameters_lat_k3_2_1_0 = all_parameters_lat_k4_2_1_0 = None
        mul_42 = understeer_corr_local_11 * mul_41;  understeer_corr_local_11 = mul_41 = None
        relation_forward_select2071_w = self.all_constants.Select2071
        einsum_62 = torch.functional.einsum('ijk,k->ij', repeat_4, relation_forward_select2071_w);  relation_forward_select2071_w = None
        unsqueeze_44 = einsum_62.unsqueeze(2);  einsum_62 = None
        relation_forward_select2069_w = self.all_constants.Select2069
        einsum_63 = torch.functional.einsum('ijk,k->ij', repeat_5, relation_forward_select2069_w);  relation_forward_select2069_w = None
        unsqueeze_45 = einsum_63.unsqueeze(2);  einsum_63 = None
        relation_forward_select2068_w = self.all_constants.Select2068
        einsum_64 = torch.functional.einsum('ijk,k->ij', repeat_6, relation_forward_select2068_w);  relation_forward_select2068_w = None
        unsqueeze_46 = einsum_64.unsqueeze(2);  einsum_64 = None
        mul_43 = unsqueeze_46 * unsqueeze_45;  unsqueeze_46 = unsqueeze_45 = None
        mul_44 = mul_43 * unsqueeze_44;  mul_43 = unsqueeze_44 = None
        all_constants_steer_center_0 = self.all_constants.steer_center_0
        all_parameters_lat_k1_2_0_2 = self.all_parameters.lat_k1_2_0_2
        all_parameters_lat_k2_2_0_2 = self.all_parameters.lat_k2_2_0_2
        all_parameters_lat_k3_2_0_2 = self.all_parameters.lat_k3_2_0_2
        all_parameters_lat_k4_2_0_2 = self.all_parameters.lat_k4_2_0_2
        understeer_corr_local_12 = nnodely_layers_parametricfunction_understeer_corr_local(einsum_28, einsum_27, all_constants_steer_center_0, all_constants_vx_center_2, einsum_26, all_constants_ax_center_2, all_parameters_lat_k1_2_0_2, all_parameters_lat_k2_2_0_2, all_parameters_lat_k3_2_0_2, all_parameters_lat_k4_2_0_2);  all_parameters_lat_k1_2_0_2 = all_parameters_lat_k2_2_0_2 = all_parameters_lat_k3_2_0_2 = all_parameters_lat_k4_2_0_2 = None
        mul_45 = understeer_corr_local_12 * mul_44;  understeer_corr_local_12 = mul_44 = None
        relation_forward_select2050_w = self.all_constants.Select2050
        einsum_65 = torch.functional.einsum('ijk,k->ij', repeat_4, relation_forward_select2050_w);  relation_forward_select2050_w = None
        unsqueeze_47 = einsum_65.unsqueeze(2);  einsum_65 = None
        relation_forward_select2048_w = self.all_constants.Select2048
        einsum_66 = torch.functional.einsum('ijk,k->ij', repeat_5, relation_forward_select2048_w);  relation_forward_select2048_w = None
        unsqueeze_48 = einsum_66.unsqueeze(2);  einsum_66 = None
        relation_forward_select2047_w = self.all_constants.Select2047
        einsum_67 = torch.functional.einsum('ijk,k->ij', repeat_6, relation_forward_select2047_w);  relation_forward_select2047_w = None
        unsqueeze_49 = einsum_67.unsqueeze(2);  einsum_67 = None
        mul_46 = unsqueeze_49 * unsqueeze_48;  unsqueeze_49 = unsqueeze_48 = None
        mul_47 = mul_46 * unsqueeze_47;  mul_46 = unsqueeze_47 = None
        all_parameters_lat_k1_2_0_1 = self.all_parameters.lat_k1_2_0_1
        all_parameters_lat_k2_2_0_1 = self.all_parameters.lat_k2_2_0_1
        all_parameters_lat_k3_2_0_1 = self.all_parameters.lat_k3_2_0_1
        all_parameters_lat_k4_2_0_1 = self.all_parameters.lat_k4_2_0_1
        understeer_corr_local_13 = nnodely_layers_parametricfunction_understeer_corr_local(einsum_28, einsum_27, all_constants_steer_center_0, all_constants_vx_center_2, einsum_26, all_constants_ax_center_1, all_parameters_lat_k1_2_0_1, all_parameters_lat_k2_2_0_1, all_parameters_lat_k3_2_0_1, all_parameters_lat_k4_2_0_1);  all_parameters_lat_k1_2_0_1 = all_parameters_lat_k2_2_0_1 = all_parameters_lat_k3_2_0_1 = all_parameters_lat_k4_2_0_1 = None
        mul_48 = understeer_corr_local_13 * mul_47;  understeer_corr_local_13 = mul_47 = None
        relation_forward_select2029_w = self.all_constants.Select2029
        einsum_68 = torch.functional.einsum('ijk,k->ij', repeat_4, relation_forward_select2029_w);  relation_forward_select2029_w = None
        unsqueeze_50 = einsum_68.unsqueeze(2);  einsum_68 = None
        relation_forward_select2027_w = self.all_constants.Select2027
        einsum_69 = torch.functional.einsum('ijk,k->ij', repeat_5, relation_forward_select2027_w);  relation_forward_select2027_w = None
        unsqueeze_51 = einsum_69.unsqueeze(2);  einsum_69 = None
        relation_forward_select2026_w = self.all_constants.Select2026
        einsum_70 = torch.functional.einsum('ijk,k->ij', repeat_6, relation_forward_select2026_w);  relation_forward_select2026_w = None
        unsqueeze_52 = einsum_70.unsqueeze(2);  einsum_70 = None
        mul_49 = unsqueeze_52 * unsqueeze_51;  unsqueeze_52 = unsqueeze_51 = None
        mul_50 = mul_49 * unsqueeze_50;  mul_49 = unsqueeze_50 = None
        all_parameters_lat_k1_2_0_0 = self.all_parameters.lat_k1_2_0_0
        all_parameters_lat_k2_2_0_0 = self.all_parameters.lat_k2_2_0_0
        all_parameters_lat_k3_2_0_0 = self.all_parameters.lat_k3_2_0_0
        all_parameters_lat_k4_2_0_0 = self.all_parameters.lat_k4_2_0_0
        understeer_corr_local_14 = nnodely_layers_parametricfunction_understeer_corr_local(einsum_28, einsum_27, all_constants_steer_center_0, all_constants_vx_center_2, einsum_26, all_constants_ax_center_0, all_parameters_lat_k1_2_0_0, all_parameters_lat_k2_2_0_0, all_parameters_lat_k3_2_0_0, all_parameters_lat_k4_2_0_0);  all_constants_vx_center_2 = all_parameters_lat_k1_2_0_0 = all_parameters_lat_k2_2_0_0 = all_parameters_lat_k3_2_0_0 = all_parameters_lat_k4_2_0_0 = None
        mul_51 = understeer_corr_local_14 * mul_50;  understeer_corr_local_14 = mul_50 = None
        relation_forward_select2008_w = self.all_constants.Select2008
        einsum_71 = torch.functional.einsum('ijk,k->ij', repeat_4, relation_forward_select2008_w);  relation_forward_select2008_w = None
        unsqueeze_53 = einsum_71.unsqueeze(2);  einsum_71 = None
        relation_forward_select2006_w = self.all_constants.Select2006
        einsum_72 = torch.functional.einsum('ijk,k->ij', repeat_5, relation_forward_select2006_w);  relation_forward_select2006_w = None
        unsqueeze_54 = einsum_72.unsqueeze(2);  einsum_72 = None
        relation_forward_select2005_w = self.all_constants.Select2005
        einsum_73 = torch.functional.einsum('ijk,k->ij', repeat_6, relation_forward_select2005_w);  relation_forward_select2005_w = None
        unsqueeze_55 = einsum_73.unsqueeze(2);  einsum_73 = None
        mul_52 = unsqueeze_55 * unsqueeze_54;  unsqueeze_55 = unsqueeze_54 = None
        mul_53 = mul_52 * unsqueeze_53;  mul_52 = unsqueeze_53 = None
        all_constants_vx_center_1 = self.all_constants.vx_center_1
        all_parameters_lat_k1_1_4_2 = self.all_parameters.lat_k1_1_4_2
        all_parameters_lat_k2_1_4_2 = self.all_parameters.lat_k2_1_4_2
        all_parameters_lat_k3_1_4_2 = self.all_parameters.lat_k3_1_4_2
        all_parameters_lat_k4_1_4_2 = self.all_parameters.lat_k4_1_4_2
        understeer_corr_local_15 = nnodely_layers_parametricfunction_understeer_corr_local(einsum_28, einsum_27, all_constants_steer_center_4, all_constants_vx_center_1, einsum_26, all_constants_ax_center_2, all_parameters_lat_k1_1_4_2, all_parameters_lat_k2_1_4_2, all_parameters_lat_k3_1_4_2, all_parameters_lat_k4_1_4_2);  all_parameters_lat_k1_1_4_2 = all_parameters_lat_k2_1_4_2 = all_parameters_lat_k3_1_4_2 = all_parameters_lat_k4_1_4_2 = None
        mul_54 = understeer_corr_local_15 * mul_53;  understeer_corr_local_15 = mul_53 = None
        relation_forward_select1987_w = self.all_constants.Select1987
        einsum_74 = torch.functional.einsum('ijk,k->ij', repeat_4, relation_forward_select1987_w);  relation_forward_select1987_w = None
        unsqueeze_56 = einsum_74.unsqueeze(2);  einsum_74 = None
        relation_forward_select1985_w = self.all_constants.Select1985
        einsum_75 = torch.functional.einsum('ijk,k->ij', repeat_5, relation_forward_select1985_w);  relation_forward_select1985_w = None
        unsqueeze_57 = einsum_75.unsqueeze(2);  einsum_75 = None
        relation_forward_select1984_w = self.all_constants.Select1984
        einsum_76 = torch.functional.einsum('ijk,k->ij', repeat_6, relation_forward_select1984_w);  relation_forward_select1984_w = None
        unsqueeze_58 = einsum_76.unsqueeze(2);  einsum_76 = None
        mul_55 = unsqueeze_58 * unsqueeze_57;  unsqueeze_58 = unsqueeze_57 = None
        mul_56 = mul_55 * unsqueeze_56;  mul_55 = unsqueeze_56 = None
        all_parameters_lat_k1_1_4_1 = self.all_parameters.lat_k1_1_4_1
        all_parameters_lat_k2_1_4_1 = self.all_parameters.lat_k2_1_4_1
        all_parameters_lat_k3_1_4_1 = self.all_parameters.lat_k3_1_4_1
        all_parameters_lat_k4_1_4_1 = self.all_parameters.lat_k4_1_4_1
        understeer_corr_local_16 = nnodely_layers_parametricfunction_understeer_corr_local(einsum_28, einsum_27, all_constants_steer_center_4, all_constants_vx_center_1, einsum_26, all_constants_ax_center_1, all_parameters_lat_k1_1_4_1, all_parameters_lat_k2_1_4_1, all_parameters_lat_k3_1_4_1, all_parameters_lat_k4_1_4_1);  all_parameters_lat_k1_1_4_1 = all_parameters_lat_k2_1_4_1 = all_parameters_lat_k3_1_4_1 = all_parameters_lat_k4_1_4_1 = None
        mul_57 = understeer_corr_local_16 * mul_56;  understeer_corr_local_16 = mul_56 = None
        relation_forward_select1966_w = self.all_constants.Select1966
        einsum_77 = torch.functional.einsum('ijk,k->ij', repeat_4, relation_forward_select1966_w);  relation_forward_select1966_w = None
        unsqueeze_59 = einsum_77.unsqueeze(2);  einsum_77 = None
        relation_forward_select1964_w = self.all_constants.Select1964
        einsum_78 = torch.functional.einsum('ijk,k->ij', repeat_5, relation_forward_select1964_w);  relation_forward_select1964_w = None
        unsqueeze_60 = einsum_78.unsqueeze(2);  einsum_78 = None
        relation_forward_select1963_w = self.all_constants.Select1963
        einsum_79 = torch.functional.einsum('ijk,k->ij', repeat_6, relation_forward_select1963_w);  relation_forward_select1963_w = None
        unsqueeze_61 = einsum_79.unsqueeze(2);  einsum_79 = None
        mul_58 = unsqueeze_61 * unsqueeze_60;  unsqueeze_61 = unsqueeze_60 = None
        mul_59 = mul_58 * unsqueeze_59;  mul_58 = unsqueeze_59 = None
        all_parameters_lat_k1_1_4_0 = self.all_parameters.lat_k1_1_4_0
        all_parameters_lat_k2_1_4_0 = self.all_parameters.lat_k2_1_4_0
        all_parameters_lat_k3_1_4_0 = self.all_parameters.lat_k3_1_4_0
        all_parameters_lat_k4_1_4_0 = self.all_parameters.lat_k4_1_4_0
        understeer_corr_local_17 = nnodely_layers_parametricfunction_understeer_corr_local(einsum_28, einsum_27, all_constants_steer_center_4, all_constants_vx_center_1, einsum_26, all_constants_ax_center_0, all_parameters_lat_k1_1_4_0, all_parameters_lat_k2_1_4_0, all_parameters_lat_k3_1_4_0, all_parameters_lat_k4_1_4_0);  all_parameters_lat_k1_1_4_0 = all_parameters_lat_k2_1_4_0 = all_parameters_lat_k3_1_4_0 = all_parameters_lat_k4_1_4_0 = None
        mul_60 = understeer_corr_local_17 * mul_59;  understeer_corr_local_17 = mul_59 = None
        relation_forward_select1945_w = self.all_constants.Select1945
        einsum_80 = torch.functional.einsum('ijk,k->ij', repeat_4, relation_forward_select1945_w);  relation_forward_select1945_w = None
        unsqueeze_62 = einsum_80.unsqueeze(2);  einsum_80 = None
        relation_forward_select1943_w = self.all_constants.Select1943
        einsum_81 = torch.functional.einsum('ijk,k->ij', repeat_5, relation_forward_select1943_w);  relation_forward_select1943_w = None
        unsqueeze_63 = einsum_81.unsqueeze(2);  einsum_81 = None
        relation_forward_select1942_w = self.all_constants.Select1942
        einsum_82 = torch.functional.einsum('ijk,k->ij', repeat_6, relation_forward_select1942_w);  relation_forward_select1942_w = None
        unsqueeze_64 = einsum_82.unsqueeze(2);  einsum_82 = None
        mul_61 = unsqueeze_64 * unsqueeze_63;  unsqueeze_64 = unsqueeze_63 = None
        mul_62 = mul_61 * unsqueeze_62;  mul_61 = unsqueeze_62 = None
        all_parameters_lat_k1_1_3_2 = self.all_parameters.lat_k1_1_3_2
        all_parameters_lat_k2_1_3_2 = self.all_parameters.lat_k2_1_3_2
        all_parameters_lat_k3_1_3_2 = self.all_parameters.lat_k3_1_3_2
        all_parameters_lat_k4_1_3_2 = self.all_parameters.lat_k4_1_3_2
        understeer_corr_local_18 = nnodely_layers_parametricfunction_understeer_corr_local(einsum_28, einsum_27, all_constants_steer_center_3, all_constants_vx_center_1, einsum_26, all_constants_ax_center_2, all_parameters_lat_k1_1_3_2, all_parameters_lat_k2_1_3_2, all_parameters_lat_k3_1_3_2, all_parameters_lat_k4_1_3_2);  all_parameters_lat_k1_1_3_2 = all_parameters_lat_k2_1_3_2 = all_parameters_lat_k3_1_3_2 = all_parameters_lat_k4_1_3_2 = None
        mul_63 = understeer_corr_local_18 * mul_62;  understeer_corr_local_18 = mul_62 = None
        relation_forward_select1924_w = self.all_constants.Select1924
        einsum_83 = torch.functional.einsum('ijk,k->ij', repeat_4, relation_forward_select1924_w);  relation_forward_select1924_w = None
        unsqueeze_65 = einsum_83.unsqueeze(2);  einsum_83 = None
        relation_forward_select1922_w = self.all_constants.Select1922
        einsum_84 = torch.functional.einsum('ijk,k->ij', repeat_5, relation_forward_select1922_w);  relation_forward_select1922_w = None
        unsqueeze_66 = einsum_84.unsqueeze(2);  einsum_84 = None
        relation_forward_select1921_w = self.all_constants.Select1921
        einsum_85 = torch.functional.einsum('ijk,k->ij', repeat_6, relation_forward_select1921_w);  relation_forward_select1921_w = None
        unsqueeze_67 = einsum_85.unsqueeze(2);  einsum_85 = None
        mul_64 = unsqueeze_67 * unsqueeze_66;  unsqueeze_67 = unsqueeze_66 = None
        mul_65 = mul_64 * unsqueeze_65;  mul_64 = unsqueeze_65 = None
        all_parameters_lat_k1_1_3_1 = self.all_parameters.lat_k1_1_3_1
        all_parameters_lat_k2_1_3_1 = self.all_parameters.lat_k2_1_3_1
        all_parameters_lat_k3_1_3_1 = self.all_parameters.lat_k3_1_3_1
        all_parameters_lat_k4_1_3_1 = self.all_parameters.lat_k4_1_3_1
        understeer_corr_local_19 = nnodely_layers_parametricfunction_understeer_corr_local(einsum_28, einsum_27, all_constants_steer_center_3, all_constants_vx_center_1, einsum_26, all_constants_ax_center_1, all_parameters_lat_k1_1_3_1, all_parameters_lat_k2_1_3_1, all_parameters_lat_k3_1_3_1, all_parameters_lat_k4_1_3_1);  all_parameters_lat_k1_1_3_1 = all_parameters_lat_k2_1_3_1 = all_parameters_lat_k3_1_3_1 = all_parameters_lat_k4_1_3_1 = None
        mul_66 = understeer_corr_local_19 * mul_65;  understeer_corr_local_19 = mul_65 = None
        relation_forward_select1903_w = self.all_constants.Select1903
        einsum_86 = torch.functional.einsum('ijk,k->ij', repeat_4, relation_forward_select1903_w);  relation_forward_select1903_w = None
        unsqueeze_68 = einsum_86.unsqueeze(2);  einsum_86 = None
        relation_forward_select1901_w = self.all_constants.Select1901
        einsum_87 = torch.functional.einsum('ijk,k->ij', repeat_5, relation_forward_select1901_w);  relation_forward_select1901_w = None
        unsqueeze_69 = einsum_87.unsqueeze(2);  einsum_87 = None
        relation_forward_select1900_w = self.all_constants.Select1900
        einsum_88 = torch.functional.einsum('ijk,k->ij', repeat_6, relation_forward_select1900_w);  relation_forward_select1900_w = None
        unsqueeze_70 = einsum_88.unsqueeze(2);  einsum_88 = None
        mul_67 = unsqueeze_70 * unsqueeze_69;  unsqueeze_70 = unsqueeze_69 = None
        mul_68 = mul_67 * unsqueeze_68;  mul_67 = unsqueeze_68 = None
        all_parameters_lat_k1_1_3_0 = self.all_parameters.lat_k1_1_3_0
        all_parameters_lat_k2_1_3_0 = self.all_parameters.lat_k2_1_3_0
        all_parameters_lat_k3_1_3_0 = self.all_parameters.lat_k3_1_3_0
        all_parameters_lat_k4_1_3_0 = self.all_parameters.lat_k4_1_3_0
        understeer_corr_local_20 = nnodely_layers_parametricfunction_understeer_corr_local(einsum_28, einsum_27, all_constants_steer_center_3, all_constants_vx_center_1, einsum_26, all_constants_ax_center_0, all_parameters_lat_k1_1_3_0, all_parameters_lat_k2_1_3_0, all_parameters_lat_k3_1_3_0, all_parameters_lat_k4_1_3_0);  all_parameters_lat_k1_1_3_0 = all_parameters_lat_k2_1_3_0 = all_parameters_lat_k3_1_3_0 = all_parameters_lat_k4_1_3_0 = None
        mul_69 = understeer_corr_local_20 * mul_68;  understeer_corr_local_20 = mul_68 = None
        relation_forward_select1882_w = self.all_constants.Select1882
        einsum_89 = torch.functional.einsum('ijk,k->ij', repeat_4, relation_forward_select1882_w);  relation_forward_select1882_w = None
        unsqueeze_71 = einsum_89.unsqueeze(2);  einsum_89 = None
        relation_forward_select1880_w = self.all_constants.Select1880
        einsum_90 = torch.functional.einsum('ijk,k->ij', repeat_5, relation_forward_select1880_w);  relation_forward_select1880_w = None
        unsqueeze_72 = einsum_90.unsqueeze(2);  einsum_90 = None
        relation_forward_select1879_w = self.all_constants.Select1879
        einsum_91 = torch.functional.einsum('ijk,k->ij', repeat_6, relation_forward_select1879_w);  relation_forward_select1879_w = None
        unsqueeze_73 = einsum_91.unsqueeze(2);  einsum_91 = None
        mul_70 = unsqueeze_73 * unsqueeze_72;  unsqueeze_73 = unsqueeze_72 = None
        mul_71 = mul_70 * unsqueeze_71;  mul_70 = unsqueeze_71 = None
        all_parameters_lat_k1_1_2_2 = self.all_parameters.lat_k1_1_2_2
        all_parameters_lat_k2_1_2_2 = self.all_parameters.lat_k2_1_2_2
        all_parameters_lat_k3_1_2_2 = self.all_parameters.lat_k3_1_2_2
        all_parameters_lat_k4_1_2_2 = self.all_parameters.lat_k4_1_2_2
        understeer_corr_local_21 = nnodely_layers_parametricfunction_understeer_corr_local(einsum_28, einsum_27, all_constants_steer_center_2, all_constants_vx_center_1, einsum_26, all_constants_ax_center_2, all_parameters_lat_k1_1_2_2, all_parameters_lat_k2_1_2_2, all_parameters_lat_k3_1_2_2, all_parameters_lat_k4_1_2_2);  all_parameters_lat_k1_1_2_2 = all_parameters_lat_k2_1_2_2 = all_parameters_lat_k3_1_2_2 = all_parameters_lat_k4_1_2_2 = None
        mul_72 = understeer_corr_local_21 * mul_71;  understeer_corr_local_21 = mul_71 = None
        relation_forward_select1861_w = self.all_constants.Select1861
        einsum_92 = torch.functional.einsum('ijk,k->ij', repeat_4, relation_forward_select1861_w);  relation_forward_select1861_w = None
        unsqueeze_74 = einsum_92.unsqueeze(2);  einsum_92 = None
        relation_forward_select1859_w = self.all_constants.Select1859
        einsum_93 = torch.functional.einsum('ijk,k->ij', repeat_5, relation_forward_select1859_w);  relation_forward_select1859_w = None
        unsqueeze_75 = einsum_93.unsqueeze(2);  einsum_93 = None
        relation_forward_select1858_w = self.all_constants.Select1858
        einsum_94 = torch.functional.einsum('ijk,k->ij', repeat_6, relation_forward_select1858_w);  relation_forward_select1858_w = None
        unsqueeze_76 = einsum_94.unsqueeze(2);  einsum_94 = None
        mul_73 = unsqueeze_76 * unsqueeze_75;  unsqueeze_76 = unsqueeze_75 = None
        mul_74 = mul_73 * unsqueeze_74;  mul_73 = unsqueeze_74 = None
        all_parameters_lat_k1_1_2_1 = self.all_parameters.lat_k1_1_2_1
        all_parameters_lat_k2_1_2_1 = self.all_parameters.lat_k2_1_2_1
        all_parameters_lat_k3_1_2_1 = self.all_parameters.lat_k3_1_2_1
        all_parameters_lat_k4_1_2_1 = self.all_parameters.lat_k4_1_2_1
        understeer_corr_local_22 = nnodely_layers_parametricfunction_understeer_corr_local(einsum_28, einsum_27, all_constants_steer_center_2, all_constants_vx_center_1, einsum_26, all_constants_ax_center_1, all_parameters_lat_k1_1_2_1, all_parameters_lat_k2_1_2_1, all_parameters_lat_k3_1_2_1, all_parameters_lat_k4_1_2_1);  all_parameters_lat_k1_1_2_1 = all_parameters_lat_k2_1_2_1 = all_parameters_lat_k3_1_2_1 = all_parameters_lat_k4_1_2_1 = None
        mul_75 = understeer_corr_local_22 * mul_74;  understeer_corr_local_22 = mul_74 = None
        relation_forward_select1840_w = self.all_constants.Select1840
        einsum_95 = torch.functional.einsum('ijk,k->ij', repeat_4, relation_forward_select1840_w);  relation_forward_select1840_w = None
        unsqueeze_77 = einsum_95.unsqueeze(2);  einsum_95 = None
        relation_forward_select1838_w = self.all_constants.Select1838
        einsum_96 = torch.functional.einsum('ijk,k->ij', repeat_5, relation_forward_select1838_w);  relation_forward_select1838_w = None
        unsqueeze_78 = einsum_96.unsqueeze(2);  einsum_96 = None
        relation_forward_select1837_w = self.all_constants.Select1837
        einsum_97 = torch.functional.einsum('ijk,k->ij', repeat_6, relation_forward_select1837_w);  relation_forward_select1837_w = None
        unsqueeze_79 = einsum_97.unsqueeze(2);  einsum_97 = None
        mul_76 = unsqueeze_79 * unsqueeze_78;  unsqueeze_79 = unsqueeze_78 = None
        mul_77 = mul_76 * unsqueeze_77;  mul_76 = unsqueeze_77 = None
        all_parameters_lat_k1_1_2_0 = self.all_parameters.lat_k1_1_2_0
        all_parameters_lat_k2_1_2_0 = self.all_parameters.lat_k2_1_2_0
        all_parameters_lat_k3_1_2_0 = self.all_parameters.lat_k3_1_2_0
        all_parameters_lat_k4_1_2_0 = self.all_parameters.lat_k4_1_2_0
        understeer_corr_local_23 = nnodely_layers_parametricfunction_understeer_corr_local(einsum_28, einsum_27, all_constants_steer_center_2, all_constants_vx_center_1, einsum_26, all_constants_ax_center_0, all_parameters_lat_k1_1_2_0, all_parameters_lat_k2_1_2_0, all_parameters_lat_k3_1_2_0, all_parameters_lat_k4_1_2_0);  all_parameters_lat_k1_1_2_0 = all_parameters_lat_k2_1_2_0 = all_parameters_lat_k3_1_2_0 = all_parameters_lat_k4_1_2_0 = None
        mul_78 = understeer_corr_local_23 * mul_77;  understeer_corr_local_23 = mul_77 = None
        relation_forward_select1819_w = self.all_constants.Select1819
        einsum_98 = torch.functional.einsum('ijk,k->ij', repeat_4, relation_forward_select1819_w);  relation_forward_select1819_w = None
        unsqueeze_80 = einsum_98.unsqueeze(2);  einsum_98 = None
        relation_forward_select1817_w = self.all_constants.Select1817
        einsum_99 = torch.functional.einsum('ijk,k->ij', repeat_5, relation_forward_select1817_w);  relation_forward_select1817_w = None
        unsqueeze_81 = einsum_99.unsqueeze(2);  einsum_99 = None
        relation_forward_select1816_w = self.all_constants.Select1816
        einsum_100 = torch.functional.einsum('ijk,k->ij', repeat_6, relation_forward_select1816_w);  relation_forward_select1816_w = None
        unsqueeze_82 = einsum_100.unsqueeze(2);  einsum_100 = None
        mul_79 = unsqueeze_82 * unsqueeze_81;  unsqueeze_82 = unsqueeze_81 = None
        mul_80 = mul_79 * unsqueeze_80;  mul_79 = unsqueeze_80 = None
        all_parameters_lat_k1_1_1_2 = self.all_parameters.lat_k1_1_1_2
        all_parameters_lat_k2_1_1_2 = self.all_parameters.lat_k2_1_1_2
        all_parameters_lat_k3_1_1_2 = self.all_parameters.lat_k3_1_1_2
        all_parameters_lat_k4_1_1_2 = self.all_parameters.lat_k4_1_1_2
        understeer_corr_local_24 = nnodely_layers_parametricfunction_understeer_corr_local(einsum_28, einsum_27, all_constants_steer_center_1, all_constants_vx_center_1, einsum_26, all_constants_ax_center_2, all_parameters_lat_k1_1_1_2, all_parameters_lat_k2_1_1_2, all_parameters_lat_k3_1_1_2, all_parameters_lat_k4_1_1_2);  all_parameters_lat_k1_1_1_2 = all_parameters_lat_k2_1_1_2 = all_parameters_lat_k3_1_1_2 = all_parameters_lat_k4_1_1_2 = None
        mul_81 = understeer_corr_local_24 * mul_80;  understeer_corr_local_24 = mul_80 = None
        relation_forward_select1798_w = self.all_constants.Select1798
        einsum_101 = torch.functional.einsum('ijk,k->ij', repeat_4, relation_forward_select1798_w);  relation_forward_select1798_w = None
        unsqueeze_83 = einsum_101.unsqueeze(2);  einsum_101 = None
        relation_forward_select1796_w = self.all_constants.Select1796
        einsum_102 = torch.functional.einsum('ijk,k->ij', repeat_5, relation_forward_select1796_w);  relation_forward_select1796_w = None
        unsqueeze_84 = einsum_102.unsqueeze(2);  einsum_102 = None
        relation_forward_select1795_w = self.all_constants.Select1795
        einsum_103 = torch.functional.einsum('ijk,k->ij', repeat_6, relation_forward_select1795_w);  relation_forward_select1795_w = None
        unsqueeze_85 = einsum_103.unsqueeze(2);  einsum_103 = None
        mul_82 = unsqueeze_85 * unsqueeze_84;  unsqueeze_85 = unsqueeze_84 = None
        mul_83 = mul_82 * unsqueeze_83;  mul_82 = unsqueeze_83 = None
        all_parameters_lat_k1_1_1_1 = self.all_parameters.lat_k1_1_1_1
        all_parameters_lat_k2_1_1_1 = self.all_parameters.lat_k2_1_1_1
        all_parameters_lat_k3_1_1_1 = self.all_parameters.lat_k3_1_1_1
        all_parameters_lat_k4_1_1_1 = self.all_parameters.lat_k4_1_1_1
        understeer_corr_local_25 = nnodely_layers_parametricfunction_understeer_corr_local(einsum_28, einsum_27, all_constants_steer_center_1, all_constants_vx_center_1, einsum_26, all_constants_ax_center_1, all_parameters_lat_k1_1_1_1, all_parameters_lat_k2_1_1_1, all_parameters_lat_k3_1_1_1, all_parameters_lat_k4_1_1_1);  all_parameters_lat_k1_1_1_1 = all_parameters_lat_k2_1_1_1 = all_parameters_lat_k3_1_1_1 = all_parameters_lat_k4_1_1_1 = None
        mul_84 = understeer_corr_local_25 * mul_83;  understeer_corr_local_25 = mul_83 = None
        relation_forward_select1777_w = self.all_constants.Select1777
        einsum_104 = torch.functional.einsum('ijk,k->ij', repeat_4, relation_forward_select1777_w);  relation_forward_select1777_w = None
        unsqueeze_86 = einsum_104.unsqueeze(2);  einsum_104 = None
        relation_forward_select1775_w = self.all_constants.Select1775
        einsum_105 = torch.functional.einsum('ijk,k->ij', repeat_5, relation_forward_select1775_w);  relation_forward_select1775_w = None
        unsqueeze_87 = einsum_105.unsqueeze(2);  einsum_105 = None
        relation_forward_select1774_w = self.all_constants.Select1774
        einsum_106 = torch.functional.einsum('ijk,k->ij', repeat_6, relation_forward_select1774_w);  relation_forward_select1774_w = None
        unsqueeze_88 = einsum_106.unsqueeze(2);  einsum_106 = None
        mul_85 = unsqueeze_88 * unsqueeze_87;  unsqueeze_88 = unsqueeze_87 = None
        mul_86 = mul_85 * unsqueeze_86;  mul_85 = unsqueeze_86 = None
        all_parameters_lat_k1_1_1_0 = self.all_parameters.lat_k1_1_1_0
        all_parameters_lat_k2_1_1_0 = self.all_parameters.lat_k2_1_1_0
        all_parameters_lat_k3_1_1_0 = self.all_parameters.lat_k3_1_1_0
        all_parameters_lat_k4_1_1_0 = self.all_parameters.lat_k4_1_1_0
        understeer_corr_local_26 = nnodely_layers_parametricfunction_understeer_corr_local(einsum_28, einsum_27, all_constants_steer_center_1, all_constants_vx_center_1, einsum_26, all_constants_ax_center_0, all_parameters_lat_k1_1_1_0, all_parameters_lat_k2_1_1_0, all_parameters_lat_k3_1_1_0, all_parameters_lat_k4_1_1_0);  all_parameters_lat_k1_1_1_0 = all_parameters_lat_k2_1_1_0 = all_parameters_lat_k3_1_1_0 = all_parameters_lat_k4_1_1_0 = None
        mul_87 = understeer_corr_local_26 * mul_86;  understeer_corr_local_26 = mul_86 = None
        relation_forward_select1756_w = self.all_constants.Select1756
        einsum_107 = torch.functional.einsum('ijk,k->ij', repeat_4, relation_forward_select1756_w);  relation_forward_select1756_w = None
        unsqueeze_89 = einsum_107.unsqueeze(2);  einsum_107 = None
        relation_forward_select1754_w = self.all_constants.Select1754
        einsum_108 = torch.functional.einsum('ijk,k->ij', repeat_5, relation_forward_select1754_w);  relation_forward_select1754_w = None
        unsqueeze_90 = einsum_108.unsqueeze(2);  einsum_108 = None
        relation_forward_select1753_w = self.all_constants.Select1753
        einsum_109 = torch.functional.einsum('ijk,k->ij', repeat_6, relation_forward_select1753_w);  relation_forward_select1753_w = None
        unsqueeze_91 = einsum_109.unsqueeze(2);  einsum_109 = None
        mul_88 = unsqueeze_91 * unsqueeze_90;  unsqueeze_91 = unsqueeze_90 = None
        mul_89 = mul_88 * unsqueeze_89;  mul_88 = unsqueeze_89 = None
        all_parameters_lat_k1_1_0_2 = self.all_parameters.lat_k1_1_0_2
        all_parameters_lat_k2_1_0_2 = self.all_parameters.lat_k2_1_0_2
        all_parameters_lat_k3_1_0_2 = self.all_parameters.lat_k3_1_0_2
        all_parameters_lat_k4_1_0_2 = self.all_parameters.lat_k4_1_0_2
        understeer_corr_local_27 = nnodely_layers_parametricfunction_understeer_corr_local(einsum_28, einsum_27, all_constants_steer_center_0, all_constants_vx_center_1, einsum_26, all_constants_ax_center_2, all_parameters_lat_k1_1_0_2, all_parameters_lat_k2_1_0_2, all_parameters_lat_k3_1_0_2, all_parameters_lat_k4_1_0_2);  all_parameters_lat_k1_1_0_2 = all_parameters_lat_k2_1_0_2 = all_parameters_lat_k3_1_0_2 = all_parameters_lat_k4_1_0_2 = None
        mul_90 = understeer_corr_local_27 * mul_89;  understeer_corr_local_27 = mul_89 = None
        relation_forward_select1735_w = self.all_constants.Select1735
        einsum_110 = torch.functional.einsum('ijk,k->ij', repeat_4, relation_forward_select1735_w);  relation_forward_select1735_w = None
        unsqueeze_92 = einsum_110.unsqueeze(2);  einsum_110 = None
        relation_forward_select1733_w = self.all_constants.Select1733
        einsum_111 = torch.functional.einsum('ijk,k->ij', repeat_5, relation_forward_select1733_w);  relation_forward_select1733_w = None
        unsqueeze_93 = einsum_111.unsqueeze(2);  einsum_111 = None
        relation_forward_select1732_w = self.all_constants.Select1732
        einsum_112 = torch.functional.einsum('ijk,k->ij', repeat_6, relation_forward_select1732_w);  relation_forward_select1732_w = None
        unsqueeze_94 = einsum_112.unsqueeze(2);  einsum_112 = None
        mul_91 = unsqueeze_94 * unsqueeze_93;  unsqueeze_94 = unsqueeze_93 = None
        mul_92 = mul_91 * unsqueeze_92;  mul_91 = unsqueeze_92 = None
        all_parameters_lat_k1_1_0_1 = self.all_parameters.lat_k1_1_0_1
        all_parameters_lat_k2_1_0_1 = self.all_parameters.lat_k2_1_0_1
        all_parameters_lat_k3_1_0_1 = self.all_parameters.lat_k3_1_0_1
        all_parameters_lat_k4_1_0_1 = self.all_parameters.lat_k4_1_0_1
        understeer_corr_local_28 = nnodely_layers_parametricfunction_understeer_corr_local(einsum_28, einsum_27, all_constants_steer_center_0, all_constants_vx_center_1, einsum_26, all_constants_ax_center_1, all_parameters_lat_k1_1_0_1, all_parameters_lat_k2_1_0_1, all_parameters_lat_k3_1_0_1, all_parameters_lat_k4_1_0_1);  all_parameters_lat_k1_1_0_1 = all_parameters_lat_k2_1_0_1 = all_parameters_lat_k3_1_0_1 = all_parameters_lat_k4_1_0_1 = None
        mul_93 = understeer_corr_local_28 * mul_92;  understeer_corr_local_28 = mul_92 = None
        relation_forward_select1714_w = self.all_constants.Select1714
        einsum_113 = torch.functional.einsum('ijk,k->ij', repeat_4, relation_forward_select1714_w);  relation_forward_select1714_w = None
        unsqueeze_95 = einsum_113.unsqueeze(2);  einsum_113 = None
        relation_forward_select1712_w = self.all_constants.Select1712
        einsum_114 = torch.functional.einsum('ijk,k->ij', repeat_5, relation_forward_select1712_w);  relation_forward_select1712_w = None
        unsqueeze_96 = einsum_114.unsqueeze(2);  einsum_114 = None
        relation_forward_select1711_w = self.all_constants.Select1711
        einsum_115 = torch.functional.einsum('ijk,k->ij', repeat_6, relation_forward_select1711_w);  relation_forward_select1711_w = None
        unsqueeze_97 = einsum_115.unsqueeze(2);  einsum_115 = None
        mul_94 = unsqueeze_97 * unsqueeze_96;  unsqueeze_97 = unsqueeze_96 = None
        mul_95 = mul_94 * unsqueeze_95;  mul_94 = unsqueeze_95 = None
        all_parameters_lat_k1_1_0_0 = self.all_parameters.lat_k1_1_0_0
        all_parameters_lat_k2_1_0_0 = self.all_parameters.lat_k2_1_0_0
        all_parameters_lat_k3_1_0_0 = self.all_parameters.lat_k3_1_0_0
        all_parameters_lat_k4_1_0_0 = self.all_parameters.lat_k4_1_0_0
        understeer_corr_local_29 = nnodely_layers_parametricfunction_understeer_corr_local(einsum_28, einsum_27, all_constants_steer_center_0, all_constants_vx_center_1, einsum_26, all_constants_ax_center_0, all_parameters_lat_k1_1_0_0, all_parameters_lat_k2_1_0_0, all_parameters_lat_k3_1_0_0, all_parameters_lat_k4_1_0_0);  all_constants_vx_center_1 = all_parameters_lat_k1_1_0_0 = all_parameters_lat_k2_1_0_0 = all_parameters_lat_k3_1_0_0 = all_parameters_lat_k4_1_0_0 = None
        mul_96 = understeer_corr_local_29 * mul_95;  understeer_corr_local_29 = mul_95 = None
        relation_forward_select1693_w = self.all_constants.Select1693
        einsum_116 = torch.functional.einsum('ijk,k->ij', repeat_4, relation_forward_select1693_w);  relation_forward_select1693_w = None
        unsqueeze_98 = einsum_116.unsqueeze(2);  einsum_116 = None
        relation_forward_select1691_w = self.all_constants.Select1691
        einsum_117 = torch.functional.einsum('ijk,k->ij', repeat_5, relation_forward_select1691_w);  relation_forward_select1691_w = None
        unsqueeze_99 = einsum_117.unsqueeze(2);  einsum_117 = None
        relation_forward_select1690_w = self.all_constants.Select1690
        einsum_118 = torch.functional.einsum('ijk,k->ij', repeat_6, relation_forward_select1690_w);  relation_forward_select1690_w = None
        unsqueeze_100 = einsum_118.unsqueeze(2);  einsum_118 = None
        mul_97 = unsqueeze_100 * unsqueeze_99;  unsqueeze_100 = unsqueeze_99 = None
        mul_98 = mul_97 * unsqueeze_98;  mul_97 = unsqueeze_98 = None
        all_constants_vx_center_0 = self.all_constants.vx_center_0
        all_parameters_lat_k1_0_4_2 = self.all_parameters.lat_k1_0_4_2
        all_parameters_lat_k2_0_4_2 = self.all_parameters.lat_k2_0_4_2
        all_parameters_lat_k3_0_4_2 = self.all_parameters.lat_k3_0_4_2
        all_parameters_lat_k4_0_4_2 = self.all_parameters.lat_k4_0_4_2
        understeer_corr_local_30 = nnodely_layers_parametricfunction_understeer_corr_local(einsum_28, einsum_27, all_constants_steer_center_4, all_constants_vx_center_0, einsum_26, all_constants_ax_center_2, all_parameters_lat_k1_0_4_2, all_parameters_lat_k2_0_4_2, all_parameters_lat_k3_0_4_2, all_parameters_lat_k4_0_4_2);  all_parameters_lat_k1_0_4_2 = all_parameters_lat_k2_0_4_2 = all_parameters_lat_k3_0_4_2 = all_parameters_lat_k4_0_4_2 = None
        mul_99 = understeer_corr_local_30 * mul_98;  understeer_corr_local_30 = mul_98 = None
        relation_forward_select1672_w = self.all_constants.Select1672
        einsum_119 = torch.functional.einsum('ijk,k->ij', repeat_4, relation_forward_select1672_w);  relation_forward_select1672_w = None
        unsqueeze_101 = einsum_119.unsqueeze(2);  einsum_119 = None
        relation_forward_select1670_w = self.all_constants.Select1670
        einsum_120 = torch.functional.einsum('ijk,k->ij', repeat_5, relation_forward_select1670_w);  relation_forward_select1670_w = None
        unsqueeze_102 = einsum_120.unsqueeze(2);  einsum_120 = None
        relation_forward_select1669_w = self.all_constants.Select1669
        einsum_121 = torch.functional.einsum('ijk,k->ij', repeat_6, relation_forward_select1669_w);  relation_forward_select1669_w = None
        unsqueeze_103 = einsum_121.unsqueeze(2);  einsum_121 = None
        mul_100 = unsqueeze_103 * unsqueeze_102;  unsqueeze_103 = unsqueeze_102 = None
        mul_101 = mul_100 * unsqueeze_101;  mul_100 = unsqueeze_101 = None
        all_parameters_lat_k1_0_4_1 = self.all_parameters.lat_k1_0_4_1
        all_parameters_lat_k2_0_4_1 = self.all_parameters.lat_k2_0_4_1
        all_parameters_lat_k3_0_4_1 = self.all_parameters.lat_k3_0_4_1
        all_parameters_lat_k4_0_4_1 = self.all_parameters.lat_k4_0_4_1
        understeer_corr_local_31 = nnodely_layers_parametricfunction_understeer_corr_local(einsum_28, einsum_27, all_constants_steer_center_4, all_constants_vx_center_0, einsum_26, all_constants_ax_center_1, all_parameters_lat_k1_0_4_1, all_parameters_lat_k2_0_4_1, all_parameters_lat_k3_0_4_1, all_parameters_lat_k4_0_4_1);  all_parameters_lat_k1_0_4_1 = all_parameters_lat_k2_0_4_1 = all_parameters_lat_k3_0_4_1 = all_parameters_lat_k4_0_4_1 = None
        mul_102 = understeer_corr_local_31 * mul_101;  understeer_corr_local_31 = mul_101 = None
        relation_forward_select1651_w = self.all_constants.Select1651
        einsum_122 = torch.functional.einsum('ijk,k->ij', repeat_4, relation_forward_select1651_w);  relation_forward_select1651_w = None
        unsqueeze_104 = einsum_122.unsqueeze(2);  einsum_122 = None
        relation_forward_select1649_w = self.all_constants.Select1649
        einsum_123 = torch.functional.einsum('ijk,k->ij', repeat_5, relation_forward_select1649_w);  relation_forward_select1649_w = None
        unsqueeze_105 = einsum_123.unsqueeze(2);  einsum_123 = None
        relation_forward_select1648_w = self.all_constants.Select1648
        einsum_124 = torch.functional.einsum('ijk,k->ij', repeat_6, relation_forward_select1648_w);  relation_forward_select1648_w = None
        unsqueeze_106 = einsum_124.unsqueeze(2);  einsum_124 = None
        mul_103 = unsqueeze_106 * unsqueeze_105;  unsqueeze_106 = unsqueeze_105 = None
        mul_104 = mul_103 * unsqueeze_104;  mul_103 = unsqueeze_104 = None
        all_parameters_lat_k1_0_4_0 = self.all_parameters.lat_k1_0_4_0
        all_parameters_lat_k2_0_4_0 = self.all_parameters.lat_k2_0_4_0
        all_parameters_lat_k3_0_4_0 = self.all_parameters.lat_k3_0_4_0
        all_parameters_lat_k4_0_4_0 = self.all_parameters.lat_k4_0_4_0
        understeer_corr_local_32 = nnodely_layers_parametricfunction_understeer_corr_local(einsum_28, einsum_27, all_constants_steer_center_4, all_constants_vx_center_0, einsum_26, all_constants_ax_center_0, all_parameters_lat_k1_0_4_0, all_parameters_lat_k2_0_4_0, all_parameters_lat_k3_0_4_0, all_parameters_lat_k4_0_4_0);  all_constants_steer_center_4 = all_parameters_lat_k1_0_4_0 = all_parameters_lat_k2_0_4_0 = all_parameters_lat_k3_0_4_0 = all_parameters_lat_k4_0_4_0 = None
        mul_105 = understeer_corr_local_32 * mul_104;  understeer_corr_local_32 = mul_104 = None
        relation_forward_select1630_w = self.all_constants.Select1630
        einsum_125 = torch.functional.einsum('ijk,k->ij', repeat_4, relation_forward_select1630_w);  relation_forward_select1630_w = None
        unsqueeze_107 = einsum_125.unsqueeze(2);  einsum_125 = None
        relation_forward_select1628_w = self.all_constants.Select1628
        einsum_126 = torch.functional.einsum('ijk,k->ij', repeat_5, relation_forward_select1628_w);  relation_forward_select1628_w = None
        unsqueeze_108 = einsum_126.unsqueeze(2);  einsum_126 = None
        relation_forward_select1627_w = self.all_constants.Select1627
        einsum_127 = torch.functional.einsum('ijk,k->ij', repeat_6, relation_forward_select1627_w);  relation_forward_select1627_w = None
        unsqueeze_109 = einsum_127.unsqueeze(2);  einsum_127 = None
        mul_106 = unsqueeze_109 * unsqueeze_108;  unsqueeze_109 = unsqueeze_108 = None
        mul_107 = mul_106 * unsqueeze_107;  mul_106 = unsqueeze_107 = None
        all_parameters_lat_k1_0_3_2 = self.all_parameters.lat_k1_0_3_2
        all_parameters_lat_k2_0_3_2 = self.all_parameters.lat_k2_0_3_2
        all_parameters_lat_k3_0_3_2 = self.all_parameters.lat_k3_0_3_2
        all_parameters_lat_k4_0_3_2 = self.all_parameters.lat_k4_0_3_2
        understeer_corr_local_33 = nnodely_layers_parametricfunction_understeer_corr_local(einsum_28, einsum_27, all_constants_steer_center_3, all_constants_vx_center_0, einsum_26, all_constants_ax_center_2, all_parameters_lat_k1_0_3_2, all_parameters_lat_k2_0_3_2, all_parameters_lat_k3_0_3_2, all_parameters_lat_k4_0_3_2);  all_parameters_lat_k1_0_3_2 = all_parameters_lat_k2_0_3_2 = all_parameters_lat_k3_0_3_2 = all_parameters_lat_k4_0_3_2 = None
        mul_108 = understeer_corr_local_33 * mul_107;  understeer_corr_local_33 = mul_107 = None
        relation_forward_select1609_w = self.all_constants.Select1609
        einsum_128 = torch.functional.einsum('ijk,k->ij', repeat_4, relation_forward_select1609_w);  relation_forward_select1609_w = None
        unsqueeze_110 = einsum_128.unsqueeze(2);  einsum_128 = None
        relation_forward_select1607_w = self.all_constants.Select1607
        einsum_129 = torch.functional.einsum('ijk,k->ij', repeat_5, relation_forward_select1607_w);  relation_forward_select1607_w = None
        unsqueeze_111 = einsum_129.unsqueeze(2);  einsum_129 = None
        relation_forward_select1606_w = self.all_constants.Select1606
        einsum_130 = torch.functional.einsum('ijk,k->ij', repeat_6, relation_forward_select1606_w);  relation_forward_select1606_w = None
        unsqueeze_112 = einsum_130.unsqueeze(2);  einsum_130 = None
        mul_109 = unsqueeze_112 * unsqueeze_111;  unsqueeze_112 = unsqueeze_111 = None
        mul_110 = mul_109 * unsqueeze_110;  mul_109 = unsqueeze_110 = None
        all_parameters_lat_k1_0_3_1 = self.all_parameters.lat_k1_0_3_1
        all_parameters_lat_k2_0_3_1 = self.all_parameters.lat_k2_0_3_1
        all_parameters_lat_k3_0_3_1 = self.all_parameters.lat_k3_0_3_1
        all_parameters_lat_k4_0_3_1 = self.all_parameters.lat_k4_0_3_1
        understeer_corr_local_34 = nnodely_layers_parametricfunction_understeer_corr_local(einsum_28, einsum_27, all_constants_steer_center_3, all_constants_vx_center_0, einsum_26, all_constants_ax_center_1, all_parameters_lat_k1_0_3_1, all_parameters_lat_k2_0_3_1, all_parameters_lat_k3_0_3_1, all_parameters_lat_k4_0_3_1);  all_parameters_lat_k1_0_3_1 = all_parameters_lat_k2_0_3_1 = all_parameters_lat_k3_0_3_1 = all_parameters_lat_k4_0_3_1 = None
        mul_111 = understeer_corr_local_34 * mul_110;  understeer_corr_local_34 = mul_110 = None
        relation_forward_select1588_w = self.all_constants.Select1588
        einsum_131 = torch.functional.einsum('ijk,k->ij', repeat_4, relation_forward_select1588_w);  relation_forward_select1588_w = None
        unsqueeze_113 = einsum_131.unsqueeze(2);  einsum_131 = None
        relation_forward_select1586_w = self.all_constants.Select1586
        einsum_132 = torch.functional.einsum('ijk,k->ij', repeat_5, relation_forward_select1586_w);  relation_forward_select1586_w = None
        unsqueeze_114 = einsum_132.unsqueeze(2);  einsum_132 = None
        relation_forward_select1585_w = self.all_constants.Select1585
        einsum_133 = torch.functional.einsum('ijk,k->ij', repeat_6, relation_forward_select1585_w);  relation_forward_select1585_w = None
        unsqueeze_115 = einsum_133.unsqueeze(2);  einsum_133 = None
        mul_112 = unsqueeze_115 * unsqueeze_114;  unsqueeze_115 = unsqueeze_114 = None
        mul_113 = mul_112 * unsqueeze_113;  mul_112 = unsqueeze_113 = None
        all_parameters_lat_k1_0_3_0 = self.all_parameters.lat_k1_0_3_0
        all_parameters_lat_k2_0_3_0 = self.all_parameters.lat_k2_0_3_0
        all_parameters_lat_k3_0_3_0 = self.all_parameters.lat_k3_0_3_0
        all_parameters_lat_k4_0_3_0 = self.all_parameters.lat_k4_0_3_0
        understeer_corr_local_35 = nnodely_layers_parametricfunction_understeer_corr_local(einsum_28, einsum_27, all_constants_steer_center_3, all_constants_vx_center_0, einsum_26, all_constants_ax_center_0, all_parameters_lat_k1_0_3_0, all_parameters_lat_k2_0_3_0, all_parameters_lat_k3_0_3_0, all_parameters_lat_k4_0_3_0);  all_constants_steer_center_3 = all_parameters_lat_k1_0_3_0 = all_parameters_lat_k2_0_3_0 = all_parameters_lat_k3_0_3_0 = all_parameters_lat_k4_0_3_0 = None
        mul_114 = understeer_corr_local_35 * mul_113;  understeer_corr_local_35 = mul_113 = None
        relation_forward_select1567_w = self.all_constants.Select1567
        einsum_134 = torch.functional.einsum('ijk,k->ij', repeat_4, relation_forward_select1567_w);  relation_forward_select1567_w = None
        unsqueeze_116 = einsum_134.unsqueeze(2);  einsum_134 = None
        relation_forward_select1565_w = self.all_constants.Select1565
        einsum_135 = torch.functional.einsum('ijk,k->ij', repeat_5, relation_forward_select1565_w);  relation_forward_select1565_w = None
        unsqueeze_117 = einsum_135.unsqueeze(2);  einsum_135 = None
        relation_forward_select1564_w = self.all_constants.Select1564
        einsum_136 = torch.functional.einsum('ijk,k->ij', repeat_6, relation_forward_select1564_w);  relation_forward_select1564_w = None
        unsqueeze_118 = einsum_136.unsqueeze(2);  einsum_136 = None
        mul_115 = unsqueeze_118 * unsqueeze_117;  unsqueeze_118 = unsqueeze_117 = None
        mul_116 = mul_115 * unsqueeze_116;  mul_115 = unsqueeze_116 = None
        all_parameters_lat_k1_0_2_2 = self.all_parameters.lat_k1_0_2_2
        all_parameters_lat_k2_0_2_2 = self.all_parameters.lat_k2_0_2_2
        all_parameters_lat_k3_0_2_2 = self.all_parameters.lat_k3_0_2_2
        all_parameters_lat_k4_0_2_2 = self.all_parameters.lat_k4_0_2_2
        understeer_corr_local_36 = nnodely_layers_parametricfunction_understeer_corr_local(einsum_28, einsum_27, all_constants_steer_center_2, all_constants_vx_center_0, einsum_26, all_constants_ax_center_2, all_parameters_lat_k1_0_2_2, all_parameters_lat_k2_0_2_2, all_parameters_lat_k3_0_2_2, all_parameters_lat_k4_0_2_2);  all_parameters_lat_k1_0_2_2 = all_parameters_lat_k2_0_2_2 = all_parameters_lat_k3_0_2_2 = all_parameters_lat_k4_0_2_2 = None
        mul_117 = understeer_corr_local_36 * mul_116;  understeer_corr_local_36 = mul_116 = None
        relation_forward_select1546_w = self.all_constants.Select1546
        einsum_137 = torch.functional.einsum('ijk,k->ij', repeat_4, relation_forward_select1546_w);  relation_forward_select1546_w = None
        unsqueeze_119 = einsum_137.unsqueeze(2);  einsum_137 = None
        relation_forward_select1544_w = self.all_constants.Select1544
        einsum_138 = torch.functional.einsum('ijk,k->ij', repeat_5, relation_forward_select1544_w);  relation_forward_select1544_w = None
        unsqueeze_120 = einsum_138.unsqueeze(2);  einsum_138 = None
        relation_forward_select1543_w = self.all_constants.Select1543
        einsum_139 = torch.functional.einsum('ijk,k->ij', repeat_6, relation_forward_select1543_w);  relation_forward_select1543_w = None
        unsqueeze_121 = einsum_139.unsqueeze(2);  einsum_139 = None
        mul_118 = unsqueeze_121 * unsqueeze_120;  unsqueeze_121 = unsqueeze_120 = None
        mul_119 = mul_118 * unsqueeze_119;  mul_118 = unsqueeze_119 = None
        all_parameters_lat_k1_0_2_1 = self.all_parameters.lat_k1_0_2_1
        all_parameters_lat_k2_0_2_1 = self.all_parameters.lat_k2_0_2_1
        all_parameters_lat_k3_0_2_1 = self.all_parameters.lat_k3_0_2_1
        all_parameters_lat_k4_0_2_1 = self.all_parameters.lat_k4_0_2_1
        understeer_corr_local_37 = nnodely_layers_parametricfunction_understeer_corr_local(einsum_28, einsum_27, all_constants_steer_center_2, all_constants_vx_center_0, einsum_26, all_constants_ax_center_1, all_parameters_lat_k1_0_2_1, all_parameters_lat_k2_0_2_1, all_parameters_lat_k3_0_2_1, all_parameters_lat_k4_0_2_1);  all_parameters_lat_k1_0_2_1 = all_parameters_lat_k2_0_2_1 = all_parameters_lat_k3_0_2_1 = all_parameters_lat_k4_0_2_1 = None
        mul_120 = understeer_corr_local_37 * mul_119;  understeer_corr_local_37 = mul_119 = None
        relation_forward_select1525_w = self.all_constants.Select1525
        einsum_140 = torch.functional.einsum('ijk,k->ij', repeat_4, relation_forward_select1525_w);  relation_forward_select1525_w = None
        unsqueeze_122 = einsum_140.unsqueeze(2);  einsum_140 = None
        relation_forward_select1523_w = self.all_constants.Select1523
        einsum_141 = torch.functional.einsum('ijk,k->ij', repeat_5, relation_forward_select1523_w);  relation_forward_select1523_w = None
        unsqueeze_123 = einsum_141.unsqueeze(2);  einsum_141 = None
        relation_forward_select1522_w = self.all_constants.Select1522
        einsum_142 = torch.functional.einsum('ijk,k->ij', repeat_6, relation_forward_select1522_w);  relation_forward_select1522_w = None
        unsqueeze_124 = einsum_142.unsqueeze(2);  einsum_142 = None
        mul_121 = unsqueeze_124 * unsqueeze_123;  unsqueeze_124 = unsqueeze_123 = None
        mul_122 = mul_121 * unsqueeze_122;  mul_121 = unsqueeze_122 = None
        all_parameters_lat_k1_0_2_0 = self.all_parameters.lat_k1_0_2_0
        all_parameters_lat_k2_0_2_0 = self.all_parameters.lat_k2_0_2_0
        all_parameters_lat_k3_0_2_0 = self.all_parameters.lat_k3_0_2_0
        all_parameters_lat_k4_0_2_0 = self.all_parameters.lat_k4_0_2_0
        understeer_corr_local_38 = nnodely_layers_parametricfunction_understeer_corr_local(einsum_28, einsum_27, all_constants_steer_center_2, all_constants_vx_center_0, einsum_26, all_constants_ax_center_0, all_parameters_lat_k1_0_2_0, all_parameters_lat_k2_0_2_0, all_parameters_lat_k3_0_2_0, all_parameters_lat_k4_0_2_0);  all_constants_steer_center_2 = all_parameters_lat_k1_0_2_0 = all_parameters_lat_k2_0_2_0 = all_parameters_lat_k3_0_2_0 = all_parameters_lat_k4_0_2_0 = None
        mul_123 = understeer_corr_local_38 * mul_122;  understeer_corr_local_38 = mul_122 = None
        relation_forward_select1504_w = self.all_constants.Select1504
        einsum_143 = torch.functional.einsum('ijk,k->ij', repeat_4, relation_forward_select1504_w);  relation_forward_select1504_w = None
        unsqueeze_125 = einsum_143.unsqueeze(2);  einsum_143 = None
        relation_forward_select1502_w = self.all_constants.Select1502
        einsum_144 = torch.functional.einsum('ijk,k->ij', repeat_5, relation_forward_select1502_w);  relation_forward_select1502_w = None
        unsqueeze_126 = einsum_144.unsqueeze(2);  einsum_144 = None
        relation_forward_select1501_w = self.all_constants.Select1501
        einsum_145 = torch.functional.einsum('ijk,k->ij', repeat_6, relation_forward_select1501_w);  relation_forward_select1501_w = None
        unsqueeze_127 = einsum_145.unsqueeze(2);  einsum_145 = None
        mul_124 = unsqueeze_127 * unsqueeze_126;  unsqueeze_127 = unsqueeze_126 = None
        mul_125 = mul_124 * unsqueeze_125;  mul_124 = unsqueeze_125 = None
        all_parameters_lat_k1_0_1_2 = self.all_parameters.lat_k1_0_1_2
        all_parameters_lat_k2_0_1_2 = self.all_parameters.lat_k2_0_1_2
        all_parameters_lat_k3_0_1_2 = self.all_parameters.lat_k3_0_1_2
        all_parameters_lat_k4_0_1_2 = self.all_parameters.lat_k4_0_1_2
        understeer_corr_local_39 = nnodely_layers_parametricfunction_understeer_corr_local(einsum_28, einsum_27, all_constants_steer_center_1, all_constants_vx_center_0, einsum_26, all_constants_ax_center_2, all_parameters_lat_k1_0_1_2, all_parameters_lat_k2_0_1_2, all_parameters_lat_k3_0_1_2, all_parameters_lat_k4_0_1_2);  all_parameters_lat_k1_0_1_2 = all_parameters_lat_k2_0_1_2 = all_parameters_lat_k3_0_1_2 = all_parameters_lat_k4_0_1_2 = None
        mul_126 = understeer_corr_local_39 * mul_125;  understeer_corr_local_39 = mul_125 = None
        relation_forward_select1483_w = self.all_constants.Select1483
        einsum_146 = torch.functional.einsum('ijk,k->ij', repeat_4, relation_forward_select1483_w);  relation_forward_select1483_w = None
        unsqueeze_128 = einsum_146.unsqueeze(2);  einsum_146 = None
        relation_forward_select1481_w = self.all_constants.Select1481
        einsum_147 = torch.functional.einsum('ijk,k->ij', repeat_5, relation_forward_select1481_w);  relation_forward_select1481_w = None
        unsqueeze_129 = einsum_147.unsqueeze(2);  einsum_147 = None
        relation_forward_select1480_w = self.all_constants.Select1480
        einsum_148 = torch.functional.einsum('ijk,k->ij', repeat_6, relation_forward_select1480_w);  relation_forward_select1480_w = None
        unsqueeze_130 = einsum_148.unsqueeze(2);  einsum_148 = None
        mul_127 = unsqueeze_130 * unsqueeze_129;  unsqueeze_130 = unsqueeze_129 = None
        mul_128 = mul_127 * unsqueeze_128;  mul_127 = unsqueeze_128 = None
        all_parameters_lat_k1_0_1_1 = self.all_parameters.lat_k1_0_1_1
        all_parameters_lat_k2_0_1_1 = self.all_parameters.lat_k2_0_1_1
        all_parameters_lat_k3_0_1_1 = self.all_parameters.lat_k3_0_1_1
        all_parameters_lat_k4_0_1_1 = self.all_parameters.lat_k4_0_1_1
        understeer_corr_local_40 = nnodely_layers_parametricfunction_understeer_corr_local(einsum_28, einsum_27, all_constants_steer_center_1, all_constants_vx_center_0, einsum_26, all_constants_ax_center_1, all_parameters_lat_k1_0_1_1, all_parameters_lat_k2_0_1_1, all_parameters_lat_k3_0_1_1, all_parameters_lat_k4_0_1_1);  all_parameters_lat_k1_0_1_1 = all_parameters_lat_k2_0_1_1 = all_parameters_lat_k3_0_1_1 = all_parameters_lat_k4_0_1_1 = None
        mul_129 = understeer_corr_local_40 * mul_128;  understeer_corr_local_40 = mul_128 = None
        relation_forward_select1462_w = self.all_constants.Select1462
        einsum_149 = torch.functional.einsum('ijk,k->ij', repeat_4, relation_forward_select1462_w);  relation_forward_select1462_w = None
        unsqueeze_131 = einsum_149.unsqueeze(2);  einsum_149 = None
        relation_forward_select1460_w = self.all_constants.Select1460
        einsum_150 = torch.functional.einsum('ijk,k->ij', repeat_5, relation_forward_select1460_w);  relation_forward_select1460_w = None
        unsqueeze_132 = einsum_150.unsqueeze(2);  einsum_150 = None
        relation_forward_select1459_w = self.all_constants.Select1459
        einsum_151 = torch.functional.einsum('ijk,k->ij', repeat_6, relation_forward_select1459_w);  relation_forward_select1459_w = None
        unsqueeze_133 = einsum_151.unsqueeze(2);  einsum_151 = None
        mul_130 = unsqueeze_133 * unsqueeze_132;  unsqueeze_133 = unsqueeze_132 = None
        mul_131 = mul_130 * unsqueeze_131;  mul_130 = unsqueeze_131 = None
        all_parameters_lat_k1_0_1_0 = self.all_parameters.lat_k1_0_1_0
        all_parameters_lat_k2_0_1_0 = self.all_parameters.lat_k2_0_1_0
        all_parameters_lat_k3_0_1_0 = self.all_parameters.lat_k3_0_1_0
        all_parameters_lat_k4_0_1_0 = self.all_parameters.lat_k4_0_1_0
        understeer_corr_local_41 = nnodely_layers_parametricfunction_understeer_corr_local(einsum_28, einsum_27, all_constants_steer_center_1, all_constants_vx_center_0, einsum_26, all_constants_ax_center_0, all_parameters_lat_k1_0_1_0, all_parameters_lat_k2_0_1_0, all_parameters_lat_k3_0_1_0, all_parameters_lat_k4_0_1_0);  all_constants_steer_center_1 = all_parameters_lat_k1_0_1_0 = all_parameters_lat_k2_0_1_0 = all_parameters_lat_k3_0_1_0 = all_parameters_lat_k4_0_1_0 = None
        mul_132 = understeer_corr_local_41 * mul_131;  understeer_corr_local_41 = mul_131 = None
        relation_forward_select1441_w = self.all_constants.Select1441
        einsum_152 = torch.functional.einsum('ijk,k->ij', repeat_4, relation_forward_select1441_w);  relation_forward_select1441_w = None
        unsqueeze_134 = einsum_152.unsqueeze(2);  einsum_152 = None
        relation_forward_select1439_w = self.all_constants.Select1439
        einsum_153 = torch.functional.einsum('ijk,k->ij', repeat_5, relation_forward_select1439_w);  relation_forward_select1439_w = None
        unsqueeze_135 = einsum_153.unsqueeze(2);  einsum_153 = None
        relation_forward_select1438_w = self.all_constants.Select1438
        einsum_154 = torch.functional.einsum('ijk,k->ij', repeat_6, relation_forward_select1438_w);  relation_forward_select1438_w = None
        unsqueeze_136 = einsum_154.unsqueeze(2);  einsum_154 = None
        mul_133 = unsqueeze_136 * unsqueeze_135;  unsqueeze_136 = unsqueeze_135 = None
        mul_134 = mul_133 * unsqueeze_134;  mul_133 = unsqueeze_134 = None
        all_parameters_lat_k1_0_0_2 = self.all_parameters.lat_k1_0_0_2
        all_parameters_lat_k2_0_0_2 = self.all_parameters.lat_k2_0_0_2
        all_parameters_lat_k3_0_0_2 = self.all_parameters.lat_k3_0_0_2
        all_parameters_lat_k4_0_0_2 = self.all_parameters.lat_k4_0_0_2
        understeer_corr_local_42 = nnodely_layers_parametricfunction_understeer_corr_local(einsum_28, einsum_27, all_constants_steer_center_0, all_constants_vx_center_0, einsum_26, all_constants_ax_center_2, all_parameters_lat_k1_0_0_2, all_parameters_lat_k2_0_0_2, all_parameters_lat_k3_0_0_2, all_parameters_lat_k4_0_0_2);  all_constants_ax_center_2 = all_parameters_lat_k1_0_0_2 = all_parameters_lat_k2_0_0_2 = all_parameters_lat_k3_0_0_2 = all_parameters_lat_k4_0_0_2 = None
        mul_135 = understeer_corr_local_42 * mul_134;  understeer_corr_local_42 = mul_134 = None
        relation_forward_select1420_w = self.all_constants.Select1420
        einsum_155 = torch.functional.einsum('ijk,k->ij', repeat_4, relation_forward_select1420_w);  relation_forward_select1420_w = None
        unsqueeze_137 = einsum_155.unsqueeze(2);  einsum_155 = None
        relation_forward_select1418_w = self.all_constants.Select1418
        einsum_156 = torch.functional.einsum('ijk,k->ij', repeat_5, relation_forward_select1418_w);  relation_forward_select1418_w = None
        unsqueeze_138 = einsum_156.unsqueeze(2);  einsum_156 = None
        relation_forward_select1417_w = self.all_constants.Select1417
        einsum_157 = torch.functional.einsum('ijk,k->ij', repeat_6, relation_forward_select1417_w);  relation_forward_select1417_w = None
        unsqueeze_139 = einsum_157.unsqueeze(2);  einsum_157 = None
        mul_136 = unsqueeze_139 * unsqueeze_138;  unsqueeze_139 = unsqueeze_138 = None
        mul_137 = mul_136 * unsqueeze_137;  mul_136 = unsqueeze_137 = None
        all_parameters_lat_k1_0_0_1 = self.all_parameters.lat_k1_0_0_1
        all_parameters_lat_k2_0_0_1 = self.all_parameters.lat_k2_0_0_1
        all_parameters_lat_k3_0_0_1 = self.all_parameters.lat_k3_0_0_1
        all_parameters_lat_k4_0_0_1 = self.all_parameters.lat_k4_0_0_1
        understeer_corr_local_43 = nnodely_layers_parametricfunction_understeer_corr_local(einsum_28, einsum_27, all_constants_steer_center_0, all_constants_vx_center_0, einsum_26, all_constants_ax_center_1, all_parameters_lat_k1_0_0_1, all_parameters_lat_k2_0_0_1, all_parameters_lat_k3_0_0_1, all_parameters_lat_k4_0_0_1);  all_constants_ax_center_1 = all_parameters_lat_k1_0_0_1 = all_parameters_lat_k2_0_0_1 = all_parameters_lat_k3_0_0_1 = all_parameters_lat_k4_0_0_1 = None
        mul_138 = understeer_corr_local_43 * mul_137;  understeer_corr_local_43 = mul_137 = None
        relation_forward_select1399_w = self.all_constants.Select1399
        einsum_158 = torch.functional.einsum('ijk,k->ij', repeat_4, relation_forward_select1399_w);  repeat_4 = relation_forward_select1399_w = None
        unsqueeze_140 = einsum_158.unsqueeze(2);  einsum_158 = None
        relation_forward_select1397_w = self.all_constants.Select1397
        einsum_159 = torch.functional.einsum('ijk,k->ij', repeat_5, relation_forward_select1397_w);  repeat_5 = relation_forward_select1397_w = None
        unsqueeze_141 = einsum_159.unsqueeze(2);  einsum_159 = None
        relation_forward_select1396_w = self.all_constants.Select1396
        einsum_160 = torch.functional.einsum('ijk,k->ij', repeat_6, relation_forward_select1396_w);  repeat_6 = relation_forward_select1396_w = None
        unsqueeze_142 = einsum_160.unsqueeze(2);  einsum_160 = None
        mul_139 = unsqueeze_142 * unsqueeze_141;  unsqueeze_142 = unsqueeze_141 = None
        mul_140 = mul_139 * unsqueeze_140;  mul_139 = unsqueeze_140 = None
        all_parameters_lat_k1_0_0_0 = self.all_parameters.lat_k1_0_0_0
        all_parameters_lat_k2_0_0_0 = self.all_parameters.lat_k2_0_0_0
        all_parameters_lat_k3_0_0_0 = self.all_parameters.lat_k3_0_0_0
        all_parameters_lat_k4_0_0_0 = self.all_parameters.lat_k4_0_0_0
        understeer_corr_local_44 = nnodely_layers_parametricfunction_understeer_corr_local(einsum_28, einsum_27, all_constants_steer_center_0, all_constants_vx_center_0, einsum_26, all_constants_ax_center_0, all_parameters_lat_k1_0_0_0, all_parameters_lat_k2_0_0_0, all_parameters_lat_k3_0_0_0, all_parameters_lat_k4_0_0_0);  einsum_28 = einsum_27 = all_constants_steer_center_0 = all_constants_vx_center_0 = einsum_26 = all_constants_ax_center_0 = all_parameters_lat_k1_0_0_0 = all_parameters_lat_k2_0_0_0 = all_parameters_lat_k3_0_0_0 = all_parameters_lat_k4_0_0_0 = None
        mul_141 = understeer_corr_local_44 * mul_140;  understeer_corr_local_44 = mul_140 = None
        add_23 = mul_141 + mul_138;  mul_141 = mul_138 = None
        add_24 = add_23 + mul_135;  add_23 = mul_135 = None
        add_25 = add_24 + mul_132;  add_24 = mul_132 = None
        add_26 = add_25 + mul_129;  add_25 = mul_129 = None
        add_27 = add_26 + mul_126;  add_26 = mul_126 = None
        add_28 = add_27 + mul_123;  add_27 = mul_123 = None
        add_29 = add_28 + mul_120;  add_28 = mul_120 = None
        add_30 = add_29 + mul_117;  add_29 = mul_117 = None
        add_31 = add_30 + mul_114;  add_30 = mul_114 = None
        add_32 = add_31 + mul_111;  add_31 = mul_111 = None
        add_33 = add_32 + mul_108;  add_32 = mul_108 = None
        add_34 = add_33 + mul_105;  add_33 = mul_105 = None
        add_35 = add_34 + mul_102;  add_34 = mul_102 = None
        add_36 = add_35 + mul_99;  add_35 = mul_99 = None
        add_37 = add_36 + mul_96;  add_36 = mul_96 = None
        add_38 = add_37 + mul_93;  add_37 = mul_93 = None
        add_39 = add_38 + mul_90;  add_38 = mul_90 = None
        add_40 = add_39 + mul_87;  add_39 = mul_87 = None
        add_41 = add_40 + mul_84;  add_40 = mul_84 = None
        add_42 = add_41 + mul_81;  add_41 = mul_81 = None
        add_43 = add_42 + mul_78;  add_42 = mul_78 = None
        add_44 = add_43 + mul_75;  add_43 = mul_75 = None
        add_45 = add_44 + mul_72;  add_44 = mul_72 = None
        add_46 = add_45 + mul_69;  add_45 = mul_69 = None
        add_47 = add_46 + mul_66;  add_46 = mul_66 = None
        add_48 = add_47 + mul_63;  add_47 = mul_63 = None
        add_49 = add_48 + mul_60;  add_48 = mul_60 = None
        add_50 = add_49 + mul_57;  add_49 = mul_57 = None
        add_51 = add_50 + mul_54;  add_50 = mul_54 = None
        add_52 = add_51 + mul_51;  add_51 = mul_51 = None
        add_53 = add_52 + mul_48;  add_52 = mul_48 = None
        add_54 = add_53 + mul_45;  add_53 = mul_45 = None
        add_55 = add_54 + mul_42;  add_54 = mul_42 = None
        add_56 = add_55 + mul_39;  add_55 = mul_39 = None
        add_57 = add_56 + mul_36;  add_56 = mul_36 = None
        add_58 = add_57 + mul_33;  add_57 = mul_33 = None
        add_59 = add_58 + mul_30;  add_58 = mul_30 = None
        add_60 = add_59 + mul_27;  add_59 = mul_27 = None
        add_61 = add_60 + mul_24;  add_60 = mul_24 = None
        add_62 = add_61 + mul_21;  add_61 = mul_21 = None
        add_63 = add_62 + mul_18;  add_62 = mul_18 = None
        add_64 = add_63 + mul_15;  add_63 = mul_15 = None
        add_65 = add_64 + mul_12;  add_64 = mul_12 = None
        add_66 = add_65 + mul_9;  add_65 = mul_9 = None
        mul_142 = add_66 * mul_6;  mul_6 = None
        size_8 = mul_142.size(0)
        relation_forward_fir2414_weights = self.all_parameters.PFir653W
        size_9 = relation_forward_fir2414_weights.size(1)
        squeeze_4 = mul_142.squeeze(-1);  mul_142 = None
        matmul_4 = torch.matmul(squeeze_4, relation_forward_fir2414_weights);  squeeze_4 = relation_forward_fir2414_weights = None
        to_4 = matmul_4.to(dtype = torch.float32);  matmul_4 = None
        view_4 = to_4.view(size_8, 1, size_9);  to_4 = size_8 = size_9 = None
        relation_forward_select2406_w = self.all_constants.Select2406
        einsum_161 = torch.functional.einsum('ijk,k->ij', repeat_2, relation_forward_select2406_w);  relation_forward_select2406_w = None
        unsqueeze_143 = einsum_161.unsqueeze(2);  einsum_161 = None
        relation_forward_select2405_w = self.all_constants.Select2405
        einsum_162 = torch.functional.einsum('ijk,k->ij', repeat_3, relation_forward_select2405_w);  relation_forward_select2405_w = None
        unsqueeze_144 = einsum_162.unsqueeze(2);  einsum_162 = None
        mul_143 = unsqueeze_144 * unsqueeze_143;  unsqueeze_144 = unsqueeze_143 = None
        mul_144 = add_66 * mul_143;  mul_143 = None
        size_10 = mul_144.size(0)
        relation_forward_fir2409_weights = self.all_parameters.PFir651W
        size_11 = relation_forward_fir2409_weights.size(1)
        squeeze_5 = mul_144.squeeze(-1);  mul_144 = None
        matmul_5 = torch.matmul(squeeze_5, relation_forward_fir2409_weights);  squeeze_5 = relation_forward_fir2409_weights = None
        to_5 = matmul_5.to(dtype = torch.float32);  matmul_5 = None
        view_5 = to_5.view(size_10, 1, size_11);  to_5 = size_10 = size_11 = None
        relation_forward_select2401_w = self.all_constants.Select2401
        einsum_163 = torch.functional.einsum('ijk,k->ij', repeat_2, relation_forward_select2401_w);  relation_forward_select2401_w = None
        unsqueeze_145 = einsum_163.unsqueeze(2);  einsum_163 = None
        relation_forward_select2400_w = self.all_constants.Select2400
        einsum_164 = torch.functional.einsum('ijk,k->ij', repeat_3, relation_forward_select2400_w);  relation_forward_select2400_w = None
        unsqueeze_146 = einsum_164.unsqueeze(2);  einsum_164 = None
        mul_145 = unsqueeze_146 * unsqueeze_145;  unsqueeze_146 = unsqueeze_145 = None
        mul_146 = add_66 * mul_145;  mul_145 = None
        size_12 = mul_146.size(0)
        relation_forward_fir2404_weights = self.all_parameters.PFir649W
        size_13 = relation_forward_fir2404_weights.size(1)
        squeeze_6 = mul_146.squeeze(-1);  mul_146 = None
        matmul_6 = torch.matmul(squeeze_6, relation_forward_fir2404_weights);  squeeze_6 = relation_forward_fir2404_weights = None
        to_6 = matmul_6.to(dtype = torch.float32);  matmul_6 = None
        view_6 = to_6.view(size_12, 1, size_13);  to_6 = size_12 = size_13 = None
        relation_forward_select2396_w = self.all_constants.Select2396
        einsum_165 = torch.functional.einsum('ijk,k->ij', repeat_2, relation_forward_select2396_w);  relation_forward_select2396_w = None
        unsqueeze_147 = einsum_165.unsqueeze(2);  einsum_165 = None
        relation_forward_select2395_w = self.all_constants.Select2395
        einsum_166 = torch.functional.einsum('ijk,k->ij', repeat_3, relation_forward_select2395_w);  relation_forward_select2395_w = None
        unsqueeze_148 = einsum_166.unsqueeze(2);  einsum_166 = None
        mul_147 = unsqueeze_148 * unsqueeze_147;  unsqueeze_148 = unsqueeze_147 = None
        mul_148 = add_66 * mul_147;  mul_147 = None
        size_14 = mul_148.size(0)
        relation_forward_fir2399_weights = self.all_parameters.PFir647W
        size_15 = relation_forward_fir2399_weights.size(1)
        squeeze_7 = mul_148.squeeze(-1);  mul_148 = None
        matmul_7 = torch.matmul(squeeze_7, relation_forward_fir2399_weights);  squeeze_7 = relation_forward_fir2399_weights = None
        to_7 = matmul_7.to(dtype = torch.float32);  matmul_7 = None
        view_7 = to_7.view(size_14, 1, size_15);  to_7 = size_14 = size_15 = None
        relation_forward_select2391_w = self.all_constants.Select2391
        einsum_167 = torch.functional.einsum('ijk,k->ij', repeat_2, relation_forward_select2391_w);  relation_forward_select2391_w = None
        unsqueeze_149 = einsum_167.unsqueeze(2);  einsum_167 = None
        relation_forward_select2390_w = self.all_constants.Select2390
        einsum_168 = torch.functional.einsum('ijk,k->ij', repeat_3, relation_forward_select2390_w);  relation_forward_select2390_w = None
        unsqueeze_150 = einsum_168.unsqueeze(2);  einsum_168 = None
        mul_149 = unsqueeze_150 * unsqueeze_149;  unsqueeze_150 = unsqueeze_149 = None
        mul_150 = add_66 * mul_149;  mul_149 = None
        size_16 = mul_150.size(0)
        relation_forward_fir2394_weights = self.all_parameters.PFir645W
        size_17 = relation_forward_fir2394_weights.size(1)
        squeeze_8 = mul_150.squeeze(-1);  mul_150 = None
        matmul_8 = torch.matmul(squeeze_8, relation_forward_fir2394_weights);  squeeze_8 = relation_forward_fir2394_weights = None
        to_8 = matmul_8.to(dtype = torch.float32);  matmul_8 = None
        view_8 = to_8.view(size_16, 1, size_17);  to_8 = size_16 = size_17 = None
        relation_forward_select2386_w = self.all_constants.Select2386
        einsum_169 = torch.functional.einsum('ijk,k->ij', repeat_2, relation_forward_select2386_w);  relation_forward_select2386_w = None
        unsqueeze_151 = einsum_169.unsqueeze(2);  einsum_169 = None
        relation_forward_select2385_w = self.all_constants.Select2385
        einsum_170 = torch.functional.einsum('ijk,k->ij', repeat_3, relation_forward_select2385_w);  relation_forward_select2385_w = None
        unsqueeze_152 = einsum_170.unsqueeze(2);  einsum_170 = None
        mul_151 = unsqueeze_152 * unsqueeze_151;  unsqueeze_152 = unsqueeze_151 = None
        mul_152 = add_66 * mul_151;  mul_151 = None
        size_18 = mul_152.size(0)
        relation_forward_fir2389_weights = self.all_parameters.PFir643W
        size_19 = relation_forward_fir2389_weights.size(1)
        squeeze_9 = mul_152.squeeze(-1);  mul_152 = None
        matmul_9 = torch.matmul(squeeze_9, relation_forward_fir2389_weights);  squeeze_9 = relation_forward_fir2389_weights = None
        to_9 = matmul_9.to(dtype = torch.float32);  matmul_9 = None
        view_9 = to_9.view(size_18, 1, size_19);  to_9 = size_18 = size_19 = None
        relation_forward_select2381_w = self.all_constants.Select2381
        einsum_171 = torch.functional.einsum('ijk,k->ij', repeat_2, relation_forward_select2381_w);  relation_forward_select2381_w = None
        unsqueeze_153 = einsum_171.unsqueeze(2);  einsum_171 = None
        relation_forward_select2380_w = self.all_constants.Select2380
        einsum_172 = torch.functional.einsum('ijk,k->ij', repeat_3, relation_forward_select2380_w);  relation_forward_select2380_w = None
        unsqueeze_154 = einsum_172.unsqueeze(2);  einsum_172 = None
        mul_153 = unsqueeze_154 * unsqueeze_153;  unsqueeze_154 = unsqueeze_153 = None
        mul_154 = add_66 * mul_153;  mul_153 = None
        size_20 = mul_154.size(0)
        relation_forward_fir2384_weights = self.all_parameters.PFir641W
        size_21 = relation_forward_fir2384_weights.size(1)
        squeeze_10 = mul_154.squeeze(-1);  mul_154 = None
        matmul_10 = torch.matmul(squeeze_10, relation_forward_fir2384_weights);  squeeze_10 = relation_forward_fir2384_weights = None
        to_10 = matmul_10.to(dtype = torch.float32);  matmul_10 = None
        view_10 = to_10.view(size_20, 1, size_21);  to_10 = size_20 = size_21 = None
        relation_forward_select2376_w = self.all_constants.Select2376
        einsum_173 = torch.functional.einsum('ijk,k->ij', repeat_2, relation_forward_select2376_w);  relation_forward_select2376_w = None
        unsqueeze_155 = einsum_173.unsqueeze(2);  einsum_173 = None
        relation_forward_select2375_w = self.all_constants.Select2375
        einsum_174 = torch.functional.einsum('ijk,k->ij', repeat_3, relation_forward_select2375_w);  relation_forward_select2375_w = None
        unsqueeze_156 = einsum_174.unsqueeze(2);  einsum_174 = None
        mul_155 = unsqueeze_156 * unsqueeze_155;  unsqueeze_156 = unsqueeze_155 = None
        mul_156 = add_66 * mul_155;  mul_155 = None
        size_22 = mul_156.size(0)
        relation_forward_fir2379_weights = self.all_parameters.PFir639W
        size_23 = relation_forward_fir2379_weights.size(1)
        squeeze_11 = mul_156.squeeze(-1);  mul_156 = None
        matmul_11 = torch.matmul(squeeze_11, relation_forward_fir2379_weights);  squeeze_11 = relation_forward_fir2379_weights = None
        to_11 = matmul_11.to(dtype = torch.float32);  matmul_11 = None
        view_11 = to_11.view(size_22, 1, size_23);  to_11 = size_22 = size_23 = None
        relation_forward_select2371_w = self.all_constants.Select2371
        einsum_175 = torch.functional.einsum('ijk,k->ij', repeat_2, relation_forward_select2371_w);  repeat_2 = relation_forward_select2371_w = None
        unsqueeze_157 = einsum_175.unsqueeze(2);  einsum_175 = None
        relation_forward_select2370_w = self.all_constants.Select2370
        einsum_176 = torch.functional.einsum('ijk,k->ij', repeat_3, relation_forward_select2370_w);  repeat_3 = relation_forward_select2370_w = None
        unsqueeze_158 = einsum_176.unsqueeze(2);  einsum_176 = None
        mul_157 = unsqueeze_158 * unsqueeze_157;  unsqueeze_158 = unsqueeze_157 = None
        mul_158 = add_66 * mul_157;  add_66 = mul_157 = None
        size_24 = mul_158.size(0)
        relation_forward_fir2374_weights = self.all_parameters.PFir637W
        size_25 = relation_forward_fir2374_weights.size(1)
        squeeze_12 = mul_158.squeeze(-1);  mul_158 = None
        matmul_12 = torch.matmul(squeeze_12, relation_forward_fir2374_weights);  squeeze_12 = relation_forward_fir2374_weights = None
        to_12 = matmul_12.to(dtype = torch.float32);  matmul_12 = None
        view_12 = to_12.view(size_24, 1, size_25);  to_12 = size_24 = size_25 = None
        add_67 = view_12 + view_11;  view_12 = view_11 = None
        add_68 = add_67 + view_10;  add_67 = view_10 = None
        add_69 = add_68 + view_9;  add_68 = view_9 = None
        add_70 = add_69 + view_8;  add_69 = view_8 = None
        add_71 = add_70 + view_7;  add_70 = view_7 = None
        add_72 = add_71 + view_6;  add_71 = view_6 = None
        add_73 = add_72 + view_5;  add_72 = view_5 = None
        add_74 = add_73 + view_4;  add_73 = view_4 = None
        getitem_16 = yaw_rate
        relation_forward_sample_part2433_w = self.all_constants.SamplePart2433
        einsum_177 = torch.functional.einsum('bij,ki->bkj', getitem_16, relation_forward_sample_part2433_w);  getitem_16 = relation_forward_sample_part2433_w = None
        getitem_17 = vel;  kwargs = None
        relation_forward_sample_part2424_w = self.all_constants.SamplePart2424
        einsum_178 = torch.functional.einsum('bij,ki->bkj', getitem_17, relation_forward_sample_part2424_w);  getitem_17 = relation_forward_sample_part2424_w = None
        mul_159 = add_74 * einsum_178;  einsum_178 = None
        outputs = ({'acceleration': acc_model_based, 'yaw_rate_': add_74, 'accy_computed': mul_159}, {'SamplePart2433': einsum_177, 'SamplePart2435': einsum_15, 'Add2422': add_74, 'ParamFun2431': acc_model_based}, {}, {})
        return (outputs[0]['acceleration'],outputs[0]['yaw_rate_'],outputs[0]['accy_computed'],), (outputs[1]['SamplePart2433'], outputs[1]['SamplePart2435'], outputs[1]['Add2422'], outputs[1]['ParamFun2431'], ), (), ()
