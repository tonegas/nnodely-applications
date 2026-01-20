import torch

def nnodely_basic_model_update_state(data_in, rel):
    data_out = data_in.clone()
    max_dim = min(rel.size(1), data_in.size(1))
    data_out[:, -max_dim:, :] = rel[:, -max_dim:, :]
    return data_out

def nnodely_basic_model_timeshift(data_in):
    return torch.cat((data_in[:, 1:, :], data_in[:, :1, :]), dim=1)

class TracerModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.all_parameters = {}
        self.all_constants = {}
        self.all_constants["Constant125"] = torch.tensor([2.0], requires_grad=False)
        self.all_constants["Constant126"] = torch.tensor([2.0], requires_grad=False)
        self.all_constants["Constant127"] = torch.tensor([2.0], requires_grad=False)
        self.all_constants["Constant128"] = torch.tensor([2.0], requires_grad=False)
        self.all_constants["Constant129"] = torch.tensor([2.0], requires_grad=False)
        self.all_constants["Constant130"] = torch.tensor([2.0], requires_grad=False)
        self.all_constants["Constant131"] = torch.tensor([2.0], requires_grad=False)
        self.all_constants["Constant132"] = torch.tensor([2.0], requires_grad=False)
        self.all_constants["Constant134"] = torch.tensor([2.0], requires_grad=False)
        self.all_constants["Constant135"] = torch.tensor([2.0], requires_grad=False)
        self.all_constants["Constant136"] = torch.tensor([2.0], requires_grad=False)
        self.all_constants["Constant137"] = torch.tensor([2.0], requires_grad=False)
        self.all_constants["Constant138"] = torch.tensor([2.0], requires_grad=False)
        self.all_constants["Constant139"] = torch.tensor([2.0], requires_grad=False)
        self.all_constants["Constant140"] = torch.tensor([2.0], requires_grad=False)
        self.all_constants["Constant141"] = torch.tensor([2.0], requires_grad=False)
        self.all_constants["Constant142"] = torch.tensor([2.0], requires_grad=False)
        self.all_constants["Constant143"] = torch.tensor([2.0], requires_grad=False)
        self.all_constants["Constant144"] = torch.tensor([2.0], requires_grad=False)
        self.all_constants["Constant146"] = torch.tensor([2.0], requires_grad=False)
        self.all_constants["Constant147"] = torch.tensor([2.0], requires_grad=False)
        self.all_constants["Constant148"] = torch.tensor([2.0], requires_grad=False)
        self.all_constants["Constant149"] = torch.tensor([2.0], requires_grad=False)
        self.all_constants["Constant150"] = torch.tensor([2.0], requires_grad=False)
        self.all_constants["Constant151"] = torch.tensor([2.0], requires_grad=False)
        self.all_constants["Constant152"] = torch.tensor([2.0], requires_grad=False)
        self.all_constants["Constant153"] = torch.tensor([2.0], requires_grad=False)
        self.all_constants["Constant154"] = torch.tensor([2.0], requires_grad=False)
        self.all_constants["Constant155"] = torch.tensor([2.0], requires_grad=False)
        self.all_constants["Constant156"] = torch.tensor([2.0], requires_grad=False)
        self.all_constants["Constant157"] = torch.tensor([2.0], requires_grad=False)
        self.all_constants["Constant158"] = torch.tensor([2.0], requires_grad=False)
        self.all_constants["Constant159"] = torch.tensor([2.0], requires_grad=False)
        self.all_constants["Constant160"] = torch.tensor([2.0], requires_grad=False)
        self.all_constants["Constant161"] = torch.tensor([2.0], requires_grad=False)
        self.all_constants["Constant162"] = torch.tensor([2.0], requires_grad=False)
        self.all_constants["Constant163"] = torch.tensor([2.0], requires_grad=False)
        self.all_constants["Constant164"] = torch.tensor([2.0], requires_grad=False)
        self.all_constants["Constant165"] = torch.tensor([2.0], requires_grad=False)
        self.all_constants["Constant166"] = torch.tensor([2.0], requires_grad=False)
        self.all_constants["Constant167"] = torch.tensor([6.0], requires_grad=False)
        self.all_constants["Constant168"] = torch.tensor([2.0], requires_grad=False)
        self.all_constants["Constant169"] = torch.tensor([2.0], requires_grad=False)
        self.all_constants["Constant170"] = torch.tensor([6.0], requires_grad=False)
        self.all_constants["Constant171"] = torch.tensor([2.0], requires_grad=False)
        self.all_constants["Constant172"] = torch.tensor([2.0], requires_grad=False)
        self.all_constants["Constant173"] = torch.tensor([6.0], requires_grad=False)
        self.all_constants["Constant174"] = torch.tensor([2.0], requires_grad=False)
        self.all_constants["Constant175"] = torch.tensor([2.0], requires_grad=False)
        self.all_constants["Constant176"] = torch.tensor([6.0], requires_grad=False)
        self.all_constants["dt"] = torch.tensor([0.019999999552965164], requires_grad=False)
        self.all_constants["g"] = torch.tensor([9.8100004196167], requires_grad=False)
        self.all_constants["gear"] = torch.tensor([100.0], requires_grad=False)
        self.all_constants["sigma_omega"] = torch.tensor([0.009999999776482582], requires_grad=False)
        self.all_constants["sigma_theta"] = torch.tensor([0.009999999776482582], requires_grad=False)
        self.all_constants["sigma_v"] = torch.tensor([0.009999999776482582], requires_grad=False)
        self.all_constants["sigma_x"] = torch.tensor([0.009999999776482582], requires_grad=False)
        self.all_parameters["I"] = torch.nn.Parameter(torch.tensor([0.3615199327468872]), requires_grad=True)
        self.all_parameters["b"] = torch.nn.Parameter(torch.tensor([0.8465597629547119]), requires_grad=True)
        self.all_parameters["d"] = torch.nn.Parameter(torch.tensor([1.0317860841751099]), requires_grad=True)
        self.all_parameters["l"] = torch.nn.Parameter(torch.tensor([0.1845705211162567]), requires_grad=True)
        self.all_parameters["m1"] = torch.nn.Parameter(torch.tensor([7.328498840332031]), requires_grad=True)
        self.all_parameters["m2"] = torch.nn.Parameter(torch.tensor([8.189186096191406]), requires_grad=True)
        self.all_parameters["tv1"] = torch.nn.Parameter(torch.tensor([[-0.03913747891783714], [-0.0462024100124836], [-0.04642180725932121], [-0.027845464646816254], [0.001236807438544929], [0.051898300647735596], [0.1236712858080864], [0.21295741200447083], [0.3219408094882965], [0.44920867681503296]]), requires_grad=True)
        self.all_parameters["tv2"] = torch.nn.Parameter(torch.tensor([[6.0792299336753786e-05], [-0.00013160287926439196], [-0.0010753690730780363], [0.0011288989335298538], [-0.0020239660516381264], [0.0011806250549852848], [-0.001147997914813459], [-0.0005478200037032366], [0.033323850482702255], [0.9687705039978027]]), requires_grad=True)
        self.all_parameters["tv3"] = torch.nn.Parameter(torch.tensor([[0.00021501831361092627], [-0.0011925705475732684], [-0.00041650698403827846], [0.0002308514667674899], [0.0008214678964577615], [-0.0008117089746519923], [0.002591873984783888], [0.0010583599796518683], [0.006278018932789564], [0.9917680025100708]]), requires_grad=True)
        self.all_parameters["tv4"] = torch.nn.Parameter(torch.tensor([[0.19513440132141113], [-0.010390308685600758], [-0.2141777127981186], [-0.10811326652765274], [0.01837822236120701], [-0.27192237973213196], [0.18312418460845947], [0.23370878398418427], [0.4989215135574341], [0.47649314999580383]]), requires_grad=True)
        self.all_constants["SamplePart1113"] = torch.tensor([[1.0]], requires_grad=True)
        self.all_constants["SamplePart1115"] = torch.tensor([[1.0]], requires_grad=True)
        self.all_constants["SamplePart1117"] = torch.tensor([[1.0]], requires_grad=True)
        self.all_constants["SamplePart1119"] = torch.tensor([[1.0]], requires_grad=True)
        self.all_constants["SamplePart561"] = torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]], requires_grad=True)
        self.all_constants["SamplePart563"] = torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]], requires_grad=True)
        self.all_constants["SamplePart569"] = torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]], requires_grad=True)
        self.all_constants["SamplePart571"] = torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]], requires_grad=True)
        self.all_constants["SamplePart577"] = torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]], requires_grad=True)
        self.all_constants["SamplePart579"] = torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]], requires_grad=True)
        self.all_constants["SamplePart585"] = torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]], requires_grad=True)
        self.all_constants["SamplePart587"] = torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]], requires_grad=True)
        self.all_constants["SamplePart593"] = torch.tensor([[1.0]], requires_grad=True)
        self.all_constants["SamplePart718"] = torch.tensor([[1.0]], requires_grad=True)
        self.all_constants["SamplePart843"] = torch.tensor([[1.0]], requires_grad=True)
        self.all_constants["SamplePart960"] = torch.tensor([[1.0]], requires_grad=True)
        self.all_parameters = torch.nn.ParameterDict(self.all_parameters)
        self.all_constants = torch.nn.ParameterDict(self.all_constants)

    def update(self, closed_loop={}, connect={}, disconnect=False):
        pass
    
    def forward(self, kwargs):
        all_parameters_m1 = self.all_parameters.m1
        all_parameters_m2 = self.all_parameters.m2
        add = all_parameters_m1 + all_parameters_m2
        add_1 = all_parameters_m1 + all_parameters_m2
        add_2 = all_parameters_m1 + all_parameters_m2
        add_3 = all_parameters_m1 + all_parameters_m2
        add_4 = all_parameters_m1 + all_parameters_m2
        add_5 = all_parameters_m1 + all_parameters_m2
        add_6 = all_parameters_m1 + all_parameters_m2
        add_7 = all_parameters_m1 + all_parameters_m2
        add_8 = all_parameters_m1 + all_parameters_m2
        add_9 = all_parameters_m1 + all_parameters_m2
        add_10 = all_parameters_m1 + all_parameters_m2
        add_11 = all_parameters_m1 + all_parameters_m2;  all_parameters_m1 = None
        mul = add * all_parameters_m2;  add = None
        all_constants_g = self.all_constants.g
        mul_1 = mul * all_constants_g;  mul = None
        all_parameters_l = self.all_parameters.l
        mul_2 = mul_1 * all_parameters_l;  mul_1 = None
        mul_3 = all_parameters_m2 * all_parameters_l
        all_parameters_b = self.all_parameters.b
        mul_4 = mul_3 * all_parameters_b;  mul_3 = None
        mul_5 = all_parameters_m2 * all_parameters_l
        mul_6 = all_parameters_m2 * all_parameters_l
        all_parameters_d = self.all_parameters.d
        mul_7 = mul_6 * all_parameters_d;  mul_6 = None
        mul_8 = all_parameters_m2 * all_parameters_l
        mul_9 = add_2 * all_parameters_d;  add_2 = None
        mul_10 = add_3 * all_parameters_m2;  add_3 = None
        mul_11 = mul_10 * all_constants_g;  mul_10 = None
        mul_12 = mul_11 * all_parameters_l;  mul_11 = None
        mul_13 = all_parameters_m2 * all_parameters_l
        mul_14 = mul_13 * all_parameters_b;  mul_13 = None
        mul_15 = all_parameters_m2 * all_parameters_l
        mul_16 = all_parameters_m2 * all_parameters_l
        mul_17 = mul_16 * all_parameters_d;  mul_16 = None
        mul_18 = all_parameters_m2 * all_parameters_l
        mul_19 = add_5 * all_parameters_d;  add_5 = None
        mul_20 = add_6 * all_parameters_m2;  add_6 = None
        mul_21 = mul_20 * all_constants_g;  mul_20 = None
        mul_22 = mul_21 * all_parameters_l;  mul_21 = None
        mul_23 = all_parameters_m2 * all_parameters_l
        mul_24 = mul_23 * all_parameters_b;  mul_23 = None
        mul_25 = all_parameters_m2 * all_parameters_l
        mul_26 = all_parameters_m2 * all_parameters_l
        mul_27 = mul_26 * all_parameters_d;  mul_26 = None
        mul_28 = all_parameters_m2 * all_parameters_l
        mul_29 = add_8 * all_parameters_d;  add_8 = None
        mul_30 = add_9 * all_parameters_m2;  add_9 = None
        mul_31 = mul_30 * all_constants_g;  mul_30 = None
        mul_32 = mul_31 * all_parameters_l;  mul_31 = None
        mul_33 = all_parameters_m2 * all_parameters_l
        mul_34 = mul_33 * all_parameters_b;  mul_33 = None
        mul_35 = all_parameters_m2 * all_parameters_l
        mul_36 = all_parameters_m2 * all_parameters_l
        mul_37 = mul_36 * all_parameters_d;  mul_36 = None
        mul_38 = all_parameters_m2 * all_parameters_l
        mul_39 = add_11 * all_parameters_d;  add_11 = all_parameters_d = None
        all_constants_constant163 = self.all_constants.Constant163
        pow_1 = torch.pow(all_parameters_m2, all_constants_constant163);  all_constants_constant163 = None
        all_constants_constant164 = self.all_constants.Constant164
        pow_2 = torch.pow(all_parameters_l, all_constants_constant164);  all_constants_constant164 = None
        all_constants_constant125 = self.all_constants.Constant125
        pow_3 = torch.pow(all_parameters_l, all_constants_constant125);  all_constants_constant125 = None
        all_constants_constant127 = self.all_constants.Constant127
        pow_4 = torch.pow(all_parameters_m2, all_constants_constant127);  all_constants_constant127 = None
        all_constants_constant128 = self.all_constants.Constant128
        pow_5 = torch.pow(all_parameters_l, all_constants_constant128);  all_constants_constant128 = None
        all_constants_constant131 = self.all_constants.Constant131
        pow_6 = torch.pow(all_parameters_m2, all_constants_constant131);  all_constants_constant131 = None
        all_constants_constant132 = self.all_constants.Constant132
        pow_7 = torch.pow(all_parameters_l, all_constants_constant132);  all_constants_constant132 = None
        all_constants_constant137 = self.all_constants.Constant137
        pow_8 = torch.pow(all_parameters_l, all_constants_constant137);  all_constants_constant137 = None
        all_constants_constant139 = self.all_constants.Constant139
        pow_9 = torch.pow(all_parameters_m2, all_constants_constant139);  all_constants_constant139 = None
        all_constants_constant140 = self.all_constants.Constant140
        pow_10 = torch.pow(all_parameters_l, all_constants_constant140);  all_constants_constant140 = None
        all_constants_constant143 = self.all_constants.Constant143
        pow_11 = torch.pow(all_parameters_m2, all_constants_constant143);  all_constants_constant143 = None
        all_constants_constant144 = self.all_constants.Constant144
        pow_12 = torch.pow(all_parameters_l, all_constants_constant144);  all_constants_constant144 = None
        all_constants_constant149 = self.all_constants.Constant149
        pow_13 = torch.pow(all_parameters_l, all_constants_constant149);  all_constants_constant149 = None
        all_constants_constant151 = self.all_constants.Constant151
        pow_14 = torch.pow(all_parameters_m2, all_constants_constant151);  all_constants_constant151 = None
        all_constants_constant152 = self.all_constants.Constant152
        pow_15 = torch.pow(all_parameters_l, all_constants_constant152);  all_constants_constant152 = None
        all_constants_constant155 = self.all_constants.Constant155
        pow_16 = torch.pow(all_parameters_m2, all_constants_constant155);  all_constants_constant155 = None
        all_constants_constant156 = self.all_constants.Constant156
        pow_17 = torch.pow(all_parameters_l, all_constants_constant156);  all_constants_constant156 = None
        all_constants_constant157 = self.all_constants.Constant157
        pow_18 = torch.pow(all_parameters_l, all_constants_constant157);  all_constants_constant157 = None
        all_constants_constant159 = self.all_constants.Constant159
        pow_19 = torch.pow(all_parameters_m2, all_constants_constant159);  all_constants_constant159 = None
        all_constants_constant160 = self.all_constants.Constant160
        pow_20 = torch.pow(all_parameters_l, all_constants_constant160);  all_constants_constant160 = None
        getitem = kwargs['Yangular_velocity']
        relation_forward_sample_part1113_w = self.all_constants.SamplePart1113
        einsum = torch.functional.einsum('bij,ki->bkj', getitem, relation_forward_sample_part1113_w);  getitem = relation_forward_sample_part1113_w = None
        getitem_1 = kwargs['Yangle']
        relation_forward_sample_part1115_w = self.all_constants.SamplePart1115
        einsum_1 = torch.functional.einsum('bij,ki->bkj', getitem_1, relation_forward_sample_part1115_w);  getitem_1 = relation_forward_sample_part1115_w = None
        getitem_2 = kwargs['Yvelocity']
        relation_forward_sample_part1117_w = self.all_constants.SamplePart1117
        einsum_2 = torch.functional.einsum('bij,ki->bkj', getitem_2, relation_forward_sample_part1117_w);  getitem_2 = relation_forward_sample_part1117_w = None
        getitem_3 = kwargs['Ypos']
        relation_forward_sample_part1119_w = self.all_constants.SamplePart1119
        einsum_3 = torch.functional.einsum('bij,ki->bkj', getitem_3, relation_forward_sample_part1119_w);  getitem_3 = relation_forward_sample_part1119_w = None
        getitem_4 = kwargs['Xpos']
        relation_forward_sample_part561_w = self.all_constants.SamplePart561
        einsum_4 = torch.functional.einsum('bij,ki->bkj', getitem_4, relation_forward_sample_part561_w);  getitem_4 = relation_forward_sample_part561_w = None
        getitem_5 = kwargs['noise2']
        relation_forward_sample_part563_w = self.all_constants.SamplePart563
        einsum_5 = torch.functional.einsum('bij,ki->bkj', getitem_5, relation_forward_sample_part563_w);  getitem_5 = relation_forward_sample_part563_w = None
        getitem_6 = kwargs['Xvelocity']
        relation_forward_sample_part569_w = self.all_constants.SamplePart569
        einsum_6 = torch.functional.einsum('bij,ki->bkj', getitem_6, relation_forward_sample_part569_w);  getitem_6 = relation_forward_sample_part569_w = None
        getitem_7 = kwargs['noise1']
        relation_forward_sample_part571_w = self.all_constants.SamplePart571
        einsum_7 = torch.functional.einsum('bij,ki->bkj', getitem_7, relation_forward_sample_part571_w);  getitem_7 = relation_forward_sample_part571_w = None
        getitem_8 = kwargs['Xangular_velocity']
        relation_forward_sample_part577_w = self.all_constants.SamplePart577
        einsum_8 = torch.functional.einsum('bij,ki->bkj', getitem_8, relation_forward_sample_part577_w);  getitem_8 = relation_forward_sample_part577_w = None
        getitem_9 = kwargs['noise3']
        relation_forward_sample_part579_w = self.all_constants.SamplePart579
        einsum_9 = torch.functional.einsum('bij,ki->bkj', getitem_9, relation_forward_sample_part579_w);  getitem_9 = relation_forward_sample_part579_w = None
        getitem_10 = kwargs['Xangle']
        relation_forward_sample_part585_w = self.all_constants.SamplePart585
        einsum_10 = torch.functional.einsum('bij,ki->bkj', getitem_10, relation_forward_sample_part585_w);  getitem_10 = relation_forward_sample_part585_w = None
        getitem_11 = kwargs['noise4']
        relation_forward_sample_part587_w = self.all_constants.SamplePart587
        einsum_11 = torch.functional.einsum('bij,ki->bkj', getitem_11, relation_forward_sample_part587_w);  getitem_11 = relation_forward_sample_part587_w = None
        getitem_12 = kwargs['action']
        relation_forward_sample_part593_w = self.all_constants.SamplePart593
        einsum_12 = torch.functional.einsum('bij,ki->bkj', getitem_12, relation_forward_sample_part593_w);  getitem_12 = relation_forward_sample_part593_w = None
        getitem_13 = kwargs['action']
        relation_forward_sample_part718_w = self.all_constants.SamplePart718
        einsum_13 = torch.functional.einsum('bij,ki->bkj', getitem_13, relation_forward_sample_part718_w);  getitem_13 = relation_forward_sample_part718_w = None
        getitem_14 = kwargs['action']
        relation_forward_sample_part843_w = self.all_constants.SamplePart843
        einsum_14 = torch.functional.einsum('bij,ki->bkj', getitem_14, relation_forward_sample_part843_w);  getitem_14 = relation_forward_sample_part843_w = None
        getitem_15 = kwargs['action'];  kwargs = None
        relation_forward_sample_part960_w = self.all_constants.SamplePart960
        einsum_15 = torch.functional.einsum('bij,ki->bkj', getitem_15, relation_forward_sample_part960_w);  getitem_15 = relation_forward_sample_part960_w = None
        mul_40 = pow_1 * pow_2;  pow_1 = pow_2 = None
        mul_41 = mul_40 * all_constants_g;  mul_40 = None
        all_constants_sigma_x = self.all_constants.sigma_x
        mul_42 = einsum_5 * all_constants_sigma_x;  einsum_5 = all_constants_sigma_x = None
        all_constants_sigma_v = self.all_constants.sigma_v
        mul_43 = einsum_7 * all_constants_sigma_v;  einsum_7 = all_constants_sigma_v = None
        all_constants_sigma_omega = self.all_constants.sigma_omega
        mul_44 = einsum_9 * all_constants_sigma_omega;  einsum_9 = all_constants_sigma_omega = None
        all_constants_sigma_theta = self.all_constants.sigma_theta
        mul_45 = einsum_11 * all_constants_sigma_theta;  einsum_11 = all_constants_sigma_theta = None
        mul_46 = all_parameters_m2 * pow_3;  pow_3 = None
        all_constants_gear = self.all_constants.gear
        mul_47 = all_constants_gear * einsum_12;  einsum_12 = None
        mul_48 = pow_6 * pow_7;  pow_6 = pow_7 = None
        mul_49 = mul_48 * all_constants_g;  mul_48 = None
        mul_50 = all_parameters_m2 * pow_8;  pow_8 = None
        mul_51 = all_constants_gear * einsum_13;  einsum_13 = None
        mul_52 = pow_11 * pow_12;  pow_11 = pow_12 = None
        mul_53 = mul_52 * all_constants_g;  mul_52 = None
        mul_54 = all_parameters_m2 * pow_13;  pow_13 = None
        mul_55 = all_constants_gear * einsum_14;  einsum_14 = None
        mul_56 = pow_16 * pow_17;  pow_16 = pow_17 = None
        mul_57 = mul_56 * all_constants_g;  mul_56 = all_constants_g = None
        mul_58 = all_parameters_m2 * pow_18;  pow_18 = None
        mul_59 = all_constants_gear * einsum_15;  all_constants_gear = einsum_15 = None
        neg = -pow_4;  pow_4 = None
        neg_1 = -pow_9;  pow_9 = None
        neg_2 = -pow_14;  pow_14 = None
        neg_3 = -pow_19;  pow_19 = None
        add_12 = einsum_4 + mul_42;  einsum_4 = mul_42 = None
        add_13 = einsum_6 + mul_43;  einsum_6 = mul_43 = None
        add_14 = einsum_8 + mul_44;  einsum_8 = mul_44 = None
        add_15 = einsum_10 + mul_45;  einsum_10 = mul_45 = None
        all_parameters_i = self.all_parameters.I
        add_16 = all_parameters_i + mul_46;  mul_46 = None
        add_17 = all_parameters_i + mul_50;  mul_50 = None
        add_18 = all_parameters_i + mul_54;  mul_54 = None
        add_19 = all_parameters_i + mul_58;  all_parameters_i = mul_58 = None
        size = add_12.size(0)
        relation_forward_fir567_weights = self.all_parameters.tv1
        size_1 = relation_forward_fir567_weights.size(1)
        squeeze = add_12.squeeze(-1);  add_12 = None
        matmul = torch.matmul(squeeze, relation_forward_fir567_weights);  squeeze = relation_forward_fir567_weights = None
        to = matmul.to(dtype = torch.float32);  matmul = None
        view = to.view(size, 1, size_1);  to = size = size_1 = None
        size_2 = add_13.size(0)
        relation_forward_fir575_weights = self.all_parameters.tv2
        size_3 = relation_forward_fir575_weights.size(1)
        squeeze_1 = add_13.squeeze(-1);  add_13 = None
        matmul_1 = torch.matmul(squeeze_1, relation_forward_fir575_weights);  squeeze_1 = relation_forward_fir575_weights = None
        to_1 = matmul_1.to(dtype = torch.float32);  matmul_1 = None
        view_1 = to_1.view(size_2, 1, size_3);  to_1 = size_2 = size_3 = None
        size_4 = add_14.size(0)
        relation_forward_fir583_weights = self.all_parameters.tv3
        size_5 = relation_forward_fir583_weights.size(1)
        squeeze_2 = add_14.squeeze(-1);  add_14 = None
        matmul_2 = torch.matmul(squeeze_2, relation_forward_fir583_weights);  squeeze_2 = relation_forward_fir583_weights = None
        to_2 = matmul_2.to(dtype = torch.float32);  matmul_2 = None
        view_2 = to_2.view(size_4, 1, size_5);  to_2 = size_4 = size_5 = None
        size_6 = add_15.size(0)
        relation_forward_fir591_weights = self.all_parameters.tv4
        size_7 = relation_forward_fir591_weights.size(1)
        squeeze_3 = add_15.squeeze(-1);  add_15 = None
        matmul_3 = torch.matmul(squeeze_3, relation_forward_fir591_weights);  squeeze_3 = relation_forward_fir591_weights = None
        to_3 = matmul_3.to(dtype = torch.float32);  matmul_3 = None
        view_3 = to_3.view(size_6, 1, size_7);  to_3 = size_6 = size_7 = None
        mul_60 = add_19 * all_parameters_m2
        mul_61 = mul_60 * all_parameters_l;  mul_60 = None
        mul_62 = mul_59 * add_19
        mul_63 = add_1 * add_16;  add_1 = None
        mul_64 = add_16 * all_parameters_b
        mul_65 = mul_64 * view_1;  mul_64 = None
        mul_66 = mul_9 * view_2;  mul_9 = None
        mul_67 = neg * pow_5;  neg = pow_5 = None
        mul_68 = mul_14 * view_1;  mul_14 = None
        mul_69 = add_16 * all_parameters_m2
        mul_70 = mul_69 * all_parameters_l;  mul_69 = None
        mul_71 = mul_17 * view_2;  mul_17 = None
        mul_72 = mul_47 * add_16;  add_16 = None
        all_constants_dt = self.all_constants.dt
        mul_73 = view_2 * all_constants_dt
        mul_74 = add_4 * add_17;  add_4 = None
        mul_75 = add_17 * all_parameters_b
        mul_76 = neg_1 * pow_10;  neg_1 = pow_10 = None
        mul_77 = add_17 * all_parameters_m2
        mul_78 = mul_77 * all_parameters_l;  mul_77 = None
        mul_79 = mul_51 * add_17;  add_17 = None
        mul_80 = add_7 * add_18;  add_7 = None
        mul_81 = add_18 * all_parameters_b
        mul_82 = neg_2 * pow_15;  neg_2 = pow_15 = None
        mul_83 = add_18 * all_parameters_m2;  all_parameters_m2 = None
        mul_84 = mul_83 * all_parameters_l;  mul_83 = all_parameters_l = None
        mul_85 = mul_55 * add_18;  add_18 = None
        mul_86 = add_10 * add_19;  add_10 = None
        mul_87 = add_19 * all_parameters_b;  add_19 = all_parameters_b = None
        mul_88 = neg_3 * pow_20;  neg_3 = pow_20 = None
        all_constants_constant129 = self.all_constants.Constant129
        pow_21 = torch.pow(view_2, all_constants_constant129);  all_constants_constant129 = None
        all_constants_constant130 = self.all_constants.Constant130
        pow_22 = torch.pow(view_2, all_constants_constant130);  all_constants_constant130 = None
        sin = torch.sin(view_3)
        cos = torch.cos(view_3)
        all_constants_constant135 = self.all_constants.Constant135
        truediv = mul_73 / all_constants_constant135;  mul_73 = all_constants_constant135 = None
        mul_89 = mul_8 * cos;  mul_8 = None
        mul_90 = mul_67 * pow_21;  mul_67 = pow_21 = None
        mul_91 = mul_90 * sin;  mul_90 = None
        mul_92 = mul_91 * cos;  mul_91 = None
        mul_93 = mul_12 * sin;  mul_12 = None
        mul_94 = mul_68 * cos;  mul_68 = None
        mul_95 = mul_15 * cos;  mul_15 = None
        mul_96 = mul_95 * mul_47;  mul_95 = mul_47 = None
        mul_97 = mul_70 * pow_22;  mul_70 = pow_22 = None
        mul_98 = mul_97 * sin;  mul_97 = None
        mul_99 = mul_49 * sin;  mul_49 = sin = None
        mul_100 = mul_99 * cos;  mul_99 = None
        mul_101 = mul_71 * cos;  mul_71 = cos = None
        all_constants_constant126 = self.all_constants.Constant126
        pow_23 = torch.pow(mul_89, all_constants_constant126);  mul_89 = all_constants_constant126 = None
        sub = mul_63 - pow_23;  mul_63 = pow_23 = None
        sub_1 = mul_92 - mul_66;  mul_92 = mul_66 = None
        sub_2 = mul_98 - mul_65;  mul_98 = mul_65 = None
        sub_3 = sub_2 - mul_100;  sub_2 = mul_100 = None
        add_20 = sub_1 + mul_93;  sub_1 = mul_93 = None
        add_21 = add_20 + mul_94;  add_20 = mul_94 = None
        add_22 = sub_3 + mul_101;  sub_3 = mul_101 = None
        add_23 = add_22 + mul_72;  add_22 = mul_72 = None
        add_24 = view_3 + truediv;  truediv = None
        cos_1 = torch.cos(add_24)
        truediv_1 = add_23 / sub;  add_23 = None
        mul_102 = truediv_1 * all_constants_dt
        mul_103 = mul_18 * cos_1;  mul_18 = None
        mul_104 = mul_25 * cos_1;  mul_25 = None
        mul_105 = mul_104 * mul_51;  mul_104 = mul_51 = None
        all_constants_constant138 = self.all_constants.Constant138
        pow_24 = torch.pow(mul_103, all_constants_constant138);  mul_103 = all_constants_constant138 = None
        sin_1 = torch.sin(add_24);  add_24 = None
        sub_4 = add_21 - mul_96;  add_21 = mul_96 = None
        sub_5 = mul_74 - pow_24;  mul_74 = pow_24 = None
        truediv_2 = sub_4 / sub;  sub_4 = sub = None
        all_constants_constant134 = self.all_constants.Constant134
        truediv_3 = mul_102 / all_constants_constant134;  mul_102 = all_constants_constant134 = None
        mul_106 = truediv_2 * all_constants_dt
        mul_107 = mul_22 * sin_1;  mul_22 = None
        mul_108 = mul_53 * sin_1;  mul_53 = None
        mul_109 = mul_108 * cos_1;  mul_108 = None
        add_25 = view_1 + truediv_3;  truediv_3 = None
        all_constants_constant136 = self.all_constants.Constant136
        truediv_4 = mul_106 / all_constants_constant136;  mul_106 = all_constants_constant136 = None
        all_constants_constant168 = self.all_constants.Constant168
        mul_110 = all_constants_constant168 * add_25;  all_constants_constant168 = None
        mul_111 = mul_75 * add_25;  mul_75 = None
        mul_112 = mul_24 * add_25;  mul_24 = add_25 = None
        mul_113 = mul_112 * cos_1;  mul_112 = None
        add_26 = view_1 + mul_110;  mul_110 = None
        add_27 = view_2 + truediv_4;  truediv_4 = None
        all_constants_constant174 = self.all_constants.Constant174
        mul_114 = all_constants_constant174 * add_27;  all_constants_constant174 = None
        mul_115 = mul_19 * add_27;  mul_19 = None
        mul_116 = mul_27 * add_27;  mul_27 = None
        mul_117 = mul_116 * cos_1;  mul_116 = None
        mul_118 = add_27 * all_constants_dt
        all_constants_constant141 = self.all_constants.Constant141
        pow_25 = torch.pow(add_27, all_constants_constant141);  all_constants_constant141 = None
        all_constants_constant142 = self.all_constants.Constant142
        pow_26 = torch.pow(add_27, all_constants_constant142);  add_27 = all_constants_constant142 = None
        add_28 = view_2 + mul_114;  mul_114 = None
        all_constants_constant147 = self.all_constants.Constant147
        truediv_5 = mul_118 / all_constants_constant147;  mul_118 = all_constants_constant147 = None
        mul_119 = mul_76 * pow_25;  mul_76 = pow_25 = None
        mul_120 = mul_119 * sin_1;  mul_119 = None
        mul_121 = mul_120 * cos_1;  mul_120 = cos_1 = None
        mul_122 = mul_78 * pow_26;  mul_78 = pow_26 = None
        mul_123 = mul_122 * sin_1;  mul_122 = sin_1 = None
        sub_6 = mul_121 - mul_115;  mul_121 = mul_115 = None
        sub_7 = mul_123 - mul_111;  mul_123 = mul_111 = None
        sub_8 = sub_7 - mul_109;  sub_7 = mul_109 = None
        add_29 = sub_6 + mul_107;  sub_6 = mul_107 = None
        add_30 = add_29 + mul_113;  add_29 = mul_113 = None
        add_31 = sub_8 + mul_117;  sub_8 = mul_117 = None
        add_32 = add_31 + mul_79;  add_31 = mul_79 = None
        add_33 = view_3 + truediv_5;  truediv_5 = None
        cos_2 = torch.cos(add_33)
        truediv_6 = add_32 / sub_5;  add_32 = None
        all_constants_constant165 = self.all_constants.Constant165
        mul_124 = all_constants_constant165 * truediv_6;  all_constants_constant165 = None
        mul_125 = truediv_6 * all_constants_dt;  truediv_6 = None
        mul_126 = mul_28 * cos_2;  mul_28 = None
        mul_127 = mul_35 * cos_2;  mul_35 = None
        mul_128 = mul_127 * mul_55;  mul_127 = mul_55 = None
        all_constants_constant150 = self.all_constants.Constant150
        pow_27 = torch.pow(mul_126, all_constants_constant150);  mul_126 = all_constants_constant150 = None
        sin_2 = torch.sin(add_33);  add_33 = None
        sub_9 = add_30 - mul_105;  add_30 = mul_105 = None
        sub_10 = mul_80 - pow_27;  mul_80 = pow_27 = None
        add_34 = truediv_1 + mul_124;  truediv_1 = mul_124 = None
        truediv_7 = sub_9 / sub_5;  sub_9 = sub_5 = None
        all_constants_constant146 = self.all_constants.Constant146
        truediv_8 = mul_125 / all_constants_constant146;  mul_125 = all_constants_constant146 = None
        all_constants_constant171 = self.all_constants.Constant171
        mul_129 = all_constants_constant171 * truediv_7;  all_constants_constant171 = None
        mul_130 = truediv_7 * all_constants_dt;  truediv_7 = None
        mul_131 = mul_32 * sin_2;  mul_32 = None
        mul_132 = mul_57 * sin_2;  mul_57 = None
        mul_133 = mul_132 * cos_2;  mul_132 = None
        add_35 = truediv_2 + mul_129;  truediv_2 = mul_129 = None
        add_36 = view_1 + truediv_8;  truediv_8 = None
        all_constants_constant148 = self.all_constants.Constant148
        truediv_9 = mul_130 / all_constants_constant148;  mul_130 = all_constants_constant148 = None
        all_constants_constant169 = self.all_constants.Constant169
        mul_134 = all_constants_constant169 * add_36;  all_constants_constant169 = None
        mul_135 = mul_81 * add_36;  mul_81 = None
        mul_136 = mul_34 * add_36;  mul_34 = add_36 = None
        mul_137 = mul_136 * cos_2;  mul_136 = None
        add_37 = add_26 + mul_134;  add_26 = mul_134 = None
        add_38 = view_2 + truediv_9;  truediv_9 = None
        all_constants_constant175 = self.all_constants.Constant175
        mul_138 = all_constants_constant175 * add_38;  all_constants_constant175 = None
        mul_139 = mul_29 * add_38;  mul_29 = None
        mul_140 = mul_37 * add_38;  mul_37 = None
        mul_141 = mul_140 * cos_2;  mul_140 = None
        mul_142 = add_38 * all_constants_dt
        all_constants_constant153 = self.all_constants.Constant153
        pow_28 = torch.pow(add_38, all_constants_constant153);  all_constants_constant153 = None
        all_constants_constant154 = self.all_constants.Constant154
        pow_29 = torch.pow(add_38, all_constants_constant154);  add_38 = all_constants_constant154 = None
        add_39 = add_28 + mul_138;  add_28 = mul_138 = None
        add_40 = view_3 + mul_142;  mul_142 = None
        cos_3 = torch.cos(add_40)
        mul_143 = mul_5 * cos_3;  mul_5 = None
        mul_144 = mul_143 * mul_59;  mul_143 = mul_59 = None
        mul_145 = mul_82 * pow_28;  mul_82 = pow_28 = None
        mul_146 = mul_145 * sin_2;  mul_145 = None
        mul_147 = mul_146 * cos_2;  mul_146 = cos_2 = None
        mul_148 = mul_84 * pow_29;  mul_84 = pow_29 = None
        mul_149 = mul_148 * sin_2;  mul_148 = sin_2 = None
        mul_150 = mul_38 * cos_3;  mul_38 = None
        all_constants_constant158 = self.all_constants.Constant158
        pow_30 = torch.pow(mul_150, all_constants_constant158);  mul_150 = all_constants_constant158 = None
        sin_3 = torch.sin(add_40);  add_40 = None
        sub_11 = mul_147 - mul_139;  mul_147 = mul_139 = None
        sub_12 = mul_149 - mul_135;  mul_149 = mul_135 = None
        sub_13 = sub_12 - mul_133;  sub_12 = mul_133 = None
        sub_14 = mul_86 - pow_30;  mul_86 = pow_30 = None
        add_41 = sub_11 + mul_131;  sub_11 = mul_131 = None
        add_42 = add_41 + mul_137;  add_41 = mul_137 = None
        add_43 = sub_13 + mul_141;  sub_13 = mul_141 = None
        add_44 = add_43 + mul_85;  add_43 = mul_85 = None
        truediv_10 = add_44 / sub_10;  add_44 = None
        mul_151 = mul_2 * sin_3;  mul_2 = None
        mul_152 = mul_41 * sin_3;  mul_41 = None
        mul_153 = mul_152 * cos_3;  mul_152 = None
        all_constants_constant166 = self.all_constants.Constant166
        mul_154 = all_constants_constant166 * truediv_10;  all_constants_constant166 = None
        mul_155 = truediv_10 * all_constants_dt;  truediv_10 = None
        sub_15 = add_42 - mul_128;  add_42 = mul_128 = None
        add_45 = add_34 + mul_154;  add_34 = mul_154 = None
        add_46 = view_1 + mul_155;  mul_155 = None
        truediv_11 = sub_15 / sub_10;  sub_15 = sub_10 = None
        mul_156 = mul_4 * add_46;  mul_4 = None
        mul_157 = mul_156 * cos_3;  mul_156 = None
        all_constants_constant172 = self.all_constants.Constant172
        mul_158 = all_constants_constant172 * truediv_11;  all_constants_constant172 = None
        mul_159 = truediv_11 * all_constants_dt;  truediv_11 = None
        mul_160 = mul_87 * add_46;  mul_87 = None
        add_47 = add_37 + add_46;  add_37 = add_46 = None
        add_48 = add_35 + mul_158;  add_35 = mul_158 = None
        add_49 = view_2 + mul_159;  mul_159 = None
        mul_161 = mul_7 * add_49;  mul_7 = None
        mul_162 = mul_161 * cos_3;  mul_161 = None
        mul_163 = add_47 * all_constants_dt;  add_47 = None
        mul_164 = mul_39 * add_49;  mul_39 = None
        all_constants_constant161 = self.all_constants.Constant161
        pow_31 = torch.pow(add_49, all_constants_constant161);  all_constants_constant161 = None
        all_constants_constant162 = self.all_constants.Constant162
        pow_32 = torch.pow(add_49, all_constants_constant162);  all_constants_constant162 = None
        add_50 = add_39 + add_49;  add_39 = add_49 = None
        all_constants_constant170 = self.all_constants.Constant170
        truediv_12 = mul_163 / all_constants_constant170;  mul_163 = all_constants_constant170 = None
        mul_165 = mul_88 * pow_31;  mul_88 = pow_31 = None
        mul_166 = mul_165 * sin_3;  mul_165 = None
        mul_167 = mul_166 * cos_3;  mul_166 = cos_3 = None
        mul_168 = mul_61 * pow_32;  mul_61 = pow_32 = None
        mul_169 = mul_168 * sin_3;  mul_168 = sin_3 = None
        mul_170 = add_50 * all_constants_dt;  add_50 = None
        sub_16 = mul_167 - mul_164;  mul_167 = mul_164 = None
        sub_17 = mul_169 - mul_160;  mul_169 = mul_160 = None
        sub_18 = sub_17 - mul_153;  sub_17 = mul_153 = None
        add_51 = sub_16 + mul_151;  sub_16 = mul_151 = None
        add_52 = add_51 + mul_157;  add_51 = mul_157 = None
        add_53 = sub_18 + mul_162;  sub_18 = mul_162 = None
        add_54 = add_53 + mul_62;  add_53 = mul_62 = None
        add_55 = view + truediv_12;  view = truediv_12 = None
        truediv_13 = add_54 / sub_14;  add_54 = None
        all_constants_constant176 = self.all_constants.Constant176
        truediv_14 = mul_170 / all_constants_constant176;  mul_170 = all_constants_constant176 = None
        sub_19 = add_52 - mul_144;  add_52 = mul_144 = None
        add_56 = add_45 + truediv_13;  add_45 = truediv_13 = None
        add_57 = view_3 + truediv_14;  view_3 = truediv_14 = None
        truediv_15 = sub_19 / sub_14;  sub_19 = sub_14 = None
        mul_171 = add_56 * all_constants_dt;  add_56 = None
        add_58 = add_48 + truediv_15;  add_48 = truediv_15 = None
        all_constants_constant167 = self.all_constants.Constant167
        truediv_16 = mul_171 / all_constants_constant167;  mul_171 = all_constants_constant167 = None
        mul_172 = add_58 * all_constants_dt;  add_58 = all_constants_dt = None
        add_59 = view_1 + truediv_16;  view_1 = truediv_16 = None
        all_constants_constant173 = self.all_constants.Constant173
        truediv_17 = mul_172 / all_constants_constant173;  mul_172 = all_constants_constant173 = None
        add_60 = view_2 + truediv_17;  view_2 = truediv_17 = None
        return ({'est_theta': add_57, 'est_thetadot': add_60, 'est_x': add_55, 'est_xdot': add_59}, {'SamplePart1113': einsum, 'SamplePart1115': einsum_1, 'SamplePart1119': einsum_3, 'SamplePart1117': einsum_2, 'Add1099': add_60, 'Add1111': add_57, 'Add1087': add_55, 'Add1075': add_59}, {'Xangle': add_57, 'Xangular_velocity': add_60, 'Xpos': add_55, 'Xvelocity': add_59}, {})
        
class RecurrentModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.Cell = TracerModel()
        self.inputs = ['Yangle', 'Yangular_velocity', 'Ypos', 'Yvelocity', 'action', 'noise1', 'noise2', 'noise3', 'noise4', ]
        self.states = dict()

    def forward(self, kwargs):
        n_samples = min([kwargs[key].size(0) for key in self.inputs])
        self.states['Xangle'] = kwargs['Xangle']
        self.states['Xangular_velocity'] = kwargs['Xangular_velocity']
        self.states['Xpos'] = kwargs['Xpos']
        self.states['Xvelocity'] = kwargs['Xvelocity']
        results = {'est_theta':[], 'est_thetadot':[], 'est_x':[], 'est_xdot':[], }
        X = dict()
        for idx in range(n_samples):
            for key in self.inputs:
                X[key] = kwargs[key][idx]
            for key, value in self.states.items():
                X[key] = value
            out, _, closed_loop, connect = self.Cell(X)
            for key, value in results.items():
                results[key].append(out[key])
            for key, val in closed_loop.items():
                self.states[key] = nnodely_basic_model_timeshift(self.states[key])
                self.states[key] = nnodely_basic_model_update_state(self.states[key], val)
            for key, val in connect.items():
                self.states[key] = nnodely_basic_model_timeshift(val)
        return results
