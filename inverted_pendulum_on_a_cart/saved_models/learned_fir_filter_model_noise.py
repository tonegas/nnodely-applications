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
        self.all_constants["Constant39"] = torch.tensor([2.0], requires_grad=False)
        self.all_constants["Constant40"] = torch.tensor([2.0], requires_grad=False)
        self.all_constants["Constant41"] = torch.tensor([2.0], requires_grad=False)
        self.all_constants["Constant42"] = torch.tensor([2.0], requires_grad=False)
        self.all_constants["Constant43"] = torch.tensor([2.0], requires_grad=False)
        self.all_constants["Constant44"] = torch.tensor([2.0], requires_grad=False)
        self.all_constants["Constant45"] = torch.tensor([2.0], requires_grad=False)
        self.all_constants["Constant46"] = torch.tensor([2.0], requires_grad=False)
        self.all_constants["Constant48"] = torch.tensor([2.0], requires_grad=False)
        self.all_constants["Constant49"] = torch.tensor([2.0], requires_grad=False)
        self.all_constants["Constant50"] = torch.tensor([2.0], requires_grad=False)
        self.all_constants["Constant51"] = torch.tensor([2.0], requires_grad=False)
        self.all_constants["Constant52"] = torch.tensor([2.0], requires_grad=False)
        self.all_constants["Constant53"] = torch.tensor([2.0], requires_grad=False)
        self.all_constants["Constant54"] = torch.tensor([2.0], requires_grad=False)
        self.all_constants["Constant55"] = torch.tensor([2.0], requires_grad=False)
        self.all_constants["Constant56"] = torch.tensor([2.0], requires_grad=False)
        self.all_constants["Constant57"] = torch.tensor([2.0], requires_grad=False)
        self.all_constants["Constant58"] = torch.tensor([2.0], requires_grad=False)
        self.all_constants["Constant60"] = torch.tensor([2.0], requires_grad=False)
        self.all_constants["Constant61"] = torch.tensor([2.0], requires_grad=False)
        self.all_constants["Constant62"] = torch.tensor([2.0], requires_grad=False)
        self.all_constants["Constant63"] = torch.tensor([2.0], requires_grad=False)
        self.all_constants["Constant64"] = torch.tensor([2.0], requires_grad=False)
        self.all_constants["Constant65"] = torch.tensor([2.0], requires_grad=False)
        self.all_constants["Constant66"] = torch.tensor([2.0], requires_grad=False)
        self.all_constants["Constant67"] = torch.tensor([2.0], requires_grad=False)
        self.all_constants["Constant68"] = torch.tensor([2.0], requires_grad=False)
        self.all_constants["Constant69"] = torch.tensor([2.0], requires_grad=False)
        self.all_constants["Constant70"] = torch.tensor([2.0], requires_grad=False)
        self.all_constants["Constant71"] = torch.tensor([2.0], requires_grad=False)
        self.all_constants["Constant72"] = torch.tensor([2.0], requires_grad=False)
        self.all_constants["Constant73"] = torch.tensor([2.0], requires_grad=False)
        self.all_constants["Constant74"] = torch.tensor([2.0], requires_grad=False)
        self.all_constants["Constant75"] = torch.tensor([2.0], requires_grad=False)
        self.all_constants["Constant76"] = torch.tensor([2.0], requires_grad=False)
        self.all_constants["Constant77"] = torch.tensor([2.0], requires_grad=False)
        self.all_constants["Constant78"] = torch.tensor([2.0], requires_grad=False)
        self.all_constants["Constant79"] = torch.tensor([2.0], requires_grad=False)
        self.all_constants["Constant80"] = torch.tensor([2.0], requires_grad=False)
        self.all_constants["Constant81"] = torch.tensor([6.0], requires_grad=False)
        self.all_constants["Constant82"] = torch.tensor([2.0], requires_grad=False)
        self.all_constants["Constant83"] = torch.tensor([2.0], requires_grad=False)
        self.all_constants["Constant84"] = torch.tensor([6.0], requires_grad=False)
        self.all_constants["Constant85"] = torch.tensor([2.0], requires_grad=False)
        self.all_constants["Constant86"] = torch.tensor([2.0], requires_grad=False)
        self.all_constants["Constant87"] = torch.tensor([6.0], requires_grad=False)
        self.all_constants["Constant88"] = torch.tensor([2.0], requires_grad=False)
        self.all_constants["Constant89"] = torch.tensor([2.0], requires_grad=False)
        self.all_constants["Constant90"] = torch.tensor([6.0], requires_grad=False)
        self.all_constants["dt"] = torch.tensor([0.019999999552965164], requires_grad=False)
        self.all_constants["g"] = torch.tensor([9.8100004196167], requires_grad=False)
        self.all_constants["gear"] = torch.tensor([100.0], requires_grad=False)
        self.all_constants["sigma_force"] = torch.tensor([0.05999999865889549], requires_grad=False)
        self.all_constants["sigma_omega"] = torch.tensor([0.10000000149011612], requires_grad=False)
        self.all_constants["sigma_theta"] = torch.tensor([0.009999999776482582], requires_grad=False)
        self.all_constants["sigma_v"] = torch.tensor([0.019999999552965164], requires_grad=False)
        self.all_constants["sigma_x"] = torch.tensor([0.009999999776482582], requires_grad=False)
        self.all_parameters["I"] = torch.nn.Parameter(torch.tensor([0.029196498915553093]), requires_grad=True)
        self.all_parameters["b"] = torch.nn.Parameter(torch.tensor([0.5518590211868286]), requires_grad=True)
        self.all_parameters["d"] = torch.nn.Parameter(torch.tensor([0.5346891283988953]), requires_grad=True)
        self.all_parameters["l"] = torch.nn.Parameter(torch.tensor([0.21498709917068481]), requires_grad=True)
        self.all_parameters["m1"] = torch.nn.Parameter(torch.tensor([6.229735374450684]), requires_grad=True)
        self.all_parameters["m2"] = torch.nn.Parameter(torch.tensor([3.577908515930176]), requires_grad=True)
        self.all_parameters["tv2"] = torch.nn.Parameter(torch.tensor([[-0.01644490845501423], [-0.016674481332302094], [-0.02302756905555725], [-0.021880563348531723], [-0.02532469853758812], [0.019463064149022102], [0.08460117131471634], [0.14615999162197113], [0.2552056908607483], [0.5938453674316406]]), requires_grad=True)
        self.all_parameters["tv3"] = torch.nn.Parameter(torch.tensor([[0.007458774838596582], [-0.011013618670403957], [-0.03344158083200455], [-0.026034535840153694], [-0.0076858410611748695], [0.03843627870082855], [0.10794762521982193], [0.20026795566082], [0.3049878776073456], [0.4230453073978424]]), requires_grad=True)
        self.all_parameters["tv4"] = torch.nn.Parameter(torch.tensor([[0.012205323204398155], [-0.0010631527984514832], [-0.0074620116502046585], [-0.0104419756680727], [-0.007607249077409506], [-0.005147782154381275], [0.008007185533642769], [0.02083374559879303], [0.03271688148379326], [0.9579738974571228]]), requires_grad=True)
        self.all_parameters["tv5"] = torch.nn.Parameter(torch.tensor([[-0.007997744716703892], [0.004633757751435041], [-0.016062814742326736], [-0.02825828641653061], [-0.02721511200070381], [-0.019219424575567245], [0.020411934703588486], [0.09304294735193253], [0.19662244617938995], [0.35025689005851746]]), requires_grad=True)
        self.all_parameters["tv1"] = torch.nn.Parameter(torch.tensor([[-0.03414878249168396], [-0.045093078166246414], [-0.03544943407177925], [-0.017527855932712555], [0.00395536795258522], [0.0578472763299942], [0.10627463459968567], [0.17552345991134644], [0.25452661514282227], [0.5350487232208252]]), requires_grad=True)
        self.all_constants["SamplePart1"] = torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]], requires_grad=True)
        self.all_constants["SamplePart11"] = torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]], requires_grad=True)
        self.all_constants["SamplePart17"] = torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]], requires_grad=True)
        self.all_constants["SamplePart19"] = torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]], requires_grad=True)
        self.all_constants["SamplePart25"] = torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]], requires_grad=True)
        self.all_constants["SamplePart27"] = torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]], requires_grad=True)
        self.all_constants["SamplePart3"] = torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]], requires_grad=True)
        self.all_constants["SamplePart33"] = torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]], requires_grad=True)
        self.all_constants["SamplePart35"] = torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]], requires_grad=True)
        self.all_constants["SamplePart557"] = torch.tensor([[1.0]], requires_grad=True)
        self.all_constants["SamplePart559"] = torch.tensor([[1.0]], requires_grad=True)
        self.all_constants["SamplePart561"] = torch.tensor([[1.0]], requires_grad=True)
        self.all_constants["SamplePart563"] = torch.tensor([[1.0]], requires_grad=True)
        self.all_constants["SamplePart9"] = torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]], requires_grad=True)
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
        all_parameters_l = self.all_parameters.l
        mul = all_parameters_m2 * all_parameters_l
        mul_1 = all_parameters_m2 * all_parameters_l
        all_parameters_d = self.all_parameters.d
        mul_2 = mul_1 * all_parameters_d;  mul_1 = None
        mul_3 = all_parameters_m2 * all_parameters_l
        mul_4 = add_1 * all_parameters_d;  add_1 = None
        mul_5 = add_2 * all_parameters_m2;  add_2 = None
        all_constants_g = self.all_constants.g
        mul_6 = mul_5 * all_constants_g;  mul_5 = None
        mul_7 = mul_6 * all_parameters_l;  mul_6 = None
        mul_8 = all_parameters_m2 * all_parameters_l
        all_parameters_b = self.all_parameters.b
        mul_9 = mul_8 * all_parameters_b;  mul_8 = None
        mul_10 = all_parameters_m2 * all_parameters_l
        mul_11 = all_parameters_m2 * all_parameters_l
        mul_12 = mul_11 * all_parameters_d;  mul_11 = None
        mul_13 = all_parameters_m2 * all_parameters_l
        mul_14 = add_4 * all_parameters_d;  add_4 = None
        mul_15 = add_5 * all_parameters_m2;  add_5 = None
        mul_16 = mul_15 * all_constants_g;  mul_15 = None
        mul_17 = mul_16 * all_parameters_l;  mul_16 = None
        mul_18 = all_parameters_m2 * all_parameters_l
        mul_19 = mul_18 * all_parameters_b;  mul_18 = None
        mul_20 = all_parameters_m2 * all_parameters_l
        mul_21 = all_parameters_m2 * all_parameters_l
        mul_22 = mul_21 * all_parameters_d;  mul_21 = None
        mul_23 = all_parameters_m2 * all_parameters_l
        mul_24 = add_7 * all_parameters_d;  add_7 = None
        mul_25 = add_8 * all_parameters_m2;  add_8 = None
        mul_26 = mul_25 * all_constants_g;  mul_25 = None
        mul_27 = mul_26 * all_parameters_l;  mul_26 = None
        mul_28 = all_parameters_m2 * all_parameters_l
        mul_29 = mul_28 * all_parameters_b;  mul_28 = None
        mul_30 = all_parameters_m2 * all_parameters_l
        mul_31 = all_parameters_m2 * all_parameters_l
        mul_32 = mul_31 * all_parameters_d;  mul_31 = None
        mul_33 = all_parameters_m2 * all_parameters_l
        mul_34 = add_10 * all_parameters_d;  add_10 = all_parameters_d = None
        mul_35 = add_11 * all_parameters_m2;  add_11 = None
        mul_36 = mul_35 * all_constants_g;  mul_35 = None
        mul_37 = mul_36 * all_parameters_l;  mul_36 = None
        mul_38 = all_parameters_m2 * all_parameters_l
        all_constants_constant45 = self.all_constants.Constant45
        pow_1 = torch.pow(all_parameters_m2, all_constants_constant45);  all_constants_constant45 = None
        all_constants_constant46 = self.all_constants.Constant46
        pow_2 = torch.pow(all_parameters_l, all_constants_constant46);  all_constants_constant46 = None
        all_constants_constant51 = self.all_constants.Constant51
        pow_3 = torch.pow(all_parameters_l, all_constants_constant51);  all_constants_constant51 = None
        all_constants_constant53 = self.all_constants.Constant53
        pow_4 = torch.pow(all_parameters_m2, all_constants_constant53);  all_constants_constant53 = None
        all_constants_constant54 = self.all_constants.Constant54
        pow_5 = torch.pow(all_parameters_l, all_constants_constant54);  all_constants_constant54 = None
        all_constants_constant57 = self.all_constants.Constant57
        pow_6 = torch.pow(all_parameters_m2, all_constants_constant57);  all_constants_constant57 = None
        all_constants_constant58 = self.all_constants.Constant58
        pow_7 = torch.pow(all_parameters_l, all_constants_constant58);  all_constants_constant58 = None
        all_constants_constant63 = self.all_constants.Constant63
        pow_8 = torch.pow(all_parameters_l, all_constants_constant63);  all_constants_constant63 = None
        all_constants_constant65 = self.all_constants.Constant65
        pow_9 = torch.pow(all_parameters_m2, all_constants_constant65);  all_constants_constant65 = None
        all_constants_constant66 = self.all_constants.Constant66
        pow_10 = torch.pow(all_parameters_l, all_constants_constant66);  all_constants_constant66 = None
        all_constants_constant69 = self.all_constants.Constant69
        pow_11 = torch.pow(all_parameters_m2, all_constants_constant69);  all_constants_constant69 = None
        all_constants_constant70 = self.all_constants.Constant70
        pow_12 = torch.pow(all_parameters_l, all_constants_constant70);  all_constants_constant70 = None
        all_constants_constant71 = self.all_constants.Constant71
        pow_13 = torch.pow(all_parameters_l, all_constants_constant71);  all_constants_constant71 = None
        all_constants_constant73 = self.all_constants.Constant73
        pow_14 = torch.pow(all_parameters_m2, all_constants_constant73);  all_constants_constant73 = None
        all_constants_constant74 = self.all_constants.Constant74
        pow_15 = torch.pow(all_parameters_l, all_constants_constant74);  all_constants_constant74 = None
        all_constants_constant39 = self.all_constants.Constant39
        pow_16 = torch.pow(all_parameters_l, all_constants_constant39);  all_constants_constant39 = None
        all_constants_constant77 = self.all_constants.Constant77
        pow_17 = torch.pow(all_parameters_m2, all_constants_constant77);  all_constants_constant77 = None
        all_constants_constant78 = self.all_constants.Constant78
        pow_18 = torch.pow(all_parameters_l, all_constants_constant78);  all_constants_constant78 = None
        all_constants_constant41 = self.all_constants.Constant41
        pow_19 = torch.pow(all_parameters_m2, all_constants_constant41);  all_constants_constant41 = None
        all_constants_constant42 = self.all_constants.Constant42
        pow_20 = torch.pow(all_parameters_l, all_constants_constant42);  all_constants_constant42 = None
        getitem = kwargs['Xpos']
        relation_forward_sample_part1_w = self.all_constants.SamplePart1
        einsum = torch.functional.einsum('bij,ki->bkj', getitem, relation_forward_sample_part1_w);  getitem = relation_forward_sample_part1_w = None
        getitem_1 = kwargs['noise1']
        relation_forward_sample_part11_w = self.all_constants.SamplePart11
        einsum_1 = torch.functional.einsum('bij,ki->bkj', getitem_1, relation_forward_sample_part11_w);  getitem_1 = relation_forward_sample_part11_w = None
        getitem_2 = kwargs['Xangular_velocity']
        relation_forward_sample_part17_w = self.all_constants.SamplePart17
        einsum_2 = torch.functional.einsum('bij,ki->bkj', getitem_2, relation_forward_sample_part17_w);  getitem_2 = relation_forward_sample_part17_w = None
        getitem_3 = kwargs['noise3']
        relation_forward_sample_part19_w = self.all_constants.SamplePart19
        einsum_3 = torch.functional.einsum('bij,ki->bkj', getitem_3, relation_forward_sample_part19_w);  getitem_3 = relation_forward_sample_part19_w = None
        getitem_4 = kwargs['Xangle']
        relation_forward_sample_part25_w = self.all_constants.SamplePart25
        einsum_4 = torch.functional.einsum('bij,ki->bkj', getitem_4, relation_forward_sample_part25_w);  getitem_4 = relation_forward_sample_part25_w = None
        getitem_5 = kwargs['noise4']
        relation_forward_sample_part27_w = self.all_constants.SamplePart27
        einsum_5 = torch.functional.einsum('bij,ki->bkj', getitem_5, relation_forward_sample_part27_w);  getitem_5 = relation_forward_sample_part27_w = None
        getitem_6 = kwargs['noise2']
        relation_forward_sample_part3_w = self.all_constants.SamplePart3
        einsum_6 = torch.functional.einsum('bij,ki->bkj', getitem_6, relation_forward_sample_part3_w);  getitem_6 = relation_forward_sample_part3_w = None
        getitem_7 = kwargs['action']
        relation_forward_sample_part33_w = self.all_constants.SamplePart33
        einsum_7 = torch.functional.einsum('bij,ki->bkj', getitem_7, relation_forward_sample_part33_w);  getitem_7 = relation_forward_sample_part33_w = None
        getitem_8 = kwargs['noise5']
        relation_forward_sample_part35_w = self.all_constants.SamplePart35
        einsum_8 = torch.functional.einsum('bij,ki->bkj', getitem_8, relation_forward_sample_part35_w);  getitem_8 = relation_forward_sample_part35_w = None
        getitem_9 = kwargs['Yangular_velocity']
        relation_forward_sample_part557_w = self.all_constants.SamplePart557
        einsum_9 = torch.functional.einsum('bij,ki->bkj', getitem_9, relation_forward_sample_part557_w);  getitem_9 = relation_forward_sample_part557_w = None
        getitem_10 = kwargs['Yangle']
        relation_forward_sample_part559_w = self.all_constants.SamplePart559
        einsum_10 = torch.functional.einsum('bij,ki->bkj', getitem_10, relation_forward_sample_part559_w);  getitem_10 = relation_forward_sample_part559_w = None
        getitem_11 = kwargs['Yvelocity']
        relation_forward_sample_part561_w = self.all_constants.SamplePart561
        einsum_11 = torch.functional.einsum('bij,ki->bkj', getitem_11, relation_forward_sample_part561_w);  getitem_11 = relation_forward_sample_part561_w = None
        getitem_12 = kwargs['Ypos']
        relation_forward_sample_part563_w = self.all_constants.SamplePart563
        einsum_12 = torch.functional.einsum('bij,ki->bkj', getitem_12, relation_forward_sample_part563_w);  getitem_12 = relation_forward_sample_part563_w = None
        getitem_13 = kwargs['Xvelocity'];  kwargs = None
        relation_forward_sample_part9_w = self.all_constants.SamplePart9
        einsum_13 = torch.functional.einsum('bij,ki->bkj', getitem_13, relation_forward_sample_part9_w);  getitem_13 = relation_forward_sample_part9_w = None
        mul_39 = mul_38 * all_parameters_b;  mul_38 = None
        mul_40 = pow_1 * pow_2;  pow_1 = pow_2 = None
        mul_41 = mul_40 * all_constants_g;  mul_40 = None
        all_constants_sigma_v = self.all_constants.sigma_v
        mul_42 = einsum_1 * all_constants_sigma_v;  einsum_1 = all_constants_sigma_v = None
        mul_43 = all_parameters_m2 * pow_3;  pow_3 = None
        all_constants_sigma_omega = self.all_constants.sigma_omega
        mul_44 = einsum_3 * all_constants_sigma_omega;  einsum_3 = all_constants_sigma_omega = None
        mul_45 = pow_6 * pow_7;  pow_6 = pow_7 = None
        mul_46 = mul_45 * all_constants_g;  mul_45 = None
        all_constants_sigma_theta = self.all_constants.sigma_theta
        mul_47 = einsum_5 * all_constants_sigma_theta;  einsum_5 = all_constants_sigma_theta = None
        mul_48 = all_parameters_m2 * pow_8;  pow_8 = None
        all_constants_sigma_force = self.all_constants.sigma_force
        mul_49 = einsum_8 * all_constants_sigma_force;  einsum_8 = all_constants_sigma_force = None
        mul_50 = pow_11 * pow_12;  pow_11 = pow_12 = None
        mul_51 = mul_50 * all_constants_g;  mul_50 = None
        mul_52 = all_parameters_m2 * pow_13;  pow_13 = None
        mul_53 = all_parameters_m2 * pow_16;  pow_16 = None
        mul_54 = pow_17 * pow_18;  pow_17 = pow_18 = None
        mul_55 = mul_54 * all_constants_g;  mul_54 = all_constants_g = None
        all_constants_sigma_x = self.all_constants.sigma_x
        mul_56 = einsum_6 * all_constants_sigma_x;  einsum_6 = all_constants_sigma_x = None
        neg = -pow_4;  pow_4 = None
        neg_1 = -pow_9;  pow_9 = None
        neg_2 = -pow_14;  pow_14 = None
        neg_3 = -pow_19;  pow_19 = None
        add_12 = einsum_13 + mul_42;  einsum_13 = mul_42 = None
        all_parameters_i = self.all_parameters.I
        add_13 = all_parameters_i + mul_43;  mul_43 = None
        add_14 = einsum_2 + mul_44;  einsum_2 = mul_44 = None
        add_15 = all_parameters_i + mul_48;  mul_48 = None
        add_16 = einsum_4 + mul_47;  einsum_4 = mul_47 = None
        add_17 = einsum_7 + mul_49;  einsum_7 = mul_49 = None
        add_18 = all_parameters_i + mul_52;  mul_52 = None
        add_19 = all_parameters_i + mul_53;  all_parameters_i = mul_53 = None
        add_20 = einsum + mul_56;  einsum = mul_56 = None
        size = add_12.size(0)
        relation_forward_fir15_weights = self.all_parameters.tv2
        size_1 = relation_forward_fir15_weights.size(1)
        squeeze = add_12.squeeze(-1);  add_12 = None
        matmul = torch.matmul(squeeze, relation_forward_fir15_weights);  squeeze = relation_forward_fir15_weights = None
        to = matmul.to(dtype = torch.float32);  matmul = None
        view = to.view(size, 1, size_1);  to = size = size_1 = None
        size_2 = add_14.size(0)
        relation_forward_fir23_weights = self.all_parameters.tv3
        size_3 = relation_forward_fir23_weights.size(1)
        squeeze_1 = add_14.squeeze(-1);  add_14 = None
        matmul_1 = torch.matmul(squeeze_1, relation_forward_fir23_weights);  squeeze_1 = relation_forward_fir23_weights = None
        to_1 = matmul_1.to(dtype = torch.float32);  matmul_1 = None
        view_1 = to_1.view(size_2, 1, size_3);  to_1 = size_2 = size_3 = None
        size_4 = add_16.size(0)
        relation_forward_fir31_weights = self.all_parameters.tv4
        size_5 = relation_forward_fir31_weights.size(1)
        squeeze_2 = add_16.squeeze(-1);  add_16 = None
        matmul_2 = torch.matmul(squeeze_2, relation_forward_fir31_weights);  squeeze_2 = relation_forward_fir31_weights = None
        to_2 = matmul_2.to(dtype = torch.float32);  matmul_2 = None
        view_2 = to_2.view(size_4, 1, size_5);  to_2 = size_4 = size_5 = None
        size_6 = add_17.size(0)
        relation_forward_fir39_weights = self.all_parameters.tv5
        size_7 = relation_forward_fir39_weights.size(1)
        squeeze_3 = add_17.squeeze(-1);  add_17 = None
        matmul_3 = torch.matmul(squeeze_3, relation_forward_fir39_weights);  squeeze_3 = relation_forward_fir39_weights = None
        to_3 = matmul_3.to(dtype = torch.float32);  matmul_3 = None
        view_3 = to_3.view(size_6, 1, size_7);  to_3 = size_6 = size_7 = None
        size_8 = add_20.size(0)
        relation_forward_fir7_weights = self.all_parameters.tv1
        size_9 = relation_forward_fir7_weights.size(1)
        squeeze_4 = add_20.squeeze(-1);  add_20 = None
        matmul_4 = torch.matmul(squeeze_4, relation_forward_fir7_weights);  squeeze_4 = relation_forward_fir7_weights = None
        to_4 = matmul_4.to(dtype = torch.float32);  matmul_4 = None
        view_4 = to_4.view(size_8, 1, size_9);  to_4 = size_8 = size_9 = None
        mul_57 = mul_39 * view;  mul_39 = None
        mul_58 = add_19 * all_parameters_m2
        mul_59 = mul_58 * all_parameters_l;  mul_58 = None
        mul_60 = mul_2 * view_1;  mul_2 = None
        all_constants_dt = self.all_constants.dt
        mul_61 = view_1 * all_constants_dt
        mul_62 = add * add_13;  add = None
        all_constants_gear = self.all_constants.gear
        mul_63 = all_constants_gear * view_3
        mul_64 = add_13 * all_parameters_b
        mul_65 = neg * pow_5;  neg = pow_5 = None
        mul_66 = add_13 * all_parameters_m2
        mul_67 = mul_66 * all_parameters_l;  mul_66 = None
        mul_68 = mul_63 * add_13;  add_13 = None
        mul_69 = add_3 * add_15;  add_3 = None
        mul_70 = all_constants_gear * view_3
        mul_71 = add_15 * all_parameters_b
        mul_72 = neg_1 * pow_10;  neg_1 = pow_10 = None
        mul_73 = add_15 * all_parameters_m2
        mul_74 = mul_73 * all_parameters_l;  mul_73 = None
        mul_75 = mul_70 * add_15;  add_15 = None
        mul_76 = add_6 * add_18;  add_6 = None
        mul_77 = all_constants_gear * view_3
        mul_78 = add_18 * all_parameters_b
        mul_79 = neg_2 * pow_15;  neg_2 = pow_15 = None
        mul_80 = add_18 * all_parameters_m2;  all_parameters_m2 = None
        mul_81 = mul_80 * all_parameters_l;  mul_80 = all_parameters_l = None
        mul_82 = mul_77 * add_18;  add_18 = None
        mul_83 = add_9 * add_19;  add_9 = None
        mul_84 = all_constants_gear * view_3;  all_constants_gear = view_3 = None
        mul_85 = add_19 * all_parameters_b;  all_parameters_b = None
        mul_86 = mul_85 * view;  mul_85 = None
        mul_87 = mul_34 * view_1;  mul_34 = None
        mul_88 = neg_3 * pow_20;  neg_3 = pow_20 = None
        all_constants_constant44 = self.all_constants.Constant44
        pow_21 = torch.pow(view_1, all_constants_constant44);  all_constants_constant44 = None
        all_constants_constant43 = self.all_constants.Constant43
        pow_22 = torch.pow(view_1, all_constants_constant43);  all_constants_constant43 = None
        sin = torch.sin(view_2)
        cos = torch.cos(view_2)
        all_constants_constant49 = self.all_constants.Constant49
        truediv = mul_61 / all_constants_constant49;  mul_61 = all_constants_constant49 = None
        mul_89 = mul_57 * cos;  mul_57 = None
        mul_90 = mul * cos;  mul = None
        mul_91 = mul_90 * mul_84;  mul_90 = None
        mul_92 = mul_59 * pow_21;  mul_59 = pow_21 = None
        mul_93 = mul_92 * sin;  mul_92 = None
        mul_94 = mul_41 * sin;  mul_41 = None
        mul_95 = mul_94 * cos;  mul_94 = None
        mul_96 = mul_60 * cos;  mul_60 = None
        mul_97 = mul_84 * add_19;  mul_84 = add_19 = None
        mul_98 = mul_33 * cos;  mul_33 = None
        mul_99 = mul_88 * pow_22;  mul_88 = pow_22 = None
        mul_100 = mul_99 * sin;  mul_99 = None
        mul_101 = mul_100 * cos;  mul_100 = cos = None
        mul_102 = mul_37 * sin;  mul_37 = sin = None
        all_constants_constant40 = self.all_constants.Constant40
        pow_23 = torch.pow(mul_98, all_constants_constant40);  mul_98 = all_constants_constant40 = None
        sub = mul_93 - mul_86;  mul_93 = mul_86 = None
        sub_1 = sub - mul_95;  sub = mul_95 = None
        sub_2 = mul_83 - pow_23;  mul_83 = pow_23 = None
        sub_3 = mul_101 - mul_87;  mul_101 = mul_87 = None
        add_21 = sub_1 + mul_96;  sub_1 = mul_96 = None
        add_22 = add_21 + mul_97;  add_21 = mul_97 = None
        add_23 = view_2 + truediv;  truediv = None
        add_24 = sub_3 + mul_102;  sub_3 = mul_102 = None
        cos_1 = torch.cos(add_23)
        truediv_1 = add_22 / sub_2;  add_22 = None
        mul_103 = truediv_1 * all_constants_dt
        mul_104 = mul_3 * cos_1;  mul_3 = None
        mul_105 = mul_10 * cos_1;  mul_10 = None
        mul_106 = mul_105 * mul_63;  mul_105 = mul_63 = None
        all_constants_constant52 = self.all_constants.Constant52
        pow_24 = torch.pow(mul_104, all_constants_constant52);  mul_104 = all_constants_constant52 = None
        sin_1 = torch.sin(add_23);  add_23 = None
        sub_4 = mul_62 - pow_24;  mul_62 = pow_24 = None
        add_25 = add_24 + mul_89;  add_24 = mul_89 = None
        all_constants_constant48 = self.all_constants.Constant48
        truediv_2 = mul_103 / all_constants_constant48;  mul_103 = all_constants_constant48 = None
        mul_107 = mul_7 * sin_1;  mul_7 = None
        mul_108 = mul_46 * sin_1;  mul_46 = None
        mul_109 = mul_108 * cos_1;  mul_108 = None
        sub_5 = add_25 - mul_91;  add_25 = mul_91 = None
        add_26 = view + truediv_2;  truediv_2 = None
        truediv_3 = sub_5 / sub_2;  sub_5 = sub_2 = None
        mul_110 = truediv_3 * all_constants_dt
        mul_111 = mul_64 * add_26;  mul_64 = None
        mul_112 = mul_9 * add_26;  mul_9 = None
        mul_113 = mul_112 * cos_1;  mul_112 = None
        all_constants_constant82 = self.all_constants.Constant82
        mul_114 = all_constants_constant82 * add_26;  all_constants_constant82 = add_26 = None
        add_27 = view + mul_114;  mul_114 = None
        all_constants_constant50 = self.all_constants.Constant50
        truediv_4 = mul_110 / all_constants_constant50;  mul_110 = all_constants_constant50 = None
        add_28 = view_1 + truediv_4;  truediv_4 = None
        mul_115 = mul_4 * add_28;  mul_4 = None
        mul_116 = mul_12 * add_28;  mul_12 = None
        mul_117 = mul_116 * cos_1;  mul_116 = None
        mul_118 = add_28 * all_constants_dt
        all_constants_constant88 = self.all_constants.Constant88
        mul_119 = all_constants_constant88 * add_28;  all_constants_constant88 = None
        all_constants_constant55 = self.all_constants.Constant55
        pow_25 = torch.pow(add_28, all_constants_constant55);  all_constants_constant55 = None
        all_constants_constant56 = self.all_constants.Constant56
        pow_26 = torch.pow(add_28, all_constants_constant56);  add_28 = all_constants_constant56 = None
        add_29 = view_1 + mul_119;  mul_119 = None
        all_constants_constant61 = self.all_constants.Constant61
        truediv_5 = mul_118 / all_constants_constant61;  mul_118 = all_constants_constant61 = None
        mul_120 = mul_65 * pow_25;  mul_65 = pow_25 = None
        mul_121 = mul_120 * sin_1;  mul_120 = None
        mul_122 = mul_121 * cos_1;  mul_121 = cos_1 = None
        mul_123 = mul_67 * pow_26;  mul_67 = pow_26 = None
        mul_124 = mul_123 * sin_1;  mul_123 = sin_1 = None
        sub_6 = mul_122 - mul_115;  mul_122 = mul_115 = None
        sub_7 = mul_124 - mul_111;  mul_124 = mul_111 = None
        sub_8 = sub_7 - mul_109;  sub_7 = mul_109 = None
        add_30 = sub_6 + mul_107;  sub_6 = mul_107 = None
        add_31 = add_30 + mul_113;  add_30 = mul_113 = None
        add_32 = sub_8 + mul_117;  sub_8 = mul_117 = None
        add_33 = add_32 + mul_68;  add_32 = mul_68 = None
        add_34 = view_2 + truediv_5;  truediv_5 = None
        cos_2 = torch.cos(add_34)
        truediv_6 = add_33 / sub_4;  add_33 = None
        mul_125 = truediv_6 * all_constants_dt
        mul_126 = mul_13 * cos_2;  mul_13 = None
        mul_127 = mul_20 * cos_2;  mul_20 = None
        mul_128 = mul_127 * mul_70;  mul_127 = mul_70 = None
        all_constants_constant79 = self.all_constants.Constant79
        mul_129 = all_constants_constant79 * truediv_6;  all_constants_constant79 = truediv_6 = None
        all_constants_constant64 = self.all_constants.Constant64
        pow_27 = torch.pow(mul_126, all_constants_constant64);  mul_126 = all_constants_constant64 = None
        sin_2 = torch.sin(add_34);  add_34 = None
        sub_9 = add_31 - mul_106;  add_31 = mul_106 = None
        sub_10 = mul_69 - pow_27;  mul_69 = pow_27 = None
        add_35 = truediv_1 + mul_129;  truediv_1 = mul_129 = None
        truediv_7 = sub_9 / sub_4;  sub_9 = sub_4 = None
        all_constants_constant60 = self.all_constants.Constant60
        truediv_8 = mul_125 / all_constants_constant60;  mul_125 = all_constants_constant60 = None
        mul_130 = truediv_7 * all_constants_dt
        mul_131 = mul_17 * sin_2;  mul_17 = None
        mul_132 = mul_51 * sin_2;  mul_51 = None
        mul_133 = mul_132 * cos_2;  mul_132 = None
        all_constants_constant85 = self.all_constants.Constant85
        mul_134 = all_constants_constant85 * truediv_7;  all_constants_constant85 = truediv_7 = None
        add_36 = view + truediv_8;  truediv_8 = None
        add_37 = truediv_3 + mul_134;  truediv_3 = mul_134 = None
        all_constants_constant62 = self.all_constants.Constant62
        truediv_9 = mul_130 / all_constants_constant62;  mul_130 = all_constants_constant62 = None
        mul_135 = mul_71 * add_36;  mul_71 = None
        mul_136 = mul_19 * add_36;  mul_19 = None
        mul_137 = mul_136 * cos_2;  mul_136 = None
        all_constants_constant83 = self.all_constants.Constant83
        mul_138 = all_constants_constant83 * add_36;  all_constants_constant83 = add_36 = None
        add_38 = view_1 + truediv_9;  truediv_9 = None
        add_39 = add_27 + mul_138;  add_27 = mul_138 = None
        mul_139 = mul_14 * add_38;  mul_14 = None
        mul_140 = mul_22 * add_38;  mul_22 = None
        mul_141 = mul_140 * cos_2;  mul_140 = None
        mul_142 = add_38 * all_constants_dt
        all_constants_constant89 = self.all_constants.Constant89
        mul_143 = all_constants_constant89 * add_38;  all_constants_constant89 = None
        all_constants_constant67 = self.all_constants.Constant67
        pow_28 = torch.pow(add_38, all_constants_constant67);  all_constants_constant67 = None
        all_constants_constant68 = self.all_constants.Constant68
        pow_29 = torch.pow(add_38, all_constants_constant68);  add_38 = all_constants_constant68 = None
        add_40 = view_2 + mul_142;  mul_142 = None
        add_41 = add_29 + mul_143;  add_29 = mul_143 = None
        cos_3 = torch.cos(add_40)
        mul_144 = mul_72 * pow_28;  mul_72 = pow_28 = None
        mul_145 = mul_144 * sin_2;  mul_144 = None
        mul_146 = mul_145 * cos_2;  mul_145 = cos_2 = None
        mul_147 = mul_74 * pow_29;  mul_74 = pow_29 = None
        mul_148 = mul_147 * sin_2;  mul_147 = sin_2 = None
        mul_149 = mul_23 * cos_3;  mul_23 = None
        mul_150 = mul_30 * cos_3;  mul_30 = None
        mul_151 = mul_150 * mul_77;  mul_150 = mul_77 = None
        all_constants_constant72 = self.all_constants.Constant72
        pow_30 = torch.pow(mul_149, all_constants_constant72);  mul_149 = all_constants_constant72 = None
        sin_3 = torch.sin(add_40);  add_40 = None
        sub_11 = mul_146 - mul_139;  mul_146 = mul_139 = None
        sub_12 = mul_148 - mul_135;  mul_148 = mul_135 = None
        sub_13 = sub_12 - mul_133;  sub_12 = mul_133 = None
        sub_14 = mul_76 - pow_30;  mul_76 = pow_30 = None
        add_42 = sub_11 + mul_131;  sub_11 = mul_131 = None
        add_43 = add_42 + mul_137;  add_42 = mul_137 = None
        add_44 = sub_13 + mul_141;  sub_13 = mul_141 = None
        add_45 = add_44 + mul_75;  add_44 = mul_75 = None
        truediv_10 = add_45 / sub_10;  add_45 = None
        mul_152 = truediv_10 * all_constants_dt
        mul_153 = mul_27 * sin_3;  mul_27 = None
        mul_154 = mul_55 * sin_3;  mul_55 = None
        mul_155 = mul_154 * cos_3;  mul_154 = None
        all_constants_constant80 = self.all_constants.Constant80
        mul_156 = all_constants_constant80 * truediv_10;  all_constants_constant80 = truediv_10 = None
        sub_15 = add_43 - mul_128;  add_43 = mul_128 = None
        add_46 = view + mul_152;  mul_152 = None
        add_47 = add_35 + mul_156;  add_35 = mul_156 = None
        add_48 = add_39 + add_46;  add_39 = None
        truediv_11 = sub_15 / sub_10;  sub_15 = sub_10 = None
        mul_157 = truediv_11 * all_constants_dt
        mul_158 = mul_78 * add_46;  mul_78 = None
        mul_159 = mul_29 * add_46;  mul_29 = add_46 = None
        mul_160 = mul_159 * cos_3;  mul_159 = None
        mul_161 = add_48 * all_constants_dt;  add_48 = None
        all_constants_constant86 = self.all_constants.Constant86
        mul_162 = all_constants_constant86 * truediv_11;  all_constants_constant86 = truediv_11 = None
        add_49 = view_1 + mul_157;  mul_157 = None
        add_50 = add_37 + mul_162;  add_37 = mul_162 = None
        add_51 = add_41 + add_49;  add_41 = None
        all_constants_constant84 = self.all_constants.Constant84
        truediv_12 = mul_161 / all_constants_constant84;  mul_161 = all_constants_constant84 = None
        mul_163 = mul_24 * add_49;  mul_24 = None
        mul_164 = mul_32 * add_49;  mul_32 = None
        mul_165 = mul_164 * cos_3;  mul_164 = None
        mul_166 = add_51 * all_constants_dt;  add_51 = None
        all_constants_constant75 = self.all_constants.Constant75
        pow_31 = torch.pow(add_49, all_constants_constant75);  all_constants_constant75 = None
        all_constants_constant76 = self.all_constants.Constant76
        pow_32 = torch.pow(add_49, all_constants_constant76);  add_49 = all_constants_constant76 = None
        add_52 = view_4 + truediv_12;  view_4 = truediv_12 = None
        all_constants_constant90 = self.all_constants.Constant90
        truediv_13 = mul_166 / all_constants_constant90;  mul_166 = all_constants_constant90 = None
        mul_167 = mul_79 * pow_31;  mul_79 = pow_31 = None
        mul_168 = mul_167 * sin_3;  mul_167 = None
        mul_169 = mul_168 * cos_3;  mul_168 = cos_3 = None
        mul_170 = mul_81 * pow_32;  mul_81 = pow_32 = None
        mul_171 = mul_170 * sin_3;  mul_170 = sin_3 = None
        sub_16 = mul_169 - mul_163;  mul_169 = mul_163 = None
        sub_17 = mul_171 - mul_158;  mul_171 = mul_158 = None
        sub_18 = sub_17 - mul_155;  sub_17 = mul_155 = None
        add_53 = sub_16 + mul_153;  sub_16 = mul_153 = None
        add_54 = add_53 + mul_160;  add_53 = mul_160 = None
        add_55 = sub_18 + mul_165;  sub_18 = mul_165 = None
        add_56 = add_55 + mul_82;  add_55 = mul_82 = None
        add_57 = view_2 + truediv_13;  view_2 = truediv_13 = None
        truediv_14 = add_56 / sub_14;  add_56 = None
        sub_19 = add_54 - mul_151;  add_54 = mul_151 = None
        add_58 = add_47 + truediv_14;  add_47 = truediv_14 = None
        truediv_15 = sub_19 / sub_14;  sub_19 = sub_14 = None
        mul_172 = add_58 * all_constants_dt;  add_58 = None
        add_59 = add_50 + truediv_15;  add_50 = truediv_15 = None
        all_constants_constant81 = self.all_constants.Constant81
        truediv_16 = mul_172 / all_constants_constant81;  mul_172 = all_constants_constant81 = None
        mul_173 = add_59 * all_constants_dt;  add_59 = all_constants_dt = None
        add_60 = view + truediv_16;  view = truediv_16 = None
        all_constants_constant87 = self.all_constants.Constant87
        truediv_17 = mul_173 / all_constants_constant87;  mul_173 = all_constants_constant87 = None
        add_61 = view_1 + truediv_17;  view_1 = truediv_17 = None
        return ({'est_theta': add_57, 'est_thetadot': add_61, 'est_x': add_52, 'est_xdot': add_60}, {'SamplePart557': einsum_9, 'SamplePart559': einsum_10, 'SamplePart563': einsum_12, 'SamplePart561': einsum_11, 'Add539': add_61, 'Add551': add_57, 'Add527': add_52, 'Add515': add_60}, {'Xangle': add_57, 'Xangular_velocity': add_61, 'Xpos': add_52, 'Xvelocity': add_60}, {})
        
class RecurrentModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.Cell = TracerModel()
        self.inputs = ['Yangle', 'Yangular_velocity', 'Ypos', 'Yvelocity', 'action', 'noise1', 'noise2', 'noise3', 'noise4', 'noise5', ]
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
