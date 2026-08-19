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
        self.all_constants["SampleTime"] = torch.tensor(0.009999999776482582, requires_grad=False)
        self.all_parameters["PLinear11W"] = torch.nn.Parameter(torch.tensor([[-0.005024774000048637, 0.0070754485204815865, -0.006699301302433014, 0.002786495955660939, 0.005666721612215042, 0.003927223384380341, 0.0051276423037052155, 0.00569740729406476, -0.0025189740117639303, 0.12829028069972992, 0.002694192575290799, -0.012417715974152088, -0.061624977737665176, -0.05015244334936142], [-0.24690237641334534, -0.20113025605678558, 0.618537425994873, 0.02111278474330902, 0.20968112349510193, -0.6530620455741882, 0.03690250590443611, -0.6463745832443237, 0.20553390681743622, -0.939859926700592, -0.1262694150209427, 0.20213864743709564, 0.5276914238929749, -1.250641107559204], [-0.2842886745929718, 0.24758891761302948, -0.4436063766479492, 0.13822385668754578, 1.046595811843872, 0.10145721584558487, -1.9004749059677124, -0.4073916971683502, 0.44326284527778625, 0.33916524052619934, 0.12422680109739304, 0.6009740233421326, -0.17909349501132965, -0.07140125334262848], [-0.0006047824281267822, 0.006958967540413141, 0.011861679144203663, 0.010378708131611347, 0.03229401260614395, -0.007714221253991127, 0.030956396833062172, -0.00238688662648201, 0.010870832949876785, -0.0033055536914616823, -0.003538951277732849, 0.006481728050857782, -0.026257218793034554, -0.051497530192136765], [0.002040404826402664, -0.01152180414646864, -0.01436692662537098, -0.012164405547082424, -0.00437884358689189, 0.005081120412796736, 0.010659253224730492, -0.003283773548901081, -0.005208977498114109, -0.006258426234126091, 0.004453802015632391, 0.020384546369314194, 0.02829315885901451, -0.016384605318307877], [0.0014811275759711862, -0.017344700172543526, -0.029048973694443703, -0.017984192818403244, -0.02976372092962265, 0.02106775902211666, 0.032418426126241684, 0.0030518684070557356, -0.01598547399044037, -0.020970789715647697, 0.005948926322162151, 0.02157912403345108, 0.02669992297887802, -0.002020308980718255], [-0.6100114583969116, -0.4712432622909546, -0.24753445386886597, 0.5373743772506714, -0.12614622712135315, 0.12248466908931732, 0.09624070674180984, 0.015742909163236618, -0.5159809589385986, 0.2599658668041229, 0.03230675682425499, 0.09847939759492874, 0.1640866994857788, 0.17894315719604492], [0.19018322229385376, 0.11691410839557648, 0.11169371008872986, -0.19859929382801056, 0.3376302719116211, -0.3198295533657074, -0.03743542730808258, 0.26954084634780884, 0.24210916459560394, 0.1336430311203003, -0.2550838589668274, -0.13442391157150269, -0.07794436812400818, -0.00802956335246563], [-0.22828422486782074, -0.2116599678993225, 0.13476483523845673, 0.05609399452805519, 0.24261006712913513, -0.24986889958381653, -0.179604634642601, 0.2513013780117035, 0.10117608308792114, 0.21767334640026093, -0.22259023785591125, -0.2547523081302643, -0.13416679203510284, 0.2581072449684143]]), requires_grad=True)
        self.all_parameters["PLinear13W"] = torch.nn.Parameter(torch.tensor([[1.0430445671081543, 2.9539401531219482, -3.6721324920654297], [1.0216633081436157, 3.156709671020508, -4.2261061668396], [-1.775482416152954, -3.180358648300171, 2.826387405395508], [2.3749964237213135, 2.2043521404266357, -3.213824987411499], [1.6129227876663208, 4.2221503257751465, -1.3658277988433838], [-1.4238938093185425, -4.113351821899414, 1.0172940492630005], [1.9487433433532715, 3.8043155670166016, -2.7885091304779053], [1.6254310607910156, 3.83312726020813, -1.224905252456665], [-2.073394298553467, -3.190133810043335, 3.2779881954193115], [1.8437706232070923, 3.8037140369415283, -1.7493656873703003], [-1.6617408990859985, -4.326606273651123, 0.9229745864868164], [1.779322862625122, 3.0580997467041016, -2.7818875312805176], [2.241752862930298, 3.6468119621276855, -3.1833512783050537], [-1.4461662769317627, -3.157651901245117, 3.929004669189453]]), requires_grad=True)
        self.all_constants["SamplePart1"] = torch.tensor([[1.0]], requires_grad=True)
        self.all_constants["SamplePart11"] = torch.tensor([[1.0]], requires_grad=True)
        self.all_constants["SamplePart14"] = torch.tensor([[1.0]], requires_grad=True)
        self.all_constants["SamplePart16"] = torch.tensor([[1.0]], requires_grad=True)
        self.all_constants["SamplePart18"] = torch.tensor([[1.0]], requires_grad=True)
        self.all_constants["SamplePart20"] = torch.tensor([[1.0]], requires_grad=True)
        self.all_constants["SamplePart3"] = torch.tensor([[1.0]], requires_grad=True)
        self.all_constants["SamplePart31"] = torch.tensor([[1.0]], requires_grad=True)
        self.all_constants["SamplePart38"] = torch.tensor([[1.0]], requires_grad=True)
        self.all_constants["SamplePart45"] = torch.tensor([[1.0]], requires_grad=True)
        self.all_constants["SamplePart51"] = torch.tensor([[1.0]], requires_grad=True)
        self.all_constants["SamplePart57"] = torch.tensor([[1.0]], requires_grad=True)
        self.all_constants["SamplePart6"] = torch.tensor([[1.0]], requires_grad=True)
        self.all_constants["SamplePart63"] = torch.tensor([[1.0]], requires_grad=True)
        self.all_constants["SamplePart78"] = torch.tensor([[1.0]], requires_grad=True)
        self.all_constants["SamplePart80"] = torch.tensor([[1.0]], requires_grad=True)
        self.all_constants["SamplePart82"] = torch.tensor([[1.0]], requires_grad=True)
        self.all_constants["SamplePart9"] = torch.tensor([[1.0]], requires_grad=True)
        self.all_constants["Select29"] = torch.tensor([0.0, 1.0, 0.0], requires_grad=True)
        self.all_constants["Select36"] = torch.tensor([0.0, 0.0, 1.0], requires_grad=True)
        self.all_constants["Select43"] = torch.tensor([1.0, 0.0, 0.0], requires_grad=True)
        self.all_constants["Select68"] = torch.tensor([0.0, 1.0, 0.0], requires_grad=True)
        self.all_constants["Select69"] = torch.tensor([0.0, 0.0, 1.0], requires_grad=True)
        self.all_constants["Select70"] = torch.tensor([1.0, 0.0, 0.0], requires_grad=True)
        self.all_parameters = torch.nn.ParameterDict(self.all_parameters)
        self.all_constants = torch.nn.ParameterDict(self.all_constants)

    def update(self, closed_loop={}, connect={}, disconnect=False):
        pass
    
    def forward(self, kwargs):
        getitem = kwargs['Xddth2']
        relation_forward_sample_part20_w = self.all_constants.SamplePart20
        einsum = torch.functional.einsum('bij,ki->bkj', getitem, relation_forward_sample_part20_w);  getitem = relation_forward_sample_part20_w = None
        getitem_1 = kwargs['Xddth1']
        relation_forward_sample_part18_w = self.all_constants.SamplePart18
        einsum_1 = torch.functional.einsum('bij,ki->bkj', getitem_1, relation_forward_sample_part18_w);  getitem_1 = relation_forward_sample_part18_w = None
        cat = torch.cat((einsum_1, einsum), dim = 2);  einsum_1 = einsum = None
        getitem_2 = kwargs['Xddx']
        relation_forward_sample_part16_w = self.all_constants.SamplePart16
        einsum_2 = torch.functional.einsum('bij,ki->bkj', getitem_2, relation_forward_sample_part16_w);  getitem_2 = relation_forward_sample_part16_w = None
        cat_1 = torch.cat((einsum_2, cat), dim = 2);  einsum_2 = cat = None
        getitem_3 = kwargs['Xth2_dot']
        relation_forward_sample_part14_w = self.all_constants.SamplePart14
        einsum_3 = torch.functional.einsum('bij,ki->bkj', getitem_3, relation_forward_sample_part14_w);  getitem_3 = relation_forward_sample_part14_w = None
        cat_2 = torch.cat((einsum_3, cat_1), dim = 2);  einsum_3 = cat_1 = None
        getitem_4 = kwargs['Xth1_dot']
        relation_forward_sample_part11_w = self.all_constants.SamplePart11
        einsum_4 = torch.functional.einsum('bij,ki->bkj', getitem_4, relation_forward_sample_part11_w);  getitem_4 = relation_forward_sample_part11_w = None
        getitem_5 = kwargs['Xvelocity']
        relation_forward_sample_part9_w = self.all_constants.SamplePart9
        einsum_5 = torch.functional.einsum('bij,ki->bkj', getitem_5, relation_forward_sample_part9_w);  getitem_5 = relation_forward_sample_part9_w = None
        cat_3 = torch.cat((einsum_5, einsum_4), dim = 2);  einsum_5 = einsum_4 = None
        cat_4 = torch.cat((cat_3, cat_2), dim = 2);  cat_3 = cat_2 = None
        getitem_6 = kwargs['Xth2']
        relation_forward_sample_part6_w = self.all_constants.SamplePart6
        einsum_6 = torch.functional.einsum('bij,ki->bkj', getitem_6, relation_forward_sample_part6_w);  getitem_6 = relation_forward_sample_part6_w = None
        getitem_7 = kwargs['Xth1']
        relation_forward_sample_part3_w = self.all_constants.SamplePart3
        einsum_7 = torch.functional.einsum('bij,ki->bkj', getitem_7, relation_forward_sample_part3_w);  getitem_7 = relation_forward_sample_part3_w = None
        getitem_8 = kwargs['Xpos']
        relation_forward_sample_part1_w = self.all_constants.SamplePart1
        einsum_8 = torch.functional.einsum('bij,ki->bkj', getitem_8, relation_forward_sample_part1_w);  getitem_8 = relation_forward_sample_part1_w = None
        cat_5 = torch.cat((einsum_8, einsum_7), dim = 2);  einsum_8 = einsum_7 = None
        cat_6 = torch.cat((cat_5, einsum_6), dim = 2);  cat_5 = einsum_6 = None
        cat_7 = torch.cat((cat_6, cat_4), dim = 2);  cat_6 = cat_4 = None
        relation_forward_linear26_weights = self.all_parameters.PLinear11W
        einsum_9 = torch.functional.einsum('bwi,io->bwo', cat_7, relation_forward_linear26_weights);  cat_7 = relation_forward_linear26_weights = None
        tanh = torch.tanh(einsum_9);  einsum_9 = None
        relation_forward_linear28_weights = self.all_parameters.PLinear13W
        einsum_10 = torch.functional.einsum('bwi,io->bwo', tanh, relation_forward_linear28_weights);  tanh = relation_forward_linear28_weights = None
        relation_forward_select69_w = self.all_constants.Select69
        einsum_11 = torch.functional.einsum('ijk,k->ij', einsum_10, relation_forward_select69_w);  relation_forward_select69_w = None
        unsqueeze = einsum_11.unsqueeze(2);  einsum_11 = None
        getitem_9 = kwargs['Xddth2']
        relation_forward_sample_part82_w = self.all_constants.SamplePart82
        einsum_12 = torch.functional.einsum('bij,ki->bkj', getitem_9, relation_forward_sample_part82_w);  getitem_9 = relation_forward_sample_part82_w = None
        relation_forward_select68_w = self.all_constants.Select68
        einsum_13 = torch.functional.einsum('ijk,k->ij', einsum_10, relation_forward_select68_w);  relation_forward_select68_w = None
        unsqueeze_1 = einsum_13.unsqueeze(2);  einsum_13 = None
        getitem_10 = kwargs['Xddth1']
        relation_forward_sample_part80_w = self.all_constants.SamplePart80
        einsum_14 = torch.functional.einsum('bij,ki->bkj', getitem_10, relation_forward_sample_part80_w);  getitem_10 = relation_forward_sample_part80_w = None
        relation_forward_select70_w = self.all_constants.Select70
        einsum_15 = torch.functional.einsum('ijk,k->ij', einsum_10, relation_forward_select70_w);  relation_forward_select70_w = None
        unsqueeze_2 = einsum_15.unsqueeze(2);  einsum_15 = None
        getitem_11 = kwargs['Xddx']
        relation_forward_sample_part78_w = self.all_constants.SamplePart78
        einsum_16 = torch.functional.einsum('bij,ki->bkj', getitem_11, relation_forward_sample_part78_w);  getitem_11 = relation_forward_sample_part78_w = None
        relation_forward_select43_w = self.all_constants.Select43
        einsum_17 = torch.functional.einsum('ijk,k->ij', einsum_10, relation_forward_select43_w);  relation_forward_select43_w = None
        unsqueeze_3 = einsum_17.unsqueeze(2);  einsum_17 = None
        all_constants_sample_time = self.all_constants.SampleTime
        mul = unsqueeze_3 * all_constants_sample_time;  unsqueeze_3 = None
        getitem_12 = kwargs['int_xdot']
        relation_forward_sample_part45_w = self.all_constants.SamplePart45
        einsum_18 = torch.functional.einsum('bij,ki->bkj', getitem_12, relation_forward_sample_part45_w);  getitem_12 = relation_forward_sample_part45_w = None
        add = einsum_18 + mul;  einsum_18 = mul = None
        mul_1 = add * all_constants_sample_time
        getitem_13 = kwargs['int_x']
        relation_forward_sample_part63_w = self.all_constants.SamplePart63
        einsum_19 = torch.functional.einsum('bij,ki->bkj', getitem_13, relation_forward_sample_part63_w);  getitem_13 = relation_forward_sample_part63_w = None
        add_1 = einsum_19 + mul_1;  einsum_19 = mul_1 = None
        relation_forward_select36_w = self.all_constants.Select36
        einsum_20 = torch.functional.einsum('ijk,k->ij', einsum_10, relation_forward_select36_w);  relation_forward_select36_w = None
        unsqueeze_4 = einsum_20.unsqueeze(2);  einsum_20 = None
        mul_2 = unsqueeze_4 * all_constants_sample_time;  unsqueeze_4 = None
        getitem_14 = kwargs['int_th2_dot']
        relation_forward_sample_part38_w = self.all_constants.SamplePart38
        einsum_21 = torch.functional.einsum('bij,ki->bkj', getitem_14, relation_forward_sample_part38_w);  getitem_14 = relation_forward_sample_part38_w = None
        add_2 = einsum_21 + mul_2;  einsum_21 = mul_2 = None
        mul_3 = add_2 * all_constants_sample_time
        getitem_15 = kwargs['int_th2']
        relation_forward_sample_part57_w = self.all_constants.SamplePart57
        einsum_22 = torch.functional.einsum('bij,ki->bkj', getitem_15, relation_forward_sample_part57_w);  getitem_15 = relation_forward_sample_part57_w = None
        add_3 = einsum_22 + mul_3;  einsum_22 = mul_3 = None
        relation_forward_select29_w = self.all_constants.Select29
        einsum_23 = torch.functional.einsum('ijk,k->ij', einsum_10, relation_forward_select29_w);  einsum_10 = relation_forward_select29_w = None
        unsqueeze_5 = einsum_23.unsqueeze(2);  einsum_23 = None
        mul_4 = unsqueeze_5 * all_constants_sample_time;  unsqueeze_5 = None
        getitem_16 = kwargs['int_th1_dot']
        relation_forward_sample_part31_w = self.all_constants.SamplePart31
        einsum_24 = torch.functional.einsum('bij,ki->bkj', getitem_16, relation_forward_sample_part31_w);  getitem_16 = relation_forward_sample_part31_w = None
        add_4 = einsum_24 + mul_4;  einsum_24 = mul_4 = None
        mul_5 = add_4 * all_constants_sample_time;  all_constants_sample_time = None
        getitem_17 = kwargs['int_th1'];  kwargs = None
        relation_forward_sample_part51_w = self.all_constants.SamplePart51
        einsum_25 = torch.functional.einsum('bij,ki->bkj', getitem_17, relation_forward_sample_part51_w);  getitem_17 = relation_forward_sample_part51_w = None
        add_5 = einsum_25 + mul_5;  einsum_25 = mul_5 = None
        return ({'theta2_dotdot_est': unsqueeze, 'theta1_dotdot_est': unsqueeze_1, 'x_dotdot_est': unsqueeze_2, 'x_est': add_1, 'theta2_est': add_3, 'theta1_est': add_5, 'x_dot_est': add, 'theta2_dot_est': add_2, 'theta1_dot_est': add_4}, {'SamplePart78': einsum_16, 'SamplePart80': einsum_14, 'SamplePart82': einsum_12, 'Select70': unsqueeze_2, 'Select68': unsqueeze_1, 'Select69': unsqueeze}, {'Xth2_dot': add_2, 'Xth1_dot': add_4, 'Xvelocity': add, 'Xth2': add_3, 'Xth1': add_5, 'Xpos': add_1, 'int_x': add_1, 'int_xdot': add, 'int_th2': add_3, 'int_th2_dot': add_2, 'int_th1': add_5, 'int_th1_dot': add_4}, {})
        
class RecurrentModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.Cell = TracerModel()
        self.inputs = ['Xddth2', 'Xddth1', 'Xddx', ]
        self.states = dict()

    def forward(self, kwargs, n_samples = None):
        n_samples = n_samples if n_samples else min([kwargs[key].size(0) for key in self.inputs])
        self.states['Xth2_dot'] = kwargs['Xth2_dot']
        self.states['Xth1_dot'] = kwargs['Xth1_dot']
        self.states['Xvelocity'] = kwargs['Xvelocity']
        self.states['Xth2'] = kwargs['Xth2']
        self.states['Xth1'] = kwargs['Xth1']
        self.states['Xpos'] = kwargs['Xpos']
        self.states['int_x'] = kwargs['int_x']
        self.states['int_xdot'] = kwargs['int_xdot']
        self.states['int_th2'] = kwargs['int_th2']
        self.states['int_th2_dot'] = kwargs['int_th2_dot']
        self.states['int_th1'] = kwargs['int_th1']
        self.states['int_th1_dot'] = kwargs['int_th1_dot']
        results = {'theta2_dotdot_est':[], 'theta1_dotdot_est':[], 'x_dotdot_est':[], 'x_est':[], 'theta2_est':[], 'theta1_est':[], 'x_dot_est':[], 'theta2_dot_est':[], 'theta1_dot_est':[], }
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
