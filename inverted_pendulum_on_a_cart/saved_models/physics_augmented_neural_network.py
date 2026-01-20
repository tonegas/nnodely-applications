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
        self.all_constants["Constant43"] = torch.tensor([2.0], requires_grad=False)
        self.all_constants["Constant44"] = torch.tensor([2.0], requires_grad=False)
        self.all_constants["Constant45"] = torch.tensor([2.0], requires_grad=False)
        self.all_constants["Constant46"] = torch.tensor([2.0], requires_grad=False)
        self.all_constants["Constant47"] = torch.tensor([2.0], requires_grad=False)
        self.all_constants["Constant48"] = torch.tensor([2.0], requires_grad=False)
        self.all_constants["Constant49"] = torch.tensor([2.0], requires_grad=False)
        self.all_constants["Constant50"] = torch.tensor([2.0], requires_grad=False)
        self.all_constants["Constant52"] = torch.tensor([2.0], requires_grad=False)
        self.all_constants["Constant53"] = torch.tensor([2.0], requires_grad=False)
        self.all_constants["Constant54"] = torch.tensor([2.0], requires_grad=False)
        self.all_constants["Constant55"] = torch.tensor([2.0], requires_grad=False)
        self.all_constants["Constant56"] = torch.tensor([2.0], requires_grad=False)
        self.all_constants["Constant57"] = torch.tensor([2.0], requires_grad=False)
        self.all_constants["Constant58"] = torch.tensor([2.0], requires_grad=False)
        self.all_constants["Constant59"] = torch.tensor([2.0], requires_grad=False)
        self.all_constants["Constant60"] = torch.tensor([2.0], requires_grad=False)
        self.all_constants["Constant61"] = torch.tensor([2.0], requires_grad=False)
        self.all_constants["Constant62"] = torch.tensor([2.0], requires_grad=False)
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
        self.all_constants["Constant81"] = torch.tensor([2.0], requires_grad=False)
        self.all_constants["Constant82"] = torch.tensor([2.0], requires_grad=False)
        self.all_constants["Constant83"] = torch.tensor([2.0], requires_grad=False)
        self.all_constants["Constant84"] = torch.tensor([2.0], requires_grad=False)
        self.all_constants["Constant85"] = torch.tensor([6.0], requires_grad=False)
        self.all_constants["Constant86"] = torch.tensor([2.0], requires_grad=False)
        self.all_constants["Constant87"] = torch.tensor([2.0], requires_grad=False)
        self.all_constants["Constant88"] = torch.tensor([6.0], requires_grad=False)
        self.all_constants["Constant89"] = torch.tensor([2.0], requires_grad=False)
        self.all_constants["Constant90"] = torch.tensor([2.0], requires_grad=False)
        self.all_constants["Constant91"] = torch.tensor([6.0], requires_grad=False)
        self.all_constants["Constant92"] = torch.tensor([2.0], requires_grad=False)
        self.all_constants["Constant93"] = torch.tensor([2.0], requires_grad=False)
        self.all_constants["Constant94"] = torch.tensor([6.0], requires_grad=False)
        self.all_constants["dt"] = torch.tensor([0.019999999552965164], requires_grad=False)
        self.all_constants["g"] = torch.tensor([9.8100004196167], requires_grad=False)
        self.all_constants["gear"] = torch.tensor([100.0], requires_grad=False)
        self.all_constants["sigma_omega"] = torch.tensor([0.0], requires_grad=False)
        self.all_constants["sigma_theta"] = torch.tensor([0.0], requires_grad=False)
        self.all_constants["sigma_v"] = torch.tensor([0.0], requires_grad=False)
        self.all_constants["sigma_x"] = torch.tensor([0.0], requires_grad=False)
        self.all_parameters["I"] = torch.nn.Parameter(torch.tensor([0.39864471554756165]), requires_grad=True)
        self.all_parameters["b"] = torch.nn.Parameter(torch.tensor([0.9730616211891174]), requires_grad=True)
        self.all_parameters["d"] = torch.nn.Parameter(torch.tensor([1.1174551248550415]), requires_grad=True)
        self.all_parameters["l"] = torch.nn.Parameter(torch.tensor([0.21911874413490295]), requires_grad=True)
        self.all_parameters["m1"] = torch.nn.Parameter(torch.tensor([6.780579090118408]), requires_grad=True)
        self.all_parameters["m2"] = torch.nn.Parameter(torch.tensor([6.922131061553955]), requires_grad=True)
        self.all_parameters["PLinear27W"] = torch.nn.Parameter(torch.tensor([[-0.0001880190393421799, 0.005893010180443525, 0.06466307491064072, 0.008933239616453648, -0.009084898978471756, 0.017183907330036163, -0.019742656499147415, 1.083087113329384e-06, 0.011865868233144283, 0.3578527569770813]]), requires_grad=True)
        self.all_parameters["PLinear29W"] = torch.nn.Parameter(torch.tensor([[-0.0003215222095604986, -0.0159317534416914, 0.0040748706087470055, 0.0024561698082834482, 0.0008748300024308264, 0.005368994083255529, 0.0011685825884342194, 4.917534897685982e-05, -0.0023861457593739033, 0.004730813670903444]]), requires_grad=True)
        self.all_parameters["PLinear31W"] = torch.nn.Parameter(torch.tensor([[-0.004730580374598503, 0.0158639308065176, -0.24506770074367523, -0.6142343282699585, -0.41605210304260254, 0.3105451464653015, -0.5566937327384949, 0.0005361298681236804, 0.7950761318206787, -0.028897438198328018]]), requires_grad=True)
        self.all_parameters["PLinear33W"] = torch.nn.Parameter(torch.tensor([[-0.022707810625433922, -0.024240678176283836, -6.05483292019926e-05, 0.0039931898936629295, 0.011852693744003773, -0.024018894881010056, 0.017550267279148102, 0.0002702289493754506, 0.027396487072110176, -0.0028888338711112738]]), requires_grad=True)
        self.all_parameters["PLinear35W"] = torch.nn.Parameter(torch.tensor([[-0.0007065734243951738, -0.004066101275384426, -0.07811392843723297, 0.11103671789169312, 0.011188536882400513, -0.03196336701512337, -0.09168338775634766, -4.609256939147599e-05, -0.025937706232070923, -0.10917679965496063]]), requires_grad=True)
        self.all_parameters["PLinear37W"] = torch.nn.Parameter(torch.tensor([[0.02063254453241825, 0.0494617260992527, 0.04872281104326248, -0.012855189852416515, -0.10390949249267578, -0.10877903550863266, 0.03513195738196373, 0.00023904282716102898, 0.009891501627862453, 0.034692347049713135]]), requires_grad=True)
        self.all_parameters["PLinear39W"] = torch.nn.Parameter(torch.tensor([[-0.014171193353831768, 0.15714071691036224, 0.1192670539021492, 0.22510765492916107, -0.03281927481293678, 0.05098520964384079, -0.19426411390304565, -0.045756496489048004, -0.04010786861181259, -0.16718825697898865], [-0.012721898965537548, -0.047862086445093155, 0.10460641980171204, 0.07384853065013885, 0.11315490305423737, 0.09630806744098663, -0.13797935843467712, 0.13106301426887512, -0.08511984348297119, -0.12640643119812012], [0.09282247722148895, 0.15721435844898224, -0.15941648185253143, 0.04045811668038368, 0.17485108971595764, 0.19064748287200928, -0.3914415240287781, -0.18088196218013763, -0.20432069897651672, -0.39218342304229736], [0.020315401256084442, 0.17174473404884338, -0.0941096842288971, 0.13578255474567413, 0.0036467043682932854, 0.03338479623198509, -0.17292973399162292, -0.002710219006985426, -0.036179885268211365, -0.168988436460495], [-0.002569112228229642, -0.0162314735352993, -0.04215381294488907, -0.08848243951797485, -0.06981850415468216, 0.139090895652771, -0.22295518219470978, -0.25321006774902344, -0.12679491937160492, -0.22471360862255096], [0.013126885518431664, -0.23434196412563324, -0.06248229742050171, -0.31178855895996094, 0.18223148584365845, -0.03691273182630539, -0.03208531066775322, 0.07352177798748016, 0.03043670579791069, 0.007704177405685186], [-0.04763412103056908, 0.0794190764427185, 0.002835692372173071, 0.2109101116657257, -0.11395229399204254, -0.10175381600856781, 0.025590883567929268, 0.14064401388168335, 0.10899025946855545, 0.03457411751151085], [0.002361225662752986, -0.005406251177191734, 0.04142311215400696, 0.00313198403455317, 0.0010670534102246165, 0.0003897600690834224, 0.01573532447218895, 0.02508380077779293, -0.00209144689142704, 0.019812580198049545], [0.006501292809844017, -0.05561191588640213, -0.2117011398077011, -0.3178425431251526, 0.44138410687446594, 0.10796058923006058, -0.5007132291793823, -0.13645219802856445, -0.10036168992519379, -0.4539477229118347], [-0.027732979506254196, -0.04559260234236717, 0.07316332310438156, 0.07365732640028, -0.11794889718294144, -0.05489973723888397, 0.18127629160881042, -0.07468268275260925, 0.05915010720491409, 0.1412903517484665]]), requires_grad=True)
        self.all_parameters["PLinear41W"] = torch.nn.Parameter(torch.tensor([[0.0008763351361267269, -0.00024157062580343336, -0.00028882664628326893, -0.0007237002719193697], [-0.30842161178588867, -0.0030950887594372034, 0.11811084300279617, 0.0013123834505677223], [0.15039990842342377, 0.0014968308387324214, 0.03533568233251572, 0.0007502621156163514], [-0.26984065771102905, -0.0026179030537605286, -0.05602499470114708, -0.0010904406663030386], [-0.5811452865600586, -0.005795136094093323, 0.18178735673427582, 0.0014936721418052912], [0.0006841715076006949, 0.0001788078952813521, -0.0019367801724001765, -0.0007646018057130277], [-0.2240845263004303, -0.0025831980165094137, 0.030911885201931, 0.0027259401977062225], [0.004272507969290018, -6.568706885445863e-05, -0.065296471118927, -0.00010297287371940911], [-0.0007754236576147377, -0.0004502884694375098, 0.002104794839397073, -4.8566282202955335e-05], [-0.15299534797668457, -0.0011870090384036303, 0.014870685525238514, -0.002811344340443611]]), requires_grad=True)
        self.all_constants["SamplePart1"] = torch.tensor([[1.0]], requires_grad=True)
        self.all_constants["SamplePart10"] = torch.tensor([[1.0]], requires_grad=True)
        self.all_constants["SamplePart15"] = torch.tensor([[1.0]], requires_grad=True)
        self.all_constants["SamplePart17"] = torch.tensor([[1.0]], requires_grad=True)
        self.all_constants["SamplePart173"] = torch.tensor([[1.0]], requires_grad=True)
        self.all_constants["SamplePart22"] = torch.tensor([[1.0]], requires_grad=True)
        self.all_constants["SamplePart24"] = torch.tensor([[1.0]], requires_grad=True)
        self.all_constants["SamplePart298"] = torch.tensor([[1.0]], requires_grad=True)
        self.all_constants["SamplePart3"] = torch.tensor([[1.0]], requires_grad=True)
        self.all_constants["SamplePart36"] = torch.tensor([[1.0]], requires_grad=True)
        self.all_constants["SamplePart415"] = torch.tensor([[1.0]], requires_grad=True)
        self.all_constants["SamplePart48"] = torch.tensor([[1.0]], requires_grad=True)
        self.all_constants["SamplePart580"] = torch.tensor([[1.0]], requires_grad=True)
        self.all_constants["SamplePart582"] = torch.tensor([[1.0]], requires_grad=True)
        self.all_constants["SamplePart584"] = torch.tensor([[1.0]], requires_grad=True)
        self.all_constants["SamplePart586"] = torch.tensor([[1.0]], requires_grad=True)
        self.all_constants["SamplePart8"] = torch.tensor([[1.0]], requires_grad=True)
        self.all_constants["Select531"] = torch.tensor([1.0, 0.0, 0.0, 0.0], requires_grad=True)
        self.all_constants["Select545"] = torch.tensor([0.0, 1.0, 0.0, 0.0], requires_grad=True)
        self.all_constants["Select559"] = torch.tensor([0.0, 0.0, 1.0, 0.0], requires_grad=True)
        self.all_constants["Select573"] = torch.tensor([0.0, 0.0, 0.0, 1.0], requires_grad=True)
        self.all_constants["Select575"] = torch.tensor([0.0, 0.0, 0.0, 1.0], requires_grad=True)
        self.all_constants["Select576"] = torch.tensor([0.0, 0.0, 1.0, 0.0], requires_grad=True)
        self.all_constants["Select577"] = torch.tensor([0.0, 1.0, 0.0, 0.0], requires_grad=True)
        self.all_constants["Select578"] = torch.tensor([1.0, 0.0, 0.0, 0.0], requires_grad=True)
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
        all_parameters_b = self.all_parameters.b
        mul_1 = mul * all_parameters_b;  mul = None
        mul_2 = all_parameters_m2 * all_parameters_l
        mul_3 = all_parameters_m2 * all_parameters_l
        all_parameters_d = self.all_parameters.d
        mul_4 = mul_3 * all_parameters_d;  mul_3 = None
        mul_5 = all_parameters_m2 * all_parameters_l
        mul_6 = add_1 * all_parameters_d;  add_1 = None
        mul_7 = add_2 * all_parameters_m2;  add_2 = None
        all_constants_g = self.all_constants.g
        mul_8 = mul_7 * all_constants_g;  mul_7 = None
        mul_9 = mul_8 * all_parameters_l;  mul_8 = None
        mul_10 = all_parameters_m2 * all_parameters_l
        mul_11 = mul_10 * all_parameters_b;  mul_10 = None
        mul_12 = all_parameters_m2 * all_parameters_l
        mul_13 = all_parameters_m2 * all_parameters_l
        mul_14 = mul_13 * all_parameters_d;  mul_13 = None
        mul_15 = all_parameters_m2 * all_parameters_l
        mul_16 = add_4 * all_parameters_d;  add_4 = None
        mul_17 = add_5 * all_parameters_m2;  add_5 = None
        mul_18 = mul_17 * all_constants_g;  mul_17 = None
        mul_19 = mul_18 * all_parameters_l;  mul_18 = None
        mul_20 = all_parameters_m2 * all_parameters_l
        mul_21 = mul_20 * all_parameters_b;  mul_20 = None
        mul_22 = all_parameters_m2 * all_parameters_l
        mul_23 = all_parameters_m2 * all_parameters_l
        mul_24 = mul_23 * all_parameters_d;  mul_23 = None
        mul_25 = all_parameters_m2 * all_parameters_l
        mul_26 = add_7 * all_parameters_d;  add_7 = None
        mul_27 = add_8 * all_parameters_m2;  add_8 = None
        mul_28 = mul_27 * all_constants_g;  mul_27 = None
        mul_29 = mul_28 * all_parameters_l;  mul_28 = None
        mul_30 = all_parameters_m2 * all_parameters_l
        mul_31 = mul_30 * all_parameters_b;  mul_30 = None
        mul_32 = all_parameters_m2 * all_parameters_l
        mul_33 = all_parameters_m2 * all_parameters_l
        mul_34 = mul_33 * all_parameters_d;  mul_33 = None
        mul_35 = all_parameters_m2 * all_parameters_l
        mul_36 = add_10 * all_parameters_d;  add_10 = all_parameters_d = None
        mul_37 = add_11 * all_parameters_m2;  add_11 = None
        all_constants_constant49 = self.all_constants.Constant49
        pow_1 = torch.pow(all_parameters_m2, all_constants_constant49);  all_constants_constant49 = None
        all_constants_constant50 = self.all_constants.Constant50
        pow_2 = torch.pow(all_parameters_l, all_constants_constant50);  all_constants_constant50 = None
        all_constants_constant55 = self.all_constants.Constant55
        pow_3 = torch.pow(all_parameters_l, all_constants_constant55);  all_constants_constant55 = None
        all_constants_constant57 = self.all_constants.Constant57
        pow_4 = torch.pow(all_parameters_m2, all_constants_constant57);  all_constants_constant57 = None
        all_constants_constant58 = self.all_constants.Constant58
        pow_5 = torch.pow(all_parameters_l, all_constants_constant58);  all_constants_constant58 = None
        all_constants_constant61 = self.all_constants.Constant61
        pow_6 = torch.pow(all_parameters_m2, all_constants_constant61);  all_constants_constant61 = None
        all_constants_constant62 = self.all_constants.Constant62
        pow_7 = torch.pow(all_parameters_l, all_constants_constant62);  all_constants_constant62 = None
        all_constants_constant67 = self.all_constants.Constant67
        pow_8 = torch.pow(all_parameters_l, all_constants_constant67);  all_constants_constant67 = None
        all_constants_constant69 = self.all_constants.Constant69
        pow_9 = torch.pow(all_parameters_m2, all_constants_constant69);  all_constants_constant69 = None
        all_constants_constant70 = self.all_constants.Constant70
        pow_10 = torch.pow(all_parameters_l, all_constants_constant70);  all_constants_constant70 = None
        all_constants_constant73 = self.all_constants.Constant73
        pow_11 = torch.pow(all_parameters_m2, all_constants_constant73);  all_constants_constant73 = None
        all_constants_constant74 = self.all_constants.Constant74
        pow_12 = torch.pow(all_parameters_l, all_constants_constant74);  all_constants_constant74 = None
        all_constants_constant75 = self.all_constants.Constant75
        pow_13 = torch.pow(all_parameters_l, all_constants_constant75);  all_constants_constant75 = None
        all_constants_constant77 = self.all_constants.Constant77
        pow_14 = torch.pow(all_parameters_m2, all_constants_constant77);  all_constants_constant77 = None
        all_constants_constant78 = self.all_constants.Constant78
        pow_15 = torch.pow(all_parameters_l, all_constants_constant78);  all_constants_constant78 = None
        all_constants_constant81 = self.all_constants.Constant81
        pow_16 = torch.pow(all_parameters_m2, all_constants_constant81);  all_constants_constant81 = None
        all_constants_constant82 = self.all_constants.Constant82
        pow_17 = torch.pow(all_parameters_l, all_constants_constant82);  all_constants_constant82 = None
        all_constants_constant43 = self.all_constants.Constant43
        pow_18 = torch.pow(all_parameters_l, all_constants_constant43);  all_constants_constant43 = None
        all_constants_constant45 = self.all_constants.Constant45
        pow_19 = torch.pow(all_parameters_m2, all_constants_constant45);  all_constants_constant45 = None
        all_constants_constant46 = self.all_constants.Constant46
        pow_20 = torch.pow(all_parameters_l, all_constants_constant46);  all_constants_constant46 = None
        getitem = kwargs['Xpos']
        relation_forward_sample_part1_w = self.all_constants.SamplePart1
        einsum = torch.functional.einsum('bij,ki->bkj', getitem, relation_forward_sample_part1_w);  getitem = relation_forward_sample_part1_w = None
        getitem_1 = kwargs['noise1']
        relation_forward_sample_part10_w = self.all_constants.SamplePart10
        einsum_1 = torch.functional.einsum('bij,ki->bkj', getitem_1, relation_forward_sample_part10_w);  getitem_1 = relation_forward_sample_part10_w = None
        getitem_2 = kwargs['Xangular_velocity']
        relation_forward_sample_part15_w = self.all_constants.SamplePart15
        einsum_2 = torch.functional.einsum('bij,ki->bkj', getitem_2, relation_forward_sample_part15_w);  getitem_2 = relation_forward_sample_part15_w = None
        getitem_3 = kwargs['noise3']
        relation_forward_sample_part17_w = self.all_constants.SamplePart17
        einsum_3 = torch.functional.einsum('bij,ki->bkj', getitem_3, relation_forward_sample_part17_w);  getitem_3 = relation_forward_sample_part17_w = None
        getitem_4 = kwargs['action']
        relation_forward_sample_part173_w = self.all_constants.SamplePart173
        einsum_4 = torch.functional.einsum('bij,ki->bkj', getitem_4, relation_forward_sample_part173_w);  getitem_4 = relation_forward_sample_part173_w = None
        getitem_5 = kwargs['Xangle']
        relation_forward_sample_part22_w = self.all_constants.SamplePart22
        einsum_5 = torch.functional.einsum('bij,ki->bkj', getitem_5, relation_forward_sample_part22_w);  getitem_5 = relation_forward_sample_part22_w = None
        getitem_6 = kwargs['noise2']
        relation_forward_sample_part24_w = self.all_constants.SamplePart24
        einsum_6 = torch.functional.einsum('bij,ki->bkj', getitem_6, relation_forward_sample_part24_w);  getitem_6 = relation_forward_sample_part24_w = None
        getitem_7 = kwargs['action']
        relation_forward_sample_part298_w = self.all_constants.SamplePart298
        einsum_7 = torch.functional.einsum('bij,ki->bkj', getitem_7, relation_forward_sample_part298_w);  getitem_7 = relation_forward_sample_part298_w = None
        getitem_8 = kwargs['noise4']
        relation_forward_sample_part3_w = self.all_constants.SamplePart3
        einsum_8 = torch.functional.einsum('bij,ki->bkj', getitem_8, relation_forward_sample_part3_w);  getitem_8 = relation_forward_sample_part3_w = None
        getitem_9 = kwargs['action']
        relation_forward_sample_part36_w = self.all_constants.SamplePart36
        einsum_9 = torch.functional.einsum('bij,ki->bkj', getitem_9, relation_forward_sample_part36_w);  getitem_9 = relation_forward_sample_part36_w = None
        getitem_10 = kwargs['action']
        relation_forward_sample_part415_w = self.all_constants.SamplePart415
        einsum_10 = torch.functional.einsum('bij,ki->bkj', getitem_10, relation_forward_sample_part415_w);  getitem_10 = relation_forward_sample_part415_w = None
        getitem_11 = kwargs['action']
        relation_forward_sample_part48_w = self.all_constants.SamplePart48
        einsum_11 = torch.functional.einsum('bij,ki->bkj', getitem_11, relation_forward_sample_part48_w);  getitem_11 = relation_forward_sample_part48_w = None
        getitem_12 = kwargs['Yangular_velocity']
        relation_forward_sample_part580_w = self.all_constants.SamplePart580
        einsum_12 = torch.functional.einsum('bij,ki->bkj', getitem_12, relation_forward_sample_part580_w);  getitem_12 = relation_forward_sample_part580_w = None
        getitem_13 = kwargs['Yangle']
        relation_forward_sample_part582_w = self.all_constants.SamplePart582
        einsum_13 = torch.functional.einsum('bij,ki->bkj', getitem_13, relation_forward_sample_part582_w);  getitem_13 = relation_forward_sample_part582_w = None
        getitem_14 = kwargs['Yvelocity']
        relation_forward_sample_part584_w = self.all_constants.SamplePart584
        einsum_14 = torch.functional.einsum('bij,ki->bkj', getitem_14, relation_forward_sample_part584_w);  getitem_14 = relation_forward_sample_part584_w = None
        getitem_15 = kwargs['Ypos']
        relation_forward_sample_part586_w = self.all_constants.SamplePart586
        einsum_15 = torch.functional.einsum('bij,ki->bkj', getitem_15, relation_forward_sample_part586_w);  getitem_15 = relation_forward_sample_part586_w = None
        getitem_16 = kwargs['Xvelocity'];  kwargs = None
        relation_forward_sample_part8_w = self.all_constants.SamplePart8
        einsum_16 = torch.functional.einsum('bij,ki->bkj', getitem_16, relation_forward_sample_part8_w);  getitem_16 = relation_forward_sample_part8_w = None
        relation_forward_linear37_weights = self.all_parameters.PLinear37W
        einsum_17 = torch.functional.einsum('bwi,io->bwo', einsum_9, relation_forward_linear37_weights);  einsum_9 = relation_forward_linear37_weights = None
        mul_38 = mul_37 * all_constants_g;  mul_37 = None
        mul_39 = mul_38 * all_parameters_l;  mul_38 = None
        all_constants_sigma_v = self.all_constants.sigma_v
        mul_40 = einsum_1 * all_constants_sigma_v;  einsum_1 = all_constants_sigma_v = None
        mul_41 = pow_1 * pow_2;  pow_1 = pow_2 = None
        mul_42 = mul_41 * all_constants_g;  mul_41 = None
        mul_43 = all_parameters_m2 * pow_3;  pow_3 = None
        all_constants_sigma_omega = self.all_constants.sigma_omega
        mul_44 = einsum_3 * all_constants_sigma_omega;  einsum_3 = all_constants_sigma_omega = None
        all_constants_gear = self.all_constants.gear
        mul_45 = all_constants_gear * einsum_4;  einsum_4 = None
        all_constants_sigma_theta = self.all_constants.sigma_theta
        mul_46 = einsum_6 * all_constants_sigma_theta;  einsum_6 = all_constants_sigma_theta = None
        mul_47 = pow_6 * pow_7;  pow_6 = pow_7 = None
        mul_48 = mul_47 * all_constants_g;  mul_47 = None
        mul_49 = all_parameters_m2 * pow_8;  pow_8 = None
        mul_50 = all_constants_gear * einsum_7;  einsum_7 = None
        mul_51 = pow_11 * pow_12;  pow_11 = pow_12 = None
        mul_52 = mul_51 * all_constants_g;  mul_51 = None
        mul_53 = all_parameters_m2 * pow_13;  pow_13 = None
        mul_54 = all_constants_gear * einsum_10;  einsum_10 = None
        all_constants_sigma_x = self.all_constants.sigma_x
        mul_55 = einsum_8 * all_constants_sigma_x;  einsum_8 = all_constants_sigma_x = None
        mul_56 = pow_16 * pow_17;  pow_16 = pow_17 = None
        mul_57 = mul_56 * all_constants_g;  mul_56 = all_constants_g = None
        mul_58 = all_parameters_m2 * pow_18;  pow_18 = None
        mul_59 = all_constants_gear * einsum_11;  all_constants_gear = einsum_11 = None
        neg = -pow_4;  pow_4 = None
        neg_1 = -pow_9;  pow_9 = None
        neg_2 = -pow_14;  pow_14 = None
        neg_3 = -pow_19;  pow_19 = None
        add_12 = einsum_16 + mul_40;  einsum_16 = mul_40 = None
        all_parameters_i = self.all_parameters.I
        add_13 = all_parameters_i + mul_43;  mul_43 = None
        add_14 = einsum_2 + mul_44;  einsum_2 = mul_44 = None
        add_15 = einsum_5 + mul_46;  einsum_5 = mul_46 = None
        add_16 = all_parameters_i + mul_49;  mul_49 = None
        add_17 = all_parameters_i + mul_53;  mul_53 = None
        add_18 = all_parameters_i + mul_58;  all_parameters_i = mul_58 = None
        add_19 = einsum + mul_55;  einsum = mul_55 = None
        cos = torch.cos(add_15)
        cos_1 = torch.cos(add_15)
        relation_forward_linear28_weights = self.all_parameters.PLinear27W
        einsum_18 = torch.functional.einsum('bwi,io->bwo', add_15, relation_forward_linear28_weights);  relation_forward_linear28_weights = None
        relation_forward_linear29_weights = self.all_parameters.PLinear29W
        einsum_19 = torch.functional.einsum('bwi,io->bwo', add_12, relation_forward_linear29_weights);  relation_forward_linear29_weights = None
        relation_forward_linear33_weights = self.all_parameters.PLinear33W
        einsum_20 = torch.functional.einsum('bwi,io->bwo', cos, relation_forward_linear33_weights);  cos = relation_forward_linear33_weights = None
        relation_forward_linear34_weights = self.all_parameters.PLinear35W
        einsum_21 = torch.functional.einsum('bwi,io->bwo', add_14, relation_forward_linear34_weights);  relation_forward_linear34_weights = None
        mul_60 = mul_1 * add_12;  mul_1 = None
        mul_61 = mul_60 * cos_1;  mul_60 = None
        mul_62 = mul_2 * cos_1;  mul_2 = None
        mul_63 = mul_62 * mul_59;  mul_62 = None
        mul_64 = add_18 * all_parameters_m2
        mul_65 = mul_64 * all_parameters_l;  mul_64 = None
        mul_66 = mul_4 * add_14;  mul_4 = None
        mul_67 = mul_66 * cos_1;  mul_66 = None
        mul_68 = mul_59 * add_18;  mul_59 = None
        all_constants_dt = self.all_constants.dt
        mul_69 = add_14 * all_constants_dt
        mul_70 = add * add_13;  add = None
        mul_71 = add_13 * all_parameters_b
        mul_72 = neg * pow_5;  neg = pow_5 = None
        mul_73 = add_13 * all_parameters_m2
        mul_74 = mul_73 * all_parameters_l;  mul_73 = None
        mul_75 = mul_45 * add_13;  add_13 = None
        mul_76 = add_3 * add_16;  add_3 = None
        mul_77 = add_16 * all_parameters_b
        mul_78 = neg_1 * pow_10;  neg_1 = pow_10 = None
        mul_79 = add_16 * all_parameters_m2
        mul_80 = mul_79 * all_parameters_l;  mul_79 = None
        mul_81 = mul_50 * add_16;  add_16 = None
        mul_82 = add_6 * add_17;  add_6 = None
        mul_83 = add_17 * all_parameters_b
        mul_84 = neg_2 * pow_15;  neg_2 = pow_15 = None
        mul_85 = add_17 * all_parameters_m2;  all_parameters_m2 = None
        mul_86 = mul_85 * all_parameters_l;  mul_85 = all_parameters_l = None
        mul_87 = mul_54 * add_17;  add_17 = None
        mul_88 = add_9 * add_18;  add_9 = None
        mul_89 = mul_35 * cos_1;  mul_35 = None
        mul_90 = add_18 * all_parameters_b;  add_18 = all_parameters_b = None
        mul_91 = mul_90 * add_12;  mul_90 = None
        mul_92 = mul_36 * add_14;  mul_36 = None
        mul_93 = neg_3 * pow_20;  neg_3 = pow_20 = None
        all_constants_constant48 = self.all_constants.Constant48
        pow_21 = torch.pow(add_14, all_constants_constant48);  all_constants_constant48 = None
        all_constants_constant44 = self.all_constants.Constant44
        pow_22 = torch.pow(mul_89, all_constants_constant44);  mul_89 = all_constants_constant44 = None
        all_constants_constant47 = self.all_constants.Constant47
        pow_23 = torch.pow(add_14, all_constants_constant47);  all_constants_constant47 = None
        sin = torch.sin(add_15)
        sin_1 = torch.sin(add_15)
        sub = mul_88 - pow_22;  mul_88 = pow_22 = None
        add_20 = einsum_18 + einsum_19;  einsum_18 = einsum_19 = None
        all_constants_constant53 = self.all_constants.Constant53
        truediv = mul_69 / all_constants_constant53;  mul_69 = all_constants_constant53 = None
        relation_forward_linear31_weights = self.all_parameters.PLinear31W
        einsum_22 = torch.functional.einsum('bwi,io->bwo', sin, relation_forward_linear31_weights);  sin = relation_forward_linear31_weights = None
        mul_94 = mul_39 * sin_1;  mul_39 = None
        mul_95 = mul_65 * pow_21;  mul_65 = pow_21 = None
        mul_96 = mul_95 * sin_1;  mul_95 = None
        mul_97 = mul_42 * sin_1;  mul_42 = None
        mul_98 = mul_97 * cos_1;  mul_97 = None
        mul_99 = mul_93 * pow_23;  mul_93 = pow_23 = None
        mul_100 = mul_99 * sin_1;  mul_99 = sin_1 = None
        mul_101 = mul_100 * cos_1;  mul_100 = cos_1 = None
        sub_1 = mul_96 - mul_91;  mul_96 = mul_91 = None
        sub_2 = sub_1 - mul_98;  sub_1 = mul_98 = None
        sub_3 = mul_101 - mul_92;  mul_101 = mul_92 = None
        add_21 = sub_3 + mul_94;  sub_3 = mul_94 = None
        add_22 = add_21 + mul_61;  add_21 = mul_61 = None
        add_23 = sub_2 + mul_67;  sub_2 = mul_67 = None
        add_24 = add_23 + mul_68;  add_23 = mul_68 = None
        add_25 = add_15 + truediv;  truediv = None
        add_26 = add_20 + einsum_22;  add_20 = einsum_22 = None
        add_27 = add_26 + einsum_20;  add_26 = einsum_20 = None
        add_28 = add_27 + einsum_21;  add_27 = einsum_21 = None
        add_29 = add_28 + einsum_17;  add_28 = einsum_17 = None
        cos_2 = torch.cos(add_25)
        truediv_1 = add_24 / sub;  add_24 = None
        mul_102 = truediv_1 * all_constants_dt
        mul_103 = mul_5 * cos_2;  mul_5 = None
        mul_104 = mul_12 * cos_2;  mul_12 = None
        mul_105 = mul_104 * mul_45;  mul_104 = mul_45 = None
        all_constants_constant56 = self.all_constants.Constant56
        pow_24 = torch.pow(mul_103, all_constants_constant56);  mul_103 = all_constants_constant56 = None
        sin_2 = torch.sin(add_25);  add_25 = None
        sub_4 = add_22 - mul_63;  add_22 = mul_63 = None
        sub_5 = mul_70 - pow_24;  mul_70 = pow_24 = None
        tanh = torch.tanh(add_29);  add_29 = None
        truediv_2 = sub_4 / sub;  sub_4 = sub = None
        all_constants_constant52 = self.all_constants.Constant52
        truediv_3 = mul_102 / all_constants_constant52;  mul_102 = all_constants_constant52 = None
        relation_forward_linear44_weights = self.all_parameters.PLinear39W
        einsum_23 = torch.functional.einsum('bwi,io->bwo', tanh, relation_forward_linear44_weights);  tanh = relation_forward_linear44_weights = None
        mul_106 = truediv_2 * all_constants_dt
        mul_107 = mul_9 * sin_2;  mul_9 = None
        mul_108 = mul_48 * sin_2;  mul_48 = None
        mul_109 = mul_108 * cos_2;  mul_108 = None
        tanh_1 = torch.tanh(einsum_23);  einsum_23 = None
        add_30 = add_12 + truediv_3;  truediv_3 = None
        all_constants_constant54 = self.all_constants.Constant54
        truediv_4 = mul_106 / all_constants_constant54;  mul_106 = all_constants_constant54 = None
        relation_forward_linear46_weights = self.all_parameters.PLinear41W
        einsum_24 = torch.functional.einsum('bwi,io->bwo', tanh_1, relation_forward_linear46_weights);  tanh_1 = relation_forward_linear46_weights = None
        mul_110 = mul_71 * add_30;  mul_71 = None
        mul_111 = mul_11 * add_30;  mul_11 = None
        mul_112 = mul_111 * cos_2;  mul_111 = None
        all_constants_constant86 = self.all_constants.Constant86
        mul_113 = all_constants_constant86 * add_30;  all_constants_constant86 = add_30 = None
        relation_forward_select531_w = self.all_constants.Select531
        einsum_25 = torch.functional.einsum('ijk,k->ij', einsum_24, relation_forward_select531_w);  relation_forward_select531_w = None
        unsqueeze = einsum_25.unsqueeze(2);  einsum_25 = None
        relation_forward_select545_w = self.all_constants.Select545
        einsum_26 = torch.functional.einsum('ijk,k->ij', einsum_24, relation_forward_select545_w);  relation_forward_select545_w = None
        unsqueeze_1 = einsum_26.unsqueeze(2);  einsum_26 = None
        relation_forward_select559_w = self.all_constants.Select559
        einsum_27 = torch.functional.einsum('ijk,k->ij', einsum_24, relation_forward_select559_w);  relation_forward_select559_w = None
        unsqueeze_2 = einsum_27.unsqueeze(2);  einsum_27 = None
        relation_forward_select573_w = self.all_constants.Select573
        einsum_28 = torch.functional.einsum('ijk,k->ij', einsum_24, relation_forward_select573_w);  relation_forward_select573_w = None
        unsqueeze_3 = einsum_28.unsqueeze(2);  einsum_28 = None
        relation_forward_select575_w = self.all_constants.Select575
        einsum_29 = torch.functional.einsum('ijk,k->ij', einsum_24, relation_forward_select575_w);  relation_forward_select575_w = None
        unsqueeze_4 = einsum_29.unsqueeze(2);  einsum_29 = None
        relation_forward_select576_w = self.all_constants.Select576
        einsum_30 = torch.functional.einsum('ijk,k->ij', einsum_24, relation_forward_select576_w);  relation_forward_select576_w = None
        unsqueeze_5 = einsum_30.unsqueeze(2);  einsum_30 = None
        relation_forward_select577_w = self.all_constants.Select577
        einsum_31 = torch.functional.einsum('ijk,k->ij', einsum_24, relation_forward_select577_w);  relation_forward_select577_w = None
        unsqueeze_6 = einsum_31.unsqueeze(2);  einsum_31 = None
        relation_forward_select578_w = self.all_constants.Select578
        einsum_32 = torch.functional.einsum('ijk,k->ij', einsum_24, relation_forward_select578_w);  einsum_24 = relation_forward_select578_w = None
        unsqueeze_7 = einsum_32.unsqueeze(2);  einsum_32 = None
        add_31 = add_14 + truediv_4;  truediv_4 = None
        add_32 = add_12 + mul_113;  mul_113 = None
        mul_114 = mul_6 * add_31;  mul_6 = None
        mul_115 = mul_14 * add_31;  mul_14 = None
        mul_116 = mul_115 * cos_2;  mul_115 = None
        mul_117 = add_31 * all_constants_dt
        all_constants_constant92 = self.all_constants.Constant92
        mul_118 = all_constants_constant92 * add_31;  all_constants_constant92 = None
        all_constants_constant59 = self.all_constants.Constant59
        pow_25 = torch.pow(add_31, all_constants_constant59);  all_constants_constant59 = None
        all_constants_constant60 = self.all_constants.Constant60
        pow_26 = torch.pow(add_31, all_constants_constant60);  add_31 = all_constants_constant60 = None
        add_33 = add_14 + mul_118;  mul_118 = None
        all_constants_constant65 = self.all_constants.Constant65
        truediv_5 = mul_117 / all_constants_constant65;  mul_117 = all_constants_constant65 = None
        mul_119 = mul_72 * pow_25;  mul_72 = pow_25 = None
        mul_120 = mul_119 * sin_2;  mul_119 = None
        mul_121 = mul_120 * cos_2;  mul_120 = cos_2 = None
        mul_122 = mul_74 * pow_26;  mul_74 = pow_26 = None
        mul_123 = mul_122 * sin_2;  mul_122 = sin_2 = None
        sub_6 = mul_121 - mul_114;  mul_121 = mul_114 = None
        sub_7 = mul_123 - mul_110;  mul_123 = mul_110 = None
        sub_8 = sub_7 - mul_109;  sub_7 = mul_109 = None
        add_34 = sub_6 + mul_107;  sub_6 = mul_107 = None
        add_35 = add_34 + mul_112;  add_34 = mul_112 = None
        add_36 = sub_8 + mul_116;  sub_8 = mul_116 = None
        add_37 = add_36 + mul_75;  add_36 = mul_75 = None
        add_38 = add_15 + truediv_5;  truediv_5 = None
        cos_3 = torch.cos(add_38)
        truediv_6 = add_37 / sub_5;  add_37 = None
        mul_124 = truediv_6 * all_constants_dt
        mul_125 = mul_15 * cos_3;  mul_15 = None
        mul_126 = mul_22 * cos_3;  mul_22 = None
        mul_127 = mul_126 * mul_50;  mul_126 = mul_50 = None
        all_constants_constant83 = self.all_constants.Constant83
        mul_128 = all_constants_constant83 * truediv_6;  all_constants_constant83 = truediv_6 = None
        all_constants_constant68 = self.all_constants.Constant68
        pow_27 = torch.pow(mul_125, all_constants_constant68);  mul_125 = all_constants_constant68 = None
        sin_3 = torch.sin(add_38);  add_38 = None
        sub_9 = add_35 - mul_105;  add_35 = mul_105 = None
        sub_10 = mul_76 - pow_27;  mul_76 = pow_27 = None
        add_39 = truediv_1 + mul_128;  truediv_1 = mul_128 = None
        truediv_7 = sub_9 / sub_5;  sub_9 = sub_5 = None
        all_constants_constant64 = self.all_constants.Constant64
        truediv_8 = mul_124 / all_constants_constant64;  mul_124 = all_constants_constant64 = None
        mul_129 = truediv_7 * all_constants_dt
        mul_130 = mul_19 * sin_3;  mul_19 = None
        mul_131 = mul_52 * sin_3;  mul_52 = None
        mul_132 = mul_131 * cos_3;  mul_131 = None
        all_constants_constant89 = self.all_constants.Constant89
        mul_133 = all_constants_constant89 * truediv_7;  all_constants_constant89 = truediv_7 = None
        add_40 = add_12 + truediv_8;  truediv_8 = None
        add_41 = truediv_2 + mul_133;  truediv_2 = mul_133 = None
        all_constants_constant66 = self.all_constants.Constant66
        truediv_9 = mul_129 / all_constants_constant66;  mul_129 = all_constants_constant66 = None
        mul_134 = mul_77 * add_40;  mul_77 = None
        mul_135 = mul_21 * add_40;  mul_21 = None
        mul_136 = mul_135 * cos_3;  mul_135 = None
        all_constants_constant87 = self.all_constants.Constant87
        mul_137 = all_constants_constant87 * add_40;  all_constants_constant87 = add_40 = None
        add_42 = add_14 + truediv_9;  truediv_9 = None
        add_43 = add_32 + mul_137;  add_32 = mul_137 = None
        mul_138 = mul_16 * add_42;  mul_16 = None
        mul_139 = mul_24 * add_42;  mul_24 = None
        mul_140 = mul_139 * cos_3;  mul_139 = None
        mul_141 = add_42 * all_constants_dt
        all_constants_constant93 = self.all_constants.Constant93
        mul_142 = all_constants_constant93 * add_42;  all_constants_constant93 = None
        all_constants_constant71 = self.all_constants.Constant71
        pow_28 = torch.pow(add_42, all_constants_constant71);  all_constants_constant71 = None
        all_constants_constant72 = self.all_constants.Constant72
        pow_29 = torch.pow(add_42, all_constants_constant72);  add_42 = all_constants_constant72 = None
        add_44 = add_15 + mul_141;  mul_141 = None
        add_45 = add_33 + mul_142;  add_33 = mul_142 = None
        cos_4 = torch.cos(add_44)
        mul_143 = mul_78 * pow_28;  mul_78 = pow_28 = None
        mul_144 = mul_143 * sin_3;  mul_143 = None
        mul_145 = mul_144 * cos_3;  mul_144 = cos_3 = None
        mul_146 = mul_80 * pow_29;  mul_80 = pow_29 = None
        mul_147 = mul_146 * sin_3;  mul_146 = sin_3 = None
        mul_148 = mul_25 * cos_4;  mul_25 = None
        mul_149 = mul_32 * cos_4;  mul_32 = None
        mul_150 = mul_149 * mul_54;  mul_149 = mul_54 = None
        all_constants_constant76 = self.all_constants.Constant76
        pow_30 = torch.pow(mul_148, all_constants_constant76);  mul_148 = all_constants_constant76 = None
        sin_4 = torch.sin(add_44);  add_44 = None
        sub_11 = mul_145 - mul_138;  mul_145 = mul_138 = None
        sub_12 = mul_147 - mul_134;  mul_147 = mul_134 = None
        sub_13 = sub_12 - mul_132;  sub_12 = mul_132 = None
        sub_14 = mul_82 - pow_30;  mul_82 = pow_30 = None
        add_46 = sub_11 + mul_130;  sub_11 = mul_130 = None
        add_47 = add_46 + mul_136;  add_46 = mul_136 = None
        add_48 = sub_13 + mul_140;  sub_13 = mul_140 = None
        add_49 = add_48 + mul_81;  add_48 = mul_81 = None
        truediv_10 = add_49 / sub_10;  add_49 = None
        mul_151 = truediv_10 * all_constants_dt
        mul_152 = mul_29 * sin_4;  mul_29 = None
        mul_153 = mul_57 * sin_4;  mul_57 = None
        mul_154 = mul_153 * cos_4;  mul_153 = None
        all_constants_constant84 = self.all_constants.Constant84
        mul_155 = all_constants_constant84 * truediv_10;  all_constants_constant84 = truediv_10 = None
        sub_15 = add_47 - mul_127;  add_47 = mul_127 = None
        add_50 = add_12 + mul_151;  mul_151 = None
        add_51 = add_39 + mul_155;  add_39 = mul_155 = None
        add_52 = add_43 + add_50;  add_43 = None
        truediv_11 = sub_15 / sub_10;  sub_15 = sub_10 = None
        mul_156 = truediv_11 * all_constants_dt
        mul_157 = mul_83 * add_50;  mul_83 = None
        mul_158 = mul_31 * add_50;  mul_31 = add_50 = None
        mul_159 = mul_158 * cos_4;  mul_158 = None
        mul_160 = add_52 * all_constants_dt;  add_52 = None
        all_constants_constant90 = self.all_constants.Constant90
        mul_161 = all_constants_constant90 * truediv_11;  all_constants_constant90 = truediv_11 = None
        add_53 = add_14 + mul_156;  mul_156 = None
        add_54 = add_41 + mul_161;  add_41 = mul_161 = None
        add_55 = add_45 + add_53;  add_45 = None
        all_constants_constant88 = self.all_constants.Constant88
        truediv_12 = mul_160 / all_constants_constant88;  mul_160 = all_constants_constant88 = None
        mul_162 = mul_26 * add_53;  mul_26 = None
        mul_163 = mul_34 * add_53;  mul_34 = None
        mul_164 = mul_163 * cos_4;  mul_163 = None
        mul_165 = add_55 * all_constants_dt;  add_55 = None
        all_constants_constant79 = self.all_constants.Constant79
        pow_31 = torch.pow(add_53, all_constants_constant79);  all_constants_constant79 = None
        all_constants_constant80 = self.all_constants.Constant80
        pow_32 = torch.pow(add_53, all_constants_constant80);  add_53 = all_constants_constant80 = None
        add_56 = add_19 + truediv_12;  add_19 = truediv_12 = None
        add_57 = add_56 + unsqueeze_1;  add_56 = unsqueeze_1 = None
        all_constants_constant94 = self.all_constants.Constant94
        truediv_13 = mul_165 / all_constants_constant94;  mul_165 = all_constants_constant94 = None
        mul_166 = mul_84 * pow_31;  mul_84 = pow_31 = None
        mul_167 = mul_166 * sin_4;  mul_166 = None
        mul_168 = mul_167 * cos_4;  mul_167 = cos_4 = None
        mul_169 = mul_86 * pow_32;  mul_86 = pow_32 = None
        mul_170 = mul_169 * sin_4;  mul_169 = sin_4 = None
        sub_16 = mul_168 - mul_162;  mul_168 = mul_162 = None
        sub_17 = mul_170 - mul_157;  mul_170 = mul_157 = None
        sub_18 = sub_17 - mul_154;  sub_17 = mul_154 = None
        add_58 = sub_16 + mul_152;  sub_16 = mul_152 = None
        add_59 = add_58 + mul_159;  add_58 = mul_159 = None
        add_60 = sub_18 + mul_164;  sub_18 = mul_164 = None
        add_61 = add_60 + mul_87;  add_60 = mul_87 = None
        add_62 = add_15 + truediv_13;  add_15 = truediv_13 = None
        add_63 = add_62 + unsqueeze_3;  add_62 = unsqueeze_3 = None
        truediv_14 = add_61 / sub_14;  add_61 = None
        sub_19 = add_59 - mul_150;  add_59 = mul_150 = None
        add_64 = add_51 + truediv_14;  add_51 = truediv_14 = None
        truediv_15 = sub_19 / sub_14;  sub_19 = sub_14 = None
        mul_171 = add_64 * all_constants_dt;  add_64 = None
        add_65 = add_54 + truediv_15;  add_54 = truediv_15 = None
        all_constants_constant85 = self.all_constants.Constant85
        truediv_16 = mul_171 / all_constants_constant85;  mul_171 = all_constants_constant85 = None
        mul_172 = add_65 * all_constants_dt;  add_65 = all_constants_dt = None
        add_66 = add_12 + truediv_16;  add_12 = truediv_16 = None
        add_67 = add_66 + unsqueeze;  add_66 = unsqueeze = None
        all_constants_constant91 = self.all_constants.Constant91
        truediv_17 = mul_172 / all_constants_constant91;  mul_172 = all_constants_constant91 = None
        add_68 = add_14 + truediv_17;  add_14 = truediv_17 = None
        add_69 = add_68 + unsqueeze_2;  add_68 = unsqueeze_2 = None
        return ({'contr_theta': unsqueeze_4, 'contr_thetadot': unsqueeze_5, 'contr_x': unsqueeze_7, 'contr_xdot': unsqueeze_6, 'est_theta': add_63, 'est_thetadot': add_69, 'est_x': add_57, 'est_xdot': add_67}, {'SamplePart580': einsum_12, 'SamplePart582': einsum_13, 'SamplePart586': einsum_15, 'SamplePart584': einsum_14, 'Add560': add_69, 'Add574': add_63, 'Add546': add_57, 'Add532': add_67}, {'Xangle': add_63, 'Xangular_velocity': add_69, 'Xpos': add_57, 'Xvelocity': add_67}, {})
        
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
        results = {'contr_theta':[], 'contr_thetadot':[], 'contr_x':[], 'contr_xdot':[], 'est_theta':[], 'est_thetadot':[], 'est_x':[], 'est_xdot':[], }
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
