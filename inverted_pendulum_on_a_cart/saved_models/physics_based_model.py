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
        self.all_constants["Constant27"] = torch.tensor([2.0], requires_grad=False)
        self.all_constants["Constant28"] = torch.tensor([2.0], requires_grad=False)
        self.all_constants["Constant29"] = torch.tensor([2.0], requires_grad=False)
        self.all_constants["Constant30"] = torch.tensor([2.0], requires_grad=False)
        self.all_constants["Constant31"] = torch.tensor([2.0], requires_grad=False)
        self.all_constants["Constant32"] = torch.tensor([2.0], requires_grad=False)
        self.all_constants["Constant33"] = torch.tensor([2.0], requires_grad=False)
        self.all_constants["Constant34"] = torch.tensor([2.0], requires_grad=False)
        self.all_constants["Constant36"] = torch.tensor([2.0], requires_grad=False)
        self.all_constants["Constant37"] = torch.tensor([2.0], requires_grad=False)
        self.all_constants["Constant38"] = torch.tensor([2.0], requires_grad=False)
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
        self.all_constants["Constant59"] = torch.tensor([2.0], requires_grad=False)
        self.all_constants["Constant60"] = torch.tensor([2.0], requires_grad=False)
        self.all_constants["Constant61"] = torch.tensor([2.0], requires_grad=False)
        self.all_constants["Constant62"] = torch.tensor([2.0], requires_grad=False)
        self.all_constants["Constant63"] = torch.tensor([2.0], requires_grad=False)
        self.all_constants["Constant64"] = torch.tensor([2.0], requires_grad=False)
        self.all_constants["Constant65"] = torch.tensor([2.0], requires_grad=False)
        self.all_constants["Constant66"] = torch.tensor([2.0], requires_grad=False)
        self.all_constants["Constant67"] = torch.tensor([2.0], requires_grad=False)
        self.all_constants["Constant68"] = torch.tensor([2.0], requires_grad=False)
        self.all_constants["Constant69"] = torch.tensor([6.0], requires_grad=False)
        self.all_constants["Constant70"] = torch.tensor([2.0], requires_grad=False)
        self.all_constants["Constant71"] = torch.tensor([2.0], requires_grad=False)
        self.all_constants["Constant72"] = torch.tensor([6.0], requires_grad=False)
        self.all_constants["Constant73"] = torch.tensor([2.0], requires_grad=False)
        self.all_constants["Constant74"] = torch.tensor([2.0], requires_grad=False)
        self.all_constants["Constant75"] = torch.tensor([6.0], requires_grad=False)
        self.all_constants["Constant76"] = torch.tensor([2.0], requires_grad=False)
        self.all_constants["Constant77"] = torch.tensor([2.0], requires_grad=False)
        self.all_constants["Constant78"] = torch.tensor([6.0], requires_grad=False)
        self.all_constants["dt"] = torch.tensor([0.019999999552965164], requires_grad=False)
        self.all_constants["g"] = torch.tensor([9.8100004196167], requires_grad=False)
        self.all_constants["gear"] = torch.tensor([100.0], requires_grad=False)
        self.all_constants["sigma_omega"] = torch.tensor([0.0], requires_grad=False)
        self.all_constants["sigma_theta"] = torch.tensor([0.0], requires_grad=False)
        self.all_constants["sigma_v"] = torch.tensor([0.0], requires_grad=False)
        self.all_constants["sigma_x"] = torch.tensor([0.0], requires_grad=False)
        self.all_parameters["I"] = torch.nn.Parameter(torch.tensor([0.33061733841896057]), requires_grad=True)
        self.all_parameters["b"] = torch.nn.Parameter(torch.tensor([0.9297953844070435]), requires_grad=True)
        self.all_parameters["d"] = torch.nn.Parameter(torch.tensor([1.0003730058670044]), requires_grad=True)
        self.all_parameters["l"] = torch.nn.Parameter(torch.tensor([0.20579937100410461]), requires_grad=True)
        self.all_parameters["m1"] = torch.nn.Parameter(torch.tensor([8.182073593139648]), requires_grad=True)
        self.all_parameters["m2"] = torch.nn.Parameter(torch.tensor([7.310985088348389]), requires_grad=True)
        self.all_constants["SamplePart1"] = torch.tensor([[1.0]], requires_grad=True)
        self.all_constants["SamplePart10"] = torch.tensor([[1.0]], requires_grad=True)
        self.all_constants["SamplePart15"] = torch.tensor([[1.0]], requires_grad=True)
        self.all_constants["SamplePart154"] = torch.tensor([[1.0]], requires_grad=True)
        self.all_constants["SamplePart17"] = torch.tensor([[1.0]], requires_grad=True)
        self.all_constants["SamplePart22"] = torch.tensor([[1.0]], requires_grad=True)
        self.all_constants["SamplePart24"] = torch.tensor([[1.0]], requires_grad=True)
        self.all_constants["SamplePart279"] = torch.tensor([[1.0]], requires_grad=True)
        self.all_constants["SamplePart29"] = torch.tensor([[1.0]], requires_grad=True)
        self.all_constants["SamplePart3"] = torch.tensor([[1.0]], requires_grad=True)
        self.all_constants["SamplePart396"] = torch.tensor([[1.0]], requires_grad=True)
        self.all_constants["SamplePart549"] = torch.tensor([[1.0]], requires_grad=True)
        self.all_constants["SamplePart551"] = torch.tensor([[1.0]], requires_grad=True)
        self.all_constants["SamplePart553"] = torch.tensor([[1.0]], requires_grad=True)
        self.all_constants["SamplePart555"] = torch.tensor([[1.0]], requires_grad=True)
        self.all_constants["SamplePart8"] = torch.tensor([[1.0]], requires_grad=True)
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
        all_parameters_d = self.all_parameters.d
        mul_1 = mul * all_parameters_d;  mul = None
        mul_2 = all_parameters_m2 * all_parameters_l
        mul_3 = add_1 * all_parameters_d;  add_1 = None
        mul_4 = add_2 * all_parameters_m2;  add_2 = None
        all_constants_g = self.all_constants.g
        mul_5 = mul_4 * all_constants_g;  mul_4 = None
        mul_6 = mul_5 * all_parameters_l;  mul_5 = None
        mul_7 = all_parameters_m2 * all_parameters_l
        all_parameters_b = self.all_parameters.b
        mul_8 = mul_7 * all_parameters_b;  mul_7 = None
        mul_9 = all_parameters_m2 * all_parameters_l
        mul_10 = all_parameters_m2 * all_parameters_l
        mul_11 = mul_10 * all_parameters_d;  mul_10 = None
        mul_12 = all_parameters_m2 * all_parameters_l
        mul_13 = add_4 * all_parameters_d;  add_4 = None
        mul_14 = add_5 * all_parameters_m2;  add_5 = None
        mul_15 = mul_14 * all_constants_g;  mul_14 = None
        mul_16 = mul_15 * all_parameters_l;  mul_15 = None
        mul_17 = all_parameters_m2 * all_parameters_l
        mul_18 = mul_17 * all_parameters_b;  mul_17 = None
        mul_19 = all_parameters_m2 * all_parameters_l
        mul_20 = all_parameters_m2 * all_parameters_l
        mul_21 = mul_20 * all_parameters_d;  mul_20 = None
        mul_22 = all_parameters_m2 * all_parameters_l
        mul_23 = add_8 * all_parameters_d;  add_8 = None
        mul_24 = add_9 * all_parameters_m2;  add_9 = None
        mul_25 = mul_24 * all_constants_g;  mul_24 = None
        mul_26 = all_parameters_m2 * all_parameters_l
        mul_27 = mul_25 * all_parameters_l;  mul_25 = None
        mul_28 = all_parameters_m2 * all_parameters_l
        mul_29 = mul_28 * all_parameters_b;  mul_28 = None
        mul_30 = all_parameters_m2 * all_parameters_l
        mul_31 = all_parameters_m2 * all_parameters_l
        mul_32 = mul_31 * all_parameters_d;  mul_31 = None
        mul_33 = add_10 * all_parameters_d;  add_10 = all_parameters_d = None
        mul_34 = add_11 * all_parameters_m2;  add_11 = None
        mul_35 = mul_34 * all_constants_g;  mul_34 = None
        mul_36 = mul_35 * all_parameters_l;  mul_35 = None
        mul_37 = all_parameters_m2 * all_parameters_l
        mul_38 = mul_37 * all_parameters_b;  mul_37 = None
        mul_39 = all_parameters_m2 * all_parameters_l
        all_constants_constant33 = self.all_constants.Constant33
        pow_1 = torch.pow(all_parameters_m2, all_constants_constant33);  all_constants_constant33 = None
        all_constants_constant34 = self.all_constants.Constant34
        pow_2 = torch.pow(all_parameters_l, all_constants_constant34);  all_constants_constant34 = None
        all_constants_constant39 = self.all_constants.Constant39
        pow_3 = torch.pow(all_parameters_l, all_constants_constant39);  all_constants_constant39 = None
        all_constants_constant41 = self.all_constants.Constant41
        pow_4 = torch.pow(all_parameters_m2, all_constants_constant41);  all_constants_constant41 = None
        all_constants_constant42 = self.all_constants.Constant42
        pow_5 = torch.pow(all_parameters_l, all_constants_constant42);  all_constants_constant42 = None
        all_constants_constant45 = self.all_constants.Constant45
        pow_6 = torch.pow(all_parameters_m2, all_constants_constant45);  all_constants_constant45 = None
        all_constants_constant46 = self.all_constants.Constant46
        pow_7 = torch.pow(all_parameters_l, all_constants_constant46);  all_constants_constant46 = None
        all_constants_constant51 = self.all_constants.Constant51
        pow_8 = torch.pow(all_parameters_l, all_constants_constant51);  all_constants_constant51 = None
        all_constants_constant53 = self.all_constants.Constant53
        pow_9 = torch.pow(all_parameters_m2, all_constants_constant53);  all_constants_constant53 = None
        all_constants_constant54 = self.all_constants.Constant54
        pow_10 = torch.pow(all_parameters_l, all_constants_constant54);  all_constants_constant54 = None
        all_constants_constant27 = self.all_constants.Constant27
        pow_11 = torch.pow(all_parameters_l, all_constants_constant27);  all_constants_constant27 = None
        all_constants_constant57 = self.all_constants.Constant57
        pow_12 = torch.pow(all_parameters_m2, all_constants_constant57);  all_constants_constant57 = None
        all_constants_constant58 = self.all_constants.Constant58
        pow_13 = torch.pow(all_parameters_l, all_constants_constant58);  all_constants_constant58 = None
        all_constants_constant59 = self.all_constants.Constant59
        pow_14 = torch.pow(all_parameters_l, all_constants_constant59);  all_constants_constant59 = None
        all_constants_constant61 = self.all_constants.Constant61
        pow_15 = torch.pow(all_parameters_m2, all_constants_constant61);  all_constants_constant61 = None
        all_constants_constant62 = self.all_constants.Constant62
        pow_16 = torch.pow(all_parameters_l, all_constants_constant62);  all_constants_constant62 = None
        all_constants_constant65 = self.all_constants.Constant65
        pow_17 = torch.pow(all_parameters_m2, all_constants_constant65);  all_constants_constant65 = None
        all_constants_constant66 = self.all_constants.Constant66
        pow_18 = torch.pow(all_parameters_l, all_constants_constant66);  all_constants_constant66 = None
        all_constants_constant29 = self.all_constants.Constant29
        pow_19 = torch.pow(all_parameters_m2, all_constants_constant29);  all_constants_constant29 = None
        all_constants_constant30 = self.all_constants.Constant30
        pow_20 = torch.pow(all_parameters_l, all_constants_constant30);  all_constants_constant30 = None
        getitem = kwargs['Xpos']
        relation_forward_sample_part1_w = self.all_constants.SamplePart1
        einsum = torch.functional.einsum('bij,ki->bkj', getitem, relation_forward_sample_part1_w);  getitem = relation_forward_sample_part1_w = None
        getitem_1 = kwargs['noise1']
        relation_forward_sample_part10_w = self.all_constants.SamplePart10
        einsum_1 = torch.functional.einsum('bij,ki->bkj', getitem_1, relation_forward_sample_part10_w);  getitem_1 = relation_forward_sample_part10_w = None
        getitem_2 = kwargs['Xangular_velocity']
        relation_forward_sample_part15_w = self.all_constants.SamplePart15
        einsum_2 = torch.functional.einsum('bij,ki->bkj', getitem_2, relation_forward_sample_part15_w);  getitem_2 = relation_forward_sample_part15_w = None
        getitem_3 = kwargs['action']
        relation_forward_sample_part154_w = self.all_constants.SamplePart154
        einsum_3 = torch.functional.einsum('bij,ki->bkj', getitem_3, relation_forward_sample_part154_w);  getitem_3 = relation_forward_sample_part154_w = None
        getitem_4 = kwargs['noise3']
        relation_forward_sample_part17_w = self.all_constants.SamplePart17
        einsum_4 = torch.functional.einsum('bij,ki->bkj', getitem_4, relation_forward_sample_part17_w);  getitem_4 = relation_forward_sample_part17_w = None
        getitem_5 = kwargs['Xangle']
        relation_forward_sample_part22_w = self.all_constants.SamplePart22
        einsum_5 = torch.functional.einsum('bij,ki->bkj', getitem_5, relation_forward_sample_part22_w);  getitem_5 = relation_forward_sample_part22_w = None
        getitem_6 = kwargs['noise2']
        relation_forward_sample_part24_w = self.all_constants.SamplePart24
        einsum_6 = torch.functional.einsum('bij,ki->bkj', getitem_6, relation_forward_sample_part24_w);  getitem_6 = relation_forward_sample_part24_w = None
        getitem_7 = kwargs['action']
        relation_forward_sample_part279_w = self.all_constants.SamplePart279
        einsum_7 = torch.functional.einsum('bij,ki->bkj', getitem_7, relation_forward_sample_part279_w);  getitem_7 = relation_forward_sample_part279_w = None
        getitem_8 = kwargs['action']
        relation_forward_sample_part29_w = self.all_constants.SamplePart29
        einsum_8 = torch.functional.einsum('bij,ki->bkj', getitem_8, relation_forward_sample_part29_w);  getitem_8 = relation_forward_sample_part29_w = None
        getitem_9 = kwargs['noise4']
        relation_forward_sample_part3_w = self.all_constants.SamplePart3
        einsum_9 = torch.functional.einsum('bij,ki->bkj', getitem_9, relation_forward_sample_part3_w);  getitem_9 = relation_forward_sample_part3_w = None
        getitem_10 = kwargs['action']
        relation_forward_sample_part396_w = self.all_constants.SamplePart396
        einsum_10 = torch.functional.einsum('bij,ki->bkj', getitem_10, relation_forward_sample_part396_w);  getitem_10 = relation_forward_sample_part396_w = None
        getitem_11 = kwargs['Yangular_velocity']
        relation_forward_sample_part549_w = self.all_constants.SamplePart549
        einsum_11 = torch.functional.einsum('bij,ki->bkj', getitem_11, relation_forward_sample_part549_w);  getitem_11 = relation_forward_sample_part549_w = None
        getitem_12 = kwargs['Yangle']
        relation_forward_sample_part551_w = self.all_constants.SamplePart551
        einsum_12 = torch.functional.einsum('bij,ki->bkj', getitem_12, relation_forward_sample_part551_w);  getitem_12 = relation_forward_sample_part551_w = None
        getitem_13 = kwargs['Yvelocity']
        relation_forward_sample_part553_w = self.all_constants.SamplePart553
        einsum_13 = torch.functional.einsum('bij,ki->bkj', getitem_13, relation_forward_sample_part553_w);  getitem_13 = relation_forward_sample_part553_w = None
        getitem_14 = kwargs['Ypos']
        relation_forward_sample_part555_w = self.all_constants.SamplePart555
        einsum_14 = torch.functional.einsum('bij,ki->bkj', getitem_14, relation_forward_sample_part555_w);  getitem_14 = relation_forward_sample_part555_w = None
        getitem_15 = kwargs['Xvelocity'];  kwargs = None
        relation_forward_sample_part8_w = self.all_constants.SamplePart8
        einsum_15 = torch.functional.einsum('bij,ki->bkj', getitem_15, relation_forward_sample_part8_w);  getitem_15 = relation_forward_sample_part8_w = None
        mul_40 = pow_1 * pow_2;  pow_1 = pow_2 = None
        mul_41 = mul_40 * all_constants_g;  mul_40 = None
        all_constants_sigma_v = self.all_constants.sigma_v
        mul_42 = einsum_1 * all_constants_sigma_v;  einsum_1 = all_constants_sigma_v = None
        mul_43 = all_parameters_m2 * pow_3;  pow_3 = None
        all_constants_gear = self.all_constants.gear
        mul_44 = all_constants_gear * einsum_3;  einsum_3 = None
        all_constants_sigma_omega = self.all_constants.sigma_omega
        mul_45 = einsum_4 * all_constants_sigma_omega;  einsum_4 = all_constants_sigma_omega = None
        mul_46 = pow_6 * pow_7;  pow_6 = pow_7 = None
        mul_47 = mul_46 * all_constants_g;  mul_46 = None
        all_constants_sigma_theta = self.all_constants.sigma_theta
        mul_48 = einsum_6 * all_constants_sigma_theta;  einsum_6 = all_constants_sigma_theta = None
        mul_49 = all_parameters_m2 * pow_8;  pow_8 = None
        mul_50 = all_constants_gear * einsum_7;  einsum_7 = None
        mul_51 = all_parameters_m2 * pow_11;  pow_11 = None
        mul_52 = pow_12 * pow_13;  pow_12 = pow_13 = None
        mul_53 = mul_52 * all_constants_g;  mul_52 = None
        mul_54 = all_parameters_m2 * pow_14;  pow_14 = None
        mul_55 = all_constants_gear * einsum_10;  einsum_10 = None
        mul_56 = pow_17 * pow_18;  pow_17 = pow_18 = None
        mul_57 = mul_56 * all_constants_g;  mul_56 = all_constants_g = None
        all_constants_sigma_x = self.all_constants.sigma_x
        mul_58 = einsum_9 * all_constants_sigma_x;  einsum_9 = all_constants_sigma_x = None
        mul_59 = all_constants_gear * einsum_8;  all_constants_gear = einsum_8 = None
        neg = -pow_4;  pow_4 = None
        neg_1 = -pow_9;  pow_9 = None
        neg_2 = -pow_15;  pow_15 = None
        neg_3 = -pow_19;  pow_19 = None
        add_12 = einsum_15 + mul_42;  einsum_15 = mul_42 = None
        all_parameters_i = self.all_parameters.I
        add_13 = all_parameters_i + mul_43;  mul_43 = None
        add_14 = einsum_2 + mul_45;  einsum_2 = mul_45 = None
        add_15 = einsum_5 + mul_48;  einsum_5 = mul_48 = None
        add_16 = all_parameters_i + mul_49;  mul_49 = None
        add_17 = all_parameters_i + mul_51;  mul_51 = None
        add_18 = all_parameters_i + mul_54;  all_parameters_i = mul_54 = None
        add_19 = einsum + mul_58;  einsum = mul_58 = None
        cos = torch.cos(add_15)
        mul_60 = add_17 * all_parameters_m2
        mul_61 = mul_60 * all_parameters_l;  mul_60 = None
        mul_62 = mul_1 * add_14;  mul_1 = None
        mul_63 = mul_62 * cos;  mul_62 = None
        mul_64 = mul_59 * add_17
        all_constants_dt = self.all_constants.dt
        mul_65 = add_14 * all_constants_dt
        mul_66 = add * add_13;  add = None
        mul_67 = add_13 * all_parameters_b
        mul_68 = neg * pow_5;  neg = pow_5 = None
        mul_69 = add_13 * all_parameters_m2
        mul_70 = mul_69 * all_parameters_l;  mul_69 = None
        mul_71 = mul_44 * add_13;  add_13 = None
        mul_72 = add_3 * add_16;  add_3 = None
        mul_73 = add_16 * all_parameters_b
        mul_74 = neg_1 * pow_10;  neg_1 = pow_10 = None
        mul_75 = add_16 * all_parameters_m2
        mul_76 = mul_75 * all_parameters_l;  mul_75 = None
        mul_77 = mul_50 * add_16;  add_16 = None
        mul_78 = add_6 * add_18;  add_6 = None
        mul_79 = add_7 * add_17;  add_7 = None
        mul_80 = add_18 * all_parameters_b
        mul_81 = neg_2 * pow_16;  neg_2 = pow_16 = None
        mul_82 = mul_26 * cos;  mul_26 = None
        mul_83 = add_18 * all_parameters_m2;  all_parameters_m2 = None
        mul_84 = mul_83 * all_parameters_l;  mul_83 = all_parameters_l = None
        mul_85 = mul_55 * add_18;  add_18 = None
        mul_86 = add_17 * all_parameters_b;  add_17 = all_parameters_b = None
        mul_87 = mul_86 * add_12;  mul_86 = None
        mul_88 = mul_33 * add_14;  mul_33 = None
        mul_89 = neg_3 * pow_20;  neg_3 = pow_20 = None
        mul_90 = mul_38 * add_12;  mul_38 = None
        mul_91 = mul_90 * cos;  mul_90 = None
        mul_92 = mul_39 * cos;  mul_39 = None
        mul_93 = mul_92 * mul_59;  mul_92 = mul_59 = None
        all_constants_constant32 = self.all_constants.Constant32
        pow_21 = torch.pow(add_14, all_constants_constant32);  all_constants_constant32 = None
        all_constants_constant28 = self.all_constants.Constant28
        pow_22 = torch.pow(mul_82, all_constants_constant28);  mul_82 = all_constants_constant28 = None
        all_constants_constant31 = self.all_constants.Constant31
        pow_23 = torch.pow(add_14, all_constants_constant31);  all_constants_constant31 = None
        sin = torch.sin(add_15)
        sub = mul_79 - pow_22;  mul_79 = pow_22 = None
        all_constants_constant37 = self.all_constants.Constant37
        truediv = mul_65 / all_constants_constant37;  mul_65 = all_constants_constant37 = None
        mul_94 = mul_61 * pow_21;  mul_61 = pow_21 = None
        mul_95 = mul_94 * sin;  mul_94 = None
        mul_96 = mul_41 * sin;  mul_41 = None
        mul_97 = mul_96 * cos;  mul_96 = None
        mul_98 = mul_89 * pow_23;  mul_89 = pow_23 = None
        mul_99 = mul_98 * sin;  mul_98 = None
        mul_100 = mul_99 * cos;  mul_99 = cos = None
        mul_101 = mul_36 * sin;  mul_36 = sin = None
        sub_1 = mul_95 - mul_87;  mul_95 = mul_87 = None
        sub_2 = sub_1 - mul_97;  sub_1 = mul_97 = None
        sub_3 = mul_100 - mul_88;  mul_100 = mul_88 = None
        add_20 = sub_2 + mul_63;  sub_2 = mul_63 = None
        add_21 = add_20 + mul_64;  add_20 = mul_64 = None
        add_22 = add_15 + truediv;  truediv = None
        add_23 = sub_3 + mul_101;  sub_3 = mul_101 = None
        add_24 = add_23 + mul_91;  add_23 = mul_91 = None
        cos_1 = torch.cos(add_22)
        truediv_1 = add_21 / sub;  add_21 = None
        mul_102 = truediv_1 * all_constants_dt
        mul_103 = mul_2 * cos_1;  mul_2 = None
        mul_104 = mul_9 * cos_1;  mul_9 = None
        mul_105 = mul_104 * mul_44;  mul_104 = mul_44 = None
        all_constants_constant40 = self.all_constants.Constant40
        pow_24 = torch.pow(mul_103, all_constants_constant40);  mul_103 = all_constants_constant40 = None
        sin_1 = torch.sin(add_22);  add_22 = None
        sub_4 = mul_66 - pow_24;  mul_66 = pow_24 = None
        sub_5 = add_24 - mul_93;  add_24 = mul_93 = None
        truediv_2 = sub_5 / sub;  sub_5 = sub = None
        all_constants_constant36 = self.all_constants.Constant36
        truediv_3 = mul_102 / all_constants_constant36;  mul_102 = all_constants_constant36 = None
        mul_106 = truediv_2 * all_constants_dt
        mul_107 = mul_6 * sin_1;  mul_6 = None
        mul_108 = mul_47 * sin_1;  mul_47 = None
        mul_109 = mul_108 * cos_1;  mul_108 = None
        add_25 = add_12 + truediv_3;  truediv_3 = None
        all_constants_constant38 = self.all_constants.Constant38
        truediv_4 = mul_106 / all_constants_constant38;  mul_106 = all_constants_constant38 = None
        mul_110 = mul_67 * add_25;  mul_67 = None
        mul_111 = mul_8 * add_25;  mul_8 = None
        mul_112 = mul_111 * cos_1;  mul_111 = None
        all_constants_constant70 = self.all_constants.Constant70
        mul_113 = all_constants_constant70 * add_25;  all_constants_constant70 = add_25 = None
        add_26 = add_14 + truediv_4;  truediv_4 = None
        add_27 = add_12 + mul_113;  mul_113 = None
        mul_114 = mul_3 * add_26;  mul_3 = None
        mul_115 = mul_11 * add_26;  mul_11 = None
        mul_116 = mul_115 * cos_1;  mul_115 = None
        mul_117 = add_26 * all_constants_dt
        all_constants_constant76 = self.all_constants.Constant76
        mul_118 = all_constants_constant76 * add_26;  all_constants_constant76 = None
        all_constants_constant43 = self.all_constants.Constant43
        pow_25 = torch.pow(add_26, all_constants_constant43);  all_constants_constant43 = None
        all_constants_constant44 = self.all_constants.Constant44
        pow_26 = torch.pow(add_26, all_constants_constant44);  add_26 = all_constants_constant44 = None
        add_28 = add_14 + mul_118;  mul_118 = None
        all_constants_constant49 = self.all_constants.Constant49
        truediv_5 = mul_117 / all_constants_constant49;  mul_117 = all_constants_constant49 = None
        mul_119 = mul_68 * pow_25;  mul_68 = pow_25 = None
        mul_120 = mul_119 * sin_1;  mul_119 = None
        mul_121 = mul_120 * cos_1;  mul_120 = cos_1 = None
        mul_122 = mul_70 * pow_26;  mul_70 = pow_26 = None
        mul_123 = mul_122 * sin_1;  mul_122 = sin_1 = None
        sub_6 = mul_121 - mul_114;  mul_121 = mul_114 = None
        sub_7 = mul_123 - mul_110;  mul_123 = mul_110 = None
        sub_8 = sub_7 - mul_109;  sub_7 = mul_109 = None
        add_29 = sub_6 + mul_107;  sub_6 = mul_107 = None
        add_30 = add_29 + mul_112;  add_29 = mul_112 = None
        add_31 = sub_8 + mul_116;  sub_8 = mul_116 = None
        add_32 = add_31 + mul_71;  add_31 = mul_71 = None
        add_33 = add_15 + truediv_5;  truediv_5 = None
        cos_2 = torch.cos(add_33)
        truediv_6 = add_32 / sub_4;  add_32 = None
        mul_124 = truediv_6 * all_constants_dt
        mul_125 = mul_12 * cos_2;  mul_12 = None
        mul_126 = mul_19 * cos_2;  mul_19 = None
        mul_127 = mul_126 * mul_50;  mul_126 = mul_50 = None
        all_constants_constant67 = self.all_constants.Constant67
        mul_128 = all_constants_constant67 * truediv_6;  all_constants_constant67 = truediv_6 = None
        all_constants_constant52 = self.all_constants.Constant52
        pow_27 = torch.pow(mul_125, all_constants_constant52);  mul_125 = all_constants_constant52 = None
        sin_2 = torch.sin(add_33);  add_33 = None
        sub_9 = add_30 - mul_105;  add_30 = mul_105 = None
        sub_10 = mul_72 - pow_27;  mul_72 = pow_27 = None
        add_34 = truediv_1 + mul_128;  truediv_1 = mul_128 = None
        truediv_7 = sub_9 / sub_4;  sub_9 = sub_4 = None
        all_constants_constant48 = self.all_constants.Constant48
        truediv_8 = mul_124 / all_constants_constant48;  mul_124 = all_constants_constant48 = None
        mul_129 = truediv_7 * all_constants_dt
        mul_130 = mul_16 * sin_2;  mul_16 = None
        mul_131 = mul_53 * sin_2;  mul_53 = None
        mul_132 = mul_131 * cos_2;  mul_131 = None
        all_constants_constant73 = self.all_constants.Constant73
        mul_133 = all_constants_constant73 * truediv_7;  all_constants_constant73 = truediv_7 = None
        add_35 = add_12 + truediv_8;  truediv_8 = None
        add_36 = truediv_2 + mul_133;  truediv_2 = mul_133 = None
        all_constants_constant50 = self.all_constants.Constant50
        truediv_9 = mul_129 / all_constants_constant50;  mul_129 = all_constants_constant50 = None
        mul_134 = mul_73 * add_35;  mul_73 = None
        mul_135 = mul_18 * add_35;  mul_18 = None
        mul_136 = mul_135 * cos_2;  mul_135 = None
        all_constants_constant71 = self.all_constants.Constant71
        mul_137 = all_constants_constant71 * add_35;  all_constants_constant71 = add_35 = None
        add_37 = add_14 + truediv_9;  truediv_9 = None
        add_38 = add_27 + mul_137;  add_27 = mul_137 = None
        mul_138 = mul_13 * add_37;  mul_13 = None
        mul_139 = mul_21 * add_37;  mul_21 = None
        mul_140 = mul_139 * cos_2;  mul_139 = None
        mul_141 = add_37 * all_constants_dt
        all_constants_constant77 = self.all_constants.Constant77
        mul_142 = all_constants_constant77 * add_37;  all_constants_constant77 = None
        all_constants_constant55 = self.all_constants.Constant55
        pow_28 = torch.pow(add_37, all_constants_constant55);  all_constants_constant55 = None
        all_constants_constant56 = self.all_constants.Constant56
        pow_29 = torch.pow(add_37, all_constants_constant56);  add_37 = all_constants_constant56 = None
        add_39 = add_15 + mul_141;  mul_141 = None
        add_40 = add_28 + mul_142;  add_28 = mul_142 = None
        cos_3 = torch.cos(add_39)
        mul_143 = mul_74 * pow_28;  mul_74 = pow_28 = None
        mul_144 = mul_143 * sin_2;  mul_143 = None
        mul_145 = mul_144 * cos_2;  mul_144 = cos_2 = None
        mul_146 = mul_76 * pow_29;  mul_76 = pow_29 = None
        mul_147 = mul_146 * sin_2;  mul_146 = sin_2 = None
        mul_148 = mul_22 * cos_3;  mul_22 = None
        mul_149 = mul_30 * cos_3;  mul_30 = None
        mul_150 = mul_149 * mul_55;  mul_149 = mul_55 = None
        all_constants_constant60 = self.all_constants.Constant60
        pow_30 = torch.pow(mul_148, all_constants_constant60);  mul_148 = all_constants_constant60 = None
        sin_3 = torch.sin(add_39);  add_39 = None
        sub_11 = mul_145 - mul_138;  mul_145 = mul_138 = None
        sub_12 = mul_147 - mul_134;  mul_147 = mul_134 = None
        sub_13 = sub_12 - mul_132;  sub_12 = mul_132 = None
        sub_14 = mul_78 - pow_30;  mul_78 = pow_30 = None
        add_41 = sub_11 + mul_130;  sub_11 = mul_130 = None
        add_42 = add_41 + mul_136;  add_41 = mul_136 = None
        add_43 = sub_13 + mul_140;  sub_13 = mul_140 = None
        add_44 = add_43 + mul_77;  add_43 = mul_77 = None
        truediv_10 = add_44 / sub_10;  add_44 = None
        mul_151 = truediv_10 * all_constants_dt
        mul_152 = mul_27 * sin_3;  mul_27 = None
        mul_153 = mul_57 * sin_3;  mul_57 = None
        mul_154 = mul_153 * cos_3;  mul_153 = None
        all_constants_constant68 = self.all_constants.Constant68
        mul_155 = all_constants_constant68 * truediv_10;  all_constants_constant68 = truediv_10 = None
        sub_15 = add_42 - mul_127;  add_42 = mul_127 = None
        add_45 = add_12 + mul_151;  mul_151 = None
        add_46 = add_34 + mul_155;  add_34 = mul_155 = None
        add_47 = add_38 + add_45;  add_38 = None
        truediv_11 = sub_15 / sub_10;  sub_15 = sub_10 = None
        mul_156 = truediv_11 * all_constants_dt
        mul_157 = mul_80 * add_45;  mul_80 = None
        mul_158 = mul_29 * add_45;  mul_29 = add_45 = None
        mul_159 = mul_158 * cos_3;  mul_158 = None
        mul_160 = add_47 * all_constants_dt;  add_47 = None
        all_constants_constant74 = self.all_constants.Constant74
        mul_161 = all_constants_constant74 * truediv_11;  all_constants_constant74 = truediv_11 = None
        add_48 = add_14 + mul_156;  mul_156 = None
        add_49 = add_36 + mul_161;  add_36 = mul_161 = None
        add_50 = add_40 + add_48;  add_40 = None
        all_constants_constant72 = self.all_constants.Constant72
        truediv_12 = mul_160 / all_constants_constant72;  mul_160 = all_constants_constant72 = None
        mul_162 = mul_23 * add_48;  mul_23 = None
        mul_163 = mul_32 * add_48;  mul_32 = None
        mul_164 = mul_163 * cos_3;  mul_163 = None
        mul_165 = add_50 * all_constants_dt;  add_50 = None
        all_constants_constant63 = self.all_constants.Constant63
        pow_31 = torch.pow(add_48, all_constants_constant63);  all_constants_constant63 = None
        all_constants_constant64 = self.all_constants.Constant64
        pow_32 = torch.pow(add_48, all_constants_constant64);  add_48 = all_constants_constant64 = None
        add_51 = add_19 + truediv_12;  add_19 = truediv_12 = None
        all_constants_constant78 = self.all_constants.Constant78
        truediv_13 = mul_165 / all_constants_constant78;  mul_165 = all_constants_constant78 = None
        mul_166 = mul_81 * pow_31;  mul_81 = pow_31 = None
        mul_167 = mul_166 * sin_3;  mul_166 = None
        mul_168 = mul_167 * cos_3;  mul_167 = cos_3 = None
        mul_169 = mul_84 * pow_32;  mul_84 = pow_32 = None
        mul_170 = mul_169 * sin_3;  mul_169 = sin_3 = None
        sub_16 = mul_168 - mul_162;  mul_168 = mul_162 = None
        sub_17 = mul_170 - mul_157;  mul_170 = mul_157 = None
        sub_18 = sub_17 - mul_154;  sub_17 = mul_154 = None
        add_52 = sub_16 + mul_152;  sub_16 = mul_152 = None
        add_53 = add_52 + mul_159;  add_52 = mul_159 = None
        add_54 = sub_18 + mul_164;  sub_18 = mul_164 = None
        add_55 = add_54 + mul_85;  add_54 = mul_85 = None
        add_56 = add_15 + truediv_13;  add_15 = truediv_13 = None
        truediv_14 = add_55 / sub_14;  add_55 = None
        sub_19 = add_53 - mul_150;  add_53 = mul_150 = None
        add_57 = add_46 + truediv_14;  add_46 = truediv_14 = None
        truediv_15 = sub_19 / sub_14;  sub_19 = sub_14 = None
        mul_171 = add_57 * all_constants_dt;  add_57 = None
        add_58 = add_49 + truediv_15;  add_49 = truediv_15 = None
        all_constants_constant69 = self.all_constants.Constant69
        truediv_16 = mul_171 / all_constants_constant69;  mul_171 = all_constants_constant69 = None
        mul_172 = add_58 * all_constants_dt;  add_58 = all_constants_dt = None
        add_59 = add_12 + truediv_16;  add_12 = truediv_16 = None
        all_constants_constant75 = self.all_constants.Constant75
        truediv_17 = mul_172 / all_constants_constant75;  mul_172 = all_constants_constant75 = None
        add_60 = add_14 + truediv_17;  add_14 = truediv_17 = None
        return ({'est_theta': add_56, 'est_thetadot': add_60, 'est_x': add_51, 'est_xdot': add_59}, {'SamplePart549': einsum_11, 'SamplePart551': einsum_12, 'SamplePart555': einsum_14, 'SamplePart553': einsum_13, 'Add535': add_60, 'Add547': add_56, 'Add523': add_51, 'Add511': add_59}, {'Xangle': add_56, 'Xangular_velocity': add_60, 'Xpos': add_51, 'Xvelocity': add_59}, {})
        
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
