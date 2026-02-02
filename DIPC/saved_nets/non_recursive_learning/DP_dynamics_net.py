import torch

def nnodely_basic_model_connect(data_in, rel):
    virtual = torch.cat((data_in[:, 1:, :], data_in[:, :1, :]), dim=1)
    max_dim = min(rel.size(1), data_in.size(1))
    virtual[:, -max_dim:, :] = rel[:, -max_dim:, :]
    return virtual

class TracerModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.all_parameters = {}
        self.all_constants = {}
        self.all_constants["Constant100"] = torch.tensor([2.0], requires_grad=False)
        self.all_constants["Constant101"] = torch.tensor([2.0], requires_grad=False)
        self.all_constants["Constant102"] = torch.tensor([2.0], requires_grad=False)
        self.all_constants["Constant103"] = torch.tensor([2.0], requires_grad=False)
        self.all_constants["Constant104"] = torch.tensor([2.0], requires_grad=False)
        self.all_constants["Constant105"] = torch.tensor([2.0], requires_grad=False)
        self.all_constants["Constant106"] = torch.tensor([2.0], requires_grad=False)
        self.all_constants["Constant107"] = torch.tensor([2.0], requires_grad=False)
        self.all_constants["Constant108"] = torch.tensor([2.0], requires_grad=False)
        self.all_constants["Constant109"] = torch.tensor([2.0], requires_grad=False)
        self.all_constants["Constant110"] = torch.tensor([2.0], requires_grad=False)
        self.all_constants["Constant111"] = torch.tensor([2.0], requires_grad=False)
        self.all_constants["Constant112"] = torch.tensor([2.0], requires_grad=False)
        self.all_constants["Constant113"] = torch.tensor([2.0], requires_grad=False)
        self.all_constants["Constant114"] = torch.tensor([2.0], requires_grad=False)
        self.all_constants["Constant115"] = torch.tensor([2.0], requires_grad=False)
        self.all_constants["Constant116"] = torch.tensor([2.0], requires_grad=False)
        self.all_constants["Constant117"] = torch.tensor([2.0], requires_grad=False)
        self.all_constants["Constant118"] = torch.tensor([2.0], requires_grad=False)
        self.all_constants["Constant119"] = torch.tensor([2.0], requires_grad=False)
        self.all_constants["Constant120"] = torch.tensor([2.0], requires_grad=False)
        self.all_constants["Constant121"] = torch.tensor([2.0], requires_grad=False)
        self.all_constants["Constant122"] = torch.tensor([2.0], requires_grad=False)
        self.all_constants["Constant123"] = torch.tensor([2.0], requires_grad=False)
        self.all_constants["Constant124"] = torch.tensor([2.0], requires_grad=False)
        self.all_constants["Constant125"] = torch.tensor([2.0], requires_grad=False)
        self.all_constants["Constant141"] = torch.tensor([0.0], requires_grad=False)
        self.all_constants["Constant142"] = torch.tensor([0.0], requires_grad=False)
        self.all_constants["Constant143"] = torch.tensor([0.0], requires_grad=False)
        self.all_constants["Constant21"] = torch.tensor([0.5], requires_grad=False)
        self.all_constants["Constant22"] = torch.tensor([0.6000000238418579], requires_grad=False)
        self.all_constants["Constant23"] = torch.tensor([0.5], requires_grad=False)
        self.all_constants["Constant24"] = torch.tensor([0.6000000238418579], requires_grad=False)
        self.all_constants["Constant25"] = torch.tensor([0.36000001430511475], requires_grad=False)
        self.all_constants["Constant26"] = torch.tensor([4.0], requires_grad=False)
        self.all_constants["Constant27"] = torch.tensor([0.36000001430511475], requires_grad=False)
        self.all_constants["Constant28"] = torch.tensor([0.6000000238418579], requires_grad=False)
        self.all_constants["Constant29"] = torch.tensor([0.6000000238418579], requires_grad=False)
        self.all_constants["Constant30"] = torch.tensor([2.0], requires_grad=False)
        self.all_constants["Constant31"] = torch.tensor([0.36000001430511475], requires_grad=False)
        self.all_constants["Constant32"] = torch.tensor([4.0], requires_grad=False)
        self.all_constants["Constant33"] = torch.tensor([0.5], requires_grad=False)
        self.all_constants["Constant34"] = torch.tensor([0.6000000238418579], requires_grad=False)
        self.all_constants["Constant35"] = torch.tensor([0.5], requires_grad=False)
        self.all_constants["Constant36"] = torch.tensor([0.6000000238418579], requires_grad=False)
        self.all_constants["Constant37"] = torch.tensor([2.0], requires_grad=False)
        self.all_constants["Constant38"] = torch.tensor([2.0], requires_grad=False)
        self.all_constants["Constant39"] = torch.tensor([2.0], requires_grad=False)
        self.all_constants["Constant40"] = torch.tensor([2.0], requires_grad=False)
        self.all_constants["Constant41"] = torch.tensor([2.0], requires_grad=False)
        self.all_constants["Constant42"] = torch.tensor([2.0], requires_grad=False)
        self.all_constants["Constant43"] = torch.tensor([2.0], requires_grad=False)
        self.all_constants["Constant44"] = torch.tensor([9.99999993922529e-09], requires_grad=False)
        self.all_constants["Constant45"] = torch.tensor([2.0], requires_grad=False)
        self.all_constants["Constant46"] = torch.tensor([2.0], requires_grad=False)
        self.all_constants["Constant47"] = torch.tensor([2.0], requires_grad=False)
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
        self.all_constants["Constant85"] = torch.tensor([2.0], requires_grad=False)
        self.all_constants["Constant86"] = torch.tensor([2.0], requires_grad=False)
        self.all_constants["Constant87"] = torch.tensor([2.0], requires_grad=False)
        self.all_constants["Constant88"] = torch.tensor([2.0], requires_grad=False)
        self.all_constants["Constant89"] = torch.tensor([2.0], requires_grad=False)
        self.all_constants["Constant90"] = torch.tensor([2.0], requires_grad=False)
        self.all_constants["Constant91"] = torch.tensor([2.0], requires_grad=False)
        self.all_constants["Constant92"] = torch.tensor([2.0], requires_grad=False)
        self.all_constants["Constant93"] = torch.tensor([2.0], requires_grad=False)
        self.all_constants["Constant94"] = torch.tensor([2.0], requires_grad=False)
        self.all_constants["Constant95"] = torch.tensor([2.0], requires_grad=False)
        self.all_constants["Constant96"] = torch.tensor([2.0], requires_grad=False)
        self.all_constants["Constant97"] = torch.tensor([2.0], requires_grad=False)
        self.all_constants["Constant98"] = torch.tensor([2.0], requires_grad=False)
        self.all_constants["Constant99"] = torch.tensor([2.0], requires_grad=False)
        self.all_constants["gravity"] = torch.tensor([9.8100004196167], requires_grad=False)
        self.all_parameters["Inertia1"] = torch.nn.Parameter(torch.tensor([0.12967057526111603]), requires_grad=True)
        self.all_parameters["Inertia2"] = torch.nn.Parameter(torch.tensor([0.07212039828300476]), requires_grad=True)
        self.all_parameters["b1"] = torch.nn.Parameter(torch.tensor([0.003206502879038453]), requires_grad=True)
        self.all_parameters["b2"] = torch.nn.Parameter(torch.tensor([0.5894838571548462]), requires_grad=True)
        self.all_parameters["bc"] = torch.nn.Parameter(torch.tensor([0.8418918251991272]), requires_grad=True)
        self.all_parameters["gear"] = torch.nn.Parameter(torch.tensor([500.9664611816406]), requires_grad=True)
        self.all_parameters["m"] = torch.nn.Parameter(torch.tensor([10.639083862304688]), requires_grad=True)
        self.all_parameters["m1"] = torch.nn.Parameter(torch.tensor([5.1800642013549805]), requires_grad=True)
        self.all_parameters["m2"] = torch.nn.Parameter(torch.tensor([3.9831883907318115]), requires_grad=True)
        self.all_constants["SamplePart1022"] = torch.tensor([[1.]], requires_grad=False)
        self.all_constants["SamplePart1024"] = torch.tensor([[1.]], requires_grad=False)
        self.all_constants["SamplePart1026"] = torch.tensor([[1.]], requires_grad=False)
        self.all_constants["SamplePart1028"] = torch.tensor([[1.]], requires_grad=False)
        self.all_constants["SamplePart1035"] = torch.tensor([[1.]], requires_grad=False)
        self.all_constants["SamplePart1042"] = torch.tensor([[1.]], requires_grad=False)
        self.all_constants["SamplePart11"] = torch.tensor([[1.]], requires_grad=False)
        self.all_constants["SamplePart13"] = torch.tensor([[1.]], requires_grad=False)
        self.all_constants["SamplePart3"] = torch.tensor([[1.]], requires_grad=False)
        self.all_constants["SamplePart5"] = torch.tensor([[1.]], requires_grad=False)
        self.all_constants["SamplePart7"] = torch.tensor([[1.]], requires_grad=False)
        self.all_constants["SamplePart9"] = torch.tensor([[1.]], requires_grad=False)
        self.all_parameters = torch.nn.ParameterDict(self.all_parameters)
        self.all_constants = torch.nn.ParameterDict(self.all_constants)

    def update(self, closed_loop={}, connect={}, disconnect=False):
        pass
    
    def forward(self, kwargs):
        all_parameters_b2 = self.all_parameters.b2
        neg = -all_parameters_b2
        relu = torch.relu(neg);  neg = None
        getitem = kwargs['action']
        relation_forward_sample_part1042_w = self.all_constants.SamplePart1042
        einsum = torch.functional.einsum('bij,ki->bkj', getitem, relation_forward_sample_part1042_w);  getitem = relation_forward_sample_part1042_w = None
        all_constants_constant143 = self.all_constants.Constant143
        mul = einsum * all_constants_constant143;  einsum = all_constants_constant143 = None
        all_parameters_b1 = self.all_parameters.b1
        neg_1 = -all_parameters_b1
        relu_1 = torch.relu(neg_1);  neg_1 = None
        getitem_1 = kwargs['action']
        relation_forward_sample_part1035_w = self.all_constants.SamplePart1035
        einsum_1 = torch.functional.einsum('bij,ki->bkj', getitem_1, relation_forward_sample_part1035_w);  getitem_1 = relation_forward_sample_part1035_w = None
        all_constants_constant142 = self.all_constants.Constant142
        mul_1 = einsum_1 * all_constants_constant142;  einsum_1 = all_constants_constant142 = None
        all_parameters_bc = self.all_parameters.bc
        neg_2 = -all_parameters_bc
        relu_2 = torch.relu(neg_2);  neg_2 = None
        getitem_2 = kwargs['action']
        relation_forward_sample_part1028_w = self.all_constants.SamplePart1028
        einsum_2 = torch.functional.einsum('bij,ki->bkj', getitem_2, relation_forward_sample_part1028_w);  getitem_2 = relation_forward_sample_part1028_w = None
        all_constants_constant141 = self.all_constants.Constant141
        mul_2 = einsum_2 * all_constants_constant141;  einsum_2 = all_constants_constant141 = None
        getitem_3 = kwargs['Xth2']
        relation_forward_sample_part9_w = self.all_constants.SamplePart9
        einsum_3 = torch.functional.einsum('bij,ki->bkj', getitem_3, relation_forward_sample_part9_w);  getitem_3 = relation_forward_sample_part9_w = None
        cos = torch.cos(einsum_3)
        getitem_4 = kwargs['Xth1']
        relation_forward_sample_part5_w = self.all_constants.SamplePart5
        einsum_4 = torch.functional.einsum('bij,ki->bkj', getitem_4, relation_forward_sample_part5_w);  getitem_4 = relation_forward_sample_part5_w = None
        cos_1 = torch.cos(einsum_4)
        add = einsum_4 + einsum_3
        cos_2 = torch.cos(add);  add = None
        all_parameters_m2 = self.all_parameters.m2
        all_constants_constant28 = self.all_constants.Constant28
        mul_3 = all_parameters_m2 * all_constants_constant28;  all_constants_constant28 = None
        all_constants_constant29 = self.all_constants.Constant29
        mul_4 = mul_3 * all_constants_constant29;  mul_3 = all_constants_constant29 = None
        all_constants_constant30 = self.all_constants.Constant30
        truediv = mul_4 / all_constants_constant30;  mul_4 = all_constants_constant30 = None
        all_constants_constant23 = self.all_constants.Constant23
        mul_5 = all_constants_constant23 * all_parameters_m2;  all_constants_constant23 = None
        all_constants_constant24 = self.all_constants.Constant24
        mul_6 = mul_5 * all_constants_constant24;  mul_5 = all_constants_constant24 = None
        all_constants_constant21 = self.all_constants.Constant21
        all_parameters_m1 = self.all_parameters.m1
        mul_7 = all_constants_constant21 * all_parameters_m1;  all_constants_constant21 = None
        add_1 = mul_7 + all_parameters_m2;  mul_7 = None
        all_constants_constant22 = self.all_constants.Constant22
        mul_8 = add_1 * all_constants_constant22;  add_1 = all_constants_constant22 = None
        all_constants_constant43 = self.all_constants.Constant43
        mul_9 = all_constants_constant43 * mul_8;  all_constants_constant43 = None
        mul_10 = mul_9 * mul_6;  mul_9 = None
        mul_11 = mul_10 * truediv;  mul_10 = None
        mul_12 = mul_11 * cos_2;  mul_11 = cos_2 = None
        mul_13 = mul_12 * cos_1;  mul_12 = cos_1 = None
        mul_14 = mul_13 * cos;  mul_13 = cos = None
        cos_3 = torch.cos(einsum_4)
        all_constants_constant42 = self.all_constants.Constant42
        pow_1 = torch.pow(cos_3, all_constants_constant42);  cos_3 = all_constants_constant42 = None
        all_constants_constant31 = self.all_constants.Constant31
        mul_15 = all_parameters_m2 * all_constants_constant31;  all_constants_constant31 = None
        all_constants_constant32 = self.all_constants.Constant32
        truediv_1 = mul_15 / all_constants_constant32;  mul_15 = all_constants_constant32 = None
        all_parameters_inertia2 = self.all_parameters.Inertia2
        add_2 = all_parameters_inertia2 + truediv_1;  all_parameters_inertia2 = truediv_1 = None
        all_constants_constant41 = self.all_constants.Constant41
        pow_2 = torch.pow(mul_8, all_constants_constant41);  all_constants_constant41 = None
        mul_16 = pow_2 * add_2;  pow_2 = None
        mul_17 = mul_16 * pow_1;  mul_16 = pow_1 = None
        cos_4 = torch.cos(einsum_3)
        all_constants_constant40 = self.all_constants.Constant40
        pow_3 = torch.pow(cos_4, all_constants_constant40);  cos_4 = all_constants_constant40 = None
        all_constants_constant39 = self.all_constants.Constant39
        pow_4 = torch.pow(truediv, all_constants_constant39);  all_constants_constant39 = None
        all_parameters_m = self.all_parameters.m
        add_3 = all_parameters_m + all_parameters_m1;  all_parameters_m = None
        add_4 = add_3 + all_parameters_m2;  add_3 = None
        mul_18 = add_4 * pow_4;  pow_4 = None
        mul_19 = mul_18 * pow_3;  mul_18 = pow_3 = None
        all_constants_constant27 = self.all_constants.Constant27
        mul_20 = all_parameters_m2 * all_constants_constant27;  all_constants_constant27 = None
        all_constants_constant25 = self.all_constants.Constant25
        mul_21 = all_parameters_m1 * all_constants_constant25;  all_constants_constant25 = None
        all_constants_constant26 = self.all_constants.Constant26
        truediv_2 = mul_21 / all_constants_constant26;  mul_21 = all_constants_constant26 = None
        all_parameters_inertia1 = self.all_parameters.Inertia1
        add_5 = all_parameters_inertia1 + truediv_2;  all_parameters_inertia1 = truediv_2 = None
        add_6 = add_5 + mul_20;  add_5 = mul_20 = None
        mul_22 = add_4 * add_6
        mul_23 = mul_22 * add_2;  mul_22 = None
        add_7 = einsum_4 + einsum_3
        cos_5 = torch.cos(add_7);  add_7 = None
        all_constants_constant38 = self.all_constants.Constant38
        pow_5 = torch.pow(cos_5, all_constants_constant38);  cos_5 = all_constants_constant38 = None
        all_constants_constant37 = self.all_constants.Constant37
        pow_6 = torch.pow(mul_6, all_constants_constant37);  all_constants_constant37 = None
        mul_24 = pow_6 * add_6;  pow_6 = None
        mul_25 = mul_24 * pow_5;  mul_24 = pow_5 = None
        sub = mul_25 - mul_23;  mul_25 = mul_23 = None
        add_8 = sub + mul_19;  sub = mul_19 = None
        add_9 = add_8 + mul_17;  add_8 = mul_17 = None
        sub_1 = add_9 - mul_14;  add_9 = mul_14 = None
        all_constants_constant44 = self.all_constants.Constant44
        add_10 = sub_1 + all_constants_constant44;  sub_1 = all_constants_constant44 = None
        cos_6 = torch.cos(einsum_3)
        cos_7 = torch.cos(einsum_4)
        add_11 = einsum_4 + einsum_3
        sin = torch.sin(add_11);  add_11 = None
        getitem_5 = kwargs['Xth2_dot']
        relation_forward_sample_part11_w = self.all_constants.SamplePart11
        einsum_5 = torch.functional.einsum('bij,ki->bkj', getitem_5, relation_forward_sample_part11_w);  getitem_5 = relation_forward_sample_part11_w = None
        getitem_6 = kwargs['Xth1_dot']
        relation_forward_sample_part7_w = self.all_constants.SamplePart7
        einsum_6 = torch.functional.einsum('bij,ki->bkj', getitem_6, relation_forward_sample_part7_w);  getitem_6 = relation_forward_sample_part7_w = None
        mul_26 = mul_8 * mul_6
        mul_27 = mul_26 * truediv;  mul_26 = None
        mul_28 = mul_27 * einsum_6;  mul_27 = None
        mul_29 = mul_28 * einsum_5;  mul_28 = None
        mul_30 = mul_29 * sin;  mul_29 = sin = None
        mul_31 = mul_30 * cos_7;  mul_30 = cos_7 = None
        mul_32 = mul_31 * cos_6;  mul_31 = cos_6 = None
        sin_1 = torch.sin(einsum_3)
        cos_8 = torch.cos(einsum_4)
        add_12 = einsum_4 + einsum_3
        cos_9 = torch.cos(add_12);  add_12 = None
        mul_33 = mul_8 * mul_6
        mul_34 = mul_33 * truediv;  mul_33 = None
        mul_35 = mul_34 * einsum_6;  mul_34 = None
        mul_36 = mul_35 * einsum_5;  mul_35 = None
        mul_37 = mul_36 * cos_9;  mul_36 = cos_9 = None
        mul_38 = mul_37 * cos_8;  mul_37 = cos_8 = None
        mul_39 = mul_38 * sin_1;  mul_38 = sin_1 = None
        cos_10 = torch.cos(einsum_3)
        cos_11 = torch.cos(einsum_4)
        add_13 = einsum_4 + einsum_3
        sin_2 = torch.sin(add_13);  add_13 = None
        all_constants_constant125 = self.all_constants.Constant125
        pow_7 = torch.pow(einsum_5, all_constants_constant125);  all_constants_constant125 = None
        mul_40 = mul_8 * mul_6
        mul_41 = mul_40 * truediv;  mul_40 = None
        mul_42 = mul_41 * pow_7;  mul_41 = pow_7 = None
        mul_43 = mul_42 * sin_2;  mul_42 = sin_2 = None
        mul_44 = mul_43 * cos_11;  mul_43 = cos_11 = None
        mul_45 = mul_44 * cos_10;  mul_44 = cos_10 = None
        sin_3 = torch.sin(einsum_3)
        cos_12 = torch.cos(einsum_4)
        add_14 = einsum_4 + einsum_3
        cos_13 = torch.cos(add_14);  add_14 = None
        all_constants_constant124 = self.all_constants.Constant124
        pow_8 = torch.pow(einsum_5, all_constants_constant124);  all_constants_constant124 = None
        mul_46 = mul_8 * mul_6
        mul_47 = mul_46 * truediv;  mul_46 = None
        mul_48 = mul_47 * pow_8;  mul_47 = pow_8 = None
        mul_49 = mul_48 * cos_13;  mul_48 = cos_13 = None
        mul_50 = mul_49 * cos_12;  mul_49 = cos_12 = None
        mul_51 = mul_50 * sin_3;  mul_50 = sin_3 = None
        sin_4 = torch.sin(einsum_4)
        cos_14 = torch.cos(einsum_4)
        add_15 = einsum_4 + einsum_3
        cos_15 = torch.cos(add_15);  add_15 = None
        all_constants_constant33 = self.all_constants.Constant33
        mul_52 = all_constants_constant33 * all_parameters_m1;  all_constants_constant33 = all_parameters_m1 = None
        add_16 = mul_52 + all_parameters_m2;  mul_52 = None
        all_constants_gravity = self.all_constants.gravity
        mul_53 = add_16 * all_constants_gravity;  add_16 = None
        all_constants_constant34 = self.all_constants.Constant34
        mul_54 = mul_53 * all_constants_constant34;  mul_53 = all_constants_constant34 = None
        mul_55 = mul_8 * mul_6
        mul_56 = mul_55 * mul_54;  mul_55 = None
        mul_57 = mul_56 * cos_15;  mul_56 = cos_15 = None
        mul_58 = mul_57 * cos_14;  mul_57 = cos_14 = None
        mul_59 = mul_58 * sin_4;  mul_58 = sin_4 = None
        sin_5 = torch.sin(einsum_4)
        add_17 = einsum_4 + einsum_3
        cos_16 = torch.cos(add_17);  add_17 = None
        all_constants_constant123 = self.all_constants.Constant123
        pow_9 = torch.pow(einsum_6, all_constants_constant123);  all_constants_constant123 = None
        mul_60 = mul_8 * mul_6
        mul_61 = mul_60 * add_6;  mul_60 = None
        mul_62 = mul_61 * pow_9;  mul_61 = pow_9 = None
        mul_63 = mul_62 * cos_16;  mul_62 = cos_16 = None
        mul_64 = mul_63 * sin_5;  mul_63 = sin_5 = None
        sin_6 = torch.sin(einsum_4)
        cos_17 = torch.cos(einsum_3)
        cos_18 = torch.cos(einsum_4)
        all_constants_constant122 = self.all_constants.Constant122
        pow_10 = torch.pow(einsum_6, all_constants_constant122);  all_constants_constant122 = None
        all_constants_constant121 = self.all_constants.Constant121
        pow_11 = torch.pow(mul_8, all_constants_constant121);  all_constants_constant121 = None
        mul_65 = pow_11 * truediv;  pow_11 = None
        mul_66 = mul_65 * pow_10;  mul_65 = pow_10 = None
        mul_67 = mul_66 * cos_18;  mul_66 = cos_18 = None
        mul_68 = mul_67 * cos_17;  mul_67 = cos_17 = None
        mul_69 = mul_68 * sin_6;  mul_68 = sin_6 = None
        cos_19 = torch.cos(einsum_3)
        cos_20 = torch.cos(einsum_4)
        getitem_7 = kwargs['Xvelocity']
        relation_forward_sample_part3_w = self.all_constants.SamplePart3
        einsum_7 = torch.functional.einsum('bij,ki->bkj', getitem_7, relation_forward_sample_part3_w);  getitem_7 = relation_forward_sample_part3_w = None
        mul_70 = all_parameters_bc * mul_8
        mul_71 = mul_70 * truediv;  mul_70 = None
        mul_72 = mul_71 * einsum_7;  mul_71 = None
        mul_73 = mul_72 * cos_20;  mul_72 = cos_20 = None
        mul_74 = mul_73 * cos_19;  mul_73 = cos_19 = None
        cos_21 = torch.cos(einsum_4)
        add_18 = einsum_4 + einsum_3
        cos_22 = torch.cos(add_18);  add_18 = None
        mul_75 = all_parameters_b2 * mul_8
        mul_76 = mul_75 * mul_6;  mul_75 = None
        mul_77 = mul_76 * einsum_5;  mul_76 = None
        mul_78 = mul_77 * cos_22;  mul_77 = cos_22 = None
        mul_79 = mul_78 * cos_21;  mul_78 = cos_21 = None
        cos_23 = torch.cos(einsum_4)
        add_19 = einsum_4 + einsum_3
        cos_24 = torch.cos(add_19);  add_19 = None
        mul_80 = all_parameters_b2 * mul_8
        mul_81 = mul_80 * mul_6;  mul_80 = None
        mul_82 = mul_81 * einsum_6;  mul_81 = None
        mul_83 = mul_82 * cos_24;  mul_82 = cos_24 = None
        mul_84 = mul_83 * cos_23;  mul_83 = cos_23 = None
        cos_25 = torch.cos(einsum_4)
        add_20 = einsum_4 + einsum_3
        cos_26 = torch.cos(add_20);  add_20 = None
        mul_85 = all_parameters_b1 * mul_8
        mul_86 = mul_85 * mul_6;  mul_85 = None
        mul_87 = mul_86 * einsum_6;  mul_86 = None
        mul_88 = mul_87 * cos_26;  mul_87 = cos_26 = None
        mul_89 = mul_88 * cos_25;  mul_88 = cos_25 = None
        all_constants_constant119 = self.all_constants.Constant119
        mul_90 = all_constants_constant119 * einsum_3;  all_constants_constant119 = None
        all_constants_constant118 = self.all_constants.Constant118
        mul_91 = all_constants_constant118 * einsum_4;  all_constants_constant118 = None
        add_21 = mul_91 + mul_90;  mul_91 = mul_90 = None
        sin_7 = torch.sin(add_21);  add_21 = None
        all_constants_constant117 = self.all_constants.Constant117
        pow_12 = torch.pow(mul_6, all_constants_constant117);  all_constants_constant117 = None
        mul_92 = pow_12 * add_6;  pow_12 = None
        mul_93 = mul_92 * einsum_6;  mul_92 = None
        mul_94 = mul_93 * einsum_5;  mul_93 = None
        mul_95 = mul_94 * sin_7;  mul_94 = sin_7 = None
        all_constants_constant120 = self.all_constants.Constant120
        truediv_3 = mul_95 / all_constants_constant120;  mul_95 = all_constants_constant120 = None
        cos_27 = torch.cos(einsum_3)
        mul_96 = all_parameters_b2 * add_4
        mul_97 = mul_96 * truediv;  mul_96 = None
        mul_98 = mul_97 * einsum_5;  mul_97 = None
        mul_99 = mul_98 * cos_27;  mul_98 = cos_27 = None
        cos_28 = torch.cos(einsum_3)
        mul_100 = all_parameters_b2 * add_4
        mul_101 = mul_100 * truediv;  mul_100 = None
        mul_102 = mul_101 * einsum_6;  mul_101 = None
        mul_103 = mul_102 * cos_28;  mul_102 = cos_28 = None
        cos_29 = torch.cos(einsum_3)
        mul_104 = all_parameters_b1 * add_4
        mul_105 = mul_104 * truediv;  mul_104 = None
        mul_106 = mul_105 * einsum_6;  mul_105 = None
        mul_107 = mul_106 * cos_29;  mul_106 = cos_29 = None
        add_22 = einsum_4 + einsum_3
        cos_30 = torch.cos(add_22);  add_22 = None
        mul_108 = all_parameters_bc * mul_6
        mul_109 = mul_108 * add_6;  mul_108 = None
        mul_110 = mul_109 * einsum_7;  mul_109 = None
        mul_111 = mul_110 * cos_30;  mul_110 = cos_30 = None
        all_constants_constant115 = self.all_constants.Constant115
        mul_112 = all_constants_constant115 * einsum_3;  all_constants_constant115 = None
        sin_8 = torch.sin(mul_112);  mul_112 = None
        all_constants_constant114 = self.all_constants.Constant114
        pow_13 = torch.pow(truediv, all_constants_constant114);  all_constants_constant114 = None
        mul_113 = add_4 * pow_13;  pow_13 = None
        mul_114 = mul_113 * einsum_6;  mul_113 = None
        mul_115 = mul_114 * einsum_5;  mul_114 = None
        mul_116 = mul_115 * sin_8;  mul_115 = sin_8 = None
        all_constants_constant116 = self.all_constants.Constant116
        truediv_4 = mul_116 / all_constants_constant116;  mul_116 = all_constants_constant116 = None
        sin_9 = torch.sin(einsum_3)
        cos_31 = torch.cos(einsum_4)
        all_constants_constant113 = self.all_constants.Constant113
        pow_14 = torch.pow(cos_31, all_constants_constant113);  cos_31 = all_constants_constant113 = None
        all_constants_constant112 = self.all_constants.Constant112
        pow_15 = torch.pow(einsum_6, all_constants_constant112);  all_constants_constant112 = None
        all_constants_constant111 = self.all_constants.Constant111
        pow_16 = torch.pow(mul_8, all_constants_constant111);  all_constants_constant111 = None
        mul_117 = pow_16 * truediv;  pow_16 = None
        mul_118 = mul_117 * pow_15;  mul_117 = pow_15 = None
        mul_119 = mul_118 * pow_14;  mul_118 = pow_14 = None
        mul_120 = mul_119 * sin_9;  mul_119 = sin_9 = None
        sin_10 = torch.sin(einsum_4)
        cos_32 = torch.cos(einsum_3)
        mul_121 = add_4 * truediv
        mul_122 = mul_121 * mul_54;  mul_121 = None
        mul_123 = mul_122 * cos_32;  mul_122 = cos_32 = None
        mul_124 = mul_123 * sin_10;  mul_123 = sin_10 = None
        sin_11 = torch.sin(einsum_3)
        all_constants_constant110 = self.all_constants.Constant110
        pow_17 = torch.pow(einsum_6, all_constants_constant110);  all_constants_constant110 = None
        mul_125 = add_4 * add_6
        mul_126 = mul_125 * truediv;  mul_125 = None
        mul_127 = mul_126 * pow_17;  mul_126 = pow_17 = None
        mul_128 = mul_127 * sin_11;  mul_127 = sin_11 = None
        cos_33 = torch.cos(einsum_3)
        cos_34 = torch.cos(einsum_4)
        getitem_8 = kwargs['action']
        relation_forward_sample_part13_w = self.all_constants.SamplePart13
        einsum_8 = torch.functional.einsum('bij,ki->bkj', getitem_8, relation_forward_sample_part13_w);  getitem_8 = relation_forward_sample_part13_w = None
        all_parameters_gear = self.all_parameters.gear
        mul_129 = einsum_8 * all_parameters_gear;  einsum_8 = all_parameters_gear = None
        mul_130 = mul_129 * mul_8
        mul_131 = mul_130 * truediv;  mul_130 = None
        mul_132 = mul_131 * cos_34;  mul_131 = cos_34 = None
        mul_133 = mul_132 * cos_33;  mul_132 = cos_33 = None
        mul_134 = all_parameters_b2 * add_4
        mul_135 = mul_134 * add_6;  mul_134 = None
        mul_136 = mul_135 * einsum_5;  mul_135 = None
        mul_137 = all_parameters_b2 * add_4
        mul_138 = mul_137 * add_6;  mul_137 = None
        mul_139 = mul_138 * einsum_6;  mul_138 = None
        all_constants_constant108 = self.all_constants.Constant108
        mul_140 = all_constants_constant108 * einsum_3;  all_constants_constant108 = None
        all_constants_constant107 = self.all_constants.Constant107
        mul_141 = all_constants_constant107 * einsum_4;  all_constants_constant107 = None
        add_23 = mul_141 + mul_140;  mul_141 = mul_140 = None
        sin_12 = torch.sin(add_23);  add_23 = None
        all_constants_constant106 = self.all_constants.Constant106
        pow_18 = torch.pow(einsum_5, all_constants_constant106);  all_constants_constant106 = None
        all_constants_constant105 = self.all_constants.Constant105
        pow_19 = torch.pow(mul_6, all_constants_constant105);  all_constants_constant105 = None
        mul_142 = pow_19 * add_6;  pow_19 = None
        mul_143 = mul_142 * pow_18;  mul_142 = pow_18 = None
        mul_144 = mul_143 * sin_12;  mul_143 = sin_12 = None
        all_constants_constant109 = self.all_constants.Constant109
        truediv_5 = mul_144 / all_constants_constant109;  mul_144 = all_constants_constant109 = None
        all_constants_constant103 = self.all_constants.Constant103
        mul_145 = all_constants_constant103 * einsum_3;  all_constants_constant103 = None
        sin_13 = torch.sin(mul_145);  mul_145 = None
        all_constants_constant102 = self.all_constants.Constant102
        pow_20 = torch.pow(einsum_5, all_constants_constant102);  all_constants_constant102 = None
        all_constants_constant101 = self.all_constants.Constant101
        pow_21 = torch.pow(truediv, all_constants_constant101);  all_constants_constant101 = None
        mul_146 = add_4 * pow_21;  pow_21 = None
        mul_147 = mul_146 * pow_20;  mul_146 = pow_20 = None
        mul_148 = mul_147 * sin_13;  mul_147 = sin_13 = None
        all_constants_constant104 = self.all_constants.Constant104
        truediv_6 = mul_148 / all_constants_constant104;  mul_148 = all_constants_constant104 = None
        cos_35 = torch.cos(einsum_4)
        all_constants_constant100 = self.all_constants.Constant100
        pow_22 = torch.pow(cos_35, all_constants_constant100);  cos_35 = all_constants_constant100 = None
        add_24 = einsum_4 + einsum_3
        sin_14 = torch.sin(add_24);  add_24 = None
        all_constants_constant35 = self.all_constants.Constant35
        mul_149 = all_constants_constant35 * all_parameters_m2;  all_constants_constant35 = all_parameters_m2 = None
        mul_150 = mul_149 * all_constants_gravity;  mul_149 = all_constants_gravity = None
        all_constants_constant36 = self.all_constants.Constant36
        mul_151 = mul_150 * all_constants_constant36;  mul_150 = all_constants_constant36 = None
        all_constants_constant99 = self.all_constants.Constant99
        pow_23 = torch.pow(mul_8, all_constants_constant99);  all_constants_constant99 = None
        mul_152 = pow_23 * mul_151;  pow_23 = None
        mul_153 = mul_152 * sin_14;  mul_152 = sin_14 = None
        mul_154 = mul_153 * pow_22;  mul_153 = pow_22 = None
        add_25 = einsum_4 + einsum_3
        sin_15 = torch.sin(add_25);  add_25 = None
        mul_155 = add_4 * add_6
        mul_156 = mul_155 * mul_151;  mul_155 = None
        mul_157 = mul_156 * sin_15;  mul_156 = sin_15 = None
        cos_36 = torch.cos(einsum_4)
        all_constants_constant98 = self.all_constants.Constant98
        pow_24 = torch.pow(cos_36, all_constants_constant98);  cos_36 = all_constants_constant98 = None
        all_constants_constant97 = self.all_constants.Constant97
        pow_25 = torch.pow(mul_8, all_constants_constant97);  all_constants_constant97 = None
        mul_158 = all_parameters_b2 * pow_25;  pow_25 = None
        mul_159 = mul_158 * einsum_5;  mul_158 = None
        mul_160 = mul_159 * pow_24;  mul_159 = pow_24 = None
        cos_37 = torch.cos(einsum_4)
        all_constants_constant96 = self.all_constants.Constant96
        pow_26 = torch.pow(cos_37, all_constants_constant96);  cos_37 = all_constants_constant96 = None
        all_constants_constant95 = self.all_constants.Constant95
        pow_27 = torch.pow(mul_8, all_constants_constant95);  all_constants_constant95 = None
        mul_161 = all_parameters_b2 * pow_27;  pow_27 = None
        mul_162 = mul_161 * einsum_6;  mul_161 = None
        mul_163 = mul_162 * pow_26;  mul_162 = pow_26 = None
        add_26 = einsum_4 + einsum_3
        cos_38 = torch.cos(add_26);  add_26 = None
        mul_164 = mul_129 * mul_6
        mul_165 = mul_164 * add_6;  mul_164 = None
        mul_166 = mul_165 * cos_38;  mul_165 = cos_38 = None
        add_27 = mul_166 + mul_163;  mul_166 = mul_163 = None
        sub_2 = add_27 - mul_160;  add_27 = mul_160 = None
        sub_3 = sub_2 - mul_157;  sub_2 = mul_157 = None
        add_28 = sub_3 + mul_154;  sub_3 = mul_154 = None
        add_29 = add_28 + truediv_6;  add_28 = truediv_6 = None
        add_30 = add_29 + truediv_5;  add_29 = truediv_5 = None
        sub_4 = add_30 - mul_139;  add_30 = mul_139 = None
        add_31 = sub_4 + mul_136;  sub_4 = mul_136 = None
        sub_5 = add_31 - mul_133;  add_31 = mul_133 = None
        add_32 = sub_5 + mul_128;  sub_5 = mul_128 = None
        add_33 = add_32 + mul_124;  add_32 = mul_124 = None
        sub_6 = add_33 - mul_120;  add_33 = mul_120 = None
        add_34 = sub_6 + truediv_4;  sub_6 = truediv_4 = None
        sub_7 = add_34 - mul_111;  add_34 = mul_111 = None
        sub_8 = sub_7 - mul_107;  sub_7 = mul_107 = None
        sub_9 = sub_8 - mul_103;  sub_8 = mul_103 = None
        add_35 = sub_9 + mul_99;  sub_9 = mul_99 = None
        add_36 = add_35 + truediv_3;  add_35 = truediv_3 = None
        add_37 = add_36 + mul_89;  add_36 = mul_89 = None
        add_38 = add_37 + mul_84;  add_37 = mul_84 = None
        sub_10 = add_38 - mul_79;  add_38 = mul_79 = None
        add_39 = sub_10 + mul_74;  sub_10 = mul_74 = None
        sub_11 = add_39 - mul_69;  add_39 = mul_69 = None
        add_40 = sub_11 + mul_64;  sub_11 = mul_64 = None
        sub_12 = add_40 - mul_59;  add_40 = mul_59 = None
        sub_13 = sub_12 - mul_51;  sub_12 = mul_51 = None
        sub_14 = sub_13 - mul_45;  sub_13 = mul_45 = None
        sub_15 = sub_14 - mul_39;  sub_14 = mul_39 = None
        sub_16 = sub_15 - mul_32;  sub_15 = mul_32 = None
        truediv_7 = sub_16 / add_10;  sub_16 = None
        getitem_9 = kwargs['Xddth2']
        relation_forward_sample_part1026_w = self.all_constants.SamplePart1026
        einsum_9 = torch.functional.einsum('bij,ki->bkj', getitem_9, relation_forward_sample_part1026_w);  getitem_9 = relation_forward_sample_part1026_w = None
        cos_39 = torch.cos(einsum_4)
        add_41 = einsum_4 + einsum_3
        sin_16 = torch.sin(add_41);  add_41 = None
        mul_167 = mul_8 * mul_6
        mul_168 = mul_167 * add_2;  mul_167 = None
        mul_169 = mul_168 * einsum_6;  mul_168 = None
        mul_170 = mul_169 * einsum_5;  mul_169 = None
        mul_171 = mul_170 * sin_16;  mul_170 = sin_16 = None
        mul_172 = mul_171 * cos_39;  mul_171 = cos_39 = None
        sin_17 = torch.sin(einsum_4)
        cos_40 = torch.cos(einsum_3)
        add_42 = einsum_4 + einsum_3
        cos_41 = torch.cos(add_42);  add_42 = None
        all_constants_constant94 = self.all_constants.Constant94
        pow_28 = torch.pow(einsum_6, all_constants_constant94);  all_constants_constant94 = None
        mul_173 = mul_8 * mul_6
        mul_174 = mul_173 * truediv;  mul_173 = None
        mul_175 = mul_174 * pow_28;  mul_174 = pow_28 = None
        mul_176 = mul_175 * cos_41;  mul_175 = cos_41 = None
        mul_177 = mul_176 * cos_40;  mul_176 = cos_40 = None
        mul_178 = mul_177 * sin_17;  mul_177 = sin_17 = None
        sin_18 = torch.sin(einsum_3)
        cos_42 = torch.cos(einsum_4)
        add_43 = einsum_4 + einsum_3
        cos_43 = torch.cos(add_43);  add_43 = None
        all_constants_constant93 = self.all_constants.Constant93
        pow_29 = torch.pow(einsum_6, all_constants_constant93);  all_constants_constant93 = None
        mul_179 = mul_8 * mul_6
        mul_180 = mul_179 * truediv;  mul_179 = None
        mul_181 = mul_180 * pow_29;  mul_180 = pow_29 = None
        mul_182 = mul_181 * cos_43;  mul_181 = cos_43 = None
        mul_183 = mul_182 * cos_42;  mul_182 = cos_42 = None
        mul_184 = mul_183 * sin_18;  mul_183 = sin_18 = None
        cos_44 = torch.cos(einsum_3)
        add_44 = einsum_4 + einsum_3
        sin_19 = torch.sin(add_44);  add_44 = None
        add_45 = einsum_4 + einsum_3
        cos_45 = torch.cos(add_45);  add_45 = None
        all_constants_constant92 = self.all_constants.Constant92
        pow_30 = torch.pow(mul_6, all_constants_constant92);  all_constants_constant92 = None
        mul_185 = pow_30 * truediv;  pow_30 = None
        mul_186 = mul_185 * einsum_6;  mul_185 = None
        mul_187 = mul_186 * einsum_5;  mul_186 = None
        mul_188 = mul_187 * cos_45;  mul_187 = cos_45 = None
        mul_189 = mul_188 * sin_19;  mul_188 = sin_19 = None
        mul_190 = mul_189 * cos_44;  mul_189 = cos_44 = None
        cos_46 = torch.cos(einsum_4)
        add_46 = einsum_4 + einsum_3
        sin_20 = torch.sin(add_46);  add_46 = None
        all_constants_constant91 = self.all_constants.Constant91
        pow_31 = torch.pow(einsum_5, all_constants_constant91);  all_constants_constant91 = None
        mul_191 = mul_8 * mul_6
        mul_192 = mul_191 * add_2;  mul_191 = None
        mul_193 = mul_192 * pow_31;  mul_192 = pow_31 = None
        mul_194 = mul_193 * sin_20;  mul_193 = sin_20 = None
        mul_195 = mul_194 * cos_46;  mul_194 = cos_46 = None
        cos_47 = torch.cos(einsum_4)
        add_47 = einsum_4 + einsum_3
        sin_21 = torch.sin(add_47);  add_47 = None
        add_48 = einsum_4 + einsum_3
        cos_48 = torch.cos(add_48);  add_48 = None
        mul_196 = mul_8 * mul_6
        mul_197 = mul_196 * mul_151;  mul_196 = None
        mul_198 = mul_197 * cos_48;  mul_197 = cos_48 = None
        mul_199 = mul_198 * sin_21;  mul_198 = sin_21 = None
        mul_200 = mul_199 * cos_47;  mul_199 = cos_47 = None
        cos_49 = torch.cos(einsum_3)
        add_49 = einsum_4 + einsum_3
        sin_22 = torch.sin(add_49);  add_49 = None
        add_50 = einsum_4 + einsum_3
        cos_50 = torch.cos(add_50);  add_50 = None
        all_constants_constant90 = self.all_constants.Constant90
        pow_32 = torch.pow(einsum_5, all_constants_constant90);  all_constants_constant90 = None
        all_constants_constant89 = self.all_constants.Constant89
        pow_33 = torch.pow(mul_6, all_constants_constant89);  all_constants_constant89 = None
        mul_201 = pow_33 * truediv;  pow_33 = None
        mul_202 = mul_201 * pow_32;  mul_201 = pow_32 = None
        mul_203 = mul_202 * cos_50;  mul_202 = cos_50 = None
        mul_204 = mul_203 * sin_22;  mul_203 = sin_22 = None
        mul_205 = mul_204 * cos_49;  mul_204 = cos_49 = None
        cos_51 = torch.cos(einsum_3)
        add_51 = einsum_4 + einsum_3
        cos_52 = torch.cos(add_51);  add_51 = None
        mul_206 = all_parameters_bc * mul_6
        mul_207 = mul_206 * truediv;  mul_206 = None
        mul_208 = mul_207 * einsum_7;  mul_207 = None
        mul_209 = mul_208 * cos_52;  mul_208 = cos_52 = None
        mul_210 = mul_209 * cos_51;  mul_209 = cos_51 = None
        cos_53 = torch.cos(einsum_4)
        add_52 = einsum_4 + einsum_3
        cos_54 = torch.cos(add_52);  add_52 = None
        mul_211 = all_parameters_b2 * mul_8
        mul_212 = mul_211 * mul_6;  mul_211 = None
        mul_213 = mul_212 * einsum_5;  mul_212 = None
        mul_214 = mul_213 * cos_54;  mul_213 = cos_54 = None
        mul_215 = mul_214 * cos_53;  mul_214 = cos_53 = None
        cos_55 = torch.cos(einsum_4)
        add_53 = einsum_4 + einsum_3
        cos_56 = torch.cos(add_53);  add_53 = None
        mul_216 = all_parameters_b2 * mul_8
        mul_217 = mul_216 * mul_6;  mul_216 = None
        mul_218 = mul_217 * einsum_6;  mul_217 = None
        mul_219 = mul_218 * cos_56;  mul_218 = cos_56 = None
        mul_220 = mul_219 * cos_55;  mul_219 = cos_55 = None
        sin_23 = torch.sin(einsum_3)
        add_54 = einsum_4 + einsum_3
        cos_57 = torch.cos(add_54);  add_54 = None
        all_constants_constant88 = self.all_constants.Constant88
        pow_34 = torch.pow(cos_57, all_constants_constant88);  cos_57 = all_constants_constant88 = None
        all_constants_constant87 = self.all_constants.Constant87
        pow_35 = torch.pow(mul_6, all_constants_constant87);  all_constants_constant87 = None
        mul_221 = pow_35 * truediv;  pow_35 = None
        mul_222 = mul_221 * einsum_6;  mul_221 = None
        mul_223 = mul_222 * einsum_5;  mul_222 = None
        mul_224 = mul_223 * pow_34;  mul_223 = pow_34 = None
        mul_225 = mul_224 * sin_23;  mul_224 = sin_23 = None
        sin_24 = torch.sin(einsum_3)
        mul_226 = add_4 * truediv
        mul_227 = mul_226 * add_2;  mul_226 = None
        mul_228 = mul_227 * einsum_6;  mul_227 = None
        mul_229 = mul_228 * einsum_5;  mul_228 = None
        mul_230 = mul_229 * sin_24;  mul_229 = sin_24 = None
        cos_58 = torch.cos(einsum_3)
        add_55 = einsum_4 + einsum_3
        cos_59 = torch.cos(add_55);  add_55 = None
        mul_231 = mul_129 * mul_6
        mul_232 = mul_231 * truediv;  mul_231 = None
        mul_233 = mul_232 * cos_59;  mul_232 = cos_59 = None
        mul_234 = mul_233 * cos_58;  mul_233 = cos_58 = None
        cos_60 = torch.cos(einsum_4)
        mul_235 = all_parameters_bc * mul_8
        mul_236 = mul_235 * add_2;  mul_235 = None
        mul_237 = mul_236 * einsum_7;  mul_236 = None
        mul_238 = mul_237 * cos_60;  mul_237 = cos_60 = None
        cos_61 = torch.cos(einsum_3)
        mul_239 = all_parameters_b2 * add_4
        mul_240 = mul_239 * truediv;  mul_239 = None
        mul_241 = mul_240 * einsum_5;  mul_240 = None
        mul_242 = mul_241 * cos_61;  mul_241 = cos_61 = None
        cos_62 = torch.cos(einsum_3)
        mul_243 = all_parameters_b2 * add_4
        mul_244 = mul_243 * truediv;  mul_243 = None
        mul_245 = mul_244 * einsum_6;  mul_244 = None
        mul_246 = mul_245 * cos_62;  mul_245 = cos_62 = None
        sin_25 = torch.sin(einsum_3)
        add_56 = einsum_4 + einsum_3
        cos_63 = torch.cos(add_56);  add_56 = None
        all_constants_constant86 = self.all_constants.Constant86
        pow_36 = torch.pow(cos_63, all_constants_constant86);  cos_63 = all_constants_constant86 = None
        all_constants_constant85 = self.all_constants.Constant85
        pow_37 = torch.pow(einsum_5, all_constants_constant85);  all_constants_constant85 = None
        all_constants_constant84 = self.all_constants.Constant84
        pow_38 = torch.pow(mul_6, all_constants_constant84);  all_constants_constant84 = None
        mul_247 = pow_38 * truediv;  pow_38 = None
        mul_248 = mul_247 * pow_37;  mul_247 = pow_37 = None
        mul_249 = mul_248 * pow_36;  mul_248 = pow_36 = None
        mul_250 = mul_249 * sin_25;  mul_249 = sin_25 = None
        sin_26 = torch.sin(einsum_3)
        all_constants_constant83 = self.all_constants.Constant83
        pow_39 = torch.pow(einsum_5, all_constants_constant83);  all_constants_constant83 = None
        mul_251 = add_4 * truediv
        mul_252 = mul_251 * add_2;  mul_251 = None
        mul_253 = mul_252 * pow_39;  mul_252 = pow_39 = None
        mul_254 = mul_253 * sin_26;  mul_253 = sin_26 = None
        cos_64 = torch.cos(einsum_3)
        add_57 = einsum_4 + einsum_3
        sin_27 = torch.sin(add_57);  add_57 = None
        mul_255 = add_4 * truediv
        mul_256 = mul_255 * mul_151;  mul_255 = None
        mul_257 = mul_256 * sin_27;  mul_256 = sin_27 = None
        mul_258 = mul_257 * cos_64;  mul_257 = cos_64 = None
        sin_28 = torch.sin(einsum_4)
        cos_65 = torch.cos(einsum_4)
        all_constants_constant82 = self.all_constants.Constant82
        pow_40 = torch.pow(einsum_6, all_constants_constant82);  all_constants_constant82 = None
        all_constants_constant81 = self.all_constants.Constant81
        pow_41 = torch.pow(mul_8, all_constants_constant81);  all_constants_constant81 = None
        mul_259 = pow_41 * add_2;  pow_41 = None
        mul_260 = mul_259 * pow_40;  mul_259 = pow_40 = None
        mul_261 = mul_260 * cos_65;  mul_260 = cos_65 = None
        mul_262 = mul_261 * sin_28;  mul_261 = sin_28 = None
        sin_29 = torch.sin(einsum_3)
        cos_66 = torch.cos(einsum_3)
        all_constants_constant80 = self.all_constants.Constant80
        pow_42 = torch.pow(einsum_6, all_constants_constant80);  all_constants_constant80 = None
        all_constants_constant79 = self.all_constants.Constant79
        pow_43 = torch.pow(truediv, all_constants_constant79);  all_constants_constant79 = None
        mul_263 = add_4 * pow_43;  pow_43 = None
        mul_264 = mul_263 * pow_42;  mul_263 = pow_42 = None
        mul_265 = mul_264 * cos_66;  mul_264 = cos_66 = None
        mul_266 = mul_265 * sin_29;  mul_265 = sin_29 = None
        add_58 = einsum_4 + einsum_3
        cos_67 = torch.cos(add_58);  add_58 = None
        all_constants_constant78 = self.all_constants.Constant78
        pow_44 = torch.pow(cos_67, all_constants_constant78);  cos_67 = all_constants_constant78 = None
        all_constants_constant77 = self.all_constants.Constant77
        pow_45 = torch.pow(mul_6, all_constants_constant77);  all_constants_constant77 = None
        mul_267 = all_parameters_b2 * pow_45;  pow_45 = None
        mul_268 = mul_267 * einsum_5;  mul_267 = None
        mul_269 = mul_268 * pow_44;  mul_268 = pow_44 = None
        add_59 = einsum_4 + einsum_3
        cos_68 = torch.cos(add_59);  add_59 = None
        all_constants_constant76 = self.all_constants.Constant76
        pow_46 = torch.pow(cos_68, all_constants_constant76);  cos_68 = all_constants_constant76 = None
        all_constants_constant75 = self.all_constants.Constant75
        pow_47 = torch.pow(mul_6, all_constants_constant75);  all_constants_constant75 = None
        mul_270 = all_parameters_b2 * pow_47;  pow_47 = None
        mul_271 = mul_270 * einsum_6;  mul_270 = None
        mul_272 = mul_271 * pow_46;  mul_271 = pow_46 = None
        add_60 = einsum_4 + einsum_3
        cos_69 = torch.cos(add_60);  add_60 = None
        all_constants_constant74 = self.all_constants.Constant74
        pow_48 = torch.pow(cos_69, all_constants_constant74);  cos_69 = all_constants_constant74 = None
        all_constants_constant73 = self.all_constants.Constant73
        pow_49 = torch.pow(mul_6, all_constants_constant73);  all_constants_constant73 = None
        mul_273 = all_parameters_b1 * pow_49;  pow_49 = None
        mul_274 = mul_273 * einsum_6;  mul_273 = None
        mul_275 = mul_274 * pow_48;  mul_274 = pow_48 = None
        mul_276 = all_parameters_b2 * add_4
        mul_277 = mul_276 * add_2;  mul_276 = None
        mul_278 = mul_277 * einsum_5;  mul_277 = None
        mul_279 = all_parameters_b2 * add_4
        mul_280 = mul_279 * add_2;  mul_279 = None
        mul_281 = mul_280 * einsum_6;  mul_280 = None
        mul_282 = all_parameters_b1 * add_4
        mul_283 = mul_282 * add_2;  mul_282 = None
        mul_284 = mul_283 * einsum_6;  mul_283 = None
        sin_30 = torch.sin(einsum_4)
        add_61 = einsum_4 + einsum_3
        cos_70 = torch.cos(add_61);  add_61 = None
        all_constants_constant72 = self.all_constants.Constant72
        pow_50 = torch.pow(cos_70, all_constants_constant72);  cos_70 = all_constants_constant72 = None
        all_constants_constant71 = self.all_constants.Constant71
        pow_51 = torch.pow(mul_6, all_constants_constant71);  all_constants_constant71 = None
        mul_285 = pow_51 * mul_54;  pow_51 = None
        mul_286 = mul_285 * pow_50;  mul_285 = pow_50 = None
        mul_287 = mul_286 * sin_30;  mul_286 = sin_30 = None
        sin_31 = torch.sin(einsum_4)
        mul_288 = add_4 * add_2;  add_4 = None
        mul_289 = mul_288 * mul_54;  mul_288 = None
        mul_290 = mul_289 * sin_31;  mul_289 = sin_31 = None
        cos_71 = torch.cos(einsum_4)
        mul_291 = mul_129 * mul_8
        mul_292 = mul_291 * add_2;  mul_291 = None
        mul_293 = mul_292 * cos_71;  mul_292 = cos_71 = None
        sub_17 = mul_293 - mul_290;  mul_293 = mul_290 = None
        add_62 = sub_17 + mul_287;  sub_17 = mul_287 = None
        add_63 = add_62 + mul_284;  add_62 = mul_284 = None
        add_64 = add_63 + mul_281;  add_63 = mul_281 = None
        sub_18 = add_64 - mul_278;  add_64 = mul_278 = None
        sub_19 = sub_18 - mul_275;  sub_18 = mul_275 = None
        sub_20 = sub_19 - mul_272;  sub_19 = mul_272 = None
        add_65 = sub_20 + mul_269;  sub_20 = mul_269 = None
        sub_21 = add_65 - mul_266;  add_65 = mul_266 = None
        add_66 = sub_21 + mul_262;  sub_21 = mul_262 = None
        add_67 = add_66 + mul_258;  add_66 = mul_258 = None
        sub_22 = add_67 - mul_254;  add_67 = mul_254 = None
        add_68 = sub_22 + mul_250;  sub_22 = mul_250 = None
        add_69 = add_68 + mul_246;  add_68 = mul_246 = None
        sub_23 = add_69 - mul_242;  add_69 = mul_242 = None
        sub_24 = sub_23 - mul_238;  sub_23 = mul_238 = None
        sub_25 = sub_24 - mul_234;  sub_24 = mul_234 = None
        sub_26 = sub_25 - mul_230;  sub_25 = mul_230 = None
        add_70 = sub_26 + mul_225;  sub_26 = mul_225 = None
        sub_27 = add_70 - mul_220;  add_70 = mul_220 = None
        add_71 = sub_27 + mul_215;  sub_27 = mul_215 = None
        add_72 = add_71 + mul_210;  add_71 = mul_210 = None
        sub_28 = add_72 - mul_205;  add_72 = mul_205 = None
        sub_29 = sub_28 - mul_200;  sub_28 = mul_200 = None
        add_73 = sub_29 + mul_195;  sub_29 = mul_195 = None
        sub_30 = add_73 - mul_190;  add_73 = mul_190 = None
        add_74 = sub_30 + mul_184;  sub_30 = mul_184 = None
        sub_31 = add_74 - mul_178;  add_74 = mul_178 = None
        add_75 = sub_31 + mul_172;  sub_31 = mul_172 = None
        truediv_8 = add_75 / add_10;  add_75 = None
        getitem_10 = kwargs['Xddth1']
        relation_forward_sample_part1024_w = self.all_constants.SamplePart1024
        einsum_10 = torch.functional.einsum('bij,ki->bkj', getitem_10, relation_forward_sample_part1024_w);  getitem_10 = relation_forward_sample_part1024_w = None
        sin_32 = torch.sin(einsum_3)
        cos_72 = torch.cos(einsum_3)
        add_76 = einsum_4 + einsum_3
        cos_73 = torch.cos(add_76);  add_76 = None
        all_constants_constant70 = self.all_constants.Constant70
        pow_52 = torch.pow(truediv, all_constants_constant70);  all_constants_constant70 = None
        mul_294 = mul_6 * pow_52;  pow_52 = None
        mul_295 = mul_294 * einsum_6;  mul_294 = None
        mul_296 = mul_295 * einsum_5;  mul_295 = None
        mul_297 = mul_296 * cos_73;  mul_296 = cos_73 = None
        mul_298 = mul_297 * cos_72;  mul_297 = cos_72 = None
        mul_299 = mul_298 * sin_32;  mul_298 = sin_32 = None
        sin_33 = torch.sin(einsum_3)
        cos_74 = torch.cos(einsum_4)
        mul_300 = mul_8 * truediv
        mul_301 = mul_300 * add_2;  mul_300 = None
        mul_302 = mul_301 * einsum_6;  mul_301 = None
        mul_303 = mul_302 * einsum_5;  mul_302 = None
        mul_304 = mul_303 * cos_74;  mul_303 = cos_74 = None
        mul_305 = mul_304 * sin_33;  mul_304 = sin_33 = None
        sin_34 = torch.sin(einsum_4)
        cos_75 = torch.cos(einsum_3)
        add_77 = einsum_4 + einsum_3
        cos_76 = torch.cos(add_77);  add_77 = None
        mul_306 = mul_6 * truediv
        mul_307 = mul_306 * mul_54;  mul_306 = None
        mul_308 = mul_307 * cos_76;  mul_307 = cos_76 = None
        mul_309 = mul_308 * cos_75;  mul_308 = cos_75 = None
        mul_310 = mul_309 * sin_34;  mul_309 = sin_34 = None
        cos_77 = torch.cos(einsum_3)
        cos_78 = torch.cos(einsum_4)
        add_78 = einsum_4 + einsum_3
        sin_35 = torch.sin(add_78);  add_78 = None
        mul_311 = mul_8 * truediv
        mul_312 = mul_311 * mul_151;  mul_311 = None
        mul_313 = mul_312 * sin_35;  mul_312 = sin_35 = None
        mul_314 = mul_313 * cos_78;  mul_313 = cos_78 = None
        mul_315 = mul_314 * cos_77;  mul_314 = cos_77 = None
        sin_36 = torch.sin(einsum_3)
        add_79 = einsum_4 + einsum_3
        cos_79 = torch.cos(add_79);  add_79 = None
        all_constants_constant69 = self.all_constants.Constant69
        pow_53 = torch.pow(einsum_6, all_constants_constant69);  all_constants_constant69 = None
        mul_316 = mul_6 * add_6
        mul_317 = mul_316 * truediv;  mul_316 = None
        mul_318 = mul_317 * pow_53;  mul_317 = pow_53 = None
        mul_319 = mul_318 * cos_79;  mul_318 = cos_79 = None
        mul_320 = mul_319 * sin_36;  mul_319 = sin_36 = None
        sin_37 = torch.sin(einsum_3)
        cos_80 = torch.cos(einsum_3)
        cos_81 = torch.cos(einsum_4)
        all_constants_constant68 = self.all_constants.Constant68
        pow_54 = torch.pow(einsum_6, all_constants_constant68);  all_constants_constant68 = None
        all_constants_constant67 = self.all_constants.Constant67
        pow_55 = torch.pow(truediv, all_constants_constant67);  all_constants_constant67 = None
        mul_321 = mul_8 * pow_55;  pow_55 = None
        mul_322 = mul_321 * pow_54;  mul_321 = pow_54 = None
        mul_323 = mul_322 * cos_81;  mul_322 = cos_81 = None
        mul_324 = mul_323 * cos_80;  mul_323 = cos_80 = None
        mul_325 = mul_324 * sin_37;  mul_324 = sin_37 = None
        sin_38 = torch.sin(einsum_3)
        cos_82 = torch.cos(einsum_3)
        add_80 = einsum_4 + einsum_3
        cos_83 = torch.cos(add_80);  add_80 = None
        all_constants_constant66 = self.all_constants.Constant66
        pow_56 = torch.pow(einsum_5, all_constants_constant66);  all_constants_constant66 = None
        all_constants_constant65 = self.all_constants.Constant65
        pow_57 = torch.pow(truediv, all_constants_constant65);  all_constants_constant65 = None
        mul_326 = mul_6 * pow_57;  pow_57 = None
        mul_327 = mul_326 * pow_56;  mul_326 = pow_56 = None
        mul_328 = mul_327 * cos_83;  mul_327 = cos_83 = None
        mul_329 = mul_328 * cos_82;  mul_328 = cos_82 = None
        mul_330 = mul_329 * sin_38;  mul_329 = sin_38 = None
        cos_84 = torch.cos(einsum_3)
        cos_85 = torch.cos(einsum_4)
        mul_331 = all_parameters_b2 * mul_8
        mul_332 = mul_331 * truediv;  mul_331 = None
        mul_333 = mul_332 * einsum_5;  mul_332 = None
        mul_334 = mul_333 * cos_85;  mul_333 = cos_85 = None
        mul_335 = mul_334 * cos_84;  mul_334 = cos_84 = None
        cos_86 = torch.cos(einsum_3)
        cos_87 = torch.cos(einsum_4)
        mul_336 = all_parameters_b2 * mul_8
        mul_337 = mul_336 * truediv;  mul_336 = None
        mul_338 = mul_337 * einsum_6;  mul_337 = None
        mul_339 = mul_338 * cos_87;  mul_338 = cos_87 = None
        mul_340 = mul_339 * cos_86;  mul_339 = cos_86 = None
        cos_88 = torch.cos(einsum_3)
        add_81 = einsum_4 + einsum_3
        cos_89 = torch.cos(add_81);  add_81 = None
        mul_341 = all_parameters_b2 * mul_6
        mul_342 = mul_341 * truediv;  mul_341 = None
        mul_343 = mul_342 * einsum_5;  mul_342 = None
        mul_344 = mul_343 * cos_89;  mul_343 = cos_89 = None
        mul_345 = mul_344 * cos_88;  mul_344 = cos_88 = None
        cos_90 = torch.cos(einsum_3)
        add_82 = einsum_4 + einsum_3
        cos_91 = torch.cos(add_82);  add_82 = None
        mul_346 = all_parameters_b2 * mul_6
        mul_347 = mul_346 * truediv;  mul_346 = None
        mul_348 = mul_347 * einsum_6;  mul_347 = None
        mul_349 = mul_348 * cos_91;  mul_348 = cos_91 = None
        mul_350 = mul_349 * cos_90;  mul_349 = cos_90 = None
        cos_92 = torch.cos(einsum_3)
        add_83 = einsum_4 + einsum_3
        cos_93 = torch.cos(add_83);  add_83 = None
        mul_351 = all_parameters_b1 * mul_6
        mul_352 = mul_351 * truediv;  mul_351 = None
        mul_353 = mul_352 * einsum_6;  mul_352 = None
        mul_354 = mul_353 * cos_93;  mul_353 = cos_93 = None
        mul_355 = mul_354 * cos_92;  mul_354 = cos_92 = None
        cos_94 = torch.cos(einsum_3)
        all_constants_constant64 = self.all_constants.Constant64
        pow_58 = torch.pow(cos_94, all_constants_constant64);  cos_94 = all_constants_constant64 = None
        add_84 = einsum_4 + einsum_3
        sin_39 = torch.sin(add_84);  add_84 = None
        all_constants_constant63 = self.all_constants.Constant63
        pow_59 = torch.pow(truediv, all_constants_constant63);  all_constants_constant63 = None
        mul_356 = mul_6 * pow_59;  pow_59 = None
        mul_357 = mul_356 * einsum_6;  mul_356 = None
        mul_358 = mul_357 * einsum_5;  mul_357 = None
        mul_359 = mul_358 * sin_39;  mul_358 = sin_39 = None
        mul_360 = mul_359 * pow_58;  mul_359 = pow_58 = None
        add_85 = einsum_4 + einsum_3
        sin_40 = torch.sin(add_85);  add_85 = None
        mul_361 = mul_6 * add_6
        mul_362 = mul_361 * add_2;  mul_361 = None
        mul_363 = mul_362 * einsum_6;  mul_362 = None
        mul_364 = mul_363 * einsum_5;  mul_363 = None
        mul_365 = mul_364 * sin_40;  mul_364 = sin_40 = None
        sin_41 = torch.sin(einsum_3)
        cos_95 = torch.cos(einsum_4)
        all_constants_constant62 = self.all_constants.Constant62
        pow_60 = torch.pow(einsum_5, all_constants_constant62);  all_constants_constant62 = None
        mul_366 = mul_8 * truediv
        mul_367 = mul_366 * add_2;  mul_366 = None
        mul_368 = mul_367 * pow_60;  mul_367 = pow_60 = None
        mul_369 = mul_368 * cos_95;  mul_368 = cos_95 = None
        mul_370 = mul_369 * sin_41;  mul_369 = sin_41 = None
        cos_96 = torch.cos(einsum_4)
        mul_371 = all_parameters_b2 * mul_8
        mul_372 = mul_371 * add_2;  mul_371 = None
        mul_373 = mul_372 * einsum_5;  mul_372 = None
        mul_374 = mul_373 * cos_96;  mul_373 = cos_96 = None
        cos_97 = torch.cos(einsum_4)
        mul_375 = all_parameters_b2 * mul_8
        mul_376 = mul_375 * add_2;  mul_375 = None
        mul_377 = mul_376 * einsum_6;  mul_376 = None
        mul_378 = mul_377 * cos_97;  mul_377 = cos_97 = None
        cos_98 = torch.cos(einsum_4)
        mul_379 = all_parameters_b1 * mul_8;  all_parameters_b1 = None
        mul_380 = mul_379 * add_2;  mul_379 = None
        mul_381 = mul_380 * einsum_6;  mul_380 = None
        mul_382 = mul_381 * cos_98;  mul_381 = cos_98 = None
        add_86 = einsum_4 + einsum_3
        cos_99 = torch.cos(add_86);  add_86 = None
        mul_383 = all_parameters_b2 * mul_6
        mul_384 = mul_383 * add_6;  mul_383 = None
        mul_385 = mul_384 * einsum_5;  mul_384 = None
        mul_386 = mul_385 * cos_99;  mul_385 = cos_99 = None
        add_87 = einsum_4 + einsum_3
        cos_100 = torch.cos(add_87);  add_87 = None
        mul_387 = all_parameters_b2 * mul_6;  all_parameters_b2 = None
        mul_388 = mul_387 * add_6;  mul_387 = None
        mul_389 = mul_388 * einsum_6;  mul_388 = None
        mul_390 = mul_389 * cos_100;  mul_389 = cos_100 = None
        sin_42 = torch.sin(einsum_4)
        cos_101 = torch.cos(einsum_3)
        all_constants_constant61 = self.all_constants.Constant61
        pow_61 = torch.pow(cos_101, all_constants_constant61);  cos_101 = all_constants_constant61 = None
        all_constants_constant60 = self.all_constants.Constant60
        pow_62 = torch.pow(einsum_6, all_constants_constant60);  all_constants_constant60 = None
        all_constants_constant59 = self.all_constants.Constant59
        pow_63 = torch.pow(truediv, all_constants_constant59);  all_constants_constant59 = None
        mul_391 = mul_8 * pow_63;  pow_63 = None
        mul_392 = mul_391 * pow_62;  mul_391 = pow_62 = None
        mul_393 = mul_392 * pow_61;  mul_392 = pow_61 = None
        mul_394 = mul_393 * sin_42;  mul_393 = sin_42 = None
        cos_102 = torch.cos(einsum_3)
        all_constants_constant58 = self.all_constants.Constant58
        pow_64 = torch.pow(cos_102, all_constants_constant58);  cos_102 = all_constants_constant58 = None
        add_88 = einsum_4 + einsum_3
        sin_43 = torch.sin(add_88);  add_88 = None
        all_constants_constant57 = self.all_constants.Constant57
        pow_65 = torch.pow(einsum_5, all_constants_constant57);  all_constants_constant57 = None
        all_constants_constant56 = self.all_constants.Constant56
        pow_66 = torch.pow(truediv, all_constants_constant56);  all_constants_constant56 = None
        mul_395 = mul_6 * pow_66;  pow_66 = None
        mul_396 = mul_395 * pow_65;  mul_395 = pow_65 = None
        mul_397 = mul_396 * sin_43;  mul_396 = sin_43 = None
        mul_398 = mul_397 * pow_64;  mul_397 = pow_64 = None
        sin_44 = torch.sin(einsum_4)
        all_constants_constant55 = self.all_constants.Constant55
        pow_67 = torch.pow(einsum_6, all_constants_constant55);  einsum_6 = all_constants_constant55 = None
        mul_399 = mul_8 * add_6
        mul_400 = mul_399 * add_2;  mul_399 = None
        mul_401 = mul_400 * pow_67;  mul_400 = pow_67 = None
        mul_402 = mul_401 * sin_44;  mul_401 = sin_44 = None
        add_89 = einsum_4 + einsum_3
        sin_45 = torch.sin(add_89);  add_89 = None
        all_constants_constant54 = self.all_constants.Constant54
        pow_68 = torch.pow(einsum_5, all_constants_constant54);  einsum_5 = all_constants_constant54 = None
        mul_403 = mul_6 * add_6
        mul_404 = mul_403 * add_2;  mul_403 = None
        mul_405 = mul_404 * pow_68;  mul_404 = pow_68 = None
        mul_406 = mul_405 * sin_45;  mul_405 = sin_45 = None
        all_constants_constant52 = self.all_constants.Constant52
        mul_407 = all_constants_constant52 * einsum_4;  all_constants_constant52 = None
        sin_46 = torch.sin(mul_407);  mul_407 = None
        mul_408 = mul_8 * add_2;  mul_8 = None
        mul_409 = mul_408 * mul_54;  mul_408 = mul_54 = None
        mul_410 = mul_409 * sin_46;  mul_409 = sin_46 = None
        all_constants_constant53 = self.all_constants.Constant53
        truediv_9 = mul_410 / all_constants_constant53;  mul_410 = all_constants_constant53 = None
        cos_103 = torch.cos(einsum_3)
        all_constants_constant51 = self.all_constants.Constant51
        pow_69 = torch.pow(cos_103, all_constants_constant51);  cos_103 = all_constants_constant51 = None
        all_constants_constant50 = self.all_constants.Constant50
        pow_70 = torch.pow(truediv, all_constants_constant50);  all_constants_constant50 = None
        mul_411 = all_parameters_bc * pow_70;  pow_70 = None
        mul_412 = mul_411 * einsum_7;  mul_411 = None
        mul_413 = mul_412 * pow_69;  mul_412 = pow_69 = None
        all_constants_constant48 = self.all_constants.Constant48
        mul_414 = all_constants_constant48 * einsum_3;  all_constants_constant48 = None
        all_constants_constant47 = self.all_constants.Constant47
        mul_415 = all_constants_constant47 * einsum_4;  all_constants_constant47 = einsum_4 = None
        add_90 = mul_415 + mul_414;  mul_415 = mul_414 = None
        sin_47 = torch.sin(add_90);  add_90 = None
        mul_416 = mul_6 * add_6;  mul_6 = None
        mul_417 = mul_416 * mul_151;  mul_416 = mul_151 = None
        mul_418 = mul_417 * sin_47;  mul_417 = sin_47 = None
        all_constants_constant49 = self.all_constants.Constant49
        truediv_10 = mul_418 / all_constants_constant49;  mul_418 = all_constants_constant49 = None
        mul_419 = all_parameters_bc * add_6;  all_parameters_bc = None
        mul_420 = mul_419 * add_2;  mul_419 = None
        mul_421 = mul_420 * einsum_7;  mul_420 = einsum_7 = None
        mul_422 = mul_129 * add_6;  add_6 = None
        mul_423 = mul_422 * add_2;  mul_422 = add_2 = None
        cos_104 = torch.cos(einsum_3);  einsum_3 = None
        all_constants_constant46 = self.all_constants.Constant46
        pow_71 = torch.pow(cos_104, all_constants_constant46);  cos_104 = all_constants_constant46 = None
        all_constants_constant45 = self.all_constants.Constant45
        pow_72 = torch.pow(truediv, all_constants_constant45);  truediv = all_constants_constant45 = None
        mul_424 = mul_129 * pow_72;  mul_129 = pow_72 = None
        mul_425 = mul_424 * pow_71;  mul_424 = pow_71 = None
        sub_32 = mul_425 - mul_423;  mul_425 = mul_423 = None
        add_91 = sub_32 + mul_421;  sub_32 = mul_421 = None
        add_92 = add_91 + truediv_10;  add_91 = truediv_10 = None
        sub_33 = add_92 - mul_413;  add_92 = mul_413 = None
        add_93 = sub_33 + truediv_9;  sub_33 = truediv_9 = None
        sub_34 = add_93 - mul_406;  add_93 = mul_406 = None
        sub_35 = sub_34 - mul_402;  sub_34 = mul_402 = None
        add_94 = sub_35 + mul_398;  sub_35 = mul_398 = None
        add_95 = add_94 + mul_394;  add_94 = mul_394 = None
        add_96 = add_95 + mul_390;  add_95 = mul_390 = None
        sub_36 = add_96 - mul_386;  add_96 = mul_386 = None
        sub_37 = sub_36 - mul_382;  sub_36 = mul_382 = None
        sub_38 = sub_37 - mul_378;  sub_37 = mul_378 = None
        add_97 = sub_38 + mul_374;  sub_38 = mul_374 = None
        add_98 = add_97 + mul_370;  add_97 = mul_370 = None
        sub_39 = add_98 - mul_365;  add_98 = mul_365 = None
        add_99 = sub_39 + mul_360;  sub_39 = mul_360 = None
        add_100 = add_99 + mul_355;  add_99 = mul_355 = None
        add_101 = add_100 + mul_350;  add_100 = mul_350 = None
        sub_40 = add_101 - mul_345;  add_101 = mul_345 = None
        sub_41 = sub_40 - mul_340;  sub_40 = mul_340 = None
        add_102 = sub_41 + mul_335;  sub_41 = mul_335 = None
        sub_42 = add_102 - mul_330;  add_102 = mul_330 = None
        add_103 = sub_42 + mul_325;  sub_42 = mul_325 = None
        sub_43 = add_103 - mul_320;  add_103 = mul_320 = None
        sub_44 = sub_43 - mul_315;  sub_43 = mul_315 = None
        sub_45 = sub_44 - mul_310;  sub_44 = mul_310 = None
        add_104 = sub_45 + mul_305;  sub_45 = mul_305 = None
        sub_46 = add_104 - mul_299;  add_104 = mul_299 = None
        truediv_11 = sub_46 / add_10;  sub_46 = add_10 = None
        getitem_11 = kwargs['Xddx'];  kwargs = None
        relation_forward_sample_part1022_w = self.all_constants.SamplePart1022
        einsum_11 = torch.functional.einsum('bij,ki->bkj', getitem_11, relation_forward_sample_part1022_w);  getitem_11 = relation_forward_sample_part1022_w = None
        return ({'th2_ddot_est': truediv_7, 'th1_ddot_est': truediv_8, 'acc_cart_est': truediv_11}, {'SamplePart1022': einsum_11, 'SamplePart1024': einsum_10, 'SamplePart1026': einsum_9, 'Mul1030': mul_2, 'Mul1037': mul_1, 'Mul1044': mul, 'Div982': truediv_11, 'Div983': truediv_8, 'Div984': truediv_7, 'Relu1033': relu_2, 'Relu1040': relu_1, 'Relu1047': relu}, {}, {})
        
