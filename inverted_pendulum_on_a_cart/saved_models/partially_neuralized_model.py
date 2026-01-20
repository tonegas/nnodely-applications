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
        self.all_constants["Constant32"] = torch.tensor([2.0], requires_grad=False)
        self.all_constants["Constant33"] = torch.tensor([2.0], requires_grad=False)
        self.all_constants["Constant34"] = torch.tensor([2.0], requires_grad=False)
        self.all_constants["Constant36"] = torch.tensor([2.0], requires_grad=False)
        self.all_constants["Constant37"] = torch.tensor([2.0], requires_grad=False)
        self.all_constants["Constant38"] = torch.tensor([2.0], requires_grad=False)
        self.all_constants["Constant39"] = torch.tensor([2.0], requires_grad=False)
        self.all_constants["Constant40"] = torch.tensor([2.0], requires_grad=False)
        self.all_constants["Constant41"] = torch.tensor([2.0], requires_grad=False)
        self.all_constants["Constant43"] = torch.tensor([2.0], requires_grad=False)
        self.all_constants["Constant44"] = torch.tensor([2.0], requires_grad=False)
        self.all_constants["Constant45"] = torch.tensor([2.0], requires_grad=False)
        self.all_constants["Constant46"] = torch.tensor([2.0], requires_grad=False)
        self.all_constants["Constant47"] = torch.tensor([2.0], requires_grad=False)
        self.all_constants["Constant48"] = torch.tensor([2.0], requires_grad=False)
        self.all_constants["Constant49"] = torch.tensor([2.0], requires_grad=False)
        self.all_constants["Constant50"] = torch.tensor([2.0], requires_grad=False)
        self.all_constants["Constant51"] = torch.tensor([2.0], requires_grad=False)
        self.all_constants["Constant52"] = torch.tensor([2.0], requires_grad=False)
        self.all_constants["Constant53"] = torch.tensor([2.0], requires_grad=False)
        self.all_constants["Constant54"] = torch.tensor([6.0], requires_grad=False)
        self.all_constants["Constant55"] = torch.tensor([2.0], requires_grad=False)
        self.all_constants["Constant56"] = torch.tensor([2.0], requires_grad=False)
        self.all_constants["Constant57"] = torch.tensor([6.0], requires_grad=False)
        self.all_constants["Constant58"] = torch.tensor([2.0], requires_grad=False)
        self.all_constants["Constant59"] = torch.tensor([2.0], requires_grad=False)
        self.all_constants["Constant60"] = torch.tensor([6.0], requires_grad=False)
        self.all_constants["Constant61"] = torch.tensor([2.0], requires_grad=False)
        self.all_constants["Constant62"] = torch.tensor([2.0], requires_grad=False)
        self.all_constants["Constant63"] = torch.tensor([6.0], requires_grad=False)
        self.all_constants["dt"] = torch.tensor([0.019999999552965164], requires_grad=False)
        self.all_constants["g"] = torch.tensor([9.8100004196167], requires_grad=False)
        self.all_constants["sigma_omega"] = torch.tensor([0.0], requires_grad=False)
        self.all_constants["sigma_theta"] = torch.tensor([0.0], requires_grad=False)
        self.all_constants["sigma_v"] = torch.tensor([0.0], requires_grad=False)
        self.all_constants["sigma_x"] = torch.tensor([0.0], requires_grad=False)
        self.all_parameters["a1"] = torch.nn.Parameter(torch.tensor([0.01585010439157486]), requires_grad=True)
        self.all_parameters["a10"] = torch.nn.Parameter(torch.tensor([0.38438400626182556]), requires_grad=True)
        self.all_parameters["a11"] = torch.nn.Parameter(torch.tensor([0.022575952112674713]), requires_grad=True)
        self.all_parameters["a12"] = torch.nn.Parameter(torch.tensor([2.4811575412750244]), requires_grad=True)
        self.all_parameters["a13"] = torch.nn.Parameter(torch.tensor([0.16351740062236786]), requires_grad=True)
        self.all_parameters["a14"] = torch.nn.Parameter(torch.tensor([0.1933160424232483]), requires_grad=True)
        self.all_parameters["a2"] = torch.nn.Parameter(torch.tensor([0.010592124424874783]), requires_grad=True)
        self.all_parameters["a4"] = torch.nn.Parameter(torch.tensor([0.037365734577178955]), requires_grad=True)
        self.all_parameters["a5"] = torch.nn.Parameter(torch.tensor([0.024878066033124924]), requires_grad=True)
        self.all_parameters["a6"] = torch.nn.Parameter(torch.tensor([1.0562189817428589]), requires_grad=True)
        self.all_parameters["a7"] = torch.nn.Parameter(torch.tensor([0.03726451098918915]), requires_grad=True)
        self.all_parameters["a8"] = torch.nn.Parameter(torch.tensor([0.25560933351516724]), requires_grad=True)
        self.all_constants["SamplePart1"] = torch.tensor([[1.0]], requires_grad=True)
        self.all_constants["SamplePart10"] = torch.tensor([[1.0]], requires_grad=True)
        self.all_constants["SamplePart105"] = torch.tensor([[1.0]], requires_grad=True)
        self.all_constants["SamplePart15"] = torch.tensor([[1.0]], requires_grad=True)
        self.all_constants["SamplePart17"] = torch.tensor([[1.0]], requires_grad=True)
        self.all_constants["SamplePart181"] = torch.tensor([[1.0]], requires_grad=True)
        self.all_constants["SamplePart22"] = torch.tensor([[1.0]], requires_grad=True)
        self.all_constants["SamplePart24"] = torch.tensor([[1.0]], requires_grad=True)
        self.all_constants["SamplePart249"] = torch.tensor([[1.0]], requires_grad=True)
        self.all_constants["SamplePart29"] = torch.tensor([[1.0]], requires_grad=True)
        self.all_constants["SamplePart3"] = torch.tensor([[1.0]], requires_grad=True)
        self.all_constants["SamplePart353"] = torch.tensor([[1.0]], requires_grad=True)
        self.all_constants["SamplePart355"] = torch.tensor([[1.0]], requires_grad=True)
        self.all_constants["SamplePart357"] = torch.tensor([[1.0]], requires_grad=True)
        self.all_constants["SamplePart359"] = torch.tensor([[1.0]], requires_grad=True)
        self.all_constants["SamplePart8"] = torch.tensor([[1.0]], requires_grad=True)
        self.all_parameters = torch.nn.ParameterDict(self.all_parameters)
        self.all_constants = torch.nn.ParameterDict(self.all_constants)

    def update(self, closed_loop={}, connect={}, disconnect=False):
        pass
    
    def forward(self, kwargs):
        all_parameters_a4 = self.all_parameters.a4
        all_constants_g = self.all_constants.g
        mul = all_parameters_a4 * all_constants_g
        all_parameters_a10 = self.all_parameters.a10
        mul_1 = all_parameters_a10 * all_constants_g
        mul_2 = all_parameters_a4 * all_constants_g
        mul_3 = all_parameters_a10 * all_constants_g
        mul_4 = all_parameters_a4 * all_constants_g
        mul_5 = all_parameters_a10 * all_constants_g
        mul_6 = all_parameters_a4 * all_constants_g;  all_parameters_a4 = None
        mul_7 = all_parameters_a10 * all_constants_g;  all_parameters_a10 = all_constants_g = None
        all_parameters_a7 = self.all_parameters.a7
        neg = -all_parameters_a7
        neg_1 = -all_parameters_a7
        neg_2 = -all_parameters_a7
        neg_3 = -all_parameters_a7;  all_parameters_a7 = None
        getitem = kwargs['Xpos']
        relation_forward_sample_part1_w = self.all_constants.SamplePart1
        einsum = torch.functional.einsum('bij,ki->bkj', getitem, relation_forward_sample_part1_w);  getitem = relation_forward_sample_part1_w = None
        getitem_1 = kwargs['noise1']
        relation_forward_sample_part10_w = self.all_constants.SamplePart10
        einsum_1 = torch.functional.einsum('bij,ki->bkj', getitem_1, relation_forward_sample_part10_w);  getitem_1 = relation_forward_sample_part10_w = None
        getitem_2 = kwargs['action']
        relation_forward_sample_part105_w = self.all_constants.SamplePart105
        einsum_2 = torch.functional.einsum('bij,ki->bkj', getitem_2, relation_forward_sample_part105_w);  getitem_2 = relation_forward_sample_part105_w = None
        getitem_3 = kwargs['Xangular_velocity']
        relation_forward_sample_part15_w = self.all_constants.SamplePart15
        einsum_3 = torch.functional.einsum('bij,ki->bkj', getitem_3, relation_forward_sample_part15_w);  getitem_3 = relation_forward_sample_part15_w = None
        getitem_4 = kwargs['noise3']
        relation_forward_sample_part17_w = self.all_constants.SamplePart17
        einsum_4 = torch.functional.einsum('bij,ki->bkj', getitem_4, relation_forward_sample_part17_w);  getitem_4 = relation_forward_sample_part17_w = None
        getitem_5 = kwargs['action']
        relation_forward_sample_part181_w = self.all_constants.SamplePart181
        einsum_5 = torch.functional.einsum('bij,ki->bkj', getitem_5, relation_forward_sample_part181_w);  getitem_5 = relation_forward_sample_part181_w = None
        getitem_6 = kwargs['Xangle']
        relation_forward_sample_part22_w = self.all_constants.SamplePart22
        einsum_6 = torch.functional.einsum('bij,ki->bkj', getitem_6, relation_forward_sample_part22_w);  getitem_6 = relation_forward_sample_part22_w = None
        getitem_7 = kwargs['noise2']
        relation_forward_sample_part24_w = self.all_constants.SamplePart24
        einsum_7 = torch.functional.einsum('bij,ki->bkj', getitem_7, relation_forward_sample_part24_w);  getitem_7 = relation_forward_sample_part24_w = None
        getitem_8 = kwargs['action']
        relation_forward_sample_part249_w = self.all_constants.SamplePart249
        einsum_8 = torch.functional.einsum('bij,ki->bkj', getitem_8, relation_forward_sample_part249_w);  getitem_8 = relation_forward_sample_part249_w = None
        getitem_9 = kwargs['action']
        relation_forward_sample_part29_w = self.all_constants.SamplePart29
        einsum_9 = torch.functional.einsum('bij,ki->bkj', getitem_9, relation_forward_sample_part29_w);  getitem_9 = relation_forward_sample_part29_w = None
        getitem_10 = kwargs['noise4']
        relation_forward_sample_part3_w = self.all_constants.SamplePart3
        einsum_10 = torch.functional.einsum('bij,ki->bkj', getitem_10, relation_forward_sample_part3_w);  getitem_10 = relation_forward_sample_part3_w = None
        getitem_11 = kwargs['Yangular_velocity']
        relation_forward_sample_part353_w = self.all_constants.SamplePart353
        einsum_11 = torch.functional.einsum('bij,ki->bkj', getitem_11, relation_forward_sample_part353_w);  getitem_11 = relation_forward_sample_part353_w = None
        getitem_12 = kwargs['Yangle']
        relation_forward_sample_part355_w = self.all_constants.SamplePart355
        einsum_12 = torch.functional.einsum('bij,ki->bkj', getitem_12, relation_forward_sample_part355_w);  getitem_12 = relation_forward_sample_part355_w = None
        getitem_13 = kwargs['Yvelocity']
        relation_forward_sample_part357_w = self.all_constants.SamplePart357
        einsum_13 = torch.functional.einsum('bij,ki->bkj', getitem_13, relation_forward_sample_part357_w);  getitem_13 = relation_forward_sample_part357_w = None
        getitem_14 = kwargs['Ypos']
        relation_forward_sample_part359_w = self.all_constants.SamplePart359
        einsum_14 = torch.functional.einsum('bij,ki->bkj', getitem_14, relation_forward_sample_part359_w);  getitem_14 = relation_forward_sample_part359_w = None
        getitem_15 = kwargs['Xvelocity'];  kwargs = None
        relation_forward_sample_part8_w = self.all_constants.SamplePart8
        einsum_15 = torch.functional.einsum('bij,ki->bkj', getitem_15, relation_forward_sample_part8_w);  getitem_15 = relation_forward_sample_part8_w = None
        all_constants_sigma_v = self.all_constants.sigma_v
        mul_8 = einsum_1 * all_constants_sigma_v;  einsum_1 = all_constants_sigma_v = None
        all_parameters_a6 = self.all_parameters.a6
        mul_9 = all_parameters_a6 * einsum_2
        all_constants_sigma_omega = self.all_constants.sigma_omega
        mul_10 = einsum_4 * all_constants_sigma_omega;  einsum_4 = all_constants_sigma_omega = None
        mul_11 = all_parameters_a6 * einsum_5
        all_constants_sigma_theta = self.all_constants.sigma_theta
        mul_12 = einsum_7 * all_constants_sigma_theta;  einsum_7 = all_constants_sigma_theta = None
        mul_13 = all_parameters_a6 * einsum_8
        all_constants_sigma_x = self.all_constants.sigma_x
        mul_14 = einsum_10 * all_constants_sigma_x;  einsum_10 = all_constants_sigma_x = None
        mul_15 = all_parameters_a6 * einsum_9;  all_parameters_a6 = None
        add = einsum_15 + mul_8;  einsum_15 = mul_8 = None
        add_1 = einsum_3 + mul_10;  einsum_3 = mul_10 = None
        add_2 = einsum_6 + mul_12;  einsum_6 = mul_12 = None
        add_3 = einsum + mul_14;  einsum = mul_14 = None
        cos = torch.cos(add_2)
        all_parameters_a14 = self.all_parameters.a14
        mul_16 = all_parameters_a14 * cos
        all_parameters_a2 = self.all_parameters.a2
        mul_17 = all_parameters_a2 * add
        all_parameters_a8 = self.all_parameters.a8
        mul_18 = all_parameters_a8 * add_1
        mul_19 = mul_6 * cos;  mul_6 = None
        all_parameters_a5 = self.all_parameters.a5
        mul_20 = all_parameters_a5 * add_1
        mul_21 = mul_20 * cos;  mul_20 = None
        all_parameters_a11 = self.all_parameters.a11
        mul_22 = all_parameters_a11 * add
        mul_23 = mul_22 * cos;  mul_22 = None
        all_parameters_a12 = self.all_parameters.a12
        mul_24 = all_parameters_a12 * cos
        mul_25 = mul_24 * einsum_9;  mul_24 = einsum_9 = None
        all_constants_dt = self.all_constants.dt
        mul_26 = add_1 * all_constants_dt
        all_constants_constant32 = self.all_constants.Constant32
        pow_1 = torch.pow(mul_16, all_constants_constant32);  mul_16 = all_constants_constant32 = None
        all_constants_constant33 = self.all_constants.Constant33
        pow_2 = torch.pow(add_1, all_constants_constant33);  all_constants_constant33 = None
        all_constants_constant34 = self.all_constants.Constant34
        pow_3 = torch.pow(add_1, all_constants_constant34);  all_constants_constant34 = None
        sin = torch.sin(add_2)
        all_parameters_a13 = self.all_parameters.a13
        sub = all_parameters_a13 - pow_1;  pow_1 = None
        all_constants_constant37 = self.all_constants.Constant37
        truediv = mul_26 / all_constants_constant37;  mul_26 = all_constants_constant37 = None
        all_parameters_a1 = self.all_parameters.a1
        mul_27 = all_parameters_a1 * pow_2;  pow_2 = None
        mul_28 = mul_27 * sin;  mul_27 = None
        mul_29 = mul_19 * sin;  mul_19 = None
        mul_30 = neg_3 * pow_3;  neg_3 = pow_3 = None
        mul_31 = mul_30 * cos;  mul_30 = cos = None
        mul_32 = mul_31 * sin;  mul_31 = None
        mul_33 = mul_7 * sin;  mul_7 = sin = None
        sub_1 = mul_28 - mul_17;  mul_28 = mul_17 = None
        sub_2 = sub_1 - mul_29;  sub_1 = mul_29 = None
        sub_3 = mul_32 - mul_18;  mul_32 = mul_18 = None
        add_4 = sub_2 + mul_21;  sub_2 = mul_21 = None
        add_5 = add_4 + mul_15;  add_4 = mul_15 = None
        add_6 = sub_3 + mul_33;  sub_3 = mul_33 = None
        add_7 = add_6 + mul_23;  add_6 = mul_23 = None
        add_8 = add_2 + truediv;  truediv = None
        cos_1 = torch.cos(add_8)
        truediv_1 = add_5 / sub;  add_5 = None
        mul_34 = all_parameters_a14 * cos_1
        mul_35 = mul * cos_1;  mul = None
        mul_36 = all_parameters_a12 * cos_1
        mul_37 = mul_36 * einsum_2;  mul_36 = einsum_2 = None
        mul_38 = truediv_1 * all_constants_dt
        all_constants_constant39 = self.all_constants.Constant39
        pow_4 = torch.pow(mul_34, all_constants_constant39);  mul_34 = all_constants_constant39 = None
        sin_1 = torch.sin(add_8);  add_8 = None
        sub_4 = all_parameters_a13 - pow_4;  pow_4 = None
        sub_5 = add_7 - mul_25;  add_7 = mul_25 = None
        truediv_2 = sub_5 / sub;  sub_5 = sub = None
        all_constants_constant36 = self.all_constants.Constant36
        truediv_3 = mul_38 / all_constants_constant36;  mul_38 = all_constants_constant36 = None
        mul_39 = truediv_2 * all_constants_dt
        mul_40 = mul_35 * sin_1;  mul_35 = None
        mul_41 = mul_1 * sin_1;  mul_1 = None
        add_9 = add + truediv_3;  truediv_3 = None
        all_constants_constant38 = self.all_constants.Constant38
        truediv_4 = mul_39 / all_constants_constant38;  mul_39 = all_constants_constant38 = None
        mul_42 = all_parameters_a2 * add_9
        mul_43 = all_parameters_a11 * add_9
        mul_44 = mul_43 * cos_1;  mul_43 = None
        all_constants_constant55 = self.all_constants.Constant55
        mul_45 = all_constants_constant55 * add_9;  all_constants_constant55 = add_9 = None
        add_10 = add_1 + truediv_4;  truediv_4 = None
        add_11 = add + mul_45;  mul_45 = None
        mul_46 = all_parameters_a8 * add_10
        mul_47 = all_parameters_a5 * add_10
        mul_48 = mul_47 * cos_1;  mul_47 = None
        mul_49 = add_10 * all_constants_dt
        all_constants_constant61 = self.all_constants.Constant61
        mul_50 = all_constants_constant61 * add_10;  all_constants_constant61 = None
        all_constants_constant40 = self.all_constants.Constant40
        pow_5 = torch.pow(add_10, all_constants_constant40);  all_constants_constant40 = None
        all_constants_constant41 = self.all_constants.Constant41
        pow_6 = torch.pow(add_10, all_constants_constant41);  add_10 = all_constants_constant41 = None
        add_12 = add_1 + mul_50;  mul_50 = None
        all_constants_constant44 = self.all_constants.Constant44
        truediv_5 = mul_49 / all_constants_constant44;  mul_49 = all_constants_constant44 = None
        mul_51 = all_parameters_a1 * pow_5;  pow_5 = None
        mul_52 = mul_51 * sin_1;  mul_51 = None
        mul_53 = neg * pow_6;  neg = pow_6 = None
        mul_54 = mul_53 * cos_1;  mul_53 = cos_1 = None
        mul_55 = mul_54 * sin_1;  mul_54 = sin_1 = None
        sub_6 = mul_52 - mul_42;  mul_52 = mul_42 = None
        sub_7 = sub_6 - mul_40;  sub_6 = mul_40 = None
        sub_8 = mul_55 - mul_46;  mul_55 = mul_46 = None
        add_13 = sub_7 + mul_48;  sub_7 = mul_48 = None
        add_14 = add_13 + mul_9;  add_13 = mul_9 = None
        add_15 = sub_8 + mul_41;  sub_8 = mul_41 = None
        add_16 = add_15 + mul_44;  add_15 = mul_44 = None
        add_17 = add_2 + truediv_5;  truediv_5 = None
        cos_2 = torch.cos(add_17)
        truediv_6 = add_14 / sub_4;  add_14 = None
        mul_56 = truediv_6 * all_constants_dt
        mul_57 = all_parameters_a14 * cos_2
        mul_58 = mul_2 * cos_2;  mul_2 = None
        mul_59 = all_parameters_a12 * cos_2
        mul_60 = mul_59 * einsum_5;  mul_59 = einsum_5 = None
        all_constants_constant52 = self.all_constants.Constant52
        mul_61 = all_constants_constant52 * truediv_6;  all_constants_constant52 = truediv_6 = None
        all_constants_constant46 = self.all_constants.Constant46
        pow_7 = torch.pow(mul_57, all_constants_constant46);  mul_57 = all_constants_constant46 = None
        sin_2 = torch.sin(add_17);  add_17 = None
        sub_9 = add_16 - mul_37;  add_16 = mul_37 = None
        sub_10 = all_parameters_a13 - pow_7;  pow_7 = None
        add_18 = truediv_1 + mul_61;  truediv_1 = mul_61 = None
        truediv_7 = sub_9 / sub_4;  sub_9 = sub_4 = None
        all_constants_constant43 = self.all_constants.Constant43
        truediv_8 = mul_56 / all_constants_constant43;  mul_56 = all_constants_constant43 = None
        mul_62 = truediv_7 * all_constants_dt
        mul_63 = mul_58 * sin_2;  mul_58 = None
        mul_64 = mul_3 * sin_2;  mul_3 = None
        all_constants_constant58 = self.all_constants.Constant58
        mul_65 = all_constants_constant58 * truediv_7;  all_constants_constant58 = truediv_7 = None
        add_19 = add + truediv_8;  truediv_8 = None
        add_20 = truediv_2 + mul_65;  truediv_2 = mul_65 = None
        all_constants_constant45 = self.all_constants.Constant45
        truediv_9 = mul_62 / all_constants_constant45;  mul_62 = all_constants_constant45 = None
        mul_66 = all_parameters_a2 * add_19
        mul_67 = all_parameters_a11 * add_19
        mul_68 = mul_67 * cos_2;  mul_67 = None
        all_constants_constant56 = self.all_constants.Constant56
        mul_69 = all_constants_constant56 * add_19;  all_constants_constant56 = add_19 = None
        add_21 = add_1 + truediv_9;  truediv_9 = None
        add_22 = add_11 + mul_69;  add_11 = mul_69 = None
        mul_70 = all_parameters_a8 * add_21
        mul_71 = all_parameters_a5 * add_21
        mul_72 = mul_71 * cos_2;  mul_71 = None
        mul_73 = add_21 * all_constants_dt
        all_constants_constant62 = self.all_constants.Constant62
        mul_74 = all_constants_constant62 * add_21;  all_constants_constant62 = None
        all_constants_constant47 = self.all_constants.Constant47
        pow_8 = torch.pow(add_21, all_constants_constant47);  all_constants_constant47 = None
        all_constants_constant48 = self.all_constants.Constant48
        pow_9 = torch.pow(add_21, all_constants_constant48);  add_21 = all_constants_constant48 = None
        add_23 = add_2 + mul_73;  mul_73 = None
        add_24 = add_12 + mul_74;  add_12 = mul_74 = None
        cos_3 = torch.cos(add_23)
        mul_75 = all_parameters_a1 * pow_8;  pow_8 = None
        mul_76 = mul_75 * sin_2;  mul_75 = None
        mul_77 = neg_1 * pow_9;  neg_1 = pow_9 = None
        mul_78 = mul_77 * cos_2;  mul_77 = cos_2 = None
        mul_79 = mul_78 * sin_2;  mul_78 = sin_2 = None
        mul_80 = all_parameters_a14 * cos_3;  all_parameters_a14 = None
        mul_81 = mul_4 * cos_3;  mul_4 = None
        mul_82 = all_parameters_a12 * cos_3;  all_parameters_a12 = None
        mul_83 = mul_82 * einsum_8;  mul_82 = einsum_8 = None
        all_constants_constant49 = self.all_constants.Constant49
        pow_10 = torch.pow(mul_80, all_constants_constant49);  mul_80 = all_constants_constant49 = None
        sin_3 = torch.sin(add_23);  add_23 = None
        sub_11 = mul_76 - mul_66;  mul_76 = mul_66 = None
        sub_12 = sub_11 - mul_63;  sub_11 = mul_63 = None
        sub_13 = mul_79 - mul_70;  mul_79 = mul_70 = None
        sub_14 = all_parameters_a13 - pow_10;  all_parameters_a13 = pow_10 = None
        add_25 = sub_12 + mul_72;  sub_12 = mul_72 = None
        add_26 = add_25 + mul_11;  add_25 = mul_11 = None
        add_27 = sub_13 + mul_64;  sub_13 = mul_64 = None
        add_28 = add_27 + mul_68;  add_27 = mul_68 = None
        truediv_10 = add_26 / sub_10;  add_26 = None
        mul_84 = truediv_10 * all_constants_dt
        mul_85 = mul_81 * sin_3;  mul_81 = None
        mul_86 = mul_5 * sin_3;  mul_5 = None
        all_constants_constant53 = self.all_constants.Constant53
        mul_87 = all_constants_constant53 * truediv_10;  all_constants_constant53 = truediv_10 = None
        sub_15 = add_28 - mul_60;  add_28 = mul_60 = None
        add_29 = add + mul_84;  mul_84 = None
        add_30 = add_18 + mul_87;  add_18 = mul_87 = None
        add_31 = add_22 + add_29;  add_22 = None
        truediv_11 = sub_15 / sub_10;  sub_15 = sub_10 = None
        mul_88 = truediv_11 * all_constants_dt
        mul_89 = all_parameters_a2 * add_29;  all_parameters_a2 = None
        mul_90 = all_parameters_a11 * add_29;  all_parameters_a11 = add_29 = None
        mul_91 = mul_90 * cos_3;  mul_90 = None
        mul_92 = add_31 * all_constants_dt;  add_31 = None
        all_constants_constant59 = self.all_constants.Constant59
        mul_93 = all_constants_constant59 * truediv_11;  all_constants_constant59 = truediv_11 = None
        add_32 = add_1 + mul_88;  mul_88 = None
        add_33 = add_20 + mul_93;  add_20 = mul_93 = None
        add_34 = add_24 + add_32;  add_24 = None
        all_constants_constant57 = self.all_constants.Constant57
        truediv_12 = mul_92 / all_constants_constant57;  mul_92 = all_constants_constant57 = None
        mul_94 = all_parameters_a8 * add_32;  all_parameters_a8 = None
        mul_95 = all_parameters_a5 * add_32;  all_parameters_a5 = None
        mul_96 = mul_95 * cos_3;  mul_95 = None
        mul_97 = add_34 * all_constants_dt;  add_34 = None
        all_constants_constant50 = self.all_constants.Constant50
        pow_11 = torch.pow(add_32, all_constants_constant50);  all_constants_constant50 = None
        all_constants_constant51 = self.all_constants.Constant51
        pow_12 = torch.pow(add_32, all_constants_constant51);  add_32 = all_constants_constant51 = None
        add_35 = add_3 + truediv_12;  add_3 = truediv_12 = None
        all_constants_constant63 = self.all_constants.Constant63
        truediv_13 = mul_97 / all_constants_constant63;  mul_97 = all_constants_constant63 = None
        mul_98 = all_parameters_a1 * pow_11;  all_parameters_a1 = pow_11 = None
        mul_99 = mul_98 * sin_3;  mul_98 = None
        mul_100 = neg_2 * pow_12;  neg_2 = pow_12 = None
        mul_101 = mul_100 * cos_3;  mul_100 = cos_3 = None
        mul_102 = mul_101 * sin_3;  mul_101 = sin_3 = None
        sub_16 = mul_99 - mul_89;  mul_99 = mul_89 = None
        sub_17 = sub_16 - mul_85;  sub_16 = mul_85 = None
        sub_18 = mul_102 - mul_94;  mul_102 = mul_94 = None
        add_36 = sub_17 + mul_96;  sub_17 = mul_96 = None
        add_37 = add_36 + mul_13;  add_36 = mul_13 = None
        add_38 = sub_18 + mul_86;  sub_18 = mul_86 = None
        add_39 = add_38 + mul_91;  add_38 = mul_91 = None
        add_40 = add_2 + truediv_13;  add_2 = truediv_13 = None
        truediv_14 = add_37 / sub_14;  add_37 = None
        sub_19 = add_39 - mul_83;  add_39 = mul_83 = None
        add_41 = add_30 + truediv_14;  add_30 = truediv_14 = None
        truediv_15 = sub_19 / sub_14;  sub_19 = sub_14 = None
        mul_103 = add_41 * all_constants_dt;  add_41 = None
        add_42 = add_33 + truediv_15;  add_33 = truediv_15 = None
        all_constants_constant54 = self.all_constants.Constant54
        truediv_16 = mul_103 / all_constants_constant54;  mul_103 = all_constants_constant54 = None
        mul_104 = add_42 * all_constants_dt;  add_42 = all_constants_dt = None
        add_43 = add + truediv_16;  add = truediv_16 = None
        all_constants_constant60 = self.all_constants.Constant60
        truediv_17 = mul_104 / all_constants_constant60;  mul_104 = all_constants_constant60 = None
        add_44 = add_1 + truediv_17;  add_1 = truediv_17 = None
        return ({'est_theta': add_40, 'est_thetadot': add_44, 'est_x': add_35, 'est_xdot': add_43}, {'SamplePart353': einsum_11, 'SamplePart355': einsum_12, 'SamplePart359': einsum_14, 'SamplePart357': einsum_13, 'Add339': add_44, 'Add351': add_40, 'Add327': add_35, 'Add315': add_43}, {'Xangle': add_40, 'Xangular_velocity': add_44, 'Xpos': add_35, 'Xvelocity': add_43}, {})
        
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
