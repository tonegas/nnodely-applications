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
        self.all_constants["Constant34"] = torch.tensor([2.0], requires_grad=False)
        self.all_constants["Constant35"] = torch.tensor([2.0], requires_grad=False)
        self.all_constants["Constant36"] = torch.tensor([2.0], requires_grad=False)
        self.all_constants["Constant38"] = torch.tensor([2.0], requires_grad=False)
        self.all_constants["Constant39"] = torch.tensor([2.0], requires_grad=False)
        self.all_constants["Constant40"] = torch.tensor([2.0], requires_grad=False)
        self.all_constants["Constant41"] = torch.tensor([2.0], requires_grad=False)
        self.all_constants["Constant42"] = torch.tensor([2.0], requires_grad=False)
        self.all_constants["Constant43"] = torch.tensor([2.0], requires_grad=False)
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
        self.all_constants["Constant56"] = torch.tensor([6.0], requires_grad=False)
        self.all_constants["Constant57"] = torch.tensor([2.0], requires_grad=False)
        self.all_constants["Constant58"] = torch.tensor([2.0], requires_grad=False)
        self.all_constants["Constant59"] = torch.tensor([6.0], requires_grad=False)
        self.all_constants["Constant60"] = torch.tensor([2.0], requires_grad=False)
        self.all_constants["Constant61"] = torch.tensor([2.0], requires_grad=False)
        self.all_constants["Constant62"] = torch.tensor([6.0], requires_grad=False)
        self.all_constants["Constant63"] = torch.tensor([2.0], requires_grad=False)
        self.all_constants["Constant64"] = torch.tensor([2.0], requires_grad=False)
        self.all_constants["Constant65"] = torch.tensor([6.0], requires_grad=False)
        self.all_constants["dt"] = torch.tensor([0.019999999552965164], requires_grad=False)
        self.all_constants["g"] = torch.tensor([9.8100004196167], requires_grad=False)
        self.all_constants["sigma_force"] = torch.tensor([0.05999999865889549], requires_grad=False)
        self.all_constants["sigma_omega"] = torch.tensor([0.10000000149011612], requires_grad=False)
        self.all_constants["sigma_theta"] = torch.tensor([0.009999999776482582], requires_grad=False)
        self.all_constants["sigma_v"] = torch.tensor([0.019999999552965164], requires_grad=False)
        self.all_constants["sigma_x"] = torch.tensor([0.009999999776482582], requires_grad=False)
        self.all_parameters["a1"] = torch.nn.Parameter(torch.tensor([0.025546101853251457]), requires_grad=True)
        self.all_parameters["a10"] = torch.nn.Parameter(torch.tensor([0.6039144992828369]), requires_grad=True)
        self.all_parameters["a11"] = torch.nn.Parameter(torch.tensor([-0.09859984368085861]), requires_grad=True)
        self.all_parameters["a12"] = torch.nn.Parameter(torch.tensor([3.735929250717163]), requires_grad=True)
        self.all_parameters["a13"] = torch.nn.Parameter(torch.tensor([0.25896498560905457]), requires_grad=True)
        self.all_parameters["a14"] = torch.nn.Parameter(torch.tensor([0.24766528606414795]), requires_grad=True)
        self.all_parameters["a2"] = torch.nn.Parameter(torch.tensor([0.021793698891997337]), requires_grad=True)
        self.all_parameters["a4"] = torch.nn.Parameter(torch.tensor([0.05778711661696434]), requires_grad=True)
        self.all_parameters["a5"] = torch.nn.Parameter(torch.tensor([0.04131729155778885]), requires_grad=True)
        self.all_parameters["a6"] = torch.nn.Parameter(torch.tensor([1.6125577688217163]), requires_grad=True)
        self.all_parameters["a7"] = torch.nn.Parameter(torch.tensor([0.06394010037183762]), requires_grad=True)
        self.all_parameters["a8"] = torch.nn.Parameter(torch.tensor([0.43495017290115356]), requires_grad=True)
        self.all_constants["SamplePart1"] = torch.tensor([[1.0]], requires_grad=True)
        self.all_constants["SamplePart10"] = torch.tensor([[1.0]], requires_grad=True)
        self.all_constants["SamplePart15"] = torch.tensor([[1.0]], requires_grad=True)
        self.all_constants["SamplePart17"] = torch.tensor([[1.0]], requires_grad=True)
        self.all_constants["SamplePart22"] = torch.tensor([[1.0]], requires_grad=True)
        self.all_constants["SamplePart24"] = torch.tensor([[1.0]], requires_grad=True)
        self.all_constants["SamplePart29"] = torch.tensor([[1.0]], requires_grad=True)
        self.all_constants["SamplePart3"] = torch.tensor([[1.0]], requires_grad=True)
        self.all_constants["SamplePart31"] = torch.tensor([[1.0]], requires_grad=True)
        self.all_constants["SamplePart356"] = torch.tensor([[1.0]], requires_grad=True)
        self.all_constants["SamplePart358"] = torch.tensor([[1.0]], requires_grad=True)
        self.all_constants["SamplePart360"] = torch.tensor([[1.0]], requires_grad=True)
        self.all_constants["SamplePart362"] = torch.tensor([[1.0]], requires_grad=True)
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
        getitem_2 = kwargs['Xangular_velocity']
        relation_forward_sample_part15_w = self.all_constants.SamplePart15
        einsum_2 = torch.functional.einsum('bij,ki->bkj', getitem_2, relation_forward_sample_part15_w);  getitem_2 = relation_forward_sample_part15_w = None
        getitem_3 = kwargs['noise3']
        relation_forward_sample_part17_w = self.all_constants.SamplePart17
        einsum_3 = torch.functional.einsum('bij,ki->bkj', getitem_3, relation_forward_sample_part17_w);  getitem_3 = relation_forward_sample_part17_w = None
        getitem_4 = kwargs['Xangle']
        relation_forward_sample_part22_w = self.all_constants.SamplePart22
        einsum_4 = torch.functional.einsum('bij,ki->bkj', getitem_4, relation_forward_sample_part22_w);  getitem_4 = relation_forward_sample_part22_w = None
        getitem_5 = kwargs['noise2']
        relation_forward_sample_part24_w = self.all_constants.SamplePart24
        einsum_5 = torch.functional.einsum('bij,ki->bkj', getitem_5, relation_forward_sample_part24_w);  getitem_5 = relation_forward_sample_part24_w = None
        getitem_6 = kwargs['action']
        relation_forward_sample_part29_w = self.all_constants.SamplePart29
        einsum_6 = torch.functional.einsum('bij,ki->bkj', getitem_6, relation_forward_sample_part29_w);  getitem_6 = relation_forward_sample_part29_w = None
        getitem_7 = kwargs['noise4']
        relation_forward_sample_part3_w = self.all_constants.SamplePart3
        einsum_7 = torch.functional.einsum('bij,ki->bkj', getitem_7, relation_forward_sample_part3_w);  getitem_7 = relation_forward_sample_part3_w = None
        getitem_8 = kwargs['noise5']
        relation_forward_sample_part31_w = self.all_constants.SamplePart31
        einsum_8 = torch.functional.einsum('bij,ki->bkj', getitem_8, relation_forward_sample_part31_w);  getitem_8 = relation_forward_sample_part31_w = None
        getitem_9 = kwargs['Yangular_velocity']
        relation_forward_sample_part356_w = self.all_constants.SamplePart356
        einsum_9 = torch.functional.einsum('bij,ki->bkj', getitem_9, relation_forward_sample_part356_w);  getitem_9 = relation_forward_sample_part356_w = None
        getitem_10 = kwargs['Yangle']
        relation_forward_sample_part358_w = self.all_constants.SamplePart358
        einsum_10 = torch.functional.einsum('bij,ki->bkj', getitem_10, relation_forward_sample_part358_w);  getitem_10 = relation_forward_sample_part358_w = None
        getitem_11 = kwargs['Yvelocity']
        relation_forward_sample_part360_w = self.all_constants.SamplePart360
        einsum_11 = torch.functional.einsum('bij,ki->bkj', getitem_11, relation_forward_sample_part360_w);  getitem_11 = relation_forward_sample_part360_w = None
        getitem_12 = kwargs['Ypos']
        relation_forward_sample_part362_w = self.all_constants.SamplePart362
        einsum_12 = torch.functional.einsum('bij,ki->bkj', getitem_12, relation_forward_sample_part362_w);  getitem_12 = relation_forward_sample_part362_w = None
        getitem_13 = kwargs['Xvelocity'];  kwargs = None
        relation_forward_sample_part8_w = self.all_constants.SamplePart8
        einsum_13 = torch.functional.einsum('bij,ki->bkj', getitem_13, relation_forward_sample_part8_w);  getitem_13 = relation_forward_sample_part8_w = None
        all_constants_sigma_v = self.all_constants.sigma_v
        mul_8 = einsum_1 * all_constants_sigma_v;  einsum_1 = all_constants_sigma_v = None
        all_constants_sigma_omega = self.all_constants.sigma_omega
        mul_9 = einsum_3 * all_constants_sigma_omega;  einsum_3 = all_constants_sigma_omega = None
        all_constants_sigma_theta = self.all_constants.sigma_theta
        mul_10 = einsum_5 * all_constants_sigma_theta;  einsum_5 = all_constants_sigma_theta = None
        all_constants_sigma_force = self.all_constants.sigma_force
        mul_11 = einsum_8 * all_constants_sigma_force;  einsum_8 = all_constants_sigma_force = None
        all_constants_sigma_x = self.all_constants.sigma_x
        mul_12 = einsum_7 * all_constants_sigma_x;  einsum_7 = all_constants_sigma_x = None
        add = einsum_13 + mul_8;  einsum_13 = mul_8 = None
        add_1 = einsum_2 + mul_9;  einsum_2 = mul_9 = None
        add_2 = einsum_4 + mul_10;  einsum_4 = mul_10 = None
        add_3 = einsum_6 + mul_11;  einsum_6 = mul_11 = None
        add_4 = einsum + mul_12;  einsum = mul_12 = None
        cos = torch.cos(add_2)
        all_constants_dt = self.all_constants.dt
        mul_13 = add_1 * all_constants_dt
        all_parameters_a6 = self.all_parameters.a6
        mul_14 = all_parameters_a6 * add_3
        mul_15 = all_parameters_a6 * add_3
        mul_16 = all_parameters_a6 * add_3
        all_parameters_a14 = self.all_parameters.a14
        mul_17 = all_parameters_a14 * cos
        all_parameters_a2 = self.all_parameters.a2
        mul_18 = all_parameters_a2 * add
        all_parameters_a8 = self.all_parameters.a8
        mul_19 = all_parameters_a8 * add_1
        mul_20 = mul_6 * cos;  mul_6 = None
        all_parameters_a5 = self.all_parameters.a5
        mul_21 = all_parameters_a5 * add_1
        mul_22 = mul_21 * cos;  mul_21 = None
        mul_23 = all_parameters_a6 * add_3;  all_parameters_a6 = None
        all_parameters_a11 = self.all_parameters.a11
        mul_24 = all_parameters_a11 * add
        mul_25 = mul_24 * cos;  mul_24 = None
        all_parameters_a12 = self.all_parameters.a12
        mul_26 = all_parameters_a12 * cos
        mul_27 = mul_26 * add_3;  mul_26 = None
        all_constants_constant34 = self.all_constants.Constant34
        pow_1 = torch.pow(mul_17, all_constants_constant34);  mul_17 = all_constants_constant34 = None
        all_constants_constant35 = self.all_constants.Constant35
        pow_2 = torch.pow(add_1, all_constants_constant35);  all_constants_constant35 = None
        all_constants_constant36 = self.all_constants.Constant36
        pow_3 = torch.pow(add_1, all_constants_constant36);  all_constants_constant36 = None
        sin = torch.sin(add_2)
        all_parameters_a13 = self.all_parameters.a13
        sub = all_parameters_a13 - pow_1;  pow_1 = None
        all_constants_constant39 = self.all_constants.Constant39
        truediv = mul_13 / all_constants_constant39;  mul_13 = all_constants_constant39 = None
        all_parameters_a1 = self.all_parameters.a1
        mul_28 = all_parameters_a1 * pow_2;  pow_2 = None
        mul_29 = mul_28 * sin;  mul_28 = None
        mul_30 = mul_20 * sin;  mul_20 = None
        mul_31 = neg_3 * pow_3;  neg_3 = pow_3 = None
        mul_32 = mul_31 * cos;  mul_31 = cos = None
        mul_33 = mul_32 * sin;  mul_32 = None
        mul_34 = mul_7 * sin;  mul_7 = sin = None
        sub_1 = mul_29 - mul_18;  mul_29 = mul_18 = None
        sub_2 = sub_1 - mul_30;  sub_1 = mul_30 = None
        sub_3 = mul_33 - mul_19;  mul_33 = mul_19 = None
        add_5 = add_2 + truediv;  truediv = None
        add_6 = sub_2 + mul_22;  sub_2 = mul_22 = None
        add_7 = add_6 + mul_23;  add_6 = mul_23 = None
        add_8 = sub_3 + mul_34;  sub_3 = mul_34 = None
        add_9 = add_8 + mul_25;  add_8 = mul_25 = None
        cos_1 = torch.cos(add_5)
        truediv_1 = add_7 / sub;  add_7 = None
        mul_35 = all_parameters_a14 * cos_1
        mul_36 = mul * cos_1;  mul = None
        mul_37 = all_parameters_a12 * cos_1
        mul_38 = mul_37 * add_3;  mul_37 = None
        mul_39 = truediv_1 * all_constants_dt
        all_constants_constant41 = self.all_constants.Constant41
        pow_4 = torch.pow(mul_35, all_constants_constant41);  mul_35 = all_constants_constant41 = None
        sin_1 = torch.sin(add_5);  add_5 = None
        sub_4 = all_parameters_a13 - pow_4;  pow_4 = None
        sub_5 = add_9 - mul_27;  add_9 = mul_27 = None
        truediv_2 = sub_5 / sub;  sub_5 = sub = None
        all_constants_constant38 = self.all_constants.Constant38
        truediv_3 = mul_39 / all_constants_constant38;  mul_39 = all_constants_constant38 = None
        mul_40 = truediv_2 * all_constants_dt
        mul_41 = mul_36 * sin_1;  mul_36 = None
        mul_42 = mul_1 * sin_1;  mul_1 = None
        add_10 = add + truediv_3;  truediv_3 = None
        all_constants_constant40 = self.all_constants.Constant40
        truediv_4 = mul_40 / all_constants_constant40;  mul_40 = all_constants_constant40 = None
        mul_43 = all_parameters_a2 * add_10
        mul_44 = all_parameters_a11 * add_10
        mul_45 = mul_44 * cos_1;  mul_44 = None
        all_constants_constant57 = self.all_constants.Constant57
        mul_46 = all_constants_constant57 * add_10;  all_constants_constant57 = add_10 = None
        add_11 = add_1 + truediv_4;  truediv_4 = None
        add_12 = add + mul_46;  mul_46 = None
        mul_47 = all_parameters_a8 * add_11
        mul_48 = all_parameters_a5 * add_11
        mul_49 = mul_48 * cos_1;  mul_48 = None
        mul_50 = add_11 * all_constants_dt
        all_constants_constant63 = self.all_constants.Constant63
        mul_51 = all_constants_constant63 * add_11;  all_constants_constant63 = None
        all_constants_constant42 = self.all_constants.Constant42
        pow_5 = torch.pow(add_11, all_constants_constant42);  all_constants_constant42 = None
        all_constants_constant43 = self.all_constants.Constant43
        pow_6 = torch.pow(add_11, all_constants_constant43);  add_11 = all_constants_constant43 = None
        add_13 = add_1 + mul_51;  mul_51 = None
        all_constants_constant46 = self.all_constants.Constant46
        truediv_5 = mul_50 / all_constants_constant46;  mul_50 = all_constants_constant46 = None
        mul_52 = all_parameters_a1 * pow_5;  pow_5 = None
        mul_53 = mul_52 * sin_1;  mul_52 = None
        mul_54 = neg * pow_6;  neg = pow_6 = None
        mul_55 = mul_54 * cos_1;  mul_54 = cos_1 = None
        mul_56 = mul_55 * sin_1;  mul_55 = sin_1 = None
        sub_6 = mul_53 - mul_43;  mul_53 = mul_43 = None
        sub_7 = sub_6 - mul_41;  sub_6 = mul_41 = None
        sub_8 = mul_56 - mul_47;  mul_56 = mul_47 = None
        add_14 = sub_7 + mul_49;  sub_7 = mul_49 = None
        add_15 = add_14 + mul_14;  add_14 = mul_14 = None
        add_16 = sub_8 + mul_42;  sub_8 = mul_42 = None
        add_17 = add_16 + mul_45;  add_16 = mul_45 = None
        add_18 = add_2 + truediv_5;  truediv_5 = None
        cos_2 = torch.cos(add_18)
        truediv_6 = add_15 / sub_4;  add_15 = None
        mul_57 = truediv_6 * all_constants_dt
        mul_58 = all_parameters_a14 * cos_2
        mul_59 = mul_2 * cos_2;  mul_2 = None
        mul_60 = all_parameters_a12 * cos_2
        mul_61 = mul_60 * add_3;  mul_60 = None
        all_constants_constant54 = self.all_constants.Constant54
        mul_62 = all_constants_constant54 * truediv_6;  all_constants_constant54 = truediv_6 = None
        all_constants_constant48 = self.all_constants.Constant48
        pow_7 = torch.pow(mul_58, all_constants_constant48);  mul_58 = all_constants_constant48 = None
        sin_2 = torch.sin(add_18);  add_18 = None
        sub_9 = add_17 - mul_38;  add_17 = mul_38 = None
        sub_10 = all_parameters_a13 - pow_7;  pow_7 = None
        add_19 = truediv_1 + mul_62;  truediv_1 = mul_62 = None
        truediv_7 = sub_9 / sub_4;  sub_9 = sub_4 = None
        all_constants_constant45 = self.all_constants.Constant45
        truediv_8 = mul_57 / all_constants_constant45;  mul_57 = all_constants_constant45 = None
        mul_63 = truediv_7 * all_constants_dt
        mul_64 = mul_59 * sin_2;  mul_59 = None
        mul_65 = mul_3 * sin_2;  mul_3 = None
        all_constants_constant60 = self.all_constants.Constant60
        mul_66 = all_constants_constant60 * truediv_7;  all_constants_constant60 = truediv_7 = None
        add_20 = add + truediv_8;  truediv_8 = None
        add_21 = truediv_2 + mul_66;  truediv_2 = mul_66 = None
        all_constants_constant47 = self.all_constants.Constant47
        truediv_9 = mul_63 / all_constants_constant47;  mul_63 = all_constants_constant47 = None
        mul_67 = all_parameters_a2 * add_20
        mul_68 = all_parameters_a11 * add_20
        mul_69 = mul_68 * cos_2;  mul_68 = None
        all_constants_constant58 = self.all_constants.Constant58
        mul_70 = all_constants_constant58 * add_20;  all_constants_constant58 = add_20 = None
        add_22 = add_1 + truediv_9;  truediv_9 = None
        add_23 = add_12 + mul_70;  add_12 = mul_70 = None
        mul_71 = all_parameters_a8 * add_22
        mul_72 = all_parameters_a5 * add_22
        mul_73 = mul_72 * cos_2;  mul_72 = None
        mul_74 = add_22 * all_constants_dt
        all_constants_constant64 = self.all_constants.Constant64
        mul_75 = all_constants_constant64 * add_22;  all_constants_constant64 = None
        all_constants_constant49 = self.all_constants.Constant49
        pow_8 = torch.pow(add_22, all_constants_constant49);  all_constants_constant49 = None
        all_constants_constant50 = self.all_constants.Constant50
        pow_9 = torch.pow(add_22, all_constants_constant50);  add_22 = all_constants_constant50 = None
        add_24 = add_2 + mul_74;  mul_74 = None
        add_25 = add_13 + mul_75;  add_13 = mul_75 = None
        cos_3 = torch.cos(add_24)
        mul_76 = all_parameters_a1 * pow_8;  pow_8 = None
        mul_77 = mul_76 * sin_2;  mul_76 = None
        mul_78 = neg_1 * pow_9;  neg_1 = pow_9 = None
        mul_79 = mul_78 * cos_2;  mul_78 = cos_2 = None
        mul_80 = mul_79 * sin_2;  mul_79 = sin_2 = None
        mul_81 = all_parameters_a14 * cos_3;  all_parameters_a14 = None
        mul_82 = mul_4 * cos_3;  mul_4 = None
        mul_83 = all_parameters_a12 * cos_3;  all_parameters_a12 = None
        mul_84 = mul_83 * add_3;  mul_83 = add_3 = None
        all_constants_constant51 = self.all_constants.Constant51
        pow_10 = torch.pow(mul_81, all_constants_constant51);  mul_81 = all_constants_constant51 = None
        sin_3 = torch.sin(add_24);  add_24 = None
        sub_11 = mul_77 - mul_67;  mul_77 = mul_67 = None
        sub_12 = sub_11 - mul_64;  sub_11 = mul_64 = None
        sub_13 = mul_80 - mul_71;  mul_80 = mul_71 = None
        sub_14 = all_parameters_a13 - pow_10;  all_parameters_a13 = pow_10 = None
        add_26 = sub_12 + mul_73;  sub_12 = mul_73 = None
        add_27 = add_26 + mul_15;  add_26 = mul_15 = None
        add_28 = sub_13 + mul_65;  sub_13 = mul_65 = None
        add_29 = add_28 + mul_69;  add_28 = mul_69 = None
        truediv_10 = add_27 / sub_10;  add_27 = None
        mul_85 = truediv_10 * all_constants_dt
        mul_86 = mul_82 * sin_3;  mul_82 = None
        mul_87 = mul_5 * sin_3;  mul_5 = None
        all_constants_constant55 = self.all_constants.Constant55
        mul_88 = all_constants_constant55 * truediv_10;  all_constants_constant55 = truediv_10 = None
        sub_15 = add_29 - mul_61;  add_29 = mul_61 = None
        add_30 = add + mul_85;  mul_85 = None
        add_31 = add_19 + mul_88;  add_19 = mul_88 = None
        add_32 = add_23 + add_30;  add_23 = None
        truediv_11 = sub_15 / sub_10;  sub_15 = sub_10 = None
        mul_89 = truediv_11 * all_constants_dt
        mul_90 = all_parameters_a2 * add_30;  all_parameters_a2 = None
        mul_91 = all_parameters_a11 * add_30;  all_parameters_a11 = add_30 = None
        mul_92 = mul_91 * cos_3;  mul_91 = None
        mul_93 = add_32 * all_constants_dt;  add_32 = None
        all_constants_constant61 = self.all_constants.Constant61
        mul_94 = all_constants_constant61 * truediv_11;  all_constants_constant61 = truediv_11 = None
        add_33 = add_1 + mul_89;  mul_89 = None
        add_34 = add_21 + mul_94;  add_21 = mul_94 = None
        add_35 = add_25 + add_33;  add_25 = None
        all_constants_constant59 = self.all_constants.Constant59
        truediv_12 = mul_93 / all_constants_constant59;  mul_93 = all_constants_constant59 = None
        mul_95 = all_parameters_a8 * add_33;  all_parameters_a8 = None
        mul_96 = all_parameters_a5 * add_33;  all_parameters_a5 = None
        mul_97 = mul_96 * cos_3;  mul_96 = None
        mul_98 = add_35 * all_constants_dt;  add_35 = None
        all_constants_constant52 = self.all_constants.Constant52
        pow_11 = torch.pow(add_33, all_constants_constant52);  all_constants_constant52 = None
        all_constants_constant53 = self.all_constants.Constant53
        pow_12 = torch.pow(add_33, all_constants_constant53);  add_33 = all_constants_constant53 = None
        add_36 = add_4 + truediv_12;  add_4 = truediv_12 = None
        all_constants_constant65 = self.all_constants.Constant65
        truediv_13 = mul_98 / all_constants_constant65;  mul_98 = all_constants_constant65 = None
        mul_99 = all_parameters_a1 * pow_11;  all_parameters_a1 = pow_11 = None
        mul_100 = mul_99 * sin_3;  mul_99 = None
        mul_101 = neg_2 * pow_12;  neg_2 = pow_12 = None
        mul_102 = mul_101 * cos_3;  mul_101 = cos_3 = None
        mul_103 = mul_102 * sin_3;  mul_102 = sin_3 = None
        sub_16 = mul_100 - mul_90;  mul_100 = mul_90 = None
        sub_17 = sub_16 - mul_86;  sub_16 = mul_86 = None
        sub_18 = mul_103 - mul_95;  mul_103 = mul_95 = None
        add_37 = sub_17 + mul_97;  sub_17 = mul_97 = None
        add_38 = add_37 + mul_16;  add_37 = mul_16 = None
        add_39 = sub_18 + mul_87;  sub_18 = mul_87 = None
        add_40 = add_39 + mul_92;  add_39 = mul_92 = None
        add_41 = add_2 + truediv_13;  add_2 = truediv_13 = None
        truediv_14 = add_38 / sub_14;  add_38 = None
        sub_19 = add_40 - mul_84;  add_40 = mul_84 = None
        add_42 = add_31 + truediv_14;  add_31 = truediv_14 = None
        truediv_15 = sub_19 / sub_14;  sub_19 = sub_14 = None
        mul_104 = add_42 * all_constants_dt;  add_42 = None
        add_43 = add_34 + truediv_15;  add_34 = truediv_15 = None
        all_constants_constant56 = self.all_constants.Constant56
        truediv_16 = mul_104 / all_constants_constant56;  mul_104 = all_constants_constant56 = None
        mul_105 = add_43 * all_constants_dt;  add_43 = all_constants_dt = None
        add_44 = add + truediv_16;  add = truediv_16 = None
        all_constants_constant62 = self.all_constants.Constant62
        truediv_17 = mul_105 / all_constants_constant62;  mul_105 = all_constants_constant62 = None
        add_45 = add_1 + truediv_17;  add_1 = truediv_17 = None
        return ({'est_theta': add_41, 'est_thetadot': add_45, 'est_x': add_36, 'est_xdot': add_44}, {'SamplePart356': einsum_9, 'SamplePart358': einsum_10, 'SamplePart362': einsum_12, 'SamplePart360': einsum_11, 'Add338': add_45, 'Add350': add_41, 'Add326': add_36, 'Add314': add_44}, {'Xangle': add_41, 'Xangular_velocity': add_45, 'Xpos': add_36, 'Xvelocity': add_44}, {})
        
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
