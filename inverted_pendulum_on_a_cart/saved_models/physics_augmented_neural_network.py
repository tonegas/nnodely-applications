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
        self.all_constants["Constant41"] = torch.tensor([2.0], requires_grad=False)
        self.all_constants["Constant42"] = torch.tensor([2.0], requires_grad=False)
        self.all_constants["Constant43"] = torch.tensor([2.0], requires_grad=False)
        self.all_constants["Constant44"] = torch.tensor([2.0], requires_grad=False)
        self.all_constants["Constant45"] = torch.tensor([2.0], requires_grad=False)
        self.all_constants["Constant46"] = torch.tensor([2.0], requires_grad=False)
        self.all_constants["Constant47"] = torch.tensor([2.0], requires_grad=False)
        self.all_constants["Constant48"] = torch.tensor([2.0], requires_grad=False)
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
        self.all_constants["Constant83"] = torch.tensor([6.0], requires_grad=False)
        self.all_constants["Constant84"] = torch.tensor([2.0], requires_grad=False)
        self.all_constants["Constant85"] = torch.tensor([2.0], requires_grad=False)
        self.all_constants["Constant86"] = torch.tensor([6.0], requires_grad=False)
        self.all_constants["Constant87"] = torch.tensor([2.0], requires_grad=False)
        self.all_constants["Constant88"] = torch.tensor([2.0], requires_grad=False)
        self.all_constants["Constant89"] = torch.tensor([6.0], requires_grad=False)
        self.all_constants["Constant90"] = torch.tensor([2.0], requires_grad=False)
        self.all_constants["Constant91"] = torch.tensor([2.0], requires_grad=False)
        self.all_constants["Constant92"] = torch.tensor([6.0], requires_grad=False)
        self.all_constants["dt"] = torch.tensor([0.019999999552965164], requires_grad=False)
        self.all_constants["g"] = torch.tensor([9.8100004196167], requires_grad=False)
        self.all_constants["sigma_force"] = torch.tensor([0.0], requires_grad=False)
        self.all_constants["sigma_omega"] = torch.tensor([0.0], requires_grad=False)
        self.all_constants["sigma_theta"] = torch.tensor([0.0], requires_grad=False)
        self.all_constants["sigma_v"] = torch.tensor([0.0], requires_grad=False)
        self.all_constants["sigma_x"] = torch.tensor([0.0], requires_grad=False)
        self.all_parameters["I"] = torch.nn.Parameter(torch.tensor([0.3665934205055237]), requires_grad=True)
        self.all_parameters["b"] = torch.nn.Parameter(torch.tensor([1.1572990417480469]), requires_grad=True)
        self.all_parameters["d"] = torch.nn.Parameter(torch.tensor([0.9200180768966675]), requires_grad=True)
        self.all_parameters["gear"] = torch.nn.Parameter(torch.tensor([92.17991638183594]), requires_grad=True)
        self.all_parameters["l"] = torch.nn.Parameter(torch.tensor([0.16484428942203522]), requires_grad=True)
        self.all_parameters["m1"] = torch.nn.Parameter(torch.tensor([5.707699298858643]), requires_grad=True)
        self.all_parameters["m2"] = torch.nn.Parameter(torch.tensor([8.428751945495605]), requires_grad=True)
        self.all_parameters["PLinear29W"] = torch.nn.Parameter(torch.tensor([[0.1339799165725708, 0.029055997729301453, 0.0901598408818245, 0.02187737263739109, -0.40687552094459534, -0.061862513422966, 0.073885478079319, 0.24921028316020966, -0.34111401438713074, -0.0335845872759819]]), requires_grad=True)
        self.all_parameters["PLinear31W"] = torch.nn.Parameter(torch.tensor([[-0.015274163335561752, 0.037064600735902786, -0.032421763986349106, 0.02278393879532814, 0.0911608561873436, 0.06094522401690483, -0.012738989666104317, -0.07245787233114243, 0.07643961161375046, -0.00038468706770800054]]), requires_grad=True)
        self.all_parameters["PLinear33W"] = torch.nn.Parameter(torch.tensor([[-0.0014596972614526749, -0.05327482149004936, 0.005350697319954634, -0.04364042729139328, 0.005023951176553965, 0.0021279947832226753, -0.013751748017966747, -0.005382625386118889, 0.003641919931396842, -0.0063543543219566345]]), requires_grad=True)
        self.all_parameters["PLinear35W"] = torch.nn.Parameter(torch.tensor([[-0.028899969533085823, 0.024189023301005363, 0.0731528252363205, 0.028463633731007576, -0.06439018994569778, -0.03797878697514534, 0.02911299467086792, 0.07108074426651001, -0.05557240545749664, 0.0163437332957983]]), requires_grad=True)
        self.all_parameters["PLinear37W"] = torch.nn.Parameter(torch.tensor([[0.1158600002527237, -0.113904669880867, 0.1473758965730667, -0.1297556310892105, 0.13717201352119446, -0.07345994561910629, -0.36757394671440125, 0.13015739619731903, -0.10960540920495987, -0.14561152458190918], [-0.04158630222082138, -0.012368869036436081, -0.03945964202284813, 0.18343447148799896, -0.03848446160554886, 0.4302118122577667, 0.10172691941261292, -0.03693754971027374, -0.027750207111239433, -0.0008555480162613094], [-0.15900832414627075, 0.07811800390481949, 0.007560492493212223, 0.05739264190196991, -0.12242865562438965, 0.09293336421251297, 0.33929768204689026, -0.03630021587014198, 0.054978467524051666, 0.09337621927261353], [0.052688438445329666, -0.011684111319482327, 0.06339462846517563, -0.23552720248699188, 0.02603539451956749, -0.42918533086776733, -0.038638196885585785, 0.007741205394268036, -0.0010974914766848087, -0.026332108303904533], [-0.4360366761684418, 0.39213937520980835, -0.257805198431015, 0.04064437374472618, -0.4452415108680725, -0.26700687408447266, 0.14069314301013947, -0.3390043377876282, 0.37585732340812683, 0.41765275597572327], [0.12035846710205078, 0.005024496931582689, -0.01806093007326126, -0.07486511021852493, 0.06955859065055847, -0.18568632006645203, -0.27796950936317444, 0.00625486159697175, 0.0037590747233480215, -0.009188300929963589], [-0.15588638186454773, 0.06122367084026337, 0.023844039067626, -0.009529123082756996, -0.13166674971580505, -0.026931453496217728, 0.531968355178833, -0.04087487608194351, 0.05037373676896095, 0.07474790513515472], [-0.23034074902534485, 0.17068135738372803, -0.006437335163354874, -0.09281938523054123, -0.2116357982158661, -0.22263112664222717, 0.5551456809043884, -0.08514589816331863, 0.15762844681739807, 0.198935866355896], [0.33641570806503296, -0.2992698550224304, 0.17985613644123077, -0.0917915552854538, 0.31408196687698364, 0.07099668681621552, 0.6257125735282898, 0.2504812479019165, -0.3045444190502167, -0.30458828806877136], [0.10669057816267014, 0.010860339738428593, 0.0019190269522368908, -0.12608498334884644, 0.07611685246229172, -0.21446296572685242, -0.29395750164985657, 0.004336431156843901, 0.022175382822752, -0.0031059144530445337]]), requires_grad=True)
        self.all_parameters["PLinear39W"] = torch.nn.Parameter(torch.tensor([[0.01584816537797451, 0.002101324964314699, 0.22798039019107819, 0.001093197613954544], [-0.030034629628062248, -0.00028604682302102447, 0.07001148164272308, 0.009054356254637241], [0.12718172371387482, 0.006596095860004425, 0.16990353167057037, 0.0014446963323280215], [-0.03711821883916855, 0.0014384817332029343, -0.032344426959753036, 0.0014871311141178012], [0.00883575715124607, 0.00472089322283864, 0.15384739637374878, 0.004416007548570633], [-0.2058391571044922, -0.00212186761200428, 0.09624464064836502, 0.0031838121358305216], [0.024641338735818863, 0.0005291750421747565, 0.12437859177589417, -0.0004970211302861571], [0.0329015776515007, -0.004327105823904276, 0.13789543509483337, -0.0029533361084759235], [-0.0411948524415493, 0.009191468358039856, 0.0880482941865921, 0.006490145344287157], [-0.01957927644252777, -4.4422609789762646e-05, 0.061949461698532104, -0.00206306716427207]]), requires_grad=True)
        self.all_constants["SamplePart1"] = torch.tensor([[1.0]], requires_grad=True)
        self.all_constants["SamplePart10"] = torch.tensor([[1.0]], requires_grad=True)
        self.all_constants["SamplePart15"] = torch.tensor([[1.0]], requires_grad=True)
        self.all_constants["SamplePart17"] = torch.tensor([[1.0]], requires_grad=True)
        self.all_constants["SamplePart22"] = torch.tensor([[1.0]], requires_grad=True)
        self.all_constants["SamplePart24"] = torch.tensor([[1.0]], requires_grad=True)
        self.all_constants["SamplePart29"] = torch.tensor([[1.0]], requires_grad=True)
        self.all_constants["SamplePart3"] = torch.tensor([[1.0]], requires_grad=True)
        self.all_constants["SamplePart31"] = torch.tensor([[1.0]], requires_grad=True)
        self.all_constants["SamplePart575"] = torch.tensor([[1.0]], requires_grad=True)
        self.all_constants["SamplePart577"] = torch.tensor([[1.0]], requires_grad=True)
        self.all_constants["SamplePart579"] = torch.tensor([[1.0]], requires_grad=True)
        self.all_constants["SamplePart581"] = torch.tensor([[1.0]], requires_grad=True)
        self.all_constants["SamplePart8"] = torch.tensor([[1.0]], requires_grad=True)
        self.all_constants["Select522"] = torch.tensor([1.0, 0.0, 0.0, 0.0], requires_grad=True)
        self.all_constants["Select536"] = torch.tensor([0.0, 1.0, 0.0, 0.0], requires_grad=True)
        self.all_constants["Select550"] = torch.tensor([0.0, 0.0, 1.0, 0.0], requires_grad=True)
        self.all_constants["Select564"] = torch.tensor([0.0, 0.0, 0.0, 1.0], requires_grad=True)
        self.all_constants["Select570"] = torch.tensor([0.0, 0.0, 0.0, 1.0], requires_grad=True)
        self.all_constants["Select571"] = torch.tensor([0.0, 0.0, 1.0, 0.0], requires_grad=True)
        self.all_constants["Select572"] = torch.tensor([0.0, 1.0, 0.0, 0.0], requires_grad=True)
        self.all_constants["Select573"] = torch.tensor([1.0, 0.0, 0.0, 0.0], requires_grad=True)
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
        mul_38 = mul_37 * all_constants_g;  mul_37 = None
        mul_39 = mul_38 * all_parameters_l;  mul_38 = None
        all_constants_constant47 = self.all_constants.Constant47
        pow_1 = torch.pow(all_parameters_m2, all_constants_constant47);  all_constants_constant47 = None
        all_constants_constant48 = self.all_constants.Constant48
        pow_2 = torch.pow(all_parameters_l, all_constants_constant48);  all_constants_constant48 = None
        all_constants_constant53 = self.all_constants.Constant53
        pow_3 = torch.pow(all_parameters_l, all_constants_constant53);  all_constants_constant53 = None
        all_constants_constant55 = self.all_constants.Constant55
        pow_4 = torch.pow(all_parameters_m2, all_constants_constant55);  all_constants_constant55 = None
        all_constants_constant56 = self.all_constants.Constant56
        pow_5 = torch.pow(all_parameters_l, all_constants_constant56);  all_constants_constant56 = None
        all_constants_constant59 = self.all_constants.Constant59
        pow_6 = torch.pow(all_parameters_m2, all_constants_constant59);  all_constants_constant59 = None
        all_constants_constant60 = self.all_constants.Constant60
        pow_7 = torch.pow(all_parameters_l, all_constants_constant60);  all_constants_constant60 = None
        all_constants_constant65 = self.all_constants.Constant65
        pow_8 = torch.pow(all_parameters_l, all_constants_constant65);  all_constants_constant65 = None
        all_constants_constant67 = self.all_constants.Constant67
        pow_9 = torch.pow(all_parameters_m2, all_constants_constant67);  all_constants_constant67 = None
        all_constants_constant68 = self.all_constants.Constant68
        pow_10 = torch.pow(all_parameters_l, all_constants_constant68);  all_constants_constant68 = None
        all_constants_constant71 = self.all_constants.Constant71
        pow_11 = torch.pow(all_parameters_m2, all_constants_constant71);  all_constants_constant71 = None
        all_constants_constant72 = self.all_constants.Constant72
        pow_12 = torch.pow(all_parameters_l, all_constants_constant72);  all_constants_constant72 = None
        all_constants_constant73 = self.all_constants.Constant73
        pow_13 = torch.pow(all_parameters_l, all_constants_constant73);  all_constants_constant73 = None
        all_constants_constant75 = self.all_constants.Constant75
        pow_14 = torch.pow(all_parameters_m2, all_constants_constant75);  all_constants_constant75 = None
        all_constants_constant76 = self.all_constants.Constant76
        pow_15 = torch.pow(all_parameters_l, all_constants_constant76);  all_constants_constant76 = None
        all_constants_constant79 = self.all_constants.Constant79
        pow_16 = torch.pow(all_parameters_m2, all_constants_constant79);  all_constants_constant79 = None
        all_constants_constant80 = self.all_constants.Constant80
        pow_17 = torch.pow(all_parameters_l, all_constants_constant80);  all_constants_constant80 = None
        all_constants_constant41 = self.all_constants.Constant41
        pow_18 = torch.pow(all_parameters_l, all_constants_constant41);  all_constants_constant41 = None
        all_constants_constant43 = self.all_constants.Constant43
        pow_19 = torch.pow(all_parameters_m2, all_constants_constant43);  all_constants_constant43 = None
        all_constants_constant44 = self.all_constants.Constant44
        pow_20 = torch.pow(all_parameters_l, all_constants_constant44);  all_constants_constant44 = None
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
        relation_forward_sample_part575_w = self.all_constants.SamplePart575
        einsum_9 = torch.functional.einsum('bij,ki->bkj', getitem_9, relation_forward_sample_part575_w);  getitem_9 = relation_forward_sample_part575_w = None
        getitem_10 = kwargs['Yangle']
        relation_forward_sample_part577_w = self.all_constants.SamplePart577
        einsum_10 = torch.functional.einsum('bij,ki->bkj', getitem_10, relation_forward_sample_part577_w);  getitem_10 = relation_forward_sample_part577_w = None
        getitem_11 = kwargs['Yvelocity']
        relation_forward_sample_part579_w = self.all_constants.SamplePart579
        einsum_11 = torch.functional.einsum('bij,ki->bkj', getitem_11, relation_forward_sample_part579_w);  getitem_11 = relation_forward_sample_part579_w = None
        getitem_12 = kwargs['Ypos']
        relation_forward_sample_part581_w = self.all_constants.SamplePart581
        einsum_12 = torch.functional.einsum('bij,ki->bkj', getitem_12, relation_forward_sample_part581_w);  getitem_12 = relation_forward_sample_part581_w = None
        getitem_13 = kwargs['Xvelocity'];  kwargs = None
        relation_forward_sample_part8_w = self.all_constants.SamplePart8
        einsum_13 = torch.functional.einsum('bij,ki->bkj', getitem_13, relation_forward_sample_part8_w);  getitem_13 = relation_forward_sample_part8_w = None
        all_constants_sigma_v = self.all_constants.sigma_v
        mul_40 = einsum_1 * all_constants_sigma_v;  einsum_1 = all_constants_sigma_v = None
        mul_41 = pow_1 * pow_2;  pow_1 = pow_2 = None
        mul_42 = mul_41 * all_constants_g;  mul_41 = None
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
        mul_53 = pow_16 * pow_17;  pow_16 = pow_17 = None
        mul_54 = mul_53 * all_constants_g;  mul_53 = all_constants_g = None
        all_constants_sigma_x = self.all_constants.sigma_x
        mul_55 = einsum_7 * all_constants_sigma_x;  einsum_7 = all_constants_sigma_x = None
        mul_56 = all_parameters_m2 * pow_18;  pow_18 = None
        neg = -pow_4;  pow_4 = None
        neg_1 = -pow_9;  pow_9 = None
        neg_2 = -pow_14;  pow_14 = None
        neg_3 = -pow_19;  pow_19 = None
        add_12 = einsum_13 + mul_40;  einsum_13 = mul_40 = None
        all_parameters_i = self.all_parameters.I
        add_13 = all_parameters_i + mul_43;  mul_43 = None
        add_14 = einsum_2 + mul_44;  einsum_2 = mul_44 = None
        add_15 = einsum_4 + mul_47;  einsum_4 = mul_47 = None
        add_16 = all_parameters_i + mul_48;  mul_48 = None
        add_17 = einsum_6 + mul_49;  einsum_6 = mul_49 = None
        add_18 = all_parameters_i + mul_52;  mul_52 = None
        add_19 = all_parameters_i + mul_56;  all_parameters_i = mul_56 = None
        add_20 = einsum + mul_55;  einsum = mul_55 = None
        cos = torch.cos(add_15)
        relation_forward_linear35_weights = self.all_parameters.PLinear29W
        einsum_14 = torch.functional.einsum('bwi,io->bwo', add_15, relation_forward_linear35_weights);  relation_forward_linear35_weights = None
        relation_forward_linear36_weights = self.all_parameters.PLinear31W
        einsum_15 = torch.functional.einsum('bwi,io->bwo', add_12, relation_forward_linear36_weights);  relation_forward_linear36_weights = None
        relation_forward_linear37_weights = self.all_parameters.PLinear33W
        einsum_16 = torch.functional.einsum('bwi,io->bwo', add_14, relation_forward_linear37_weights);  relation_forward_linear37_weights = None
        relation_forward_linear38_weights = self.all_parameters.PLinear35W
        einsum_17 = torch.functional.einsum('bwi,io->bwo', add_17, relation_forward_linear38_weights);  relation_forward_linear38_weights = None
        mul_57 = mul_1 * add_12;  mul_1 = None
        mul_58 = mul_57 * cos;  mul_57 = None
        mul_59 = mul_2 * cos;  mul_2 = None
        mul_60 = add_19 * all_parameters_m2
        mul_61 = mul_60 * all_parameters_l;  mul_60 = None
        mul_62 = mul_4 * add_14;  mul_4 = None
        mul_63 = mul_62 * cos;  mul_62 = None
        all_constants_dt = self.all_constants.dt
        mul_64 = add_14 * all_constants_dt
        mul_65 = add * add_13;  add = None
        all_parameters_gear = self.all_parameters.gear
        mul_66 = all_parameters_gear * add_17
        mul_67 = add_13 * all_parameters_b
        mul_68 = neg * pow_5;  neg = pow_5 = None
        mul_69 = add_13 * all_parameters_m2
        mul_70 = mul_69 * all_parameters_l;  mul_69 = None
        mul_71 = mul_66 * add_13;  add_13 = None
        mul_72 = add_3 * add_16;  add_3 = None
        mul_73 = all_parameters_gear * add_17
        mul_74 = add_16 * all_parameters_b
        mul_75 = neg_1 * pow_10;  neg_1 = pow_10 = None
        mul_76 = add_16 * all_parameters_m2
        mul_77 = mul_76 * all_parameters_l;  mul_76 = None
        mul_78 = mul_73 * add_16;  add_16 = None
        mul_79 = add_6 * add_18;  add_6 = None
        mul_80 = all_parameters_gear * add_17
        mul_81 = add_18 * all_parameters_b
        mul_82 = neg_2 * pow_15;  neg_2 = pow_15 = None
        mul_83 = add_18 * all_parameters_m2;  all_parameters_m2 = None
        mul_84 = mul_83 * all_parameters_l;  mul_83 = all_parameters_l = None
        mul_85 = mul_80 * add_18;  add_18 = None
        mul_86 = add_9 * add_19;  add_9 = None
        mul_87 = mul_35 * cos;  mul_35 = None
        mul_88 = all_parameters_gear * add_17;  all_parameters_gear = add_17 = None
        mul_89 = add_19 * all_parameters_b;  all_parameters_b = None
        mul_90 = mul_89 * add_12;  mul_89 = None
        mul_91 = mul_36 * add_14;  mul_36 = None
        mul_92 = neg_3 * pow_20;  neg_3 = pow_20 = None
        all_constants_constant46 = self.all_constants.Constant46
        pow_21 = torch.pow(add_14, all_constants_constant46);  all_constants_constant46 = None
        all_constants_constant42 = self.all_constants.Constant42
        pow_22 = torch.pow(mul_87, all_constants_constant42);  mul_87 = all_constants_constant42 = None
        all_constants_constant45 = self.all_constants.Constant45
        pow_23 = torch.pow(add_14, all_constants_constant45);  all_constants_constant45 = None
        sin = torch.sin(add_15)
        sub = mul_86 - pow_22;  mul_86 = pow_22 = None
        add_21 = einsum_14 + einsum_15;  einsum_14 = einsum_15 = None
        add_22 = add_21 + einsum_16;  add_21 = einsum_16 = None
        add_23 = add_22 + einsum_17;  add_22 = einsum_17 = None
        all_constants_constant51 = self.all_constants.Constant51
        truediv = mul_64 / all_constants_constant51;  mul_64 = all_constants_constant51 = None
        mul_93 = mul_39 * sin;  mul_39 = None
        mul_94 = mul_59 * mul_88;  mul_59 = None
        mul_95 = mul_61 * pow_21;  mul_61 = pow_21 = None
        mul_96 = mul_95 * sin;  mul_95 = None
        mul_97 = mul_42 * sin;  mul_42 = None
        mul_98 = mul_97 * cos;  mul_97 = None
        mul_99 = mul_88 * add_19;  mul_88 = add_19 = None
        mul_100 = mul_92 * pow_23;  mul_92 = pow_23 = None
        mul_101 = mul_100 * sin;  mul_100 = sin = None
        mul_102 = mul_101 * cos;  mul_101 = cos = None
        sub_1 = mul_96 - mul_90;  mul_96 = mul_90 = None
        sub_2 = sub_1 - mul_98;  sub_1 = mul_98 = None
        sub_3 = mul_102 - mul_91;  mul_102 = mul_91 = None
        tanh = torch.tanh(add_23);  add_23 = None
        add_24 = sub_3 + mul_93;  sub_3 = mul_93 = None
        add_25 = add_24 + mul_58;  add_24 = mul_58 = None
        add_26 = sub_2 + mul_63;  sub_2 = mul_63 = None
        add_27 = add_26 + mul_99;  add_26 = mul_99 = None
        add_28 = add_15 + truediv;  truediv = None
        cos_1 = torch.cos(add_28)
        truediv_1 = add_27 / sub;  add_27 = None
        relation_forward_linear43_weights = self.all_parameters.PLinear37W
        einsum_18 = torch.functional.einsum('bwi,io->bwo', tanh, relation_forward_linear43_weights);  tanh = relation_forward_linear43_weights = None
        mul_103 = truediv_1 * all_constants_dt
        mul_104 = mul_5 * cos_1;  mul_5 = None
        mul_105 = mul_12 * cos_1;  mul_12 = None
        mul_106 = mul_105 * mul_66;  mul_105 = mul_66 = None
        all_constants_constant54 = self.all_constants.Constant54
        pow_24 = torch.pow(mul_104, all_constants_constant54);  mul_104 = all_constants_constant54 = None
        sin_1 = torch.sin(add_28);  add_28 = None
        sub_4 = add_25 - mul_94;  add_25 = mul_94 = None
        sub_5 = mul_65 - pow_24;  mul_65 = pow_24 = None
        tanh_1 = torch.tanh(einsum_18);  einsum_18 = None
        truediv_2 = sub_4 / sub;  sub_4 = sub = None
        all_constants_constant50 = self.all_constants.Constant50
        truediv_3 = mul_103 / all_constants_constant50;  mul_103 = all_constants_constant50 = None
        relation_forward_linear45_weights = self.all_parameters.PLinear39W
        einsum_19 = torch.functional.einsum('bwi,io->bwo', tanh_1, relation_forward_linear45_weights);  tanh_1 = relation_forward_linear45_weights = None
        mul_107 = truediv_2 * all_constants_dt
        mul_108 = mul_9 * sin_1;  mul_9 = None
        mul_109 = mul_46 * sin_1;  mul_46 = None
        mul_110 = mul_109 * cos_1;  mul_109 = None
        relation_forward_select522_w = self.all_constants.Select522
        einsum_20 = torch.functional.einsum('ijk,k->ij', einsum_19, relation_forward_select522_w);  relation_forward_select522_w = None
        unsqueeze = einsum_20.unsqueeze(2);  einsum_20 = None
        relation_forward_select536_w = self.all_constants.Select536
        einsum_21 = torch.functional.einsum('ijk,k->ij', einsum_19, relation_forward_select536_w);  relation_forward_select536_w = None
        unsqueeze_1 = einsum_21.unsqueeze(2);  einsum_21 = None
        relation_forward_select550_w = self.all_constants.Select550
        einsum_22 = torch.functional.einsum('ijk,k->ij', einsum_19, relation_forward_select550_w);  relation_forward_select550_w = None
        unsqueeze_2 = einsum_22.unsqueeze(2);  einsum_22 = None
        relation_forward_select564_w = self.all_constants.Select564
        einsum_23 = torch.functional.einsum('ijk,k->ij', einsum_19, relation_forward_select564_w);  relation_forward_select564_w = None
        unsqueeze_3 = einsum_23.unsqueeze(2);  einsum_23 = None
        relation_forward_select570_w = self.all_constants.Select570
        einsum_24 = torch.functional.einsum('ijk,k->ij', einsum_19, relation_forward_select570_w);  relation_forward_select570_w = None
        unsqueeze_4 = einsum_24.unsqueeze(2);  einsum_24 = None
        relation_forward_select571_w = self.all_constants.Select571
        einsum_25 = torch.functional.einsum('ijk,k->ij', einsum_19, relation_forward_select571_w);  relation_forward_select571_w = None
        unsqueeze_5 = einsum_25.unsqueeze(2);  einsum_25 = None
        relation_forward_select572_w = self.all_constants.Select572
        einsum_26 = torch.functional.einsum('ijk,k->ij', einsum_19, relation_forward_select572_w);  relation_forward_select572_w = None
        unsqueeze_6 = einsum_26.unsqueeze(2);  einsum_26 = None
        relation_forward_select573_w = self.all_constants.Select573
        einsum_27 = torch.functional.einsum('ijk,k->ij', einsum_19, relation_forward_select573_w);  einsum_19 = relation_forward_select573_w = None
        unsqueeze_7 = einsum_27.unsqueeze(2);  einsum_27 = None
        add_29 = add_12 + truediv_3;  truediv_3 = None
        all_constants_constant52 = self.all_constants.Constant52
        truediv_4 = mul_107 / all_constants_constant52;  mul_107 = all_constants_constant52 = None
        mul_111 = mul_67 * add_29;  mul_67 = None
        mul_112 = mul_11 * add_29;  mul_11 = None
        mul_113 = mul_112 * cos_1;  mul_112 = None
        all_constants_constant84 = self.all_constants.Constant84
        mul_114 = all_constants_constant84 * add_29;  all_constants_constant84 = add_29 = None
        add_30 = add_14 + truediv_4;  truediv_4 = None
        add_31 = add_12 + mul_114;  mul_114 = None
        mul_115 = mul_6 * add_30;  mul_6 = None
        mul_116 = mul_14 * add_30;  mul_14 = None
        mul_117 = mul_116 * cos_1;  mul_116 = None
        mul_118 = add_30 * all_constants_dt
        all_constants_constant90 = self.all_constants.Constant90
        mul_119 = all_constants_constant90 * add_30;  all_constants_constant90 = None
        all_constants_constant57 = self.all_constants.Constant57
        pow_25 = torch.pow(add_30, all_constants_constant57);  all_constants_constant57 = None
        all_constants_constant58 = self.all_constants.Constant58
        pow_26 = torch.pow(add_30, all_constants_constant58);  add_30 = all_constants_constant58 = None
        add_32 = add_14 + mul_119;  mul_119 = None
        all_constants_constant63 = self.all_constants.Constant63
        truediv_5 = mul_118 / all_constants_constant63;  mul_118 = all_constants_constant63 = None
        mul_120 = mul_68 * pow_25;  mul_68 = pow_25 = None
        mul_121 = mul_120 * sin_1;  mul_120 = None
        mul_122 = mul_121 * cos_1;  mul_121 = cos_1 = None
        mul_123 = mul_70 * pow_26;  mul_70 = pow_26 = None
        mul_124 = mul_123 * sin_1;  mul_123 = sin_1 = None
        sub_6 = mul_122 - mul_115;  mul_122 = mul_115 = None
        sub_7 = mul_124 - mul_111;  mul_124 = mul_111 = None
        sub_8 = sub_7 - mul_110;  sub_7 = mul_110 = None
        add_33 = sub_6 + mul_108;  sub_6 = mul_108 = None
        add_34 = add_33 + mul_113;  add_33 = mul_113 = None
        add_35 = sub_8 + mul_117;  sub_8 = mul_117 = None
        add_36 = add_35 + mul_71;  add_35 = mul_71 = None
        add_37 = add_15 + truediv_5;  truediv_5 = None
        cos_2 = torch.cos(add_37)
        truediv_6 = add_36 / sub_5;  add_36 = None
        mul_125 = truediv_6 * all_constants_dt
        mul_126 = mul_15 * cos_2;  mul_15 = None
        mul_127 = mul_22 * cos_2;  mul_22 = None
        mul_128 = mul_127 * mul_73;  mul_127 = mul_73 = None
        all_constants_constant81 = self.all_constants.Constant81
        mul_129 = all_constants_constant81 * truediv_6;  all_constants_constant81 = truediv_6 = None
        all_constants_constant66 = self.all_constants.Constant66
        pow_27 = torch.pow(mul_126, all_constants_constant66);  mul_126 = all_constants_constant66 = None
        sin_2 = torch.sin(add_37);  add_37 = None
        sub_9 = add_34 - mul_106;  add_34 = mul_106 = None
        sub_10 = mul_72 - pow_27;  mul_72 = pow_27 = None
        add_38 = truediv_1 + mul_129;  truediv_1 = mul_129 = None
        truediv_7 = sub_9 / sub_5;  sub_9 = sub_5 = None
        all_constants_constant62 = self.all_constants.Constant62
        truediv_8 = mul_125 / all_constants_constant62;  mul_125 = all_constants_constant62 = None
        mul_130 = truediv_7 * all_constants_dt
        mul_131 = mul_19 * sin_2;  mul_19 = None
        mul_132 = mul_51 * sin_2;  mul_51 = None
        mul_133 = mul_132 * cos_2;  mul_132 = None
        all_constants_constant87 = self.all_constants.Constant87
        mul_134 = all_constants_constant87 * truediv_7;  all_constants_constant87 = truediv_7 = None
        add_39 = add_12 + truediv_8;  truediv_8 = None
        add_40 = truediv_2 + mul_134;  truediv_2 = mul_134 = None
        all_constants_constant64 = self.all_constants.Constant64
        truediv_9 = mul_130 / all_constants_constant64;  mul_130 = all_constants_constant64 = None
        mul_135 = mul_74 * add_39;  mul_74 = None
        mul_136 = mul_21 * add_39;  mul_21 = None
        mul_137 = mul_136 * cos_2;  mul_136 = None
        all_constants_constant85 = self.all_constants.Constant85
        mul_138 = all_constants_constant85 * add_39;  all_constants_constant85 = add_39 = None
        add_41 = add_14 + truediv_9;  truediv_9 = None
        add_42 = add_31 + mul_138;  add_31 = mul_138 = None
        mul_139 = mul_16 * add_41;  mul_16 = None
        mul_140 = mul_24 * add_41;  mul_24 = None
        mul_141 = mul_140 * cos_2;  mul_140 = None
        mul_142 = add_41 * all_constants_dt
        all_constants_constant91 = self.all_constants.Constant91
        mul_143 = all_constants_constant91 * add_41;  all_constants_constant91 = None
        all_constants_constant69 = self.all_constants.Constant69
        pow_28 = torch.pow(add_41, all_constants_constant69);  all_constants_constant69 = None
        all_constants_constant70 = self.all_constants.Constant70
        pow_29 = torch.pow(add_41, all_constants_constant70);  add_41 = all_constants_constant70 = None
        add_43 = add_15 + mul_142;  mul_142 = None
        add_44 = add_32 + mul_143;  add_32 = mul_143 = None
        cos_3 = torch.cos(add_43)
        mul_144 = mul_75 * pow_28;  mul_75 = pow_28 = None
        mul_145 = mul_144 * sin_2;  mul_144 = None
        mul_146 = mul_145 * cos_2;  mul_145 = cos_2 = None
        mul_147 = mul_77 * pow_29;  mul_77 = pow_29 = None
        mul_148 = mul_147 * sin_2;  mul_147 = sin_2 = None
        mul_149 = mul_25 * cos_3;  mul_25 = None
        mul_150 = mul_32 * cos_3;  mul_32 = None
        mul_151 = mul_150 * mul_80;  mul_150 = mul_80 = None
        all_constants_constant74 = self.all_constants.Constant74
        pow_30 = torch.pow(mul_149, all_constants_constant74);  mul_149 = all_constants_constant74 = None
        sin_3 = torch.sin(add_43);  add_43 = None
        sub_11 = mul_146 - mul_139;  mul_146 = mul_139 = None
        sub_12 = mul_148 - mul_135;  mul_148 = mul_135 = None
        sub_13 = sub_12 - mul_133;  sub_12 = mul_133 = None
        sub_14 = mul_79 - pow_30;  mul_79 = pow_30 = None
        add_45 = sub_11 + mul_131;  sub_11 = mul_131 = None
        add_46 = add_45 + mul_137;  add_45 = mul_137 = None
        add_47 = sub_13 + mul_141;  sub_13 = mul_141 = None
        add_48 = add_47 + mul_78;  add_47 = mul_78 = None
        truediv_10 = add_48 / sub_10;  add_48 = None
        mul_152 = truediv_10 * all_constants_dt
        mul_153 = mul_29 * sin_3;  mul_29 = None
        mul_154 = mul_54 * sin_3;  mul_54 = None
        mul_155 = mul_154 * cos_3;  mul_154 = None
        all_constants_constant82 = self.all_constants.Constant82
        mul_156 = all_constants_constant82 * truediv_10;  all_constants_constant82 = truediv_10 = None
        sub_15 = add_46 - mul_128;  add_46 = mul_128 = None
        add_49 = add_12 + mul_152;  mul_152 = None
        add_50 = add_38 + mul_156;  add_38 = mul_156 = None
        add_51 = add_42 + add_49;  add_42 = None
        truediv_11 = sub_15 / sub_10;  sub_15 = sub_10 = None
        mul_157 = truediv_11 * all_constants_dt
        mul_158 = mul_81 * add_49;  mul_81 = None
        mul_159 = mul_31 * add_49;  mul_31 = add_49 = None
        mul_160 = mul_159 * cos_3;  mul_159 = None
        mul_161 = add_51 * all_constants_dt;  add_51 = None
        all_constants_constant88 = self.all_constants.Constant88
        mul_162 = all_constants_constant88 * truediv_11;  all_constants_constant88 = truediv_11 = None
        add_52 = add_14 + mul_157;  mul_157 = None
        add_53 = add_40 + mul_162;  add_40 = mul_162 = None
        add_54 = add_44 + add_52;  add_44 = None
        all_constants_constant86 = self.all_constants.Constant86
        truediv_12 = mul_161 / all_constants_constant86;  mul_161 = all_constants_constant86 = None
        mul_163 = mul_26 * add_52;  mul_26 = None
        mul_164 = mul_34 * add_52;  mul_34 = None
        mul_165 = mul_164 * cos_3;  mul_164 = None
        mul_166 = add_54 * all_constants_dt;  add_54 = None
        all_constants_constant77 = self.all_constants.Constant77
        pow_31 = torch.pow(add_52, all_constants_constant77);  all_constants_constant77 = None
        all_constants_constant78 = self.all_constants.Constant78
        pow_32 = torch.pow(add_52, all_constants_constant78);  add_52 = all_constants_constant78 = None
        add_55 = add_20 + truediv_12;  add_20 = truediv_12 = None
        add_56 = add_55 + unsqueeze_1;  add_55 = unsqueeze_1 = None
        all_constants_constant92 = self.all_constants.Constant92
        truediv_13 = mul_166 / all_constants_constant92;  mul_166 = all_constants_constant92 = None
        mul_167 = mul_82 * pow_31;  mul_82 = pow_31 = None
        mul_168 = mul_167 * sin_3;  mul_167 = None
        mul_169 = mul_168 * cos_3;  mul_168 = cos_3 = None
        mul_170 = mul_84 * pow_32;  mul_84 = pow_32 = None
        mul_171 = mul_170 * sin_3;  mul_170 = sin_3 = None
        sub_16 = mul_169 - mul_163;  mul_169 = mul_163 = None
        sub_17 = mul_171 - mul_158;  mul_171 = mul_158 = None
        sub_18 = sub_17 - mul_155;  sub_17 = mul_155 = None
        add_57 = sub_16 + mul_153;  sub_16 = mul_153 = None
        add_58 = add_57 + mul_160;  add_57 = mul_160 = None
        add_59 = sub_18 + mul_165;  sub_18 = mul_165 = None
        add_60 = add_59 + mul_85;  add_59 = mul_85 = None
        add_61 = add_15 + truediv_13;  add_15 = truediv_13 = None
        add_62 = add_61 + unsqueeze_3;  add_61 = unsqueeze_3 = None
        truediv_14 = add_60 / sub_14;  add_60 = None
        sub_19 = add_58 - mul_151;  add_58 = mul_151 = None
        add_63 = add_50 + truediv_14;  add_50 = truediv_14 = None
        truediv_15 = sub_19 / sub_14;  sub_19 = sub_14 = None
        mul_172 = add_63 * all_constants_dt;  add_63 = None
        add_64 = add_53 + truediv_15;  add_53 = truediv_15 = None
        all_constants_constant83 = self.all_constants.Constant83
        truediv_16 = mul_172 / all_constants_constant83;  mul_172 = all_constants_constant83 = None
        mul_173 = add_64 * all_constants_dt;  add_64 = all_constants_dt = None
        add_65 = add_12 + truediv_16;  add_12 = truediv_16 = None
        add_66 = add_65 + unsqueeze;  add_65 = unsqueeze = None
        all_constants_constant89 = self.all_constants.Constant89
        truediv_17 = mul_173 / all_constants_constant89;  mul_173 = all_constants_constant89 = None
        add_67 = add_14 + truediv_17;  add_14 = truediv_17 = None
        add_68 = add_67 + unsqueeze_2;  add_67 = unsqueeze_2 = None
        return ({'contr_theta': unsqueeze_4, 'contr_thetadot': unsqueeze_5, 'contr_x': unsqueeze_7, 'contr_xdot': unsqueeze_6, 'est_theta': add_62, 'est_thetadot': add_68, 'est_x': add_56, 'est_xdot': add_66}, {'SamplePart575': einsum_9, 'SamplePart577': einsum_10, 'SamplePart581': einsum_12, 'SamplePart579': einsum_11, 'Add551': add_68, 'Add565': add_62, 'Add537': add_56, 'Add523': add_66}, {'Xangle': add_62, 'Xangular_velocity': add_68, 'Xpos': add_56, 'Xvelocity': add_66}, {})
        
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
