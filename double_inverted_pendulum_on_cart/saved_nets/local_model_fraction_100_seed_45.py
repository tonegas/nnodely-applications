import torch

def nnodely_basic_model_update_state(data_in, rel):
    data_out = data_in.clone()
    max_dim = min(rel.size(1), data_in.size(1))
    data_out[:, -max_dim:, :] = rel[:, -max_dim:, :]
    return data_out

def nnodely_basic_model_timeshift(data_in):
    return torch.cat((data_in[:, 1:, :], data_in[:, :1, :]), dim=1)

def nnodely_layers_fuzzify_slicing(res, i, x):
    res[:, :, i:i+1] = x

class TracerModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.all_parameters = {}
        self.all_constants = {}
        self._tensor_constant0 = torch.tensor(0.0)
        self._tensor_constant1 = torch.tensor(1.0)
        self._tensor_constant10 = torch.tensor(0.0)
        self._tensor_constant11 = torch.tensor(3)
        self._tensor_constant12 = torch.tensor(0.0)
        self._tensor_constant13 = torch.tensor(1.0)
        self._tensor_constant14 = torch.tensor(4)
        self._tensor_constant15 = torch.tensor(0.0)
        self._tensor_constant16 = torch.tensor(1.0)
        self._tensor_constant17 = torch.tensor(0)
        self._tensor_constant18 = torch.tensor(0.0)
        self._tensor_constant19 = torch.tensor(0.0)
        self._tensor_constant2 = torch.tensor(0)
        self._tensor_constant20 = torch.tensor(1)
        self._tensor_constant21 = torch.tensor(0.0)
        self._tensor_constant22 = torch.tensor(0.0)
        self._tensor_constant23 = torch.tensor(2)
        self._tensor_constant24 = torch.tensor(0.0)
        self._tensor_constant25 = torch.tensor(0.0)
        self._tensor_constant26 = torch.tensor(3)
        self._tensor_constant27 = torch.tensor(0.0)
        self._tensor_constant28 = torch.tensor(1.0)
        self._tensor_constant29 = torch.tensor(4)
        self._tensor_constant3 = torch.tensor(0.0)
        self._tensor_constant30 = torch.tensor(0.0)
        self._tensor_constant31 = torch.tensor(1.0)
        self._tensor_constant32 = torch.tensor(0)
        self._tensor_constant33 = torch.tensor(0.0)
        self._tensor_constant34 = torch.tensor(0.0)
        self._tensor_constant35 = torch.tensor(1)
        self._tensor_constant36 = torch.tensor(0.0)
        self._tensor_constant37 = torch.tensor(0.0)
        self._tensor_constant38 = torch.tensor(2)
        self._tensor_constant39 = torch.tensor(0.0)
        self._tensor_constant4 = torch.tensor(0.0)
        self._tensor_constant40 = torch.tensor(0.0)
        self._tensor_constant41 = torch.tensor(3)
        self._tensor_constant42 = torch.tensor(0.0)
        self._tensor_constant43 = torch.tensor(1.0)
        self._tensor_constant44 = torch.tensor(4)
        self._tensor_constant5 = torch.tensor(1)
        self._tensor_constant6 = torch.tensor(0.0)
        self._tensor_constant7 = torch.tensor(0.0)
        self._tensor_constant8 = torch.tensor(2)
        self._tensor_constant9 = torch.tensor(0.0)
        self.all_constants["SampleTime"] = torch.tensor(0.009999999776482582, requires_grad=False)
        self.all_parameters["PLinear15b"] = torch.nn.Parameter(torch.tensor([0.7320238947868347]), requires_grad=True)
        self.all_parameters["Lin_cart_0"] = torch.nn.Parameter(torch.tensor([[42.22535705566406], [0.03298371657729149], [-0.006843626964837313], [5.830634117126465], [0.5146127939224243], [-0.5593050718307495], [0.19870060682296753], [-0.011298584751784801], [0.00863967277109623]]), requires_grad=True)
        self.all_parameters["PLinear18b"] = torch.nn.Parameter(torch.tensor([0.18206217885017395]), requires_grad=True)
        self.all_parameters["Lin_cart_1"] = torch.nn.Parameter(torch.tensor([[42.25515365600586], [0.03289857506752014], [-0.0009934157133102417], [5.809866905212402], [0.2111794650554657], [-0.5541363954544067], [-0.034249551594257355], [-0.0046072728000581264], [0.004806199111044407]]), requires_grad=True)
        self.all_parameters["PLinear21b"] = torch.nn.Parameter(torch.tensor([0.03284340724349022]), requires_grad=True)
        self.all_parameters["Lin_cart_2"] = torch.nn.Parameter(torch.tensor([[42.249176025390625], [0.02419828437268734], [-0.0008445162093266845], [5.814788341522217], [0.022990694269537926], [-0.5541952252388], [0.009926180355250835], [-0.01368597149848938], [0.011378702707588673]]), requires_grad=True)
        self.all_parameters["PLinear24b"] = torch.nn.Parameter(torch.tensor([0.12788943946361542]), requires_grad=True)
        self.all_parameters["Lin_cart_3"] = torch.nn.Parameter(torch.tensor([[42.25447463989258], [0.03449968993663788], [0.0012061852030456066], [5.801887035369873], [-0.07059016823768616], [-0.5551334023475647], [0.20665980875492096], [-0.0031802018638700247], [0.0022170094307512045]]), requires_grad=True)
        self.all_parameters["PLinear27b"] = torch.nn.Parameter(torch.tensor([0.2696792781352997]), requires_grad=True)
        self.all_parameters["Lin_cart_4"] = torch.nn.Parameter(torch.tensor([[42.24095153808594], [0.000494120002258569], [-0.005494824610650539], [5.845009803771973], [-0.021854743361473083], [-0.572150468826294], [0.2958143353462219], [-0.019475558772683144], [0.018814217299222946]]), requires_grad=True)
        self.all_parameters["PLinear31b"] = torch.nn.Parameter(torch.tensor([0.04819480702280998]), requires_grad=True)
        self.all_parameters["Lin_pend1_0"] = torch.nn.Parameter(torch.tensor([[86.242919921875], [0.04137282073497772], [0.09739992022514343], [32.52565002441406], [0.3962099850177765], [-19.397289276123047], [0.3010002076625824], [-1.168418288230896], [0.05577128008008003]]), requires_grad=True)
        self.all_parameters["PLinear34b"] = torch.nn.Parameter(torch.tensor([0.40968120098114014]), requires_grad=True)
        self.all_parameters["Lin_pend1_1"] = torch.nn.Parameter(torch.tensor([[86.40929412841797], [-0.006560610607266426], [-0.006743579637259245], [31.964670181274414], [0.512851893901825], [-18.939970016479492], [0.37656131386756897], [-1.9910985231399536], [-0.02273460291326046]]), requires_grad=True)
        self.all_parameters["PLinear37b"] = torch.nn.Parameter(torch.tensor([0.16823852062225342]), requires_grad=True)
        self.all_parameters["Lin_pend1_2"] = torch.nn.Parameter(torch.tensor([[86.3125991821289], [-0.008489664644002914], [-1.482704101363197e-05], [31.934019088745117], [0.09269718080759048], [-18.913108825683594], [0.07755149900913239], [-2.0008697509765625], [-0.008712418377399445]]), requires_grad=True)
        self.all_parameters["PLinear40b"] = torch.nn.Parameter(torch.tensor([0.30829092860221863]), requires_grad=True)
        self.all_parameters["Lin_pend1_3"] = torch.nn.Parameter(torch.tensor([[86.3825454711914], [-0.006354339420795441], [-0.003608567873016], [31.955074310302734], [-0.13040581345558167], [-18.932430267333984], [-0.04599098116159439], [-2.008570432662964], [-0.016474539414048195]]), requires_grad=True)
        self.all_parameters["PLinear43b"] = torch.nn.Parameter(torch.tensor([0.7862429022789001]), requires_grad=True)
        self.all_parameters["Lin_pend1_4"] = torch.nn.Parameter(torch.tensor([[86.20155334472656], [-0.14070655405521393], [-0.253192275762558], [32.8763542175293], [-0.02936507947742939], [-19.527427673339844], [0.05133116990327835], [-1.3990744352340698], [-0.06340057402849197]]), requires_grad=True)
        self.all_parameters["PLinear47b"] = torch.nn.Parameter(torch.tensor([0.18953174352645874]), requires_grad=True)
        self.all_parameters["Lin_pend2_0"] = torch.nn.Parameter(torch.tensor([[-62.40509796142578], [-0.19577676057815552], [0.19814454019069672], [-22.53580665588379], [0.20036061108112335], [-30.27839469909668], [0.07644934207201004], [-0.5846542119979858], [0.42505863308906555]]), requires_grad=True)
        self.all_parameters["PLinear50b"] = torch.nn.Parameter(torch.tensor([0.17495958507061005]), requires_grad=True)
        self.all_parameters["Lin_pend2_1"] = torch.nn.Parameter(torch.tensor([[-63.714168548583984], [0.006936203222721815], [0.00048626060015521944], [-23.52730369567871], [0.38436150550842285], [-29.640029907226562], [0.14330953359603882], [-0.0015545905334874988], [-1.5065127611160278]]), requires_grad=True)
        self.all_parameters["PLinear53b"] = torch.nn.Parameter(torch.tensor([0.11867731809616089]), requires_grad=True)
        self.all_parameters["Lin_pend2_2"] = torch.nn.Parameter(torch.tensor([[-63.5864372253418], [0.005254864692687988], [-0.0021521009039133787], [-23.54424476623535], [0.03592071309685707], [-29.661745071411133], [0.08169130980968475], [-0.00908749457448721], [-1.518660545349121]]), requires_grad=True)
        self.all_parameters["PLinear56b"] = torch.nn.Parameter(torch.tensor([0.34491419792175293]), requires_grad=True)
        self.all_parameters["Lin_pend2_3"] = torch.nn.Parameter(torch.tensor([[-63.583396911621094], [0.004186008125543594], [-0.0018641221104189754], [-23.523391723632812], [-0.15115413069725037], [-29.63206672668457], [0.14697115123271942], [-0.0018971865065395832], [-1.5130950212478638]]), requires_grad=True)
        self.all_parameters["PLinear59b"] = torch.nn.Parameter(torch.tensor([0.25941482186317444]), requires_grad=True)
        self.all_parameters["Lin_pend2_4"] = torch.nn.Parameter(torch.tensor([[-63.30183410644531], [0.15996426343917847], [0.6254562139511108], [-22.33116912841797], [-0.2461412250995636], [-30.69117546081543], [-0.16854606568813324], [-0.3949339687824249], [-1.4652094841003418]]), requires_grad=True)
        self.all_constants["SamplePart1"] = torch.tensor([[1.0]], requires_grad=True)
        self.all_constants["SamplePart10"] = torch.tensor([[1.0]], requires_grad=True)
        self.all_constants["SamplePart103"] = torch.tensor([[1.0]], requires_grad=True)
        self.all_constants["SamplePart109"] = torch.tensor([[1.0]], requires_grad=True)
        self.all_constants["SamplePart115"] = torch.tensor([[1.0]], requires_grad=True)
        self.all_constants["SamplePart121"] = torch.tensor([[1.0]], requires_grad=True)
        self.all_constants["SamplePart127"] = torch.tensor([[1.0]], requires_grad=True)
        self.all_constants["SamplePart13"] = torch.tensor([[1.0]], requires_grad=True)
        self.all_constants["SamplePart139"] = torch.tensor([[1.0]], requires_grad=True)
        self.all_constants["SamplePart141"] = torch.tensor([[1.0]], requires_grad=True)
        self.all_constants["SamplePart143"] = torch.tensor([[1.0]], requires_grad=True)
        self.all_constants["SamplePart17"] = torch.tensor([[1.0]], requires_grad=True)
        self.all_constants["SamplePart20"] = torch.tensor([[1.0]], requires_grad=True)
        self.all_constants["SamplePart25"] = torch.tensor([[1.0]], requires_grad=True)
        self.all_constants["SamplePart27"] = torch.tensor([[1.0]], requires_grad=True)
        self.all_constants["SamplePart30"] = torch.tensor([[1.0]], requires_grad=True)
        self.all_constants["SamplePart32"] = torch.tensor([[1.0]], requires_grad=True)
        self.all_constants["SamplePart34"] = torch.tensor([[1.0]], requires_grad=True)
        self.all_constants["SamplePart4"] = torch.tensor([[1.0]], requires_grad=True)
        self.all_constants["SamplePart7"] = torch.tensor([[1.0]], requires_grad=True)
        self.all_constants["SamplePart97"] = torch.tensor([[1.0]], requires_grad=True)
        self.all_constants["Select40"] = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0], requires_grad=True)
        self.all_constants["Select43"] = torch.tensor([0.0, 1.0, 0.0, 0.0, 0.0], requires_grad=True)
        self.all_constants["Select46"] = torch.tensor([0.0, 0.0, 1.0, 0.0, 0.0], requires_grad=True)
        self.all_constants["Select49"] = torch.tensor([0.0, 0.0, 0.0, 1.0, 0.0], requires_grad=True)
        self.all_constants["Select52"] = torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0], requires_grad=True)
        self.all_constants["Select59"] = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0], requires_grad=True)
        self.all_constants["Select62"] = torch.tensor([0.0, 1.0, 0.0, 0.0, 0.0], requires_grad=True)
        self.all_constants["Select65"] = torch.tensor([0.0, 0.0, 1.0, 0.0, 0.0], requires_grad=True)
        self.all_constants["Select68"] = torch.tensor([0.0, 0.0, 0.0, 1.0, 0.0], requires_grad=True)
        self.all_constants["Select71"] = torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0], requires_grad=True)
        self.all_constants["Select78"] = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0], requires_grad=True)
        self.all_constants["Select81"] = torch.tensor([0.0, 1.0, 0.0, 0.0, 0.0], requires_grad=True)
        self.all_constants["Select84"] = torch.tensor([0.0, 0.0, 1.0, 0.0, 0.0], requires_grad=True)
        self.all_constants["Select87"] = torch.tensor([0.0, 0.0, 0.0, 1.0, 0.0], requires_grad=True)
        self.all_constants["Select90"] = torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0], requires_grad=True)
        self.all_parameters = torch.nn.ParameterDict(self.all_parameters)
        self.all_constants = torch.nn.ParameterDict(self.all_constants)

    def update(self, closed_loop={}, connect={}, disconnect=False):
        pass
    def forward(self, kwargs):
        getitem = kwargs['Xth2_dot']
        relation_forward_sample_part7_w = self.all_constants.SamplePart7
        einsum = torch.functional.einsum('bij,ki->bkj', getitem, relation_forward_sample_part7_w);  getitem = relation_forward_sample_part7_w = None
        zeros_like = torch.zeros_like(einsum)
        repeat = zeros_like.repeat(1, 1, 5);  zeros_like = None
        sub = einsum - -0.5
        neg = -sub;  sub = None
        truediv = neg / 0.25;  neg = None
        add = truediv + 1;  truediv = None
        _tensor_constant0 = self._tensor_constant0
        maximum = torch.maximum(add, _tensor_constant0);  add = _tensor_constant0 = None
        _tensor_constant1 = self._tensor_constant1
        minimum = torch.minimum(maximum, _tensor_constant1);  maximum = _tensor_constant1 = None
        _tensor_constant2 = self._tensor_constant2
        slicing = nnodely_layers_fuzzify_slicing(repeat, _tensor_constant2, minimum);  _tensor_constant2 = minimum = slicing = None
        sub_1 = einsum - -0.5
        truediv_1 = sub_1 / 0.25;  sub_1 = None
        _tensor_constant3 = self._tensor_constant3
        maximum_1 = torch.maximum(truediv_1, _tensor_constant3);  truediv_1 = _tensor_constant3 = None
        sub_2 = einsum - -0.25
        neg_1 = -sub_2;  sub_2 = None
        truediv_2 = neg_1 / 0.25;  neg_1 = None
        add_1 = truediv_2 + 1;  truediv_2 = None
        _tensor_constant4 = self._tensor_constant4
        maximum_2 = torch.maximum(add_1, _tensor_constant4);  add_1 = _tensor_constant4 = None
        minimum_1 = torch.minimum(maximum_1, maximum_2);  maximum_1 = maximum_2 = None
        _tensor_constant5 = self._tensor_constant5
        slicing_1 = nnodely_layers_fuzzify_slicing(repeat, _tensor_constant5, minimum_1);  _tensor_constant5 = minimum_1 = slicing_1 = None
        sub_3 = einsum - -0.25
        truediv_3 = sub_3 / 0.25;  sub_3 = None
        _tensor_constant6 = self._tensor_constant6
        maximum_3 = torch.maximum(truediv_3, _tensor_constant6);  truediv_3 = _tensor_constant6 = None
        sub_4 = einsum - 0.0
        neg_2 = -sub_4;  sub_4 = None
        truediv_4 = neg_2 / 0.25;  neg_2 = None
        add_2 = truediv_4 + 1;  truediv_4 = None
        _tensor_constant7 = self._tensor_constant7
        maximum_4 = torch.maximum(add_2, _tensor_constant7);  add_2 = _tensor_constant7 = None
        minimum_2 = torch.minimum(maximum_3, maximum_4);  maximum_3 = maximum_4 = None
        _tensor_constant8 = self._tensor_constant8
        slicing_2 = nnodely_layers_fuzzify_slicing(repeat, _tensor_constant8, minimum_2);  _tensor_constant8 = minimum_2 = slicing_2 = None
        sub_5 = einsum - 0.0
        truediv_5 = sub_5 / 0.25;  sub_5 = None
        _tensor_constant9 = self._tensor_constant9
        maximum_5 = torch.maximum(truediv_5, _tensor_constant9);  truediv_5 = _tensor_constant9 = None
        sub_6 = einsum - 0.25
        neg_3 = -sub_6;  sub_6 = None
        truediv_6 = neg_3 / 0.25;  neg_3 = None
        add_3 = truediv_6 + 1;  truediv_6 = None
        _tensor_constant10 = self._tensor_constant10
        maximum_6 = torch.maximum(add_3, _tensor_constant10);  add_3 = _tensor_constant10 = None
        minimum_3 = torch.minimum(maximum_5, maximum_6);  maximum_5 = maximum_6 = None
        _tensor_constant11 = self._tensor_constant11
        slicing_3 = nnodely_layers_fuzzify_slicing(repeat, _tensor_constant11, minimum_3);  _tensor_constant11 = minimum_3 = slicing_3 = None
        sub_7 = einsum - 0.25;  einsum = None
        truediv_7 = sub_7 / 0.25;  sub_7 = None
        _tensor_constant12 = self._tensor_constant12
        maximum_7 = torch.maximum(truediv_7, _tensor_constant12);  truediv_7 = _tensor_constant12 = None
        _tensor_constant13 = self._tensor_constant13
        minimum_4 = torch.minimum(maximum_7, _tensor_constant13);  maximum_7 = _tensor_constant13 = None
        _tensor_constant14 = self._tensor_constant14
        slicing_4 = nnodely_layers_fuzzify_slicing(repeat, _tensor_constant14, minimum_4);  _tensor_constant14 = minimum_4 = slicing_4 = None
        relation_forward_select90_w = self.all_constants.Select90
        einsum_1 = torch.functional.einsum('ijk,k->ij', repeat, relation_forward_select90_w);  relation_forward_select90_w = None
        unsqueeze = einsum_1.unsqueeze(2);  einsum_1 = None
        getitem_1 = kwargs['Xth2_dot']
        relation_forward_sample_part27_w = self.all_constants.SamplePart27
        einsum_2 = torch.functional.einsum('bij,ki->bkj', getitem_1, relation_forward_sample_part27_w);  getitem_1 = relation_forward_sample_part27_w = None
        getitem_2 = kwargs['Xth1_dot']
        relation_forward_sample_part25_w = self.all_constants.SamplePart25
        einsum_3 = torch.functional.einsum('bij,ki->bkj', getitem_2, relation_forward_sample_part25_w);  getitem_2 = relation_forward_sample_part25_w = None
        cat = torch.cat((einsum_3, einsum_2), dim = 2);  einsum_3 = einsum_2 = None
        getitem_3 = kwargs['Xth2']
        relation_forward_sample_part20_w = self.all_constants.SamplePart20
        einsum_4 = torch.functional.einsum('bij,ki->bkj', getitem_3, relation_forward_sample_part20_w);  getitem_3 = relation_forward_sample_part20_w = None
        cos = torch.cos(einsum_4);  einsum_4 = None
        getitem_4 = kwargs['Xth2']
        relation_forward_sample_part17_w = self.all_constants.SamplePart17
        einsum_5 = torch.functional.einsum('bij,ki->bkj', getitem_4, relation_forward_sample_part17_w);  getitem_4 = relation_forward_sample_part17_w = None
        sin = torch.sin(einsum_5);  einsum_5 = None
        cat_1 = torch.cat((sin, cos), dim = 2);  sin = cos = None
        getitem_5 = kwargs['Xth1']
        relation_forward_sample_part13_w = self.all_constants.SamplePart13
        einsum_6 = torch.functional.einsum('bij,ki->bkj', getitem_5, relation_forward_sample_part13_w);  getitem_5 = relation_forward_sample_part13_w = None
        cos_1 = torch.cos(einsum_6);  einsum_6 = None
        getitem_6 = kwargs['Xth1']
        relation_forward_sample_part10_w = self.all_constants.SamplePart10
        einsum_7 = torch.functional.einsum('bij,ki->bkj', getitem_6, relation_forward_sample_part10_w);  getitem_6 = relation_forward_sample_part10_w = None
        sin_1 = torch.sin(einsum_7);  einsum_7 = None
        cat_2 = torch.cat((sin_1, cos_1), dim = 2);  sin_1 = cos_1 = None
        cat_3 = torch.cat((cat_2, cat_1), dim = 2);  cat_2 = cat_1 = None
        cat_4 = torch.cat((cat_3, cat), dim = 2);  cat_3 = cat = None
        getitem_7 = kwargs['Xpos']
        relation_forward_sample_part34_w = self.all_constants.SamplePart34
        einsum_8 = torch.functional.einsum('bij,ki->bkj', getitem_7, relation_forward_sample_part34_w);  getitem_7 = relation_forward_sample_part34_w = None
        getitem_8 = kwargs['Xvelocity']
        relation_forward_sample_part32_w = self.all_constants.SamplePart32
        einsum_9 = torch.functional.einsum('bij,ki->bkj', getitem_8, relation_forward_sample_part32_w);  getitem_8 = relation_forward_sample_part32_w = None
        cat_5 = torch.cat((einsum_9, einsum_8), dim = 2);  einsum_9 = einsum_8 = None
        getitem_9 = kwargs['action']
        relation_forward_sample_part30_w = self.all_constants.SamplePart30
        einsum_10 = torch.functional.einsum('bij,ki->bkj', getitem_9, relation_forward_sample_part30_w);  getitem_9 = relation_forward_sample_part30_w = None
        cat_6 = torch.cat((einsum_10, cat_5), dim = 2);  einsum_10 = cat_5 = None
        cat_7 = torch.cat((cat_6, cat_4), dim = 2);  cat_6 = cat_4 = None
        relation_forward_linear89_weights = self.all_parameters.Lin_pend2_4
        einsum_11 = torch.functional.einsum('bwi,io->bwo', cat_7, relation_forward_linear89_weights);  relation_forward_linear89_weights = None
        relation_forward_linear89_bias = self.all_parameters.PLinear59b
        add_4 = einsum_11 + relation_forward_linear89_bias;  einsum_11 = relation_forward_linear89_bias = None
        mul = add_4 * unsqueeze;  add_4 = unsqueeze = None
        relation_forward_select87_w = self.all_constants.Select87
        einsum_12 = torch.functional.einsum('ijk,k->ij', repeat, relation_forward_select87_w);  relation_forward_select87_w = None
        unsqueeze_1 = einsum_12.unsqueeze(2);  einsum_12 = None
        relation_forward_linear86_weights = self.all_parameters.Lin_pend2_3
        einsum_13 = torch.functional.einsum('bwi,io->bwo', cat_7, relation_forward_linear86_weights);  relation_forward_linear86_weights = None
        relation_forward_linear86_bias = self.all_parameters.PLinear56b
        add_5 = einsum_13 + relation_forward_linear86_bias;  einsum_13 = relation_forward_linear86_bias = None
        mul_1 = add_5 * unsqueeze_1;  add_5 = unsqueeze_1 = None
        relation_forward_select84_w = self.all_constants.Select84
        einsum_14 = torch.functional.einsum('ijk,k->ij', repeat, relation_forward_select84_w);  relation_forward_select84_w = None
        unsqueeze_2 = einsum_14.unsqueeze(2);  einsum_14 = None
        relation_forward_linear83_weights = self.all_parameters.Lin_pend2_2
        einsum_15 = torch.functional.einsum('bwi,io->bwo', cat_7, relation_forward_linear83_weights);  relation_forward_linear83_weights = None
        relation_forward_linear83_bias = self.all_parameters.PLinear53b
        add_6 = einsum_15 + relation_forward_linear83_bias;  einsum_15 = relation_forward_linear83_bias = None
        mul_2 = add_6 * unsqueeze_2;  add_6 = unsqueeze_2 = None
        relation_forward_select81_w = self.all_constants.Select81
        einsum_16 = torch.functional.einsum('ijk,k->ij', repeat, relation_forward_select81_w);  relation_forward_select81_w = None
        unsqueeze_3 = einsum_16.unsqueeze(2);  einsum_16 = None
        relation_forward_linear80_weights = self.all_parameters.Lin_pend2_1
        einsum_17 = torch.functional.einsum('bwi,io->bwo', cat_7, relation_forward_linear80_weights);  relation_forward_linear80_weights = None
        relation_forward_linear80_bias = self.all_parameters.PLinear50b
        add_7 = einsum_17 + relation_forward_linear80_bias;  einsum_17 = relation_forward_linear80_bias = None
        mul_3 = add_7 * unsqueeze_3;  add_7 = unsqueeze_3 = None
        relation_forward_select78_w = self.all_constants.Select78
        einsum_18 = torch.functional.einsum('ijk,k->ij', repeat, relation_forward_select78_w);  repeat = relation_forward_select78_w = None
        unsqueeze_4 = einsum_18.unsqueeze(2);  einsum_18 = None
        relation_forward_linear77_weights = self.all_parameters.Lin_pend2_0
        einsum_19 = torch.functional.einsum('bwi,io->bwo', cat_7, relation_forward_linear77_weights);  relation_forward_linear77_weights = None
        relation_forward_linear77_bias = self.all_parameters.PLinear47b
        add_8 = einsum_19 + relation_forward_linear77_bias;  einsum_19 = relation_forward_linear77_bias = None
        mul_4 = add_8 * unsqueeze_4;  add_8 = unsqueeze_4 = None
        add_9 = mul_4 + mul_3;  mul_4 = mul_3 = None
        add_10 = add_9 + mul_2;  add_9 = mul_2 = None
        add_11 = add_10 + mul_1;  add_10 = mul_1 = None
        add_12 = add_11 + mul;  add_11 = mul = None
        getitem_10 = kwargs['Xddth2']
        relation_forward_sample_part143_w = self.all_constants.SamplePart143
        einsum_20 = torch.functional.einsum('bij,ki->bkj', getitem_10, relation_forward_sample_part143_w);  getitem_10 = relation_forward_sample_part143_w = None
        getitem_11 = kwargs['Xth1_dot']
        relation_forward_sample_part4_w = self.all_constants.SamplePart4
        einsum_21 = torch.functional.einsum('bij,ki->bkj', getitem_11, relation_forward_sample_part4_w);  getitem_11 = relation_forward_sample_part4_w = None
        zeros_like_1 = torch.zeros_like(einsum_21)
        repeat_1 = zeros_like_1.repeat(1, 1, 5);  zeros_like_1 = None
        sub_8 = einsum_21 - -0.5
        neg_4 = -sub_8;  sub_8 = None
        truediv_8 = neg_4 / 0.25;  neg_4 = None
        add_13 = truediv_8 + 1;  truediv_8 = None
        _tensor_constant15 = self._tensor_constant15
        maximum_8 = torch.maximum(add_13, _tensor_constant15);  add_13 = _tensor_constant15 = None
        _tensor_constant16 = self._tensor_constant16
        minimum_5 = torch.minimum(maximum_8, _tensor_constant16);  maximum_8 = _tensor_constant16 = None
        _tensor_constant17 = self._tensor_constant17
        slicing_5 = nnodely_layers_fuzzify_slicing(repeat_1, _tensor_constant17, minimum_5);  _tensor_constant17 = minimum_5 = slicing_5 = None
        sub_9 = einsum_21 - -0.5
        truediv_9 = sub_9 / 0.25;  sub_9 = None
        _tensor_constant18 = self._tensor_constant18
        maximum_9 = torch.maximum(truediv_9, _tensor_constant18);  truediv_9 = _tensor_constant18 = None
        sub_10 = einsum_21 - -0.25
        neg_5 = -sub_10;  sub_10 = None
        truediv_10 = neg_5 / 0.25;  neg_5 = None
        add_14 = truediv_10 + 1;  truediv_10 = None
        _tensor_constant19 = self._tensor_constant19
        maximum_10 = torch.maximum(add_14, _tensor_constant19);  add_14 = _tensor_constant19 = None
        minimum_6 = torch.minimum(maximum_9, maximum_10);  maximum_9 = maximum_10 = None
        _tensor_constant20 = self._tensor_constant20
        slicing_6 = nnodely_layers_fuzzify_slicing(repeat_1, _tensor_constant20, minimum_6);  _tensor_constant20 = minimum_6 = slicing_6 = None
        sub_11 = einsum_21 - -0.25
        truediv_11 = sub_11 / 0.25;  sub_11 = None
        _tensor_constant21 = self._tensor_constant21
        maximum_11 = torch.maximum(truediv_11, _tensor_constant21);  truediv_11 = _tensor_constant21 = None
        sub_12 = einsum_21 - 0.0
        neg_6 = -sub_12;  sub_12 = None
        truediv_12 = neg_6 / 0.25;  neg_6 = None
        add_15 = truediv_12 + 1;  truediv_12 = None
        _tensor_constant22 = self._tensor_constant22
        maximum_12 = torch.maximum(add_15, _tensor_constant22);  add_15 = _tensor_constant22 = None
        minimum_7 = torch.minimum(maximum_11, maximum_12);  maximum_11 = maximum_12 = None
        _tensor_constant23 = self._tensor_constant23
        slicing_7 = nnodely_layers_fuzzify_slicing(repeat_1, _tensor_constant23, minimum_7);  _tensor_constant23 = minimum_7 = slicing_7 = None
        sub_13 = einsum_21 - 0.0
        truediv_13 = sub_13 / 0.25;  sub_13 = None
        _tensor_constant24 = self._tensor_constant24
        maximum_13 = torch.maximum(truediv_13, _tensor_constant24);  truediv_13 = _tensor_constant24 = None
        sub_14 = einsum_21 - 0.25
        neg_7 = -sub_14;  sub_14 = None
        truediv_14 = neg_7 / 0.25;  neg_7 = None
        add_16 = truediv_14 + 1;  truediv_14 = None
        _tensor_constant25 = self._tensor_constant25
        maximum_14 = torch.maximum(add_16, _tensor_constant25);  add_16 = _tensor_constant25 = None
        minimum_8 = torch.minimum(maximum_13, maximum_14);  maximum_13 = maximum_14 = None
        _tensor_constant26 = self._tensor_constant26
        slicing_8 = nnodely_layers_fuzzify_slicing(repeat_1, _tensor_constant26, minimum_8);  _tensor_constant26 = minimum_8 = slicing_8 = None
        sub_15 = einsum_21 - 0.25;  einsum_21 = None
        truediv_15 = sub_15 / 0.25;  sub_15 = None
        _tensor_constant27 = self._tensor_constant27
        maximum_15 = torch.maximum(truediv_15, _tensor_constant27);  truediv_15 = _tensor_constant27 = None
        _tensor_constant28 = self._tensor_constant28
        minimum_9 = torch.minimum(maximum_15, _tensor_constant28);  maximum_15 = _tensor_constant28 = None
        _tensor_constant29 = self._tensor_constant29
        slicing_9 = nnodely_layers_fuzzify_slicing(repeat_1, _tensor_constant29, minimum_9);  _tensor_constant29 = minimum_9 = slicing_9 = None
        relation_forward_select71_w = self.all_constants.Select71
        einsum_22 = torch.functional.einsum('ijk,k->ij', repeat_1, relation_forward_select71_w);  relation_forward_select71_w = None
        unsqueeze_5 = einsum_22.unsqueeze(2);  einsum_22 = None
        relation_forward_linear70_weights = self.all_parameters.Lin_pend1_4
        einsum_23 = torch.functional.einsum('bwi,io->bwo', cat_7, relation_forward_linear70_weights);  relation_forward_linear70_weights = None
        relation_forward_linear70_bias = self.all_parameters.PLinear43b
        add_17 = einsum_23 + relation_forward_linear70_bias;  einsum_23 = relation_forward_linear70_bias = None
        mul_5 = add_17 * unsqueeze_5;  add_17 = unsqueeze_5 = None
        relation_forward_select68_w = self.all_constants.Select68
        einsum_24 = torch.functional.einsum('ijk,k->ij', repeat_1, relation_forward_select68_w);  relation_forward_select68_w = None
        unsqueeze_6 = einsum_24.unsqueeze(2);  einsum_24 = None
        relation_forward_linear67_weights = self.all_parameters.Lin_pend1_3
        einsum_25 = torch.functional.einsum('bwi,io->bwo', cat_7, relation_forward_linear67_weights);  relation_forward_linear67_weights = None
        relation_forward_linear67_bias = self.all_parameters.PLinear40b
        add_18 = einsum_25 + relation_forward_linear67_bias;  einsum_25 = relation_forward_linear67_bias = None
        mul_6 = add_18 * unsqueeze_6;  add_18 = unsqueeze_6 = None
        relation_forward_select65_w = self.all_constants.Select65
        einsum_26 = torch.functional.einsum('ijk,k->ij', repeat_1, relation_forward_select65_w);  relation_forward_select65_w = None
        unsqueeze_7 = einsum_26.unsqueeze(2);  einsum_26 = None
        relation_forward_linear64_weights = self.all_parameters.Lin_pend1_2
        einsum_27 = torch.functional.einsum('bwi,io->bwo', cat_7, relation_forward_linear64_weights);  relation_forward_linear64_weights = None
        relation_forward_linear64_bias = self.all_parameters.PLinear37b
        add_19 = einsum_27 + relation_forward_linear64_bias;  einsum_27 = relation_forward_linear64_bias = None
        mul_7 = add_19 * unsqueeze_7;  add_19 = unsqueeze_7 = None
        relation_forward_select62_w = self.all_constants.Select62
        einsum_28 = torch.functional.einsum('ijk,k->ij', repeat_1, relation_forward_select62_w);  relation_forward_select62_w = None
        unsqueeze_8 = einsum_28.unsqueeze(2);  einsum_28 = None
        relation_forward_linear61_weights = self.all_parameters.Lin_pend1_1
        einsum_29 = torch.functional.einsum('bwi,io->bwo', cat_7, relation_forward_linear61_weights);  relation_forward_linear61_weights = None
        relation_forward_linear61_bias = self.all_parameters.PLinear34b
        add_20 = einsum_29 + relation_forward_linear61_bias;  einsum_29 = relation_forward_linear61_bias = None
        mul_8 = add_20 * unsqueeze_8;  add_20 = unsqueeze_8 = None
        relation_forward_select59_w = self.all_constants.Select59
        einsum_30 = torch.functional.einsum('ijk,k->ij', repeat_1, relation_forward_select59_w);  repeat_1 = relation_forward_select59_w = None
        unsqueeze_9 = einsum_30.unsqueeze(2);  einsum_30 = None
        relation_forward_linear58_weights = self.all_parameters.Lin_pend1_0
        einsum_31 = torch.functional.einsum('bwi,io->bwo', cat_7, relation_forward_linear58_weights);  relation_forward_linear58_weights = None
        relation_forward_linear58_bias = self.all_parameters.PLinear31b
        add_21 = einsum_31 + relation_forward_linear58_bias;  einsum_31 = relation_forward_linear58_bias = None
        mul_9 = add_21 * unsqueeze_9;  add_21 = unsqueeze_9 = None
        add_22 = mul_9 + mul_8;  mul_9 = mul_8 = None
        add_23 = add_22 + mul_7;  add_22 = mul_7 = None
        add_24 = add_23 + mul_6;  add_23 = mul_6 = None
        add_25 = add_24 + mul_5;  add_24 = mul_5 = None
        getitem_12 = kwargs['Xddth1']
        relation_forward_sample_part141_w = self.all_constants.SamplePart141
        einsum_32 = torch.functional.einsum('bij,ki->bkj', getitem_12, relation_forward_sample_part141_w);  getitem_12 = relation_forward_sample_part141_w = None
        getitem_13 = kwargs['Xvelocity']
        relation_forward_sample_part1_w = self.all_constants.SamplePart1
        einsum_33 = torch.functional.einsum('bij,ki->bkj', getitem_13, relation_forward_sample_part1_w);  getitem_13 = relation_forward_sample_part1_w = None
        zeros_like_2 = torch.zeros_like(einsum_33)
        repeat_2 = zeros_like_2.repeat(1, 1, 5);  zeros_like_2 = None
        sub_16 = einsum_33 - -0.35
        neg_8 = -sub_16;  sub_16 = None
        truediv_16 = neg_8 / 0.175;  neg_8 = None
        add_26 = truediv_16 + 1;  truediv_16 = None
        _tensor_constant30 = self._tensor_constant30
        maximum_16 = torch.maximum(add_26, _tensor_constant30);  add_26 = _tensor_constant30 = None
        _tensor_constant31 = self._tensor_constant31
        minimum_10 = torch.minimum(maximum_16, _tensor_constant31);  maximum_16 = _tensor_constant31 = None
        _tensor_constant32 = self._tensor_constant32
        slicing_10 = nnodely_layers_fuzzify_slicing(repeat_2, _tensor_constant32, minimum_10);  _tensor_constant32 = minimum_10 = slicing_10 = None
        sub_17 = einsum_33 - -0.35
        truediv_17 = sub_17 / 0.175;  sub_17 = None
        _tensor_constant33 = self._tensor_constant33
        maximum_17 = torch.maximum(truediv_17, _tensor_constant33);  truediv_17 = _tensor_constant33 = None
        sub_18 = einsum_33 - -0.175
        neg_9 = -sub_18;  sub_18 = None
        truediv_18 = neg_9 / 0.175;  neg_9 = None
        add_27 = truediv_18 + 1;  truediv_18 = None
        _tensor_constant34 = self._tensor_constant34
        maximum_18 = torch.maximum(add_27, _tensor_constant34);  add_27 = _tensor_constant34 = None
        minimum_11 = torch.minimum(maximum_17, maximum_18);  maximum_17 = maximum_18 = None
        _tensor_constant35 = self._tensor_constant35
        slicing_11 = nnodely_layers_fuzzify_slicing(repeat_2, _tensor_constant35, minimum_11);  _tensor_constant35 = minimum_11 = slicing_11 = None
        sub_19 = einsum_33 - -0.175
        truediv_19 = sub_19 / 0.175;  sub_19 = None
        _tensor_constant36 = self._tensor_constant36
        maximum_19 = torch.maximum(truediv_19, _tensor_constant36);  truediv_19 = _tensor_constant36 = None
        sub_20 = einsum_33 - 0.0
        neg_10 = -sub_20;  sub_20 = None
        truediv_20 = neg_10 / 0.17499999999999993;  neg_10 = None
        add_28 = truediv_20 + 1;  truediv_20 = None
        _tensor_constant37 = self._tensor_constant37
        maximum_20 = torch.maximum(add_28, _tensor_constant37);  add_28 = _tensor_constant37 = None
        minimum_12 = torch.minimum(maximum_19, maximum_20);  maximum_19 = maximum_20 = None
        _tensor_constant38 = self._tensor_constant38
        slicing_12 = nnodely_layers_fuzzify_slicing(repeat_2, _tensor_constant38, minimum_12);  _tensor_constant38 = minimum_12 = slicing_12 = None
        sub_21 = einsum_33 - 0.0
        truediv_21 = sub_21 / 0.17499999999999993;  sub_21 = None
        _tensor_constant39 = self._tensor_constant39
        maximum_21 = torch.maximum(truediv_21, _tensor_constant39);  truediv_21 = _tensor_constant39 = None
        sub_22 = einsum_33 - 0.17499999999999993
        neg_11 = -sub_22;  sub_22 = None
        truediv_22 = neg_11 / 0.17500000000000004;  neg_11 = None
        add_29 = truediv_22 + 1;  truediv_22 = None
        _tensor_constant40 = self._tensor_constant40
        maximum_22 = torch.maximum(add_29, _tensor_constant40);  add_29 = _tensor_constant40 = None
        minimum_13 = torch.minimum(maximum_21, maximum_22);  maximum_21 = maximum_22 = None
        _tensor_constant41 = self._tensor_constant41
        slicing_13 = nnodely_layers_fuzzify_slicing(repeat_2, _tensor_constant41, minimum_13);  _tensor_constant41 = minimum_13 = slicing_13 = None
        sub_23 = einsum_33 - 0.17499999999999993;  einsum_33 = None
        truediv_23 = sub_23 / 0.17500000000000004;  sub_23 = None
        _tensor_constant42 = self._tensor_constant42
        maximum_23 = torch.maximum(truediv_23, _tensor_constant42);  truediv_23 = _tensor_constant42 = None
        _tensor_constant43 = self._tensor_constant43
        minimum_14 = torch.minimum(maximum_23, _tensor_constant43);  maximum_23 = _tensor_constant43 = None
        _tensor_constant44 = self._tensor_constant44
        slicing_14 = nnodely_layers_fuzzify_slicing(repeat_2, _tensor_constant44, minimum_14);  _tensor_constant44 = minimum_14 = slicing_14 = None
        relation_forward_select52_w = self.all_constants.Select52
        einsum_34 = torch.functional.einsum('ijk,k->ij', repeat_2, relation_forward_select52_w);  relation_forward_select52_w = None
        unsqueeze_10 = einsum_34.unsqueeze(2);  einsum_34 = None
        relation_forward_linear51_weights = self.all_parameters.Lin_cart_4
        einsum_35 = torch.functional.einsum('bwi,io->bwo', cat_7, relation_forward_linear51_weights);  relation_forward_linear51_weights = None
        relation_forward_linear51_bias = self.all_parameters.PLinear27b
        add_30 = einsum_35 + relation_forward_linear51_bias;  einsum_35 = relation_forward_linear51_bias = None
        mul_10 = add_30 * unsqueeze_10;  add_30 = unsqueeze_10 = None
        relation_forward_select49_w = self.all_constants.Select49
        einsum_36 = torch.functional.einsum('ijk,k->ij', repeat_2, relation_forward_select49_w);  relation_forward_select49_w = None
        unsqueeze_11 = einsum_36.unsqueeze(2);  einsum_36 = None
        relation_forward_linear48_weights = self.all_parameters.Lin_cart_3
        einsum_37 = torch.functional.einsum('bwi,io->bwo', cat_7, relation_forward_linear48_weights);  relation_forward_linear48_weights = None
        relation_forward_linear48_bias = self.all_parameters.PLinear24b
        add_31 = einsum_37 + relation_forward_linear48_bias;  einsum_37 = relation_forward_linear48_bias = None
        mul_11 = add_31 * unsqueeze_11;  add_31 = unsqueeze_11 = None
        relation_forward_select46_w = self.all_constants.Select46
        einsum_38 = torch.functional.einsum('ijk,k->ij', repeat_2, relation_forward_select46_w);  relation_forward_select46_w = None
        unsqueeze_12 = einsum_38.unsqueeze(2);  einsum_38 = None
        relation_forward_linear45_weights = self.all_parameters.Lin_cart_2
        einsum_39 = torch.functional.einsum('bwi,io->bwo', cat_7, relation_forward_linear45_weights);  relation_forward_linear45_weights = None
        relation_forward_linear45_bias = self.all_parameters.PLinear21b
        add_32 = einsum_39 + relation_forward_linear45_bias;  einsum_39 = relation_forward_linear45_bias = None
        mul_12 = add_32 * unsqueeze_12;  add_32 = unsqueeze_12 = None
        relation_forward_select43_w = self.all_constants.Select43
        einsum_40 = torch.functional.einsum('ijk,k->ij', repeat_2, relation_forward_select43_w);  relation_forward_select43_w = None
        unsqueeze_13 = einsum_40.unsqueeze(2);  einsum_40 = None
        relation_forward_linear42_weights = self.all_parameters.Lin_cart_1
        einsum_41 = torch.functional.einsum('bwi,io->bwo', cat_7, relation_forward_linear42_weights);  relation_forward_linear42_weights = None
        relation_forward_linear42_bias = self.all_parameters.PLinear18b
        add_33 = einsum_41 + relation_forward_linear42_bias;  einsum_41 = relation_forward_linear42_bias = None
        mul_13 = add_33 * unsqueeze_13;  add_33 = unsqueeze_13 = None
        relation_forward_select40_w = self.all_constants.Select40
        einsum_42 = torch.functional.einsum('ijk,k->ij', repeat_2, relation_forward_select40_w);  repeat_2 = relation_forward_select40_w = None
        unsqueeze_14 = einsum_42.unsqueeze(2);  einsum_42 = None
        relation_forward_linear39_weights = self.all_parameters.Lin_cart_0
        einsum_43 = torch.functional.einsum('bwi,io->bwo', cat_7, relation_forward_linear39_weights);  cat_7 = relation_forward_linear39_weights = None
        relation_forward_linear39_bias = self.all_parameters.PLinear15b
        add_34 = einsum_43 + relation_forward_linear39_bias;  einsum_43 = relation_forward_linear39_bias = None
        mul_14 = add_34 * unsqueeze_14;  add_34 = unsqueeze_14 = None
        add_35 = mul_14 + mul_13;  mul_14 = mul_13 = None
        add_36 = add_35 + mul_12;  add_35 = mul_12 = None
        add_37 = add_36 + mul_11;  add_36 = mul_11 = None
        add_38 = add_37 + mul_10;  add_37 = mul_10 = None
        getitem_14 = kwargs['Xddx']
        relation_forward_sample_part139_w = self.all_constants.SamplePart139
        einsum_44 = torch.functional.einsum('bij,ki->bkj', getitem_14, relation_forward_sample_part139_w);  getitem_14 = relation_forward_sample_part139_w = None
        all_constants_sample_time = self.all_constants.SampleTime
        mul_15 = add_12 * all_constants_sample_time
        getitem_15 = kwargs['int_th2_dot']
        relation_forward_sample_part109_w = self.all_constants.SamplePart109
        einsum_45 = torch.functional.einsum('bij,ki->bkj', getitem_15, relation_forward_sample_part109_w);  getitem_15 = relation_forward_sample_part109_w = None
        add_39 = einsum_45 + mul_15;  einsum_45 = mul_15 = None
        mul_16 = add_39 * all_constants_sample_time
        getitem_16 = kwargs['int_th2']
        relation_forward_sample_part127_w = self.all_constants.SamplePart127
        einsum_46 = torch.functional.einsum('bij,ki->bkj', getitem_16, relation_forward_sample_part127_w);  getitem_16 = relation_forward_sample_part127_w = None
        add_40 = einsum_46 + mul_16;  einsum_46 = mul_16 = None
        mul_17 = add_25 * all_constants_sample_time
        getitem_17 = kwargs['int_th1_dot']
        relation_forward_sample_part103_w = self.all_constants.SamplePart103
        einsum_47 = torch.functional.einsum('bij,ki->bkj', getitem_17, relation_forward_sample_part103_w);  getitem_17 = relation_forward_sample_part103_w = None
        add_41 = einsum_47 + mul_17;  einsum_47 = mul_17 = None
        mul_18 = add_41 * all_constants_sample_time
        getitem_18 = kwargs['int_th1']
        relation_forward_sample_part121_w = self.all_constants.SamplePart121
        einsum_48 = torch.functional.einsum('bij,ki->bkj', getitem_18, relation_forward_sample_part121_w);  getitem_18 = relation_forward_sample_part121_w = None
        add_42 = einsum_48 + mul_18;  einsum_48 = mul_18 = None
        mul_19 = add_38 * all_constants_sample_time
        getitem_19 = kwargs['int_xdot']
        relation_forward_sample_part97_w = self.all_constants.SamplePart97
        einsum_49 = torch.functional.einsum('bij,ki->bkj', getitem_19, relation_forward_sample_part97_w);  getitem_19 = relation_forward_sample_part97_w = None
        add_43 = einsum_49 + mul_19;  einsum_49 = mul_19 = None
        mul_20 = add_43 * all_constants_sample_time;  all_constants_sample_time = None
        getitem_20 = kwargs['int_x'];  kwargs = None
        relation_forward_sample_part115_w = self.all_constants.SamplePart115
        einsum_50 = torch.functional.einsum('bij,ki->bkj', getitem_20, relation_forward_sample_part115_w);  getitem_20 = relation_forward_sample_part115_w = None
        add_44 = einsum_50 + mul_20;  einsum_50 = mul_20 = None
        return ({'th2_ddot_est': add_12, 'th1_ddot_est': add_25, 'acc_cart_est': add_38, 'th2_est': add_40, 'omega2_est': add_39, 'th1_est': add_42, 'omega1_est': add_41, 'x_est': add_44, 'xdot_est': add_43}, {'SamplePart139': einsum_44, 'SamplePart141': einsum_32, 'SamplePart143': einsum_20, 'Add57': add_38, 'Add76': add_25, 'Add95': add_12}, {'Xth2_dot': add_39, 'Xth1_dot': add_41, 'Xth2': add_40, 'Xth1': add_42, 'Xpos': add_44, 'Xvelocity': add_43, 'int_th2': add_40, 'int_th2_dot': add_39, 'int_th1': add_42, 'int_th1_dot': add_41, 'int_x': add_44, 'int_xdot': add_43}, {})
        
class RecurrentModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.Cell = TracerModel()
        self.inputs = ['action', 'Xddth2', 'Xddth1', 'Xddx', ]
        self.states = dict()

    def forward(self, kwargs, n_samples = None):
        n_samples = n_samples if n_samples else min([kwargs[key].size(0) for key in self.inputs])
        self.states['Xth2_dot'] = kwargs['Xth2_dot']
        self.states['Xth1_dot'] = kwargs['Xth1_dot']
        self.states['Xth2'] = kwargs['Xth2']
        self.states['Xth1'] = kwargs['Xth1']
        self.states['Xpos'] = kwargs['Xpos']
        self.states['Xvelocity'] = kwargs['Xvelocity']
        self.states['int_th2'] = kwargs['int_th2']
        self.states['int_th2_dot'] = kwargs['int_th2_dot']
        self.states['int_th1'] = kwargs['int_th1']
        self.states['int_th1_dot'] = kwargs['int_th1_dot']
        self.states['int_x'] = kwargs['int_x']
        self.states['int_xdot'] = kwargs['int_xdot']
        results = {'th2_ddot_est':[], 'th1_ddot_est':[], 'acc_cart_est':[], 'th2_est':[], 'omega2_est':[], 'th1_est':[], 'omega1_est':[], 'x_est':[], 'xdot_est':[], }
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
