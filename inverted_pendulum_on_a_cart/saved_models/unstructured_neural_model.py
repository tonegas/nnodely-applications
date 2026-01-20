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
        self.all_constants["sigma_omega"] = torch.tensor([0.0], requires_grad=False)
        self.all_constants["sigma_theta"] = torch.tensor([0.0], requires_grad=False)
        self.all_constants["sigma_v"] = torch.tensor([0.0], requires_grad=False)
        self.all_constants["sigma_x"] = torch.tensor([0.0], requires_grad=False)
        self.all_parameters["PFir89W"] = torch.nn.Parameter(torch.tensor([[0.8416784405708313]]), requires_grad=True)
        self.all_parameters["PFir83W"] = torch.nn.Parameter(torch.tensor([[0.1227167621254921]]), requires_grad=True)
        self.all_parameters["PFir85W"] = torch.nn.Parameter(torch.tensor([[0.08001725375652313]]), requires_grad=True)
        self.all_parameters["PFir87W"] = torch.nn.Parameter(torch.tensor([[0.12916341423988342]]), requires_grad=True)
        self.all_parameters["PLinear91W"] = torch.nn.Parameter(torch.tensor([[0.10566435754299164, -0.08787209540605545, 0.3666980266571045, 0.19626909494400024, 1.0757936239242554, -0.13800734281539917, -0.06865554302930832, 0.6059689521789551, 0.007938618771731853, 0.04481298103928566]]), requires_grad=True)
        self.all_parameters["PLinear93W"] = torch.nn.Parameter(torch.tensor([[0.20302139222621918, 0.031443189829587936, -0.11687294393777847, 0.19198575615882874, 0.033028535544872284, -0.03829606622457504, 0.45180872082710266, 0.044927217066287994, 1.0330973863601685, -0.0068347561173141]]), requires_grad=True)
        self.all_parameters["PLinear95W"] = torch.nn.Parameter(torch.tensor([[-0.0828372910618782, 0.13086044788360596, 0.36676883697509766, -0.1677926480770111, 0.11068268120288849, 0.08424267172813416, -0.006758241914212704, 0.8284241557121277, -0.5778492093086243, 1.3914775848388672]]), requires_grad=True)
        self.all_parameters["PLinear97W"] = torch.nn.Parameter(torch.tensor([[0.40788063406944275, 0.526215136051178, 1.1058486700057983, 0.45316094160079956, -0.466362327337265, 0.15977849066257477, 0.2088443636894226, 0.2625473737716675, 0.026950068771839142, -0.07458783686161041]]), requires_grad=True)
        self.all_parameters["PLinear99W"] = torch.nn.Parameter(torch.tensor([[-0.3203251361846924, -0.2527702748775482, 0.12889790534973145, -0.15453289449214935, -0.2838740646839142, 0.47001487016677856, 0.45195460319519043, -0.018180465325713158, -0.02246868424117565, 0.0789511576294899]]), requires_grad=True)
        self.all_parameters["PLinear101W"] = torch.nn.Parameter(torch.tensor([[0.5711999535560608, -0.045563679188489914, 0.0070588914677500725, -0.45005714893341064, 0.04028822109103203, 0.07227244973182678, 0.01751856692135334, -0.08169938623905182, 0.011485951952636242, 0.003663351759314537]]), requires_grad=True)
        self.all_parameters["PLinear103W"] = torch.nn.Parameter(torch.tensor([[-0.02232731506228447, -0.018502919003367424, 0.16818350553512573, -0.04413715749979019, -0.09407293051481247, 0.07266207784414291, -0.10003166645765305, 0.04317149892449379, 0.004465627484023571, 0.009477022103965282]]), requires_grad=True)
        self.all_parameters["PLinear105W"] = torch.nn.Parameter(torch.tensor([[0.1686074137687683, -0.5748383402824402, -0.10994458198547363, 0.11033552885055542, -0.16182006895542145, 0.227553129196167, 0.04999234527349472, -0.38133499026298523, -0.19922159612178802, -0.16112546622753143], [0.5962351560592651, 0.4346374273300171, 0.3109723627567291, 0.7136629223823547, 0.976896345615387, 0.15204083919525146, -0.22465839982032776, -0.3019057512283325, 0.7414509654045105, -0.14246289432048798], [-0.08892549574375153, 0.27332839369773865, -0.5205637216567993, 0.5571541786193848, -0.025607705116271973, -0.3311350345611572, 0.39572322368621826, 0.02575763687491417, 0.07086458802223206, 0.3743473291397095], [0.17964106798171997, -0.11032609641551971, 0.17581912875175476, 0.004615219309926033, -0.3107468783855438, 0.31539157032966614, -0.2590113580226898, 0.47966939210891724, -0.3407376706600189, -0.09044050425291061], [-0.011972428299486637, -0.06419694423675537, -0.11421208828687668, -0.08474656194448471, 0.508674681186676, -0.35735011100769043, -0.1256050318479538, -0.3134627938270569, 0.3138655424118042, 0.21152396500110626], [0.15414512157440186, -0.2977851629257202, 0.2902003824710846, -0.17781108617782593, 0.9790985584259033, 0.3613426089286804, -0.17400698363780975, -0.016708848997950554, -0.07130908221006393, 0.30713891983032227], [-0.16398857533931732, -0.01604621857404709, -0.03379465267062187, 0.20763467252254486, 0.653001070022583, -0.2684141993522644, -0.03883110731840134, 0.020578021183609962, -0.2421327531337738, 0.23703402280807495], [0.025585345923900604, 0.19678975641727448, -0.03830381855368614, 0.21190641820430756, 0.37903982400894165, 0.23299729824066162, -0.3491519093513489, 0.11481929570436478, -0.15946511924266815, -0.4549933075904846], [-0.22504130005836487, 0.09129028767347336, 0.25172701478004456, -0.008388413116335869, -0.38941115140914917, -0.10471872985363007, -0.11615593731403351, 0.17827850580215454, 0.10405968129634857, 0.23577794432640076], [1.0970638990402222, -1.468384861946106, 1.5831080675125122, 0.5785375833511353, -1.1889196634292603, 1.3271502256393433, 0.9620514512062073, 1.6231260299682617, 1.0236533880233765, 0.7740903496742249]]), requires_grad=True)
        self.all_parameters["PLinear107W"] = torch.nn.Parameter(torch.tensor([[1.289291262626648], [-0.6081482172012329], [1.0318515300750732], [0.3900676965713501], [-0.36081480979919434], [0.8329543471336365], [1.1391496658325195], [0.6765429377555847], [1.211871862411499], [1.0841323137283325]]), requires_grad=True)
        self.all_parameters["PLinear109W"] = torch.nn.Parameter(torch.tensor([[0.3351389765739441, 1.1056753396987915, -0.03248303011059761, -0.12002471089363098, -0.1536858081817627, -0.3876519799232483, -0.5825278759002686, 0.5866972208023071, 0.3401072919368744, 0.10798407346010208], [0.04001225531101227, -0.913703441619873, 0.17073558270931244, -0.515258252620697, 0.6388874650001526, 0.08376860618591309, 0.5204358696937561, -0.3470103144645691, 0.8038610219955444, -0.5311955809593201], [0.22756195068359375, 0.11250582337379456, 0.10314925014972687, -0.13691508769989014, -0.5467584133148193, 0.2754257321357727, 0.2532079517841339, 0.06188181787729263, 0.021940365433692932, 0.09879394620656967], [0.34527236223220825, 1.1933231353759766, -0.21185185015201569, -0.05727069079875946, -0.07428944110870361, -0.20616239309310913, 0.24635003507137299, 0.05250917002558708, 0.08942347019910812, 0.42093148827552795], [-0.027291640639305115, 1.1143467426300049, 0.12080666422843933, -0.022379467263817787, -0.17785276472568512, -0.18779638409614563, 0.1537226438522339, -0.4085084795951843, 0.2930475175380707, 0.20863908529281616], [0.24639980494976044, -2.248633861541748, -0.1517050415277481, 0.12972071766853333, -0.39798304438591003, 0.6366106867790222, -0.46326667070388794, 0.216453418135643, -0.3891525864601135, -0.08900592476129532], [0.5859012007713318, -0.49442166090011597, -0.4215604066848755, 0.14421847462654114, 0.05972931534051895, 0.2641385793685913, 0.30787020921707153, 0.27689439058303833, 0.08764326572418213, 0.63480144739151], [-0.18515080213546753, 2.015047311782837, -0.048804499208927155, 0.005417323671281338, -0.15937532484531403, -0.1283196061849594, -0.4266829788684845, -0.031120525673031807, -0.24988284707069397, 0.6054531931877136], [-1.043843150138855, 6.453720569610596, 0.9726628661155701, 0.8350223898887634, 2.6492459774017334, 1.0732955932617188, 1.312579870223999, 0.7272279262542725, 0.8502452969551086, 2.547415256500244], [-0.6930896639823914, 1.2759833335876465, 0.2193545401096344, 0.5448135733604431, 1.0260270833969116, 0.7237148284912109, 0.5042695999145508, 0.4827488958835602, 0.16572540998458862, 0.8412517309188843]]), requires_grad=True)
        self.all_parameters["PLinear111W"] = torch.nn.Parameter(torch.tensor([[-0.4042825996875763], [-0.459232896566391], [1.151382565498352], [1.6244195699691772], [0.6452844738960266], [0.5750159025192261], [1.0456578731536865], [1.200089454650879], [0.8044319748878479], [1.0572324991226196]]), requires_grad=True)
        self.all_parameters["PLinear113W"] = torch.nn.Parameter(torch.tensor([[-0.7650732398033142, -0.26037874817848206, 0.4648684561252594, 1.5334469079971313, -0.0546724833548069, -0.6928558349609375, -0.0011288216337561607, 1.0192621946334839, 0.23635625839233398, 0.7644299864768982], [1.0631482601165771, 1.2667332887649536, 0.9612788558006287, 0.814130425453186, 2.3413548469543457, -0.7621850371360779, 1.650965690612793, 1.1404234170913696, 0.5812704563140869, 1.171276330947876], [0.29685425758361816, 0.030664799734950066, 0.6228449940681458, 0.2918458878993988, -1.1106655597686768, -0.500067949295044, -0.37736034393310547, -0.1825782209634781, 0.12937846779823303, -0.7462899088859558], [0.8865786194801331, -0.037244100123643875, 0.6473543047904968, -0.30303704738616943, 0.18463635444641113, -0.2667180001735687, 0.5450273156166077, 0.2040368914604187, 0.5643481016159058, 0.5549779534339905], [-0.08950351923704147, 0.8444698452949524, -0.19410893321037292, 0.6745433211326599, 0.36393114924430847, -0.5759389996528625, 0.2518478333950043, 0.25063204765319824, 0.6787486672401428, 2.0723156929016113], [1.1097335815429688, 0.7595548033714294, 0.5923506021499634, 1.3672914505004883, 1.7922652959823608, -0.7170969843864441, 0.6676839590072632, 0.9820489883422852, 1.0487865209579468, -1.284509301185608], [0.4488784372806549, 0.08261600881814957, -1.1353973150253296, 0.8491522669792175, 0.18744567036628723, -0.4206537902355194, -0.8906650543212891, -0.7247441411018372, 0.8173158168792725, -1.2375212907791138], [0.19846832752227783, -0.04492487013339996, 0.07355108857154846, -0.3221363425254822, -0.40117505192756653, -0.20151910185813904, -0.00448019290342927, -0.3074030578136444, 0.1367269903421402, -0.08969764411449432], [-0.18757809698581696, -0.12560072541236877, 0.4516364336013794, -0.4869301915168762, -0.2510451078414917, 0.10600335896015167, 0.3556767702102661, 0.07983781397342682, -0.44160908460617065, -0.31744685769081116], [-0.1202675923705101, 0.23176443576812744, -0.03772096708416939, -0.13836169242858887, -0.06637287884950638, 0.5871118903160095, -0.2317105531692505, -0.32776814699172974, -0.2188900262117386, 0.18950575590133667]]), requires_grad=True)
        self.all_parameters["PLinear115W"] = torch.nn.Parameter(torch.tensor([[1.2629163265228271], [1.9945895671844482], [1.258370041847229], [0.9414674639701843], [1.7888829708099365], [-1.1484260559082031], [2.0045464038848877], [1.3389488458633423], [1.5577892065048218], [0.6789577603340149]]), requires_grad=True)
        self.all_parameters["PLinear117W"] = torch.nn.Parameter(torch.tensor([[0.5353698134422302, -0.6699849367141724, 1.4482778310775757, 0.23883061110973358, -0.4098929166793823, -0.6556970477104187, 1.087274432182312, -0.8039222955703735, 0.7158417701721191, 0.843367338180542], [-0.4941578805446625, 0.4798400104045868, -0.7028713822364807, 0.49517112970352173, 0.18531294167041779, 0.25575563311576843, -0.791546106338501, 0.16592977941036224, -0.3574520945549011, -1.2504740953445435], [0.13996638357639313, -0.18242168426513672, 0.2192315012216568, 0.09036879241466522, 0.025068974122405052, -0.26629066467285156, 0.6430698037147522, 0.2215600311756134, -0.599562406539917, 0.09944914281368256], [0.16770675778388977, -0.9929742813110352, -0.030813854187726974, -0.031629279255867004, -0.6977342963218689, 0.8559200763702393, -0.3652174770832062, 0.6462593674659729, -0.09182970225811005, 0.852665901184082], [0.3505862355232239, -0.9160879254341125, 0.33387261629104614, -0.045782171189785004, -1.2187329530715942, -0.1992378830909729, 0.09285014867782593, 0.35810601711273193, 0.017764965072274208, 0.362834632396698], [0.04318251088261604, 0.590979278087616, 0.1814134269952774, -0.5048409104347229, 1.5882006883621216, -0.48319846391677856, 0.3912985324859619, 0.10828999429941177, -0.05104036629199982, 0.19858317077159882], [0.18499071896076202, 0.9059181809425354, -0.24012842774391174, -0.391086608171463, 2.318652629852295, -0.47572192549705505, 0.21576496958732605, -0.4342440068721771, -0.22387681901454926, 0.7231854796409607], [0.6019194722175598, -1.186425805091858, 0.7748550176620483, 0.2499426156282425, -0.6633814573287964, 0.036008745431900024, -0.018978489562869072, 0.5017654299736023, 0.48888230323791504, 0.5936568379402161], [0.08361201733350754, -0.032254669815301895, 0.045960504561662674, 0.2479972243309021, -0.2994464039802551, 0.12718236446380615, -0.6430074572563171, -0.03194911405444145, -0.06391645967960358, -0.36420750617980957], [-0.5624468922615051, 1.8317126035690308, -0.37880173325538635, 0.013357431627810001, -0.02422226034104824, 0.23719412088394165, 0.34010374546051025, 0.0058952998369932175, 0.14571832120418549, -0.5180050134658813]]), requires_grad=True)
        self.all_parameters["PLinear119W"] = torch.nn.Parameter(torch.tensor([[0.5836499333381653], [-0.4410580098628998], [0.5740870833396912], [1.1592715978622437], [-0.5013313889503479], [0.8534936308860779], [0.40318939089775085], [0.6437219977378845], [0.7972379922866821], [0.3962278366088867]]), requires_grad=True)
        self.all_constants["SamplePart113"] = torch.tensor([[1.0]], requires_grad=True)
        self.all_constants["SamplePart135"] = torch.tensor([[1.0]], requires_grad=True)
        self.all_constants["SamplePart137"] = torch.tensor([[1.0]], requires_grad=True)
        self.all_constants["SamplePart139"] = torch.tensor([[1.0]], requires_grad=True)
        self.all_constants["SamplePart141"] = torch.tensor([[1.0]], requires_grad=True)
        self.all_constants["SamplePart73"] = torch.tensor([[1.0]], requires_grad=True)
        self.all_constants["SamplePart75"] = torch.tensor([[1.0]], requires_grad=True)
        self.all_constants["SamplePart81"] = torch.tensor([[1.0]], requires_grad=True)
        self.all_constants["SamplePart83"] = torch.tensor([[1.0]], requires_grad=True)
        self.all_constants["SamplePart89"] = torch.tensor([[1.0]], requires_grad=True)
        self.all_constants["SamplePart91"] = torch.tensor([[1.0]], requires_grad=True)
        self.all_constants["SamplePart97"] = torch.tensor([[1.0]], requires_grad=True)
        self.all_constants["SamplePart99"] = torch.tensor([[1.0]], requires_grad=True)
        self.all_parameters = torch.nn.ParameterDict(self.all_parameters)
        self.all_constants = torch.nn.ParameterDict(self.all_constants)

    def update(self, closed_loop={}, connect={}, disconnect=False):
        pass
    
    def forward(self, kwargs):
        getitem = kwargs['action']
        relation_forward_sample_part113_w = self.all_constants.SamplePart113
        einsum = torch.functional.einsum('bij,ki->bkj', getitem, relation_forward_sample_part113_w);  getitem = relation_forward_sample_part113_w = None
        getitem_1 = kwargs['Yangular_velocity']
        relation_forward_sample_part135_w = self.all_constants.SamplePart135
        einsum_1 = torch.functional.einsum('bij,ki->bkj', getitem_1, relation_forward_sample_part135_w);  getitem_1 = relation_forward_sample_part135_w = None
        getitem_2 = kwargs['Yangle']
        relation_forward_sample_part137_w = self.all_constants.SamplePart137
        einsum_2 = torch.functional.einsum('bij,ki->bkj', getitem_2, relation_forward_sample_part137_w);  getitem_2 = relation_forward_sample_part137_w = None
        getitem_3 = kwargs['Yvelocity']
        relation_forward_sample_part139_w = self.all_constants.SamplePart139
        einsum_3 = torch.functional.einsum('bij,ki->bkj', getitem_3, relation_forward_sample_part139_w);  getitem_3 = relation_forward_sample_part139_w = None
        getitem_4 = kwargs['Ypos']
        relation_forward_sample_part141_w = self.all_constants.SamplePart141
        einsum_4 = torch.functional.einsum('bij,ki->bkj', getitem_4, relation_forward_sample_part141_w);  getitem_4 = relation_forward_sample_part141_w = None
        getitem_5 = kwargs['Xpos']
        relation_forward_sample_part73_w = self.all_constants.SamplePart73
        einsum_5 = torch.functional.einsum('bij,ki->bkj', getitem_5, relation_forward_sample_part73_w);  getitem_5 = relation_forward_sample_part73_w = None
        getitem_6 = kwargs['noise2']
        relation_forward_sample_part75_w = self.all_constants.SamplePart75
        einsum_6 = torch.functional.einsum('bij,ki->bkj', getitem_6, relation_forward_sample_part75_w);  getitem_6 = relation_forward_sample_part75_w = None
        getitem_7 = kwargs['Xvelocity']
        relation_forward_sample_part81_w = self.all_constants.SamplePart81
        einsum_7 = torch.functional.einsum('bij,ki->bkj', getitem_7, relation_forward_sample_part81_w);  getitem_7 = relation_forward_sample_part81_w = None
        getitem_8 = kwargs['noise1']
        relation_forward_sample_part83_w = self.all_constants.SamplePart83
        einsum_8 = torch.functional.einsum('bij,ki->bkj', getitem_8, relation_forward_sample_part83_w);  getitem_8 = relation_forward_sample_part83_w = None
        getitem_9 = kwargs['Xangular_velocity']
        relation_forward_sample_part89_w = self.all_constants.SamplePart89
        einsum_9 = torch.functional.einsum('bij,ki->bkj', getitem_9, relation_forward_sample_part89_w);  getitem_9 = relation_forward_sample_part89_w = None
        getitem_10 = kwargs['noise3']
        relation_forward_sample_part91_w = self.all_constants.SamplePart91
        einsum_10 = torch.functional.einsum('bij,ki->bkj', getitem_10, relation_forward_sample_part91_w);  getitem_10 = relation_forward_sample_part91_w = None
        getitem_11 = kwargs['Xangle']
        relation_forward_sample_part97_w = self.all_constants.SamplePart97
        einsum_11 = torch.functional.einsum('bij,ki->bkj', getitem_11, relation_forward_sample_part97_w);  getitem_11 = relation_forward_sample_part97_w = None
        getitem_12 = kwargs['noise4'];  kwargs = None
        relation_forward_sample_part99_w = self.all_constants.SamplePart99
        einsum_12 = torch.functional.einsum('bij,ki->bkj', getitem_12, relation_forward_sample_part99_w);  getitem_12 = relation_forward_sample_part99_w = None
        relation_forward_linear114_weights = self.all_parameters.PLinear103W
        einsum_13 = torch.functional.einsum('bwi,io->bwo', einsum, relation_forward_linear114_weights);  einsum = relation_forward_linear114_weights = None
        all_constants_sigma_theta = self.all_constants.sigma_theta
        mul = einsum_12 * all_constants_sigma_theta;  einsum_12 = all_constants_sigma_theta = None
        all_constants_sigma_x = self.all_constants.sigma_x
        mul_1 = einsum_6 * all_constants_sigma_x;  einsum_6 = all_constants_sigma_x = None
        all_constants_sigma_v = self.all_constants.sigma_v
        mul_2 = einsum_8 * all_constants_sigma_v;  einsum_8 = all_constants_sigma_v = None
        all_constants_sigma_omega = self.all_constants.sigma_omega
        mul_3 = einsum_10 * all_constants_sigma_omega;  einsum_10 = all_constants_sigma_omega = None
        add = einsum_11 + mul;  einsum_11 = mul = None
        add_1 = einsum_5 + mul_1;  einsum_5 = mul_1 = None
        add_2 = einsum_7 + mul_2;  einsum_7 = mul_2 = None
        add_3 = einsum_9 + mul_3;  einsum_9 = mul_3 = None
        size = add.size(0)
        relation_forward_fir103_weights = self.all_parameters.PFir89W
        size_1 = relation_forward_fir103_weights.size(1)
        squeeze = add.squeeze(-1);  add = None
        matmul = torch.matmul(squeeze, relation_forward_fir103_weights);  squeeze = relation_forward_fir103_weights = None
        to = matmul.to(dtype = torch.float32);  matmul = None
        view = to.view(size, 1, size_1);  to = size = size_1 = None
        size_2 = add_1.size(0)
        relation_forward_fir79_weights = self.all_parameters.PFir83W
        size_3 = relation_forward_fir79_weights.size(1)
        squeeze_1 = add_1.squeeze(-1);  add_1 = None
        matmul_1 = torch.matmul(squeeze_1, relation_forward_fir79_weights);  squeeze_1 = relation_forward_fir79_weights = None
        to_1 = matmul_1.to(dtype = torch.float32);  matmul_1 = None
        view_1 = to_1.view(size_2, 1, size_3);  to_1 = size_2 = size_3 = None
        size_4 = add_2.size(0)
        relation_forward_fir87_weights = self.all_parameters.PFir85W
        size_5 = relation_forward_fir87_weights.size(1)
        squeeze_2 = add_2.squeeze(-1);  add_2 = None
        matmul_2 = torch.matmul(squeeze_2, relation_forward_fir87_weights);  squeeze_2 = relation_forward_fir87_weights = None
        to_2 = matmul_2.to(dtype = torch.float32);  matmul_2 = None
        view_2 = to_2.view(size_4, 1, size_5);  to_2 = size_4 = size_5 = None
        size_6 = add_3.size(0)
        relation_forward_fir95_weights = self.all_parameters.PFir87W
        size_7 = relation_forward_fir95_weights.size(1)
        squeeze_3 = add_3.squeeze(-1);  add_3 = None
        matmul_3 = torch.matmul(squeeze_3, relation_forward_fir95_weights);  squeeze_3 = relation_forward_fir95_weights = None
        to_3 = matmul_3.to(dtype = torch.float32);  matmul_3 = None
        view_3 = to_3.view(size_6, 1, size_7);  to_3 = size_6 = size_7 = None
        relation_forward_linear104_weights = self.all_parameters.PLinear91W
        einsum_14 = torch.functional.einsum('bwi,io->bwo', view, relation_forward_linear104_weights);  relation_forward_linear104_weights = None
        relation_forward_linear105_weights = self.all_parameters.PLinear93W
        einsum_15 = torch.functional.einsum('bwi,io->bwo', view_1, relation_forward_linear105_weights);  view_1 = relation_forward_linear105_weights = None
        relation_forward_linear106_weights = self.all_parameters.PLinear95W
        einsum_16 = torch.functional.einsum('bwi,io->bwo', view_2, relation_forward_linear106_weights);  view_2 = relation_forward_linear106_weights = None
        relation_forward_linear107_weights = self.all_parameters.PLinear97W
        einsum_17 = torch.functional.einsum('bwi,io->bwo', view_3, relation_forward_linear107_weights);  view_3 = relation_forward_linear107_weights = None
        sin = torch.sin(view)
        add_4 = einsum_14 + einsum_15;  einsum_14 = einsum_15 = None
        add_5 = add_4 + einsum_16;  add_4 = einsum_16 = None
        add_6 = add_5 + einsum_17;  add_5 = einsum_17 = None
        cos = torch.cos(view);  view = None
        relation_forward_linear109_weights = self.all_parameters.PLinear99W
        einsum_18 = torch.functional.einsum('bwi,io->bwo', sin, relation_forward_linear109_weights);  sin = relation_forward_linear109_weights = None
        relation_forward_linear111_weights = self.all_parameters.PLinear101W
        einsum_19 = torch.functional.einsum('bwi,io->bwo', cos, relation_forward_linear111_weights);  cos = relation_forward_linear111_weights = None
        add_7 = add_6 + einsum_18;  add_6 = einsum_18 = None
        add_8 = add_7 + einsum_19;  add_7 = einsum_19 = None
        add_9 = add_8 + einsum_13;  add_8 = einsum_13 = None
        tanh = torch.tanh(add_9);  add_9 = None
        relation_forward_linear122_weights = self.all_parameters.PLinear105W
        einsum_20 = torch.functional.einsum('bwi,io->bwo', tanh, relation_forward_linear122_weights);  relation_forward_linear122_weights = None
        relation_forward_linear125_weights = self.all_parameters.PLinear109W
        einsum_21 = torch.functional.einsum('bwi,io->bwo', tanh, relation_forward_linear125_weights);  relation_forward_linear125_weights = None
        relation_forward_linear128_weights = self.all_parameters.PLinear113W
        einsum_22 = torch.functional.einsum('bwi,io->bwo', tanh, relation_forward_linear128_weights);  relation_forward_linear128_weights = None
        relation_forward_linear131_weights = self.all_parameters.PLinear117W
        einsum_23 = torch.functional.einsum('bwi,io->bwo', tanh, relation_forward_linear131_weights);  tanh = relation_forward_linear131_weights = None
        tanh_1 = torch.tanh(einsum_20);  einsum_20 = None
        tanh_2 = torch.tanh(einsum_21);  einsum_21 = None
        tanh_3 = torch.tanh(einsum_22);  einsum_22 = None
        tanh_4 = torch.tanh(einsum_23);  einsum_23 = None
        relation_forward_linear124_weights = self.all_parameters.PLinear107W
        einsum_24 = torch.functional.einsum('bwi,io->bwo', tanh_1, relation_forward_linear124_weights);  tanh_1 = relation_forward_linear124_weights = None
        relation_forward_linear127_weights = self.all_parameters.PLinear111W
        einsum_25 = torch.functional.einsum('bwi,io->bwo', tanh_2, relation_forward_linear127_weights);  tanh_2 = relation_forward_linear127_weights = None
        relation_forward_linear130_weights = self.all_parameters.PLinear115W
        einsum_26 = torch.functional.einsum('bwi,io->bwo', tanh_3, relation_forward_linear130_weights);  tanh_3 = relation_forward_linear130_weights = None
        relation_forward_linear133_weights = self.all_parameters.PLinear119W
        einsum_27 = torch.functional.einsum('bwi,io->bwo', tanh_4, relation_forward_linear133_weights);  tanh_4 = relation_forward_linear133_weights = None
        return ({'est_theta': einsum_27, 'est_thetadot': einsum_26, 'est_x': einsum_25, 'est_xdot': einsum_24}, {'SamplePart135': einsum_1, 'SamplePart137': einsum_2, 'SamplePart141': einsum_4, 'SamplePart139': einsum_3, 'Linear130': einsum_26, 'Linear133': einsum_27, 'Linear127': einsum_25, 'Linear124': einsum_24}, {'Xangle': einsum_27, 'Xangular_velocity': einsum_26, 'Xpos': einsum_25, 'Xvelocity': einsum_24}, {})
        
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
