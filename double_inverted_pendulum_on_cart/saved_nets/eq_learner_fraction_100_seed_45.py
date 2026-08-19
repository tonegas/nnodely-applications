import torch

def nnodely_basic_model_update_state(data_in, rel):
    data_out = data_in.clone()
    max_dim = min(rel.size(1), data_in.size(1))
    data_out[:, -max_dim:, :] = rel[:, -max_dim:, :]
    return data_out

def nnodely_basic_model_timeshift(data_in):
    return torch.cat((data_in[:, 1:, :], data_in[:, :1, :]), dim=1)

def nnodely_layers_parametricfunction_Pow2(x):
    return x **2

class TracerModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.all_parameters = {}
        self.all_constants = {}
        self.all_constants["SampleTime"] = torch.tensor(0.009999999776482582, requires_grad=False)
        self.all_parameters["PLinear17W"] = torch.nn.Parameter(torch.tensor([[2.029775619506836, -0.2621805965900421, -1.39096200466156, 2.3120319843292236, 0.9104390740394592, -0.6552103161811829, -0.6484490633010864, 0.035572733730077744, 0.03711334988474846, -0.09738340228796005, 0.09729617089033127, 0.9647720456123352, 0.0005364841199479997, -8.08812546893023e-05, 1.5168475329119246e-05, -0.0009600765770301223], [0.48206448554992676, -1.7095946073532104, -1.4585800170898438, 2.0377492904663086, -0.32924166321754456, -0.013139992952346802, -0.02039998397231102, 0.1267349123954773, 0.12656548619270325, -0.12121715396642685, 0.12084010988473892, 0.6474688649177551, 0.0007501691579818726, 0.0006347948801703751, -0.0003928877122234553, 7.517186895711347e-05], [0.15543925762176514, 1.0513263940811157, -1.379576563835144, 1.6075283288955688, 0.2281867414712906, 0.3337938189506531, 0.38797321915626526, 0.29166653752326965, 0.2922538220882416, -0.33765584230422974, 0.3375316262245178, -0.13161003589630127, 0.00022846291540190578, 2.6767675080918707e-05, -0.0007427692762576044, -0.0007124621770344675], [0.04743442311882973, 2.159156084060669, -0.8370015025138855, 0.9938234686851501, -0.20395123958587646, -1.0917408466339111, -1.035099744796753, -0.1193513572216034, -0.11903447657823563, 0.14505714178085327, -0.14642861485481262, -0.2685215473175049, 0.00012512982357293367, 0.0007333609974011779, 0.0002718767791520804, 0.0006260382942855358], [0.8082680702209473, 0.2578916549682617, 0.7706727981567383, -0.7716408967971802, -0.21946224570274353, -0.24656495451927185, -0.2549259066581726, -0.055707912892103195, -0.057523611932992935, 0.08159354329109192, -0.08213390409946442, 0.283803254365921, -7.265865860972553e-05, 0.00022223095584195107, -0.0005639826413244009, -0.0003367011668160558], [0.9830236434936523, 0.3307971656322479, 0.27272555232048035, -0.10052316635847092, -0.42030540108680725, 0.011582906357944012, 0.012157810851931572, 0.517397403717041, 0.5179622173309326, -0.5264477133750916, 0.5257837176322937, 0.3686217963695526, 0.0004606457659974694, 0.0007587543223053217, 0.0008245939970947802, -0.0001990660821320489], [0.6873002648353577, 0.019854987040162086, 0.8414864540100098, -0.5634669661521912, 0.16736438870429993, -0.25702282786369324, -0.26491209864616394, -0.21769174933433533, -0.21788069605827332, 0.22225745022296906, -0.22152845561504364, 0.31079819798469543, 0.000768760044593364, -0.00011794789315899834, -6.849241617601365e-05, -0.0001548788568470627], [0.6424185633659363, -0.010218131355941296, 0.865203857421875, -0.5230798721313477, 0.257159948348999, -0.2616574466228485, -0.2708011567592621, -0.2553933560848236, -0.2566939890384674, 0.2576076090335846, -0.2580240070819855, 0.32294124364852905, -0.0009995268192142248, -0.0008926254813559353, -0.0005956885870546103, -0.0006120550096966326], [-0.22911781072616577, -0.4101276695728302, -0.17476625740528107, 0.3603571653366089, 0.6303334832191467, -0.08203016966581345, -0.07985509186983109, -0.45488348603248596, -0.4542888104915619, 0.4441019296646118, -0.4438444972038269, -0.37008970975875854, 0.000568016548641026, -0.0007728142663836479, -0.0008035637438297272, 0.0007619266398251057]]), requires_grad=True)
        self.all_parameters["PLinear19W"] = torch.nn.Parameter(torch.tensor([[0.17295730113983154, 1.4423831701278687, -0.9360105991363525], [-0.4688326120376587, 0.6565104722976685, 1.836804747581482], [0.1658681333065033, 1.310525894165039, 1.4260083436965942], [0.315313458442688, 1.1903051137924194, 1.9577198028564453], [0.20273685455322266, -0.2527713477611542, 0.5832845568656921], [-0.16428902745246887, -0.3634073734283447, -0.033583689481019974], [0.23487672209739685, 0.3495773375034332, -0.33458903431892395], [-0.2708480656147003, -0.3530040681362152, 0.32098230719566345], [0.2331703156232834, 0.503942608833313, -0.30754950642585754]]), requires_grad=True)
        self.all_parameters["PLinear11W"] = torch.nn.Parameter(torch.tensor([[-0.07714127749204636, 0.07544731348752975, -0.0063252998515963554, -0.011583585292100906, 0.00906442292034626, -0.01920131966471672, 0.024996556341648102, -0.025809694081544876, -0.02490457147359848, 0.009776666760444641, -0.009647583588957787, 0.0010006290394812822, -0.0006940894527360797, -0.0009512905380688608, -0.00011961876589339226, 5.565180617850274e-05], [-0.012227139435708523, 0.013450045138597488, 0.0018407671013846993, 0.006628007162362337, 0.0020213432144373655, -0.009060045704245567, 0.005575756076723337, 0.008087567053735256, 0.008671491406857967, -0.004281462170183659, 0.004620802588760853, 9.865208994597197e-06, 0.0002105470048263669, -0.0003777867532335222, -0.0005723765352740884, 3.639535862021148e-05], [0.0038989742752164602, -0.03477455675601959, -0.4730522334575653, 0.4541471004486084, 0.013173257932066917, -0.019851306453347206, -0.026070963591337204, 0.00546335568651557, 0.0055547612719237804, 0.0020773238502442837, -0.0024711976293474436, -0.009981329552829266, 0.0007973216124810278, -0.0004518748610280454, -0.000388193700928241, -0.0003186841495335102], [0.002233007224276662, -0.0028037000447511673, 0.0028168014250695705, 0.00987508799880743, -0.00835060328245163, -0.009127449244260788, 0.004485039506107569, 0.010643561370670795, 0.011444245465099812, -0.00011026795255020261, 0.00098402239382267, 0.0009432467631995678, -0.00027182529447600245, 5.854563642060384e-05, 0.0007755617843940854, 0.00048304081428796053], [0.08601811528205872, 0.05653097480535507, 0.021479293704032898, 0.02865891344845295, 0.0606331005692482, 0.25599202513694763, 0.25534331798553467, -0.00827824231237173, -0.007788915187120438, 0.0577850379049778, -0.05731979385018349, -0.4341239631175995, 0.0008457473013550043, -0.0008014105260372162, 0.00035416841274127364, 0.00037008762592449784], [0.010467990301549435, -0.010967015288770199, 0.002630088711157441, -0.002263810485601425, 0.004379461985081434, -0.0014113365905359387, 0.007223319262266159, -0.01027281116694212, -0.00985171739012003, 0.0010001029586419463, -0.0002447600127197802, 0.0027699468191713095, 0.00035432277945801616, 9.13203475647606e-05, -0.0006637830520048738, -0.0005255073774605989], [-2.159736156463623, -1.9687107801437378, 0.1440776139497757, -0.04098568111658096, 1.4871957302093506, -0.2317054122686386, -0.20319896936416626, 1.6069939136505127, 1.606846809387207, 1.506130337715149, -1.5058931112289429, -0.10811091959476471, 6.070102608646266e-05, -0.0005460134707391262, 0.00027317649801261723, -0.0005515309749171138]]), requires_grad=True)
        self.all_parameters["PLinear13W"] = torch.nn.Parameter(torch.tensor([[-0.45428666472435, 0.754141628742218, 0.8448415398597717, -0.881198525428772, -0.24398064613342285, 0.2782360017299652, 0.28409305214881897, 0.45370498299598694, 0.4526588022708893, 0.48380348086357117, -0.4842440187931061, 0.21728819608688354, -0.00013467975077219307, 0.000585579895414412, -0.0004853869031649083, -8.044206879276317e-06], [-0.26994606852531433, 0.615341305732727, 0.7015805840492249, -0.7993984222412109, -0.41310614347457886, 0.33514875173568726, 0.34141454100608826, 0.4843519628047943, 0.4833934009075165, 0.49168699979782104, -0.49091392755508423, 0.4219927191734314, -0.00030207104282453656, 9.465339098824188e-05, -0.00028350556385703385, 0.0009496629936620593], [0.4073994755744934, -0.18414820730686188, -0.3505934774875641, 0.29728415608406067, -0.7771183848381042, -0.05224447697401047, -0.05900804325938225, -0.4016319811344147, -0.4016600549221039, -0.5245758891105652, 0.5233133435249329, -1.9928343296051025, 0.0005351982545107603, -0.00029843320953659713, 0.0004823340568691492, -0.0009577813907526433], [0.25873127579689026, -0.001979227876290679, -0.1809072047472, 0.1922191083431244, -0.33639493584632874, 0.0099245086312294, 0.009258796460926533, -0.1576722413301468, -0.15732760727405548, -0.2211810201406479, 0.2215295284986496, -1.7708520889282227, -0.00013049360131844878, -0.00018664552771952003, 0.000502424722071737, 0.00015842363063711673], [0.4604671001434326, -0.6462579965591431, 0.2050882875919342, -0.12948617339134216, 0.6026961803436279, -0.2887708842754364, -0.28954315185546875, -0.3674880266189575, -0.3659566640853882, -0.33639994263648987, 0.33602985739707947, -0.22787444293498993, -0.00058377580717206, -0.00024848984321579337, -0.0005873409099876881, 0.00011864191765198484], [0.008839046582579613, 0.00014280056348070502, -0.5102956295013428, 0.521069347858429, 0.32930389046669006, 0.35720664262771606, 0.35697561502456665, -0.010817425325512886, -0.009299482218921185, 0.026591524481773376, -0.026760734617710114, -0.16614863276481628, -0.0005694442661479115, 0.0001894415618153289, 9.626761311665177e-05, 0.0005265058716759086], [0.272964745759964, -0.8116617202758789, -0.023880673572421074, 0.1538892239332199, 0.415299654006958, -0.1864568144083023, -0.19275763630867004, -0.2820394039154053, -0.28119635581970215, -0.2603611946105957, 0.2593730688095093, 0.18132109940052032, 0.0006986483931541443, 0.0006666812696494162, -0.0009263110696338117, -9.447753836866468e-05], [0.5033323764801025, -0.679639995098114, 0.1844775378704071, -0.12446914613246918, 0.5677657723426819, -0.2749830484390259, -0.2736814022064209, -0.35480648279190063, -0.352939248085022, -0.32775208353996277, 0.3266792595386505, -0.17743995785713196, 0.00014867825666442513, 1.975121267605573e-05, 0.0002595095429569483, -0.00040059798629954457], [-0.10284200310707092, 0.0306010153144598, -0.6940840482711792, 0.6943243145942688, 0.15249212086200714, 0.40098699927330017, 0.4005805253982544, 0.03539412468671799, 0.03551995009183884, 0.05599828064441681, -0.05651877820491791, -0.2667769491672516, -0.0009375972440466285, 0.0005799990613013506, 0.00013729982310906053, -0.000744683900848031]]), requires_grad=True)
        self.all_parameters["PLinear15W"] = torch.nn.Parameter(torch.tensor([[-0.7913134694099426, 0.06324651092290878, -0.8159177303314209, -0.6594147086143494, -0.6477721333503723, 0.10475693643093109, -0.10501833260059357, -0.47909054160118103, -0.4791794717311859, -0.44262319803237915, 0.44298774003982544, -0.32154130935668945, -0.0008249054080806673, -0.0001194238502648659, -4.549406367004849e-05, -0.00020136298553552479], [0.8273311853408813, 0.26227790117263794, 0.3001255989074707, 0.41977202892303467, 0.7822454571723938, -0.06525078415870667, 0.08289667218923569, 0.5525550842285156, 0.5536282658576965, 0.47800663113594055, -0.47852638363838196, 0.6143949627876282, -0.0003176276804879308, -0.0009558926685713232, 0.0008353921002708375, -0.00033280643401667476], [0.06912274658679962, 0.6365675926208496, -1.8011765480041504, -1.4421740770339966, -0.24729451537132263, -0.2832247018814087, 0.275579571723938, -0.01991746388375759, -0.019258344545960426, 0.021176891401410103, -0.02147611789405346, 0.07081056386232376, -0.0009726438438519835, -0.0002231022372143343, -0.00014127072063274682, -4.650903429137543e-05], [0.08981429040431976, 0.5868310332298279, -1.8678603172302246, -1.3763172626495361, -0.26837214827537537, -0.29354551434516907, 0.2863742709159851, -0.016528937965631485, -0.016617927700281143, 0.03054545260965824, -0.030667563900351524, 0.1426229029893875, 0.0003055271226912737, -0.0005462672561407089, 7.047564577078447e-05, 0.0002442319237161428], [0.13070423901081085, 0.15447765588760376, 0.047642871737480164, -0.08662199229001999, -0.29214414954185486, 0.9061580300331116, -0.8789106011390686, -0.3662232756614685, -0.3665335178375244, -0.3967167139053345, 0.39754486083984375, -0.4673807621002197, 0.0008440471137873828, 0.00015811191406100988, 0.0005022400291636586, 0.0003179755003657192], [0.5556551218032837, 0.5901309251785278, 0.4017702043056488, 0.2717628479003906, 0.2609291970729828, -0.01678626984357834, 0.020879710093140602, 0.15025492012500763, 0.15179871022701263, 0.12169835716485977, -0.12188772112131119, -0.002743688179180026, -0.00018375365471001714, 0.00023119259276427329, -0.0008782981894910336, 0.0009411501814611256], [0.4894857108592987, 0.1374126374721527, 0.05836685001850128, 0.2174486219882965, 0.3259364366531372, -0.1695973426103592, 0.1649262011051178, 0.367297887802124, 0.36607664823532104, 0.3821311593055725, -0.38095754384994507, 0.2969059348106384, 0.0008819833747111261, 0.0008154567331075668, -0.0006380985723808408, -0.00022244310821406543], [0.5846779346466064, 0.17789651453495026, 0.06927379220724106, 0.2288558930158615, 0.34352701902389526, -0.06691034883260727, 0.06303872913122177, 0.3703339397907257, 0.3703781068325043, 0.3831225633621216, -0.382734477519989, 0.2727249264717102, 1.8436303435009904e-05, 0.0008696914883330464, -0.0008818075293675065, -3.743268098332919e-05], [-0.5230079293251038, -0.17759543657302856, 0.01772320084273815, -0.2171422690153122, -0.2300974428653717, -1.1254467964172363, 1.1359087228775024, -0.2750205397605896, -0.2735193967819214, -0.2884432375431061, 0.28697678446769714, 0.5340220928192139, 2.2635416826233268e-05, 2.7370550014893524e-05, 0.00015067093772813678, 0.0008340107160620391]]), requires_grad=True)
        self.all_constants["SamplePart1"] = torch.tensor([[1.0]], requires_grad=True)
        self.all_constants["SamplePart11"] = torch.tensor([[1.0]], requires_grad=True)
        self.all_constants["SamplePart13"] = torch.tensor([[1.0]], requires_grad=True)
        self.all_constants["SamplePart145"] = torch.tensor([[1.0]], requires_grad=True)
        self.all_constants["SamplePart151"] = torch.tensor([[1.0]], requires_grad=True)
        self.all_constants["SamplePart157"] = torch.tensor([[1.0]], requires_grad=True)
        self.all_constants["SamplePart163"] = torch.tensor([[1.0]], requires_grad=True)
        self.all_constants["SamplePart169"] = torch.tensor([[1.0]], requires_grad=True)
        self.all_constants["SamplePart175"] = torch.tensor([[1.0]], requires_grad=True)
        self.all_constants["SamplePart187"] = torch.tensor([[1.0]], requires_grad=True)
        self.all_constants["SamplePart189"] = torch.tensor([[1.0]], requires_grad=True)
        self.all_constants["SamplePart191"] = torch.tensor([[1.0]], requires_grad=True)
        self.all_constants["SamplePart3"] = torch.tensor([[1.0]], requires_grad=True)
        self.all_constants["SamplePart5"] = torch.tensor([[1.0]], requires_grad=True)
        self.all_constants["SamplePart7"] = torch.tensor([[1.0]], requires_grad=True)
        self.all_constants["SamplePart9"] = torch.tensor([[1.0]], requires_grad=True)
        self.all_constants["Select100"] = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], requires_grad=True)
        self.all_constants["Select103"] = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], requires_grad=True)
        self.all_constants["Select104"] = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0], requires_grad=True)
        self.all_constants["Select107"] = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0], requires_grad=True)
        self.all_constants["Select111"] = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], requires_grad=True)
        self.all_constants["Select113"] = torch.tensor([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], requires_grad=True)
        self.all_constants["Select116"] = torch.tensor([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], requires_grad=True)
        self.all_constants["Select119"] = torch.tensor([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], requires_grad=True)
        self.all_constants["Select122"] = torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], requires_grad=True)
        self.all_constants["Select125"] = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], requires_grad=True)
        self.all_constants["Select126"] = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], requires_grad=True)
        self.all_constants["Select129"] = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], requires_grad=True)
        self.all_constants["Select130"] = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], requires_grad=True)
        self.all_constants["Select133"] = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], requires_grad=True)
        self.all_constants["Select134"] = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0], requires_grad=True)
        self.all_constants["Select137"] = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0], requires_grad=True)
        self.all_constants["Select141"] = torch.tensor([1.0, 0.0, 0.0], requires_grad=True)
        self.all_constants["Select142"] = torch.tensor([0.0, 1.0, 0.0], requires_grad=True)
        self.all_constants["Select143"] = torch.tensor([0.0, 0.0, 1.0], requires_grad=True)
        self.all_constants["Select21"] = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], requires_grad=True)
        self.all_constants["Select23"] = torch.tensor([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], requires_grad=True)
        self.all_constants["Select26"] = torch.tensor([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], requires_grad=True)
        self.all_constants["Select29"] = torch.tensor([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], requires_grad=True)
        self.all_constants["Select32"] = torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], requires_grad=True)
        self.all_constants["Select35"] = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], requires_grad=True)
        self.all_constants["Select36"] = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], requires_grad=True)
        self.all_constants["Select39"] = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], requires_grad=True)
        self.all_constants["Select40"] = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], requires_grad=True)
        self.all_constants["Select43"] = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], requires_grad=True)
        self.all_constants["Select44"] = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0], requires_grad=True)
        self.all_constants["Select47"] = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0], requires_grad=True)
        self.all_constants["Select51"] = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], requires_grad=True)
        self.all_constants["Select53"] = torch.tensor([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], requires_grad=True)
        self.all_constants["Select56"] = torch.tensor([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], requires_grad=True)
        self.all_constants["Select59"] = torch.tensor([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], requires_grad=True)
        self.all_constants["Select62"] = torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], requires_grad=True)
        self.all_constants["Select65"] = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], requires_grad=True)
        self.all_constants["Select66"] = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], requires_grad=True)
        self.all_constants["Select69"] = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], requires_grad=True)
        self.all_constants["Select70"] = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], requires_grad=True)
        self.all_constants["Select73"] = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], requires_grad=True)
        self.all_constants["Select74"] = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0], requires_grad=True)
        self.all_constants["Select77"] = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0], requires_grad=True)
        self.all_constants["Select81"] = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], requires_grad=True)
        self.all_constants["Select83"] = torch.tensor([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], requires_grad=True)
        self.all_constants["Select86"] = torch.tensor([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], requires_grad=True)
        self.all_constants["Select89"] = torch.tensor([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], requires_grad=True)
        self.all_constants["Select92"] = torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], requires_grad=True)
        self.all_constants["Select95"] = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], requires_grad=True)
        self.all_constants["Select96"] = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], requires_grad=True)
        self.all_constants["Select99"] = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], requires_grad=True)
        self.all_parameters = torch.nn.ParameterDict(self.all_parameters)
        self.all_constants = torch.nn.ParameterDict(self.all_constants)

    def update(self, closed_loop={}, connect={}, disconnect=False):
        pass
    def forward(self, kwargs):
        getitem = kwargs['action']
        relation_forward_sample_part13_w = self.all_constants.SamplePart13
        einsum = torch.functional.einsum('bij,ki->bkj', getitem, relation_forward_sample_part13_w);  getitem = relation_forward_sample_part13_w = None
        getitem_1 = kwargs['Xth2_dot']
        relation_forward_sample_part11_w = self.all_constants.SamplePart11
        einsum_1 = torch.functional.einsum('bij,ki->bkj', getitem_1, relation_forward_sample_part11_w);  getitem_1 = relation_forward_sample_part11_w = None
        getitem_2 = kwargs['Xth2']
        relation_forward_sample_part9_w = self.all_constants.SamplePart9
        einsum_2 = torch.functional.einsum('bij,ki->bkj', getitem_2, relation_forward_sample_part9_w);  getitem_2 = relation_forward_sample_part9_w = None
        getitem_3 = kwargs['Xth1_dot']
        relation_forward_sample_part7_w = self.all_constants.SamplePart7
        einsum_3 = torch.functional.einsum('bij,ki->bkj', getitem_3, relation_forward_sample_part7_w);  getitem_3 = relation_forward_sample_part7_w = None
        getitem_4 = kwargs['Xth1']
        relation_forward_sample_part5_w = self.all_constants.SamplePart5
        einsum_4 = torch.functional.einsum('bij,ki->bkj', getitem_4, relation_forward_sample_part5_w);  getitem_4 = relation_forward_sample_part5_w = None
        getitem_5 = kwargs['Xvelocity']
        relation_forward_sample_part3_w = self.all_constants.SamplePart3
        einsum_5 = torch.functional.einsum('bij,ki->bkj', getitem_5, relation_forward_sample_part3_w);  getitem_5 = relation_forward_sample_part3_w = None
        getitem_6 = kwargs['Xpos']
        relation_forward_sample_part1_w = self.all_constants.SamplePart1
        einsum_6 = torch.functional.einsum('bij,ki->bkj', getitem_6, relation_forward_sample_part1_w);  getitem_6 = relation_forward_sample_part1_w = None
        cat = torch.cat((einsum_6, einsum_5), dim = 2);  einsum_6 = einsum_5 = None
        cat_1 = torch.cat((cat, einsum_4), dim = 2);  cat = einsum_4 = None
        cat_2 = torch.cat((cat_1, einsum_3), dim = 2);  cat_1 = einsum_3 = None
        cat_3 = torch.cat((cat_2, einsum_2), dim = 2);  cat_2 = einsum_2 = None
        cat_4 = torch.cat((cat_3, einsum_1), dim = 2);  cat_3 = einsum_1 = None
        cat_5 = torch.cat((cat_4, einsum), dim = 2);  cat_4 = einsum = None
        relation_forward_linear20_weights = self.all_parameters.PLinear11W
        einsum_7 = torch.functional.einsum('bwi,io->bwo', cat_5, relation_forward_linear20_weights);  cat_5 = relation_forward_linear20_weights = None
        relation_forward_select47_w = self.all_constants.Select47
        einsum_8 = torch.functional.einsum('ijk,k->ij', einsum_7, relation_forward_select47_w);  relation_forward_select47_w = None
        unsqueeze = einsum_8.unsqueeze(2);  einsum_8 = None
        pow2 = nnodely_layers_parametricfunction_Pow2(unsqueeze);  unsqueeze = None
        relation_forward_select44_w = self.all_constants.Select44
        einsum_9 = torch.functional.einsum('ijk,k->ij', einsum_7, relation_forward_select44_w);  relation_forward_select44_w = None
        unsqueeze_1 = einsum_9.unsqueeze(2);  einsum_9 = None
        relation_forward_select43_w = self.all_constants.Select43
        einsum_10 = torch.functional.einsum('ijk,k->ij', einsum_7, relation_forward_select43_w);  relation_forward_select43_w = None
        unsqueeze_2 = einsum_10.unsqueeze(2);  einsum_10 = None
        sub = unsqueeze_2 - unsqueeze_1;  unsqueeze_2 = unsqueeze_1 = None
        relation_forward_select40_w = self.all_constants.Select40
        einsum_11 = torch.functional.einsum('ijk,k->ij', einsum_7, relation_forward_select40_w);  relation_forward_select40_w = None
        unsqueeze_3 = einsum_11.unsqueeze(2);  einsum_11 = None
        relation_forward_select39_w = self.all_constants.Select39
        einsum_12 = torch.functional.einsum('ijk,k->ij', einsum_7, relation_forward_select39_w);  relation_forward_select39_w = None
        unsqueeze_4 = einsum_12.unsqueeze(2);  einsum_12 = None
        add = unsqueeze_4 + unsqueeze_3;  unsqueeze_4 = unsqueeze_3 = None
        relation_forward_select36_w = self.all_constants.Select36
        einsum_13 = torch.functional.einsum('ijk,k->ij', einsum_7, relation_forward_select36_w);  relation_forward_select36_w = None
        unsqueeze_5 = einsum_13.unsqueeze(2);  einsum_13 = None
        relation_forward_select35_w = self.all_constants.Select35
        einsum_14 = torch.functional.einsum('ijk,k->ij', einsum_7, relation_forward_select35_w);  relation_forward_select35_w = None
        unsqueeze_6 = einsum_14.unsqueeze(2);  einsum_14 = None
        mul = unsqueeze_6 * unsqueeze_5;  unsqueeze_6 = unsqueeze_5 = None
        relation_forward_select32_w = self.all_constants.Select32
        einsum_15 = torch.functional.einsum('ijk,k->ij', einsum_7, relation_forward_select32_w);  relation_forward_select32_w = None
        unsqueeze_7 = einsum_15.unsqueeze(2);  einsum_15 = None
        relation_forward_select29_w = self.all_constants.Select29
        einsum_16 = torch.functional.einsum('ijk,k->ij', einsum_7, relation_forward_select29_w);  relation_forward_select29_w = None
        unsqueeze_8 = einsum_16.unsqueeze(2);  einsum_16 = None
        cos = torch.cos(unsqueeze_8);  unsqueeze_8 = None
        relation_forward_select26_w = self.all_constants.Select26
        einsum_17 = torch.functional.einsum('ijk,k->ij', einsum_7, relation_forward_select26_w);  relation_forward_select26_w = None
        unsqueeze_9 = einsum_17.unsqueeze(2);  einsum_17 = None
        cos_1 = torch.cos(unsqueeze_9);  unsqueeze_9 = None
        relation_forward_select23_w = self.all_constants.Select23
        einsum_18 = torch.functional.einsum('ijk,k->ij', einsum_7, relation_forward_select23_w);  relation_forward_select23_w = None
        unsqueeze_10 = einsum_18.unsqueeze(2);  einsum_18 = None
        sin = torch.sin(unsqueeze_10);  unsqueeze_10 = None
        relation_forward_select21_w = self.all_constants.Select21
        einsum_19 = torch.functional.einsum('ijk,k->ij', einsum_7, relation_forward_select21_w);  einsum_7 = relation_forward_select21_w = None
        unsqueeze_11 = einsum_19.unsqueeze(2);  einsum_19 = None
        sin_1 = torch.sin(unsqueeze_11);  unsqueeze_11 = None
        cat_6 = torch.cat((sin_1, sin), dim = 2);  sin_1 = sin = None
        cat_7 = torch.cat((cat_6, cos_1), dim = 2);  cat_6 = cos_1 = None
        cat_8 = torch.cat((cat_7, cos), dim = 2);  cat_7 = cos = None
        cat_9 = torch.cat((cat_8, unsqueeze_7), dim = 2);  cat_8 = unsqueeze_7 = None
        cat_10 = torch.cat((cat_9, mul), dim = 2);  cat_9 = mul = None
        cat_11 = torch.cat((cat_10, add), dim = 2);  cat_10 = add = None
        cat_12 = torch.cat((cat_11, sub), dim = 2);  cat_11 = sub = None
        cat_13 = torch.cat((cat_12, pow2), dim = 2);  cat_12 = pow2 = None
        relation_forward_linear50_weights = self.all_parameters.PLinear13W
        einsum_20 = torch.functional.einsum('bwi,io->bwo', cat_13, relation_forward_linear50_weights);  cat_13 = relation_forward_linear50_weights = None
        relation_forward_select77_w = self.all_constants.Select77
        einsum_21 = torch.functional.einsum('ijk,k->ij', einsum_20, relation_forward_select77_w);  relation_forward_select77_w = None
        unsqueeze_12 = einsum_21.unsqueeze(2);  einsum_21 = None
        pow2_1 = nnodely_layers_parametricfunction_Pow2(unsqueeze_12);  unsqueeze_12 = None
        relation_forward_select74_w = self.all_constants.Select74
        einsum_22 = torch.functional.einsum('ijk,k->ij', einsum_20, relation_forward_select74_w);  relation_forward_select74_w = None
        unsqueeze_13 = einsum_22.unsqueeze(2);  einsum_22 = None
        relation_forward_select73_w = self.all_constants.Select73
        einsum_23 = torch.functional.einsum('ijk,k->ij', einsum_20, relation_forward_select73_w);  relation_forward_select73_w = None
        unsqueeze_14 = einsum_23.unsqueeze(2);  einsum_23 = None
        sub_1 = unsqueeze_14 - unsqueeze_13;  unsqueeze_14 = unsqueeze_13 = None
        relation_forward_select70_w = self.all_constants.Select70
        einsum_24 = torch.functional.einsum('ijk,k->ij', einsum_20, relation_forward_select70_w);  relation_forward_select70_w = None
        unsqueeze_15 = einsum_24.unsqueeze(2);  einsum_24 = None
        relation_forward_select69_w = self.all_constants.Select69
        einsum_25 = torch.functional.einsum('ijk,k->ij', einsum_20, relation_forward_select69_w);  relation_forward_select69_w = None
        unsqueeze_16 = einsum_25.unsqueeze(2);  einsum_25 = None
        add_1 = unsqueeze_16 + unsqueeze_15;  unsqueeze_16 = unsqueeze_15 = None
        relation_forward_select66_w = self.all_constants.Select66
        einsum_26 = torch.functional.einsum('ijk,k->ij', einsum_20, relation_forward_select66_w);  relation_forward_select66_w = None
        unsqueeze_17 = einsum_26.unsqueeze(2);  einsum_26 = None
        relation_forward_select65_w = self.all_constants.Select65
        einsum_27 = torch.functional.einsum('ijk,k->ij', einsum_20, relation_forward_select65_w);  relation_forward_select65_w = None
        unsqueeze_18 = einsum_27.unsqueeze(2);  einsum_27 = None
        mul_1 = unsqueeze_18 * unsqueeze_17;  unsqueeze_18 = unsqueeze_17 = None
        relation_forward_select62_w = self.all_constants.Select62
        einsum_28 = torch.functional.einsum('ijk,k->ij', einsum_20, relation_forward_select62_w);  relation_forward_select62_w = None
        unsqueeze_19 = einsum_28.unsqueeze(2);  einsum_28 = None
        relation_forward_select59_w = self.all_constants.Select59
        einsum_29 = torch.functional.einsum('ijk,k->ij', einsum_20, relation_forward_select59_w);  relation_forward_select59_w = None
        unsqueeze_20 = einsum_29.unsqueeze(2);  einsum_29 = None
        cos_2 = torch.cos(unsqueeze_20);  unsqueeze_20 = None
        relation_forward_select56_w = self.all_constants.Select56
        einsum_30 = torch.functional.einsum('ijk,k->ij', einsum_20, relation_forward_select56_w);  relation_forward_select56_w = None
        unsqueeze_21 = einsum_30.unsqueeze(2);  einsum_30 = None
        cos_3 = torch.cos(unsqueeze_21);  unsqueeze_21 = None
        relation_forward_select53_w = self.all_constants.Select53
        einsum_31 = torch.functional.einsum('ijk,k->ij', einsum_20, relation_forward_select53_w);  relation_forward_select53_w = None
        unsqueeze_22 = einsum_31.unsqueeze(2);  einsum_31 = None
        sin_2 = torch.sin(unsqueeze_22);  unsqueeze_22 = None
        relation_forward_select51_w = self.all_constants.Select51
        einsum_32 = torch.functional.einsum('ijk,k->ij', einsum_20, relation_forward_select51_w);  einsum_20 = relation_forward_select51_w = None
        unsqueeze_23 = einsum_32.unsqueeze(2);  einsum_32 = None
        sin_3 = torch.sin(unsqueeze_23);  unsqueeze_23 = None
        cat_14 = torch.cat((sin_3, sin_2), dim = 2);  sin_3 = sin_2 = None
        cat_15 = torch.cat((cat_14, cos_3), dim = 2);  cat_14 = cos_3 = None
        cat_16 = torch.cat((cat_15, cos_2), dim = 2);  cat_15 = cos_2 = None
        cat_17 = torch.cat((cat_16, unsqueeze_19), dim = 2);  cat_16 = unsqueeze_19 = None
        cat_18 = torch.cat((cat_17, mul_1), dim = 2);  cat_17 = mul_1 = None
        cat_19 = torch.cat((cat_18, add_1), dim = 2);  cat_18 = add_1 = None
        cat_20 = torch.cat((cat_19, sub_1), dim = 2);  cat_19 = sub_1 = None
        cat_21 = torch.cat((cat_20, pow2_1), dim = 2);  cat_20 = pow2_1 = None
        relation_forward_linear80_weights = self.all_parameters.PLinear15W
        einsum_33 = torch.functional.einsum('bwi,io->bwo', cat_21, relation_forward_linear80_weights);  cat_21 = relation_forward_linear80_weights = None
        relation_forward_select107_w = self.all_constants.Select107
        einsum_34 = torch.functional.einsum('ijk,k->ij', einsum_33, relation_forward_select107_w);  relation_forward_select107_w = None
        unsqueeze_24 = einsum_34.unsqueeze(2);  einsum_34 = None
        pow2_2 = nnodely_layers_parametricfunction_Pow2(unsqueeze_24);  unsqueeze_24 = None
        relation_forward_select104_w = self.all_constants.Select104
        einsum_35 = torch.functional.einsum('ijk,k->ij', einsum_33, relation_forward_select104_w);  relation_forward_select104_w = None
        unsqueeze_25 = einsum_35.unsqueeze(2);  einsum_35 = None
        relation_forward_select103_w = self.all_constants.Select103
        einsum_36 = torch.functional.einsum('ijk,k->ij', einsum_33, relation_forward_select103_w);  relation_forward_select103_w = None
        unsqueeze_26 = einsum_36.unsqueeze(2);  einsum_36 = None
        sub_2 = unsqueeze_26 - unsqueeze_25;  unsqueeze_26 = unsqueeze_25 = None
        relation_forward_select100_w = self.all_constants.Select100
        einsum_37 = torch.functional.einsum('ijk,k->ij', einsum_33, relation_forward_select100_w);  relation_forward_select100_w = None
        unsqueeze_27 = einsum_37.unsqueeze(2);  einsum_37 = None
        relation_forward_select99_w = self.all_constants.Select99
        einsum_38 = torch.functional.einsum('ijk,k->ij', einsum_33, relation_forward_select99_w);  relation_forward_select99_w = None
        unsqueeze_28 = einsum_38.unsqueeze(2);  einsum_38 = None
        add_2 = unsqueeze_28 + unsqueeze_27;  unsqueeze_28 = unsqueeze_27 = None
        relation_forward_select96_w = self.all_constants.Select96
        einsum_39 = torch.functional.einsum('ijk,k->ij', einsum_33, relation_forward_select96_w);  relation_forward_select96_w = None
        unsqueeze_29 = einsum_39.unsqueeze(2);  einsum_39 = None
        relation_forward_select95_w = self.all_constants.Select95
        einsum_40 = torch.functional.einsum('ijk,k->ij', einsum_33, relation_forward_select95_w);  relation_forward_select95_w = None
        unsqueeze_30 = einsum_40.unsqueeze(2);  einsum_40 = None
        mul_2 = unsqueeze_30 * unsqueeze_29;  unsqueeze_30 = unsqueeze_29 = None
        relation_forward_select92_w = self.all_constants.Select92
        einsum_41 = torch.functional.einsum('ijk,k->ij', einsum_33, relation_forward_select92_w);  relation_forward_select92_w = None
        unsqueeze_31 = einsum_41.unsqueeze(2);  einsum_41 = None
        relation_forward_select89_w = self.all_constants.Select89
        einsum_42 = torch.functional.einsum('ijk,k->ij', einsum_33, relation_forward_select89_w);  relation_forward_select89_w = None
        unsqueeze_32 = einsum_42.unsqueeze(2);  einsum_42 = None
        cos_4 = torch.cos(unsqueeze_32);  unsqueeze_32 = None
        relation_forward_select86_w = self.all_constants.Select86
        einsum_43 = torch.functional.einsum('ijk,k->ij', einsum_33, relation_forward_select86_w);  relation_forward_select86_w = None
        unsqueeze_33 = einsum_43.unsqueeze(2);  einsum_43 = None
        cos_5 = torch.cos(unsqueeze_33);  unsqueeze_33 = None
        relation_forward_select83_w = self.all_constants.Select83
        einsum_44 = torch.functional.einsum('ijk,k->ij', einsum_33, relation_forward_select83_w);  relation_forward_select83_w = None
        unsqueeze_34 = einsum_44.unsqueeze(2);  einsum_44 = None
        sin_4 = torch.sin(unsqueeze_34);  unsqueeze_34 = None
        relation_forward_select81_w = self.all_constants.Select81
        einsum_45 = torch.functional.einsum('ijk,k->ij', einsum_33, relation_forward_select81_w);  einsum_33 = relation_forward_select81_w = None
        unsqueeze_35 = einsum_45.unsqueeze(2);  einsum_45 = None
        sin_5 = torch.sin(unsqueeze_35);  unsqueeze_35 = None
        cat_22 = torch.cat((sin_5, sin_4), dim = 2);  sin_5 = sin_4 = None
        cat_23 = torch.cat((cat_22, cos_5), dim = 2);  cat_22 = cos_5 = None
        cat_24 = torch.cat((cat_23, cos_4), dim = 2);  cat_23 = cos_4 = None
        cat_25 = torch.cat((cat_24, unsqueeze_31), dim = 2);  cat_24 = unsqueeze_31 = None
        cat_26 = torch.cat((cat_25, mul_2), dim = 2);  cat_25 = mul_2 = None
        cat_27 = torch.cat((cat_26, add_2), dim = 2);  cat_26 = add_2 = None
        cat_28 = torch.cat((cat_27, sub_2), dim = 2);  cat_27 = sub_2 = None
        cat_29 = torch.cat((cat_28, pow2_2), dim = 2);  cat_28 = pow2_2 = None
        relation_forward_linear110_weights = self.all_parameters.PLinear17W
        einsum_46 = torch.functional.einsum('bwi,io->bwo', cat_29, relation_forward_linear110_weights);  cat_29 = relation_forward_linear110_weights = None
        relation_forward_select137_w = self.all_constants.Select137
        einsum_47 = torch.functional.einsum('ijk,k->ij', einsum_46, relation_forward_select137_w);  relation_forward_select137_w = None
        unsqueeze_36 = einsum_47.unsqueeze(2);  einsum_47 = None
        pow2_3 = nnodely_layers_parametricfunction_Pow2(unsqueeze_36);  unsqueeze_36 = None
        relation_forward_select134_w = self.all_constants.Select134
        einsum_48 = torch.functional.einsum('ijk,k->ij', einsum_46, relation_forward_select134_w);  relation_forward_select134_w = None
        unsqueeze_37 = einsum_48.unsqueeze(2);  einsum_48 = None
        relation_forward_select133_w = self.all_constants.Select133
        einsum_49 = torch.functional.einsum('ijk,k->ij', einsum_46, relation_forward_select133_w);  relation_forward_select133_w = None
        unsqueeze_38 = einsum_49.unsqueeze(2);  einsum_49 = None
        sub_3 = unsqueeze_38 - unsqueeze_37;  unsqueeze_38 = unsqueeze_37 = None
        relation_forward_select130_w = self.all_constants.Select130
        einsum_50 = torch.functional.einsum('ijk,k->ij', einsum_46, relation_forward_select130_w);  relation_forward_select130_w = None
        unsqueeze_39 = einsum_50.unsqueeze(2);  einsum_50 = None
        relation_forward_select129_w = self.all_constants.Select129
        einsum_51 = torch.functional.einsum('ijk,k->ij', einsum_46, relation_forward_select129_w);  relation_forward_select129_w = None
        unsqueeze_40 = einsum_51.unsqueeze(2);  einsum_51 = None
        add_3 = unsqueeze_40 + unsqueeze_39;  unsqueeze_40 = unsqueeze_39 = None
        relation_forward_select126_w = self.all_constants.Select126
        einsum_52 = torch.functional.einsum('ijk,k->ij', einsum_46, relation_forward_select126_w);  relation_forward_select126_w = None
        unsqueeze_41 = einsum_52.unsqueeze(2);  einsum_52 = None
        relation_forward_select125_w = self.all_constants.Select125
        einsum_53 = torch.functional.einsum('ijk,k->ij', einsum_46, relation_forward_select125_w);  relation_forward_select125_w = None
        unsqueeze_42 = einsum_53.unsqueeze(2);  einsum_53 = None
        mul_3 = unsqueeze_42 * unsqueeze_41;  unsqueeze_42 = unsqueeze_41 = None
        relation_forward_select122_w = self.all_constants.Select122
        einsum_54 = torch.functional.einsum('ijk,k->ij', einsum_46, relation_forward_select122_w);  relation_forward_select122_w = None
        unsqueeze_43 = einsum_54.unsqueeze(2);  einsum_54 = None
        relation_forward_select119_w = self.all_constants.Select119
        einsum_55 = torch.functional.einsum('ijk,k->ij', einsum_46, relation_forward_select119_w);  relation_forward_select119_w = None
        unsqueeze_44 = einsum_55.unsqueeze(2);  einsum_55 = None
        cos_6 = torch.cos(unsqueeze_44);  unsqueeze_44 = None
        relation_forward_select116_w = self.all_constants.Select116
        einsum_56 = torch.functional.einsum('ijk,k->ij', einsum_46, relation_forward_select116_w);  relation_forward_select116_w = None
        unsqueeze_45 = einsum_56.unsqueeze(2);  einsum_56 = None
        cos_7 = torch.cos(unsqueeze_45);  unsqueeze_45 = None
        relation_forward_select113_w = self.all_constants.Select113
        einsum_57 = torch.functional.einsum('ijk,k->ij', einsum_46, relation_forward_select113_w);  relation_forward_select113_w = None
        unsqueeze_46 = einsum_57.unsqueeze(2);  einsum_57 = None
        sin_6 = torch.sin(unsqueeze_46);  unsqueeze_46 = None
        relation_forward_select111_w = self.all_constants.Select111
        einsum_58 = torch.functional.einsum('ijk,k->ij', einsum_46, relation_forward_select111_w);  einsum_46 = relation_forward_select111_w = None
        unsqueeze_47 = einsum_58.unsqueeze(2);  einsum_58 = None
        sin_7 = torch.sin(unsqueeze_47);  unsqueeze_47 = None
        cat_30 = torch.cat((sin_7, sin_6), dim = 2);  sin_7 = sin_6 = None
        cat_31 = torch.cat((cat_30, cos_7), dim = 2);  cat_30 = cos_7 = None
        cat_32 = torch.cat((cat_31, cos_6), dim = 2);  cat_31 = cos_6 = None
        cat_33 = torch.cat((cat_32, unsqueeze_43), dim = 2);  cat_32 = unsqueeze_43 = None
        cat_34 = torch.cat((cat_33, mul_3), dim = 2);  cat_33 = mul_3 = None
        cat_35 = torch.cat((cat_34, add_3), dim = 2);  cat_34 = add_3 = None
        cat_36 = torch.cat((cat_35, sub_3), dim = 2);  cat_35 = sub_3 = None
        cat_37 = torch.cat((cat_36, pow2_3), dim = 2);  cat_36 = pow2_3 = None
        relation_forward_linear140_weights = self.all_parameters.PLinear19W
        einsum_59 = torch.functional.einsum('bwi,io->bwo', cat_37, relation_forward_linear140_weights);  cat_37 = relation_forward_linear140_weights = None
        relation_forward_select143_w = self.all_constants.Select143
        einsum_60 = torch.functional.einsum('ijk,k->ij', einsum_59, relation_forward_select143_w);  relation_forward_select143_w = None
        unsqueeze_48 = einsum_60.unsqueeze(2);  einsum_60 = None
        getitem_7 = kwargs['Xddth2']
        relation_forward_sample_part191_w = self.all_constants.SamplePart191
        einsum_61 = torch.functional.einsum('bij,ki->bkj', getitem_7, relation_forward_sample_part191_w);  getitem_7 = relation_forward_sample_part191_w = None
        relation_forward_select142_w = self.all_constants.Select142
        einsum_62 = torch.functional.einsum('ijk,k->ij', einsum_59, relation_forward_select142_w);  relation_forward_select142_w = None
        unsqueeze_49 = einsum_62.unsqueeze(2);  einsum_62 = None
        getitem_8 = kwargs['Xddth1']
        relation_forward_sample_part189_w = self.all_constants.SamplePart189
        einsum_63 = torch.functional.einsum('bij,ki->bkj', getitem_8, relation_forward_sample_part189_w);  getitem_8 = relation_forward_sample_part189_w = None
        relation_forward_select141_w = self.all_constants.Select141
        einsum_64 = torch.functional.einsum('ijk,k->ij', einsum_59, relation_forward_select141_w);  einsum_59 = relation_forward_select141_w = None
        unsqueeze_50 = einsum_64.unsqueeze(2);  einsum_64 = None
        getitem_9 = kwargs['Xddx']
        relation_forward_sample_part187_w = self.all_constants.SamplePart187
        einsum_65 = torch.functional.einsum('bij,ki->bkj', getitem_9, relation_forward_sample_part187_w);  getitem_9 = relation_forward_sample_part187_w = None
        all_constants_sample_time = self.all_constants.SampleTime
        mul_4 = unsqueeze_48 * all_constants_sample_time
        getitem_10 = kwargs['int_th2_dot']
        relation_forward_sample_part157_w = self.all_constants.SamplePart157
        einsum_66 = torch.functional.einsum('bij,ki->bkj', getitem_10, relation_forward_sample_part157_w);  getitem_10 = relation_forward_sample_part157_w = None
        add_4 = einsum_66 + mul_4;  einsum_66 = mul_4 = None
        mul_5 = add_4 * all_constants_sample_time
        getitem_11 = kwargs['int_th2']
        relation_forward_sample_part175_w = self.all_constants.SamplePart175
        einsum_67 = torch.functional.einsum('bij,ki->bkj', getitem_11, relation_forward_sample_part175_w);  getitem_11 = relation_forward_sample_part175_w = None
        add_5 = einsum_67 + mul_5;  einsum_67 = mul_5 = None
        mul_6 = unsqueeze_49 * all_constants_sample_time
        getitem_12 = kwargs['int_th1_dot']
        relation_forward_sample_part151_w = self.all_constants.SamplePart151
        einsum_68 = torch.functional.einsum('bij,ki->bkj', getitem_12, relation_forward_sample_part151_w);  getitem_12 = relation_forward_sample_part151_w = None
        add_6 = einsum_68 + mul_6;  einsum_68 = mul_6 = None
        mul_7 = add_6 * all_constants_sample_time
        getitem_13 = kwargs['int_th1']
        relation_forward_sample_part169_w = self.all_constants.SamplePart169
        einsum_69 = torch.functional.einsum('bij,ki->bkj', getitem_13, relation_forward_sample_part169_w);  getitem_13 = relation_forward_sample_part169_w = None
        add_7 = einsum_69 + mul_7;  einsum_69 = mul_7 = None
        mul_8 = unsqueeze_50 * all_constants_sample_time
        getitem_14 = kwargs['int_xdot']
        relation_forward_sample_part145_w = self.all_constants.SamplePart145
        einsum_70 = torch.functional.einsum('bij,ki->bkj', getitem_14, relation_forward_sample_part145_w);  getitem_14 = relation_forward_sample_part145_w = None
        add_8 = einsum_70 + mul_8;  einsum_70 = mul_8 = None
        mul_9 = add_8 * all_constants_sample_time;  all_constants_sample_time = None
        getitem_15 = kwargs['int_x'];  kwargs = None
        relation_forward_sample_part163_w = self.all_constants.SamplePart163
        einsum_71 = torch.functional.einsum('bij,ki->bkj', getitem_15, relation_forward_sample_part163_w);  getitem_15 = relation_forward_sample_part163_w = None
        add_9 = einsum_71 + mul_9;  einsum_71 = mul_9 = None
        return ({'th2_ddot_est': unsqueeze_48, 'th1_ddot_est': unsqueeze_49, 'acc_cart_est': unsqueeze_50, 'th2_est': add_5, 'omega2_est': add_4, 'th1_est': add_7, 'omega1_est': add_6, 'x_est': add_9, 'xdot_est': add_8}, {'SamplePart187': einsum_65, 'SamplePart189': einsum_63, 'SamplePart191': einsum_61, 'Select141': unsqueeze_50, 'Select142': unsqueeze_49, 'Select143': unsqueeze_48}, {'Xth2_dot': add_4, 'Xth2': add_5, 'Xth1_dot': add_6, 'Xth1': add_7, 'Xvelocity': add_8, 'Xpos': add_9, 'int_th2': add_5, 'int_th2_dot': add_4, 'int_th1': add_7, 'int_th1_dot': add_6, 'int_x': add_9, 'int_xdot': add_8}, {})
        
class RecurrentModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.Cell = TracerModel()
        self.inputs = ['action', 'Xddth2', 'Xddth1', 'Xddx', ]
        self.states = dict()

    def forward(self, kwargs, n_samples = None):
        n_samples = n_samples if n_samples else min([kwargs[key].size(0) for key in self.inputs])
        self.states['Xth2_dot'] = kwargs['Xth2_dot']
        self.states['Xth2'] = kwargs['Xth2']
        self.states['Xth1_dot'] = kwargs['Xth1_dot']
        self.states['Xth1'] = kwargs['Xth1']
        self.states['Xvelocity'] = kwargs['Xvelocity']
        self.states['Xpos'] = kwargs['Xpos']
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
