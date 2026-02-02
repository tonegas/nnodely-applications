import torch

def nnodely_basic_model_connect(data_in, rel):
    virtual = torch.cat((data_in[:, 1:, :], data_in[:, :1, :]), dim=1)
    max_dim = min(rel.size(1), data_in.size(1))
    virtual[:, -max_dim:, :] = rel[:, -max_dim:, :]
    return virtual

def nnodely_layers_parametricfunction_Pow2(x):
    return x **2

class TracerModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.all_parameters = {}
        self.all_constants = {}
        self.all_constants["Constant31"] = torch.tensor([1.000000013351432e-10], requires_grad=False)
        self.all_constants["Constant32"] = torch.tensor([1.000000013351432e-10], requires_grad=False)
        self.all_constants["Constant33"] = torch.tensor([1.000000013351432e-10], requires_grad=False)
        self.all_constants["SampleTime"] = torch.tensor(0.009999999776482582, requires_grad=False)
        self.all_parameters["PLinear11W"] = torch.nn.Parameter(torch.tensor([[-1.3528778553009033, -1.197702169418335, -1.01039457321167, -0.006181459873914719, -0.10385538637638092, -1.2940645217895508, -1.2942085266113281, -1.1160240173339844, 1.1156129837036133, 0.06840094923973083, 0.0003562530328053981, -0.000912685296498239, 0.0009190026903524995, -5.856451389263384e-05, 0.0005035487120039761, -0.0008890252211131155], [-0.008702542632818222, 0.028879603371024132, 0.029129911214113235, 0.4400346279144287, 0.6404422521591187, -0.04129841923713684, -0.040612511336803436, -0.0032073832117021084, 0.0038052622694522142, 0.11847797781229019, 5.16000873176381e-05, -0.00046665073023177683, 4.238805558998138e-05, -0.0005897785304114223, 0.0006400797283276916, 0.0009283064282499254], [0.07278305292129517, -0.02438485249876976, -0.021400516852736473, -0.07725560665130615, -0.3516107201576233, -0.05069880560040474, -0.05018385127186775, 0.05487307161092758, -0.055831484496593475, -0.1905922144651413, 0.000621558865532279, -0.0008817358175292611, 0.0001551450986880809, -0.0006114810239523649, -1.3978487913846038e-05, 0.0008346210815943778], [-0.025603098794817924, -0.02262748032808304, 0.009953131899237633, 0.08252615481615067, 0.09715460240840912, -0.018960101529955864, -0.018350373953580856, 0.005217196419835091, -0.005098446737974882, 0.054797641932964325, -0.0007865990628488362, 9.801710984902456e-05, 0.0004694427188951522, 0.0008994172094389796, 0.000777796667534858, -0.00023743926431052387], [0.09542535990476608, 0.05552985891699791, 0.022790512070059776, 0.040575187653303146, -0.22604721784591675, -0.08111406862735748, -0.08163280785083771, 0.04027310758829117, -0.04096013307571411, -0.212638720870018, 0.0002658228331711143, -0.0003004575555678457, -0.0002738349430728704, -0.0005389482830651104, -0.0006859080749563873, 0.000749787490349263], [0.3933517038822174, 0.7619304656982422, 1.308672547340393, 0.06529968976974487, -0.21420255303382874, -0.18100857734680176, -0.1820543110370636, -0.1392735242843628, 0.14047464728355408, 0.05455075204372406, 0.0009443467715755105, 0.0005268005770631135, -0.00040925072971731424, 0.0009117767913267016, 0.0007198106031864882, 0.00010151088645216078], [-0.05425079166889191, -0.018591051921248436, -0.027202796190977097, 0.12982389330863953, -0.01137352455407381, 0.17197534441947937, 0.17095080018043518, -0.08719463646411896, 0.08748136460781097, -0.15275955200195312, 0.000990438275039196, -0.0009282047394663095, -0.0008680797764100134, -3.596238821046427e-05, -0.0001498653437010944, 0.000998975825496018]]), requires_grad=True)
        self.all_parameters["PLinear13W"] = torch.nn.Parameter(torch.tensor([[-0.5782089233398438, -0.12072131037712097, -0.1670076549053192, 0.1933613270521164, -0.06454258412122726, 0.054969388991594315, 0.054866302758455276, -0.5753000378608704, 0.5765814185142517, 0.17057356238365173, -0.0005760364583693445, 0.0006557730375789106, 0.0001738541468512267, -8.536185487173498e-05, -0.0003228849673178047, -0.00011861764505738392], [0.2746467590332031, -0.4457990229129791, -0.23768168687820435, 0.07316314429044724, 0.11571314930915833, -0.04478474706411362, -0.04498276859521866, 0.02443006820976734, -0.024975473061203957, -0.30611762404441833, 0.00011203023314010352, -0.0004874830774497241, -0.0009575069416314363, 5.035495269112289e-05, -0.0004617647500708699, -0.0006058667786419392], [-0.43173548579216003, -0.5136255025863647, -0.24066492915153503, 0.08026555925607681, -0.048185840249061584, -0.14553695917129517, -0.14502090215682983, 0.30260229110717773, -0.3026231527328491, 0.0922544002532959, -0.0005746489623561502, -0.0009859970305114985, 0.000563585024792701, -0.0002315556484973058, 0.0006867614574730396, 0.0006547546945512295], [-0.02389657497406006, 0.03036770224571228, 0.018790559843182564, 0.004462850745767355, -0.5356543064117432, 0.0019712778739631176, 0.0028222091495990753, -0.02104969322681427, 0.020265908911824226, 0.0038422001525759697, 0.000387736625270918, -9.270579903386533e-05, -2.756091271294281e-05, -0.00020384257368277758, -0.000177639871253632, -4.671894566854462e-05], [-0.19796118140220642, 0.06911389529705048, -0.08103623986244202, 0.09860549122095108, 0.027149183675646782, -0.11643119156360626, -0.11630046367645264, -0.25188392400741577, 0.2505962550640106, -0.06783684343099594, 0.0008709188550710678, 0.0004212293424643576, 0.0005438243388198316, -0.00010584267874946818, 0.0006986977532505989, 0.0006457953131757677], [-0.13242557644844055, 0.2363239973783493, 0.30859750509262085, 0.21596424281597137, 0.007825227454304695, -0.13292567431926727, -0.13131606578826904, 0.01709180697798729, -0.018034901469945908, 0.028307829052209854, 0.0006382835563272238, -0.0008689006790518761, -0.0002582398592494428, -0.00019984037498943508, 0.0007047514664009213, 0.0003401864960324019], [-0.1028875857591629, 0.1524282544851303, 0.10349471122026443, -0.0010562158422544599, -1.4672069549560547, 0.020052429288625717, 0.021872147917747498, -0.02828720584511757, 0.026959948241710663, -0.07988433539867401, 0.00045765048707835376, -1.4517864656227175e-05, 0.00038659365964122117, -5.6352866522502154e-05, 0.0007504753884859383, -0.0006176550523377955]]), requires_grad=True)
        self.all_parameters["PLinear15W"] = torch.nn.Parameter(torch.tensor([[-0.05295371264219284, -0.617633044719696, 0.08251166343688965, -0.5372825860977173, -0.10536953061819077, -0.08264755457639694, -0.08118846267461777, -0.16777533292770386, 0.1685282588005066, 0.5994567275047302, 0.0005581589066423476, -3.96833784179762e-05, 0.0007302239537239075, -0.00012192311260150746, 0.0005749387782998383, -0.0006614740123040974], [0.03640201315283775, 0.4517880380153656, -0.4130822420120239, 0.42663922905921936, -0.25758445262908936, -0.21425175666809082, -0.21416616439819336, -0.22704194486141205, 0.2268076092004776, -0.13849179446697235, 0.000271181866992265, -0.0007939901552163064, -0.000639371108263731, 0.0008714202558621764, -0.0006433506496250629, 0.0006858243723399937], [0.0981413871049881, -0.3299817144870758, -0.06743570417165756, 0.19696912169456482, 0.005563416983932257, 0.1566932499408722, 0.15694254636764526, -0.10282071679830551, 0.10270925611257553, 0.17039969563484192, -0.0005832894239574671, 5.1601095037767664e-05, -0.0004054380115121603, -0.0005007882718928158, 2.6868239729083143e-05, 0.0005431516910903156], [0.04794556275010109, -0.17742012441158295, -0.16053828597068787, 0.356343150138855, 0.4063376784324646, -0.014994754455983639, -0.013639659620821476, 0.011617759242653847, -0.011496404185891151, 0.10137395560741425, 3.629504863056354e-05, -2.1405055576906307e-06, -0.0005260555190034211, -0.0002874605997931212, 0.0004847102100029588, -0.0004404672945383936], [-0.31941455602645874, -0.04642550274729729, -0.09589270502328873, 0.10329916328191757, 0.05895096808671951, -0.07131222635507584, -0.07172050327062607, 0.17575551569461823, -0.17606313526630402, 0.10888849943876266, -0.00025368257774971426, -0.0003814521769527346, 0.0008728976245038211, 0.0005198820144869387, -0.0008803498931229115, 0.00011138372065033764], [-0.011194744147360325, 0.1829666644334793, -0.10421006381511688, 0.480762779712677, 0.2736356854438782, 0.04086478799581528, 0.041389018297195435, 0.01643730141222477, -0.017315398901700974, -0.24082022905349731, -0.0004968214780092239, -0.0002827304997481406, 0.0006681729573756456, 2.506502096366603e-05, -0.0008947218884713948, -0.0009119423339143395], [-0.00652447110041976, 0.6933597326278687, 0.507922887802124, 0.34888172149658203, -0.4304167628288269, 0.017876407131552696, 0.01629999466240406, 0.17422696948051453, -0.17266157269477844, -0.5953245759010315, -0.0003230813017580658, -0.0004697717376984656, 0.0004046509857289493, -0.0006420559948310256, 0.00046241405652835965, 0.00026196049293503165]]), requires_grad=True)
        self.all_parameters["PLinear17W"] = torch.nn.Parameter(torch.tensor([[-3.269441604614258, -0.30887043476104736, -3.512974262237549, 3.8778388500213623, 3.874152660369873, 2.5814225673675537, 2.581441640853882, -2.971438407897949, 2.970551013946533, 4.532593250274658, -0.0008719873148947954, 0.0008006697753444314, -0.0005547032342292368, 0.0008850719314068556, -0.000900939223356545, 0.0002701602934394032], [-0.7354493141174316, 2.645170211791992, -0.49901166558265686, 0.2832795977592468, 0.28438684344291687, -0.769756019115448, -0.7694880366325378, 0.7561246752738953, -0.7573756575584412, -0.13608016073703766, -0.00048347245319746435, -0.0005041956901550293, 0.0007047888939268887, 8.15960083855316e-05, 0.00021681752696167678, -0.0006141273188404739], [-0.10497292876243591, 1.920396327972412, 0.5511963963508606, 1.135060429573059, 1.1401344537734985, -1.4417235851287842, -1.4417309761047363, -0.24126824736595154, 0.24023234844207764, 0.5699265599250793, 0.00035205495078116655, -0.0008043724810704589, 0.00017460617527831346, 0.00017502362607046962, -0.0007617064984515309, -6.300795212155208e-05], [0.03857419639825821, 1.316137433052063, 0.05765911936759949, 1.0631649494171143, 1.0612826347351074, 0.12489543110132217, 0.12544524669647217, 0.024761240929365158, -0.023692216724157333, -1.1256109476089478, 0.0007840512553229928, 0.0007528192945756018, -0.0008285307558253407, -0.0007449081749655306, 0.00042242940980941057, -0.00011478176020318642], [-1.3589844703674316, 0.33727169036865234, -1.459439754486084, 0.7196765542030334, 0.7312577366828918, -1.2660537958145142, -1.2660473585128784, 0.1991044580936432, -0.19825664162635803, 0.11854322254657745, 0.0004349005757831037, -0.0007544542313553393, -0.0008842484676279128, -0.0008981908904388547, 0.0003804480074904859, 0.0005070955376140773], [-0.3653894066810608, 3.982247829437256, -0.24285632371902466, -0.7713018655776978, -0.7690299153327942, 0.8770894408226013, 0.8777654767036438, 1.3688682317733765, -1.368906855583191, -1.0537034273147583, 5.5805270676501095e-05, 0.00045989584759809077, 0.0003496817662380636, 1.688458360149525e-05, 0.000527962576597929, -0.0007816210854798555], [0.640576958656311, 2.921182155609131, 0.3551841378211975, 1.0165603160858154, 1.016121506690979, 0.3483865559101105, 0.347940593957901, -0.2061879187822342, 0.20729543268680573, -1.0220887660980225, -0.0009631947614252567, -7.45274664950557e-05, -0.00030289971618913114, -0.0003365800657775253, 0.000560891639906913, 0.0005769066046923399]]), requires_grad=True)
        self.all_parameters["PLinear19W"] = torch.nn.Parameter(torch.tensor([[-0.955195426940918, 4.988913536071777, -9.66148853302002], [-0.12637317180633545, -0.5987063050270081, 0.885278582572937], [1.556978464126587, -0.6628533005714417, 1.0684380531311035], [-0.43663763999938965, -0.737395703792572, 1.1034256219863892], [-0.1029847040772438, 0.4695146977901459, -3.4172720909118652], [0.06284455955028534, -1.67068612575531, 1.6681509017944336], [0.39110392332077026, 0.6596296429634094, -0.980027973651886]]), requires_grad=True)
        self.all_parameters["PLinear21W"] = torch.nn.Parameter(torch.tensor([[-0.014860376715660095], [0.11112932115793228], [0.044473353773355484], [0.1374933421611786], [0.05163102596998215], [0.03046674095094204], [0.14146041870117188]]), requires_grad=True)
        self.all_constants["SamplePart1"] = torch.tensor([[1.]], requires_grad=False)
        self.all_constants["SamplePart100"] = torch.tensor([[1., 0.]], requires_grad=False)
        self.all_constants["SamplePart102"] = torch.tensor([[1., 0.]], requires_grad=False)
        self.all_constants["SamplePart104"] = torch.tensor([[1., 0.]], requires_grad=False)
        self.all_constants["SamplePart106"] = torch.tensor([[1., 0.]], requires_grad=False)
        self.all_constants["SamplePart11"] = torch.tensor([[1., 0.]], requires_grad=False)
        self.all_constants["SamplePart13"] = torch.tensor([[1., 0.]], requires_grad=False)
        self.all_constants["SamplePart223"] = torch.tensor([[1.]], requires_grad=False)
        self.all_constants["SamplePart229"] = torch.tensor([[1.]], requires_grad=False)
        self.all_constants["SamplePart235"] = torch.tensor([[1.]], requires_grad=False)
        self.all_constants["SamplePart241"] = torch.tensor([[1.]], requires_grad=False)
        self.all_constants["SamplePart247"] = torch.tensor([[1.]], requires_grad=False)
        self.all_constants["SamplePart253"] = torch.tensor([[1.]], requires_grad=False)
        self.all_constants["SamplePart259"] = torch.tensor([[1.]], requires_grad=False)
        self.all_constants["SamplePart261"] = torch.tensor([[1.]], requires_grad=False)
        self.all_constants["SamplePart263"] = torch.tensor([[1.]], requires_grad=False)
        self.all_constants["SamplePart265"] = torch.tensor([[0., 1.]], requires_grad=False)
        self.all_constants["SamplePart267"] = torch.tensor([[0., 1.]], requires_grad=False)
        self.all_constants["SamplePart269"] = torch.tensor([[0., 1.]], requires_grad=False)
        self.all_constants["SamplePart271"] = torch.tensor([[0., 1.]], requires_grad=False)
        self.all_constants["SamplePart273"] = torch.tensor([[0., 1.]], requires_grad=False)
        self.all_constants["SamplePart275"] = torch.tensor([[0., 1.]], requires_grad=False)
        self.all_constants["SamplePart3"] = torch.tensor([[1., 0.]], requires_grad=False)
        self.all_constants["SamplePart5"] = torch.tensor([[1., 0.]], requires_grad=False)
        self.all_constants["SamplePart7"] = torch.tensor([[1., 0.]], requires_grad=False)
        self.all_constants["SamplePart9"] = torch.tensor([[1., 0.]], requires_grad=False)
        self.all_constants["SamplePart94"] = torch.tensor([[1.]], requires_grad=False)
        self.all_constants["SamplePart96"] = torch.tensor([[1., 0.]], requires_grad=False)
        self.all_constants["SamplePart98"] = torch.tensor([[1., 0.]], requires_grad=False)
        self.all_constants["Select114"] = torch.tensor([1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.], requires_grad=False)
        self.all_constants["Select116"] = torch.tensor([0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.], requires_grad=False)
        self.all_constants["Select119"] = torch.tensor([0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.], requires_grad=False)
        self.all_constants["Select122"] = torch.tensor([0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.], requires_grad=False)
        self.all_constants["Select123"] = torch.tensor([0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.], requires_grad=False)
        self.all_constants["Select126"] = torch.tensor([0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.], requires_grad=False)
        self.all_constants["Select127"] = torch.tensor([0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.], requires_grad=False)
        self.all_constants["Select130"] = torch.tensor([0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.], requires_grad=False)
        self.all_constants["Select131"] = torch.tensor([0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.], requires_grad=False)
        self.all_constants["Select134"] = torch.tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.], requires_grad=False)
        self.all_constants["Select138"] = torch.tensor([1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.], requires_grad=False)
        self.all_constants["Select140"] = torch.tensor([0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.], requires_grad=False)
        self.all_constants["Select143"] = torch.tensor([0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.], requires_grad=False)
        self.all_constants["Select146"] = torch.tensor([0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.], requires_grad=False)
        self.all_constants["Select147"] = torch.tensor([0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.], requires_grad=False)
        self.all_constants["Select150"] = torch.tensor([0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.], requires_grad=False)
        self.all_constants["Select151"] = torch.tensor([0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.], requires_grad=False)
        self.all_constants["Select154"] = torch.tensor([0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.], requires_grad=False)
        self.all_constants["Select155"] = torch.tensor([0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.], requires_grad=False)
        self.all_constants["Select158"] = torch.tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.], requires_grad=False)
        self.all_constants["Select162"] = torch.tensor([1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.], requires_grad=False)
        self.all_constants["Select164"] = torch.tensor([0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.], requires_grad=False)
        self.all_constants["Select167"] = torch.tensor([0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.], requires_grad=False)
        self.all_constants["Select170"] = torch.tensor([0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.], requires_grad=False)
        self.all_constants["Select171"] = torch.tensor([0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.], requires_grad=False)
        self.all_constants["Select174"] = torch.tensor([0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.], requires_grad=False)
        self.all_constants["Select175"] = torch.tensor([0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.], requires_grad=False)
        self.all_constants["Select178"] = torch.tensor([0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.], requires_grad=False)
        self.all_constants["Select179"] = torch.tensor([0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.], requires_grad=False)
        self.all_constants["Select182"] = torch.tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.], requires_grad=False)
        self.all_constants["Select186"] = torch.tensor([1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.], requires_grad=False)
        self.all_constants["Select188"] = torch.tensor([0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.], requires_grad=False)
        self.all_constants["Select191"] = torch.tensor([0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.], requires_grad=False)
        self.all_constants["Select194"] = torch.tensor([0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.], requires_grad=False)
        self.all_constants["Select195"] = torch.tensor([0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.], requires_grad=False)
        self.all_constants["Select198"] = torch.tensor([0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.], requires_grad=False)
        self.all_constants["Select199"] = torch.tensor([0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.], requires_grad=False)
        self.all_constants["Select202"] = torch.tensor([0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.], requires_grad=False)
        self.all_constants["Select203"] = torch.tensor([0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.], requires_grad=False)
        self.all_constants["Select206"] = torch.tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.], requires_grad=False)
        self.all_constants["Select21"] = torch.tensor([1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.], requires_grad=False)
        self.all_constants["Select210"] = torch.tensor([1., 0., 0.], requires_grad=False)
        self.all_constants["Select214"] = torch.tensor([0., 1., 0.], requires_grad=False)
        self.all_constants["Select218"] = torch.tensor([0., 0., 1.], requires_grad=False)
        self.all_constants["Select23"] = torch.tensor([0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.], requires_grad=False)
        self.all_constants["Select26"] = torch.tensor([0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.], requires_grad=False)
        self.all_constants["Select29"] = torch.tensor([0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.], requires_grad=False)
        self.all_constants["Select30"] = torch.tensor([0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.], requires_grad=False)
        self.all_constants["Select33"] = torch.tensor([0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.], requires_grad=False)
        self.all_constants["Select34"] = torch.tensor([0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.], requires_grad=False)
        self.all_constants["Select37"] = torch.tensor([0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.], requires_grad=False)
        self.all_constants["Select38"] = torch.tensor([0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.], requires_grad=False)
        self.all_constants["Select41"] = torch.tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.], requires_grad=False)
        self.all_constants["Select45"] = torch.tensor([1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.], requires_grad=False)
        self.all_constants["Select47"] = torch.tensor([0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.], requires_grad=False)
        self.all_constants["Select50"] = torch.tensor([0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.], requires_grad=False)
        self.all_constants["Select53"] = torch.tensor([0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.], requires_grad=False)
        self.all_constants["Select54"] = torch.tensor([0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.], requires_grad=False)
        self.all_constants["Select57"] = torch.tensor([0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.], requires_grad=False)
        self.all_constants["Select58"] = torch.tensor([0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.], requires_grad=False)
        self.all_constants["Select61"] = torch.tensor([0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.], requires_grad=False)
        self.all_constants["Select62"] = torch.tensor([0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.], requires_grad=False)
        self.all_constants["Select65"] = torch.tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.], requires_grad=False)
        self.all_constants["Select69"] = torch.tensor([1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.], requires_grad=False)
        self.all_constants["Select71"] = torch.tensor([0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.], requires_grad=False)
        self.all_constants["Select74"] = torch.tensor([0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.], requires_grad=False)
        self.all_constants["Select77"] = torch.tensor([0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.], requires_grad=False)
        self.all_constants["Select78"] = torch.tensor([0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.], requires_grad=False)
        self.all_constants["Select81"] = torch.tensor([0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.], requires_grad=False)
        self.all_constants["Select82"] = torch.tensor([0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.], requires_grad=False)
        self.all_constants["Select85"] = torch.tensor([0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.], requires_grad=False)
        self.all_constants["Select86"] = torch.tensor([0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.], requires_grad=False)
        self.all_constants["Select89"] = torch.tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.], requires_grad=False)
        self.all_parameters = torch.nn.ParameterDict(self.all_parameters)
        self.all_constants = torch.nn.ParameterDict(self.all_constants)

    def update(self, closed_loop={}, connect={}, disconnect=False):
        pass
    def forward(self, kwargs):
        getitem = kwargs['Xth2_dot']
        relation_forward_sample_part13_w = self.all_constants.SamplePart13
        einsum = torch.functional.einsum('bij,ki->bkj', getitem, relation_forward_sample_part13_w);  getitem = relation_forward_sample_part13_w = None
        getitem_1 = kwargs['Xth2']
        relation_forward_sample_part11_w = self.all_constants.SamplePart11
        einsum_1 = torch.functional.einsum('bij,ki->bkj', getitem_1, relation_forward_sample_part11_w);  getitem_1 = relation_forward_sample_part11_w = None
        getitem_2 = kwargs['Xth1_dot']
        relation_forward_sample_part9_w = self.all_constants.SamplePart9
        einsum_2 = torch.functional.einsum('bij,ki->bkj', getitem_2, relation_forward_sample_part9_w);  getitem_2 = relation_forward_sample_part9_w = None
        getitem_3 = kwargs['Xth1']
        relation_forward_sample_part7_w = self.all_constants.SamplePart7
        einsum_3 = torch.functional.einsum('bij,ki->bkj', getitem_3, relation_forward_sample_part7_w);  getitem_3 = relation_forward_sample_part7_w = None
        getitem_4 = kwargs['Xvelocity']
        relation_forward_sample_part5_w = self.all_constants.SamplePart5
        einsum_4 = torch.functional.einsum('bij,ki->bkj', getitem_4, relation_forward_sample_part5_w);  getitem_4 = relation_forward_sample_part5_w = None
        getitem_5 = kwargs['Xpos']
        relation_forward_sample_part3_w = self.all_constants.SamplePart3
        einsum_5 = torch.functional.einsum('bij,ki->bkj', getitem_5, relation_forward_sample_part3_w);  getitem_5 = relation_forward_sample_part3_w = None
        getitem_6 = kwargs['action']
        relation_forward_sample_part1_w = self.all_constants.SamplePart1
        einsum_6 = torch.functional.einsum('bij,ki->bkj', getitem_6, relation_forward_sample_part1_w);  getitem_6 = relation_forward_sample_part1_w = None
        cat = torch.cat((einsum_6, einsum_5), dim = 2);  einsum_6 = einsum_5 = None
        cat_1 = torch.cat((cat, einsum_4), dim = 2);  cat = einsum_4 = None
        cat_2 = torch.cat((cat_1, einsum_3), dim = 2);  cat_1 = einsum_3 = None
        cat_3 = torch.cat((cat_2, einsum_2), dim = 2);  cat_2 = einsum_2 = None
        cat_4 = torch.cat((cat_3, einsum_1), dim = 2);  cat_3 = einsum_1 = None
        cat_5 = torch.cat((cat_4, einsum), dim = 2);  cat_4 = einsum = None
        relation_forward_linear113_weights = self.all_parameters.PLinear11W
        einsum_7 = torch.functional.einsum('bwi,io->bwo', cat_5, relation_forward_linear113_weights);  cat_5 = None
        relation_forward_select41_w = self.all_constants.Select41
        einsum_8 = torch.functional.einsum('ijk,k->ij', einsum_7, relation_forward_select41_w);  relation_forward_select41_w = None
        unsqueeze = einsum_8.unsqueeze(2);  einsum_8 = None
        pow2 = nnodely_layers_parametricfunction_Pow2(unsqueeze);  unsqueeze = None
        relation_forward_select38_w = self.all_constants.Select38
        einsum_9 = torch.functional.einsum('ijk,k->ij', einsum_7, relation_forward_select38_w);  relation_forward_select38_w = None
        unsqueeze_1 = einsum_9.unsqueeze(2);  einsum_9 = None
        relation_forward_select37_w = self.all_constants.Select37
        einsum_10 = torch.functional.einsum('ijk,k->ij', einsum_7, relation_forward_select37_w);  relation_forward_select37_w = None
        unsqueeze_2 = einsum_10.unsqueeze(2);  einsum_10 = None
        sub = unsqueeze_2 - unsqueeze_1;  unsqueeze_2 = unsqueeze_1 = None
        relation_forward_select34_w = self.all_constants.Select34
        einsum_11 = torch.functional.einsum('ijk,k->ij', einsum_7, relation_forward_select34_w);  relation_forward_select34_w = None
        unsqueeze_3 = einsum_11.unsqueeze(2);  einsum_11 = None
        relation_forward_select33_w = self.all_constants.Select33
        einsum_12 = torch.functional.einsum('ijk,k->ij', einsum_7, relation_forward_select33_w);  relation_forward_select33_w = None
        unsqueeze_4 = einsum_12.unsqueeze(2);  einsum_12 = None
        add = unsqueeze_4 + unsqueeze_3;  unsqueeze_4 = unsqueeze_3 = None
        relation_forward_select30_w = self.all_constants.Select30
        einsum_13 = torch.functional.einsum('ijk,k->ij', einsum_7, relation_forward_select30_w);  relation_forward_select30_w = None
        unsqueeze_5 = einsum_13.unsqueeze(2);  einsum_13 = None
        relation_forward_select29_w = self.all_constants.Select29
        einsum_14 = torch.functional.einsum('ijk,k->ij', einsum_7, relation_forward_select29_w);  relation_forward_select29_w = None
        unsqueeze_6 = einsum_14.unsqueeze(2);  einsum_14 = None
        mul = unsqueeze_6 * unsqueeze_5;  unsqueeze_6 = unsqueeze_5 = None
        relation_forward_select26_w = self.all_constants.Select26
        einsum_15 = torch.functional.einsum('ijk,k->ij', einsum_7, relation_forward_select26_w);  relation_forward_select26_w = None
        unsqueeze_7 = einsum_15.unsqueeze(2);  einsum_15 = None
        relation_forward_select23_w = self.all_constants.Select23
        einsum_16 = torch.functional.einsum('ijk,k->ij', einsum_7, relation_forward_select23_w);  relation_forward_select23_w = None
        unsqueeze_8 = einsum_16.unsqueeze(2);  einsum_16 = None
        cos = torch.cos(unsqueeze_8);  unsqueeze_8 = None
        relation_forward_select21_w = self.all_constants.Select21
        einsum_17 = torch.functional.einsum('ijk,k->ij', einsum_7, relation_forward_select21_w);  einsum_7 = relation_forward_select21_w = None
        unsqueeze_9 = einsum_17.unsqueeze(2);  einsum_17 = None
        sin = torch.sin(unsqueeze_9);  unsqueeze_9 = None
        cat_6 = torch.cat((sin, cos), dim = 2);  sin = cos = None
        cat_7 = torch.cat((cat_6, unsqueeze_7), dim = 2);  cat_6 = unsqueeze_7 = None
        cat_8 = torch.cat((cat_7, mul), dim = 2);  cat_7 = mul = None
        cat_9 = torch.cat((cat_8, add), dim = 2);  cat_8 = add = None
        cat_10 = torch.cat((cat_9, sub), dim = 2);  cat_9 = sub = None
        cat_11 = torch.cat((cat_10, pow2), dim = 2);  cat_10 = pow2 = None
        relation_forward_linear137_weights = self.all_parameters.PLinear13W
        einsum_18 = torch.functional.einsum('bwi,io->bwo', cat_11, relation_forward_linear137_weights);  cat_11 = None
        relation_forward_select65_w = self.all_constants.Select65
        einsum_19 = torch.functional.einsum('ijk,k->ij', einsum_18, relation_forward_select65_w);  relation_forward_select65_w = None
        unsqueeze_10 = einsum_19.unsqueeze(2);  einsum_19 = None
        pow2_1 = nnodely_layers_parametricfunction_Pow2(unsqueeze_10);  unsqueeze_10 = None
        relation_forward_select62_w = self.all_constants.Select62
        einsum_20 = torch.functional.einsum('ijk,k->ij', einsum_18, relation_forward_select62_w);  relation_forward_select62_w = None
        unsqueeze_11 = einsum_20.unsqueeze(2);  einsum_20 = None
        relation_forward_select61_w = self.all_constants.Select61
        einsum_21 = torch.functional.einsum('ijk,k->ij', einsum_18, relation_forward_select61_w);  relation_forward_select61_w = None
        unsqueeze_12 = einsum_21.unsqueeze(2);  einsum_21 = None
        sub_1 = unsqueeze_12 - unsqueeze_11;  unsqueeze_12 = unsqueeze_11 = None
        relation_forward_select58_w = self.all_constants.Select58
        einsum_22 = torch.functional.einsum('ijk,k->ij', einsum_18, relation_forward_select58_w);  relation_forward_select58_w = None
        unsqueeze_13 = einsum_22.unsqueeze(2);  einsum_22 = None
        relation_forward_select57_w = self.all_constants.Select57
        einsum_23 = torch.functional.einsum('ijk,k->ij', einsum_18, relation_forward_select57_w);  relation_forward_select57_w = None
        unsqueeze_14 = einsum_23.unsqueeze(2);  einsum_23 = None
        add_1 = unsqueeze_14 + unsqueeze_13;  unsqueeze_14 = unsqueeze_13 = None
        relation_forward_select54_w = self.all_constants.Select54
        einsum_24 = torch.functional.einsum('ijk,k->ij', einsum_18, relation_forward_select54_w);  relation_forward_select54_w = None
        unsqueeze_15 = einsum_24.unsqueeze(2);  einsum_24 = None
        relation_forward_select53_w = self.all_constants.Select53
        einsum_25 = torch.functional.einsum('ijk,k->ij', einsum_18, relation_forward_select53_w);  relation_forward_select53_w = None
        unsqueeze_16 = einsum_25.unsqueeze(2);  einsum_25 = None
        mul_1 = unsqueeze_16 * unsqueeze_15;  unsqueeze_16 = unsqueeze_15 = None
        relation_forward_select50_w = self.all_constants.Select50
        einsum_26 = torch.functional.einsum('ijk,k->ij', einsum_18, relation_forward_select50_w);  relation_forward_select50_w = None
        unsqueeze_17 = einsum_26.unsqueeze(2);  einsum_26 = None
        relation_forward_select47_w = self.all_constants.Select47
        einsum_27 = torch.functional.einsum('ijk,k->ij', einsum_18, relation_forward_select47_w);  relation_forward_select47_w = None
        unsqueeze_18 = einsum_27.unsqueeze(2);  einsum_27 = None
        cos_1 = torch.cos(unsqueeze_18);  unsqueeze_18 = None
        relation_forward_select45_w = self.all_constants.Select45
        einsum_28 = torch.functional.einsum('ijk,k->ij', einsum_18, relation_forward_select45_w);  einsum_18 = relation_forward_select45_w = None
        unsqueeze_19 = einsum_28.unsqueeze(2);  einsum_28 = None
        sin_1 = torch.sin(unsqueeze_19);  unsqueeze_19 = None
        cat_12 = torch.cat((sin_1, cos_1), dim = 2);  sin_1 = cos_1 = None
        cat_13 = torch.cat((cat_12, unsqueeze_17), dim = 2);  cat_12 = unsqueeze_17 = None
        cat_14 = torch.cat((cat_13, mul_1), dim = 2);  cat_13 = mul_1 = None
        cat_15 = torch.cat((cat_14, add_1), dim = 2);  cat_14 = add_1 = None
        cat_16 = torch.cat((cat_15, sub_1), dim = 2);  cat_15 = sub_1 = None
        cat_17 = torch.cat((cat_16, pow2_1), dim = 2);  cat_16 = pow2_1 = None
        relation_forward_linear161_weights = self.all_parameters.PLinear15W
        einsum_29 = torch.functional.einsum('bwi,io->bwo', cat_17, relation_forward_linear161_weights);  cat_17 = None
        relation_forward_select89_w = self.all_constants.Select89
        einsum_30 = torch.functional.einsum('ijk,k->ij', einsum_29, relation_forward_select89_w);  relation_forward_select89_w = None
        unsqueeze_20 = einsum_30.unsqueeze(2);  einsum_30 = None
        pow2_2 = nnodely_layers_parametricfunction_Pow2(unsqueeze_20);  unsqueeze_20 = None
        relation_forward_select86_w = self.all_constants.Select86
        einsum_31 = torch.functional.einsum('ijk,k->ij', einsum_29, relation_forward_select86_w);  relation_forward_select86_w = None
        unsqueeze_21 = einsum_31.unsqueeze(2);  einsum_31 = None
        relation_forward_select85_w = self.all_constants.Select85
        einsum_32 = torch.functional.einsum('ijk,k->ij', einsum_29, relation_forward_select85_w);  relation_forward_select85_w = None
        unsqueeze_22 = einsum_32.unsqueeze(2);  einsum_32 = None
        sub_2 = unsqueeze_22 - unsqueeze_21;  unsqueeze_22 = unsqueeze_21 = None
        relation_forward_select82_w = self.all_constants.Select82
        einsum_33 = torch.functional.einsum('ijk,k->ij', einsum_29, relation_forward_select82_w);  relation_forward_select82_w = None
        unsqueeze_23 = einsum_33.unsqueeze(2);  einsum_33 = None
        relation_forward_select81_w = self.all_constants.Select81
        einsum_34 = torch.functional.einsum('ijk,k->ij', einsum_29, relation_forward_select81_w);  relation_forward_select81_w = None
        unsqueeze_24 = einsum_34.unsqueeze(2);  einsum_34 = None
        add_2 = unsqueeze_24 + unsqueeze_23;  unsqueeze_24 = unsqueeze_23 = None
        relation_forward_select78_w = self.all_constants.Select78
        einsum_35 = torch.functional.einsum('ijk,k->ij', einsum_29, relation_forward_select78_w);  relation_forward_select78_w = None
        unsqueeze_25 = einsum_35.unsqueeze(2);  einsum_35 = None
        relation_forward_select77_w = self.all_constants.Select77
        einsum_36 = torch.functional.einsum('ijk,k->ij', einsum_29, relation_forward_select77_w);  relation_forward_select77_w = None
        unsqueeze_26 = einsum_36.unsqueeze(2);  einsum_36 = None
        mul_2 = unsqueeze_26 * unsqueeze_25;  unsqueeze_26 = unsqueeze_25 = None
        relation_forward_select74_w = self.all_constants.Select74
        einsum_37 = torch.functional.einsum('ijk,k->ij', einsum_29, relation_forward_select74_w);  relation_forward_select74_w = None
        unsqueeze_27 = einsum_37.unsqueeze(2);  einsum_37 = None
        relation_forward_select71_w = self.all_constants.Select71
        einsum_38 = torch.functional.einsum('ijk,k->ij', einsum_29, relation_forward_select71_w);  relation_forward_select71_w = None
        unsqueeze_28 = einsum_38.unsqueeze(2);  einsum_38 = None
        cos_2 = torch.cos(unsqueeze_28);  unsqueeze_28 = None
        relation_forward_select69_w = self.all_constants.Select69
        einsum_39 = torch.functional.einsum('ijk,k->ij', einsum_29, relation_forward_select69_w);  einsum_29 = relation_forward_select69_w = None
        unsqueeze_29 = einsum_39.unsqueeze(2);  einsum_39 = None
        sin_2 = torch.sin(unsqueeze_29);  unsqueeze_29 = None
        cat_18 = torch.cat((sin_2, cos_2), dim = 2);  sin_2 = cos_2 = None
        cat_19 = torch.cat((cat_18, unsqueeze_27), dim = 2);  cat_18 = unsqueeze_27 = None
        cat_20 = torch.cat((cat_19, mul_2), dim = 2);  cat_19 = mul_2 = None
        cat_21 = torch.cat((cat_20, add_2), dim = 2);  cat_20 = add_2 = None
        cat_22 = torch.cat((cat_21, sub_2), dim = 2);  cat_21 = sub_2 = None
        cat_23 = torch.cat((cat_22, pow2_2), dim = 2);  cat_22 = pow2_2 = None
        relation_forward_linear92_weights = self.all_parameters.PLinear21W
        einsum_40 = torch.functional.einsum('bwi,io->bwo', cat_23, relation_forward_linear92_weights);  cat_23 = relation_forward_linear92_weights = None
        all_constants_constant33 = self.all_constants.Constant33
        add_3 = einsum_40 + all_constants_constant33;  all_constants_constant33 = None
        getitem_7 = kwargs['Xth2_dot']
        relation_forward_sample_part106_w = self.all_constants.SamplePart106
        einsum_41 = torch.functional.einsum('bij,ki->bkj', getitem_7, relation_forward_sample_part106_w);  getitem_7 = relation_forward_sample_part106_w = None
        getitem_8 = kwargs['Xth2']
        relation_forward_sample_part104_w = self.all_constants.SamplePart104
        einsum_42 = torch.functional.einsum('bij,ki->bkj', getitem_8, relation_forward_sample_part104_w);  getitem_8 = relation_forward_sample_part104_w = None
        getitem_9 = kwargs['Xth1_dot']
        relation_forward_sample_part102_w = self.all_constants.SamplePart102
        einsum_43 = torch.functional.einsum('bij,ki->bkj', getitem_9, relation_forward_sample_part102_w);  getitem_9 = relation_forward_sample_part102_w = None
        getitem_10 = kwargs['Xth1']
        relation_forward_sample_part100_w = self.all_constants.SamplePart100
        einsum_44 = torch.functional.einsum('bij,ki->bkj', getitem_10, relation_forward_sample_part100_w);  getitem_10 = relation_forward_sample_part100_w = None
        getitem_11 = kwargs['Xvelocity']
        relation_forward_sample_part98_w = self.all_constants.SamplePart98
        einsum_45 = torch.functional.einsum('bij,ki->bkj', getitem_11, relation_forward_sample_part98_w);  getitem_11 = relation_forward_sample_part98_w = None
        getitem_12 = kwargs['Xpos']
        relation_forward_sample_part96_w = self.all_constants.SamplePart96
        einsum_46 = torch.functional.einsum('bij,ki->bkj', getitem_12, relation_forward_sample_part96_w);  getitem_12 = relation_forward_sample_part96_w = None
        getitem_13 = kwargs['action']
        relation_forward_sample_part94_w = self.all_constants.SamplePart94
        einsum_47 = torch.functional.einsum('bij,ki->bkj', getitem_13, relation_forward_sample_part94_w);  getitem_13 = relation_forward_sample_part94_w = None
        cat_24 = torch.cat((einsum_47, einsum_46), dim = 2);  einsum_47 = einsum_46 = None
        cat_25 = torch.cat((cat_24, einsum_45), dim = 2);  cat_24 = einsum_45 = None
        cat_26 = torch.cat((cat_25, einsum_44), dim = 2);  cat_25 = einsum_44 = None
        cat_27 = torch.cat((cat_26, einsum_43), dim = 2);  cat_26 = einsum_43 = None
        cat_28 = torch.cat((cat_27, einsum_42), dim = 2);  cat_27 = einsum_42 = None
        cat_29 = torch.cat((cat_28, einsum_41), dim = 2);  cat_28 = einsum_41 = None
        einsum_48 = torch.functional.einsum('bwi,io->bwo', cat_29, relation_forward_linear113_weights);  cat_29 = relation_forward_linear113_weights = None
        relation_forward_select134_w = self.all_constants.Select134
        einsum_49 = torch.functional.einsum('ijk,k->ij', einsum_48, relation_forward_select134_w);  relation_forward_select134_w = None
        unsqueeze_30 = einsum_49.unsqueeze(2);  einsum_49 = None
        pow2_3 = nnodely_layers_parametricfunction_Pow2(unsqueeze_30);  unsqueeze_30 = None
        relation_forward_select131_w = self.all_constants.Select131
        einsum_50 = torch.functional.einsum('ijk,k->ij', einsum_48, relation_forward_select131_w);  relation_forward_select131_w = None
        unsqueeze_31 = einsum_50.unsqueeze(2);  einsum_50 = None
        relation_forward_select130_w = self.all_constants.Select130
        einsum_51 = torch.functional.einsum('ijk,k->ij', einsum_48, relation_forward_select130_w);  relation_forward_select130_w = None
        unsqueeze_32 = einsum_51.unsqueeze(2);  einsum_51 = None
        sub_3 = unsqueeze_32 - unsqueeze_31;  unsqueeze_32 = unsqueeze_31 = None
        relation_forward_select127_w = self.all_constants.Select127
        einsum_52 = torch.functional.einsum('ijk,k->ij', einsum_48, relation_forward_select127_w);  relation_forward_select127_w = None
        unsqueeze_33 = einsum_52.unsqueeze(2);  einsum_52 = None
        relation_forward_select126_w = self.all_constants.Select126
        einsum_53 = torch.functional.einsum('ijk,k->ij', einsum_48, relation_forward_select126_w);  relation_forward_select126_w = None
        unsqueeze_34 = einsum_53.unsqueeze(2);  einsum_53 = None
        add_4 = unsqueeze_34 + unsqueeze_33;  unsqueeze_34 = unsqueeze_33 = None
        relation_forward_select123_w = self.all_constants.Select123
        einsum_54 = torch.functional.einsum('ijk,k->ij', einsum_48, relation_forward_select123_w);  relation_forward_select123_w = None
        unsqueeze_35 = einsum_54.unsqueeze(2);  einsum_54 = None
        relation_forward_select122_w = self.all_constants.Select122
        einsum_55 = torch.functional.einsum('ijk,k->ij', einsum_48, relation_forward_select122_w);  relation_forward_select122_w = None
        unsqueeze_36 = einsum_55.unsqueeze(2);  einsum_55 = None
        mul_3 = unsqueeze_36 * unsqueeze_35;  unsqueeze_36 = unsqueeze_35 = None
        relation_forward_select119_w = self.all_constants.Select119
        einsum_56 = torch.functional.einsum('ijk,k->ij', einsum_48, relation_forward_select119_w);  relation_forward_select119_w = None
        unsqueeze_37 = einsum_56.unsqueeze(2);  einsum_56 = None
        relation_forward_select116_w = self.all_constants.Select116
        einsum_57 = torch.functional.einsum('ijk,k->ij', einsum_48, relation_forward_select116_w);  relation_forward_select116_w = None
        unsqueeze_38 = einsum_57.unsqueeze(2);  einsum_57 = None
        cos_3 = torch.cos(unsqueeze_38);  unsqueeze_38 = None
        relation_forward_select114_w = self.all_constants.Select114
        einsum_58 = torch.functional.einsum('ijk,k->ij', einsum_48, relation_forward_select114_w);  einsum_48 = relation_forward_select114_w = None
        unsqueeze_39 = einsum_58.unsqueeze(2);  einsum_58 = None
        sin_3 = torch.sin(unsqueeze_39);  unsqueeze_39 = None
        cat_30 = torch.cat((sin_3, cos_3), dim = 2);  sin_3 = cos_3 = None
        cat_31 = torch.cat((cat_30, unsqueeze_37), dim = 2);  cat_30 = unsqueeze_37 = None
        cat_32 = torch.cat((cat_31, mul_3), dim = 2);  cat_31 = mul_3 = None
        cat_33 = torch.cat((cat_32, add_4), dim = 2);  cat_32 = add_4 = None
        cat_34 = torch.cat((cat_33, sub_3), dim = 2);  cat_33 = sub_3 = None
        cat_35 = torch.cat((cat_34, pow2_3), dim = 2);  cat_34 = pow2_3 = None
        einsum_59 = torch.functional.einsum('bwi,io->bwo', cat_35, relation_forward_linear137_weights);  cat_35 = relation_forward_linear137_weights = None
        relation_forward_select158_w = self.all_constants.Select158
        einsum_60 = torch.functional.einsum('ijk,k->ij', einsum_59, relation_forward_select158_w);  relation_forward_select158_w = None
        unsqueeze_40 = einsum_60.unsqueeze(2);  einsum_60 = None
        pow2_4 = nnodely_layers_parametricfunction_Pow2(unsqueeze_40);  unsqueeze_40 = None
        relation_forward_select155_w = self.all_constants.Select155
        einsum_61 = torch.functional.einsum('ijk,k->ij', einsum_59, relation_forward_select155_w);  relation_forward_select155_w = None
        unsqueeze_41 = einsum_61.unsqueeze(2);  einsum_61 = None
        relation_forward_select154_w = self.all_constants.Select154
        einsum_62 = torch.functional.einsum('ijk,k->ij', einsum_59, relation_forward_select154_w);  relation_forward_select154_w = None
        unsqueeze_42 = einsum_62.unsqueeze(2);  einsum_62 = None
        sub_4 = unsqueeze_42 - unsqueeze_41;  unsqueeze_42 = unsqueeze_41 = None
        relation_forward_select151_w = self.all_constants.Select151
        einsum_63 = torch.functional.einsum('ijk,k->ij', einsum_59, relation_forward_select151_w);  relation_forward_select151_w = None
        unsqueeze_43 = einsum_63.unsqueeze(2);  einsum_63 = None
        relation_forward_select150_w = self.all_constants.Select150
        einsum_64 = torch.functional.einsum('ijk,k->ij', einsum_59, relation_forward_select150_w);  relation_forward_select150_w = None
        unsqueeze_44 = einsum_64.unsqueeze(2);  einsum_64 = None
        add_5 = unsqueeze_44 + unsqueeze_43;  unsqueeze_44 = unsqueeze_43 = None
        relation_forward_select147_w = self.all_constants.Select147
        einsum_65 = torch.functional.einsum('ijk,k->ij', einsum_59, relation_forward_select147_w);  relation_forward_select147_w = None
        unsqueeze_45 = einsum_65.unsqueeze(2);  einsum_65 = None
        relation_forward_select146_w = self.all_constants.Select146
        einsum_66 = torch.functional.einsum('ijk,k->ij', einsum_59, relation_forward_select146_w);  relation_forward_select146_w = None
        unsqueeze_46 = einsum_66.unsqueeze(2);  einsum_66 = None
        mul_4 = unsqueeze_46 * unsqueeze_45;  unsqueeze_46 = unsqueeze_45 = None
        relation_forward_select143_w = self.all_constants.Select143
        einsum_67 = torch.functional.einsum('ijk,k->ij', einsum_59, relation_forward_select143_w);  relation_forward_select143_w = None
        unsqueeze_47 = einsum_67.unsqueeze(2);  einsum_67 = None
        relation_forward_select140_w = self.all_constants.Select140
        einsum_68 = torch.functional.einsum('ijk,k->ij', einsum_59, relation_forward_select140_w);  relation_forward_select140_w = None
        unsqueeze_48 = einsum_68.unsqueeze(2);  einsum_68 = None
        cos_4 = torch.cos(unsqueeze_48);  unsqueeze_48 = None
        relation_forward_select138_w = self.all_constants.Select138
        einsum_69 = torch.functional.einsum('ijk,k->ij', einsum_59, relation_forward_select138_w);  einsum_59 = relation_forward_select138_w = None
        unsqueeze_49 = einsum_69.unsqueeze(2);  einsum_69 = None
        sin_4 = torch.sin(unsqueeze_49);  unsqueeze_49 = None
        cat_36 = torch.cat((sin_4, cos_4), dim = 2);  sin_4 = cos_4 = None
        cat_37 = torch.cat((cat_36, unsqueeze_47), dim = 2);  cat_36 = unsqueeze_47 = None
        cat_38 = torch.cat((cat_37, mul_4), dim = 2);  cat_37 = mul_4 = None
        cat_39 = torch.cat((cat_38, add_5), dim = 2);  cat_38 = add_5 = None
        cat_40 = torch.cat((cat_39, sub_4), dim = 2);  cat_39 = sub_4 = None
        cat_41 = torch.cat((cat_40, pow2_4), dim = 2);  cat_40 = pow2_4 = None
        einsum_70 = torch.functional.einsum('bwi,io->bwo', cat_41, relation_forward_linear161_weights);  cat_41 = relation_forward_linear161_weights = None
        relation_forward_select182_w = self.all_constants.Select182
        einsum_71 = torch.functional.einsum('ijk,k->ij', einsum_70, relation_forward_select182_w);  relation_forward_select182_w = None
        unsqueeze_50 = einsum_71.unsqueeze(2);  einsum_71 = None
        pow2_5 = nnodely_layers_parametricfunction_Pow2(unsqueeze_50);  unsqueeze_50 = None
        relation_forward_select179_w = self.all_constants.Select179
        einsum_72 = torch.functional.einsum('ijk,k->ij', einsum_70, relation_forward_select179_w);  relation_forward_select179_w = None
        unsqueeze_51 = einsum_72.unsqueeze(2);  einsum_72 = None
        relation_forward_select178_w = self.all_constants.Select178
        einsum_73 = torch.functional.einsum('ijk,k->ij', einsum_70, relation_forward_select178_w);  relation_forward_select178_w = None
        unsqueeze_52 = einsum_73.unsqueeze(2);  einsum_73 = None
        sub_5 = unsqueeze_52 - unsqueeze_51;  unsqueeze_52 = unsqueeze_51 = None
        relation_forward_select175_w = self.all_constants.Select175
        einsum_74 = torch.functional.einsum('ijk,k->ij', einsum_70, relation_forward_select175_w);  relation_forward_select175_w = None
        unsqueeze_53 = einsum_74.unsqueeze(2);  einsum_74 = None
        relation_forward_select174_w = self.all_constants.Select174
        einsum_75 = torch.functional.einsum('ijk,k->ij', einsum_70, relation_forward_select174_w);  relation_forward_select174_w = None
        unsqueeze_54 = einsum_75.unsqueeze(2);  einsum_75 = None
        add_6 = unsqueeze_54 + unsqueeze_53;  unsqueeze_54 = unsqueeze_53 = None
        relation_forward_select171_w = self.all_constants.Select171
        einsum_76 = torch.functional.einsum('ijk,k->ij', einsum_70, relation_forward_select171_w);  relation_forward_select171_w = None
        unsqueeze_55 = einsum_76.unsqueeze(2);  einsum_76 = None
        relation_forward_select170_w = self.all_constants.Select170
        einsum_77 = torch.functional.einsum('ijk,k->ij', einsum_70, relation_forward_select170_w);  relation_forward_select170_w = None
        unsqueeze_56 = einsum_77.unsqueeze(2);  einsum_77 = None
        mul_5 = unsqueeze_56 * unsqueeze_55;  unsqueeze_56 = unsqueeze_55 = None
        relation_forward_select167_w = self.all_constants.Select167
        einsum_78 = torch.functional.einsum('ijk,k->ij', einsum_70, relation_forward_select167_w);  relation_forward_select167_w = None
        unsqueeze_57 = einsum_78.unsqueeze(2);  einsum_78 = None
        relation_forward_select164_w = self.all_constants.Select164
        einsum_79 = torch.functional.einsum('ijk,k->ij', einsum_70, relation_forward_select164_w);  relation_forward_select164_w = None
        unsqueeze_58 = einsum_79.unsqueeze(2);  einsum_79 = None
        cos_5 = torch.cos(unsqueeze_58);  unsqueeze_58 = None
        relation_forward_select162_w = self.all_constants.Select162
        einsum_80 = torch.functional.einsum('ijk,k->ij', einsum_70, relation_forward_select162_w);  einsum_70 = relation_forward_select162_w = None
        unsqueeze_59 = einsum_80.unsqueeze(2);  einsum_80 = None
        sin_5 = torch.sin(unsqueeze_59);  unsqueeze_59 = None
        cat_42 = torch.cat((sin_5, cos_5), dim = 2);  sin_5 = cos_5 = None
        cat_43 = torch.cat((cat_42, unsqueeze_57), dim = 2);  cat_42 = unsqueeze_57 = None
        cat_44 = torch.cat((cat_43, mul_5), dim = 2);  cat_43 = mul_5 = None
        cat_45 = torch.cat((cat_44, add_6), dim = 2);  cat_44 = add_6 = None
        cat_46 = torch.cat((cat_45, sub_5), dim = 2);  cat_45 = sub_5 = None
        cat_47 = torch.cat((cat_46, pow2_5), dim = 2);  cat_46 = pow2_5 = None
        relation_forward_linear185_weights = self.all_parameters.PLinear17W
        einsum_81 = torch.functional.einsum('bwi,io->bwo', cat_47, relation_forward_linear185_weights);  cat_47 = relation_forward_linear185_weights = None
        relation_forward_select206_w = self.all_constants.Select206
        einsum_82 = torch.functional.einsum('ijk,k->ij', einsum_81, relation_forward_select206_w);  relation_forward_select206_w = None
        unsqueeze_60 = einsum_82.unsqueeze(2);  einsum_82 = None
        pow2_6 = nnodely_layers_parametricfunction_Pow2(unsqueeze_60);  unsqueeze_60 = None
        relation_forward_select203_w = self.all_constants.Select203
        einsum_83 = torch.functional.einsum('ijk,k->ij', einsum_81, relation_forward_select203_w);  relation_forward_select203_w = None
        unsqueeze_61 = einsum_83.unsqueeze(2);  einsum_83 = None
        relation_forward_select202_w = self.all_constants.Select202
        einsum_84 = torch.functional.einsum('ijk,k->ij', einsum_81, relation_forward_select202_w);  relation_forward_select202_w = None
        unsqueeze_62 = einsum_84.unsqueeze(2);  einsum_84 = None
        sub_6 = unsqueeze_62 - unsqueeze_61;  unsqueeze_62 = unsqueeze_61 = None
        relation_forward_select199_w = self.all_constants.Select199
        einsum_85 = torch.functional.einsum('ijk,k->ij', einsum_81, relation_forward_select199_w);  relation_forward_select199_w = None
        unsqueeze_63 = einsum_85.unsqueeze(2);  einsum_85 = None
        relation_forward_select198_w = self.all_constants.Select198
        einsum_86 = torch.functional.einsum('ijk,k->ij', einsum_81, relation_forward_select198_w);  relation_forward_select198_w = None
        unsqueeze_64 = einsum_86.unsqueeze(2);  einsum_86 = None
        add_7 = unsqueeze_64 + unsqueeze_63;  unsqueeze_64 = unsqueeze_63 = None
        relation_forward_select195_w = self.all_constants.Select195
        einsum_87 = torch.functional.einsum('ijk,k->ij', einsum_81, relation_forward_select195_w);  relation_forward_select195_w = None
        unsqueeze_65 = einsum_87.unsqueeze(2);  einsum_87 = None
        relation_forward_select194_w = self.all_constants.Select194
        einsum_88 = torch.functional.einsum('ijk,k->ij', einsum_81, relation_forward_select194_w);  relation_forward_select194_w = None
        unsqueeze_66 = einsum_88.unsqueeze(2);  einsum_88 = None
        mul_6 = unsqueeze_66 * unsqueeze_65;  unsqueeze_66 = unsqueeze_65 = None
        relation_forward_select191_w = self.all_constants.Select191
        einsum_89 = torch.functional.einsum('ijk,k->ij', einsum_81, relation_forward_select191_w);  relation_forward_select191_w = None
        unsqueeze_67 = einsum_89.unsqueeze(2);  einsum_89 = None
        relation_forward_select188_w = self.all_constants.Select188
        einsum_90 = torch.functional.einsum('ijk,k->ij', einsum_81, relation_forward_select188_w);  relation_forward_select188_w = None
        unsqueeze_68 = einsum_90.unsqueeze(2);  einsum_90 = None
        cos_6 = torch.cos(unsqueeze_68);  unsqueeze_68 = None
        relation_forward_select186_w = self.all_constants.Select186
        einsum_91 = torch.functional.einsum('ijk,k->ij', einsum_81, relation_forward_select186_w);  einsum_81 = relation_forward_select186_w = None
        unsqueeze_69 = einsum_91.unsqueeze(2);  einsum_91 = None
        sin_6 = torch.sin(unsqueeze_69);  unsqueeze_69 = None
        cat_48 = torch.cat((sin_6, cos_6), dim = 2);  sin_6 = cos_6 = None
        cat_49 = torch.cat((cat_48, unsqueeze_67), dim = 2);  cat_48 = unsqueeze_67 = None
        cat_50 = torch.cat((cat_49, mul_6), dim = 2);  cat_49 = mul_6 = None
        cat_51 = torch.cat((cat_50, add_7), dim = 2);  cat_50 = add_7 = None
        cat_52 = torch.cat((cat_51, sub_6), dim = 2);  cat_51 = sub_6 = None
        cat_53 = torch.cat((cat_52, pow2_6), dim = 2);  cat_52 = pow2_6 = None
        relation_forward_linear209_weights = self.all_parameters.PLinear19W
        einsum_92 = torch.functional.einsum('bwi,io->bwo', cat_53, relation_forward_linear209_weights);  cat_53 = relation_forward_linear209_weights = None
        relation_forward_select218_w = self.all_constants.Select218
        einsum_93 = torch.functional.einsum('ijk,k->ij', einsum_92, relation_forward_select218_w);  relation_forward_select218_w = None
        unsqueeze_70 = einsum_93.unsqueeze(2);  einsum_93 = None
        truediv = unsqueeze_70 / add_3;  unsqueeze_70 = add_3 = None
        all_constants_sample_time = self.all_constants.SampleTime
        mul_7 = truediv * all_constants_sample_time
        getitem_14 = kwargs['int_th2_dot']
        relation_forward_sample_part235_w = self.all_constants.SamplePart235
        einsum_94 = torch.functional.einsum('bij,ki->bkj', getitem_14, relation_forward_sample_part235_w);  getitem_14 = relation_forward_sample_part235_w = None
        add_8 = einsum_94 + mul_7;  einsum_94 = mul_7 = None
        mul_8 = add_8 * all_constants_sample_time
        getitem_15 = kwargs['int_th2']
        relation_forward_sample_part253_w = self.all_constants.SamplePart253
        einsum_95 = torch.functional.einsum('bij,ki->bkj', getitem_15, relation_forward_sample_part253_w);  getitem_15 = relation_forward_sample_part253_w = None
        add_9 = einsum_95 + mul_8;  einsum_95 = mul_8 = None
        getitem_16 = kwargs['Xth2']
        relation_forward_sample_part275_w = self.all_constants.SamplePart275
        einsum_96 = torch.functional.einsum('bij,ki->bkj', getitem_16, relation_forward_sample_part275_w);  getitem_16 = relation_forward_sample_part275_w = None
        getitem_17 = kwargs['Xth2_dot']
        relation_forward_sample_part273_w = self.all_constants.SamplePart273
        einsum_97 = torch.functional.einsum('bij,ki->bkj', getitem_17, relation_forward_sample_part273_w);  getitem_17 = relation_forward_sample_part273_w = None
        all_constants_constant32 = self.all_constants.Constant32
        add_10 = einsum_40 + all_constants_constant32;  all_constants_constant32 = None
        relation_forward_select214_w = self.all_constants.Select214
        einsum_98 = torch.functional.einsum('ijk,k->ij', einsum_92, relation_forward_select214_w);  relation_forward_select214_w = None
        unsqueeze_71 = einsum_98.unsqueeze(2);  einsum_98 = None
        truediv_1 = unsqueeze_71 / add_10;  unsqueeze_71 = add_10 = None
        mul_9 = truediv_1 * all_constants_sample_time
        getitem_18 = kwargs['int_th1_dot']
        relation_forward_sample_part229_w = self.all_constants.SamplePart229
        einsum_99 = torch.functional.einsum('bij,ki->bkj', getitem_18, relation_forward_sample_part229_w);  getitem_18 = relation_forward_sample_part229_w = None
        add_11 = einsum_99 + mul_9;  einsum_99 = mul_9 = None
        mul_10 = add_11 * all_constants_sample_time
        getitem_19 = kwargs['int_th1']
        relation_forward_sample_part247_w = self.all_constants.SamplePart247
        einsum_100 = torch.functional.einsum('bij,ki->bkj', getitem_19, relation_forward_sample_part247_w);  getitem_19 = relation_forward_sample_part247_w = None
        add_12 = einsum_100 + mul_10;  einsum_100 = mul_10 = None
        getitem_20 = kwargs['Xth1']
        relation_forward_sample_part271_w = self.all_constants.SamplePart271
        einsum_101 = torch.functional.einsum('bij,ki->bkj', getitem_20, relation_forward_sample_part271_w);  getitem_20 = relation_forward_sample_part271_w = None
        getitem_21 = kwargs['Xth1_dot']
        relation_forward_sample_part269_w = self.all_constants.SamplePart269
        einsum_102 = torch.functional.einsum('bij,ki->bkj', getitem_21, relation_forward_sample_part269_w);  getitem_21 = relation_forward_sample_part269_w = None
        all_constants_constant31 = self.all_constants.Constant31
        add_13 = einsum_40 + all_constants_constant31;  einsum_40 = all_constants_constant31 = None
        relation_forward_select210_w = self.all_constants.Select210
        einsum_103 = torch.functional.einsum('ijk,k->ij', einsum_92, relation_forward_select210_w);  einsum_92 = relation_forward_select210_w = None
        unsqueeze_72 = einsum_103.unsqueeze(2);  einsum_103 = None
        truediv_2 = unsqueeze_72 / add_13;  unsqueeze_72 = add_13 = None
        mul_11 = truediv_2 * all_constants_sample_time
        getitem_22 = kwargs['int_xdot']
        relation_forward_sample_part223_w = self.all_constants.SamplePart223
        einsum_104 = torch.functional.einsum('bij,ki->bkj', getitem_22, relation_forward_sample_part223_w);  getitem_22 = relation_forward_sample_part223_w = None
        add_14 = einsum_104 + mul_11;  einsum_104 = mul_11 = None
        mul_12 = add_14 * all_constants_sample_time;  all_constants_sample_time = None
        getitem_23 = kwargs['int_x']
        relation_forward_sample_part241_w = self.all_constants.SamplePart241
        einsum_105 = torch.functional.einsum('bij,ki->bkj', getitem_23, relation_forward_sample_part241_w);  getitem_23 = relation_forward_sample_part241_w = None
        add_15 = einsum_105 + mul_12;  einsum_105 = mul_12 = None
        getitem_24 = kwargs['Xpos']
        relation_forward_sample_part267_w = self.all_constants.SamplePart267
        einsum_106 = torch.functional.einsum('bij,ki->bkj', getitem_24, relation_forward_sample_part267_w);  getitem_24 = relation_forward_sample_part267_w = None
        getitem_25 = kwargs['Xvelocity']
        relation_forward_sample_part265_w = self.all_constants.SamplePart265
        einsum_107 = torch.functional.einsum('bij,ki->bkj', getitem_25, relation_forward_sample_part265_w);  getitem_25 = relation_forward_sample_part265_w = None
        getitem_26 = kwargs['Xddth2']
        relation_forward_sample_part263_w = self.all_constants.SamplePart263
        einsum_108 = torch.functional.einsum('bij,ki->bkj', getitem_26, relation_forward_sample_part263_w);  getitem_26 = relation_forward_sample_part263_w = None
        getitem_27 = kwargs['Xddth1']
        relation_forward_sample_part261_w = self.all_constants.SamplePart261
        einsum_109 = torch.functional.einsum('bij,ki->bkj', getitem_27, relation_forward_sample_part261_w);  getitem_27 = relation_forward_sample_part261_w = None
        getitem_28 = kwargs['Xddx'];  kwargs = None
        relation_forward_sample_part259_w = self.all_constants.SamplePart259
        einsum_110 = torch.functional.einsum('bij,ki->bkj', getitem_28, relation_forward_sample_part259_w);  getitem_28 = relation_forward_sample_part259_w = None
        return ({'th2_est': add_9, 'omega2_est': add_8, 'th1_est': add_12, 'omega1_est': add_11, 'x_est': add_15, 'xdot_est': add_14, 'th2_ddot_est': truediv, 'th1_ddot_est': truediv_1, 'acc_cart_est': truediv_2}, {'SamplePart259': einsum_110, 'SamplePart261': einsum_109, 'SamplePart263': einsum_108, 'SamplePart265': einsum_107, 'SamplePart267': einsum_106, 'SamplePart269': einsum_102, 'SamplePart271': einsum_101, 'SamplePart273': einsum_97, 'SamplePart275': einsum_96, 'Div213': truediv_2, 'Div217': truediv_1, 'Div221': truediv, 'Add226': add_14, 'Add244': add_15, 'Add232': add_11, 'Add250': add_12, 'Add238': add_8, 'Add256': add_9}, {'Xth2': add_9, 'int_th2': add_9, 'Xth2_dot': add_8, 'int_th2_dot': add_8, 'Xth1_dot': add_11, 'Xth1': add_12, 'Xvelocity': add_14, 'Xpos': add_15, 'int_th1': add_12, 'int_th1_dot': add_11, 'int_x': add_15, 'int_xdot': add_14}, {})
        
class RecurrentModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.Cell = TracerModel()
        self.inputs = ['action', 'Xddth2', 'Xddth1', 'Xddx', ]
        self.states = dict()

    def forward(self, kwargs):
        n_samples = min([kwargs[key].size(0) for key in self.inputs])
        self.states['Xth2'] = kwargs['Xth2']
        self.states['int_th2'] = kwargs['int_th2']
        self.states['Xth2_dot'] = kwargs['Xth2_dot']
        self.states['int_th2_dot'] = kwargs['int_th2_dot']
        self.states['Xth1_dot'] = kwargs['Xth1_dot']
        self.states['Xth1'] = kwargs['Xth1']
        self.states['Xvelocity'] = kwargs['Xvelocity']
        self.states['Xpos'] = kwargs['Xpos']
        self.states['int_th1'] = kwargs['int_th1']
        self.states['int_th1_dot'] = kwargs['int_th1_dot']
        self.states['int_x'] = kwargs['int_x']
        self.states['int_xdot'] = kwargs['int_xdot']
        results = {'th2_est':[], 'omega2_est':[], 'th1_est':[], 'omega1_est':[], 'x_est':[], 'xdot_est':[], 'th2_ddot_est':[], 'th1_ddot_est':[], 'acc_cart_est':[], }
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
                self.states[key] = nnodely_basic_model_connect(self.states[key], val)
        return results
