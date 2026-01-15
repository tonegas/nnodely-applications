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

def nnodely_layers_parametricfunction_understeer_corr_local_control(vx,curv,  # inputs
                                  A,        # constant
                                  ):
    return curv * (1 + A * torch.pow(vx, 2))

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
        self._tensor_constant13 = torch.tensor(0.0)
        self._tensor_constant14 = torch.tensor(4)
        self._tensor_constant15 = torch.tensor(0.0)
        self._tensor_constant16 = torch.tensor(0.0)
        self._tensor_constant17 = torch.tensor(5)
        self._tensor_constant18 = torch.tensor(0.0)
        self._tensor_constant19 = torch.tensor(0.0)
        self._tensor_constant2 = torch.tensor(0)
        self._tensor_constant20 = torch.tensor(6)
        self._tensor_constant21 = torch.tensor(0.0)
        self._tensor_constant22 = torch.tensor(1.0)
        self._tensor_constant23 = torch.tensor(7)
        self._tensor_constant24 = torch.tensor(0.0)
        self._tensor_constant25 = torch.tensor(1.0)
        self._tensor_constant26 = torch.tensor(0)
        self._tensor_constant27 = torch.tensor(0.0)
        self._tensor_constant28 = torch.tensor(0.0)
        self._tensor_constant29 = torch.tensor(1)
        self._tensor_constant3 = torch.tensor(0.0)
        self._tensor_constant30 = torch.tensor(0.0)
        self._tensor_constant31 = torch.tensor(0.0)
        self._tensor_constant32 = torch.tensor(2)
        self._tensor_constant33 = torch.tensor(0.0)
        self._tensor_constant34 = torch.tensor(0.0)
        self._tensor_constant35 = torch.tensor(3)
        self._tensor_constant36 = torch.tensor(0.0)
        self._tensor_constant37 = torch.tensor(1.0)
        self._tensor_constant38 = torch.tensor(4)
        self._tensor_constant4 = torch.tensor(0.0)
        self._tensor_constant5 = torch.tensor(1)
        self._tensor_constant6 = torch.tensor(0.0)
        self._tensor_constant7 = torch.tensor(0.0)
        self._tensor_constant8 = torch.tensor(2)
        self._tensor_constant9 = torch.tensor(0.0)
        self.all_constants["A"] = torch.tensor([0.12241744995117188, 0.13215704262256622, 0.13487723469734192, 0.13838686048984528, 0.14319394528865814], requires_grad=False)
        self.all_constants["W_fir_init"] = torch.tensor([[0.03067079372704029], [0.039163894951343536], [0.04956628382205963], [0.062176525592803955], [0.07730474323034286], [0.09526325762271881], [0.116354800760746], [0.14085842669010162], [0.16901332139968872], [0.20100119709968567], [0.23692776262760162], [0.2768043279647827], [0.3205305337905884], [0.3678794503211975], [0.4184862971305847], [0.4718419909477234], [0.5272924304008484], [0.5840446949005127], [0.6411803960800171], [0.6976763010025024], [0.7524321675300598], [0.8043041825294495], [0.8521437644958496], [0.894839346408844], [0.9313583970069885], [0.9607894420623779], [0.9823793172836304], [0.9955654144287109], [1.0]], requires_grad=False)
        self.all_constants["W_fir_target"] = torch.tensor([[1.0], [0.9955654144287109], [0.9823793172836304], [0.9607894420623779], [0.9313583970069885], [0.894839346408844], [0.8521437644958496], [0.8043041825294495], [0.7524321675300598], [0.6976763010025024], [0.6411803960800171], [0.5840446949005127], [0.5272924304008484], [0.4718419909477234], [0.4184862971305847], [0.3678794503211975], [0.3205305337905884], [0.2768043279647827], [0.23692776262760162], [0.20100119709968567], [0.16901332139968872], [0.14085842669010162], [0.116354800760746], [0.09526325762271881], [0.07730474323034286], [0.062176525592803955], [0.04956628382205963], [0.039163894951343536], [0.03067079372704029], [0.023806948214769363]], requires_grad=False)
        self.all_parameters["Fir_InitCondition"] = torch.nn.Parameter(torch.tensor([[-0.005676846019923687], [-5.078164394944906e-05], [0.007623068057000637], [0.015457557514309883], [0.0211738720536232], [0.026559188961982727], [0.03333593159914017], [0.041086673736572266], [0.04691801965236664], [0.051395390182733536], [0.05676743760704994], [0.061077240854501724], [0.06495276093482971], [0.06868331879377365], [0.07179466634988785], [0.07229167222976685], [0.07215118408203125], [0.07177411019802094], [0.07180652767419815], [0.07127676159143448], [0.0690212994813919], [0.06502213329076767], [0.06130627170205116], [0.05942361429333687], [-0.20321491360664368], [-0.10378878563642502], [-0.2889885902404785], [-0.506409227848053], [-0.10591235756874084]]), requires_grad=True)
        self.all_parameters["Fir_target_0"] = torch.nn.Parameter(torch.tensor([[1.0860881805419922], [1.0301192998886108], [0.9732351303100586], [0.9155109524726868], [0.8570305109024048], [0.7978867888450623], [0.7381616830825806], [0.6779631972312927], [0.6173838973045349], [0.5565256476402283], [0.4954935610294342], [0.43438994884490967], [0.3733194172382355], [0.3123920261859894], [0.2517073452472687], [0.19137363135814667], [0.13149100542068481], [0.07216159254312515], [0.01348485890775919], [-0.044451192021369934], [-0.10155246406793594], [-0.15772926807403564], [-0.21290872991085052], [-0.2670043408870697], [-0.3199438750743866], [-0.3716489374637604], [-0.4220503866672516], [-0.47107553482055664], [-0.5186495184898376], [-0.5646901726722717]]), requires_grad=True)
        self.all_parameters["Fir_target_1"] = torch.nn.Parameter(torch.tensor([[0.8347728252410889], [0.7831745147705078], [0.7308439016342163], [0.6778424382209778], [0.6242359280586243], [0.570091724395752], [0.5154778957366943], [0.46047136187553406], [0.4051465094089508], [0.34957823157310486], [0.29384687542915344], [0.23803307116031647], [0.18221771717071533], [0.1264772117137909], [0.07089433073997498], [0.015551033429801464], [-0.03947228193283081], [-0.09409701079130173], [-0.1482478678226471], [-0.20184631645679474], [-0.25481826066970825], [-0.30708977580070496], [-0.358586847782135], [-0.4092375636100769], [-0.45896923542022705], [-0.5077088475227356], [-0.5553777813911438], [-0.6018968820571899], [-0.6471809148788452], [-0.6911268830299377]]), requires_grad=True)
        self.all_parameters["Fir_target_2"] = torch.nn.Parameter(torch.tensor([[0.5733448266983032], [0.5387258529663086], [0.5038601160049438], [0.4687975347042084], [0.4335801303386688], [0.3982495069503784], [0.3628474473953247], [0.3274137079715729], [0.2919822633266449], [0.256589412689209], [0.2212648093700409], [0.1860395222902298], [0.1509401649236679], [0.11599166691303253], [0.0812096893787384], [0.04661747068166733], [0.012225558049976826], [-0.021954987198114395], [-0.055915407836437225], [-0.08965113759040833], [-0.1231599673628807], [-0.15644104778766632], [-0.18949411809444427], [-0.22231924533843994], [-0.25491759181022644], [-0.287289023399353], [-0.3194332718849182], [-0.35134485363960266], [-0.3830169439315796], [-0.4144284725189209]]), requires_grad=True)
        self.all_parameters["Fir_target_3"] = torch.nn.Parameter(torch.tensor([[0.3524777293205261], [0.33502820134162903], [0.31745919585227966], [0.2998187839984894], [0.2821652889251709], [0.2645634114742279], [0.24707533419132233], [0.2297789752483368], [0.21274466812610626], [0.19604426622390747], [0.1797492951154709], [0.16392557322978973], [0.1486373245716095], [0.13394057750701904], [0.11988916248083115], [0.10652604699134827], [0.09388204663991928], [0.08198556303977966], [0.07084733992815018], [0.06047321856021881], [0.050854846835136414], [0.04197553172707558], [0.03381021320819855], [0.026326296851038933], [0.01948302797973156], [0.013237235136330128], [0.007539053447544575], [0.0023330524563789368], [-0.0024375305511057377], [-0.006836494896560907]]), requires_grad=True)
        self.all_parameters["Fir_target_4"] = torch.nn.Parameter(torch.tensor([[0.29996585845947266], [0.2844270169734955], [0.2686833441257477], [0.2527642548084259], [0.23669418692588806], [0.22050198912620544], [0.20421387255191803], [0.187852144241333], [0.17144276201725006], [0.15500293672084808], [0.1385558396577835], [0.12212695926427841], [0.10573706030845642], [0.0894150361418724], [0.07319120317697525], [0.0570993646979332], [0.04117598384618759], [0.02546309307217598], [0.01000835932791233], [-0.0051408628933131695], [-0.01993296481668949], [-0.03431183472275734], [-0.048220764845609665], [-0.061600446701049805], [-0.0743916928768158], [-0.08653859794139862], [-0.09798769652843475], [-0.10869012773036957], [-0.11859968304634094], [-0.1276792734861374]]), requires_grad=True)
        self.all_parameters["Fir_target_5"] = torch.nn.Parameter(torch.tensor([[0.31530195474624634], [0.29468315839767456], [0.273494690656662], [0.25176700949668884], [0.22953493893146515], [0.20682759582996368], [0.18348942697048187], [0.15865054726600647], [0.13493667542934418], [0.111105777323246], [0.08720876276493073], [0.06333461403846741], [0.03958143666386604], [0.01604893058538437], [-0.007159021683037281], [-0.029941871762275696], [-0.05220096558332443], [-0.0738447830080986], [-0.09479029476642609], [-0.11496301740407944], [-0.13430044054985046], [-0.1527561992406845], [-0.17031586170196533], [-0.18727216124534607], [-0.2037562131881714], [-0.21811345219612122], [-0.23139715194702148], [-0.2435224950313568], [-0.25440123677253723], [-0.26392608880996704]]), requires_grad=True)
        self.all_parameters["Fir_target_6"] = torch.nn.Parameter(torch.tensor([[0.28236696124076843], [0.25818634033203125], [0.23391760885715485], [0.20979410409927368], [0.1860314905643463], [0.16271598637104034], [0.1379116028547287], [0.11588051915168762], [0.09518630057573318], [0.07580560445785522], [0.0577576644718647], [0.041072387248277664], [0.02577033080160618], [0.011852824129164219], [-0.0006969108944758773], [-0.011910236440598965], [-0.021831152960658073], [-0.030516035854816437], [-0.038026582449674606], [-0.04442623630166054], [-0.049773529171943665], [-0.05411948636174202], [-0.057508137077093124], [-0.05997495353221893], [-0.06154481694102287], [-0.06223170459270477], [-0.062035489827394485], [-0.06094639003276825], [-0.05894255265593529], [-0.05598975718021393]]), requires_grad=True)
        self.all_parameters["Fir_target_7"] = torch.nn.Parameter(torch.tensor([[0.17653429508209229], [0.16239482164382935], [0.14905300736427307], [0.13639730215072632], [0.12432683259248734], [0.11275620013475418], [0.10160979628562927], [0.09082990884780884], [0.08037010580301285], [0.0701960027217865], [0.060282137244939804], [0.050608839839696884], [0.041154809296131134], [0.03190069645643234], [0.02282974123954773], [0.013926276937127113], [0.005177393089979887], [-0.003427061252295971], [-0.01189208310097456], [-0.020217526704072952], [-0.028400296345353127], [-0.03643697127699852], [-0.04432457312941551], [-0.05205713212490082], [-0.05962911993265152], [-0.06703425943851471], [-0.07426738739013672], [-0.08131909370422363], [-0.08817297965288162], [-0.09478789567947388]]), requires_grad=True)
        self.all_constants["SamplePart100"] = torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], requires_grad=True)
        self.all_constants["SamplePart106"] = torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]], requires_grad=True)
        self.all_constants["SamplePart111"] = torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]], requires_grad=True)
        self.all_constants["SamplePart113"] = torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]], requires_grad=True)
        self.all_constants["SamplePart97"] = torch.tensor([[1.0]], requires_grad=True)
        self.all_constants["Select118"] = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], requires_grad=True)
        self.all_constants["Select121"] = torch.tensor([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], requires_grad=True)
        self.all_constants["Select124"] = torch.tensor([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0], requires_grad=True)
        self.all_constants["Select127"] = torch.tensor([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0], requires_grad=True)
        self.all_constants["Select130"] = torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], requires_grad=True)
        self.all_constants["Select133"] = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], requires_grad=True)
        self.all_constants["Select136"] = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0], requires_grad=True)
        self.all_constants["Select139"] = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], requires_grad=True)
        self.all_parameters = torch.nn.ParameterDict(self.all_parameters)
        self.all_constants = torch.nn.ParameterDict(self.all_constants)

    def update(self, closed_loop={}, connect={}, disconnect=False):
        pass
    def forward(self, controller_curv_in, controller_vx_in, controller_ax_in, controller_steer_in):
        getitem = controller_steer_in
        relation_forward_sample_part106_w = self.all_constants.SamplePart106
        einsum = torch.functional.einsum('bij,ki->bkj', getitem, relation_forward_sample_part106_w);  getitem = relation_forward_sample_part106_w = None
        all_constants_w_fir_init = self.all_constants.W_fir_init
        mul = einsum * all_constants_w_fir_init;  einsum = all_constants_w_fir_init = None
        size = mul.size(0)
        relation_forward_fir109_weights = self.all_parameters.Fir_InitCondition
        size_1 = relation_forward_fir109_weights.size(1)
        squeeze = mul.squeeze(-1);  mul = None
        matmul = torch.matmul(squeeze, relation_forward_fir109_weights);  squeeze = relation_forward_fir109_weights = None
        to = matmul.to(dtype = torch.float32);  matmul = None
        view = to.view(size, 1, size_1);  to = size = size_1 = None
        getitem_1 = controller_vx_in
        relation_forward_sample_part100_w = self.all_constants.SamplePart100
        einsum_1 = torch.functional.einsum('bij,ki->bkj', getitem_1, relation_forward_sample_part100_w);  getitem_1 = relation_forward_sample_part100_w = None
        zeros_like = torch.zeros_like(einsum_1)
        repeat = zeros_like.repeat(1, 1, 8);  zeros_like = None
        sub = einsum_1 - 8.0
        neg = -sub;  sub = None
        truediv = neg / 2.1428571428571423;  neg = None
        add = truediv + 1;  truediv = None
        _tensor_constant0 = self._tensor_constant0
        maximum = torch.maximum(add, _tensor_constant0);  add = _tensor_constant0 = None
        _tensor_constant1 = self._tensor_constant1
        minimum = torch.minimum(maximum, _tensor_constant1);  maximum = _tensor_constant1 = None
        _tensor_constant2 = self._tensor_constant2
        slicing = nnodely_layers_fuzzify_slicing(repeat, _tensor_constant2, minimum);  _tensor_constant2 = minimum = None
        sub_1 = einsum_1 - 8.0
        truediv_1 = sub_1 / 2.1428571428571423;  sub_1 = None
        _tensor_constant3 = self._tensor_constant3
        maximum_1 = torch.maximum(truediv_1, _tensor_constant3);  truediv_1 = _tensor_constant3 = None
        sub_2 = einsum_1 - 10.142857142857142
        neg_1 = -sub_2;  sub_2 = None
        truediv_2 = neg_1 / 2.1428571428571423;  neg_1 = None
        add_1 = truediv_2 + 1;  truediv_2 = None
        _tensor_constant4 = self._tensor_constant4
        maximum_2 = torch.maximum(add_1, _tensor_constant4);  add_1 = _tensor_constant4 = None
        minimum_1 = torch.minimum(maximum_1, maximum_2);  maximum_1 = maximum_2 = None
        _tensor_constant5 = self._tensor_constant5
        slicing_1 = nnodely_layers_fuzzify_slicing(repeat, _tensor_constant5, minimum_1);  _tensor_constant5 = minimum_1 = None
        sub_3 = einsum_1 - 10.142857142857142
        truediv_3 = sub_3 / 2.1428571428571423;  sub_3 = None
        _tensor_constant6 = self._tensor_constant6
        maximum_3 = torch.maximum(truediv_3, _tensor_constant6);  truediv_3 = _tensor_constant6 = None
        sub_4 = einsum_1 - 12.285714285714285
        neg_2 = -sub_4;  sub_4 = None
        truediv_4 = neg_2 / 2.142857142857144;  neg_2 = None
        add_2 = truediv_4 + 1;  truediv_4 = None
        _tensor_constant7 = self._tensor_constant7
        maximum_4 = torch.maximum(add_2, _tensor_constant7);  add_2 = _tensor_constant7 = None
        minimum_2 = torch.minimum(maximum_3, maximum_4);  maximum_3 = maximum_4 = None
        _tensor_constant8 = self._tensor_constant8
        slicing_2 = nnodely_layers_fuzzify_slicing(repeat, _tensor_constant8, minimum_2);  _tensor_constant8 = minimum_2 = None
        sub_5 = einsum_1 - 12.285714285714285
        truediv_5 = sub_5 / 2.142857142857144;  sub_5 = None
        _tensor_constant9 = self._tensor_constant9
        maximum_5 = torch.maximum(truediv_5, _tensor_constant9);  truediv_5 = _tensor_constant9 = None
        sub_6 = einsum_1 - 14.428571428571429
        neg_3 = -sub_6;  sub_6 = None
        truediv_6 = neg_3 / 2.1428571428571406;  neg_3 = None
        add_3 = truediv_6 + 1;  truediv_6 = None
        _tensor_constant10 = self._tensor_constant10
        maximum_6 = torch.maximum(add_3, _tensor_constant10);  add_3 = _tensor_constant10 = None
        minimum_3 = torch.minimum(maximum_5, maximum_6);  maximum_5 = maximum_6 = None
        _tensor_constant11 = self._tensor_constant11
        slicing_3 = nnodely_layers_fuzzify_slicing(repeat, _tensor_constant11, minimum_3);  _tensor_constant11 = minimum_3 = None
        sub_7 = einsum_1 - 14.428571428571429
        truediv_7 = sub_7 / 2.1428571428571406;  sub_7 = None
        _tensor_constant12 = self._tensor_constant12
        maximum_7 = torch.maximum(truediv_7, _tensor_constant12);  truediv_7 = _tensor_constant12 = None
        sub_8 = einsum_1 - 16.57142857142857
        neg_4 = -sub_8;  sub_8 = None
        truediv_8 = neg_4 / 2.142857142857146;  neg_4 = None
        add_4 = truediv_8 + 1;  truediv_8 = None
        _tensor_constant13 = self._tensor_constant13
        maximum_8 = torch.maximum(add_4, _tensor_constant13);  add_4 = _tensor_constant13 = None
        minimum_4 = torch.minimum(maximum_7, maximum_8);  maximum_7 = maximum_8 = None
        _tensor_constant14 = self._tensor_constant14
        slicing_4 = nnodely_layers_fuzzify_slicing(repeat, _tensor_constant14, minimum_4);  _tensor_constant14 = minimum_4 = None
        sub_9 = einsum_1 - 16.57142857142857
        truediv_9 = sub_9 / 2.142857142857146;  sub_9 = None
        _tensor_constant15 = self._tensor_constant15
        maximum_9 = torch.maximum(truediv_9, _tensor_constant15);  truediv_9 = _tensor_constant15 = None
        sub_10 = einsum_1 - 18.714285714285715
        neg_5 = -sub_10;  sub_10 = None
        truediv_10 = neg_5 / 2.1428571428571423;  neg_5 = None
        add_5 = truediv_10 + 1;  truediv_10 = None
        _tensor_constant16 = self._tensor_constant16
        maximum_10 = torch.maximum(add_5, _tensor_constant16);  add_5 = _tensor_constant16 = None
        minimum_5 = torch.minimum(maximum_9, maximum_10);  maximum_9 = maximum_10 = None
        _tensor_constant17 = self._tensor_constant17
        slicing_5 = nnodely_layers_fuzzify_slicing(repeat, _tensor_constant17, minimum_5);  _tensor_constant17 = minimum_5 = None
        sub_11 = einsum_1 - 18.714285714285715
        truediv_11 = sub_11 / 2.1428571428571423;  sub_11 = None
        _tensor_constant18 = self._tensor_constant18
        maximum_11 = torch.maximum(truediv_11, _tensor_constant18);  truediv_11 = _tensor_constant18 = None
        sub_12 = einsum_1 - 20.857142857142858
        neg_6 = -sub_12;  sub_12 = None
        truediv_12 = neg_6 / 2.1428571428571423;  neg_6 = None
        add_6 = truediv_12 + 1;  truediv_12 = None
        _tensor_constant19 = self._tensor_constant19
        maximum_12 = torch.maximum(add_6, _tensor_constant19);  add_6 = _tensor_constant19 = None
        minimum_6 = torch.minimum(maximum_11, maximum_12);  maximum_11 = maximum_12 = None
        _tensor_constant20 = self._tensor_constant20
        slicing_6 = nnodely_layers_fuzzify_slicing(repeat, _tensor_constant20, minimum_6);  _tensor_constant20 = minimum_6 = None
        sub_13 = einsum_1 - 20.857142857142858;  einsum_1 = None
        truediv_13 = sub_13 / 2.1428571428571423;  sub_13 = None
        _tensor_constant21 = self._tensor_constant21
        maximum_13 = torch.maximum(truediv_13, _tensor_constant21);  truediv_13 = _tensor_constant21 = None
        _tensor_constant22 = self._tensor_constant22
        minimum_7 = torch.minimum(maximum_13, _tensor_constant22);  maximum_13 = _tensor_constant22 = None
        _tensor_constant23 = self._tensor_constant23
        slicing_7 = nnodely_layers_fuzzify_slicing(repeat, _tensor_constant23, minimum_7);  _tensor_constant23 = minimum_7 = None
        relation_forward_select139_w = self.all_constants.Select139
        einsum_2 = torch.functional.einsum('ijk,k->ij', repeat, relation_forward_select139_w);  relation_forward_select139_w = None
        unsqueeze = einsum_2.unsqueeze(2);  einsum_2 = None
        getitem_2 = controller_ax_in
        relation_forward_sample_part97_w = self.all_constants.SamplePart97
        einsum_3 = torch.functional.einsum('bij,ki->bkj', getitem_2, relation_forward_sample_part97_w);  getitem_2 = relation_forward_sample_part97_w = None
        zeros_like_1 = torch.zeros_like(einsum_3)
        repeat_1 = zeros_like_1.repeat(1, 1, 5);  zeros_like_1 = None
        sub_14 = einsum_3 - -2.0
        neg_7 = -sub_14;  sub_14 = None
        truediv_14 = neg_7 / 1.0;  neg_7 = None
        add_7 = truediv_14 + 1;  truediv_14 = None
        _tensor_constant24 = self._tensor_constant24
        maximum_14 = torch.maximum(add_7, _tensor_constant24);  add_7 = _tensor_constant24 = None
        _tensor_constant25 = self._tensor_constant25
        minimum_8 = torch.minimum(maximum_14, _tensor_constant25);  maximum_14 = _tensor_constant25 = None
        _tensor_constant26 = self._tensor_constant26
        slicing_8 = nnodely_layers_fuzzify_slicing(repeat_1, _tensor_constant26, minimum_8);  _tensor_constant26 = minimum_8 = None
        sub_15 = einsum_3 - -2.0
        truediv_15 = sub_15 / 1.0;  sub_15 = None
        _tensor_constant27 = self._tensor_constant27
        maximum_15 = torch.maximum(truediv_15, _tensor_constant27);  truediv_15 = _tensor_constant27 = None
        sub_16 = einsum_3 - -1.0
        neg_8 = -sub_16;  sub_16 = None
        truediv_16 = neg_8 / 1.0;  neg_8 = None
        add_8 = truediv_16 + 1;  truediv_16 = None
        _tensor_constant28 = self._tensor_constant28
        maximum_16 = torch.maximum(add_8, _tensor_constant28);  add_8 = _tensor_constant28 = None
        minimum_9 = torch.minimum(maximum_15, maximum_16);  maximum_15 = maximum_16 = None
        _tensor_constant29 = self._tensor_constant29
        slicing_9 = nnodely_layers_fuzzify_slicing(repeat_1, _tensor_constant29, minimum_9);  _tensor_constant29 = minimum_9 = None
        sub_17 = einsum_3 - -1.0
        truediv_17 = sub_17 / 1.0;  sub_17 = None
        _tensor_constant30 = self._tensor_constant30
        maximum_17 = torch.maximum(truediv_17, _tensor_constant30);  truediv_17 = _tensor_constant30 = None
        sub_18 = einsum_3 - 0.0
        neg_9 = -sub_18;  sub_18 = None
        truediv_18 = neg_9 / 1.0;  neg_9 = None
        add_9 = truediv_18 + 1;  truediv_18 = None
        _tensor_constant31 = self._tensor_constant31
        maximum_18 = torch.maximum(add_9, _tensor_constant31);  add_9 = _tensor_constant31 = None
        minimum_10 = torch.minimum(maximum_17, maximum_18);  maximum_17 = maximum_18 = None
        _tensor_constant32 = self._tensor_constant32
        slicing_10 = nnodely_layers_fuzzify_slicing(repeat_1, _tensor_constant32, minimum_10);  _tensor_constant32 = minimum_10 = None
        sub_19 = einsum_3 - 0.0
        truediv_19 = sub_19 / 1.0;  sub_19 = None
        _tensor_constant33 = self._tensor_constant33
        maximum_19 = torch.maximum(truediv_19, _tensor_constant33);  truediv_19 = _tensor_constant33 = None
        sub_20 = einsum_3 - 1.0
        neg_10 = -sub_20;  sub_20 = None
        truediv_20 = neg_10 / 1.0;  neg_10 = None
        add_10 = truediv_20 + 1;  truediv_20 = None
        _tensor_constant34 = self._tensor_constant34
        maximum_20 = torch.maximum(add_10, _tensor_constant34);  add_10 = _tensor_constant34 = None
        minimum_11 = torch.minimum(maximum_19, maximum_20);  maximum_19 = maximum_20 = None
        _tensor_constant35 = self._tensor_constant35
        slicing_11 = nnodely_layers_fuzzify_slicing(repeat_1, _tensor_constant35, minimum_11);  _tensor_constant35 = minimum_11 = None
        sub_21 = einsum_3 - 1.0;  einsum_3 = None
        truediv_21 = sub_21 / 1.0;  sub_21 = None
        _tensor_constant36 = self._tensor_constant36
        maximum_21 = torch.maximum(truediv_21, _tensor_constant36);  truediv_21 = _tensor_constant36 = None
        _tensor_constant37 = self._tensor_constant37
        minimum_12 = torch.minimum(maximum_21, _tensor_constant37);  maximum_21 = _tensor_constant37 = None
        _tensor_constant38 = self._tensor_constant38
        slicing_12 = nnodely_layers_fuzzify_slicing(repeat_1, _tensor_constant38, minimum_12);  _tensor_constant38 = minimum_12 = None
        all_constants_a = self.all_constants.A
        mul_1 = repeat_1 * all_constants_a;  repeat_1 = all_constants_a = None
        sum_1 = torch.sum(mul_1, dim = 2, keepdim = True);  mul_1 = None
        getitem_3 = controller_curv_in
        relation_forward_sample_part113_w = self.all_constants.SamplePart113
        einsum_4 = torch.functional.einsum('bij,ki->bkj', getitem_3, relation_forward_sample_part113_w);  getitem_3 = relation_forward_sample_part113_w = None
        getitem_4 = controller_vx_in;  kwargs = None
        relation_forward_sample_part111_w = self.all_constants.SamplePart111
        einsum_5 = torch.functional.einsum('bij,ki->bkj', getitem_4, relation_forward_sample_part111_w);  getitem_4 = relation_forward_sample_part111_w = None
        understeer_corr_local_control = nnodely_layers_parametricfunction_understeer_corr_local_control(einsum_5, einsum_4, sum_1);  einsum_5 = einsum_4 = sum_1 = None
        all_constants_w_fir_target = self.all_constants.W_fir_target
        mul_2 = understeer_corr_local_control * all_constants_w_fir_target;  understeer_corr_local_control = all_constants_w_fir_target = None
        size_2 = mul_2.size(0)
        relation_forward_fir138_weights = self.all_parameters.Fir_target_7
        size_3 = relation_forward_fir138_weights.size(1)
        squeeze_1 = mul_2.squeeze(-1)
        matmul_1 = torch.matmul(squeeze_1, relation_forward_fir138_weights);  squeeze_1 = relation_forward_fir138_weights = None
        to_1 = matmul_1.to(dtype = torch.float32);  matmul_1 = None
        view_1 = to_1.view(size_2, 1, size_3);  to_1 = size_2 = size_3 = None
        mul_3 = view_1 * unsqueeze;  view_1 = unsqueeze = None
        relation_forward_select136_w = self.all_constants.Select136
        einsum_6 = torch.functional.einsum('ijk,k->ij', repeat, relation_forward_select136_w);  relation_forward_select136_w = None
        unsqueeze_1 = einsum_6.unsqueeze(2);  einsum_6 = None
        size_4 = mul_2.size(0)
        relation_forward_fir135_weights = self.all_parameters.Fir_target_6
        size_5 = relation_forward_fir135_weights.size(1)
        squeeze_2 = mul_2.squeeze(-1)
        matmul_2 = torch.matmul(squeeze_2, relation_forward_fir135_weights);  squeeze_2 = relation_forward_fir135_weights = None
        to_2 = matmul_2.to(dtype = torch.float32);  matmul_2 = None
        view_2 = to_2.view(size_4, 1, size_5);  to_2 = size_4 = size_5 = None
        mul_4 = view_2 * unsqueeze_1;  view_2 = unsqueeze_1 = None
        relation_forward_select133_w = self.all_constants.Select133
        einsum_7 = torch.functional.einsum('ijk,k->ij', repeat, relation_forward_select133_w);  relation_forward_select133_w = None
        unsqueeze_2 = einsum_7.unsqueeze(2);  einsum_7 = None
        size_6 = mul_2.size(0)
        relation_forward_fir132_weights = self.all_parameters.Fir_target_5
        size_7 = relation_forward_fir132_weights.size(1)
        squeeze_3 = mul_2.squeeze(-1)
        matmul_3 = torch.matmul(squeeze_3, relation_forward_fir132_weights);  squeeze_3 = relation_forward_fir132_weights = None
        to_3 = matmul_3.to(dtype = torch.float32);  matmul_3 = None
        view_3 = to_3.view(size_6, 1, size_7);  to_3 = size_6 = size_7 = None
        mul_5 = view_3 * unsqueeze_2;  view_3 = unsqueeze_2 = None
        relation_forward_select130_w = self.all_constants.Select130
        einsum_8 = torch.functional.einsum('ijk,k->ij', repeat, relation_forward_select130_w);  relation_forward_select130_w = None
        unsqueeze_3 = einsum_8.unsqueeze(2);  einsum_8 = None
        size_8 = mul_2.size(0)
        relation_forward_fir129_weights = self.all_parameters.Fir_target_4
        size_9 = relation_forward_fir129_weights.size(1)
        squeeze_4 = mul_2.squeeze(-1)
        matmul_4 = torch.matmul(squeeze_4, relation_forward_fir129_weights);  squeeze_4 = relation_forward_fir129_weights = None
        to_4 = matmul_4.to(dtype = torch.float32);  matmul_4 = None
        view_4 = to_4.view(size_8, 1, size_9);  to_4 = size_8 = size_9 = None
        mul_6 = view_4 * unsqueeze_3;  view_4 = unsqueeze_3 = None
        relation_forward_select127_w = self.all_constants.Select127
        einsum_9 = torch.functional.einsum('ijk,k->ij', repeat, relation_forward_select127_w);  relation_forward_select127_w = None
        unsqueeze_4 = einsum_9.unsqueeze(2);  einsum_9 = None
        size_10 = mul_2.size(0)
        relation_forward_fir126_weights = self.all_parameters.Fir_target_3
        size_11 = relation_forward_fir126_weights.size(1)
        squeeze_5 = mul_2.squeeze(-1)
        matmul_5 = torch.matmul(squeeze_5, relation_forward_fir126_weights);  squeeze_5 = relation_forward_fir126_weights = None
        to_5 = matmul_5.to(dtype = torch.float32);  matmul_5 = None
        view_5 = to_5.view(size_10, 1, size_11);  to_5 = size_10 = size_11 = None
        mul_7 = view_5 * unsqueeze_4;  view_5 = unsqueeze_4 = None
        relation_forward_select124_w = self.all_constants.Select124
        einsum_10 = torch.functional.einsum('ijk,k->ij', repeat, relation_forward_select124_w);  relation_forward_select124_w = None
        unsqueeze_5 = einsum_10.unsqueeze(2);  einsum_10 = None
        size_12 = mul_2.size(0)
        relation_forward_fir123_weights = self.all_parameters.Fir_target_2
        size_13 = relation_forward_fir123_weights.size(1)
        squeeze_6 = mul_2.squeeze(-1)
        matmul_6 = torch.matmul(squeeze_6, relation_forward_fir123_weights);  squeeze_6 = relation_forward_fir123_weights = None
        to_6 = matmul_6.to(dtype = torch.float32);  matmul_6 = None
        view_6 = to_6.view(size_12, 1, size_13);  to_6 = size_12 = size_13 = None
        mul_8 = view_6 * unsqueeze_5;  view_6 = unsqueeze_5 = None
        relation_forward_select121_w = self.all_constants.Select121
        einsum_11 = torch.functional.einsum('ijk,k->ij', repeat, relation_forward_select121_w);  relation_forward_select121_w = None
        unsqueeze_6 = einsum_11.unsqueeze(2);  einsum_11 = None
        size_14 = mul_2.size(0)
        relation_forward_fir120_weights = self.all_parameters.Fir_target_1
        size_15 = relation_forward_fir120_weights.size(1)
        squeeze_7 = mul_2.squeeze(-1)
        matmul_7 = torch.matmul(squeeze_7, relation_forward_fir120_weights);  squeeze_7 = relation_forward_fir120_weights = None
        to_7 = matmul_7.to(dtype = torch.float32);  matmul_7 = None
        view_7 = to_7.view(size_14, 1, size_15);  to_7 = size_14 = size_15 = None
        mul_9 = view_7 * unsqueeze_6;  view_7 = unsqueeze_6 = None
        relation_forward_select118_w = self.all_constants.Select118
        einsum_12 = torch.functional.einsum('ijk,k->ij', repeat, relation_forward_select118_w);  repeat = relation_forward_select118_w = None
        unsqueeze_7 = einsum_12.unsqueeze(2);  einsum_12 = None
        size_16 = mul_2.size(0)
        relation_forward_fir117_weights = self.all_parameters.Fir_target_0
        size_17 = relation_forward_fir117_weights.size(1)
        squeeze_8 = mul_2.squeeze(-1);  mul_2 = None
        matmul_8 = torch.matmul(squeeze_8, relation_forward_fir117_weights);  squeeze_8 = relation_forward_fir117_weights = None
        to_8 = matmul_8.to(dtype = torch.float32);  matmul_8 = None
        view_8 = to_8.view(size_16, 1, size_17);  to_8 = size_16 = size_17 = None
        mul_10 = view_8 * unsqueeze_7;  view_8 = unsqueeze_7 = None
        add_11 = mul_10 + mul_9;  mul_10 = mul_9 = None
        add_12 = add_11 + mul_8;  add_11 = mul_8 = None
        add_13 = add_12 + mul_7;  add_12 = mul_7 = None
        add_14 = add_13 + mul_6;  add_13 = mul_6 = None
        add_15 = add_14 + mul_5;  add_14 = mul_5 = None
        add_16 = add_15 + mul_4;  add_15 = mul_4 = None
        add_17 = add_16 + mul_3;  add_16 = mul_3 = None
        add_18 = add_17 + view
        outputs = ({'controller_steer_from_ic': view, 'controller_steer_from_target': add_17, 'controller_steer_out': add_18}, {}, {'controller_steer_in': add_18}, {})
        return (outputs[0]['controller_steer_out'],outputs[0]['controller_steer_from_ic'],outputs[0]['controller_steer_from_target'],), (), (outputs[2]['controller_steer_in'], ), ()

class RecurrentModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.Cell = TracerModel()
    def forward(self, controller_curv_in, controller_vx_in, controller_ax_in, controller_steer_in, ):
        n_samples = min([controller_curv_in.size(0), controller_vx_in.size(0), controller_ax_in.size(0)])
        results_controller_steer_out = []
        results_controller_steer_from_ic = []
        results_controller_steer_from_target = []
        for idx in range(n_samples):
            out, losses, closed_loop, connect = self.Cell(controller_curv_in[idx], controller_vx_in[idx], controller_ax_in[idx], controller_steer_in, )
            results_controller_steer_out.append(out[0])
            results_controller_steer_from_ic.append(out[1])
            results_controller_steer_from_target.append(out[2])
            controller_steer_in = nnodely_basic_model_timeshift(controller_steer_in)
            controller_steer_in = nnodely_basic_model_update_state(controller_steer_in, closed_loop[0])
        results_controller_steer_out = torch.stack(results_controller_steer_out, dim=0)
        results_controller_steer_from_ic = torch.stack(results_controller_steer_from_ic, dim=0)
        results_controller_steer_from_target = torch.stack(results_controller_steer_from_target, dim=0)
        return results_controller_steer_out, results_controller_steer_from_ic, results_controller_steer_from_target, 