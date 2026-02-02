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
        self.all_parameters["PLinear11W"] = torch.nn.Parameter(torch.tensor([[-0.05396796017885208, -0.051660869270563126, 1.2701075077056885, 0.3034270107746124, -0.19626079499721527, -0.03862933814525604, -0.03909926116466522, -0.02090965397655964, 0.01998959667980671, 0.15167498588562012, -0.0009713638573884964, -0.0005728565738536417, -0.0008836605702526867, 0.0005065530422143638, -0.00014217158604878932, 0.00046061386819928885], [-0.06227584555745125, -0.41335803270339966, 0.01145391445606947, 0.000793071580119431, 0.033802732825279236, -0.04997151345014572, -0.04895996302366257, 0.05799904465675354, -0.0572163350880146, -0.20078477263450623, -0.0007106741541065276, 0.0008353741141036153, 0.00048444943968206644, -0.000407478742999956, 0.0006562787457369268, -0.0005950283957645297], [0.11711250990629196, -0.06566427648067474, -0.02714340202510357, 0.007616002578288317, -0.03569070249795914, 0.0391271598637104, 0.03965962305665016, -0.11482985317707062, 0.11480505764484406, 0.19680923223495483, 0.00029738625744357705, -0.000898603058885783, -0.0008469593012705445, -0.0009339434327557683, 3.264948463765904e-05, -0.00037090410478413105], [-0.10156387835741043, -0.007704285904765129, 0.003505390603095293, 0.016187818720936775, 0.0018947250209748745, -0.04545183852314949, -0.0456518791615963, 0.026377523317933083, -0.026810171082615852, -0.08087953180074692, -0.0009724479168653488, 0.00016208700253628194, 0.0004601279797498137, 0.00024389720056205988, 0.00027906164177693427, -0.0004927701083943248], [0.1528097242116928, -0.08365369588136673, -0.012282827869057655, -0.030080704018473625, -0.053095851093530655, 0.035193149000406265, 0.034287385642528534, -0.12451771646738052, 0.12335886061191559, 0.20378591120243073, -6.895640126458602e-06, -0.0008122068247757852, 6.50845468044281e-05, 0.0007863436476327479, -0.000217584049096331, -0.00026353835710324347], [-0.6240378022193909, -0.5600491762161255, -0.007440411020070314, -0.7180415987968445, -0.24646300077438354, -0.469043105840683, -0.468610018491745, -0.25323814153671265, 0.25214460492134094, 0.112564317882061, 0.0006872166995890439, 0.0004982667160220444, -0.0009862290462478995, 0.00046099183964543045, -0.0006061584572307765, -0.0007681436254642904], [0.14823965728282928, 0.0689183846116066, -0.018643464893102646, -0.011715870350599289, -0.11356692016124725, 0.06815411150455475, 0.0687134712934494, -0.15538880228996277, 0.15531831979751587, 0.041090283542871475, 0.0002035340148722753, 0.0002970089844893664, 0.0006380889681167901, -0.0007647587917745113, -0.0008456358918920159, -8.626909402664751e-05]]), requires_grad=True)
        self.all_parameters["PLinear13W"] = torch.nn.Parameter(torch.tensor([[-0.27017441391944885, -0.3677360713481903, -0.03785458207130432, 0.20526909828186035, 0.47232556343078613, 0.6491338610649109, 0.6501571536064148, -0.01497641857713461, 0.014742711558938026, -0.564993679523468, 0.00023268333461601287, 3.206961991963908e-05, -5.0293812819290906e-05, 0.0008134889649227262, 0.0007716042455285788, 0.0006237021298147738], [0.5334225296974182, 0.08597514778375626, 0.2341241091489792, -0.043690670281648636, 0.8006253838539124, 0.049305278807878494, 0.04865049943327904, 0.03121231496334076, -0.03165551647543907, 0.03680303692817688, 0.0003386216121725738, 5.0120241212425753e-05, -7.636614100192674e-06, -0.0007700849091634154, -0.0005391771555878222, 0.0003321795957162976], [0.07346263527870178, -0.11377014219760895, 0.018126219511032104, -0.536567747592926, -0.000300384039292112, 0.018276505172252655, 0.018167905509471893, 0.5748807191848755, -0.5750939846038818, 0.07349929958581924, -0.0003014864050783217, -0.000934708456043154, 0.0006909055518917739, 0.00029243051540106535, -2.3617549231857993e-05, -0.0003969364915974438], [0.04782935231924057, -0.5449182391166687, 0.2893286645412445, 0.15631990134716034, -0.15683650970458984, 0.19074185192584991, 0.1911829560995102, 0.006364063825458288, -0.007584899663925171, 0.23521044850349426, 0.0003924008924514055, 0.0006217332556843758, 0.0009298835066147149, -0.00022152547899167985, 0.0005759382620453835, -8.920844993554056e-05], [0.012138202786445618, -0.09379446506500244, -0.4882730543613434, 0.0558178536593914, 0.06200491264462471, 0.09402197599411011, 0.09279154241085052, 0.008245762437582016, -0.0083573367446661, -0.16046057641506195, 1.5246016118908301e-05, 0.00011137480032630265, -0.0009252741001546383, -0.00016007639351300895, -0.0009793667122721672, -8.022928523132578e-05], [0.06950796395540237, 0.26889559626579285, -0.2158375084400177, -0.046695828437805176, -0.15153425931930542, 0.15571942925453186, 0.15644362568855286, -0.026121612638235092, 0.025658294558525085, 0.27292129397392273, -0.00024111494712997228, -0.0008723391219973564, 0.0007705016178078949, -9.183886504615657e-06, -0.0007908797706477344, -0.0005399584770202637], [-0.31036365032196045, 0.01834513247013092, -0.1459016352891922, -0.04063156247138977, -0.5877846479415894, -0.14157578349113464, -0.14199037849903107, -0.02139466069638729, 0.020493214949965477, 0.019268741831183434, 0.0002088472683681175, 0.0009807591559365392, -0.0007888150284998119, 0.0001673705701250583, -0.0001473081938456744, 0.0009387898608110845]]), requires_grad=True)
        self.all_parameters["PLinear15W"] = torch.nn.Parameter(torch.tensor([[0.12009918689727783, -0.4220503568649292, 1.0039184093475342, -0.7939313054084778, 0.2981888949871063, -0.19741758704185486, -0.1989533007144928, -0.9654124975204468, 0.9662209749221802, 0.23610694706439972, 0.0003618014743551612, 0.0007794553530402482, -0.000331463961629197, -0.0009570351103320718, -7.702963193878531e-05, -0.000989177031442523], [-1.2772208452224731, -0.12462861835956573, -0.13905362784862518, 0.29666444659233093, 0.5099812150001526, 0.002258682856336236, 0.0006519406451843679, 0.17543214559555054, -0.17518913745880127, 0.1374296247959137, 0.00015831406926736236, -0.0009506313945166767, 0.0008128504268825054, 0.0006815607193857431, 0.0008612220408394933, -0.00035367789678275585], [0.25918158888816833, -0.02856486663222313, -0.33221620321273804, -0.02009359374642372, -0.2504792809486389, -0.38039880990982056, -0.38007885217666626, -0.1039993092417717, 0.10459641367197037, -0.562994658946991, -0.00018443171575199813, 0.0008973564254119992, 0.00025265556178055704, 0.0006527031655423343, -0.00035679549910128117, 0.0009267539717257023], [0.9144573211669922, 0.024511266499757767, 0.7054464221000671, 0.13852538168430328, 0.11606227606534958, 0.6434437036514282, 0.6438668966293335, 0.09918675571680069, -0.09967175871133804, -0.018538780510425568, 0.000602978456299752, -0.0003120327601209283, 0.0007860964979045093, 0.0006354249198921025, -0.0008717015152797103, 0.0008336151950061321], [-0.23760171234607697, -0.1686117947101593, 0.041383299976587296, -0.047511324286460876, -0.15260161459445953, -0.13754677772521973, -0.1378629207611084, -0.14747081696987152, 0.1476794332265854, 0.40633726119995117, 0.0006168585969135165, -0.000676647003274411, 0.0008825189433991909, -0.0009887493215501308, 0.0006337818340398371, -0.00015634126611985266], [0.10114099830389023, 0.07654790580272675, -0.4961816966533661, 0.1807769387960434, -0.38483545184135437, 0.43842464685440063, 0.4391162395477295, -0.1484154760837555, 0.1474253535270691, -0.019528374075889587, 0.0009011919028125703, -0.00043150808778591454, -0.0008605962502770126, 0.0005408090073615313, -0.0007719695568084717, 0.0008062771521508694], [-0.13427238166332245, 0.1462155282497406, -0.08910772949457169, 0.8806106448173523, 0.09898283332586288, 0.2614208161830902, 0.2609970271587372, 0.36020928621292114, -0.3593583106994629, 0.049281373620033264, 0.0004037925391457975, 0.0002889517636504024, 0.0002735763555392623, -8.654601697344333e-05, 0.0003788203757721931, -0.0008406806737184525]]), requires_grad=True)
        self.all_parameters["PLinear17W"] = torch.nn.Parameter(torch.tensor([[-2.2818641662597656, -2.705190658569336, -1.2002754211425781, -1.5004502534866333, 0.21109724044799805, -0.4207266867160797, -0.4222750961780548, -0.6485060453414917, 0.6476781368255615, 0.8685688972473145, -0.0009157342137768865, -0.000865911366418004, -0.00026782549684867263, -0.0009592515416443348, -0.0005679624737240374, 0.0005039810785092413], [-0.056083813309669495, -0.000611721770837903, -0.01814018003642559, 1.6679219007492065, 1.6193077564239502, -1.4068894386291504, -1.406820297241211, -0.37870916724205017, 0.3791501224040985, -0.09070861339569092, 0.0007693744846619666, 0.0009146449738182127, 2.2192387405084446e-05, -0.0006224442622624338, 0.00042747266707010567, -0.00032183891744352877], [0.8892371654510498, 0.5402072668075562, -1.1276239156723022, 0.8423755764961243, 2.15609073638916, -1.7659194469451904, -1.765749216079712, 2.033757448196411, -2.033463954925537, -1.1109251976013184, 0.00014634204853791744, -0.0002375620388193056, -0.0009161567431874573, -0.00012903862807434052, -9.978334128390998e-05, -0.0008111844654195011], [0.01882357895374298, 1.9862078428268433, -0.5796030163764954, 0.017262613400816917, -0.09687329083681107, 5.452556133270264, 5.450911045074463, 2.1626601219177246, -2.161738395690918, -0.31227436661720276, -0.0002473554923199117, -0.000666772888507694, 0.0004341640742495656, -0.00045649564708583057, 0.0008921894477680326, 1.722702472761739e-05], [0.8226611614227295, -0.34116360545158386, 2.0360984802246094, -0.5615633130073547, -1.2803430557250977, -1.632811188697815, -1.6332424879074097, -1.623518466949463, 1.6235618591308594, 1.0622318983078003, -9.120965114561841e-05, -0.000872429518494755, 0.0005115997046232224, 3.365517477504909e-05, -0.0008960979175753891, 1.4257453585742041e-05], [0.4152791202068329, -0.24388089776039124, -1.4569742679595947, -0.7189589142799377, 1.1166239976882935, -0.9341872334480286, -0.9328077435493469, 0.3357539474964142, -0.3367243707180023, -1.121719241142273, 0.0005925146979279816, 0.00016061170026659966, -0.0008362037478946149, 0.0004444069054443389, 0.00027270943974144757, -0.0003879487339872867], [0.23094305396080017, -0.6859961748123169, 1.2625590562820435, -3.575223922729492, -0.1106402575969696, -0.040145393460989, -0.040058281272649765, -0.6144916415214539, 0.6137873530387878, -0.931857705116272, 0.0005191775853745639, -7.226247544167563e-05, 0.0007920574280433357, 0.0008974527008831501, 0.0007440807530656457, -0.0003948893863707781]]), requires_grad=True)
        self.all_parameters["PLinear19W"] = torch.nn.Parameter(torch.tensor([[3.2027275562286377, -1.456723690032959, 21.362451553344727], [0.02204504795372486, -3.32538104057312, 15.011308670043945], [-1.9377580881118774, -0.7721957564353943, -1.8474262952804565], [-0.7597042918205261, 0.28816932439804077, -0.4087508022785187], [-0.23618224263191223, 4.353035926818848, -5.384149551391602], [1.5364446640014648, 0.048778291791677475, -2.76452898979187], [0.42838120460510254, 0.6477887630462646, -1.1367207765579224]]), requires_grad=True)
        self.all_parameters["PLinear21W"] = torch.nn.Parameter(torch.tensor([[-0.033776525408029556], [-0.3710744082927704], [0.07168954610824585], [-0.013632066547870636], [-0.02793237194418907], [-0.15484404563903809], [-0.4111219346523285]]), requires_grad=True)
        self.all_constants["SamplePart1"] = torch.tensor([[1.]], requires_grad=False)
        self.all_constants["SamplePart100"] = torch.tensor([[1.]], requires_grad=False)
        self.all_constants["SamplePart102"] = torch.tensor([[1.]], requires_grad=False)
        self.all_constants["SamplePart104"] = torch.tensor([[1.]], requires_grad=False)
        self.all_constants["SamplePart106"] = torch.tensor([[1.]], requires_grad=False)
        self.all_constants["SamplePart11"] = torch.tensor([[1.]], requires_grad=False)
        self.all_constants["SamplePart13"] = torch.tensor([[1.]], requires_grad=False)
        self.all_constants["SamplePart259"] = torch.tensor([[1.]], requires_grad=False)
        self.all_constants["SamplePart261"] = torch.tensor([[1.]], requires_grad=False)
        self.all_constants["SamplePart263"] = torch.tensor([[1.]], requires_grad=False)
        self.all_constants["SamplePart3"] = torch.tensor([[1.]], requires_grad=False)
        self.all_constants["SamplePart5"] = torch.tensor([[1.]], requires_grad=False)
        self.all_constants["SamplePart7"] = torch.tensor([[1.]], requires_grad=False)
        self.all_constants["SamplePart9"] = torch.tensor([[1.]], requires_grad=False)
        self.all_constants["SamplePart94"] = torch.tensor([[1.]], requires_grad=False)
        self.all_constants["SamplePart96"] = torch.tensor([[1.]], requires_grad=False)
        self.all_constants["SamplePart98"] = torch.tensor([[1.]], requires_grad=False)
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
        getitem_14 = kwargs['Xddth2']
        relation_forward_sample_part263_w = self.all_constants.SamplePart263
        einsum_94 = torch.functional.einsum('bij,ki->bkj', getitem_14, relation_forward_sample_part263_w);  getitem_14 = relation_forward_sample_part263_w = None
        all_constants_constant32 = self.all_constants.Constant32
        add_8 = einsum_40 + all_constants_constant32;  all_constants_constant32 = None
        relation_forward_select214_w = self.all_constants.Select214
        einsum_95 = torch.functional.einsum('ijk,k->ij', einsum_92, relation_forward_select214_w);  relation_forward_select214_w = None
        unsqueeze_71 = einsum_95.unsqueeze(2);  einsum_95 = None
        truediv_1 = unsqueeze_71 / add_8;  unsqueeze_71 = add_8 = None
        getitem_15 = kwargs['Xddth1']
        relation_forward_sample_part261_w = self.all_constants.SamplePart261
        einsum_96 = torch.functional.einsum('bij,ki->bkj', getitem_15, relation_forward_sample_part261_w);  getitem_15 = relation_forward_sample_part261_w = None
        all_constants_constant31 = self.all_constants.Constant31
        add_9 = einsum_40 + all_constants_constant31;  einsum_40 = all_constants_constant31 = None
        relation_forward_select210_w = self.all_constants.Select210
        einsum_97 = torch.functional.einsum('ijk,k->ij', einsum_92, relation_forward_select210_w);  einsum_92 = relation_forward_select210_w = None
        unsqueeze_72 = einsum_97.unsqueeze(2);  einsum_97 = None
        truediv_2 = unsqueeze_72 / add_9;  unsqueeze_72 = add_9 = None
        getitem_16 = kwargs['Xddx'];  kwargs = None
        relation_forward_sample_part259_w = self.all_constants.SamplePart259
        einsum_98 = torch.functional.einsum('bij,ki->bkj', getitem_16, relation_forward_sample_part259_w);  getitem_16 = relation_forward_sample_part259_w = None
        return ({'th2_ddot_est': truediv, 'th1_ddot_est': truediv_1, 'acc_cart_est': truediv_2}, {'SamplePart259': einsum_98, 'SamplePart261': einsum_96, 'SamplePart263': einsum_94, 'Div213': truediv_2, 'Div217': truediv_1, 'Div221': truediv}, {}, {})
        
