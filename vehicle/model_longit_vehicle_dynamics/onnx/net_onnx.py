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
        self._tensor_constant0 = torch.tensor(0)
        self._tensor_constant1 = torch.tensor(1)
        self._tensor_constant2 = torch.tensor(2)
        self._tensor_constant3 = torch.tensor(3)
        self._tensor_constant4 = torch.tensor(4)
        self._tensor_constant5 = torch.tensor(5)
        self.all_constants["Constant10"] = torch.tensor([2.0], requires_grad=False)
        self.all_parameters["PFir17W"] = torch.nn.Parameter(torch.tensor([[-0.008135760203003883], [0.0004996120696887374], [0.004309962037950754], [0.004002674948424101], [0.002453196793794632], [0.0007812415133230388], [-0.0003985333605669439], [-0.0006848013144917786], [-0.0008657987345941365], [-0.0004540160298347473], [-9.57158554228954e-05], [0.0005813991883769631], [0.0009609049302525818], [0.0020991507917642593], [0.002836525673046708], [0.003802570281550288], [0.005496923346072435], [0.006507326848804951], [0.007515795063227415], [0.008500070311129093], [0.008230590261518955], [0.007287502288818359], [0.006341757718473673], [0.006181484553962946], [0.007319348398596048]]), requires_grad=True)
        self.all_parameters["PFir19W"] = torch.nn.Parameter(torch.tensor([[0.002957673976197839], [0.0019238614477217197], [-0.0003223423846065998], [-0.00046587895485572517], [0.0001462403597543016], [0.0006810129270888865], [-0.00013726229371968657], [-0.0006233762833289802], [-0.0006556377629749477], [-0.0004681602877099067], [-0.00036322750383988023], [8.073947537923232e-05], [0.001206699525937438], [0.0014884568518027663], [0.0014454127522185445], [0.0011810465948656201], [0.001092472462914884], [0.0008893698104657233], [0.001785975182428956], [0.002921110251918435], [0.0037832220550626516], [0.005944572854787111], [0.007672573439776897], [0.007852957583963871], [0.00616086320951581]]), requires_grad=True)
        self.all_parameters["PFir21W"] = torch.nn.Parameter(torch.tensor([[0.002462932374328375], [0.0015282232780009508], [-0.0002149208594346419], [-0.0008415934280492365], [-0.0007885030354373157], [-0.00024792723706923425], [0.00046445115003734827], [0.0007007668609730899], [0.00018747488502413034], [-0.00012529283412732184], [-2.344283348065801e-05], [2.7445204977993853e-05], [-0.00026831720606423914], [-0.00031946218223311007], [0.000279019441222772], [0.001101460075005889], [0.002364309271797538], [0.0026065586134791374], [0.0008173316018655896], [0.00160711829084903], [0.0021603398490697145], [0.002691823523491621], [0.004665428772568703], [0.006260385271161795], [0.004034966696053743]]), requires_grad=True)
        self.all_parameters["PFir23W"] = torch.nn.Parameter(torch.tensor([[-0.0005539869889616966], [0.0010661132400855422], [0.0005811607697978616], [0.00012066643103025854], [4.343838372733444e-05], [-3.483053660602309e-05], [6.282547110458836e-05], [0.00012555952707771212], [7.61906267143786e-05], [0.00015493073442485183], [6.131002737674862e-05], [-2.384301660640631e-05], [3.87910331482999e-05], [2.579169813543558e-05], [9.785821748664603e-05], [4.431992056197487e-05], [0.00040023220935836434], [0.00113433669321239], [0.002760229166597128], [0.0010034460574388504], [0.0011995090171694756], [0.0029544762801378965], [0.0035775229334831238], [0.005169224459677935], [0.0027265604585409164]]), requires_grad=True)
        self.all_parameters["PFir25W"] = torch.nn.Parameter(torch.tensor([[6.949935777811334e-05], [0.0006358008831739426], [0.0004081868100911379], [-9.335689128420199e-07], [-0.00022903659555595368], [-0.0002558834385126829], [-8.76335398061201e-05], [0.0001335157867288217], [0.00025910328258760273], [8.047556912060827e-05], [0.00031920932815410197], [0.000490573060233146], [0.0005941720446571708], [0.0001927856501424685], [-6.964457134017721e-05], [-0.00024098512949422002], [9.968446101993322e-05], [0.0007178228115662932], [0.0014348794938996434], [0.0019577140919864178], [0.001660904847085476], [0.0014155323151499033], [0.003231754992157221], [0.0036110240034759045], [0.0018623671494424343]]), requires_grad=True)
        self.all_parameters["PFir27W"] = torch.nn.Parameter(torch.tensor([[-0.0007988717989064753], [0.00048245626385323703], [0.00040841029840521514], [4.5573375246021897e-05], [-0.0001252640358870849], [-9.60647885221988e-05], [1.1713585081452038e-05], [0.00010226163431070745], [0.00014543661382049322], [0.00019973848247900605], [0.00016277917893603444], [7.184610876720399e-05], [-3.0018836696399376e-05], [-0.00012141145271016285], [-0.00012333030463196337], [1.635169974179007e-05], [0.0003255648189224303], [0.0006765684811398387], [0.000913522788323462], [0.0009344323771074414], [0.001248107641004026], [0.00195650989189744], [0.0028539299964904785], [0.0031563197262585163], [0.0017614654498174787]]), requires_grad=True)
        self.all_parameters["PFir11W"] = torch.nn.Parameter(torch.tensor([[-0.0012907714117318392], [-0.0012035410618409514], [-0.000557518273126334], [0.0001647373428568244], [0.00039622827898710966], [0.0006678080535493791], [0.0011032273760065436], [0.0010254142107442021], [0.0007919071358628571], [0.0006284426199272275], [0.0005191003438085318], [0.0009084712946787477], [0.0015025349566712976], [0.002325770677998662], [0.002565741306170821], [0.0032002092339098454], [0.004275183659046888], [0.005975808948278427], [0.007214523386210203], [0.00854186899960041], [0.011146784760057926], [0.016347911208868027], [0.020513171330094337], [0.017901470884680748], [0.013373982161283493]]), requires_grad=True)
        self.Linear12 = torch.nn.Dropout(p=0.1)
        self.all_parameters["gravity"] = torch.nn.Parameter(torch.tensor([[-0.01631101593375206], [-0.009384186938405037], [-0.004503757227212191], [-0.0004733556997962296], [0.0014779327902942896], [0.0027063952293246984], [0.00124755606520921], [-0.00021601120533887297], [-0.0021409939508885145], [-0.00041964874253608286], [0.0], [0.00030287308618426323], [0.0025914558209478855], [0.001507107517682016], [-0.00012877301196567714], [-0.0005666159559041262], [0.0004749408981297165], [0.00210341508500278], [0.0051531000062823296], [0.009914634749293327], [0.016117699444293976]]), requires_grad=True)
        self.all_parameters["PLinear7b"] = torch.nn.Parameter(torch.tensor([-0.08660857379436493]), requires_grad=True)
        self.all_parameters["PLinear7W"] = torch.nn.Parameter(torch.tensor([[-0.00021852669306099415]]), requires_grad=True)
        self.all_constants["SamplePart1"] = torch.tensor([[1.0]], requires_grad=True)
        self.all_constants["SamplePart11"] = torch.tensor([[1.0]], requires_grad=True)
        self.all_constants["SamplePart14"] = torch.tensor([[1.0]], requires_grad=True)
        self.all_constants["SamplePart17"] = torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]], requires_grad=True)
        self.all_constants["SamplePart45"] = torch.tensor([[1.0]], requires_grad=True)
        self.all_constants["SamplePart6"] = torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]], requires_grad=True)
        self.all_constants["Select19"] = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0], requires_grad=True)
        self.all_constants["Select22"] = torch.tensor([0.0, 1.0, 0.0, 0.0, 0.0, 0.0], requires_grad=True)
        self.all_constants["Select25"] = torch.tensor([0.0, 0.0, 1.0, 0.0, 0.0, 0.0], requires_grad=True)
        self.all_constants["Select28"] = torch.tensor([0.0, 0.0, 0.0, 1.0, 0.0, 0.0], requires_grad=True)
        self.all_constants["Select31"] = torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0, 0.0], requires_grad=True)
        self.all_constants["Select34"] = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 1.0], requires_grad=True)
        self.all_parameters = torch.nn.ParameterDict(self.all_parameters)
        self.all_constants = torch.nn.ParameterDict(self.all_constants)

    def update(self, closed_loop={}, connect={}, disconnect=False):
        pass
    def forward(self, gear, trq, alt, brk, vel, acc):
        getitem = gear
        relation_forward_sample_part14_w = self.all_constants.SamplePart14
        einsum = torch.functional.einsum('bij,ki->bkj', getitem, relation_forward_sample_part14_w);  getitem = relation_forward_sample_part14_w = None
        zeros_like = torch.zeros_like(einsum)
        repeat = zeros_like.repeat(1, 1, 6);  zeros_like = None
        lt = einsum < 2.5
        where = torch.where(lt, 1.0, 0.0);  lt = None
        _tensor_constant0 = self._tensor_constant0
        slicing = nnodely_layers_fuzzify_slicing(repeat, _tensor_constant0, where);  _tensor_constant0 = where = slicing = None
        ge = einsum >= 2.5
        lt_1 = einsum < 3.5
        and_ = ge & lt_1;  ge = lt_1 = None
        where_1 = torch.where(and_, 1.0, 0.0);  and_ = None
        _tensor_constant1 = self._tensor_constant1
        slicing_1 = nnodely_layers_fuzzify_slicing(repeat, _tensor_constant1, where_1);  _tensor_constant1 = where_1 = slicing_1 = None
        ge_1 = einsum >= 3.5
        lt_2 = einsum < 4.5
        and__1 = ge_1 & lt_2;  ge_1 = lt_2 = None
        where_2 = torch.where(and__1, 1.0, 0.0);  and__1 = None
        _tensor_constant2 = self._tensor_constant2
        slicing_2 = nnodely_layers_fuzzify_slicing(repeat, _tensor_constant2, where_2);  _tensor_constant2 = where_2 = slicing_2 = None
        ge_2 = einsum >= 4.5
        lt_3 = einsum < 5.5
        and__2 = ge_2 & lt_3;  ge_2 = lt_3 = None
        where_3 = torch.where(and__2, 1.0, 0.0);  and__2 = None
        _tensor_constant3 = self._tensor_constant3
        slicing_3 = nnodely_layers_fuzzify_slicing(repeat, _tensor_constant3, where_3);  _tensor_constant3 = where_3 = slicing_3 = None
        ge_3 = einsum >= 5.5
        lt_4 = einsum < 6.5
        and__3 = ge_3 & lt_4;  ge_3 = lt_4 = None
        where_4 = torch.where(and__3, 1.0, 0.0);  and__3 = None
        _tensor_constant4 = self._tensor_constant4
        slicing_4 = nnodely_layers_fuzzify_slicing(repeat, _tensor_constant4, where_4);  _tensor_constant4 = where_4 = slicing_4 = None
        ge_4 = einsum >= 6.5;  einsum = None
        where_5 = torch.where(ge_4, 1.0, 0.0);  ge_4 = None
        _tensor_constant5 = self._tensor_constant5
        slicing_5 = nnodely_layers_fuzzify_slicing(repeat, _tensor_constant5, where_5);  _tensor_constant5 = where_5 = slicing_5 = None
        relation_forward_select34_w = self.all_constants.Select34
        einsum_1 = torch.functional.einsum('ijk,k->ij', repeat, relation_forward_select34_w);  relation_forward_select34_w = None
        unsqueeze = einsum_1.unsqueeze(2);  einsum_1 = None
        getitem_1 = trq
        relation_forward_sample_part17_w = self.all_constants.SamplePart17
        einsum_2 = torch.functional.einsum('bij,ki->bkj', getitem_1, relation_forward_sample_part17_w);  getitem_1 = relation_forward_sample_part17_w = None
        size = einsum_2.size(0)
        relation_forward_fir33_weights = self.all_parameters.PFir27W
        size_1 = relation_forward_fir33_weights.size(1)
        squeeze = einsum_2.squeeze(-1)
        matmul = torch.matmul(squeeze, relation_forward_fir33_weights);  squeeze = relation_forward_fir33_weights = None
        to = matmul.to(dtype = torch.float32);  matmul = None
        view = to.view(size, 1, size_1);  to = size = size_1 = None
        mul = view * unsqueeze;  view = unsqueeze = None
        relation_forward_select31_w = self.all_constants.Select31
        einsum_3 = torch.functional.einsum('ijk,k->ij', repeat, relation_forward_select31_w);  relation_forward_select31_w = None
        unsqueeze_1 = einsum_3.unsqueeze(2);  einsum_3 = None
        size_2 = einsum_2.size(0)
        relation_forward_fir30_weights = self.all_parameters.PFir25W
        size_3 = relation_forward_fir30_weights.size(1)
        squeeze_1 = einsum_2.squeeze(-1)
        matmul_1 = torch.matmul(squeeze_1, relation_forward_fir30_weights);  squeeze_1 = relation_forward_fir30_weights = None
        to_1 = matmul_1.to(dtype = torch.float32);  matmul_1 = None
        view_1 = to_1.view(size_2, 1, size_3);  to_1 = size_2 = size_3 = None
        mul_1 = view_1 * unsqueeze_1;  view_1 = unsqueeze_1 = None
        relation_forward_select28_w = self.all_constants.Select28
        einsum_4 = torch.functional.einsum('ijk,k->ij', repeat, relation_forward_select28_w);  relation_forward_select28_w = None
        unsqueeze_2 = einsum_4.unsqueeze(2);  einsum_4 = None
        size_4 = einsum_2.size(0)
        relation_forward_fir27_weights = self.all_parameters.PFir23W
        size_5 = relation_forward_fir27_weights.size(1)
        squeeze_2 = einsum_2.squeeze(-1)
        matmul_2 = torch.matmul(squeeze_2, relation_forward_fir27_weights);  squeeze_2 = relation_forward_fir27_weights = None
        to_2 = matmul_2.to(dtype = torch.float32);  matmul_2 = None
        view_2 = to_2.view(size_4, 1, size_5);  to_2 = size_4 = size_5 = None
        mul_2 = view_2 * unsqueeze_2;  view_2 = unsqueeze_2 = None
        relation_forward_select25_w = self.all_constants.Select25
        einsum_5 = torch.functional.einsum('ijk,k->ij', repeat, relation_forward_select25_w);  relation_forward_select25_w = None
        unsqueeze_3 = einsum_5.unsqueeze(2);  einsum_5 = None
        size_6 = einsum_2.size(0)
        relation_forward_fir24_weights = self.all_parameters.PFir21W
        size_7 = relation_forward_fir24_weights.size(1)
        squeeze_3 = einsum_2.squeeze(-1)
        matmul_3 = torch.matmul(squeeze_3, relation_forward_fir24_weights);  squeeze_3 = relation_forward_fir24_weights = None
        to_3 = matmul_3.to(dtype = torch.float32);  matmul_3 = None
        view_3 = to_3.view(size_6, 1, size_7);  to_3 = size_6 = size_7 = None
        mul_3 = view_3 * unsqueeze_3;  view_3 = unsqueeze_3 = None
        relation_forward_select22_w = self.all_constants.Select22
        einsum_6 = torch.functional.einsum('ijk,k->ij', repeat, relation_forward_select22_w);  relation_forward_select22_w = None
        unsqueeze_4 = einsum_6.unsqueeze(2);  einsum_6 = None
        size_8 = einsum_2.size(0)
        relation_forward_fir21_weights = self.all_parameters.PFir19W
        size_9 = relation_forward_fir21_weights.size(1)
        squeeze_4 = einsum_2.squeeze(-1)
        matmul_4 = torch.matmul(squeeze_4, relation_forward_fir21_weights);  squeeze_4 = relation_forward_fir21_weights = None
        to_4 = matmul_4.to(dtype = torch.float32);  matmul_4 = None
        view_4 = to_4.view(size_8, 1, size_9);  to_4 = size_8 = size_9 = None
        mul_4 = view_4 * unsqueeze_4;  view_4 = unsqueeze_4 = None
        relation_forward_select19_w = self.all_constants.Select19
        einsum_7 = torch.functional.einsum('ijk,k->ij', repeat, relation_forward_select19_w);  repeat = relation_forward_select19_w = None
        unsqueeze_5 = einsum_7.unsqueeze(2);  einsum_7 = None
        size_10 = einsum_2.size(0)
        relation_forward_fir18_weights = self.all_parameters.PFir17W
        size_11 = relation_forward_fir18_weights.size(1)
        squeeze_5 = einsum_2.squeeze(-1);  einsum_2 = None
        matmul_5 = torch.matmul(squeeze_5, relation_forward_fir18_weights);  squeeze_5 = relation_forward_fir18_weights = None
        to_5 = matmul_5.to(dtype = torch.float32);  matmul_5 = None
        view_5 = to_5.view(size_10, 1, size_11);  to_5 = size_10 = size_11 = None
        mul_5 = view_5 * unsqueeze_5;  view_5 = unsqueeze_5 = None
        add = mul_5 + mul_4;  mul_5 = mul_4 = None
        add_1 = add + mul_3;  add = mul_3 = None
        add_2 = add_1 + mul_2;  add_1 = mul_2 = None
        add_3 = add_2 + mul_1;  add_2 = mul_1 = None
        add_4 = add_3 + mul;  add_3 = mul = None
        getitem_2 = alt
        relation_forward_sample_part11_w = self.all_constants.SamplePart11
        einsum_8 = torch.functional.einsum('bij,ki->bkj', getitem_2, relation_forward_sample_part11_w);  getitem_2 = relation_forward_sample_part11_w = None
        relation_forward_linear12_weights = self.all_parameters.gravity
        einsum_9 = torch.functional.einsum('bwi,io->bwo', einsum_8, relation_forward_linear12_weights);  einsum_8 = relation_forward_linear12_weights = None
        relation_forward_linear12_dropout = self.Linear12(einsum_9);  einsum_9 = None
        getitem_3 = brk
        relation_forward_sample_part6_w = self.all_constants.SamplePart6
        einsum_10 = torch.functional.einsum('bij,ki->bkj', getitem_3, relation_forward_sample_part6_w);  getitem_3 = relation_forward_sample_part6_w = None
        size_12 = einsum_10.size(0)
        relation_forward_fir7_weights = self.all_parameters.PFir11W
        size_13 = relation_forward_fir7_weights.size(1)
        squeeze_6 = einsum_10.squeeze(-1);  einsum_10 = None
        matmul_6 = torch.matmul(squeeze_6, relation_forward_fir7_weights);  squeeze_6 = relation_forward_fir7_weights = None
        to_6 = matmul_6.to(dtype = torch.float32);  matmul_6 = None
        view_6 = to_6.view(size_12, 1, size_13);  to_6 = size_12 = size_13 = None
        relu = torch.relu(view_6);  view_6 = None
        neg = -relu;  relu = None
        getitem_4 = vel
        relation_forward_sample_part1_w = self.all_constants.SamplePart1
        einsum_11 = torch.functional.einsum('bij,ki->bkj', getitem_4, relation_forward_sample_part1_w);  getitem_4 = relation_forward_sample_part1_w = None
        all_constants_constant10 = self.all_constants.Constant10
        pow_1 = torch.pow(einsum_11, all_constants_constant10);  einsum_11 = all_constants_constant10 = None
        relation_forward_linear4_weights = self.all_parameters.PLinear7W
        einsum_12 = torch.functional.einsum('bwi,io->bwo', pow_1, relation_forward_linear4_weights);  pow_1 = relation_forward_linear4_weights = None
        relation_forward_linear4_bias = self.all_parameters.PLinear7b
        add_5 = einsum_12 + relation_forward_linear4_bias;  einsum_12 = relation_forward_linear4_bias = None
        add_6 = add_5 + neg;  add_5 = neg = None
        add_7 = add_6 + relation_forward_linear12_dropout;  add_6 = relation_forward_linear12_dropout = None
        add_8 = add_7 + add_4;  add_7 = add_4 = None
        getitem_5 = acc;  kwargs = None
        relation_forward_sample_part45_w = self.all_constants.SamplePart45
        einsum_13 = torch.functional.einsum('bij,ki->bkj', getitem_5, relation_forward_sample_part45_w);  getitem_5 = relation_forward_sample_part45_w = None
        outputs = ({'accelleration': add_8}, {'SamplePart45': einsum_13, 'Add43': add_8}, {}, {})
        return (outputs[0]['accelleration'],), (outputs[1]['SamplePart45'], outputs[1]['Add43'], ), (), ()
