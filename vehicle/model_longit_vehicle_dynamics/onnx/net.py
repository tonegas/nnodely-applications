import torch

def nnodely_basic_model_connect(data_in, rel):
    virtual = torch.cat((data_in[:, 1:, :], data_in[:, :1, :]), dim=1)
    max_dim = min(rel.size(1), data_in.size(1))
    virtual[:, -max_dim:, :] = rel[:, -max_dim:, :]
    return virtual

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
        self.all_parameters["PFir17W"] = torch.nn.Parameter(torch.tensor([[-0.005591336637735367], [-0.00556231988593936], [-0.005551861599087715], [-0.005533216986805201], [-0.0055033620446920395], [-0.005462788511067629], [-0.005426831543445587], [-0.005402027163654566], [-0.005352351348847151], [-0.0053106010891497135], [-0.00526846619322896], [-0.005227808374911547], [-0.0051879179663956165], [-0.005124595947563648], [-0.005033878143876791], [-0.0049131144769489765], [-0.004801461938768625], [-0.004685236141085625], [-0.0045578316785395145], [-0.004401085432618856], [-0.0042215557768940926], [-0.00395407248288393], [-0.003614486660808325], [-0.003207604866474867], [-0.002775771077722311]]), requires_grad=True)
        self.all_parameters["PFir19W"] = torch.nn.Parameter(torch.tensor([[-0.007270080503076315], [-0.007235011085867882], [-0.007217633072286844], [-0.007223965600132942], [-0.007242000196129084], [-0.007222650572657585], [-0.007221532985568047], [-0.00719186058267951], [-0.007137169595807791], [-0.007078750059008598], [-0.00703083910048008], [-0.007009277120232582], [-0.00697826873511076], [-0.006927921436727047], [-0.006884141359478235], [-0.00682104704901576], [-0.006717521231621504], [-0.006593525409698486], [-0.006427885964512825], [-0.006307789124548435], [-0.0061520542949438095], [-0.0059730918146669865], [-0.00578729622066021], [-0.005560920108109713], [-0.0053426604717969894]]), requires_grad=True)
        self.all_parameters["PFir21W"] = torch.nn.Parameter(torch.tensor([[-0.008361199870705605], [-0.008324740454554558], [-0.00827372819185257], [-0.008218483068048954], [-0.008173203095793724], [-0.008123013190925121], [-0.008083424530923367], [-0.008062543347477913], [-0.008022258058190346], [-0.007971413433551788], [-0.007923848927021027], [-0.007880287244915962], [-0.007819324731826782], [-0.007742381654679775], [-0.007654882036149502], [-0.007576752919703722], [-0.0074759977869689465], [-0.007348516024649143], [-0.007219363935291767], [-0.007067101076245308], [-0.006891506724059582], [-0.006725314073264599], [-0.006545338314026594], [-0.00633225729689002], [-0.006092165596783161]]), requires_grad=True)
        self.all_parameters["PFir23W"] = torch.nn.Parameter(torch.tensor([[-0.008370306342840195], [-0.008339863270521164], [-0.008325650356709957], [-0.008308257907629013], [-0.00830310583114624], [-0.008280674926936626], [-0.008280080743134022], [-0.008278941735625267], [-0.008278376422822475], [-0.008240717463195324], [-0.008215689100325108], [-0.008165626786649227], [-0.008112192153930664], [-0.008078096434473991], [-0.00800139456987381], [-0.007913544774055481], [-0.007827638648450375], [-0.007756868842989206], [-0.007635686080902815], [-0.007517281454056501], [-0.007343887817114592], [-0.007162525784224272], [-0.00695779686793685], [-0.006733263377100229], [-0.006472156383097172]]), requires_grad=True)
        self.all_parameters["PFir25W"] = torch.nn.Parameter(torch.tensor([[-0.008025185205042362], [-0.008025972172617912], [-0.008015881292521954], [-0.008002463728189468], [-0.007981868460774422], [-0.007957161404192448], [-0.007934058085083961], [-0.007912679575383663], [-0.007895911112427711], [-0.007863505743443966], [-0.007835433818399906], [-0.007805307861417532], [-0.007779840379953384], [-0.00773268099874258], [-0.007667612284421921], [-0.00759255001321435], [-0.007509528659284115], [-0.00741309579461813], [-0.007285016588866711], [-0.007142023183405399], [-0.0069765918888151646], [-0.0067956955172121525], [-0.006609996315091848], [-0.006396244280040264], [-0.006158989854156971]]), requires_grad=True)
        self.all_parameters["PFir27W"] = torch.nn.Parameter(torch.tensor([[-0.008939730934798717], [-0.008927649818360806], [-0.008915845304727554], [-0.008901924826204777], [-0.008885661140084267], [-0.008864733390510082], [-0.008837919682264328], [-0.008807669393718243], [-0.00877176783978939], [-0.008733057416975498], [-0.008690640330314636], [-0.008641085587441921], [-0.008580422960221767], [-0.008515367284417152], [-0.008445271290838718], [-0.008369709365069866], [-0.00828090962022543], [-0.008180927485227585], [-0.008066436275839806], [-0.007940453477203846], [-0.007797221187502146], [-0.007633603643625975], [-0.007448225282132626], [-0.007241092622280121], [-0.007004931569099426]]), requires_grad=True)
        self.all_parameters["PFir11W"] = torch.nn.Parameter(torch.tensor([[0.008527202531695366], [0.008539924398064613], [0.008539069443941116], [0.008601736277341843], [0.008667985908687115], [0.008701938204467297], [0.008717993274331093], [0.008755896240472794], [0.008807212114334106], [0.008839099667966366], [0.008838022127747536], [0.00887773372232914], [0.008953974582254887], [0.008948948234319687], [0.00903286226093769], [0.009079908952116966], [0.00915302149951458], [0.0092434948310256], [0.00931038148701191], [0.009453165344893932], [0.00963855441659689], [0.009852924384176731], [0.010064595378935337], [0.010149502195417881], [0.010393128730356693]]), requires_grad=True)
        self.Linear12 = torch.nn.Dropout(p=0.1)
        self.all_parameters["gravity"] = torch.nn.Parameter(torch.tensor([[-0.0024196570739150047], [-0.002382629318162799], [-0.0023432299494743347], [-0.002304633380845189], [-0.0022655713837593794], [-0.0022282323334366083], [-0.0021909296046942472], [-0.0021552026737481356], [-0.0021135322749614716], [-0.002046094974502921], [0.0], [0.002014478435739875], [0.0020477771759033203], [0.002056571887806058], [0.002063042251393199], [0.0020693170372396708], [0.0020805869717150927], [0.0020932035986334085], [0.002112589543685317], [0.002133534988388419], [0.002160219708457589]]), requires_grad=True)
        self.all_parameters["PLinear7b"] = torch.nn.Parameter(torch.tensor([0.6055392622947693]), requires_grad=True)
        self.all_parameters["PLinear7W"] = torch.nn.Parameter(torch.tensor([[0.37189188599586487]]), requires_grad=True)
        self.all_constants["SamplePart1"] = torch.tensor([[1.]], requires_grad=False)
        self.all_constants["SamplePart11"] = torch.tensor([[1.]], requires_grad=False)
        self.all_constants["SamplePart14"] = torch.tensor([[1.]], requires_grad=False)
        self.all_constants["SamplePart17"] = torch.tensor([[1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
         0., 0., 0., 0., 0., 0., 0.],
        [0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
         0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
         0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
         0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
         0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
         0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
         0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
         0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
         0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.,
         0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.,
         0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.,
         0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.,
         0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.,
         0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.,
         0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.,
         0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.,
         0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.,
         0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
         1., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
         0., 1., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
         0., 0., 1., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
         0., 0., 0., 1., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
         0., 0., 0., 0., 1., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
         0., 0., 0., 0., 0., 1., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
         0., 0., 0., 0., 0., 0., 1.]], requires_grad=False)
        self.all_constants["SamplePart45"] = torch.tensor([[1.]], requires_grad=False)
        self.all_constants["SamplePart6"] = torch.tensor([[1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
         0., 0., 0., 0., 0., 0., 0.],
        [0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
         0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
         0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
         0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
         0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
         0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
         0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
         0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
         0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.,
         0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.,
         0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.,
         0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.,
         0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.,
         0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.,
         0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.,
         0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.,
         0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.,
         0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
         1., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
         0., 1., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
         0., 0., 1., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
         0., 0., 0., 1., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
         0., 0., 0., 0., 1., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
         0., 0., 0., 0., 0., 1., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
         0., 0., 0., 0., 0., 0., 1.]], requires_grad=False)
        self.all_constants["Select19"] = torch.tensor([1., 0., 0., 0., 0., 0.], requires_grad=False)
        self.all_constants["Select22"] = torch.tensor([0., 1., 0., 0., 0., 0.], requires_grad=False)
        self.all_constants["Select25"] = torch.tensor([0., 0., 1., 0., 0., 0.], requires_grad=False)
        self.all_constants["Select28"] = torch.tensor([0., 0., 0., 1., 0., 0.], requires_grad=False)
        self.all_constants["Select31"] = torch.tensor([0., 0., 0., 0., 1., 0.], requires_grad=False)
        self.all_constants["Select34"] = torch.tensor([0., 0., 0., 0., 0., 1.], requires_grad=False)
        self.all_parameters = torch.nn.ParameterDict(self.all_parameters)
        self.all_constants = torch.nn.ParameterDict(self.all_constants)

    def update(self, closed_loop={}, connect={}, disconnect=False):
        pass
    def forward(self, kwargs):
        getitem = kwargs['gear']
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
        getitem_1 = kwargs['trq']
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
        getitem_2 = kwargs['alt']
        relation_forward_sample_part11_w = self.all_constants.SamplePart11
        einsum_8 = torch.functional.einsum('bij,ki->bkj', getitem_2, relation_forward_sample_part11_w);  getitem_2 = relation_forward_sample_part11_w = None
        relation_forward_linear12_weights = self.all_parameters.gravity
        einsum_9 = torch.functional.einsum('bwi,io->bwo', einsum_8, relation_forward_linear12_weights);  einsum_8 = relation_forward_linear12_weights = None
        relation_forward_linear12_dropout = self.Linear12(einsum_9);  einsum_9 = None
        getitem_3 = kwargs['brk']
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
        getitem_4 = kwargs['vel']
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
        getitem_5 = kwargs['acc'];  kwargs = None
        relation_forward_sample_part45_w = self.all_constants.SamplePart45
        einsum_13 = torch.functional.einsum('bij,ki->bkj', getitem_5, relation_forward_sample_part45_w);  getitem_5 = relation_forward_sample_part45_w = None
        return ({'accelleration': add_8}, {'SamplePart45': einsum_13, 'Add43': add_8}, {}, {})
        
