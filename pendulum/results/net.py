import torch

class TracerModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.all_parameters = {}
        self.all_constants = {}
        self.all_parameters["PFir5p"] = torch.nn.Parameter(torch.tensor([[0.14562338590621948]]), requires_grad=True)
        self.all_parameters["PFir3p"] = torch.nn.Parameter(torch.tensor([[0.03213246539235115], [-0.021538889035582542], [0.053095363080501556], [0.34138908982276917], [0.11427462100982666], [0.6046465039253235], [0.38414159417152405], [0.5539029836654663], [0.856288731098175], [0.6727710962295532]]), requires_grad=True)
        self.all_parameters["PFir4p"] = torch.nn.Parameter(torch.tensor([[-0.8496224284172058], [-0.8890238404273987], [-0.730859100818634], [-0.47441890835762024], [-0.08027598261833191], [-0.009404491633176804], [0.10820397734642029], [0.8905107378959656], [1.0606800317764282], [0.9743391871452332]]), requires_grad=True)
        self.all_parameters = torch.nn.ParameterDict(self.all_parameters)
        self.all_constants = torch.nn.ParameterDict(self.all_constants)
    def init_states(self, state_model, connect = {}, reset_states = False):
        pass
    def reset_connect_variables(self, connect, values = None, only = True):
        pass
    def reset_states(self, values = None, only = True):
        pass
    
    
    def forward(self, kwargs):
        getitem = kwargs['torque']
        getitem_1 = getitem[(slice(None, None, None), slice(0, 1, None))];  getitem = None
        size = getitem_1.size(0)
        relation_forward_fir12_weights = self.all_parameters.PFir5p
        size_1 = relation_forward_fir12_weights.size(1)
        squeeze = getitem_1.squeeze(-1);  getitem_1 = None
        matmul = torch.matmul(squeeze, relation_forward_fir12_weights);  squeeze = relation_forward_fir12_weights = None
        view = matmul.view(size, 1, size_1);  matmul = size = size_1 = None
        getitem_2 = kwargs['theta']
        getitem_3 = getitem_2[(slice(None, None, None), slice(0, 10, None))];  getitem_2 = None
        size_2 = getitem_3.size(0)
        relation_forward_fir9_weights = self.all_parameters.PFir4p
        size_3 = relation_forward_fir9_weights.size(1)
        squeeze_1 = getitem_3.squeeze(-1);  getitem_3 = None
        matmul_1 = torch.matmul(squeeze_1, relation_forward_fir9_weights);  squeeze_1 = relation_forward_fir9_weights = None
        view_1 = matmul_1.view(size_2, 1, size_3);  matmul_1 = size_2 = size_3 = None
        getitem_4 = kwargs['theta']
        getitem_5 = getitem_4[(slice(None, None, None), slice(0, 10, None))];  getitem_4 = None
        sin = torch.sin(getitem_5);  getitem_5 = None
        size_4 = sin.size(0)
        relation_forward_fir6_weights = self.all_parameters.PFir3p
        size_5 = relation_forward_fir6_weights.size(1)
        squeeze_2 = sin.squeeze(-1);  sin = None
        matmul_2 = torch.matmul(squeeze_2, relation_forward_fir6_weights);  squeeze_2 = relation_forward_fir6_weights = None
        view_2 = matmul_2.view(size_4, 1, size_5);  matmul_2 = size_4 = size_5 = None
        add = torch.add(view_2, view_1);  view_2 = view_1 = None
        add_1 = torch.add(add, view);  add = view = None
        getitem_6 = kwargs['omega'];  kwargs = None
        getitem_7 = getitem_6[(slice(None, None, None), slice(0, 1, None))];  getitem_6 = None
        return ({'omega_pred': add_1}, {'SamplePart17': getitem_7, 'omega_pred': add_1})
        
