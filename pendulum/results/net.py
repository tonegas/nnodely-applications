import torch

def nnodely_model_connect(data_in, rel):
    virtual = torch.cat((data_in[:, 1:, :], data_in[:, :1, :]), dim=1)
    max_dim = min(rel.size(1), data_in.size(1))
    virtual[:, -max_dim:, :] = rel[:, -max_dim:, :]
    return virtual

class TracerModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.all_parameters = {}
        self.all_constants = {}
        self.all_parameters["PFir4W"] = torch.nn.Parameter(torch.tensor([[0.063806913793087], [-0.13234031200408936], [0.22834041714668274], [0.1591425985097885], [0.48751300573349], [0.06571295857429504], [0.7737712860107422], [0.6233242750167847], [0.6471247673034668], [0.7498874068260193]]), requires_grad=True)
        self.all_parameters["PFir6W"] = torch.nn.Parameter(torch.tensor([[-1.0791332721710205], [-0.5110344290733337], [-0.9365184903144836], [-0.6643281579017639], [-0.13577714562416077], [0.5677571296691895], [0.07190157473087311], [0.8866100907325745], [0.7644850015640259], [1.0357720851898193]]), requires_grad=True)
        self.all_parameters["PFir8W"] = torch.nn.Parameter(torch.tensor([[0.15335263311862946]]), requires_grad=True)
        self.all_constants["SamplePart13"] = torch.tensor([[1.]], requires_grad=False)
        self.all_constants["SamplePart8"] = torch.tensor([[1.]], requires_grad=False)
        self.all_constants["TimePart1"] = torch.tensor([[1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 1., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 1., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 1., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 1., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 1.]], requires_grad=False)
        self.all_constants["TimePart5"] = torch.tensor([[1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 1., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 1., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 1., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 1., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 1.]], requires_grad=False)
        self.all_parameters = torch.nn.ParameterDict(self.all_parameters)
        self.all_constants = torch.nn.ParameterDict(self.all_constants)

    def update(self, closed_loop={}, connect={}):
        pass
    
    def forward(self, kwargs):
        getitem = kwargs['torque']
        relation_forward_sample_part8_w = self.all_constants.SamplePart8
        einsum = torch.functional.einsum('bij,ki->bkj', getitem, relation_forward_sample_part8_w);  getitem = relation_forward_sample_part8_w = None
        size = einsum.size(0)
        relation_forward_fir9_weights = self.all_parameters.PFir8W
        size_1 = relation_forward_fir9_weights.size(1)
        squeeze = einsum.squeeze(-1);  einsum = None
        matmul = torch.matmul(squeeze, relation_forward_fir9_weights);  squeeze = relation_forward_fir9_weights = None
        to = matmul.to(dtype = torch.float32);  matmul = None
        view = to.view(size, 1, size_1);  to = size = size_1 = None
        getitem_1 = kwargs['theta']
        relation_forward_time_part5_w = self.all_constants.TimePart5
        einsum_1 = torch.functional.einsum('bij,ki->bkj', getitem_1, relation_forward_time_part5_w);  getitem_1 = relation_forward_time_part5_w = None
        size_2 = einsum_1.size(0)
        relation_forward_fir6_weights = self.all_parameters.PFir6W
        size_3 = relation_forward_fir6_weights.size(1)
        squeeze_1 = einsum_1.squeeze(-1);  einsum_1 = None
        matmul_1 = torch.matmul(squeeze_1, relation_forward_fir6_weights);  squeeze_1 = relation_forward_fir6_weights = None
        to_1 = matmul_1.to(dtype = torch.float32);  matmul_1 = None
        view_1 = to_1.view(size_2, 1, size_3);  to_1 = size_2 = size_3 = None
        getitem_2 = kwargs['theta']
        relation_forward_time_part1_w = self.all_constants.TimePart1
        einsum_2 = torch.functional.einsum('bij,ki->bkj', getitem_2, relation_forward_time_part1_w);  getitem_2 = relation_forward_time_part1_w = None
        sin = torch.sin(einsum_2);  einsum_2 = None
        size_4 = sin.size(0)
        relation_forward_fir3_weights = self.all_parameters.PFir4W
        size_5 = relation_forward_fir3_weights.size(1)
        squeeze_2 = sin.squeeze(-1);  sin = None
        matmul_2 = torch.matmul(squeeze_2, relation_forward_fir3_weights);  squeeze_2 = relation_forward_fir3_weights = None
        to_2 = matmul_2.to(dtype = torch.float32);  matmul_2 = None
        view_2 = to_2.view(size_4, 1, size_5);  to_2 = size_4 = size_5 = None
        add = view_2 + view_1;  view_2 = view_1 = None
        add_1 = add + view;  add = view = None
        getitem_3 = kwargs['omega'];  kwargs = None
        relation_forward_sample_part13_w = self.all_constants.SamplePart13
        einsum_3 = torch.functional.einsum('bij,ki->bkj', getitem_3, relation_forward_sample_part13_w);  getitem_3 = relation_forward_sample_part13_w = None
        return ({'omega_pred': add_1}, {'SamplePart13': einsum_3, 'omega_pred': add_1}, {}, {})
        
