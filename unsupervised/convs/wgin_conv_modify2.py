from typing import Callable, Union, Optional

import torch
from torch import Tensor
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import OptPairTensor, Adj, Size
# from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops

from unsupervised.convs.inits import reset


class WGINConv(MessagePassing):
    def __init__(self, nn: Callable, eps: float = 0., train_eps: bool = False,
                 **kwargs):
        kwargs.setdefault('aggr', 'add')
        super(WGINConv, self).__init__(**kwargs)
        self.nn = nn
        self.initial_eps = eps
        if train_eps:
            self.eps = torch.nn.Parameter(torch.Tensor([eps]))
        else:
            self.register_buffer('eps', torch.Tensor([eps]))
        self.reset_parameters()

    def reset_parameters(self):
        reset(self.nn)
        self.eps.data.fill_(self.initial_eps)

    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj, edge_weight = None, filter = 'low-pass', 
                size: Size = None) -> Tensor:
        """"""
        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)

        # propagate_type: (x: OptPairTensor)
        out = self.propagate1(edge_index, x=x)
        out_raw = self.propagate2(edge_index, x=x)

        x_r = x[1]
        if x_r is not None:
            if filter == 'low-pass':
                out = (1 + self.eps) * x_r + out
            elif filter == 'high-pass':
                out = (1 + self.eps) * x_r - out
            elif filter == 'middle-pass':
                out2 = self.propagate(edge_index, x=out, edge_weight=edge_weight, size=size)
                out = out2 - (1 + self.eps) * x_r
            else:
                out = out_raw

        return self.nn(out)

    def propagate1(self, x, edge_index):
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        out = self.message1(x, edge_index)
        out = self.aggregate(out, edge_index)
        out = self.update(out)
        
        return out
    
    def message1(self, x, edge_index):
        x = self.lin(x)
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        x_j = x[row]
        x_j = norm.view(-1, 1) * x_j
        return x_j

    def aggregate(self, x_j, edge_index):
        row, col = edge_index
        aggr_out = scatter(x_j, col, dim=-2, reduce='sum')
        return aggr_out
    
    def update(self, aggr_out):
        return aggr_out

    def propagate2(self, x, edge_index):
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        out = self.message2(x, edge_index)
        out = self.aggregate(out, edge_index)
        out = self.update(out)
        
        return out
    
    def message2(self, x, edge_index):
        x = self.lin(x)
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        x_j = x[row]
        x_j = norm.view(-1, 1) * x_j
        return x_j

    def __repr__(self):
        return '{}(nn={})'.format(self.__class__.__name__, self.nn)