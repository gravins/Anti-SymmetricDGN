
import torch

from torch.nn import Module, Parameter, init, Linear, ModuleList, Sequential, LeakyReLU
from torch_geometric.nn import MessagePassing, GCNConv, global_add_pool, global_max_pool, global_mean_pool
from torch_geometric.data import Data
from typing import Optional
import math
from collections import OrderedDict


class AntiSymmetricConv(MessagePassing):
    def __init__(self, 
                 in_channels: int,
                 num_iters: int = 1, 
                 gamma: float = 0.1, 
                 epsilon : float = 0.1, 
                 activ_fun: str = 'tanh', # it should be monotonically non-decreasing
                 gcn_conv: bool = False,
                 bias: bool = True) -> None:

        super().__init__(aggr = 'add')
        self.W = Parameter(torch.empty((in_channels, in_channels)), requires_grad=True)
        self.bias = Parameter(torch.empty(in_channels), requires_grad=True) if bias else None

        self.lin = Linear(in_channels, in_channels, bias=False) # for simple aggregation
        self.I = Parameter(torch.eye(in_channels), requires_grad=False)

        self.gcn_conv = GCNConv(in_channels, in_channels, bias=False) if gcn_conv else None

        self.num_iters = num_iters
        self.gamma = gamma
        self.epsilon = epsilon
        self.activation = getattr(torch, activ_fun)

        self.reset_parameters()

    def __repr__(self):
        return f'self.__class__.__name__(\
                        in_channels: {self.in_channels}, \
                        num_iters: {self.num_iters}, \
                        gamma: {self.gamma.data if isinstance(self.gamma, Parameter) else self.gamma}, \
                        epsilon: {self.epsilon.data if isinstance(self.epsilon, Parameter) else self.epsilon}, \
                        activ_fun: {self.activation}, \
                        gcn_conv: {self.gcn_conv}, \
                        bias: {self.bias is not None})'

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        antisymmetric_W = self.W - self.W.T - self.gamma * self.I

        for _ in range(self.num_iters):
            if self.gcn_conv is None:
                # simple aggregation
                neigh_x = self.lin(x) 
                neigh_x = self.propagate(edge_index, x=neigh_x, edge_weight=edge_weight)
            else:
                # gcn aggregation
                neigh_x = self.gcn_conv(x, edge_index=edge_index, edge_weight=edge_weight)

            conv = x @ antisymmetric_W.T + neigh_x

            if self.bias is not None:
                conv += self.bias

            x = x + self.epsilon * self.activation(conv)
        return x

    def message(self, x_j: torch.Tensor, edge_weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j

    def reset_parameters(self):
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        init.kaiming_uniform_(self.W, a=math.sqrt(5))
        self.lin.reset_parameters()
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.W)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)


class GraphAntiSymmetricNN_GraphProp(Module):
    def __init__(self, 
                 input_dim: int,
                 output_dim: int,
                 hidden_dim: Optional[int] = None,
                 num_layers: int = 1,
                 epsilon: float = 0.1,
                 gamma: float = 0.1,
                 node_level_task: bool = False,
                 activ_fun: str = 'tanh',
                 gcn_norm: bool = False,
                 bias: bool = True) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.epsilon = epsilon
        self.gamma = gamma
        self.activ_fun = activ_fun
        self.bias = bias

        inp = self.input_dim
        self.emb = None
        if self.hidden_dim is not None:
            self.emb = Linear(self.input_dim, self.hidden_dim)
            inp = self.hidden_dim

        self.conv = AntiSymmetricConv(in_channels = inp,
                                  num_iters = self.num_layers,
                                  gamma = self.gamma,
                                  epsilon = self.epsilon,
                                  activ_fun = self.activ_fun,
                                  gcn_conv = gcn_norm,
                                  bias = self.bias)
            
        self.node_level_task = node_level_task 
        if self.node_level_task:
            self.readout = Sequential(OrderedDict([
                ('L1', Linear(inp, inp // 2)),
                ('LeakyReLU1', LeakyReLU()),
                ('L2', Linear(inp // 2, self.output_dim)),
                ('LeakyReLU2', LeakyReLU())
            ]))
        else:
            self.readout = Sequential(OrderedDict([
                ('L1', Linear(inp * 3, (inp * 3) // 2)),
                ('LeakyReLU1', LeakyReLU()),
                ('L2', Linear((inp * 3) // 2, self.output_dim)),
                ('LeakyReLU2', LeakyReLU())
            ]))


    def forward(self, data: Data) -> torch.Tensor:
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.emb(x) if self.emb else x
        x = self.conv(x, edge_index)

        if not self.node_level_task:
            x = torch.cat([global_add_pool(x, batch), global_max_pool(x, batch), global_mean_pool(x, batch)], dim=1)

        x = self.readout(x)

        return x

