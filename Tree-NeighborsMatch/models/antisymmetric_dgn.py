import torch

from torch.nn import Module, Parameter, init, Linear, ModuleList
from torch_geometric.nn import MessagePassing, GCNConv
from torch_geometric.transforms import GCNNorm
from torch_geometric.data import Data
from typing import Optional
import math


class AntiSymmetricConv(MessagePassing):
    '''
    VERSIONE 22/2/2022
    La matrice dei pesi antisimmetrica su previous hidden state
    z^l_v = (W^l_H - (W^l_H)^T - g * I) *  h^(l-1)_v + sum(W^(l-1)_n  h^(l-1)_u) + b
    h^l_v = h^(l-1)_v + e * tanh(z^l_v)
    '''
    def __init__(self, 
                 in_channels: int,
                 num_iters: int = 1, 
                 gamma: float = 0.1, 
                 epsilon : float = 0.1, 
                 activ_fun: str = 'tanh', # it should be monotonically non-decreasing
                 train_weights: bool = True,
                 gcn_conv: bool = False,
                 bias: bool = False) -> None:

        super().__init__(aggr = 'add')
        #super().__init__(aggr = 'max')
        self.train_weights = train_weights
        self.W = Parameter(torch.empty((in_channels, in_channels)), requires_grad=self.train_weights)
        self.bias = Parameter(torch.empty(in_channels), requires_grad=self.train_weights) if bias else None

        self.lin = Linear(in_channels, in_channels, bias=False) # Computes: W^(l-1)_n  h^(l-1)_u
        if not self.train_weights:
            self.lin.weight.requires_grad = False

        self.I = Parameter(torch.eye(in_channels), requires_grad=False)

        self.gcn_conv = GCNConv(in_channels, in_channels, bias=False) if gcn_conv else None
        if gcn_conv and not train_weights:
            raise NotImplementedError('Non implementato untrained con GCN')

        self.reset_parameters()

        self.num_iters = num_iters
        self.gamma = gamma
        self.epsilon = epsilon
        self.activation = getattr(torch, activ_fun)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        #TODO: matrice antisimmetrica la costruisco anche facendo cosi: W.triu(1) - W.triu(1).transpose(-1, -2), ma e' piu' lento
        antisymmetric_W = self.W - self.W.T - self.gamma * self.I

        for _ in range(self.num_iters):
            if self.gcn_conv is None:
                neigh_x = self.lin(x) 
                neigh_x = self.propagate(edge_index, x=neigh_x, edge_weight=edge_weight)
            else:
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


class GraphAntiSymmetricNN(Module):
    def __init__(self, 
                 input_dim: int,
                 output_dim: int,
                 hidden_dim: Optional[int] = None,
                 num_layers: int = 1,
                 weight_sharing: bool = True,
                 epsilon: float = 0.1,
                 gamma: float = 0.1,
                 activ_fun: str = 'tanh',
                 trainable_conv_layer: bool = True,
                 gcn_norm: bool = False,
                 bias: bool = False) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.epsilon = epsilon
        self.gamma = gamma
        self.activ_fun = activ_fun
        self.trainable_conv_layer = trainable_conv_layer
        #self.gcn_norm = GCNNorm() if gcn_norm else None
        self.bias = bias

        inp = self.input_dim
        self.emb = None
        if self.hidden_dim is not None:
            self.emb = Linear(self.input_dim, self.hidden_dim)
            if not self.trainable_conv_layer:
                self.emb.weight.requires_grad = False 
                self.emb.bias.requires_grad = False
            inp = self.hidden_dim

        self.conv = ModuleList()
        if weight_sharing:
            self.conv.append(
                AntiSymmetricConv(in_channels = inp,
                                  num_iters = self.num_layers,
                                  gamma = self.gamma,
                                  epsilon = self.epsilon,
                                  activ_fun = self.activ_fun,
                                  train_weights = self.trainable_conv_layer,
                                  gcn_conv = gcn_norm,
                                  bias = self.bias)
            )
        else:
            for _ in range(num_layers):
                self.conv.append(
                    AntiSymmetricConv(in_channels = inp,
                                    num_iters = 1,
                                    gamma = self.gamma,
                                    epsilon = self.epsilon,
                                    activ_fun = self.activ_fun,
                                    train_weights = self.trainable_conv_layer,
                                    gcn_conv = gcn_norm,
                                    bias = self.bias)
                )
            
        self.readout = Linear(inp, self.output_dim)

    def forward(self, data: Data) -> torch.Tensor:
        #if self.gcn_norm is not None:
        #    data = self.gcn_norm(data)

        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_weight

        x = self.emb(x) if self.emb else x # TODO: activation function?
        for conv in self.conv:
            x = conv(x, edge_index, edge_weight)
        x = self.readout(x)

        return x

"""
##class AntiSymmetricConv(MessagePassing):
##    '''
##    VERSIONE 23/02/2022
##    La matrice dei pesi antisimmetrica sia su previous hidden state che su vicinato
##    z^l_1 = (W^l_1 - (W^l_1)^T - g * I) *  h^(l-1)_v 
##    z^l_2 = (W^l_2 - (W^l_2)^T - g * I) *  sum(h^(l-1)_u)
##    h^l_v = h^(l-1)_v + e * sigma(z^l_1 + z^l_2 + b)
##    '''
##    def __init__(self, 
##                 in_channels: int, 
##                 num_iters: int, 
##                 gamma: float = 0.1, 
##                 epsilon : float = 0.1, 
##                 activ_fun: str = 'tanh', # it should be monotonically non-decreasing
##                 bias: bool = False) -> None:
##
##        super().__init__(aggr = 'add')
##        self.W1 = Parameter(torch.empty((in_channels, in_channels)), requires_grad=True)
##        self.W2 = Parameter(torch.empty((in_channels, in_channels)), requires_grad=True)
##        
##        self.bias = Parameter(torch.empty(in_channels), requires_grad=True) if bias else None
##        self.I = Parameter(torch.eye(in_channels), requires_grad=False)
##
##        self.reset_parameters()
##
##        self.num_iters = num_iters
##        self.gamma = gamma
##        self.epsilon = epsilon
##        self.activation = getattr(torch, activ_fun)
##
##    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
##        for _ in range(self.num_iters):
##            neigh_x = self.propagate(edge_index, x=x)
##
##            #TODO: matrice antisimmetrica la costruisco anche facendo cosi: W.triu(1) - W.triu(1).transpose(-1, -2), ma e' piu' lento
##            antisymmetric_W1 = self.W1 - self.W1.T - self.gamma * self.I
##            antisymmetric_W2 = self.W2 - self.W2.T - self.gamma * self.I
##
##            conv = (x @ antisymmetric_W1.T) + (neigh_x @ antisymmetric_W2.T)
##            
##            if self.bias is not None:
##                conv += self.bias
##
##            x = x + self.epsilon * self.activation(conv)
##        return x
##
##    def message(self, x_j: torch.Tensor) -> torch.Tensor:
##        return x_j
##
##    def reset_parameters(self):
##        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
##        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
##        # https://github.com/pytorch/pytorch/issues/57109
##        init.kaiming_uniform_(self.W1, a=math.sqrt(5))
##        init.kaiming_uniform_(self.W2, a=math.sqrt(5))
##        if self.bias is not None:
##            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
##            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
##            init.uniform_(self.bias, -bound, bound)
##
##
###########
##class AntiSymmetricConv(MessagePassing):
##    '''
##    VERSIONE 24/2/2022
##    La matrice dei pesi antisimmetrica sul vicinato
##    z^l_v = (W^l_x *  h^(l-1)_v + (W^l_H - (W^l_H)^T - g * I) * sum(h^(l-1)_u) + b
##    h^l_v = h^(l-1)_v + e * tanh(z^l_v)
##    '''
##    def __init__(self, 
##                 in_channels: int, 
##                 num_iters: int, 
##                 gamma: float = 0.1, 
##                 epsilon : float = 0.1, 
##                 activ_fun: str = 'tanh', # it should be monotonically non-decreasing
##                 bias: bool = False) -> None:
##
##        super().__init__(aggr = 'add')
##        self.W = Parameter(torch.empty((in_channels, in_channels)), requires_grad=True)
##        self.lin = Linear(in_channels, in_channels, bias=False) # Computes: W^(l-1)_n  h^(l-1)_u
##        self.bias = Parameter(torch.empty(in_channels), requires_grad=True) if bias else None
##        self.I = Parameter(torch.eye(in_channels), requires_grad=False)
##
##        self.reset_parameters()
##
##        self.num_iters = num_iters
##        self.gamma = gamma
##        self.epsilon = epsilon
##        self.activation = getattr(torch, activ_fun)
##
##    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
##        for _ in range(self.num_iters):
##            neigh_x = self.propagate(edge_index, x=x)
##            x_ = self.lin(x) 
##
##            #TODO: matrice antisimmetrica la costruisco anche facendo cosi: W.triu(1) - W.triu(1).transpose(-1, -2), ma e' piu' lento
##            antisymmetric_W = self.W - self.W.T - self.gamma * self.I
##
##            conv = x_ + neigh_x @ antisymmetric_W.T
##            
##            if self.bias is not None:
##                conv += self.bias
##
##            x = x + self.epsilon * self.activation(conv)
##        return x
##
##    def message(self, x_j: torch.Tensor) -> torch.Tensor:
##        return x_j
##
##    def reset_parameters(self):
##        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
##        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
##        # https://github.com/pytorch/pytorch/issues/57109
##        init.kaiming_uniform_(self.W, a=math.sqrt(5))
##        self.lin.reset_parameters()
##        if self.bias is not None:
##            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
##            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
##            init.uniform_(self.bias, -bound, bound)
"""   

