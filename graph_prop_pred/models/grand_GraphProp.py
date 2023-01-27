import torch
from torch.nn import Module, Linear, Sequential, LeakyReLU
from torch_geometric.nn import global_add_pool, global_max_pool, global_mean_pool
from torch_geometric.data import Data
from torch_geometric.nn import GATConv
#from torchdyn.models import NeuralDE
from typing import Optional
from torch import tanh
from collections import OrderedDict

from torchdyn.core import NeuralODE

from torch import nn


class AttentionLaplacianODEFunc(Module):

    def __init__(self, input_size, opt) -> None:
        super().__init__()
        self.attention = GATConv(
            in_channels = input_size, 
            out_channels = input_size, 
            heads = opt['heads'],
            negative_slope = opt['leaky_relu_slope']
        )
        self.opt = opt
        self.alpha_train = nn.Parameter(torch.tensor(0.0))
        self.beta_train = nn.Parameter(torch.tensor(0.0))
        self.edge_index = None
        self.x0 = None

    def forward(self, t, x):  # the t param is needed by the ODE solver.
        ax = self.attention(x, self.edge_index)
        if not self.opt['no_alpha_sigmoid']:
            alpha = torch.sigmoid(self.alpha_train)
        else:
            alpha = self.alpha_train
        f = alpha * (ax - x)
        if self.opt['add_source']:
            f = f + self.beta_train * self.x0
        return f


class GRAND_GraphProp(Module):
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 hidden_dim: Optional[int] = None,
                 epsilon: float = 0.1,
                 iterations: int = 1,
                 cached: bool = False,
                 node_level_task: bool = False) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.epsilon = epsilon
        self.iterations = iterations
        self.cached = cached

        inp = self.input_dim
        self.emb = None
        if self.hidden_dim is not None:
            self.emb = Linear(self.input_dim, self.hidden_dim)
            inp = self.hidden_dim

        opt = {
            'hidden_dim': inp,
            'block': 'attention',
            'add_source': False,
            'beltrami': False,
            'no_alpha_sigmoid': False,
            'attention_type': 'cosine_sim',
            'leaky_relu_slope': 0.2,
            'reweight_attention': False,
            'square_plus': False,
            'attention_norm_idx': 0,
            'heads': 1,
        }

        self.func = AttentionLaplacianODEFunc(inp, opt)

        t_span = [0.]
        for _ in range(self.iterations):
            t_span.append(t_span[-1] + self.epsilon)
        self.t_span = torch.tensor(t_span) # the evaluation timesteps
        self.conv = NeuralODE(self.func, sensitivity='adjoint', solver='rk4', solver_adjoint='rk4', return_t_eval=False)

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

        if (not self.cached) or self.func.edge_index is None:
            self.func.edge_index = edge_index

        x = self.conv(x, t_span=self.t_span)
        x = x[-1] # conv returns node states at each evaluation step

        x = tanh(x)

        if not self.node_level_task:
            x = torch.cat([global_add_pool(x, batch), global_max_pool(x, batch), global_mean_pool(x, batch)], dim=1)

        x = self.readout(x)

        return x
 
