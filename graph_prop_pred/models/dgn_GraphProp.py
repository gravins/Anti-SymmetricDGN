import torch

from torch.nn import Module, Linear, ModuleList, Sequential, LeakyReLU
from torch_geometric.data import Data
import torch_geometric.nn as pyg_nn
from torch_geometric.nn import global_add_pool, global_max_pool, global_mean_pool
from typing import Optional
from collections import OrderedDict
from torch import tanh

class DGN_GraphProp(Module):
    def __init__(self, 
                 input_dim: int,
                 output_dim: int,
                 hidden_dim: Optional[int] = None,
                 num_layers: int = 1,
                 node_level_task: bool = False,
                 conv_layer: str = 'GCNConv',
                 alpha: Optional[float] = None) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.alpha = alpha
        
        inp = self.input_dim
        self.emb = None
        if self.hidden_dim is not None:
            self.emb = Linear(self.input_dim, self.hidden_dim)
            inp = self.hidden_dim

        self.conv_layer = getattr(pyg_nn, conv_layer)
        self.conv = ModuleList()
        for _ in range(num_layers):
            if conv_layer == 'GINConv':
                mlp = Linear(inp, inp)
                self.conv.append(self.conv_layer(nn=mlp,
                                                 train_eps = True))
            elif conv_layer == 'GCN2Conv':
                self.conv.append(self.conv_layer(channels = inp,
                                                 alpha = self.alpha))
            else:
                self.conv.append(self.conv_layer(in_channels = inp,
                                                 out_channels = inp))

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

        if self.conv_name == 'GCN2Conv':
            x_0 = x

        for conv in self.conv:
            if self.conv_name == 'GCN2Conv':
                x = tanh(conv(x, x_0, edge_index))
            else:
                x = tanh(conv(x, edge_index))

        if not self.node_level_task:
            x = torch.cat([global_add_pool(x, batch), global_max_pool(x, batch), global_mean_pool(x, batch)], dim=1)
        x = self.readout(x)

        return x

