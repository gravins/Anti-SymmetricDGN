import torch

from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import Adj, OptTensor
from torch_sparse import SparseTensor, matmul
from torch_geometric.data import Data
from torch.nn import Module, Linear
from torch import Tensor, tanh
from typing import Optional



class DGCConv(MessagePassing):
    def __init__(self,
                 input_dim: int,
                 epsilon: float = 0.1,
                 iterations: int = 1,
                 cached: bool = False,
                 add_self_loops: bool = True) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.epsilon = epsilon
        self.iterations = iterations
        self.cached = cached
        self.add_self_loops = add_self_loops

        self.cached_edge_index = None

    def forward(self, x: Tensor, edge_index: Adj, edge_weight: OptTensor = None) -> Tensor:

        if self.cached_edge_index is None:
            edge_index, edge_weight = gcn_norm(  # yapf: disable
                edge_index, edge_weight=edge_weight, num_nodes=x.size(self.node_dim), improved=True,
                add_self_loops=self.add_self_loops, dtype=x.dtype)

            if self.cached:
                self.cached_edge_index = (edge_index, edge_weight)
        else:
            edge_index, edge_weight = self.cached_edge_index

        for _ in range(self.iterations):
            epsilon_LX = self.epsilon * self.propagate(edge_index, x=x, edge_weight=edge_weight)
            x = x - epsilon_LX
           
        return x

    def message(self, x_j: Tensor, edge_weight: OptTensor) -> Tensor:
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        return matmul(adj_t, x, reduce=self.aggr)


class DGC(Module):
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 hidden_dim: Optional[int] = None,
                 epsilon: float = 0.1,
                 iterations: int = 1,
                 cached: bool = False,
                 add_self_loops: bool = True) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.epsilon = epsilon
        self.iterations = iterations
        self.cached = cached
        self.add_self_loops = add_self_loops

        inp = self.input_dim
        self.emb = None
        if self.hidden_dim is not None:
            self.emb = Linear(self.input_dim, self.hidden_dim)
            inp = self.hidden_dim

        self.conv = DGCConv(inp, self.epsilon, self.iterations, self.cached, self.add_self_loops)

        self.readout = Linear(inp, self.output_dim)

    def forward(self, data: Data) -> torch.Tensor:
        x, edge_index= data.x, data.edge_index

        x = self.emb(x) if self.emb else x

        x = tanh(self.conv(x, edge_index))
        x = self.readout(x)

        return x