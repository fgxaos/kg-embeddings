### LIBRARIES ###
# Global libraries
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn.conv import MessagePassing

# Custom libraries
from models.rgcn.utils import uniform

### CLASS DEFINITION ###
class RGCNConv(MessagePassing):
    """Relational Graph Convolutional operator.

    From the paper "Modeling Relational Data with Graph Convolutional Networks"."""

    def __init__(
        self,
        in_channels,
        out_channels,
        n_relations,
        n_bases,
        root_weight=True,
        bias=True,
        **kwargs,
    ):
        """Initiates a RGCNConv operator.

        Args:
            in_channels: int
                size of each input sample
            out_channels: int
                size of each output sample
            n_relations: int
                number of relations
            n_bases: int
                number of bases used for basis-decomposition
            root_weight: bool
                if `False`, the layer will not add transformed root
                node features to the output.
            bias: bool
                if `False`, the layer will not learn an additive bias
        """
        super(RGCNConv, self).__init__(aggr="mean", **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_relations = n_relations
        self.n_bases = n_bases

        self.basis = nn.Parameter(torch.Tensor(n_bases, in_channels, out_channels))
        self.att = nn.Parameter(torch.Tensor(n_relations, n_bases))

        if root_weight:
            self.root = nn.Parameter(torch.Tensor(in_channels, out_channels))
        else:
            self.register_parameter("root", None)

        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        size = self.n_bases * self.in_channels
        uniform(size, self.basis)
        uniform(size, self.att)
        uniform(size, self.root)
        uniform(size, self.bias)

    def forward(self, x, edge_idx, edge_type, edge_norm=None, size=None):
        return self.propagate(
            edge_idx,
            edge_idx=edge_idx,
            size=size,
            x=x,
            edge_type=edge_type,
            edge_norm=edge_norm,
        )

    def message(self, x_j, edge_idx_j, edge_type, edge_norm):
        w = torch.matmul(self.att, self.basis.view(self.n_bases, -1))

        print("Shape x_j: ", x_j.shape)

        # If no node features are given, we implement a simple embedding
        # lookup based on the target node index and its edge type.
        if x_j is None:
            w = w.view(-1, self.out_channels)
            idx = edge_type * self.in_channels + edge_idx_j
            out = torch.index_select(w, 0, idx)
        else:
            w = w.view(self.n_relations, self.in_channels, self.out_channels)
            w = torch.idx_select(w, 0, edge_type)
            out = torch.bmm(x_j.unsqueeze(1), w).squeeze(-2)

        return out if edge_norm is None else out * edge_norm.view(-1, 1)

    def update(self, aggr_out, x):
        if self.root is not None:
            if x is None:
                out = aggr_out + self.root
            else:
                out = aggr_out + torch.matmul(x, self.root)

        if self.bias is not None:
            out = out + self.bias
        return out

    def __repr__(self):
        return f"{self.__class__.__name__} ({self.in_channels}, {self.out_channels}, n_relations={self.n_relations})"
