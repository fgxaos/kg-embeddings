### LIBRARIES ###
# Global libraries
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn.conv import MessagePassing

# Custom libraries
from models.rgcn.utils import uniform

### CLASS DEFINITION ###
# class RGCNConv(MessagePassing):
#     """Relational Graph Convolutional operator.
#
#     From the paper "Modeling Relational Data with Graph Convolutional Networks"."""
#
#     def __init__(
#         self,
#         in_channels,
#         out_channels,
#         n_relations,
#         n_bases,
#         root_weight=True,
#         bias=True,
#         **kwargs,
#     ):
#         """Initiates a RGCNConv operator#.
#
#         Args:
#             in_channels: int
#                 size of each input sample
#             out_channels: int
#                 size of each output sample
#             n_relations: int
#                 number of relations
#             n_bases: int
#                 number of bases used for basis-decomposition
#             root_weight: bool
#                 if `False`, the layer will not add transformed root
#                 node features to the output.
#             bias: bool
#                 if `False`, the layer will not learn an additive bias
#         """
#         super(RGCNConv, self).__init__(aggr="mean", **kwargs)
#
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.n_relations = n_relations
#         self.n_bases = n_bases
#
#         self.basis = nn.Parameter(torch.Tensor(n_bases, in_channels, out_channels))
#         self.att = nn.Parameter(torch.Tensor(n_relations, n_bases))
#
#         if root_weight:
#             self.root = nn.Parameter(torch.Tensor(in_channels, out_channels))
#         else:
#             self.register_parameter("root", None)
#
#        if bias:
#             self.bias = nn.Parameter(torch.Tensor(out_channels))
#         else:
#             self.register_parameter("bias", None)
#
#         self.reset_parameters()
#
#     def reset_parameters(self):
#         size = self.n_bases * self.in_channels
#         uniform(size, self.basis)
#         uniform(size, self.att)
#         uniform(size, self.root)
#         uniform(size, self.bias)
#
#     def forward(self, x, edge_index, edge_type, edge_norm=None, size=None):
#         return self.propagate(
#             edge_index,
#             size=size,
#             x=x,
#             edge_type=edge_type,
#             edge_norm=edge_norm,
#         )
#
#     def message(self, x_j, edge_index_j, edge_type, edge_norm):
#         w = torch.matmul(self.att, self.basis.view(self.n_bases, -1))
#
#         # If no node features are given, we implement a simple embedding
#         # lookup based on the target node index and its edge type.
#         if x_j is None:
#             w = w.view(-1, self.out_channels)
#             index = edge_type * self.in_channels + edge_index_j
#             out = torch.index_select(w, 0, index)
#         else:
#             w = w.view(self.n_relations, self.in_channels, self.out_channels)
#             w = torch.index_select(w, 0, edge_type)
#             out = torch.bmm(x_j.unsqueeze(1), w).squeeze(-2)
#
#         return out if edge_norm is None else out * edge_norm.view(-1, 1)
#
#     def update(self, aggr_out, x):
#        if self.root is not None:
#             if x is None:
#                 out = aggr_out + self.root
#             else:
#                 out = aggr_out + torch.matmul(x, self.root)
#
#         if self.bias is not None:
#             out = out + self.bias
#         return out
#
#     def __repr__(self):
#         return f"{self.__class__.__name__} ({self.in_channels}, {self.out_channels}, n_relations={self.n_relations})"
class RGCNConv(MessagePassing):
    r"""The relational graph convolutional operator from the `"Modeling
    Relational Data with Graph Convolutional Networks"
    <https://arxiv.org/abs/1703.06103>`_ paper
    .. math::
        \mathbf{x}^{\prime}_i = \mathbf{\Theta}_{\textrm{root}} \cdot
        \mathbf{x}_i + \sum_{r \in \mathcal{R}} \sum_{j \in \mathcal{N}_r(i)}
        \frac{1}{|\mathcal{N}_r(i)|} \mathbf{\Theta}_r \cdot \mathbf{x}_j,
    where :math:`\mathcal{R}` denotes the set of relations, *i.e.* edge types.
    Edge type needs to be a one-dimensional :obj:`torch.long` tensor which
    stores a relation identifier
    :math:`\in \{ 0, \ldots, |\mathcal{R}| - 1\}` for each edge.
    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        num_relations (int): Number of relations.
        num_bases (int): Number of bases used for basis-decomposition.
        root_weight (bool, optional): If set to :obj:`False`, the layer will
            not add transformed root node features to the output.
            (default: :obj:`True`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        num_relations,
        num_bases,
        root_weight=True,
        bias=True,
        **kwargs
    ):
        super(RGCNConv, self).__init__(aggr="mean", **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_relations = num_relations
        self.num_bases = num_bases

        self.basis = nn.Parameter(torch.Tensor(num_bases, in_channels, out_channels))
        self.att = nn.Parameter(torch.Tensor(num_relations, num_bases))

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
        size = self.num_bases * self.in_channels
        uniform(size, self.basis)
        uniform(size, self.att)
        uniform(size, self.root)
        uniform(size, self.bias)

    def forward(self, x, edge_index, edge_type, edge_norm=None, size=None):
        """"""
        return self.propagate(
            edge_index, size=size, x=x, edge_type=edge_type, edge_norm=edge_norm
        )

    def message(self, x_j, edge_index_j, edge_type, edge_norm):
        w = torch.matmul(self.att, self.basis.view(self.num_bases, -1))

        # If no node features are given, we implement a simple embedding
        # loopkup based on the target node index and its edge type.
        if x_j is None:
            w = w.view(-1, self.out_channels)
            index = edge_type * self.in_channels + edge_index_j
            out = torch.index_select(w, 0, index)
        else:
            w = w.view(self.num_relations, self.in_channels, self.out_channels)
            w = torch.index_select(w, 0, edge_type)
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
        return "{}({}, {}, num_relations={})".format(
            self.__class__.__name__,
            self.in_channels,
            self.out_channels,
            self.num_relations,
        )
