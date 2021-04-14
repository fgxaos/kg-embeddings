### LIBRARIES ###
# Global libraries
import torch
import torch.nn as nn
import torch.nn.functional as F

# Custom libraries
from models.rgcn.utils import uniform
from models.rgcn.rgcn_conv import RGCNConv

### CLASS DEFINITION ###
class RGCN(nn.Module):
    def __init__(
        self,
        n_entities,
        n_relations,
        n_bases,
        dropout,
        reg_ratio=0.0,
    ):
        super(RGCN, self).__init__()

        self.entity_embedding = nn.Embedding(n_entities, 100)
        self.relation_embedding = nn.Parameter(torch.Tensor(n_relations, 100))

        nn.init.xavier_uniform_(
            self.relation_embedding, gain=nn.init.calculate_gain("relu")
        )

        self.conv1 = RGCNConv(100, 100, n_relations * 2, num_bases=n_bases)
        self.conv2 = RGCNConv(100, 100, n_relations * 2, num_bases=n_bases)

        self.dropout_ratio = dropout
        self.reg_ratio = reg_ratio

    def forward(self, entity, edge_idx, edge_type, edge_norm):
        x = self.entity_embedding(entity)
        x = self.conv1(x, edge_idx, edge_type, edge_norm)
        x = F.relu(self.conv1(x, edge_idx, edge_type, edge_norm))
        x = F.dropout(x, p=self.dropout_ratio, training=self.training)
        x = self.conv2(x, edge_idx, edge_type, edge_norm)

        return x

    def distmult(self, embedding, triplets):
        s = embedding[triplets[:, 0]]
        r = self.relation_embedding[triplets[:, 1]]
        o = embedding[triplets[:, 2]]
        score = torch.sum(s * r * o, dim=1)
        return score

    def score_loss(self, embedding, triplets, target):
        score = self.distmult(embedding, triplets)
        return F.binary_cross_entropy_with_logits(score, target)

    def reg_loss(self, embedding):
        return torch.mean(embedding.pow(2) + torch.mean(self.relation_embedding.pow(2)))
