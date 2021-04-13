### LIBRARIES ###
# Global libraries
import torch
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl

# Custom libraries
from models.rgcn.utils import compute_metrics, uniform
from models.rgcn.rgcn_conv import RGCNConv

### CLASS DEFINITION ###
class RGCN(pl.LightningModule):
    def __init__(
        self,
        n_entities,
        n_relations,
        all_triples,
        n_bases,
        dropout,
        learning_rate,
        reg_ratio=0.0,
    ):
        super(RGCN, self).__init__()

        self.entity_embedding = nn.Embedding(n_entities, 100)
        self.relation_embedding = nn.Parameter(torch.Tensor(n_relations, 100))

        nn.init.xavier_uniform_(
            self.relation_embedding, gain=nn.init.calculate_gain("relu")
        )

        self.conv1 = RGCNConv(100, 100, n_relations * 2, n_bases=n_bases)
        self.conv2 = RGCNConv(100, 100, n_relations * 2, n_bases=n_bases)

        self.dropout_ratio = dropout
        self.reg_ratio = reg_ratio

        self.learning_rate = learning_rate
        self.lr = None

        # self.save_hyperparameters()

        self.all_triples = all_triples

    def forward(self, entity, edge_idx, edge_type, edge_norm):
        x = self.entity_embedding(entity)
        x = F.relu(self.conv1(x, edge_idx, edge_type, edge_norm))
        x = F.dropout(x, p=self.dropout_ratio, training=self.training)
        x = self.conv2(x, edge_idx, edge_type, edge_norm)

        return x

    def training_step(self, batch, batch_idx):
        entity, edge_idx, edge_type, edge_norm, samples, labels = batch
        entity_embedding = self(entity, edge_idx, edge_type, edge_norm)
        loss = self.score_loss(
            entity_embedding, samples, labels
        ) + self.reg_ratio * self.reg_loss(entity_embedding)

        return loss

    def validation_step(self, batch, batch_idx):
        entity, edge_idx, edge_type, edge_norm, val_triplets = batch
        edge_idx = edge_idx.view(2, -1)
        entity_embedding = self(entity, edge_idx, edge_type, edge_norm)
        metrics = compute_metrics(
            entity_embedding,
            self.relation_embedding,
            val_triplets,
            self.all_triples,
            hits=[1, 3, 10],
        )
        self.log("MRR", metrics["mrr"])
        self.log("MR", metrics["mr"])
        self.log("HITS@1", metrics["hits@1"])
        self.log("HITS@3", metrics["hits@3"])
        self.log("HITS@10", metrics["hits@10"])
        self.log("val_score", 1.0 / metrics["mrr"])

    def test_step(self, batch, batch_idx):
        entity, edge_idx, edge_type, edge_norm, test_triplets = batch
        entity_embedding = self(entity, edge_idx, edge_type, edge_norm)
        metrics = compute_metrics(
            entity_embedding,
            self.relation_embedding,
            test_triplets,
            self.all_triples,
            hits=[1, 3, 10],
        )
        self.log("MRR", metrics["mrr"])
        self.log("MR", metrics["mr"])
        self.log("HITS@1", metrics["hits@1"])
        self.log("HITS@3", metrics["hits@3"])
        self.log("HITS@10", metrics["hits@10"])
        self.log("val_score", 1.0 / metrics["mrr"])

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr or self.learning_rate)

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
