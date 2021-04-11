### LIBRARIES ###
# Global libraries
import torch
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl

### UTILS CLASS ###
class Flatten(nn.Module):
    def forward(self, x):
        n, _, _, _ = x.size()
        x = x.view(n, -1)
        return x


### CLASS DEFINITION ###
class ConvE(pl.LightningModule):
    """ConvE model.

    Model introduced in "Convolutional 2D Knowledge Graph Embeddings".

    Link to the paper: https://arxiv.org/pdf/1412.6575
    """

    def __init__(
        self,
        n_entities: int,
        n_relations: int,
        embedding_size_h: int,
        embedding_size_w: int,
        conv_channels: int,
        conv_kernel_size: int,
        embedding_dropout: float,
        feature_map_dropout: float,
        proj_layer_dropout: float,
        regularization: float,
        learning_rate: float,
        use_wandb: bool,
        label_smooth: float,
    ):
        """Initiates a ConvE model.

        Args:
            n_entities: int
            n_relations: int
            embedding_size_h: int
            embedding_size_w: int
            conv_channels: int
            conv_kernel_size: int
            embedding_dropout: float
            feature_map_dropout: float
            proj_layer_dropout: float
            regularization: float
            learning_rate: float
            use_wandb: bool
                whether to use wandb
            label_smooth: float
        """
        super(ConvE, self).__init__()

        self.n_entities = n_entities
        self.n_relations = n_relations
        self.embedding_size_h = embedding_size_h
        self.embedding_size_w = embedding_size_w

        self.lr = None
        self.learning_rate = learning_rate

        self.label_smooth = label_smooth
        self.use_wandb = use_wandb
        self.criterion = StableBCELoss()

        embedding_size = embedding_size_h * embedding_size_w
        flattened_size = (
            (embedding_size_w * 2 - conv_kernel_size + 1)
            * (embedding_size_h - conv_kernel_size + 1)
            * conv_channels
        )

        self.embed_e = nn.Embedding(
            num_embeddings=self.n_entities, embedding_dim=embedding_size
        )
        self.embed_r = nn.Embedding(
            num_embeddings=self.n_relations, embedding_dim=embedding_size
        )

        self.conv_e = nn.Sequential(
            nn.Dropout(p=embedding_dropout),
            nn.Conv2d(
                in_channels=1, out_channels=conv_channels, kernel_size=conv_kernel_size
            ),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=conv_channels),
            nn.Dropout2d(p=feature_map_dropout),
            Flatten(),
            nn.Linear(in_features=flattened_size, out_features=embedding_size),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=embedding_size),
            nn.Dropout(p=proj_layer_dropout),
        )

        self.save_hyperparameters()

    def forward(self, entity, relation):
        embed_e = self.embed_e(entity)
        embed_r = self.embed_r(relation)

        embed_e = embed_e.view(-1, self.embedding_size_w, self.embedding_size_h)
        embed_r = embed_r.view(-1, self.embedding_size_w, self.embedding_size_h)
        conv_input = torch.cat([embed_e, embed_r], dim=1).unsqueeze(1)
        out = self.conv_e(conv_input)

        scores = out.mm(self.embed_e.weight.t())

        return scores

    def training_step(self, batch, batch_idx):
        entity, relation, indices = batch
        y_multihot = torch.LongTensor(entity.shape[0], self.n_entities)
        y_multihot.zero_()
        y_multihot = y_multihot.scatter_(1, indices, 1)
        targets = (
            1 - self.label_smooth
        ) * y_multihot.float() + self.label_smooth / self.n_entities

        output = self(entity, relation)
        loss = self.criterion(output, targets)

        return loss

        # TODO: Add `self.label_smooth`

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            entity, relation, indices = batch
            output = self(entity, relation)

            val_score = 0
            ranks = []
            for i in range(entity.shape[0]):
                _, top_indices = output[i].topk(output.shape[1])
                for idx in indices[i]:
                    _, rank = (top_indices == idx).max(dim=0)
                    ranking = rank.item() + 1
                    val_score += ranking

                self.log("MRR", 1.0 / ranking)
                self.log("MR", float(ranking))
                self.log("HITS@1", 1.0 if ranking <= 1 else 0.0)
                self.log("HITS@3", 1.0 if ranking <= 3 else 0.0)
                self.log("HITS@10", 1.0 if ranking <= 10 else 0.0)
            self.log("val_score", val_score)

    def test_step(self, batch, batch_idx):
        with torch.no_grad():
            entity, relation, indices = batch
            output = self(entity, relation)

            test_score = 0
            ranks = []
            for i in range(entity.shape[0]):
                _, top_indices = output[i].topk(output.shape[1])
                for idx in indices[i]:
                    _, rank = (top_indices == idx).max(dim=0)
                    ranking = rank.item() + 1
                    test_score += ranking

                self.log("MRR", 1.0 / ranking)
                self.log("MR", float(ranking))
                self.log("HITS@1", 1.0 if ranking <= 1 else 0.0)
                self.log("HITS@3", 1.0 if ranking <= 3 else 0.0)
                self.log("HITS@10", 1.0 if ranking <= 10 else 0.0)
            self.log("test_score", test_score)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr or self.learning_rate)


class StableBCELoss(nn.modules.Module):
    def __init__(self):
        super(StableBCELoss, self).__init__()

    def forward(self, preds, target):
        neg_abs = -abs(preds)
        loss = preds.clamp(min=0) - preds * target + (1 + neg_abs.exp()).log()
        return loss.mean()