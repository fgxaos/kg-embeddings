### LIBRARIES ###
# Global libraries
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl

### CLASS DEFINITION ###
class ComplEx(pl.LightningModule):
    """ComplEx model.

    Model introduced in "Knowledge Graph Completion via Complex Tensor Factorization".

    Link to the paper: https://arxiv.org/abs/1702.06879
    """

    def __init__(
        self,
        n_entity: int,
        n_relation: int,
        hidden_dim: int,
        gamma: float,
        learning_rate: float,
        double_entity_embedding: bool = True,
        double_relation_embedding: bool = True,
        negative_adversarial_sampling: bool = False,
        adversarial_temperature: float = 1.0,
        uni_weight: bool = False,
        regularization: float = 0.0,
    ):
        """Initiates the ComplEx model.

        Args:
            n_entity: int
                total number of entities
            n_relation: int
                total number of relations
            hidden_dim: int
                dimension of the embeddings (entities and relations)
            gamma: float

            learning_rate: float
                learning rate to use for the optimizer
            double_entity_embedding: bool
                whether to double the size of the entity embedding
            double_relation_embedding: bool
                whether to double the size of the relation embedding
            negative_adversarial_sampling: bool

            adversarial_temperature: float

            uni_weight: bool

            regularization: float

        """
        super(ComplEx, self).__init__()
        self.n_entity = n_entity
        self.n_relation = n_relation
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        self.negative_adversarial_sampling = negative_adversarial_sampling
        self.adversarial_temperature = adversarial_temperature
        self.epsilon = 2.0
        self.uni_weight = uni_weight
        self.regularization = regularization
        self.lr = None

        self.gamma = nn.Parameter(torch.Tensor([gamma]), requires_grad=False)

        self.embedding_range = nn.Parameter(
            torch.Tensor([(self.gamma.item() + self.epsilon) / hidden_dim]),
            requires_grad=False,
        )

        self.entity_dim = hidden_dim * 2 if double_entity_embedding else hidden_dim
        self.relation_dim = hidden_dim * 2 if double_relation_embedding else hidden_dim

        self.entity_embedding = nn.Parameter(torch.zeros(n_entity, self.entity_dim))
        # TODO: Try to replace the uniform initialization with a Xavier
        nn.init.uniform_(
            tensor=self.entity_embedding,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item(),
        )

        self.relation_embedding = nn.Parameter(
            torch.zeros(n_relation, self.relation_dim)
        )
        # TODO: Try to replace the uniform initialization with a Xavier
        nn.init.uniform_(
            tensor=self.relation_embedding,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item(),
        )

        if not double_entity_embedding or not double_relation_embedding:
            raise ValueError(
                "ComplEx should use --double_entity_embedding and --double_relation_embedding"
            )

        self.save_hyperparameters()

    def forward(self, sample, mode="single"):
        if mode == "single":
            batch_size, negative_sample_size = sample.size(0), 1

            head = torch.index_select(
                self.entity_embedding, dim=0, index=sample[:, 0]
            ).unsqueeze(1)

            relation = torch.index_select(
                self.relation_embedding, dim=0, index=sample[:, 1]
            ).unsqueeze(1)

            tail = torch.index_select(
                self.entity_embedding, dim=0, index=sample[:, 2]
            ).unsqueeze(1)

        elif mode == "head-batch":
            tail_part, head_part = sample
            batch_size, negative_sample_size = head_part.size(0), head_part.size(1)

            head = torch.index_select(
                self.entity_embedding, dim=0, index=head_part.view(-1)
            ).view(batch_size, negative_sample_size, -1)

            relation = torch.index_select(
                self.relation_embedding, dim=0, index=tail_part[:, 1]
            ).unsqueeze(1)

            tail = torch.index_select(
                self.entity_embedding, dim=0, index=tail_part[:, 2]
            ).unsqueeze(1)

        elif mode == "tail-batch":
            head_part, tail_part = sample
            batch_size, negative_sample_size = tail_part.size(0), tail_part.size(1)

            head = torch.index_select(
                self.entity_embedding, dim=0, index=head_part[:, 0]
            ).unsqueeze(1)

            relation = torch.index_select(
                self.relation_embedding, dim=0, index=head_part[:, 1]
            ).unsqueeze(1)

            tail = torch.index_select(
                self.entity_embedding, dim=0, index=tail_part.view(-1)
            ).view(batch_size, negative_sample_size, -1)

        else:
            raise ValueError(f"Mode {mode} not supported")

        re_head, im_head = torch.chunk(head, 2, dim=2)
        re_relation, im_relation = torch.chunk(relation, 2, dim=2)
        re_tail, im_tail = torch.chunk(tail, 2, dim=2)

        if mode == "head-batch":
            re_score = re_relation * re_tail + im_relation * im_tail
            im_score = re_relation * im_tail - im_relation * re_tail
            score = re_head * re_score + im_head * im_score
        else:
            re_score = re_head * re_relation - im_head * im_relation
            im_score = re_head * im_relation + im_head * re_relation
            score = re_score * re_tail + im_score * im_tail

        score = score.sum(dim=2)

        return score

    def training_step(self, batch, batch_idx):
        batch_loss = 0
        for sub_batch in batch:
            positive_sample, negative_sample, subsampling_weight, mode = sub_batch
            negative_score = self((positive_sample, negative_sample), mode=mode)

            if self.negative_adversarial_sampling:
                # Don't apply back-propagation on the sampling weight
                negative_score = (
                    F.softmax(
                        negative_score * self.adversarial_temperature, dim=1
                    ).detach()
                    * F.logsigmoid(-negative_score)
                ).sum(dim=1)
            else:
                negative_score = F.logsigmoid(-negative_score).mean(dim=1)

            positive_score = self(positive_sample)
            positive_score = F.logsigmoid(positive_score).squeeze(dim=1)

            if self.uni_weight:
                positive_sample_loss = -positive_score.mean()
                negative_sample_loss = -negative_score.mean()
            else:
                positive_sample_loss = (
                    -(subsampling_weight * positive_score).sum()
                    / subsampling_weight.sum()
                )
                negative_sample_loss = (
                    -(subsampling_weight * negative_score).sum()
                    / subsampling_weight.sum()
                )

            loss = (positive_sample_loss + negative_sample_loss) / 2

            if self.regularization != 0.0:
                regularization = args.regularization * (
                    self.entity_embedding.norm(p=3) ** 3
                    + self.relation_embedding.norm(p=3).norm(p=3) ** 3
                )
                loss += regularization

            batch_loss += loss

        return batch_loss / 2

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            val_score = 0
            for sub_batch in batch:
                positive_sample, negative_sample, filter_bias, mode = sub_batch
                score = self((positive_sample, negative_sample), mode)
                score += filter_bias

                # Explicitly sort all the entities to ensure that there
                # is no text exposure bias
                argsort = torch.argsort(score, dim=1, descending=True)

                if mode == "head-batch":
                    positive_arg = positive_sample[:, 0]
                elif mode == "tail-batch":
                    positive_arg = positive_sample[:, 2]
                else:
                    raise ValueError(f"Mode {mode} not supported")

                for i in range(len(batch)):
                    ranking = torch.nonzero(argsort[i, :] == positive_arg[i])
                    assert ranking.size(0) == 1

                    # `ranking + 1` is the true ranking, used in evaluation metrics
                    ranking = 1 + ranking.item()
                    val_score += ranking
                    self.log("MRR", 1.0 / ranking)
                    self.log("MR", float(ranking))
                    self.log("HITS@1", 1.0 if ranking <= 1 else 0.0)
                    self.log("HITS@3", 1.0 if ranking <= 3 else 0.0)
                    self.log("HITS@10", 1.0 if ranking <= 10 else 0.0)
            self.log("val_score", val_score)

    def test_step(self, batch, batch_idx):
        with torch.no_grad():
            test_score = 0
            for sub_batch in batch:
                positive_sample, negative_sample, filter_bias, mode = sub_batch
                score = self((positive_sample, negative_sample), mode)
                score += filter_bias

                # Explicitly sort all the entities to ensure that there
                # is no text exposure bias
                argsort = torch.argsort(score, dim=1, descending=True)

                if mode == "head-batch":
                    positive_arg = positive_sample[:, 0]
                elif mode == "tail-batch":
                    positive_arg = positive_sample[:, 2]
                else:
                    raise ValueError(f"Mode {mode} not supported")

                for i in range(len(batch)):
                    ranking = torch.nonzero(argsort[i, :] == positive_arg[i])
                    assert ranking.size(0) == 1

                    # `ranking + 1` is the true ranking, used in evaluation metrics
                    ranking = 1 + ranking.item()
                    test_score += ranking
                    self.log("MRR", 1.0 / ranking)
                    self.log("MR", float(ranking))
                    self.log("HITS@1", 1.0 if ranking <= 1 else 0.0)
                    self.log("HITS@3", 1.0 if ranking <= 3 else 0.0)
                    self.log("HITS@10", 1.0 if ranking <= 10 else 0.0)
            self.log("test_score", test_score)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr or self.learning_rate)
