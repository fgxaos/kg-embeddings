### LIBRARIES ###
# Global libraries
import math

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_scatter import scatter_add

### FUNCTION DEFINITIONS ###
def build_test_graph(n_nodes: int, n_relations: int, triples: list):
    """Builds the test graph used at validation and test.

    Args:
        n_nodes: int
            total number of nodes
        n_relations: int
            total number of relations
        triples: List[int, int, int]
            list of triples (head, relation, tail)
    Returns:
        data: ?
    """
    triples = np.array(triples)
    origin, relation, target = triples.transpose()

    origin = torch.Tensor(origin)
    relation = torch.Tensor(relation)
    target = torch.Tensor(target)

    origin, target = torch.cat((origin, target)), torch.cat((target, origin))
    relation = torch.cat((relation, relation + n_relations))

    edge_idx = torch.stack((origin, target))
    edge_type = relation

    data = {"edge_idx": edge_idx}
    data["entity"] = torch.from_numpy(np.arange(n_nodes))
    data["edge_type"] = edge_type
    data["edge_norm"] = edge_normalization(edge_type, edge_idx, n_nodes, n_relations)


def edge_normalization(edge_type, edge_idx, n_entity: int, n_relation: int):
    """Edge normalization trick.

    Args:
        edge_type:
        edge_idx:
        n_entity:
        n_relation:
    Returns:
        edge_norm:
    """
    edge_type = edge_type.long()
    edge_idx = edge_idx.long()
    one_hot = F.one_hot(edge_type, num_classes=2 * n_relation).to(torch.float)
    deg = scatter_add(one_hot, edge_idx[0], dim=0, dim_size=n_entity)
    idx = edge_type + torch.arange(len(edge_idx[0])) * (2 * n_relation)
    edge_norm = 1 / deg[edge_idx[0]].view(-1)[idx]

    return edge_norm


def compute_metrics(embedding, w, test_triplets, all_triplets, hits):
    """Computes MRR and Hit@ metrics.

    Args:
        embedding:
        w:
        test_triplets:
        all_triplets:
        hits:
    Returns:
        metrics: dict
            computed metrics (MRR and Hit@)
    """
    with torch.no_grad():
        n_entity = len(embedding)

        ranks_s = []
        ranks_o = []

        head_relation_triplets = all_triplets[:, :2]
        tail_relation_triplets = torch.stack(
            (all_triplets[:, 2], all_triplets[:, 1])
        ).transpose(0, 1)

        for test_triplet in test_triplets:
            # Perturb object
            subject, relation, object_ = test_triplet

            subject_relation = test_triplet[:2]
            delete_idx = torch.sum(head_relation_triplets == subject_relation, dim=1)
            delete_idx = torch.nonzero(delete_idx == 2).squeeze()
            delte_entity_idx = all_triplets[delete_idx, 2].view(-1).numpy()
            perturb_entity_idx = np.array(
                list(set(np.arange(n_entity)) - set(delte_entity_idx))
            )
            perturb_entity_idx = torch.from_numpy(perturb_entity_idx)
            perturb_entity_idx = torch.cat((perturb_entity_idx, object_.view(-1)))

            emb_ar = embedding[subject] * w[relation]
            emb_ar = emb_ar.view(-1, 1, 1)

            emb_c = embedding[perturb_entity_idx]
            emb_c = emb_c.transpose(0, 1).unsqueeze(1)

            out_prod = torch.bmm(emb_ar, emb_c)
            score = torch.sum(out_prod, dim=0)
            score = torch.sigmoid(score)

            target = torch.tensor(len(perturb_entity_idx) - 1)
            ranks_s.append(sort_and_rank(score, target))

            # Perturb subject
            object_relation = torch.tensor([object_, relation])
            delete_idx = torch.sum(tail_relation_triplets == object_relation, dim=1)
            delete_idx = torch.nonzero(delete_idx == 2).squeeze()

            delete_entity_idx = all_triplets[delete_idx, 0].view(-1).numpy()
            perturb_entity_idx = np.array(
                list(set(np.arange(n_entity)) - set(delete_entity_idx))
            )
            perturb_entity_idx = torch.from_numpy(perturb_entity_idx)
            perturb_entity_idx = torch.cat((perturb_entity_idx, subject.view(-1)))

            emb_ar = embedding[object_] * w[relation]
            emb_ar = emb_ar.view(-1, 1, 1)

            emb_c = embedding[perturb_entity_idx]
            emb_c = emb_c.transpose(0, 1).unsqueeze(1)

            out_prod = torch.bmm(emb_ar, emb_c)
            score = torch.sum(out_prod, dim=0)
            score = torch.sigmoid(score)

            target = torch.tensor(len(perturb_entity_idx) - 1)
            ranks_o.append(sort_and_rank(score, target))

        ranks_s = torch.cat(ranks_s)
        ranks_o = torch.cat(ranks_o)

        ranks = torch.cat([ranks_s, ranks_o])
        ranks += 1

        metrics["mrr"] = torch.mean(1.0 / ranks.float())
        for hit in hits:
            metrics[f"hits@{hit}"] = torch.mean((ranks <= hit).float())

    return metrics


def uniform(size, tensor):
    """Initiates a tensor with a uniform distribution.

    Args:
        size: int
            integer used to determine the bounds of the distribution
        tensor: torch.Tensor
            tensor to initiate
    """
    bound = 1.0 / math.sqrt(size)
    if tensor is not None:
        tensor.data.uniform_(-bound, bound)


def sort_and_rank(score, target):
    _, indices = torch.sort(score, dim=1, descending=True)
    indices = torch.nonzero(indices == target.view(-1, 1))
    indices = indices[:, 1].view(-1)
    return indices