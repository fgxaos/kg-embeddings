### LIBRARIES ###
# Global libraries
import math

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.data import Data

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
    origin, relation, target = triples.transpose()

    origin = torch.from_numpy(origin)
    relation = torch.from_numpy(relation)
    target = torch.from_numpy(target)

    origin, target = torch.cat((origin, target)), torch.cat((target, origin))
    relation = torch.cat((relation, relation + n_relations))

    edge_idx = torch.stack((origin, target))
    edge_type = relation

    data = Data(edge_index=edge_idx)
    data.entity = torch.from_numpy(np.arange(n_nodes))
    data.edge_type = edge_type
    data.edge_norm = edge_normalization(edge_type, edge_idx, n_nodes, n_relations)

    return data


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

        metrics = {}
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


def generate_sampled_graph_and_labels(
    triplets, sample_size, split_size, n_entity, n_relation, negative_rate
):
    edges = sample_edge_uniform(len(triplets), sample_size)

    # Select sampled edges
    edges = triplets[edges]
    src, rel, dst = edges.transpose()
    uniq_entity, edges = np.unique((src, dst), return_inverse=True)
    src, dst = np.reshape(edges, (2, -1))
    relabeled_edges = np.stack((src, rel, dst)).transpose()

    # Negative sampling
    samples, labels = negative_sampling(
        relabeled_edges, len(uniq_entity), negative_rate
    )

    # Further split the graph, only half of the edges will be used as graph
    # structure, while the remaining half is used as unseen positive samples
    split_size = int(sample_size * split_size)
    graph_split_ids = np.random.choice(
        np.arange(sample_size), size=split_size, replace=False
    )

    src = torch.tensor(src[graph_split_ids], dtype=torch.long).contiguous()
    dst = torch.tensor(dst[graph_split_ids], dtype=torch.long).contiguous()
    rel = torch.tensor(rel[graph_split_ids], dtype=torch.long).contiguous()

    # Create bi-directional graph
    src, dst = torch.cat((src, dst)), torch.cat((dst, src))
    rel = torch.cat((rel, rel + n_relation))

    edge_index = torch.stack((src, dst))
    edge_type = rel

    data = Data(edge_index=edge_index)
    data.entity = torch.from_numpy(uniq_entity)
    data.edge_type = edge_type
    data.edge_norm = edge_normalization(
        edge_type, edge_index, len(uniq_entity), n_relation
    )
    data.samples = torch.from_numpy(samples)
    data.labels = torch.from_numpy(labels)

    return data


def sample_edge_uniform(n_triples, sample_size):
    """Uniformly samples edges from all the edges.

    Args:
        n_triples: int
            total number of triples
        sample_size: int
            number of triples to sample
    Returns:
        np.array
            array with the indexes of the triples to sample
    """
    all_edges = np.arange(n_triples)
    return np.random.choice(all_edges, sample_size, replace=False)


def negative_sampling(pos_samples, n_entity, negative_rate):
    """Samples negative triples.

    Args:
        pos_samples: np.array
            positive samples
        n_entity: int
            total number of entities
        negative_rate
    Returns:
        samples: np.array
            concatenated positive and negative triples
        labels: np.array
            labels of the samples (1 for positive, 0 for negative)
    """
    batch_size = len(pos_samples)
    num_to_generate = batch_size * negative_rate
    neg_samples = np.tile(pos_samples, (negative_rate, 1))
    labels = np.zeros(batch_size * (negative_rate + 1), dtype=np.float32)
    labels[:batch_size] = 1
    values = np.random.choice(n_entity, size=num_to_generate)
    choices = np.random.uniform(size=num_to_generate)
    subj = choices > 0.5
    obj = choices <= 0.5
    neg_samples[subj, 0] = values[subj]
    neg_samples[obj, 2] = values[obj]

    return np.concatenate((pos_samples, neg_samples)), labels