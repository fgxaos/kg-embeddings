### LIBRARIES ###
# Global libraries
import os
from tqdm import tqdm

import numpy as np

import torch
import torch.nn as nn
from torch_geometric.data import DataLoader, Dataset

# Custom libraries
from data.datasets.utils import load_entities, load_relations, read_triple
from models.rgcn.utils import (
    build_test_graph,
    generate_sampled_graph_and_labels,
    compute_metrics,
)
from models.rgcn.rgcn import RGCN

### FUNCTION DEFINITION ###
def run_rgcn(
    dataset_path: str,
    ckpt_file: str,
    cfg: dict,
):
    """Runs experiment with the R-GCN model.

    Args:
        dataset_path: str
            path to the dataset
        ckpt_file: str
            checkpoint file to the pretrained model
        cfg: dict
            configuration dictionary to use
    """
    cfg_model = cfg["rgcn"]["model"]
    cfg_data = cfg["rgcn"]["data"]
    cfg_training = cfg["rgcn"]["training"]

    ## Load the dataset
    # Build dictionaries to translate entities/relations to their ID
    entity2id = load_entities(os.path.join(dataset_path, "entities.dict"))
    relation2id = load_relations(os.path.join(dataset_path, "relations.dict"))

    # Load training, validation and test triples
    train_triples = read_triple(
        os.path.join(dataset_path, "train.txt"), entity2id, relation2id
    )
    val_triples = read_triple(
        os.path.join(dataset_path, "valid.txt"), entity2id, relation2id
    )
    test_triples = read_triple(
        os.path.join(dataset_path, "test.txt"), entity2id, relation2id
    )

    # Build the data objects used by the model
    all_triples = torch.LongTensor(train_triples + val_triples + test_triples)
    train_triples = np.array(train_triples, dtype=int)
    val_triples = torch.LongTensor(val_triples)
    test_triples = torch.LongTensor(test_triples)

    # Build the test graph
    test_graph = build_test_graph(len(entity2id), len(relation2id), train_triples)

    # Create a model instance
    model = RGCN(
        len(entity2id),
        len(relation2id),
        cfg_model["n_bases"],
        cfg_model["dropout"],
        cfg_model["reg_ratio"],
    )

    # Load the pretrained model
    if not os.path.exists(ckpt_file):
        optimizer = torch.optim.Adam(
            model.parameters(), lr=cfg_training["learning_rate"]
        )
        best_mrr = 0.0

        for i in tqdm(range(cfg_training["n_epochs"]), desc="Training epochs"):
            model.train()
            optimizer.zero_grad()
            loss = train_epoch(
                train_triples,
                model,
                cfg_data,
                cfg_model["reg_ratio"],
                len(entity2id),
                len(relation2id),
            )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg_training["grad_norm"])
            optimizer.step()

            # Evaluate the model
            val_metrics = validate(val_triples, test_graph, model, all_triples)

            # Save model checkpoint if best
            if val_metrics["mrr"] > best_mrr:
                best_mrr = val_metrics["mrr"]
                torch.save(model.state_dict(), ckpt_file)

    # Test the pretrained model
    model_test = model.load_state_dict(torch.load(ckpt_file))
    metrics = test(test_triples, model, test_graph, all_triples)
    print("Test metrics: ", metrics)


def train_epoch(
    train_triples,
    model,
    cfg_data,
    reg_ratio,
    n_entities,
    n_relations,
):
    # Dataset used by the training model
    train_data = generate_sampled_graph_and_labels(
        train_triples,
        cfg_data["batch_size"],
        cfg_data["split_size"],
        n_entities,
        n_relations,
        cfg_data["negative_sample_size"],
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_data.to(device)

    entity_embedding = model(
        train_data.entity,
        train_data.edge_index,
        train_data.edge_type,
        train_data.edge_norm,
    )
    loss = model.score_loss(
        entity_embedding, train_data.samples, train_data.labels
    ) + reg_ratio * model.reg_loss(entity_embedding)
    return loss


def validate(val_triples, test_graph, model, all_triples):
    model.eval()
    entity_embedding = model(
        test_graph.entity,
        test_graph.edge_index,
        test_graph.edge_type,
        test_graph.edge_norm,
    )
    val_metrics = compute_metrics(
        entity_embedding,
        model.relation_embedding,
        val_triples,
        all_triples,
        hits=[1, 3, 10],
    )
    return val_metrics


def test(test_triples, model, test_graph, all_triples):
    model.eval()
    entity_embedding = model(
        test_graph.entity,
        test_graph.edge_index,
        test_graph.edge_type,
        test_graph.edge_norm,
    )
    metrics = compute_metrics(
        entity_embedding,
        model.relation_embedding,
        test_triples,
        all_triples,
        hits=[1, 3, 10],
    )
    return metrics