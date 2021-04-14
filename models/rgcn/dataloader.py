### LIBRARIES ###
# Global libraries
import os

import numpy as np

import torch
from torch_geometric.data import DataLoader, Dataset

from pytorch_lightning import LightningDataModule

# Custom libraries
from models.rgcn.utils import build_test_graph, generate_sampled_graph_and_labels
from data.datasets.utils import load_entities, load_relations, read_triple

### CLASS DEFINITION ###
class KGDataModule(LightningDataModule):
    """Knowledge Graph module."""

    def __init__(
        self,
        dataset_dir: str,
        num_workers: int,
        batch_size: int,
        negative_sample_size: int,
        *args,
        **kwargs
    ):
        """Initiates a Knowledge Graph dataset.

        Args:
            dataset_dir: str
                path of the dataset directory to use
            num_workers: int
                number of workers to use
            batch_size: int
                batch size to use
            negative_sample_size: int
                size of the negative samples
        """
        super().__init__(*args, **kwargs)

        self.dataset_dir = dataset_dir
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.negative_sample_size = negative_sample_size

        # Build dictionaries to translate entities/relations to their ID
        self.entity2id = load_entities(os.path.join(self.dataset_dir, "entities.dict"))
        self.relation2id = load_relations(
            os.path.join(self.dataset_dir, "relations.dict")
        )
        # Load training, validation and test triples
        train_triples = read_triple(
            os.path.join(self.dataset_dir, "train.txt"),
            self.entity2id,
            self.relation2id,
        )
        val_triples = read_triple(
            os.path.join(self.dataset_dir, "valid.txt"),
            self.entity2id,
            self.relation2id,
        )
        test_triples = read_triple(
            os.path.join(self.dataset_dir, "test.txt"), self.entity2id, self.relation2id
        )

        # Gather all the triples
        self.all_triples = torch.LongTensor(train_triples + val_triples + test_triples)

        self.train_triples = np.array(train_triples, dtype=int)
        self.val_triples = torch.LongTensor(val_triples)
        self.test_triples = torch.LongTensor(test_triples)

        # Build the test graph
        self.test_graph = build_test_graph(
            len(self.entity2id), len(self.relation2id), self.train_triples
        )

    def train_dataloader(self):
        # Build the train data
        train_data = generate_sampled_graph_and_labels(
            self.train_triples,
            self.batch_size,
            self.split_size,
            len(self.entity2id),
            len(self.relation2id),
            self.negative_sample_size,
        )

        return DataLoader(
            train_data,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader([self.test_graph], batch_size=1)
        # return DataLoader(
        #     self.test_graph,
        #     batch_size=self.batch_size,
        #     shuffle=False,
        #     num_workers=self.num_workers,
        # )

    def test_dataloader(self):
        return self.test_graph
        # return DataLoader(
        #     self.test_graph,
        #     batch_size=self.batch_size,
        #     shuffle=False,
        #     num_workers=self.num_workers,
        # )
