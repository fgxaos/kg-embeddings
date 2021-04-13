### LIBRARIES ###
# Global libraries
import os

import torch
from torch.utils.data import DataLoader

from pytorch_lightning import LightningDataModule

# Custom libraries
from models.rgcn.datasets import TrainDataset, TestDataset

from models.rgcn.utils import build_test_graph
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
        self.train_triples = read_triple(
            os.path.join(self.dataset_dir, "train.txt"),
            self.entity2id,
            self.relation2id,
        )
        self.val_triples = read_triple(
            os.path.join(self.dataset_dir, "valid.txt"),
            self.entity2id,
            self.relation2id,
        )
        self.test_triples = read_triple(
            os.path.join(self.dataset_dir, "test.txt"), self.entity2id, self.relation2id
        )

        # Gather all the triples
        self.all_triples = torch.LongTensor(
            self.train_triples + self.val_triples + self.test_triples
        )

        # Build the test graph
        self.test_graph = build_test_graph(
            len(self.entity2id), len(self.relation2id), self.train_triples
        )

    def train_dataloader(self):
        return DataLoader(
            TrainDataset(
                self.train_triples,
                len(self.entity2id),
                len(self.relation2id),
                self.negative_sample_size,
            ),
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=TrainDataset.collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            TestDataset(
                self.val_triples,
                self.all_triples,
                len(self.entity2id),
                len(self.relation2id),
            ),
            batch_size=17535,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=TestDataset.collate_fn,
        )

    def test_dataloader(self):
        return DataLoader(
            TestDataset(
                self.test_triples,
                self.all_triples,
                len(self.entity2id),
                len(self.relation2id),
            ),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=TestDataset.collate_fn,
        )
