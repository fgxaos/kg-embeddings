### LIBRARIES ###
# Global libraries
import os

from torch.utils.data import DataLoader

from pytorch_lightning import LightningDataModule

# Custom libraries
from models.ConvE.datasets import ConvEDataset
from data.datasets.utils import (
    load_entities,
    load_relations,
    read_triple,
    separate_triples,
)

### CLASS DEFINITION ###
class KGDataModule(LightningDataModule):
    """Knowledge Graph module for ConvE"""

    def __init__(
        self, dataset_dir: str, num_workers: int, batch_size: int, *args, **kwargs
    ):
        """Initiates a Knowledge Graph dataset.

        Args:
            dataset_dir: str
                path of the dataset directory to use
            num_workers: int
                number of workers to use
            train_batch_size: int
                batch size to use for training
            val_batch_size: int
                batch size to use for validation and test
        """
        super().__init__(*args, **kwargs)

        self.dataset_dir = dataset_dir
        self.num_workers = num_workers
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size

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
        self.x_train, self.y_train = separate_triples(train_triples)

        val_triples = read_triple(
            os.path.join(self.dataset_dir, "valid.txt"),
            self.entity2id,
            self.relation2id,
        )
        self.x_val, self.y_val = separate_triples(val_triples)

        test_triples = read_triple(
            os.path.join(self.dataset_dir, "test.txt"), self.entity2id, self.relation2id
        )
        self.x_test, self.y_test = separate_triples(test_triples)

    def train_dataloader(self):
        return DataLoader(
            ConvEDataset(self.x_train, self.y_train),
            batch_size=self.train_batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=ConvEDataset.collate_train,
        )

    def val_dataloader(self):
        return DataLoader(
            ConvEDataset(self.x_val, self.y_val),
            batch_size=self.val_batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=ConvEDataset.collate_test,
        )

    def test_dataloader(self):
        return DataLoader(
            ConvEDataset(self.x_test, self.y_test),
            batch_size=self.val_batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=ConvEDataset.collate_test,
        )
