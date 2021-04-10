### LIBRARIES ###
# Global libraries
import os

from torch.utils.data import DataLoader

from pytorch_lightning import LightningDataModule
from pytorch_lightning.trainer.supporters import CombinedLoader

# Custom libraries
from data.datasets.datasets import TrainDataset, TestDataset
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

    def train_dataloader(self):
        train_dataloader_head = DataLoader(
            TrainDataset(
                self.train_triples,
                len(self.entity2id),
                len(self.relation2id),
                self.negative_sample_size,
                "head-batch",
            ),
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=TrainDataset.collate_fn,
            drop_last=True,
            pin_memory=True,
        )

        train_dataloader_tail = DataLoader(
            TrainDataset(
                self.train_triples,
                len(self.entity2id),
                len(self.relation2id),
                self.negative_sample_size,
                "tail-batch",
            ),
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=TrainDataset.collate_fn,
            drop_last=True,
            pin_memory=True,
        )

        return [train_dataloader_head, train_dataloader_tail]

    def val_dataloader(self):
        val_dataloader_head = DataLoader(
            TestDataset(
                self.val_triples,
                self.train_triples + self.val_triples + self.test_triples,
                len(self.entity2id),
                len(self.relation2id),
                "head-batch",
            ),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=TestDataset.collate_fn,
            drop_last=True,
            pin_memory=True,
        )

        val_dataloader_tail = DataLoader(
            TestDataset(
                self.val_triples,
                self.train_triples + self.val_triples + self.test_triples,
                len(self.entity2id),
                len(self.relation2id),
                "tail-batch",
            ),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=TestDataset.collate_fn,
            drop_last=True,
            pin_memory=True,
        )

        return CombinedLoader([val_dataloader_head, val_dataloader_tail])

    def test_dataloader(self):
        test_dataloader_head = DataLoader(
            TestDataset(
                self.test_triples,
                self.train_triples + self.val_triples + self.test_triples,
                len(self.entity2id),
                len(self.relation2id),
                "head-batch",
            ),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=TestDataset.collate_fn,
            drop_last=True,
            pin_memory=True,
        )

        test_dataloader_tail = DataLoader(
            TestDataset(
                self.test_triples,
                self.train_triples + self.val_triples + self.test_triples,
                len(self.entity2id),
                len(self.relation2id),
                "tail-batch",
            ),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=TestDataset.collate_fn,
            drop_last=True,
            pin_memory=True,
        )

        return CombinedLoader([test_dataloader_head, test_dataloader_tail])
