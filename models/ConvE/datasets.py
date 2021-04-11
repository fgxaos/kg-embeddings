### LIBRARIES ###
# Global libraries
from itertools import repeat

import numpy as np
import torch
from torch.utils.data import Dataset

### CLASS DEFINITION ###
class ConvEDataset(Dataset):
    """Training dataset for the FB15k-237 or wn18rr dataset."""

    def __init__(self, x, y):
        """Initiates the FB15k-237 or wn18rr dataset for training.

        Dataset used by the ConvE model.

        Args:
            x:
            y:
        """
        self.x = x
        self.y = y

        assert len(x) == len(y)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        s, r = self.x[idx]
        os = self.y[idx]
        indices = [o for o in os]
        return s, r, indices

    @staticmethod
    def collate_train(batch):
        max_len = max(map(lambda x: len(x[2]), batch))

        # Each object index list must have the same length (to use `torch.scatter_`),
        # therefore we pad with the first index.
        for _, _, indices in batch:
            indices.extend(repeat(indices[0], max_len - len(indices)))

        s, o, i = zip(*batch)
        return torch.LongTensor(s), torch.LongTensor(o), torch.LongTensor(i)

    @staticmethod
    def collate_test(batch):
        s, o, i = zip(*batch)
        return torch.LongTensor(s), torch.LongTensor(o), list(i)