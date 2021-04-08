### LIBRARIES ###
# Global libraries
import numpy as np
import torch
from torch.utils.data import Dataset

### CLASS DEFINITION ###
class TrainDataset(Dataset):
    """Training dataset for the FB15k-237 dataset."""

    def __init__(
        self, triples: list, n_entity: int, n_relation: int, negative_sample_size: int
    ):
        """Initiates the FB15k-237 dataset for training.

        Each even element corresponds to a (`relation`, `tail`) element,
        each odd element corresponds to a (`head`, `relation`) element.

        Args:
            triples: List[(int, int, int)]
                list of triples representing a head, relationship and tail
            n_entity: int
                total number of entities
            n_relation: int
                total number of relations
            negative_sample_size: int
                size of negative samples
        """
        self.triples = triples
        self.triple_set = set(triples)
        self.n_entity = n_entity
        self.n_relation = n_relation
        self.negative_sample_size = negative_sample_size
        self.count = self.count_frequency(triples)
        self.true_head, self.true_tail = self.get_true_head_and_tail(self.triples)

    def __len__(self):
        return len(self.triples)

    def __getitem__(self, idx):
        positive_sample = self.triples[idx // 2]
        if idx % 2 == 0:
            mode = "head-batch"
        else:
            mode = "tail-batch"

        head, relation, tail = positive_sample

        subsampling_weight = (
            self.count[(head, relation)] + self.count[(tail, -relation - 1)]
        )
        subsampling_weight = torch.sqrt(1 / torch.Tensor([subsampling_weight]))

        negative_sample_list = []
        negative_sample_size = 0

        while negative_sample_size < self.negative_sample_size:
            negative_sample = np.random.randint(
                self.n_entity, size=self.negative_sample_size * 2
            )
            if mode == "head-batch":
                mask = np.in1d(
                    negative_sample,
                    self.true_head[(relation, tail)],
                    assume_unique=True,
                    invert=True,
                )
            elif mode == "tail-batch":
                mask = np.in1d(
                    negative_sample,
                    self.true_tail[(head, relation)],
                    assume_unique=True,
                    invert=True,
                )
            else:
                raise ValueError(f"Training batch mode {mode} not supported")

            negative_sample = negative_sample[mask]
            negative_sample_list.append(negative_sample)
            negative_sample_size += negative_sample.size

        negative_sample = np.concatenate(negative_sample_list)[
            : self.negative_sample_size
        ]

        negative_sample = torch.LongTensor(negative_sample)

        positive_sample = torch.LongTensor(positive_sample)

        return positive_sample, negative_sample, subsampling_weight, mode

    @staticmethod
    def collate_fn(data):
        positive_sample = torch.stack([_[0] for _ in data], dim=0)
        negative_sample = torch.stack([_[1] for _ in data], dim=0)
        subsample_weight = torch.cat([_[2] for _ in data], dim=0)
        mode = data[0][3]
        return positive_sample, negative_sample, subsample_weight, mode

    @staticmethod
    def count_frequency(triples: tuple, start=4) -> dict:
        """Gets the frequency of a partial triple

        Gets the frequency of a partial triple, like (head, relation) or (relation, tail).
        This frequency will be used for subsampling (like word2vec).

        Args:
            triples: (int, int, int)
                triples to consider (head, relation, tail)
            start: int
                initial value of the counter of a partial triple
        Returns:
            count: dict
                number of appearances of each partial triple
        """
        count = {}
        for head, relation, tail in triples:
            if (head, relation) not in count:
                count[(head, relation)] = start
            else:
                count[(head, relation)] += 1

            if (tail, -relation - 1) not in count:
                count[(tail, -relation - 1)] = start
            else:
                count[(tail, -relation - 1)] += 1
        return count

    @staticmethod
    def get_true_head_and_tail(triples: list) -> (dict, dict):
        """Builds dictionaries to use for negative sampling.

        Builds a dictionary of true triples that will be used
        to filter these true triples for negative sampling.

        Args:
            triples: list[(int, int, int)]
                list of true triples
        Returns:
            true_head: dict
                true triples for a given head
            true_tail: dict
                true triples for a given tail
        """
        true_head = {}
        true_tail = {}

        for head, relation, tail in triples:
            if (head, relation) not in true_tail:
                true_tail[(head, relation)] = []
            true_tail[(head, relation)].append(tail)
            if (relation, tail) not in true_head:
                true_head[(relation, tail)] = []
            true_head[(relation, tail)].append(head)

        for relation, tail in true_head:
            true_head[(relation, tail)] = np.array(
                list(set(true_head[(relation, tail)]))
            )
        for head, relation in true_tail:
            true_tail[(head, relation)] = np.array(
                list(set(true_tail[(head, relation)]))
            )

        return true_head, true_tail


class TestDataset(Dataset):
    """Validation and test datasets for the FB15k-237 dataset."""

    def __init__(
        self, triples: list, all_true_triples: list, n_entity: int, n_relation: int
    ):
        """Initiates the FB15k-237 dataset for training.

        Each even element corresponds to a (`relation`, `tail`) element,
        each odd element corresponds to a (`head`, `relation`) element.

        Args:
            triples: List[(int, int, int)]
                list of triples representing a head, relationship and tail
            all_true_triples: List[(int, int, int)]
                list of true triples from the training, validation and test datasets
            n_entity: int
                total number of entities
            n_relation: int
                total number of relations
        """
        self.triple_set = set(all_true_triples)
        self.triples = triples
        self.n_entity = n_entity
        self.n_relation = n_relation

    def __len__(self):
        return len(self.triples)

    def __getitem__(self, idx):
        head, relation, tail = self.triples[idx // 2]
        if idx % 2 == 0:
            mode = "head-batch"
        else:
            mode = "tail-batch"

        if mode == "head-batch":
            tmp = [
                (0, rand_head)
                if (rand_head, relation, tail) not in self.triple_set
                else (-1, head)
                for rand_head in range(self.n_entity)
            ]
            tmp[head] = (0, head)
        elif mode == "tail-batch":
            tmp = [
                (0, rand_tail)
                if (head, relation, rand_tail) not in self.triple_set
                else (-1, tail)
                for rand_tail in range(self.n_entity)
            ]
            tmp[tail] = (0, tail)
        else:
            raise ValueError(f"Negative batch mode {self.mode} not supported")

        tmp = torch.LongTensor(tmp)
        filter_bias = tmp[:, 0].float()
        negative_sample = tmp[:, 1]

        positive_sample = torch.LongTensor((head, relation, tail))

        return positive_sample, negative_sample, filter_bias, mode

    @staticmethod
    def collate_fn(data):
        positive_sample = torch.stack([_[0] for _ in data], dim=0)
        negative_sample = torch.stack([_[1] for _ in data], dim=0)
        filter_bias = torch.stack([_[2] for _ in data], dim=0)
        mode = data[0][3]
        return positive_sample, negative_sample, filter_bias, mode
