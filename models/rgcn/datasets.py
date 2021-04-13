### LIBRARIES ###
# Global libraries
import numpy as np
import torch
from torch.utils.data import Dataset

### CLASS DEFINITION ###
class TrainDataset(Dataset):
    """Training dataset for the FB15k-237 or wn18rr dataset."""

    def __init__(
        self,
        triples: list,
        n_entity: int,
        n_relation: int,
        negative_sample_size: int,
    ):
        """Initiates the FB15k-237 or wn18rr dataset for training.

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
        positive_sample = self.triples[idx]

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
            mask = np.isin(
                negative_sample,
                self.true_head[(relation, tail)],
                assume_unique=True,
                invert=True,
            )

            negative_sample = negative_sample[mask]
            negative_sample_list.append(negative_sample)
            negative_sample_size += negative_sample.size

        negative_sample = np.concatenate(negative_sample_list)[
            : self.negative_sample_size
        ]

        negative_sample = torch.LongTensor(negative_sample)

        positive_sample = torch.LongTensor(positive_sample)

        samples = torch.cat((positive_sample, negative_sample))
        labels = torch.cat(
            (torch.ones_like(positive_sample), torch.zeros_like(negative_sample))
        )

        entity = torch.unique(samples)
        edge_idx = -1
        edge_type = relation
        edge_norm = subsampling_weight

        return entity, edge_idx, edge_type, edge_norm, samples, labels

    @staticmethod
    def collate_fn(data):
        entity = torch.stack([_[0] for _ in data], dim=0)
        edge_idx = torch.stack([_[1] for _ in data], dim=0)
        edge_type = torch.cat([_[2] for _ in data], dim=0)
        edge_norm = torch.cat([_[3] for _ in data], dim=0)
        samples = torch.cat([_[4] for _ in data], dim=0)
        labels = torch.cat([_[5] for _ in data], dim=0)
        return entity, edge_idx, edge_type, edge_norm, samples, labels

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
    """Validation and test datasets for the FB15k-237 or wn18rr dataset."""

    def __init__(
        self,
        triples: list,
        all_triplets: torch.LongTensor,
        n_entity: int,
        n_relation: int,
    ):
        """Initiates the FB15k-237 or wn18rr dataset for validation and testing.

        Each even element corresponds to a (`relation`, `tail`) element,
        each odd element corresponds to a (`head`, `relation`) element.

        Args:
            triples: List[(int, int, int)]
                list of triples representing a head, relationship and tail
            all_triplets: List[(int, int, int)]
                list of true triples from the training, validation and test datasets
            n_entity: int
                total number of entities
            n_relation: int
                total number of relations
            mode: str
                mode
        """
        self.all_triplets = set(all_triplets)
        self.triples = triples
        self.n_entity = n_entity
        self.n_relation = n_relation

    def __len__(self):
        return len(self.triples)

    def __getitem__(self, idx):
        head, relation, tail = self.triples[idx]

        tmp = [
            (0, rand_head)
            if (rand_head, relation, tail) not in self.all_triplets
            else (-1, head)
            for rand_head in range(self.n_entity)
        ]
        tmp[head] = (0, head)

        tmp = torch.LongTensor(tmp)
        filter_bias = tmp[:, 0].float()
        negative_sample = tmp[:, 1]

        positive_sample = torch.LongTensor((head, relation, tail))

        samples = torch.cat((positive_sample, negative_sample))
        labels = torch.cat(
            (torch.ones_like(positive_sample), torch.zeros_like(negative_sample))
        )
        entity = torch.unique(samples)
        heads = torch.cat(
            (torch.LongTensor([head, tail]), torch.LongTensor([tail, head]))
        )
        tails = torch.cat(
            (torch.LongTensor([tail, head]), torch.LongTensor([tail, head]))
        )
        edge_idx = torch.stack((heads, tails))
        edge_type = torch.LongTensor([relation])
        edge_norm = filter_bias

        return (
            entity,
            edge_idx,
            edge_type,
            edge_norm,
            torch.LongTensor(self.triples[idx]),
        )

    @staticmethod
    def collate_fn(data):
        entity = torch.stack([_[0] for _ in data], dim=0)
        edge_idx = torch.stack([_[1] for _ in data], dim=0)
        edge_type = torch.stack([_[2] for _ in data], dim=0)
        edge_norm = torch.stack([_[3] for _ in data], dim=0)
        triplet = torch.stack([_[4] for _ in data], dim=0)
        return entity, edge_idx, edge_type, edge_norm, triplet
