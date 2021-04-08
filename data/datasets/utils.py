### FUNCTION DEFINITIONS ###
def read_triple(file_path: str, entity2id: dict, relation2id: dict) -> list:
    """Reads triples and maps them to IDs.

    Args:
        file_path: str
            path to the file which contains the data to read
        entity2id: dict
            dictionary mapping an entity to its ID
        relation2id: dict
            dictionary mapping a relation to its ID
    Returns:
        triples: list[(int, int, int)]
            list of triples, representing each relation by their IDs (head, relation, tail)
    """
    triples = []
    with open(file_path) as f:
        for line in f:
            h, r, t = line.strip().split("\t")
            triples.append((entity2id[h], relation2id[r], entity2id[t]))
    return triples


def load_entities(file_path: str) -> dict:
    """Loads the entities.

    Args:
        file_path: str
            path to the file which contains the data to read
    Returns:
        entity2id: dict
            dictionary mapping an entity to its ID
    """
    with open(file_path) as f:
        entity2id = {}
        for line in f:
            e_id, entity = line.strip().split("\t")
            entity2id[entity] = int(e_id)
    return entity2id


def load_relations(file_path: str) -> dict:
    """Loads the relations.

    Args:
        file_path: str
            path to the file which contains the data to read
    Returns:
        relation2id: dict
            dictionary mapping a relation to its ID
    """
    with open(file_path) as f:
        relation2id = {}
        for line in f:
            r_id, relation = line.strip().split("\t")
            relation2id[relation] = int(r_id)
    return relation2id
