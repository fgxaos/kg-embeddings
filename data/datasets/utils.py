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
            list of triples, representing each entity/relation by their IDs (head, relation, tail)
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


def separate_triples(triples):
    """Separates the triples in two lists.

    Given a list of triples (s, r, o), creates a list of tuples (s, r) and a list of o.

    Args:
        triples: List[(int, int, int)]
            triples to separate
    Returns:
        x: List[(int, int)]
            list of pairs of (source entity, relation)
        y: List[int]
            list of associated target entity
    """
    dict_source = {}
    for triple in triples:
        source = triple[0]
        relation = triple[1]
        target = triple[2]

        if source not in dict_source:
            dict_source[source] = {relation: [target]}
        else:
            if relation not in dict_source[source]:
                dict_source[source][relation] = [target]
            else:
                dict_source[source][relation].append(target)

    x = []
    y = []
    for i, triple in enumerate(triples):
        x.append((triple[0], triple[1]))
        y.append(dict_source[triple[0]][triple[1]])
    return x, y