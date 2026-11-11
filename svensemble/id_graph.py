"""Define the SVEnsemble ID graph, which links every matched breakpoint to its SV.

Matching (chunk_match.get_pairwise_matches) builds one graph over all callers
and samples. Each node is one breakpoint, named by its (source, event, sample)
tuple. Once SVs are assigned (chunk_match.assign_svensemble_ids), each node
also carries, under the attribute the caller names with common_id_name
('svensemble_id' by default):

    The cross-sample SV it belongs to (None for a ground-truth breakpoint
    left without matches).


Each edge is one pairwise match and carries:

    pair: The unordered source pair, encoded against IdGraph.sources.
    same_sample: Whether both ends come from the same sample.

IdGraph only holds data. The functions below read it and never change it;
drop_edges returns a new IdGraph.
"""

import dataclasses
from dataclasses import dataclass, field

import igraph as ig
import numpy as np
import pandas as pd


NODE_ATTRS = ('name',)
EDGE_ATTRS = ('pair', 'same_sample')


@dataclass(frozen=True, slots=True)
class IdGraph:
    """Hold a match graph with its node and edge annotations.

    Frozen fields cannot be rebound, but the igraph object itself is mutable:
    derive new graphs with drop_edges rather than editing *graph* in place.

    Attributes:
        graph (ig.Graph): Match graph whose nodes and edges carry the attributes
            the module docstring lists.
        sources (tuple[str, ...]): Sorted, unique source names; the edges' pair
            codes index into this order.

    Raises:
        ValueError: If *sources* is not sorted and unique, the nodes or edges
            lack their attributes, a pair code falls outside the range for
            *sources*, or a node's source is not in *sources*.
    """
    graph: ig.Graph = field(compare=False)
    sources: tuple[str, ...]

    def __post_init__(self):
        if list(self.sources) != sorted(self.sources) or len(set(self.sources)) != len(self.sources):
            raise ValueError(f"IdGraph sources must be sorted and unique, got {self.sources}")
        missing_v = [a for a in NODE_ATTRS if a not in self.graph.vs.attributes()]
        missing_e = [a for a in EDGE_ATTRS if a not in self.graph.es.attributes()]
        if self.graph.vcount() and missing_v:
            raise ValueError(f"IdGraph nodes lack attributes {missing_v}")
        if self.graph.ecount() and missing_e:
            raise ValueError(f"IdGraph edges lack attributes {missing_e}")
        if self.graph.ecount():
            codes = np.asarray(self.graph.es['pair'])
            if codes.min() < 0 or codes.max() >= n_pair_codes(len(self.sources)):
                raise ValueError(f"IdGraph pair codes out of range for {len(self.sources)} sources")
        unknown = {name[0] for name in self.graph.vs['name']} - set(self.sources) if self.graph.vcount() else set()
        if unknown:
            raise ValueError(f"IdGraph nodes have sources {sorted(unknown)} not in {self.sources}")


# -- Pair codes ---------------------------------------------------------------

def n_pair_codes(n_sources: int) -> int:
    """Count the unordered source pairs, same-source pairs included.

    Args:
        n_sources (int): Number of sources.

    Returns:
        int: n_sources * (n_sources + 1) / 2, the number of distinct pair codes.
    """
    return n_sources * (n_sources + 1) // 2


def pair_codes(i: np.ndarray, j: np.ndarray) -> np.ndarray:
    """Encode unordered pairs of source indices as integers.

    Takes the larger index of each pair as hi and the smaller as lo and returns
    hi*(hi+1)/2 + lo, so (i, j) and (j, i) get the same code.

    Args:
        i (np.ndarray): First source index of each pair.
        j (np.ndarray): Second source index of each pair, aligned with i.

    Returns:
        np.ndarray: The code of each pair.
    """
    i, j = np.asarray(i), np.asarray(j)
    lo, hi = np.minimum(i, j), np.maximum(i, j)
    return hi * (hi + 1) // 2 + lo


def encode_pair(idg: IdGraph, a: str, b: str) -> int:
    """Encode the unordered pair of sources *a* and *b*.

    Args:
        idg (IdGraph): ID graph whose sources give each source its index.
        a (str): One source.
        b (str): The other source.

    Returns:
        int: The pair code.
    """
    return int(pair_codes(idg.sources.index(a), idg.sources.index(b)))


def decode_pair(idg: IdGraph, code: int) -> tuple[str, str]:
    """Decode a pair code into the two sources it stands for.

    Args:
        idg (IdGraph): ID graph whose sources the code refers to.
        code (int): Pair code, as encode_pair or pair_codes make it.

    Returns:
        tuple[str, str]: The (lower, higher) source pair, by index in idg.sources.
    """
    hi = int((np.sqrt(8 * code + 1) - 1) // 2)
    lo = code - hi * (hi + 1) // 2
    return idg.sources[lo], idg.sources[hi]


# -- Derived graphs and tables ------------------------------------------------

def drop_edges(idg: IdGraph, edge_ids) -> IdGraph:
    """Copy an ID graph without the given edges, keeping every node.

    Args:
        idg (IdGraph): ID graph to copy; stays unchanged.
        edge_ids (Iterable[int]): IDs of the edges to drop; repeats are ignored.

    Returns:
        IdGraph: A new IdGraph without *edge_ids*.
    """
    graph = idg.graph.copy()
    graph.delete_edges(sorted(set(int(e) for e in edge_ids)))
    return dataclasses.replace(idg, graph=graph)


def node_table(idg: IdGraph) -> pd.DataFrame:
    """List the nodes of an ID graph as a table.

    Unpacks each node name into source, event and sample columns, then appends
    any other node attributes.

    Args:
        idg (IdGraph): ID graph to read.

    Returns:
        pd.DataFrame: One row per node, indexed by node ID. Holds only the
            source, event and sample columns, and no rows, when the graph has
            no nodes.
    """
    if idg.graph.vcount() == 0:
        return pd.DataFrame(columns=['source', 'event', 'sample'])
    nodes = idg.graph.get_vertex_dataframe()
    names = pd.DataFrame(nodes.pop('name').tolist(), columns=['source', 'event', 'sample'], index=nodes.index)
    return names.join(nodes)


def edge_table(idg: IdGraph) -> pd.DataFrame:
    """List the edges of an ID graph as a table.

    Args:
        idg (IdGraph): ID graph to read.

    Returns:
        pd.DataFrame: One row per edge, indexed by edge ID, with the source and
            target node IDs plus the edge attributes. Has no rows when the graph
            has no edges.
    """
    if idg.graph.ecount() == 0:
        return pd.DataFrame(columns=['source', 'target', *EDGE_ATTRS])
    return idg.graph.get_edge_dataframe()
