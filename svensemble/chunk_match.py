"""
Contains functions used for matching breakpoints from different sets
Following a similar algorith with chunking and creating similarity
matrices to the one used by Truvari
"""
import itertools
from collections.abc import Iterable
from collections import defaultdict, namedtuple
from functools import lru_cache

import igraph as ig
import numpy as np
import pandas as pd

from . import breakpoints as bps
from ._constants import BND1, BND2, GROUND_TRUTH_SAMPLE, SAMPLE_ID_COL, SV_ID_PREFIX
from .id_graph import IdGraph, drop_edges, pair_codes

# Maps each BND relative-position string to a unique integer code.
# Combined code = enc(bnd1) * 4 + enc(bnd2) gives a unique scalar per orientation pair,
# allowing a single np.equal.outer on integers instead of two on strings.
_ORIENT_ENCODE: dict[str, int] = {'N[': 0, ']N': 1, 'N]': 2, '[N': 3}

# Combined codes for insertion ('N[', ']N') and tandem-duplication (']N', 'N[') orientations.
# These are the two SV types that are indistinguishable at short-read resolution and
# should be allowed to match each other when ins_dup_match=True.
_INS_DUP_CODES: frozenset[int] = frozenset([
    _ORIENT_ENCODE['N['] * 4 + _ORIENT_ENCODE[']N'],  # 1  – insertion
    _ORIENT_ENCODE[']N'] * 4 + _ORIENT_ENCODE['N['],  # 4  – tandem duplication
])

# Chunks of more rows than this get re-cut by POS_bnd2 (_split_by_bnd2), so a
# locus that piles thousands of breakends onto one bnd1 spot, with mates spread
# far apart, never builds an n x n matrix over all of them
CHUNK_BND2_MATCH_THRESHOLD = 1000

def mark_ground_truth(bp_df: pd.DataFrame) -> pd.DataFrame:
    """Give ground-truth breakpoints the GROUND_TRUTH_SAMPLE placeholder as their sample.

    The ground truth is sites-only, so it has no sample of its own; the
    placeholder lets it match within every sample. Rejects a frame that
    already carries SAMPLE_ID_COL rather than silently relabelling it.
    """
    if SAMPLE_ID_COL in bp_df.columns:
        raise ValueError(
            f"Ground truth already has a {SAMPLE_ID_COL} column; truth sets must be sample-less"
        )
    return bp_df.assign(**{SAMPLE_ID_COL: GROUND_TRUTH_SAMPLE})


def concat_bp_dfs(bp_df_dict: dict[str, pd.DataFrame]):
    """Stack the non-empty breakpoint frames into one, labelling each row with its source in 'source'.

    Sorts the rows by chromosome pair, then by breakend positions, so that
    get_pairwise_matches can cut them into chunks of neighbouring breakpoints
    (_chunk_bounds).
    """
    bp_df = pd.concat([
        df.assign(source=name)
        for name, df in bp_df_dict.items()
        if not df.empty
    ]).sort_values(by=['CHROM_bnd1', 'CHROM_bnd2', 'POS_bnd1', 'POS_bnd2'])
    return bp_df

def _chunk_bounds(
    chrom1: np.ndarray,
    chrom2: np.ndarray,
    pos1: np.ndarray,
    chunk_size: int = 10000,
    chunk_max_n: int | None = None,
    chunk_max_n_dist: int | None = None,
) -> list[tuple[int, int]]:
    """Split rows into chunks of nearby breakpoints; return each chunk's [start, end) row range.

    Takes the rows grouped by chromosome pair and, within a pair, in POS_bnd1
    order. Starts a new chunk at every new chromosome pair and wherever
    POS_bnd1 rises by more than chunk_size over the previous row's. Once a
    chunk holds chunk_max_n rows, also starts one wherever POS_bnd1 rises by
    more than chunk_max_n_dist, a soft cap on chunk size. Walks plain lists of
    numbers, so it costs well under a microsecond per row.

    Raises:
        ValueError: If only one of chunk_max_n and chunk_max_n_dist is given,
            or either is not positive.
    """
    if (chunk_max_n is None) != (chunk_max_n_dist is None):
        raise ValueError("chunk_max_n and chunk_max_n_dist must be set together")
    if chunk_max_n is not None and (chunk_max_n <= 0 or chunk_max_n_dist <= 0):
        raise ValueError("chunk_max_n and chunk_max_n_dist must be greater than 0")
    n = len(pos1)
    if n == 0:
        return []
    new_pair = np.ones(n, dtype=bool)
    new_pair[1:] = (chrom1[1:] != chrom1[:-1]) | (chrom2[1:] != chrom2[:-1])
    new_pair, pos = new_pair.tolist(), pos1.tolist()
    capped = chunk_max_n is not None
    bounds, start = [], 0
    for i in range(1, n):
        gap = pos[i] - pos[i - 1]
        # 'not gap <= chunk_size' also splits at a missing position
        if new_pair[i] or (capped and i - start >= chunk_max_n and gap > chunk_max_n_dist) or not gap <= chunk_size:
            bounds.append((start, i))
            start = i
    bounds.append((start, n))
    return bounds


def _split_by_bnd2(
    pos2: np.ndarray,
    bounds: list[tuple[int, int]],
    max_n: int,
    max_dist: int,
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """Re-cut every chunk of more than max_n rows by POS_bnd2.

    Sorts such a chunk's rows by POS_bnd2 and starts a new chunk wherever
    POS_bnd2 rises by more than max_dist over the previous row's, or is
    missing. Leaves smaller chunks, and their row order, as they are. Like
    _chunk_bounds, ignores confidence intervals, so it may separate two
    breakpoints whose widened bnd2 intervals still lie within max_dist.

    Returns:
        The order to read the rows in, and each chunk's [start, end) range in that order.
    """
    order = np.arange(len(pos2))
    out = []
    for start, end in bounds:
        if end - start <= max_n:
            out.append((start, end))
            continue
        rows = start + np.argsort(pos2[start:end], kind='stable')
        order[start:end] = rows
        # 'not gap <= max_dist' also cuts at a missing position
        cuts = (start + 1 + np.flatnonzero(~(np.diff(pos2[rows]) <= max_dist))).tolist()
        edges = [start, *cuts, end]
        out.extend(zip(edges[:-1], edges[1:]))
    return order, out


def _interval_dists(ref_lo: np.ndarray, ref_hi: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """Measure the gap between reference interval i and interval j, for every (i, j).

    Returns 0 where the two intervals overlap or a bound is missing.
    """
    ref_right = np.subtract.outer(ref_lo, hi)         # reference interval lies right of the other
    other_right = np.subtract.outer(lo, ref_hi).T     # other interval lies right of the reference
    return np.where(other_right > 0, other_right, np.where(ref_right > 0, ref_right, 0))


def dist_matrix_ref(ref_bp_df: pd.DataFrame, bp_df: pd.DataFrame, which_bnd: str) -> np.ndarray:
    """Measure how far each reference breakpoint lies from each other breakpoint at one breakend.

    Widens each breakend position by its confidence interval (POS + CIPOS) and
    measures the gap between the reference and the other interval
    (_interval_dists): 0 where they overlap, otherwise the bases between them.

    Args:
        ref_bp_df (pd.DataFrame): DataFrame containing reference breakpoint data
        bp_df (pd.DataFrame): DataFrame containing breakpoint data from other sources
        which_bnd (str): String specifying which breakend to analyze ('bnd1' or 'bnd2')

    Returns:
        np.ndarray: 2D numpy array where element [i,j] is the distance between
                   reference breakpoint i and other breakpoint j. Distance is 0
                   if intervals overlap, otherwise it's the minimum separation
                   between the intervals.
    """
    def bounds(df):
        pos = df[f'POS_{which_bnd}'].to_numpy(dtype=np.float64)
        return (pos + df[f'INFO_CIPOS_l_{which_bnd}'].to_numpy(dtype=np.float64),
                pos + df[f'INFO_CIPOS_r_{which_bnd}'].to_numpy(dtype=np.float64))
    return _interval_dists(*bounds(ref_bp_df), *bounds(bp_df))


def rel_insert_delta_matrix_ref(ref_bp_df, bp_df, insert_delta_margin=30, ignore_size=False):
    """Compare the 'insert_delta' of every reference breakpoint with every other one
    (_insert_delta_similarity); with ignore_size, rate every pair 1."""
    if ignore_size:
        return np.ones((len(ref_bp_df), len(bp_df)))
    return _insert_delta_similarity(
        ref_bp_df['insert_delta'].to_numpy(np.float64), bp_df['insert_delta'].to_numpy(np.float64),
        insert_delta_margin,
    )


def _insert_delta_similarity(ref: np.ndarray, comp: np.ndarray, insert_delta_margin: float = 30) -> np.ndarray:
    """Rate how alike in size every (ref, comp) pair of insert deltas is.

    Divides the smaller delta by the larger, so 1 means equal sizes; rates a
    pair 1 when both deltas are at most insert_delta_margin.
    """
    ref_deltas = ref[:, np.newaxis]
    comp_deltas = comp[np.newaxis, :]

    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = np.where(comp_deltas > 0, ref_deltas / comp_deltas, 0.0)
        ratio_inv = np.where(ref_deltas > 0, comp_deltas / ref_deltas, 0.0)

    delta_matrix = np.minimum(ratio, ratio_inv)
    both_near_zero = (ref_deltas <= insert_delta_margin) & (comp_deltas <= insert_delta_margin)
    return np.where(both_near_zero, 1.0, delta_matrix)


def _extract_relative_pos(alt_series: pd.Series) -> np.ndarray:
    """Extract BND relative-position strings from an ALT series without full regex parsing.

    relative_pos is determined solely by whether the ALT string starts with a bracket
    (orientation is to the left of the anchor base → [N / ]N) or with a nucleotide
    (orientation is to the right → N[ / N]).  A single bracket-extraction regex is
    an order of magnitude faster than the full process_breakend_alt parse.

    Returns:
        np.ndarray of str with values in {'N[', ']N', 'N]', '[N'}.
    """
    alt = alt_series.fillna('.').astype(str)
    starts_with_bracket = alt.str[0].isin(['[', ']'])
    bracket_char = alt.str.extract(r'(\[|\])', expand=False)
    return np.where(starts_with_bracket, bracket_char + 'N', 'N' + bracket_char)


def alt_match_matrix(chunk_df: pd.DataFrame, ins_dup_match: bool = False, match_dist: int = 30) -> np.ndarray:
    """Mark which pairs of breakpoints share an orientation, reading it from their ALT strings.

    Codes each row's (bnd1, bnd2) orientation pair as one integer
    (_orientation_codes), so one O(n²) equality on integers compares the
    pairs, rather than two equalities on strings joined by a bitwise AND.

    Args:
        chunk_df (pd.DataFrame): DataFrame containing breakpoint data.
        ins_dup_match (bool): Whether to allow insertions and tandem-duplications to
            match each other.  Enable when merging long- and short-read call-sets.
    Returns:
        np.ndarray: Boolean match matrix of shape (n, n).
    """
    return _orientation_match(_orientation_codes(chunk_df), ins_dup_match)


def _orientation_codes(bp_df: pd.DataFrame) -> np.ndarray:
    """Code each row's (bnd1, bnd2) orientation pair as enc(bnd1) * 4 + enc(bnd2) (_ORIENT_ENCODE),
    taking enc = -1 for an ALT without a bracket."""
    enc1, enc2 = (
        pd.Series(_extract_relative_pos(bp_df[col])).map(_ORIENT_ENCODE).to_numpy(dtype=np.int8, na_value=-1)
        for col in ('ALT_bnd1', 'ALT_bnd2')
    )
    return enc1 * 4 + enc2


def _orientation_match(codes: np.ndarray, ins_dup_match: bool = False) -> np.ndarray:
    """Mark, in an (n, n) boolean matrix, which rows i and j share an orientation code.

    With ins_dup_match, also marks every pair whose rows are both insertion-
    or duplication-like, so the two SV types can match each other.
    """
    orientation_mask = np.equal.outer(codes, codes)
    if ins_dup_match:
        ins_dup_arr = np.isin(codes, list(_INS_DUP_CODES))
        orientation_mask |= np.logical_and.outer(ins_dup_arr, ins_dup_arr)
    return orientation_mask


# def ins_dup_match_matrix(chunk_df: pd.DataFrame, match_dist: int = 30) -> np.ndarray:
#     """
#     Generate a match matrix for insertions and duplications based on their insert delta.
#     Each entry where the estimated end of a variant via insert_delta is within match_dist of the end of another variant
#     that entry is set to True.
#     """
#     est_end_pos_matr =


def _get_match_matrix(
    pos_match_matrix: np.ndarray,
    delta_matrix: np.ndarray,
    alt_match_matrix: np.ndarray,
    triu_mask: np.ndarray,
    cross_source_mask: np.ndarray
) -> np.ndarray:
    """Match rows i and j where every mask agrees and i > j, so each pair appears once.

    Combines the position, size, orientation and source masks, and removes the
    diagonal and everything above it (*triu_mask*).
    """
    # delta_matrix may arrive as float (all-ones sentinel) or bool (threshold result);
    # cast to bool so bitwise-and works uniformly under NumPy 2.x
    return pos_match_matrix & delta_matrix.astype(bool) & alt_match_matrix & ~triu_mask & cross_source_mask

_MATCH_COLS = [
    'source', 'INFO_EVENT', SAMPLE_ID_COL, 'CHROM_bnd1', 'CHROM_bnd2', 'POS_bnd1', 'POS_bnd2',
    'ALT_bnd1', 'ALT_bnd2', 'INFO_CIPOS_l_bnd1', 'INFO_CIPOS_r_bnd1', 'INFO_CIPOS_l_bnd2',
    'INFO_CIPOS_r_bnd2', 'INFO_SVTYPE',
]
_OPTIONAL_MATCH_COLS = ['INFO_SVLEN', 'svensemble2_insert_size']


def _match_columns(bp_df: pd.DataFrame) -> list[str]:
    """List the columns get_pairwise_matches reads: _MATCH_COLS plus whichever optional ones *bp_df* has."""
    return _MATCH_COLS + [c for c in _OPTIONAL_MATCH_COLS if c in bp_df.columns]


@lru_cache(maxsize=100)
def _get_triu_mask(n: int) -> np.ndarray:
    """Build, and cache by size, a read-only (n, n) mask of the diagonal and everything above it."""
    mask = np.triu(np.ones(n, dtype=bool))
    mask.setflags(write=False)
    return mask

def get_pairwise_matches(
    zipped_bp_df: pd.DataFrame,
    insert_delta_threshold: float = 0.7,
    max_dist: int = 10,
    ignore_size: bool = False,
    ins_dup_match: bool = False,
    return_graph: bool = True,
    return_unmatched: bool = False,
    skip_single_source_for: str | list[str] | None = 'all',
    no_self_match: list[str] | None = None,
) -> ig.Graph | list[tuple[tuple[str, str], tuple[str, str]] | tuple[str, str]]:
    """
    Match breakpoints from multiple sources based on position and insert delta similarity.

    For intra-chromosomal events the insert delta is the net event size
    abs(insert_size - (pos_diff - 1)); for inter-chromosomal events it
    equals the raw insert size.

    Args:
        zipped_bp_df (pd.DataFrame): DataFrame containing breakpoint data from multiple sources
        insert_delta_threshold (float): Minimum similarity threshold for insert delta (0-1)
        max_dist (int): Maximum allowed distance between breakpoints
        ignore_size (bool): If True, skip insert delta filtering
        ins_dup_match (bool): If True, allow INS/DUP cross-type matching
        return_graph (bool): If True, return an undirected ig.Graph where
            each vertex name is a (source, event_id, sample) tuple. Default
            False returns the raw matched pairs list.
        return_unmatched (bool): If True, also add unmatched breakpoint pairs to the result as single items.
        no_self_match (list[str] | None): Sources whose breakpoints never match
            each other, e.g. the ground truth. Breakpoints of any other source
            may match others from the same source, within and across samples.
            Defaults to None.

    Returns:
        ig.Graph: Undirected graph of matched breakpoints. Each vertex has a
            'name' attribute holding a (source, event_id, sample) tuple; the
            ground truth's sample is GROUND_TRUTH_SAMPLE. Matching ignores
            samples, so edges may join different samples: split the graph
            with split_graph_by_sample before using its components. Returned
            when return_graph is True.
        list[tuple[tuple[str, str], tuple[str, str]]]: Raw matched pairs,
            returned when return_graph is False (default).
    """
    if return_unmatched and skip_single_source_for == 'all':
        raise ValueError("return_unmatched=True requires skip_single_source_for != 'all'")
    if SAMPLE_ID_COL not in zipped_bp_df.columns or zipped_bp_df[SAMPLE_ID_COL].isna().any():
        raise ValueError(
            f"Every breakpoint needs a {SAMPLE_ID_COL}; give the ground truth one with mark_ground_truth"
        )

    col_retain_list = _match_columns(zipped_bp_df)
    if not set(col_retain_list).issubset(zipped_bp_df.columns):
        raise ValueError(f"Some of the required columns are missing from the input dataframe: {set(col_retain_list) - set(zipped_bp_df.columns)}")
    # Rows missing a chromosome match nothing. Group the rest by chromosome
    # pair, in sorted pair order, keeping each pair's rows in input order
    bp_df = zipped_bp_df[col_retain_list]
    bp_df = bp_df[bp_df['CHROM_bnd1'].notna() & bp_df['CHROM_bnd2'].notna()]
    bp_df = bp_df.sort_values(['CHROM_bnd1', 'CHROM_bnd2'], kind='stable')
    bounds = _chunk_bounds(
        bp_df['CHROM_bnd1'].to_numpy(), bp_df['CHROM_bnd2'].to_numpy(), bp_df['POS_bnd1'].to_numpy(),
        chunk_max_n=100, chunk_max_n_dist=max_dist,
    )
    order, bounds = _split_by_bnd2(bp_df['POS_bnd2'].to_numpy(), bounds, CHUNK_BND2_MATCH_THRESHOLD, max_dist)
    bp_df = bp_df.iloc[order]

    # Read every per-row value once; each chunk below is a slice of these arrays
    same_chrom = (bp_df['CHROM_bnd1'] == bp_df['CHROM_bnd2']).to_numpy()
    pos_diff = (bp_df['POS_bnd2'] - bp_df['POS_bnd1']).to_numpy(dtype=np.float64)
    insert_delta = bps.get_insert_size(bp_df).to_numpy(dtype=np.float64)
    insert_delta[same_chrom] = np.abs(insert_delta[same_chrom] - (pos_diff[same_chrom] - 1))
    lo, hi = {}, {}
    for bnd in (BND1, BND2):
        pos = bp_df[f'POS_{bnd}'].to_numpy(dtype=np.float64)
        lo[bnd] = pos + bp_df[f'INFO_CIPOS_l_{bnd}'].to_numpy(dtype=np.float64)
        hi[bnd] = pos + bp_df[f'INFO_CIPOS_r_{bnd}'].to_numpy(dtype=np.float64)
    orientation = _orientation_codes(bp_df)
    source = bp_df['source'].to_numpy()
    source_code = pd.factorize(source)[0]
    no_self = np.isin(source, list(no_self_match or []))
    nodes = list(zip(source.tolist(), bp_df['INFO_EVENT'].tolist(), bp_df[SAMPLE_ID_COL].astype(str).tolist()))
    skip_all = skip_single_source_for == 'all'
    skip = set(skip_single_source_for) if isinstance(skip_single_source_for, list) else set()

    matched_pairs = []
    unmatched = []
    print(f'Number of chunks: {len(bounds)}')
    for start, end in bounds:
        n = end - start
        chunk_nodes = nodes[start:end]
        codes = source_code[start:end]
        # Skip lone breakpoints, and chunks of one source, of the skipped sources
        if (codes == codes[0]).all() and (skip_all or chunk_nodes[0][0] in skip):
            continue
        if n == 1:
            if return_unmatched:
                unmatched.append(chunk_nodes[0])
            continue

        rows = slice(start, end)
        pos_match_matrix = (
            (_interval_dists(lo[BND1][rows], hi[BND1][rows], lo[BND1][rows], hi[BND1][rows]) <= max_dist)
            & (_interval_dists(lo[BND2][rows], hi[BND2][rows], lo[BND2][rows], hi[BND2][rows]) <= max_dist)
        )
        if ignore_size:
            delta_matrix = np.ones((n, n), dtype=bool)
        else:
            delta_matrix = _insert_delta_similarity(insert_delta[rows], insert_delta[rows]) >= insert_delta_threshold
        alt_mask = _orientation_match(orientation[rows], ins_dup_match=ins_dup_match)
        # False where rows i and j both come from one no_self_match source
        in_no_self = no_self[rows]
        cross_source_mask = ~(np.logical_and.outer(in_no_self, in_no_self) & np.equal.outer(codes, codes))
        match_matrix = _get_match_matrix(pos_match_matrix, delta_matrix, alt_mask, _get_triu_mask(n), cross_source_mask)

        i_indices, j_indices = np.nonzero(match_matrix)
        matched_pairs.extend((chunk_nodes[i], chunk_nodes[j]) for i, j in zip(i_indices.tolist(), j_indices.tolist()))
        if return_unmatched:
            matched = np.zeros(n, dtype=bool)
            matched[i_indices] = True
            matched[j_indices] = True
            unmatched.extend(chunk_nodes[i] for i in np.flatnonzero(~matched).tolist())

    if isinstance(skip_single_source_for, list):
        unmatched = [node for node in unmatched if node[0] not in skip_single_source_for]
    elif skip_single_source_for == "all":
        unmatched = []

    if return_graph:
        out = ig.Graph.TupleList(matched_pairs, directed=False)
        out.add_vertices(unmatched)

        # Edge annotations: the unordered source pair, coded against the sorted
        # sources (IdGraph.sources), and whether both breakpoints come from one sample
        names = out.vs['name'] if out.vcount() else []
        sources = tuple(sorted({name[0] for name in names}))
        source_index = np.array([sources.index(name[0]) for name in names], dtype=np.int64)
        sample = np.array([name[2] for name in names], dtype=object)
        ends = np.asarray(out.get_edgelist(), dtype=np.int64).reshape(-1, 2)
        out.es['pair'] = pair_codes(source_index[ends[:, 0]], source_index[ends[:, 1]]).tolist()
        out.es['same_sample'] = (sample[ends[:, 0]] == sample[ends[:, 1]]).tolist()
        out = IdGraph(graph=out, sources=sources)
    else:
        out = matched_pairs + unmatched

    return out

def _node_samples(graph: ig.Graph) -> list[str]:
    """Read the sample of every node of *graph* from its name, in node order."""
    return [name[2] for name in graph.vs['name']] if graph.vcount() else []


def _wildcard_mask(graph: ig.Graph, wildcard_sample: str | None) -> np.ndarray:
    """Mark, over graph.vs, the nodes whose sample is *wildcard_sample*; mark none if it is None."""
    return np.array([s == wildcard_sample for s in _node_samples(graph)], dtype=bool)


def sample_vertex_mask(
    graph_in: ig.Graph,
    samples: Iterable[str],
    wildcard_sample: str | None = None,
) -> np.ndarray:
    """Mark, over graph_in.vs, the nodes whose sample is in *samples*.

    Also marks the nodes whose sample is *wildcard_sample*, since those
    belong to every sample.
    """
    wanted = set(samples) | ({wildcard_sample} if wildcard_sample is not None else set())
    return np.array([s in wanted for s in _node_samples(graph_in)], dtype=bool)


def split_graph_by_sample(graph_in: ig.Graph, wildcard_sample: str | None = None) -> dict[str, ig.Graph]:
    """Split the graph into one subgraph per sample, holding that sample's nodes and the wildcard nodes.

    Induces each subgraph from the sample's nodes and the nodes whose sample is
    *wildcard_sample* (as sample_vertex_mask selects them), which drops every
    edge between different samples. Removes the wildcard nodes left without a
    neighbour in a sample from that sample's subgraph, so they join only the
    samples they match. Returns the samples in sorted order.
    """
    # Read the names once: per-vertex attribute access in a per-sample loop is
    # O(samples x nodes) in Python
    sample_of = np.array(_node_samples(graph_in), dtype=object)
    is_wild = _wildcard_mask(graph_in, wildcard_sample)
    out = {}
    for sample in sorted(set(sample_of[~is_wild])):
        sub = graph_in.induced_subgraph(np.flatnonzero((sample_of == sample) | is_wild))
        stray = _wildcard_mask(sub, wildcard_sample) & (np.array(sub.degree(), dtype=np.int64) == 0)
        sub.delete_vertices(np.flatnonzero(stray).tolist())
        out[sample] = sub
    return out


def component_sample(component: list[tuple], wildcard_sample: str | None = None) -> str:
    """Find the one sample, other than *wildcard_sample*, that a per-sample component's nodes come from.

    Raises ValueError if the component spans several samples.
    """
    samples = {node[2] for node in component} - {wildcard_sample}
    if len(samples) != 1:
        raise ValueError(f"Component spans samples {sorted(samples)}; split the graph by sample first")
    return samples.pop()


def merge_bp_pairs_dfs(pairs):
    """
    Merge breakpoint pairs into connected components using depth-first search.

    Args:
        pairs: List of tuples representing connected breakpoint pairs

    Returns:
        List of lists, where each inner list contains the connected breakpoints
        in a component, sorted alphabetically
    """
    # Create adjacency list representation of the graph
    adj = defaultdict(set)
    for call_a, call_b in pairs:
        adj[call_a].add(call_b)
        adj[call_b].add(call_a)

    # Track visited nodes and store components
    seen = set()
    components = []

    # Perform DFS for each unvisited node
    for node in adj:
        if node in seen:
            continue

        # Initialize stack and component for current DFS
        stack = [node]
        comp = []

        # DFS traversal
        while stack:
            call_v = stack.pop()
            if call_v in seen:
                continue
            seen.add(call_v)
            comp.append(call_v)
            # Add unvisited neighbors to stack
            stack.extend(adj[call_v] - seen)

        # Add sorted component to results
        components.append(sorted(comp))

    return components

_TOPOLOGY_PREFIX = 'svensembletopology'

def smoothe_by_source(graph_in: ig.Graph, source_to_drop: str) -> ig.Graph:
    """Remove one source's vertices from a match graph and join their neighbours.

    Joins every two neighbours of each vertex of *source_to_drop* by an edge,
    unless they already share one, then deletes the vertices of that source.
    Added edges carry no attributes (None).

    Args:
        graph_in (ig.Graph): Match graph whose vertex names start with the
            source; stays unchanged.
        source_to_drop (str): Source whose vertices to remove.

    Returns:
        ig.Graph: A new graph without the vertices of *source_to_drop*.
    """
    graph = graph_in.copy()
    drop_indices = [v.index for v in graph.vs if v['name'][0] == source_to_drop]
    pairs = []
    for v in drop_indices:
        neighbors = np.asarray(graph.neighbors(v), dtype=np.int64)
        i, j = np.triu_indices(len(neighbors), k=1)
        pairs.append(np.column_stack([neighbors[i], neighbors[j]]))
    if pairs:
        graph.add_edges(np.concatenate(pairs).tolist())
        # Drop the added edges that duplicate existing ones; 'first' keeps the
        # existing edge's attributes, as existing edges precede added ones
        graph.simplify(multiple=True, loops=False, combine_edges='first')
    graph.delete_vertices(drop_indices)
    return graph

def calc_component_stats(graph_in: ig.Graph, ignore_callsets: list[str] | None = None) -> pd.DataFrame:
    """Compute topological features for each connected component of a match graph.

    Args:
        graph_in (ig.Graph): Undirected graph where each vertex name is a
            (source, event_id) tuple, as returned by get_pairwise_matches.
        ignore_callsets (list[str], optional): List of callset names to ignore when calculating component stats. Defaults to None.

    Returns:
        pd.DataFrame: One row per connected component (in iteration order of
            graph_in.connected_components()), with columns for degree, density,
            diameter, and clustering coefficient statistics.
    """
    out = {
        f'{_TOPOLOGY_PREFIX}_n_nodes': [],
        f'{_TOPOLOGY_PREFIX}_degree_mean': [],
        f'{_TOPOLOGY_PREFIX}_degree_max': [],
        f'{_TOPOLOGY_PREFIX}_density': [],
        f'{_TOPOLOGY_PREFIX}_diameter': [],
        f'{_TOPOLOGY_PREFIX}_clustering_coeff_mean': [],
    }
    for component in graph_in.connected_components():
        subgraph = graph_in.subgraph(component)
        if ignore_callsets is not None:
            keep = [v.index for v in subgraph.vs if v['name'][0] not in ignore_callsets]
            subgraph = subgraph.induced_subgraph(keep)
        if subgraph.vcount() == 0:
            continue
        out[f'{_TOPOLOGY_PREFIX}_n_nodes'].append(subgraph.vcount())
        degrees = subgraph.degree()
        out[f'{_TOPOLOGY_PREFIX}_degree_mean'].append(np.nanmean(degrees))
        out[f'{_TOPOLOGY_PREFIX}_degree_max'].append(np.max(degrees))
        out[f'{_TOPOLOGY_PREFIX}_density'].append(subgraph.density())
        out[f'{_TOPOLOGY_PREFIX}_diameter'].append(subgraph.diameter())
        out[f'{_TOPOLOGY_PREFIX}_clustering_coeff_mean'].append(
            np.nanmean(subgraph.transitivity_local_undirected())
        )
    return pd.DataFrame(out)


def graph_to_unique_component_lists(
    graph_in: ig.Graph,
) -> list[list[tuple[str, str]]]:
    """Convert a match graph into per-component node lists with one node per source.

    Within each component, nodes are sorted by descending degree so that the
    best-connected representative for each source is kept.

    Args:
        graph_in (ig.Graph): Undirected graph where each vertex name is a
            (source, event_id, sample) tuple.

    Returns:
        list[list[tuple]]: One inner list per connected component, each
            containing at most one (source, event_id, sample) tuple per source.
    """
    out = []
    for component in graph_in.connected_components():
        subgraph = graph_in.subgraph(component)
        degrees = subgraph.degree()
        sources_seen = []
        component_list = []
        for node in sorted(subgraph.vs, key=lambda v: degrees[v.index], reverse=True):
            if node['name'][0] in sources_seen:
                continue
            sources_seen.append(node['name'][0])
            component_list.append(node['name'])
        out.append(component_list)
    return out


def build_matched_bp_df(
    bp_df_dict: dict[str, pd.DataFrame],
    matched_lists: list[list[tuple[str, str]]],
    ref_callset: str | None = None,
    ids_only: bool = False,
    topology_df: pd.DataFrame = None,
    wildcard_sample: str | None = None,
) -> pd.DataFrame:
    """Build a DataFrame containing matched breakpoints from different sources.

    Each row represents a connected component of breakpoints that were matched across
    different sources. The data from each source is prefixed with the source name.

    Args:
        bp_df_dict (dict[str, pd.DataFrame]): Dictionary mapping source names to their
            breakpoint DataFrames
        matched_lists (list[list[tuple[str, str]]]): List of lists, where each inner list
            contains tuples of (source, event_id) representing connected breakpoints
        ids_only (bool, optional): Whether to return only the event IDs. Defaults to False.
        topology_df (pd.DataFrame, optional): Per-component topological features,
            one row per component aligned with matched_lists. Concatenated into the
            result when provided.
        wildcard_sample (str | None, optional): Sample of the nodes that belong
            to every sample; a row takes its sample from its other nodes.
    Returns:
        pd.DataFrame: DataFrame where each row contains the matched breakpoint data from
            different sources side by side, with source-specific column prefixes. Only
            includes rows where breakpoints were matched across at least 2 sources.
    """
    # Get unique matched events across all components
    unique_matched_events = {
        node[1]
        for matched_list in matched_lists
        for node in matched_list
    }

    # Create prefixed DataFrames containing only matched events
    bp_df_dict_prefixes = {
        source: bp_df[bp_df['INFO_EVENT'].isin(unique_matched_events)].assign(
            INFO_EVENT=lambda d: d['INFO_EVENT'].astype('string')
        ).add_prefix(source + '_')
        for source, bp_df in bp_df_dict.items()
    }

    # Create a table of {source}_INFO_EVENT entries populated by IDs from the matched_lists
    # This table will be used to join the breakpoint dataframes
    id_table_columns = [f'{source}_INFO_EVENT' for source in bp_df_dict.keys()]
    id_dict = {col: [pd.NA] * len(matched_lists) for col in id_table_columns}
    id_dict[SAMPLE_ID_COL] = [component_sample(matched_list, wildcard_sample) for matched_list in matched_lists]
    for i, matched_list in enumerate(matched_lists):
        sources_seen = set()
        for bp_source, bp_event, _sample in matched_list:
            if bp_source in sources_seen:
                continue
            sources_seen.add(bp_source)
            id_dict[f'{bp_source}_INFO_EVENT'][i] = bp_event
    id_table = pd.DataFrame(id_dict, dtype='string')

    if topology_df is not None:
        id_table = pd.concat([id_table, topology_df.reset_index(drop=True)], axis=1)

    # Remove rows where the only event is from the reference callset
    if ref_callset is not None:
        non_ref_event_cols = [
            f'{source}_INFO_EVENT' for source in bp_df_dict if source != ref_callset
        ]
        non_ref_notna_mask = np.any(id_table.loc[:, non_ref_event_cols].notna(), axis=1)
        id_table = id_table[non_ref_notna_mask]

    if ids_only:
        return id_table

    # Left join the id-table and each of the breakpoint dataframes
    for source, bp_df in bp_df_dict_prefixes.items():
        if source == ref_callset:
            continue

        try:
            id_table = _merge_on_event_and_sample(id_table, bp_df, source, how='left')
        except ValueError:
            raise ValueError(f"Error merging {source} breakpoint dataframe with id_table")

    return id_table


def _merge_on_event_and_sample(
    left: pd.DataFrame,
    source_df: pd.DataFrame,
    source: str,
    how: str,
) -> pd.DataFrame:
    """Merge a source's prefixed breakpoints into *left* on (event, sample).

    *source_df* carries {source}_INFO_EVENT and {source}_SAMPLE_ID_COL;
    *left* carries {source}_INFO_EVENT and the shared SAMPLE_ID_COL. The
    source's sample column is folded into the shared one (filling it for rows
    the merge adds) and dropped. Keys are compared as strings.
    """
    event_col = f'{source}_INFO_EVENT'
    source_sample_col = f'{source}_{SAMPLE_ID_COL}'

    def with_text_keys(df: pd.DataFrame, cols: tuple[str, str]) -> pd.DataFrame:
        # Replace only the key columns that are not text yet, on a shallow copy:
        # the caller's frame stays unchanged and no data is duplicated
        cast = [col for col in cols if df[col].dtype != 'string']
        if cast:
            df = df.copy(deep=False)
            for col in cast:
                df[col] = df[col].astype('string')
        return df

    merged = with_text_keys(left, (event_col, SAMPLE_ID_COL)).merge(
        with_text_keys(source_df, (event_col, source_sample_col)),
        left_on=[event_col, SAMPLE_ID_COL], right_on=[event_col, source_sample_col], how=how,
    )
    merged[SAMPLE_ID_COL] = merged[SAMPLE_ID_COL].fillna(merged[source_sample_col])
    del merged[source_sample_col]
    return merged


def _attach_source_columns(left: pd.DataFrame, bp_df_dict: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Append each source's breakpoint columns to *left*, prefixed with the source name.

    For every row of *left*, looks up the source's breakpoint with the row's
    {source}_INFO_EVENT and SAMPLE_ID_COL, compared as text, and takes that
    breakpoint's other columns, or NaN where the row has none. Joins all the
    sources' columns to *left* in one concat that copies nothing, so no step
    copies the growing frame. Gives the rows and columns that a left merge per
    source (_merge_on_event_and_sample) gives, with a fresh RangeIndex.

    Raises:
        ValueError: If a source holds two breakpoints with one event and sample,
            which a merge would turn into duplicate rows.
    """
    left = left.reset_index(drop=True)
    blocks = []
    for source, bp_df in bp_df_dict.items():
        keys = [f'{source}_INFO_EVENT', SAMPLE_ID_COL]
        right = (
            bp_df[['INFO_EVENT', SAMPLE_ID_COL]].astype('string')
            .set_axis(keys, axis=1).reset_index(drop=True)
            .assign(_row=np.arange(len(bp_df)))
        )
        if right.duplicated(keys).any():
            raise ValueError(f"{source} holds more than one breakpoint with the same event and sample")
        # how='left' keeps left's rows in order; -1 marks rows without the source's breakpoint
        rows = left[keys].astype('string').merge(right, on=keys, how='left')['_row'].fillna(-1).to_numpy(dtype=np.int64)
        body = bp_df.drop(columns=['INFO_EVENT', SAMPLE_ID_COL])
        body.index = pd.RangeIndex(len(body))
        block = body.reindex(rows)
        block.index = left.index
        block.columns = [f'{source}_{col}' for col in block.columns]
        blocks.append(block)
    return pd.concat([left, *blocks], axis=1, copy=False)


def _kept_view(graph: ig.Graph, dropped: set[int]) -> ig.Graph:
    """Copy *graph* without the *dropped* edges, tagging every edge with its ID in
    *graph* ('eid') and every node with its ID ('vid').

    igraph carries the tags through subgraphs, so cut edges map straight back to
    *graph*. subgraph_edges keeps the edges in ID order, which matches *kept*.
    """
    keep = np.ones(graph.ecount(), dtype=bool)
    keep[list(dropped)] = False
    kept = np.flatnonzero(keep).tolist()
    view = graph.subgraph_edges(kept, delete_vertices=False)
    view.es['eid'] = kept
    view.vs['vid'] = list(range(view.vcount()))
    return view


def _edge_costs(graph: ig.Graph, is_wild: np.ndarray) -> list[int]:
    """Price every edge of *graph* at 1, or, if it touches a wildcard node, above all edges together.

    A minimum cut over these prices removes as few wildcard edges as it can and,
    among those cuts, as few other edges as it can, so breakpoints stay with the
    wildcard (ground-truth) site they match.
    """
    ends = np.asarray(graph.get_edgelist(), dtype=np.int64).reshape(-1, 2)
    is_wild = np.asarray(is_wild, dtype=bool)
    return np.where(is_wild[ends[:, 0]] | is_wild[ends[:, 1]], graph.ecount() + 1, 1).tolist()


def _most_separating(cuts: list, groups: list[list[int]], cost: list[float]):
    """Pick the cut that separates the most conflicting vertex pairs per unit of edge *cost*
    it removes; the first wins ties.

    Every two vertices of one group conflict, so a cut that leaves k of a group's
    n vertices on one side separates k * (n - k) of its pairs; counting them
    this way avoids listing the n * (n - 1) / 2 pairs.
    """
    def separated_per_cost(cut):
        side = set(cut.partition[0])
        separated = 0
        for group in groups:
            k = sum(v in side for v in group)
            separated += k * (len(group) - k)
        return separated / sum(cost[e] for e in cut.cut)
    return max(cuts, key=separated_per_cost)


def _members(membership: np.ndarray, components: np.ndarray) -> list[np.ndarray]:
    """Gather the vertex IDs of each of *components* from every vertex's component number.

    Takes *components* in ascending order and returns one array per component,
    its IDs ascending, as connected_components() lists them.
    """
    order = np.argsort(membership, kind='stable')
    lo, hi = np.searchsorted(membership[order], np.stack([components, components + 1]))
    return [order[a:b] for a, b in zip(lo, hi)]


def _components_with_repeats(graph: ig.Graph, keys: np.ndarray) -> list[np.ndarray]:
    """Find the components of *graph* in which two vertices share a key; return their vertex IDs.

    Counts (component, key) pairs with numpy, so a graph of mostly lone
    breakpoints costs no Python work per component. Returns the components in
    component order.
    """
    if graph.vcount() == 0:
        return []
    membership = np.asarray(graph.connected_components().membership, dtype=np.int64)
    n_keys = int(np.max(keys)) + 1
    values, counts = np.unique(membership * n_keys + np.asarray(keys, dtype=np.int64), return_counts=True)
    return _members(membership, np.unique(values[counts > 1] // n_keys))


def _same_caller_cut(sub: ig.Graph, wildcard_sample: str | None) -> list[int]:
    """Choose which edges to cut in *sub*, a component holding some source twice; return their IDs in *sub*.

    Computes one candidate cut per duplicated source and picks one
    (_most_separating), as same_caller_split_edges describes.
    """
    names = sub.vs['name']
    by_source = defaultdict(list)
    for i, name in enumerate(names):
        by_source[name[0]].append(i)
    duplicated = sorted(s for s, members in by_source.items() if len(members) > 1)
    cost = _edge_costs(sub, _wildcard_mask(sub, wildcard_sample))
    cost_arr = np.asarray(cost, dtype=np.float64)
    ends = np.asarray(sub.get_edgelist(), dtype=np.int64).reshape(-1, 2)
    degree = sub.degree()
    cuts = []
    for s in duplicated:
        # Cut between the source's two best-connected breakpoints, adding eps to the kept one's edges
        kept, other = sorted(by_source[s], key=lambda i: (-degree[i], str(names[i][1])))[:2]
        eps = 1 / (2 * (degree[kept] + 1))
        capacity = np.where((ends == kept).any(axis=1), cost_arr + eps, cost_arr)
        cuts.append(sub.mincut(source=kept, target=other, capacity=capacity.tolist()))
    return _most_separating(cuts, [by_source[s] for s in duplicated], cost).cut


def same_caller_split_edges(
    idg: IdGraph, dropped: Iterable[int] = (), wildcard_sample: str | None = None,
) -> np.ndarray:
    """Collect edges whose removal leaves no source twice in a per-sample component.

    Removes the marked edges from a copy of the graph and splits it by sample
    (split_graph_by_sample, with the wildcard nodes in every sample they
    match). In every component that holds two breakpoints of one source,
    separates them with a minimum cut and marks the cut edges, then repeats on
    the pieces of that component until none holds a source twice. Components
    never share edges, so each one is split on its own; its pieces keep their
    vertices in ID order and get igraph's canonical edge order, as
    split_graph_by_sample's subgraphs do. Wildcard (ground-truth) breakpoints
    conflict like any other source's.

    Within a component, computes one candidate cut per duplicated source,
    between its two breakpoints with the most edges in the component (ties:
    event ID), and applies the candidate that separates the most same-source
    pairs per unit of cost cut (ties: alphabetically first source). Prices
    every edge at 1, and an edge to a wildcard node above all edges together
    (_edge_costs), so the cut keeps breakpoints with the wildcard site they
    match whenever other edges can separate them. Each candidate keeps the
    better-connected breakpoint: adds eps to its edges, so that among equally
    cheap cuts its neighbours stay with it. eps never changes the cut's cost.

    Args:
        idg (IdGraph): ID graph to split; stays unchanged.
        dropped (Iterable[int]): Edge IDs already marked for removal.
        wildcard_sample (str | None): Sample of the nodes that belong to every
            sample they match (the ground truth). Defaults to None.

    Returns:
        np.ndarray: The sorted IDs (in idg.graph) of the marked edges: the
            input ones plus the new ones.
    """
    graph = idg.graph
    dropped = {int(e) for e in dropped}
    if graph.vcount() == 0:
        return np.array(sorted(dropped), dtype=np.int64)
    _, source_code = np.unique([name[0] for name in graph.vs['name']], return_inverse=True)

    def split(g: ig.Graph) -> list[ig.Graph]:
        """Extract, as subgraphs, the components of *g* that hold some source twice."""
        return [
            g.induced_subgraph(component, implementation='create_from_scratch')
            for component in _components_with_repeats(g, source_code[g.vs['vid']])
        ]

    view = _kept_view(graph, dropped)
    pending = [sub for sample_graph in split_graph_by_sample(view, wildcard_sample).values()
               for sub in split(sample_graph)]
    while pending:
        current, pending = pending, []
        for sub in current:
            cut = _same_caller_cut(sub, wildcard_sample)
            dropped.update(sub.es[cut]['eid'])
            sub.delete_edges(cut)
            pending.extend(split(sub))
    return np.array(sorted(dropped), dtype=np.int64)


_TreeCut = namedtuple('_TreeCut', ['partition', 'cut'])


def _gomory_hu_cuts(graph: ig.Graph, cost: list[float], pairs: list[tuple[int, int]]) -> list[_TreeCut]:
    """Find a cheapest cut between the vertices of every pair from one Gomory-Hu tree of *graph*.

    Builds the tree with n - 1 maximum flows for n vertices, where cutting each
    pair directly would take one per pair. The lightest tree edge on the path
    between a pair's vertices splits the tree into two sides that the pair's
    minimum cut separates at the same cost. Returns one cut per distinct tree
    edge, in the order the pairs first pick it; each cut carries the two
    vertex sides (partition) and the IDs of the *graph* edges crossing them
    (cut), as igraph's Cut does.
    """
    # Loops never cross a cut and parallel edges add up, so build the tree on a simple copy
    simple = graph.copy()
    simple.es['cost'] = cost
    simple.simplify(multiple=True, loops=True, combine_edges={'cost': 'sum'})
    tree = simple.gomory_hu_tree(capacity='cost')
    flow = tree.es['flow']
    picked = {}
    for a, b in pairs:
        picked.setdefault(min(tree.get_shortest_path(a, b, output='epath'), key=flow.__getitem__), None)
    ends = np.asarray(graph.get_edgelist(), dtype=np.int64).reshape(-1, 2)
    cuts = []
    for e in picked:
        rest = tree.copy()
        rest.delete_edges([e])
        side = np.zeros(graph.vcount(), dtype=bool)
        side[rest.subcomponent(tree.es[e].source)] = True
        crossing = np.flatnonzero(side[ends[:, 0]] != side[ends[:, 1]]).tolist()
        cuts.append(_TreeCut([np.flatnonzero(side).tolist(), np.flatnonzero(~side).tolist()], crossing))
    return cuts


def same_sample_split_edges(
    idg: IdGraph, dropped: Iterable[int] = (), wildcard_sample: str | None = None,
) -> np.ndarray:
    """Collect edges whose removal leaves no sample twice in an SV.

    Each round, removes the marked edges from a copy of the graph, numbers its
    per-sample components (split_graph_by_sample), and groups all breakpoints,
    wildcard ones included, into SVs by the edges between them. In every SV
    that holds two per-sample components of one sample, separates those two
    with a minimum cut and marks the cut edges. The cut runs on a graph whose
    vertices are the SV's per-sample components and its wildcard breakpoints,
    one vertex each, and which keeps every edge between two of them. Wildcard
    breakpoints belong to no single sample, so they never conflict. Stops
    when no SV holds a sample twice.

    Within an SV, finds a cheapest cut for every pair of same-sample
    components from one Gomory-Hu tree (_gomory_hu_cuts), and applies the
    candidate that separates the most such pairs per unit of cost cut (ties:
    alphabetically first sample, then lowest-numbered components). Where
    several cuts are equally cheap for a pair, the tree may pick a different
    one than a direct minimum cut between the pair would. Prices every edge at 1, and an edge to a
    wildcard node above all edges together (_edge_costs), so the cut keeps
    every sample's breakpoints with the wildcard site they match whenever
    other edges can separate the conflict.

    Args:
        idg (IdGraph): ID graph to split; stays unchanged.
        dropped (Iterable[int]): Edge IDs already marked for removal.
        wildcard_sample (str | None): Sample of the nodes that belong to every
            sample they match (the ground truth). Defaults to None.

    Returns:
        np.ndarray: The sorted IDs (in idg.graph) of the marked edges: the
            input ones plus the new ones.
    """
    graph = idg.graph
    sample_of = _node_samples(graph)
    is_wild = _wildcard_mask(graph, wildcard_sample)
    dropped = {int(e) for e in dropped}
    if graph.vcount() == 0:
        return np.array(sorted(dropped), dtype=np.int64)
    _, sample_code = np.unique(np.asarray(sample_of, dtype=object), return_inverse=True)
    tame = np.flatnonzero(~is_wild)
    while True:
        view = _kept_view(graph, dropped)
        # Per-sample component of every non-wildcard breakpoint; each wildcard
        # breakpoint gets a number of its own
        component_of = np.full(graph.vcount(), -1, dtype=np.int64)
        n_components = 0
        for sample_graph in split_graph_by_sample(view, wildcard_sample).values():
            components = sample_graph.connected_components()
            vid = np.array(sample_graph.vs['vid'], dtype=np.int64)
            component_of[vid] = n_components + np.array(components.membership, dtype=np.int64)
            n_components += len(components)
        component_of[is_wild] = n_components + np.arange(is_wild.sum())

        # Find, with numpy, the SVs that hold two per-sample components of one sample
        membership = np.asarray(view.connected_components().membership, dtype=np.int64)
        triples = np.unique(np.column_stack([membership[tame], sample_code[tame], component_of[tame]]), axis=0)
        pairs, counts = np.unique(triples[:, :2], axis=0, return_counts=True)
        new = []
        for sv in _members(membership, np.unique(pairs[counts > 1, 0])):
            sv = sv.tolist()
            per_sample = defaultdict(set)
            for v in sv:
                if not is_wild[v]:
                    per_sample[sample_of[v]].add(component_of[v])
            conflicting = sorted(s for s, comps in per_sample.items() if len(comps) > 1)
            # Price the SV's edges, then merge each per-sample component (and
            # wildcard breakpoint) into one vertex, keeping every edge between them
            quotient = view.subgraph(sv)
            vid = quotient.vs['vid']
            quotient.es['cost'] = _edge_costs(quotient, is_wild[vid])
            components, local = np.unique(component_of[vid], return_inverse=True)
            quotient.contract_vertices(local.tolist())
            cost = quotient.es['cost']
            index = {c: i for i, c in enumerate(components)}
            # Every two per-sample components of one sample conflict; cut each pair
            conflicts = [
                (index[a], index[b])
                for sample in conflicting
                for a, b in itertools.combinations(sorted(per_sample[sample]), 2)
            ]
            cuts = _gomory_hu_cuts(quotient, cost, conflicts)
            groups = [[index[c] for c in sorted(per_sample[sample])] for sample in conflicting]
            new.extend(quotient.es[_most_separating(cuts, groups, cost).cut]['eid'])
        if not new:
            return np.array(sorted(dropped), dtype=np.int64)
        dropped.update(new)


def assign_svensemble_ids(
    idg: IdGraph, dropped: Iterable[int], wildcard_sample: str | None = None,
    common_id_name: str = 'svensemble_id',
) -> IdGraph:
    """Drop the marked edges and label every SV with its SVEnsemble ID.

    Groups all breakpoints, wildcard ones included, into SVs by the edges
    between them, once before and once after removing the marked edges,
    numbering the groups by their smallest node ID. Labels each SV after the
    removal with SV_ID_PREFIX plus the number of the group it came from, adding
    '.k' when the removal split that group (k numbers the pieces). Leaves a
    wildcard breakpoint without a label when it stands alone.

    Args:
        idg (IdGraph): ID graph to label; stays unchanged.
        dropped (Iterable[int]): Edge IDs marked for removal.
        wildcard_sample (str | None): Sample of the nodes that belong to every
            sample they match (the ground truth). Defaults to None.
        common_id_name (str): Node attribute to write the labels to. Defaults
            to 'svensemble_id'.

    Returns:
        IdGraph: A new ID graph without the marked edges, with the label in the
            common_id_name node attribute (None for a wildcard breakpoint that
            stands alone).
    """
    is_wild = _wildcard_mask(idg.graph, wildcard_sample)
    pruned = drop_edges(idg, dropped)

    # SV components before and after the drop, lone wildcard nodes left out
    before, after = (
        sorted(
            (c for c in graph.connected_components() if not is_wild[c].all()),
            key=min,
        )
        for graph in (idg.graph, pruned.graph)
    )
    parent_of = np.empty(len(is_wild), dtype=np.int64)
    for number, component in enumerate(before):
        parent_of[component] = number
    pieces = defaultdict(list)
    for component in after:
        pieces[parent_of[component[0]]].append(component)

    pruned.graph.vs[common_id_name] = None
    for parent, parts in pieces.items():
        for k, component in enumerate(parts):
            pruned.graph.vs[component][common_id_name] = (
                f'{SV_ID_PREFIX}{parent}' if len(parts) == 1 else f'{SV_ID_PREFIX}{parent}.{k}'
            )
    return pruned


def flatten_idgraph(
    idg: IdGraph,
    sources: Iterable[str],
    wildcard_sample: str | None = None,
    common_id_name: str = 'svensemble_id',
    add_common_id: bool = False,
) -> pd.DataFrame:
    """Turn a pruned ID graph into the ID table, one row per per-sample component.

    Splits the graph by sample (split_graph_by_sample), so a wildcard node
    appears once in every sample it matches, and lists every node as
    (source, event, sample, SV ID). Pivots that list on (SV ID, sample) into
    one <source>_INFO_EVENT column per source. Pruning leaves at most one node
    per source in each (SV ID, sample) pair, so the pivot meets no duplicates;
    DataFrame.pivot raises if it does.

    Args:
        idg (IdGraph): Pruned ID graph with SV IDs assigned (assign_svensemble_ids).
        sources (Iterable[str]): Sources to give a column, in column order.
        wildcard_sample (str | None): Sample of the nodes that belong to every
            sample they match (the ground truth). Defaults to None.
        common_id_name (str): Node attribute holding the SV IDs. Defaults to
            'svensemble_id'.
        add_common_id (bool): Keep the SV ID as a column named common_id_name.
            Defaults to False.

    Returns:
        pd.DataFrame: One <source>_INFO_EVENT column per source and
            SAMPLE_ID_COL, plus common_id_name with add_common_id; all text.
    """
    long = pd.DataFrame(
        [
            (name[0], name[1], sample, common_id)
            for sample, sub in split_graph_by_sample(idg.graph, wildcard_sample).items()
            for name, common_id in zip(sub.vs['name'], sub.vs[common_id_name])
        ],
        columns=['source', 'event', SAMPLE_ID_COL, common_id_name],
    )
    table = (
        long.pivot(index=[common_id_name, SAMPLE_ID_COL], columns='source', values='event')
        .reindex(columns=list(sources))
        .rename(columns=lambda source: f'{source}_INFO_EVENT')
        .reset_index()
    )
    if not add_common_id:
        table = table.drop(columns=common_id_name)
    return table.astype('string')


def join_calls(
    bp_df_dict: dict[str, pd.DataFrame],
    ref_callset: str | None = None,
    insert_delta_threshold: float = 0.7,
    max_dist: int = 10,
    ids_only: bool = False,
    add_topology: bool = False,
    ignore_size: bool = False,
    workflow_config: dict | None = None,
    keep_unmatched: bool = False,
    wildcard_sample: str | None = None,
    no_self_match: list[str] | None = None,
    common_id_name: str = 'svensemble_id',
    add_common_id: bool = False,
    verbose: bool = False,
) -> tuple[pd.DataFrame, IdGraph]:
    """Match breakpoints across sources and join them side by side.

    Matches the breakpoints (get_pairwise_matches), then marks and drops edges
    until no per-sample component holds two breakpoints of one source and no
    SV holds two per-sample components of one sample
    (same_caller_split_edges, same_sample_split_edges, assign_svensemble_ids).
    Splits the result by sample and puts each per-sample component on one row,
    prefixing each source's columns with its name. Keeps unmatched breakpoints
    as rows of their own only with keep_unmatched. Treats the breakpoints whose
    sample is wildcard_sample as belonging to every sample they match: they
    join each such sample's components and rows, link those samples into one
    SV, and cost more to cut away than any other match.

    Args:
        bp_df_dict (dict[str, pd.DataFrame]): Breakpoint frame of each source,
            keyed by source name.
        ref_callset (str | None): Reference (ground truth) source; matched
            against but never merged in or matched to itself. Defaults to None.
        insert_delta_threshold (float): Minimum similarity threshold for insert
            delta. Defaults to 0.7.
        max_dist (int): Maximum allowed distance between breakpoints. Defaults
            to 10.
        ids_only (bool): Whether to return only the event IDs. Defaults to
            False.
        add_topology (bool): Whether to add columns with topological
            characteristics of breakpoint clusters. Defaults to False.
        ignore_size (bool): If True, skips insert delta filtering. Defaults to
            False.
        workflow_config (dict | None): If given, its
            model.training_params.add_topology overrides add_topology.
            Defaults to None.
        keep_unmatched (bool): Keep every breakpoint, unmatched ones as rows of
            their own (outer join); otherwise only matched ones (inner join).
            Defaults to False.
        wildcard_sample (str | None): Sample of the breakpoints that belong to
            every sample they match, such as GROUND_TRUTH_SAMPLE from
            mark_ground_truth. Defaults to None (no wildcard).
        no_self_match (list[str] | None): Further sources whose breakpoints
            never match each other, besides ref_callset. Defaults to None.
        common_id_name (str): Node attribute (in the returned ID graph) that
            holds the SV IDs. Defaults to 'svensemble_id'.
        add_common_id (bool): Also return the SV IDs as a column named
            common_id_name. Defaults to False.
        verbose (bool): Print how many match edges the pruning drops.
            Defaults to False.

    Returns:
        tuple[pd.DataFrame, IdGraph]: The matched breakpoints from different
            sources side by side, one row per per-sample component; and the
            pruned ID graph, whose nodes carry their SV ID (common_id_name).
    """
    # Combine the matching columns of all breakpoint frames into one with source
    # labels; the other columns join back per source at the end
    zipped_bp_df = concat_bp_dfs({
        name: df[[c for c in df.columns if c in _MATCH_COLS or c in _OPTIONAL_MATCH_COLS]]
        for name, df in bp_df_dict.items()
    })

    if workflow_config is not None:
        add_topology = workflow_config['model']['training_params'].get('add_topology', False)

    match_graph = get_pairwise_matches(
        zipped_bp_df,
        insert_delta_threshold=insert_delta_threshold,
        max_dist=max_dist,
        ignore_size=ignore_size,
        return_graph=True,
        return_unmatched=keep_unmatched,
        skip_single_source_for=([ref_callset] if ref_callset is not None else None),
        no_self_match=sorted(set(no_self_match or []) | ({ref_callset} if ref_callset is not None else set())),
    )

    # Prune: collect the edges whose removal leaves at most one breakpoint per
    # caller in each per-sample component and one per-sample component per
    # sample in each SV, then drop them and label the SVs
    dropped = same_caller_split_edges(match_graph, wildcard_sample=wildcard_sample)
    dropped = same_sample_split_edges(match_graph, dropped, wildcard_sample=wildcard_sample)
    id_graph = assign_svensemble_ids(
        match_graph, dropped, wildcard_sample=wildcard_sample, common_id_name=common_id_name,
    )
    if verbose:
        print(f"Pruned {len(dropped)} of {match_graph.graph.ecount()} match edges")

    # One row per per-sample component, keyed by (SV ID, sample) until the end
    matched_df = flatten_idgraph(
        id_graph, bp_df_dict.keys(), wildcard_sample=wildcard_sample,
        common_id_name=common_id_name, add_common_id=True,
    )
    # Drop rows whose only event is the reference's (without a wildcard sample,
    # reference breakpoints form rows of their own)
    if ref_callset is not None:
        call_cols = [f'{source}_INFO_EVENT' for source in bp_df_dict if source != ref_callset]
        matched_df = matched_df[matched_df[call_cols].notna().any(axis=1)].reset_index(drop=True)

    # Topology per SV: computed on the pruned graph across samples, so every
    # sample's row of one SV shares it; joined on the SV ID alone
    if add_topology:
        ignore_callsets = [ref_callset] if ref_callset is not None else []
        stats = calc_component_stats(id_graph.graph, ignore_callsets=ignore_callsets or None)
        # calc_component_stats skips components made only of ignored sources;
        # key the rest by the SV ID of their first other node
        keys = [
            next(id_graph.graph.vs[v][common_id_name] for v in c
                 if id_graph.graph.vs[v]['name'][0] not in ignore_callsets)
            for c in id_graph.graph.connected_components()
            if any(id_graph.graph.vs[v]['name'][0] not in ignore_callsets for v in c)
        ]
        topology = stats.assign(**{common_id_name: keys}).astype({common_id_name: 'string'})
        matched_df = matched_df.merge(topology, on=common_id_name, how='left')

    # Add each caller's breakpoint columns on (event, sample)
    if not ids_only:
        matched_df = _attach_source_columns(
            matched_df, {source: bp_df for source, bp_df in bp_df_dict.items() if source != ref_callset},
        )

    if not add_common_id:
        matched_df = matched_df.drop(columns=common_id_name)
    return matched_df, id_graph


def inner_join_bps(
    bp_df_dict: dict[str, pd.DataFrame],
    ref_callset: str | None = None,
    insert_delta_threshold: float = 0.7,
    max_dist: int = 10,
    ids_only: bool = False,
    add_topology: bool = False,
    ignore_size: bool = False,
    workflow_config: dict | None = None,
    wildcard_sample: str | None = None,
    no_self_match: list[str] | None = None,
    common_id_name: str = 'svensemble_id',
    add_common_id: bool = False,
    verbose: bool = False,
) -> tuple[pd.DataFrame, IdGraph]:
    """Join matched breakpoints only.

    Calls join_calls with keep_unmatched=False and passes every other argument
    through.

    Args:
        bp_df_dict (dict[str, pd.DataFrame]): Breakpoint frame of each source,
            keyed by source name.
        ref_callset (str | None): Reference (ground truth) source. Defaults to
            None.
        insert_delta_threshold (float): Minimum similarity threshold for insert
            delta. Defaults to 0.7.
        max_dist (int): Maximum allowed distance between breakpoints. Defaults
            to 10.
        ids_only (bool): Whether to return only the event IDs. Defaults to
            False.
        add_topology (bool): Whether to add topology columns. Defaults to False.
        ignore_size (bool): If True, skips insert delta filtering. Defaults to
            False.
        workflow_config (dict | None): Workflow config; see join_calls.
            Defaults to None.
        wildcard_sample (str | None): Sample of the breakpoints that belong to
            every sample they match; see join_calls. Defaults to None.
        no_self_match (list[str] | None): Further sources whose breakpoints
            never match each other; see join_calls. Defaults to None.
        common_id_name (str): Node attribute that holds the SV IDs; see
            join_calls. Defaults to 'svensemble_id'.
        add_common_id (bool): Also return the SV IDs as a column; see
            join_calls. Defaults to False.
        verbose (bool): Print how many match edges the pruning drops.
            Defaults to False.

    Returns:
        tuple[pd.DataFrame, IdGraph]: Matched breakpoints from different sources
            side by side, and the pruned ID graph; see join_calls.
    """
    return join_calls(
        bp_df_dict, ref_callset=ref_callset, insert_delta_threshold=insert_delta_threshold,
        max_dist=max_dist, ids_only=ids_only, add_topology=add_topology, ignore_size=ignore_size,
        workflow_config=workflow_config, keep_unmatched=False, wildcard_sample=wildcard_sample,
        no_self_match=no_self_match, common_id_name=common_id_name,
        add_common_id=add_common_id, verbose=verbose,
    )


def outer_join_bps(
    bp_df_dict: dict,
    ref_callset: str | None = None,
    insert_delta_threshold: float = 0.7,
    max_dist: int = 200,
    ids_only: bool = False,
    ignore_size: bool = False,
    add_topology: bool = False,
    workflow_config: dict | None = None,
    wildcard_sample: str | None = None,
    no_self_match: list[str] | None = None,
    common_id_name: str = 'svensemble_id',
    add_common_id: bool = False,
    verbose: bool = False,
) -> tuple[pd.DataFrame, IdGraph]:
    """
    Merge breakpoint dataframes from multiple callers, aligning on matching events.

    Args:
        bp_df_dict (dict): Dictionary mapping caller names to their breakpoint DataFrames.
        ref_callset (str): Name of the reference callset to use for matching.
        insert_delta_threshold (float, optional): Threshold for insert delta similarity. Defaults to 0.7.
        max_dist (int, optional): Maximum allowed distance between breakpoints to consider a match. Defaults to 200.
        ids_only (bool, optional): Whether to return only the event IDs. Defaults to False.
        add_topology (bool, optional): Whether to add columns with topological
            characteristics of breakpoint clusters. Defaults to False.
        wildcard_sample (str | None, optional): Sample of the breakpoints that
            belong to every sample they match; see join_calls. Defaults to None.
        no_self_match (list[str] | None, optional): Further sources whose
            breakpoints never match each other; see join_calls. Defaults to None.
        common_id_name (str, optional): Node attribute that holds the SV IDs;
            see join_calls. Defaults to 'svensemble_id'.
        add_common_id (bool, optional): Also return the SV IDs as a column;
            see join_calls. Defaults to False.
        verbose (bool, optional): Print how many match edges the pruning drops.
            Defaults to False.

    Returns:
        tuple[pd.DataFrame, IdGraph]: Outer-joined DataFrame containing breakpoints
            from all callers, with columns prefixed by caller name, and the
            pruned ID graph; see join_calls.
    """
    return join_calls(
        bp_df_dict, ref_callset=ref_callset, insert_delta_threshold=insert_delta_threshold,
        max_dist=max_dist, ids_only=ids_only, add_topology=add_topology, ignore_size=ignore_size,
        workflow_config=workflow_config, keep_unmatched=True, wildcard_sample=wildcard_sample,
        no_self_match=no_self_match, common_id_name=common_id_name,
        add_common_id=add_common_id, verbose=verbose,
    )
