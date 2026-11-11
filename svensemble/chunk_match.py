"""
Contains functions used for matching breakpoints from different sets
Following a similar algorith with chunking and creating similarity
matrices to the one used by Truvari
"""
import itertools
from collections.abc import Iterable
from collections import defaultdict
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

def mark_ground_truth(bp_df: pd.DataFrame) -> pd.DataFrame:
    """Ground-truth breakpoints with the GROUND_TRUTH_SAMPLE placeholder sample.

    The ground truth is sites-only, so it has no sample of its own; the
    placeholder lets it match within every sample. A frame that already
    carries SAMPLE_ID_COL is rejected rather than silently relabelled.
    """
    if SAMPLE_ID_COL in bp_df.columns:
        raise ValueError(
            f"Ground truth already has a {SAMPLE_ID_COL} column; truth sets must be sample-less"
        )
    return bp_df.assign(**{SAMPLE_ID_COL: GROUND_TRUTH_SAMPLE})


def concat_bp_dfs(bp_df_dict: dict[str, pd.DataFrame]):
    """
    Concatenates multiple breakpoints dataframes into one, with a new column 'source'
    indicating the source of each breakpoint
    Sorts the resulting dataframe by chromosome, orientation, and positions of breakends.
    This is necessary for the chunker function to work correctly.
    """
    bp_df = pd.concat([
        df.assign(source=name)
        for name, df in bp_df_dict.items()
        if not df.empty
    ]).sort_values(by=['CHROM_bnd1', 'CHROM_bnd2', 'POS_bnd1', 'POS_bnd2'])
    return bp_df

def chunker(bp_df: pd.DataFrame, chunk_size: int = 10000, chunk_max_n: int|None = None, chunk_max_n_dist: int|None = None):
    """
    Splits a breakpoints DataFrame into chunks based on chromosome, orientation, and proximity.

    This function groups breakpoints that are:
    1. On the same chromosome pair with the same relative orientation
    2. Spatially proximate (within chunk_size base pairs)

    Args:
        df (pd.DataFrame): Breakpoints DataFrame that must be pre-sorted by:
                          - chromosome pair
                          - relative orientation of breakends
                          - position of breakend1 and breakend2
        chunk_size (int, optional): Maximum distance in base pairs between
                                   breakpoints to group them together. Defaults to 10000.
        chunk_max_n (int, optional): Requires chunk_max_n_dist to be set. Soft upper limit on the number of breakpoints in a chunk. Defaults to 100.
                                    If the number of breakpoints in a chunk exceeds this value,
                                    and the next element is more than chunk_max_n_dist base pairs away,
                                    the chunk is yielded.
        chunk_max_n_dist (int, optional): Requires chunk_max_n to be set. Soft upper limit on the distance in base pairs between breakpoints in a chunk. Defaults to None.
                                    If the number of breakpoints in a chunk exceeds chunk_max_n,
                                    and the next element is more than chunk_max_n_dist base pairs away,
                                    the chunk is yielded.
    Yields:
        list: Lists of pandas Series representing chunked breakpoints.
              Each chunk contains breakpoints where bnd1 position is within
              chunk_size of the previous breakpoint's bnd2 position.

    Note:
        A breakpoint is grouped with the previous one if:
        row['POS_bnd1'] <= previous_row['POS_bnd1'] + chunk_size
    """
    if (chunk_max_n is not None and chunk_max_n_dist is None) or (chunk_max_n is None and chunk_max_n_dist is not None):
        raise ValueError("chunk_max_n and chunk_max_n_dist must be set together")
    if chunk_max_n is not None and chunk_max_n_dist is not None:
        if chunk_max_n_dist <= 0:
            raise ValueError("chunk_max_n_dist must be greater than 0")
        if chunk_max_n <= 0:
            raise ValueError("chunk_max_n must be greater than 0")

    # Get chromosome and orientation identifiers for grouping
    # chrom_orientation_labels = bps.get_bp_chroms_orientations(bp_df)

    n_chunks = 0
    for _, group in bp_df.groupby(['CHROM_bnd1', 'CHROM_bnd2']):
        current_chunk = []

        # Iterate through each breakpoint in the current group
        for _, breakpoint_row in group.iterrows():
            # Start a new chunk if this is the first breakpoint
            if len(current_chunk) == 0:
                current_chunk.append(breakpoint_row)
                continue

            # Check if the chunk has reached the maximum number of breakpoints and can be yielded
            if chunk_max_n is not None:
                if len(current_chunk) >= chunk_max_n:
                    if breakpoint_row['POS_bnd1'] > current_chunk[-1]['POS_bnd1'] + chunk_max_n_dist:
                        n_chunks += 1
                        yield current_chunk
                        current_chunk = [breakpoint_row]
                        continue

            # Check if current breakpoint is within proximity of the last one in chunk
            if breakpoint_row['POS_bnd1'] <= current_chunk[-1]['POS_bnd1'] + chunk_size:
                # Add to current chunk if within proximity threshold
                current_chunk.append(breakpoint_row)
            else:
                n_chunks += 1
                # Yield completed chunk and start a new one with the current breakpoint added
                yield current_chunk
                current_chunk = [breakpoint_row]
        # Yield the last chunk if it exists
        if len(current_chunk) > 0:
            n_chunks += 1
            yield current_chunk
    print(f'Number of chunks: {n_chunks}')

def dist_matrix_ref(ref_bp_df: pd.DataFrame, bp_df: pd.DataFrame, which_bnd: str) -> np.ndarray:
    """
    Calculate distance matrix between reference breakpoints and other breakpoints.

    This function computes the minimum distance between confidence intervals
    of breakpoints from reference data and breakpoints from other sources.
    The distance is 0 if intervals overlap, otherwise it's the minimum
    separation between the intervals.

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
    # Extract position and confidence interval data for reference breakpoints
    ref_bnd_pos = ref_bp_df[f'POS_{which_bnd}'].values
    ref_bnd_cipos_l = ref_bp_df[f'INFO_CIPOS_l_{which_bnd}'].values
    ref_bnd_cipos_r = ref_bp_df[f'INFO_CIPOS_r_{which_bnd}'].values

    # Extract position and confidence interval data for other breakpoints
    bnd_pos = bp_df[f'POS_{which_bnd}'].values
    bnd_cipos_l = bp_df[f'INFO_CIPOS_l_{which_bnd}'].values
    bnd_cipos_r = bp_df[f'INFO_CIPOS_r_{which_bnd}'].values

    # Initialize distance matrix with zeros (overlapping intervals)
    dists = np.zeros((len(ref_bp_df), len(bp_df)))

    # Calculate distances between interval boundaries
    # Distance from left edge of ref interval to right edge of other interval
    bnd_dist_v1 = np.subtract.outer(
        ref_bnd_pos + ref_bnd_cipos_l,
        bnd_pos + bnd_cipos_r
    )

    # Distance from left edge of other interval to right edge of ref interval
    bnd_dist_v2 = np.subtract.outer(
        bnd_pos + bnd_cipos_l,
        ref_bnd_pos + ref_bnd_cipos_r
    ).T

    # Create boolean masks for non-overlapping intervals
    mask_v1 = bnd_dist_v1 > 0  # Ref interval is to the right of other interval
    mask_v2 = bnd_dist_v2 > 0  # Other interval is to the right of ref interval

    # Assign distances only where intervals don't overlap
    dists[mask_v1] = bnd_dist_v1[mask_v1]
    dists[mask_v2] = bnd_dist_v2[mask_v2]

    return dists


def rel_insert_delta_matrix_ref(ref_bp_df, bp_df, insert_delta_margin=30, ignore_size=False):
    if ignore_size:
        return np.ones((len(ref_bp_df), len(bp_df)))

    ref_deltas = ref_bp_df['insert_delta'].to_numpy(np.float64)[:, np.newaxis]
    comp_deltas = bp_df['insert_delta'].to_numpy(np.float64)[np.newaxis, :]

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
    """
    Generate a match matrix for breakpoints based on their ALT strings.

    Each (bnd1, bnd2) orientation pair is encoded as a single integer so that the
    orientation check is a single O(n²) outer-equality on integers rather than two
    outer-equalities on strings combined with a bitwise AND.

    Args:
        chunk_df (pd.DataFrame): DataFrame containing breakpoint data.
        ins_dup_match (bool): Whether to allow insertions and tandem-duplications to
            match each other.  Enable when merging long- and short-read call-sets.
    Returns:
        np.ndarray: Boolean match matrix of shape (n, n).
    """
    rel1 = _extract_relative_pos(chunk_df['ALT_bnd1'])
    rel2 = _extract_relative_pos(chunk_df['ALT_bnd2'])

    enc1 = pd.Series(rel1).map(_ORIENT_ENCODE).to_numpy(dtype=np.int8, na_value=-1)
    enc2 = pd.Series(rel2).map(_ORIENT_ENCODE).to_numpy(dtype=np.int8, na_value=-1)
    combined = enc1 * 4 + enc2

    orientation_mask = np.equal.outer(combined, combined)

    if ins_dup_match:
        ins_dup_arr = np.isin(combined, list(_INS_DUP_CODES))
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
    # delta_matrix may arrive as float (all-ones sentinel) or bool (threshold result);
    # cast to bool so bitwise-and works uniformly under NumPy 2.x
    return pos_match_matrix & delta_matrix.astype(bool) & alt_match_matrix & ~triu_mask & cross_source_mask

@lru_cache(maxsize=100)
def _get_triu_mask(n: int) -> np.ndarray:
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

    # First, minimize the number of columns in the input dataframe
    # by removing columns that are not needed for matching
    col_retain_list = ['source', 'INFO_EVENT', SAMPLE_ID_COL, 'CHROM_bnd1', 'CHROM_bnd2', 'POS_bnd1', 'POS_bnd2', 'ALT_bnd1', 'ALT_bnd2', 'INFO_CIPOS_l_bnd1', 'INFO_CIPOS_r_bnd1', 'INFO_CIPOS_l_bnd2', 'INFO_CIPOS_r_bnd2', 'INFO_SVTYPE']
    if 'INFO_SVLEN' in zipped_bp_df.columns:
        col_retain_list.append('INFO_SVLEN')
    if 'svensemble2_insert_size' in zipped_bp_df.columns:
        col_retain_list.append('svensemble2_insert_size')
    if not set(col_retain_list).issubset(zipped_bp_df.columns):
        raise ValueError(f"Some of the required columns are missing from the input dataframe: {set(col_retain_list) - set(zipped_bp_df.columns)}")
    zipped_bp_df_ = zipped_bp_df[col_retain_list].copy()

    raw_insert_size = bps.get_insert_size(zipped_bp_df_)

    same_chrom = zipped_bp_df_['CHROM_bnd1'] == zipped_bp_df_['CHROM_bnd2']
    pos_diff = zipped_bp_df_['POS_bnd2'] - zipped_bp_df_['POS_bnd1']

    insert_delta = raw_insert_size.copy()
    insert_delta[same_chrom] = np.abs(
        raw_insert_size[same_chrom] - (pos_diff[same_chrom] - 1)
    )
    zipped_bp_df_.loc[:, 'insert_delta'] = insert_delta

    matched_pairs = []
    unmatched = []

    chunk_max_n = 100
    chunk_max_n_dist = max_dist
    for chunk in chunker(zipped_bp_df_, chunk_max_n=chunk_max_n, chunk_max_n_dist=chunk_max_n_dist):
        # Skip chunks with insufficient breakpoints or single source
        if len(chunk) <= 1:
            if (skip_single_source_for == "all") or (isinstance(skip_single_source_for, list) and (chunk[0]['source'] in skip_single_source_for)):
                continue
            if return_unmatched:
                unmatched.append((chunk[0]['source'], chunk[0]['INFO_EVENT'], str(chunk[0][SAMPLE_ID_COL])))
            continue


        chunk_df = pd.DataFrame(chunk)
        if chunk_df['source'].nunique() == 1:
            single_source = chunk_df['source'].values[0]
            if (skip_single_source_for == "all") or (isinstance(skip_single_source_for, list) and (single_source in skip_single_source_for)):
                continue

        n = len(chunk_df)

        # Build cross-source mask: False where row i and column j both come from one
        # no_self_match source. Uses the outer-product subtraction trick: start from
        # all-ones and zero out each listed source's self-pair block.
        cross_source_mask = np.ones((n, n), dtype=float)
        for source in np.intersect1d(chunk_df['source'].values, no_self_match or []):
            m = chunk_df['source'].values == source
            cross_source_mask -= np.outer(m, m)
        cross_source_mask = cross_source_mask.astype(bool)

        # Calculate distance matrices
        bp1_matrix = dist_matrix_ref(chunk_df, chunk_df, BND1)
        bp2_matrix = dist_matrix_ref(chunk_df, chunk_df, BND2)

        # Create position match matrix
        pos_match_matrix = (bp1_matrix <= max_dist) & (bp2_matrix <= max_dist)

        # Calculate insert delta similarity
        if ignore_size:
            delta_matrix = np.ones((n, n))
        else:
            delta_matrix = (
                rel_insert_delta_matrix_ref(chunk_df, chunk_df)
                >= insert_delta_threshold
            )

        alt_mask = alt_match_matrix(chunk_df, ins_dup_match=ins_dup_match)


        # Use cached triu mask; cross_source_mask already filters no_self_match pairs
        triu_mask = _get_triu_mask(n)

        # Create final match matrix (cross-source filter applied inside the matrix)
        match_matrix = _get_match_matrix(pos_match_matrix, delta_matrix, alt_mask, triu_mask, cross_source_mask)

        nodes = list(zip(
            chunk_df['source'].values,
            chunk_df['INFO_EVENT'].values,
            chunk_df[SAMPLE_ID_COL].astype(str).values,
        ))

        # Check if any matches exist in the matrix
        matched_indices = np.zeros(n, dtype=int)
        if np.any(match_matrix):
            i_indices, j_indices = np.where(match_matrix)

            matched_indices[i_indices] = 1
            matched_indices[j_indices] = 1

            matched_pairs.extend((nodes[i], nodes[j]) for i, j in zip(i_indices, j_indices))

        if return_unmatched:
            unmatched.extend(nodes[i] for i in np.where(matched_indices == 0)[0])

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
    """The sample of every node of *graph*, in node order."""
    return [name[2] for name in graph.vs['name']] if graph.vcount() else []


def _wildcard_mask(graph: ig.Graph, wildcard_sample: str | None) -> np.ndarray:
    """Boolean mask over graph.vs: nodes whose sample is *wildcard_sample* (none if it is None)."""
    return np.array([s == wildcard_sample for s in _node_samples(graph)], dtype=bool)


def sample_vertex_mask(
    graph_in: ig.Graph,
    samples: Iterable[str],
    wildcard_sample: str | None = None,
) -> np.ndarray:
    """Boolean mask over graph_in.vs: nodes whose sample is in *samples*.

    Also selects the nodes whose sample is *wildcard_sample*, since those
    belong to every sample.
    """
    wanted = set(samples) | ({wildcard_sample} if wildcard_sample is not None else set())
    return np.array([s in wanted for s in _node_samples(graph_in)], dtype=bool)


def split_graph_by_sample(graph_in: ig.Graph, wildcard_sample: str | None = None) -> dict[str, ig.Graph]:
    """One subgraph per sample: that sample's nodes plus the wildcard nodes.

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
    """The sample of a per-sample component: its one sample other than *wildcard_sample*."""
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
    left = left.copy()
    source_df = source_df.copy()
    for df, cols in ((left, (event_col, SAMPLE_ID_COL)), (source_df, (event_col, source_sample_col))):
        for col in cols:
            df[col] = df[col].astype('string')
    merged = left.merge(
        source_df, left_on=[event_col, SAMPLE_ID_COL], right_on=[event_col, source_sample_col], how=how,
    )
    merged[SAMPLE_ID_COL] = merged[SAMPLE_ID_COL].fillna(merged[source_sample_col])
    return merged.drop(columns=[source_sample_col])


def _kept_view(graph: ig.Graph, dropped: set[int]) -> ig.Graph:
    """Copy *graph* without the *dropped* edges, tagging every edge with its ID in
    *graph* ('eid') and every node with its ID ('vid').

    igraph carries the tags through subgraphs, so cut edges map straight back to
    *graph*. subgraph_edges keeps the edges in ID order, which matches *kept*.
    """
    kept = [e for e in range(graph.ecount()) if e not in dropped]
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
    heavy = graph.ecount() + 1
    return [heavy if is_wild[u] or is_wild[v] else 1 for u, v in graph.get_edgelist()]


def _most_separating(cuts: list, conflicts: list[tuple[int, int]], cost: list[float]):
    """The cut that separates the most conflicting vertex pairs per unit of edge *cost*
    it removes; the first wins ties."""
    def separated_per_cost(cut):
        side = set(cut.partition[0])
        return sum((a in side) != (b in side) for a, b in conflicts) / sum(cost[e] for e in cut.cut)
    return max(cuts, key=separated_per_cost)


def same_caller_split_edges(
    idg: IdGraph, dropped: Iterable[int] = (), wildcard_sample: str | None = None,
) -> np.ndarray:
    """Collect edges whose removal leaves no source twice in a per-sample component.

    Each round, removes the marked edges from a copy of the graph, splits it by
    sample (split_graph_by_sample, with the wildcard nodes in every sample they
    match), and in every component that holds two breakpoints of one source,
    separates them with a minimum cut and marks the cut edges. Stops when no
    component holds a source twice. Wildcard (ground-truth) breakpoints
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
    while True:
        view = _kept_view(graph, dropped)
        new = []
        for sample_graph in split_graph_by_sample(view, wildcard_sample).values():
            for component in sample_graph.connected_components():
                sub = sample_graph.subgraph(component)
                names = sub.vs['name']
                by_source = defaultdict(list)
                for i, name in enumerate(names):
                    by_source[name[0]].append(i)
                duplicated = sorted(s for s, members in by_source.items() if len(members) > 1)
                if not duplicated:
                    continue
                # Every two breakpoints of one source conflict
                conflicts = [p for s in duplicated for p in itertools.combinations(by_source[s], 2)]
                # One candidate cut per duplicated source, between its two best-connected
                # breakpoints, adding eps to the kept one's edges
                cost = _edge_costs(sub, _wildcard_mask(sub, wildcard_sample))
                edges = sub.get_edgelist()
                degree = sub.degree()
                cuts = []
                for s in duplicated:
                    kept, other = sorted(by_source[s], key=lambda i: (-degree[i], str(names[i][1])))[:2]
                    eps = 1 / (2 * (degree[kept] + 1))
                    capacity = [c + eps if kept in e else c for c, e in zip(cost, edges)]
                    cuts.append(sub.mincut(source=kept, target=other, capacity=capacity))
                new.extend(sub.es[_most_separating(cuts, conflicts, cost).cut]['eid'])
        if not new:
            return np.array(sorted(dropped), dtype=np.int64)
        dropped.update(new)


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

    Within an SV, computes one candidate cut per pair of same-sample
    components and applies the candidate that separates the most such pairs
    per unit of cost cut (ties: alphabetically first sample, then
    lowest-numbered components). Prices every edge at 1, and an edge to a
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

        new = []
        for sv in view.connected_components():
            per_sample = defaultdict(set)
            for v in sv:
                if not is_wild[v]:
                    per_sample[sample_of[v]].add(component_of[v])
            conflicting = sorted(s for s, comps in per_sample.items() if len(comps) > 1)
            if not conflicting:
                continue
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
            cuts = [quotient.mincut(source=a, target=b, capacity=cost) for a, b in conflicts]
            new.extend(quotient.es[_most_separating(cuts, conflicts, cost).cut]['eid'])
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
    # Combine all breakpoint dataframes into one with source labels
    zipped_bp_df = concat_bp_dfs(bp_df_dict)

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
        for source, bp_df in bp_df_dict.items():
            if source == ref_callset:
                continue
            prefixed = bp_df.assign(INFO_EVENT=bp_df['INFO_EVENT'].astype('string')).add_prefix(f'{source}_')
            matched_df = _merge_on_event_and_sample(matched_df, prefixed, source, how='left')

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
