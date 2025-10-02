"""
Contains functions used for matching breakpoints from different sets
Following a similar algorith with chunking and creating similarity
matrices to the one used by Truvari
"""
from collections import defaultdict
from functools import lru_cache

import numpy as np
import pandas as pd

from . import breakpoints as bps
from . import breakends as bnds
from . import events as evts
from . import _caller_params as caller_params

def concat_bp_dfs(bp_df_dict: dict[str, pd.DataFrame]):
    """
    Concatenates multiple breakpoints dataframes into one, with a new column 'source'
    indicating the source of each breakpoint
    Sorts the resulting dataframe by chromosome, orientation, and positions of breakends.
    This is necessary for the chunker function to work correctly.
    """
    bp_df_dict_ = {source: bp_df.copy() for source, bp_df in bp_df_dict.items()}
    for name, bp_df in bp_df_dict_.items():
        bp_df.loc[:, 'source'] = name
    bp_df = pd.concat(bp_df_dict_.values())
    bp_df = bp_df.sort_values(
        by=['CHROM_bnd1', 'CHROM_bnd2', 'POS_bnd1', 'POS_bnd2']
    )
    return bp_df

def chunker(bp_df: pd.DataFrame, chunk_size: int = 10000):
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
                                   breakpoints to group them together. Defaults to 5000.

    Yields:
        list: Lists of pandas Series representing chunked breakpoints.
              Each chunk contains breakpoints where bnd1 position is within
              chunk_size of the previous breakpoint's bnd2 position.

    Note:
        A breakpoint is grouped with the previous one if:
        row['POS_bnd1'] <= previous_row['POS_bnd1'] + chunk_size
    """
    # Get chromosome and orientation identifiers for grouping
    chrom_orientation_labels = bps.get_bp_chroms_orientations(bp_df)

    # Process each chromosome-orientation group separately
    n_chunks = 0
    for chrom_orientation_key, group in bp_df.groupby(chrom_orientation_labels):
        current_chunk = []

        # Iterate through each breakpoint in the current group
        for row_index, breakpoint_row in group.iterrows():
            # Start a new chunk if this is the first breakpoint
            if len(current_chunk) == 0:
                current_chunk.append(breakpoint_row)
                continue

            # Check if current breakpoint is within proximity of the last one in chunk
            if breakpoint_row['POS_bnd1'] <= current_chunk[-1]['POS_bnd1'] + chunk_size:
                # Add to current chunk if within proximity threshold
                current_chunk.append(breakpoint_row)
            else:
                n_chunks += 1
                # Yield completed chunk and start a new one
                yield current_chunk
                current_chunk = []
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
        ref_bnd_pos - ref_bnd_cipos_l,
        bnd_pos + bnd_cipos_r
    )

    # Distance from left edge of other interval to right edge of ref interval
    bnd_dist_v2 = np.subtract.outer(
        bnd_pos - bnd_cipos_l,
        ref_bnd_pos + ref_bnd_cipos_r
    ).T

    # Create boolean masks for non-overlapping intervals
    mask_v1 = bnd_dist_v1 > 0  # Ref interval is to the right of other interval
    mask_v2 = bnd_dist_v2 > 0  # Other interval is to the right of ref interval

    # Assign distances only where intervals don't overlap
    dists[mask_v1] = bnd_dist_v1[mask_v1]
    dists[mask_v2] = bnd_dist_v2[mask_v2]

    return dists

insert_source_dispatch = {
    'svlen': evts.get_insert_seq_len_from_svlen,
    'alt': bnds.get_insert_seq_len_from_alt,
    'svensemble2_col': bps.get_insert_seq_len_from_svensemble2_len_col
}

def get_insert_size(bp_df: pd.DataFrame) -> pd.Series:
    """
    Calculates insert sequence length from ALT or SVLEN column
    Requires 'source' column to be present in the dataframe
    """
    insert_sizes = pd.Series(index=bp_df.index)
    for source in bp_df['source'].unique():
        insert_sizes.loc[bp_df['source'] == source] = (
            insert_source_dispatch[caller_params.caller_param_dict[source]['svlen_source']](
                bp_df[bp_df['source'] == source]
            )
        )
    insert_sizes.loc[insert_sizes < 0] = 0
    insert_sizes = insert_sizes.fillna(0)  # Convert NAs to 0
    insert_sizes = insert_sizes.astype(int)
    return insert_sizes


def _get_zeros(shape: tuple[int, int]) -> np.ndarray:
    return np.zeros(shape, dtype=np.float64)


def rel_insert_size_matrix_ref(
    ref_bp_df: pd.DataFrame,
    bp_df: pd.DataFrame,
    insert_size_margin: int = 10,
    ignore_size: bool = False
):
    """
    Calculate relative insert size matrix between reference breakpoints and other breakpoints.
    To use insert size, requires 'insert_size' column to be present in the dataframe.
    Insert sizes must be integers.
    """
    if ignore_size:
        return np.ones((len(ref_bp_df), len(bp_df)))

    insert_sizes_ref = pd.Series(index=ref_bp_df.index)
    insert_sizes_comp = pd.Series(index=bp_df.index)

    # Validate and collect insert sizes
    for source in ref_bp_df['source'].unique():
        source_sizes = ref_bp_df[ref_bp_df['source'] == source]['insert_size']
        if source_sizes.dtype.kind not in 'iu':  # Check if integer or unsigned integer
            raise TypeError(
                f"Insert sizes for source {source} must be integers, got {source_sizes.dtype}"
            )
        insert_sizes_ref.loc[ref_bp_df['source'] == source] = source_sizes

    for source in bp_df['source'].unique():
        source_sizes = bp_df[bp_df['source'] == source]['insert_size']
        if source_sizes.dtype.kind not in 'iu':  # Check if integer or unsigned integer
            raise TypeError(
                f"Insert sizes for source {source} must be integers, got {source_sizes.dtype}"
            )
        insert_sizes_comp.loc[bp_df['source'] == source] = source_sizes

    # Create matrix where smaller insert sizes are divided by larger ones
    # Convert to numpy arrays for broadcasting
    ref_sizes = insert_sizes_ref.values[:, np.newaxis]  # Shape: (n_ref, 1)
    comp_sizes = insert_sizes_comp.values[np.newaxis, :]  # Shape: (1, n_comp)

    # Use broadcasting to create division matrices with explicit type handling
    # Convert to float64 for division to avoid integer division issues
    # Result shape should be (n_ref, n_comp)
    out_shape = (ref_sizes.shape[0], comp_sizes.shape[1])
    insert_matrix1 = np.divide(
        ref_sizes.astype(np.float64),
        comp_sizes.astype(np.float64),
        out=_get_zeros(out_shape),
        where=comp_sizes > 0
    )
    insert_matrix2 = np.divide(
        comp_sizes.astype(np.float64),
        ref_sizes.astype(np.float64),
        out=_get_zeros(out_shape),
        where=ref_sizes > 0
    )

    # For each element, ensure we divide smaller by larger
    insert_matrix = np.minimum(insert_matrix1, insert_matrix2)

    # If both insert sizes are near-zero (within insert_size_margin), set the similarity to 1.0
    both_near_zero = (ref_sizes <= insert_size_margin) & (comp_sizes <= insert_size_margin)
    insert_matrix = np.where(both_near_zero, 1.0, insert_matrix)

    return insert_matrix

def _get_match_matrix(
    pos_match_matrix: np.ndarray,
    insert_matrix: np.ndarray,
    triu_mask: np.ndarray
) -> np.ndarray:
    return pos_match_matrix & insert_matrix & ~triu_mask

@lru_cache(maxsize=100)
def _get_triu_mask(n: int) -> np.ndarray:
    return np.triu(np.ones(n, dtype=bool))

def get_pairwise_matches(
    zipped_bp_df: pd.DataFrame,
    insert_size_threshold: float = 0.7,
    max_dist: int = 10,
    ignore_size: bool = False
) -> list[tuple[tuple[str, str], tuple[str, str]]]:
    """
    Match breakpoints from multiple sources based on position and insert size similarity.

    Args:
        zipped_bp_df (pd.DataFrame): DataFrame containing breakpoint data from multiple sources
        insert_size_threshold (float): Minimum similarity threshold for insert size (0-1)
        max_dist (int): Maximum allowed distance between breakpoints

    Returns:
        list[tuple[tuple[str, str], tuple[str, str]]]: List of matched breakpoint pairs,
        where each pair contains tuples of (source, event_id) for both breakpoints
    """
    # First, minimize the number of columns in the input dataframe
    # by removing columns that are not needed for matching
    col_retain_list = ['source', 'INFO_EVENT', 'CHROM_bnd1', 'CHROM_bnd2', 'POS_bnd1', 'POS_bnd2', 'ALT_bnd1', 'ALT_bnd2', 'INFO_CIPOS_l_bnd1', 'INFO_CIPOS_r_bnd1', 'INFO_CIPOS_l_bnd2', 'INFO_CIPOS_r_bnd2', 'INFO_SVTYPE']
    if 'INFO_SVLEN' in zipped_bp_df.columns:
        col_retain_list.append('INFO_SVLEN')
    if 'svensemble2_insert_size' in zipped_bp_df.columns:
        col_retain_list.append('svensemble2_insert_size')
    zipped_bp_df_ = zipped_bp_df[col_retain_list].copy()

    # unique_sources = list(zipped_bp_df['source'].unique())
    zipped_bp_df_.loc[:, 'insert_size'] = get_insert_size(zipped_bp_df_)

    # Generate all possible source combinations
    # source_combinations = [
    #     f'{source1}_{source2}'
    #     for source1 in unique_sources
    #     for source2 in unique_sources
    # ]

    # Generate combinations between different sources only
    # source_combinations_diff = [
    #     f'{source1}_{source2}'
    #     for source1 in unique_sources
    #     for source2 in unique_sources
    #     if source1 != source2
    # ]

    matched_pairs = []



    for chunk in chunker(zipped_bp_df_):
        # Skip chunks with insufficient breakpoints or single source
        if len(chunk) <= 1:
            continue

        chunk_df = pd.DataFrame(chunk)
        if len(set(chunk_df['source'])) == 1:
            continue

        # # Create source combination masks
        # source_masks = {
        #     source_pair: np.zeros((len(chunk_df), len(chunk_df)), dtype=bool)
        #     for source_pair in source_combinations
        # }

        # # Populate source masks
        # for i, source1 in enumerate(chunk_df['source'].values):
        #     for j, source2 in enumerate(chunk_df['source'].values):
        #         source_masks[f'{source1}_{source2}'][i, j] = True

        # # Calculate matrix for different source breakpoints
        # sources_matrix = np.sum(
        #     [
        #         mask for pair, mask in source_masks.items()
        #         if pair in source_combinations_diff
        #     ],
        #     axis=0
        # ).astype(bool)

        # Calculate distance matrices
        bp1_matrix = dist_matrix_ref(chunk_df, chunk_df, 'bnd1')
        bp2_matrix = dist_matrix_ref(chunk_df, chunk_df, 'bnd2')

        # Create position match matrix
        pos_match_matrix = (bp1_matrix <= max_dist) & (bp2_matrix <= max_dist)

        # # Calculate minimum distance matches per source combination
        # dist_sum_matrix = bp1_matrix + bp2_matrix
        # dist_sum_matrix[np.triu_indices_from(dist_sum_matrix)] = np.inf

        # is_min_dist_matrix = np.zeros((len(chunk_df), len(chunk_df)), dtype=bool)
        # for source_pair in source_combinations_diff:
        #     dist_sum_copy = dist_sum_matrix.copy()
        #     dist_sum_copy[~source_masks[source_pair]] = np.inf

        #     # Find minimum in each row and column
        #     row_mins = np.min(dist_sum_copy, axis=1, keepdims=True)
        #     col_mins = np.min(dist_sum_copy, axis=0, keepdims=True)

        #     # Element is minimum if it equals both row and column minimum
        #     min_dist_mask = (dist_sum_copy == row_mins) & (dist_sum_copy == col_mins)
        #     is_min_dist_matrix |= min_dist_mask

        # Calculate insert size similarity
        insert_matrix = (
            rel_insert_size_matrix_ref(chunk_df, chunk_df, ignore_size=ignore_size)
            > insert_size_threshold
        )

        # Use cached triangular matrix
        n = len(chunk_df)
        triu_mask = np.triu(np.ones(n, dtype=bool))

        # Create final match matrix
        match_matrix = _get_match_matrix(pos_match_matrix, insert_matrix, triu_mask)

        # Check if any matches exist in the matrix
        if np.any(match_matrix):
            i_indices, j_indices = np.where(match_matrix)
            sources = chunk_df['source'].values
            events = chunk_df['INFO_EVENT'].values

            matched_pairs.extend(
                [
                    ((sources[i], events[i]), (sources[j], events[j]))
                    for i, j in zip(i_indices, j_indices)
                ]
            )

    return matched_pairs

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

def build_matched_bp_df(
    bp_df_dict: dict[str, pd.DataFrame],
    matched_lists: list[list[tuple[str, str]]],
    ref_callset: str | None = None,
    ids_only: bool = False
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
    Returns:
        pd.DataFrame: DataFrame where each row contains the matched breakpoint data from
            different sources side by side, with source-specific column prefixes. Only
            includes rows where breakpoints were matched across at least 2 sources.
    """
    # Get unique matched events across all components
    unique_matched_events = {
        bp_event
        for matched_list in matched_lists
        for _, bp_event in matched_list
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
    id_table = pd.DataFrame(index=range(len(matched_lists)), columns=id_table_columns, dtype='string')
    for i, matched_list in enumerate(matched_lists):
        sources_seen = set()
        for bp_source, bp_event in matched_list:
            if bp_source in sources_seen:
                continue
            sources_seen.add(bp_source)
            id_table.loc[i, f'{bp_source}_INFO_EVENT'] = bp_event

    # Remove empty rows from the id_table
    id_table = id_table.dropna(axis=0, how='all')

    # Remove rows where the only event is from the reference callset
    if ref_callset is not None:
        non_ref_notna_mask = np.any(id_table.loc[:, ~id_table.columns.str.startswith(ref_callset)].notna(), axis=1)
        id_table = id_table[non_ref_notna_mask]

    if ids_only:
        return id_table

    # Left join the id-table and each of the breakpoint dataframes
    for source, bp_df in bp_df_dict_prefixes.items():
        if source == ref_callset:
            continue

        try:
            join_key = f'{source}_INFO_EVENT'
            # Ensure both sides are string dtype on the join key
            if join_key in id_table.columns:
                id_table.loc[:, join_key] = id_table.loc[:, join_key].astype('string')
            if join_key in bp_df.columns:
                bp_df = bp_df.copy()
                bp_df.loc[:, join_key] = bp_df.loc[:, join_key].astype('string')
            id_table = id_table.merge(bp_df, left_on=join_key, right_on=join_key, how='left')
        except ValueError:
            raise ValueError(f"Error merging {source} breakpoint dataframe with id_table")

    return id_table


def inner_join_bps(
    bp_df_dict: dict[str, pd.DataFrame],
    ref_callset: str | None = None,
    insert_size_threshold: float = 0.7,
    max_dist: int = 10,
    ids_only: bool = False,
    ignore_size: bool = False
) -> pd.DataFrame:
    """Find all pairwise matches in the breakpoint dataframe dictionary
    and return a dataframe with the matches.

    Args:
        bp_df_dict (dict[str, pd.DataFrame]): Dictionary mapping source names
        to their breakpoint DataFrames
        insert_size_threshold (float, optional): Minimum similarity threshold
        for insert size. Defaults to 0.7.
        max_dist (int, optional): Maximum allowed distance between breakpoints. Defaults to 10.
        ids_only (bool, optional): Whether to return only the event IDs. Defaults to False.
    Returns:
        pd.DataFrame: DataFrame containing matched breakpoints from different sources side by side
    """
    # Combine all breakpoint dataframes into one with source labels
    zipped_bp_df = concat_bp_dfs(bp_df_dict)

    # Find all pairwise matches between breakpoints
    matched_pairs = get_pairwise_matches(
        zipped_bp_df,
        insert_size_threshold=insert_size_threshold,
        max_dist=max_dist,
        ignore_size=ignore_size
    )

    # Merge connected pairs into lists of related breakpoints
    matched_lists = merge_bp_pairs_dfs(matched_pairs)

    # Build final dataframe with matched breakpoints side by side
    matched_bp_df = build_matched_bp_df(bp_df_dict, matched_lists, ref_callset, ids_only)

    return matched_bp_df

def outer_join_bps(
    bp_df_dict: dict,
    ref_callset: str | None = None,
    insert_size_threshold: float = 0.7,
    max_dist: int = 200,
    ids_only: bool = False,
    ignore_size: bool = False
) -> pd.DataFrame:
    """
    Merge breakpoint dataframes from multiple callers, aligning on matching events.

    Args:
        bp_df_dict (dict): Dictionary mapping caller names to their breakpoint DataFrames.
        ref_callset (str): Name of the reference callset to use for matching.
        insert_size_threshold (float, optional): Threshold for insert size similarity. Defaults to 0.7.
        max_dist (int, optional): Maximum allowed distance between breakpoints to consider a match. Defaults to 200.
        ids_only (bool, optional): Whether to return only the event IDs. Defaults to False.

    Returns:
        pd.DataFrame: Outer-joined DataFrame containing breakpoints from all callers,
                      with columns suffixed by caller name.
    """
    # Obtain matched breakpoints using the inner_join_bps utility
    matched_bp_df = inner_join_bps(
        bp_df_dict=bp_df_dict,
        ref_callset=ref_callset,
        insert_size_threshold=insert_size_threshold,
        max_dist=max_dist,
        ids_only=True,
        ignore_size=ignore_size
    )

    # Ensure DataFrame type for subsequent merges
    if not isinstance(matched_bp_df, pd.DataFrame):
        raise ValueError('matched_bp_df is not a DataFrame')

    # Start with a copy of the matched breakpoints DataFrame
    joint_bp_df = matched_bp_df.copy()

    # Merge in each caller's DataFrame, except the reference callset
    for caller, df in bp_df_dict.items():
        if caller == ref_callset:
            # Skip the reference callset; only need to know if calls match to it
            continue

        # Only keep the INFO_EVENT column if ids_only is True
        if ids_only:
            df_prefixed = df[['INFO_EVENT']]
        else:
            df_prefixed = df.copy()

        # Add a prefix to all columns to distinguish between callers
        df_prefixed = df_prefixed.add_prefix(f'{caller}_')

        # Align join key dtype on both sides
        join_key = f'{caller}_INFO_EVENT'
        if join_key in joint_bp_df.columns:
            joint_bp_df.loc[:, join_key] = joint_bp_df.loc[:, join_key].astype('string')
        if join_key in df_prefixed.columns:
            df_prefixed = df_prefixed.copy()
            df_prefixed.loc[:, join_key] = df_prefixed.loc[:, join_key].astype('string')

        # Merge on the INFO_EVENT column, using an outer join to retain all breakpoints
        joint_bp_df = joint_bp_df.merge(
            df_prefixed,
            left_on=join_key,
            right_on=join_key,
            how='outer'
        )

    return joint_bp_df
