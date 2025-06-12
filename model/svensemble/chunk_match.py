"""
Contains functions used for matching breakpoints from different sets
Following a similar algorith with chunking and creating similarity
matrices to the one used by Truvari
"""
from collections import defaultdict

import numpy as np
import pandas as pd

from . import breakpoints as bps
from . import breakends as bnds
from . import events as evts
from . import _caller_params as caller_params

def zip_bp_dfs(bp_df_dict: dict[str, pd.DataFrame]):
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

def chunker(bp_df: pd.DataFrame, chunk_size: int = 5000):
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
                # Yield completed chunk and start a new one
                yield current_chunk
                current_chunk = []
        # Yield the last chunk if it exists
        if len(current_chunk) > 0:
            yield current_chunk

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
    'alt': bnds.get_insert_seq_len_from_alt
}

def get_insert_size(bp_df: pd.DataFrame) -> pd.Series:
    """
    Calculates insert sequence length from ALT or SVLEN column
    Requires 'source' column to be present in the dataframe
    """
    insert_sizes = pd.Series(index=bp_df.index)
    for source in bp_df['source'].unique():
        insert_sizes.loc[bp_df['source'] == source] = (
            insert_source_dispatch[caller_params.insert_seq_source[source]](
                bp_df[bp_df['source'] == source]
            )
        )
    insert_sizes.loc[insert_sizes < 0] = 0
    insert_sizes = insert_sizes.fillna(0)  # Convert NAs to 0
    insert_sizes = insert_sizes.astype(int)
    return insert_sizes

def rel_insert_size_matrix_ref(
    ref_bp_df: pd.DataFrame,
    bp_df: pd.DataFrame,
    insert_size_margin: int = 10
):
    """
    Calculate relative insert size matrix between reference breakpoints and other breakpoints.
    To use insert size, requires 'insert_size' column to be present in the dataframe.
    Insert sizes must be integers.
    """
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
    insert_matrix1 = np.divide(
        ref_sizes.astype(np.float64),
        comp_sizes.astype(np.float64),
        out=np.zeros_like(ref_sizes * comp_sizes, dtype=np.float64),
        where=comp_sizes > 0
    )
    insert_matrix2 = np.divide(
        comp_sizes.astype(np.float64),
        ref_sizes.astype(np.float64),
        out=np.zeros_like(comp_sizes * ref_sizes, dtype=np.float64),
        where=ref_sizes > 0
    )

    # For each element, ensure we divide smaller by larger
    insert_matrix = np.minimum(insert_matrix1, insert_matrix2)

    # If both insert sizes are near-zero (within insert_size_margin), set the value to 1
    insert_matrix = np.where(
        np.logical_and(
            np.abs(insert_matrix1) < insert_size_margin,
            np.abs(insert_matrix2) < insert_size_margin
        ),
        1,
        insert_matrix
    )

    return insert_matrix


def get_ref_comp_matches(
    zipped_bp_df: pd.DataFrame,
    sources_ref: list[str],
    sources_comp: list[str],
    max_dist: int = 10,
    insert_size_threshold: float = 0.7
) -> list[pd.DataFrame]:
    """
    Find matches between reference and comparison breakpoint datasets.

    Args:
        zipped_bp_df: DataFrame containing breakpoint data from multiple sources
        sources_ref: List of reference source names
        sources_comp: List of comparison source names
        max_dist: Maximum allowed distance between breakpoints
        insert_size_threshold: Minimum similarity threshold for insert size (0-1)

    Returns:
        List of DataFrames containing matches between reference and comparison breakpoints
    """
    matches = []
    zipped_bp_df.loc[:, 'insert_size'] = get_insert_size(zipped_bp_df)

    for chunk in chunker(zipped_bp_df):
        if len(chunk) <= 1:
            continue

        chunk_df = pd.DataFrame(chunk)
        if len(set(chunk_df['source'])) == 1:
            continue

        # Split into reference and comparison datasets
        match_to_df = chunk_df[chunk_df['source'].isin(sources_ref)]
        match_df = chunk_df[chunk_df['source'].isin(sources_comp)]

        # Calculate distance matrices for both breakpoint ends
        bp1_matrix = dist_matrix_ref(match_to_df, match_df, 'bnd1')
        bp2_matrix = dist_matrix_ref(match_to_df, match_df, 'bnd2')

        # Create position match matrix
        pos_match_matrix = (bp1_matrix <= max_dist) & (bp2_matrix <= max_dist)

        # Calculate insert size similarity matrix
        insert_matrix = rel_insert_size_matrix_ref(match_to_df, match_df) > insert_size_threshold

        # Combine conditions for final match matrix
        match_matrix = pos_match_matrix & insert_matrix

        if np.any(match_matrix):
            comp_events = match_df['INFO_EVENT'].values
            ref_events = match_to_df['INFO_EVENT'].values

            match_matrix_df = pd.DataFrame(
                match_matrix,
                index=ref_events,
                columns=comp_events
            )
            matches.append(
                match_matrix_df.loc[
                    match_matrix_df.any(axis=1),
                    match_matrix_df.any(axis=0)
                ]
            )

    return matches


def matches_pick_single(matches: list[pd.DataFrame]) -> list[pd.DataFrame]:
    """
    Select only the first match for each event when multiple matches exist.

    Args:
        matches: List of DataFrames containing matches

    Returns:
        List of DataFrames with only single matches per event
    """
    matches_single_only = []

    for match in matches:
        if match.shape[0] > 1 or match.shape[1] > 1:
            for i, match_row in match.iterrows():
                for j, match_status in match_row.items():
                    if match_status:
                        matches_single_only.append(
                            pd.DataFrame({j: match_status}, index=[i])
                        )
                        break
        else:
            matches_single_only.append(match)

    return matches_single_only


def matches_pick_all(matches: list[pd.DataFrame]) -> list[pd.DataFrame]:
    """
    Keep all matches for each event when multiple matches exist.

    Args:
        matches: List of DataFrames containing matches

    Returns:
        List of DataFrames with all matches per event
    """
    matches_all = []

    for match in matches:
        if match.shape[0] > 1 or match.shape[1] > 1:
            for i, match_row in match.iterrows():
                for j, match_status in match_row.items():
                    if match_status:
                        matches_all.append(
                            pd.DataFrame({j: match_status}, index=[i])
                        )
        else:
            matches_all.append(match)

    return matches_all


def matches_to_df(matches: list[pd.DataFrame]) -> pd.DataFrame:
    """
    Convert list of match DataFrames into a single DataFrame with match pairs.

    Args:
        matches: List of DataFrames containing matches

    Returns:
        DataFrame with columns 'match_to' and 'match' containing matched event IDs
    """
    matches_for_df = {'match_to': [], 'match': []}

    for match in matches:
        try:
            matches_for_df['match_to'].append(match.index[0])
            matches_for_df['match'].append(match.columns[0])
        except AttributeError:
            print(f"Error processing match: {match}")
            break

    return pd.DataFrame(matches_for_df)

def get_ref_comp_match_df(
    zipped_bp_df: pd.DataFrame,
    sources_ref: list[str],
    sources_comp: list[str],
    max_dist: int = 10,
    insert_size_threshold: float = 0.7
) -> pd.DataFrame:
    """
    Get a DataFrame of matches between reference and comparison sources.

    Args:
        zipped_bp_df (pd.DataFrame): DataFrame containing breakpoint data from multiple sources
        sources_ref (list[str]): List of reference source names
        sources_comp (list[str]): List of comparison source names
        max_dist (int, optional): Maximum allowed distance between breakpoints. Defaults to 10.
        insert_size_threshold (float, optional): Minimum similarity threshold for insert size (0-1).
            Defaults to 0.7.

    Returns:
        pd.DataFrame: DataFrame containing matches between reference and comparison sources,
            with columns 'match_to' and 'match'
    """
    matches = get_ref_comp_matches(
        zipped_bp_df,
        sources_ref,
        sources_comp,
        max_dist,
        insert_size_threshold
    )
    matches = matches_pick_all(matches)
    return matches_to_df(matches)

def get_pairwise_matches(
    zipped_bp_df: pd.DataFrame,
    insert_size_threshold: float = 0.7,
    max_dist: int = 10
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
    unique_sources = list(zipped_bp_df['source'].unique())
    zipped_bp_df.loc[:, 'insert_size'] = get_insert_size(zipped_bp_df)

    # Generate all possible source combinations
    source_combinations = [
        f'{source1}_{source2}'
        for source1 in unique_sources
        for source2 in unique_sources
    ]

    # Generate combinations between different sources only
    source_combinations_diff = [
        f'{source1}_{source2}'
        for source1 in unique_sources
        for source2 in unique_sources
        if source1 != source2
    ]

    matched_pairs = []

    for chunk in chunker(zipped_bp_df):
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
            rel_insert_size_matrix_ref(chunk_df, chunk_df)
            > insert_size_threshold
        )

        # Create final match matrix
        match_matrix = (
            pos_match_matrix
            & insert_matrix
            # & is_min_dist_matrix
            & ~np.triu(np.ones(len(chunk_df), dtype=bool))
            # & sources_matrix
        )
        # Check if any matches exist in the matrix
        if np.any(match_matrix):
            # Iterate through matrix indices
            for i in range(len(chunk_df)):
                for j in range(len(chunk_df)):
                    # If match found, append source and event info for both breakpoints
                    if match_matrix[i, j]:
                        matched_pairs.append([
                            (chunk_df['source'].iloc[i], chunk_df['INFO_EVENT'].iloc[i]),
                            (chunk_df['source'].iloc[j], chunk_df['INFO_EVENT'].iloc[j])
                        ])

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
    matched_lists: list[list[tuple[str, str]]]
) -> pd.DataFrame:
    """Build a DataFrame containing matched breakpoints from different sources.

    Each row represents a connected component of breakpoints that were matched across
    different sources. The data from each source is prefixed with the source name.

    Args:
        bp_df_dict (dict[str, pd.DataFrame]): Dictionary mapping source names to their
            breakpoint DataFrames
        matched_lists (list[list[tuple[str, str]]]): List of lists, where each inner list
            contains tuples of (source, event_id) representing connected breakpoints

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
        source: bp_df[bp_df['INFO_EVENT'].isin(unique_matched_events)].add_prefix(source + '_')
        for source, bp_df in bp_df_dict.items()
    }

    rows = []
    for matched_list in matched_lists:
        per_source = []
        seen = set()

        for bp_source, bp_event in matched_list:
            if bp_source in seen:
                continue

            # Get one row for this source/event pair
            match_df = bp_df_dict_prefixes[bp_source].loc[
                bp_df_dict_prefixes[bp_source][f'{bp_source}_INFO_EVENT'] == bp_event
            ].iloc[[0]].reset_index(drop=True)

            per_source.append(match_df)
            seen.add(bp_source)

        # If only one source is present, skip this row
        # Since we are only interested in cross-source matches
        if len(per_source) < 2:
            continue

        # Join source-specific data side by side
        component_row = pd.concat(per_source, axis=1)
        rows.append(component_row)

    # Stack all component rows vertically
    return pd.concat(rows, ignore_index=True)


def get_connected_pairwise_matches_df(
    bp_df_dict: dict[str, pd.DataFrame],
    insert_size_threshold: float = 0.7,
    max_dist: int = 10
) -> pd.DataFrame:
    """Find all pairwise matches in the breakpoint dataframe dictionary
    and return a dataframe with the matches.

    Args:
        bp_df_dict (dict[str, pd.DataFrame]): Dictionary mapping source names
        to their breakpoint DataFrames
        insert_size_threshold (float, optional): Minimum similarity threshold
        for insert size. Defaults to 0.7.
        max_dist (int, optional): Maximum allowed distance between breakpoints. Defaults to 10.

    Returns:
        pd.DataFrame: DataFrame containing matched breakpoints from different sources side by side
    """
    # Combine all breakpoint dataframes into one with source labels
    zipped_bp_df = zip_bp_dfs(bp_df_dict)

    # Find all pairwise matches between breakpoints
    matched_pairs = get_pairwise_matches(
        zipped_bp_df,
        insert_size_threshold=insert_size_threshold,
        max_dist=max_dist
    )

    # Merge connected pairs into lists of related breakpoints
    matched_lists = merge_bp_pairs_dfs(matched_pairs)

    # Build final dataframe with matched breakpoints side by side
    matched_bp_df = build_matched_bp_df(bp_df_dict, matched_lists)

    return matched_bp_df
