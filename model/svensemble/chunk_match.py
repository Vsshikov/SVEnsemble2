"""
Contains functions used for matching breakpoints from different sets
Following a similar algorith with chunking and creating similarity matrices to the one used by Truvari
"""
import numpy as np
import pandas as pd

from . import breakpoints as bps
from . import breakends as bnds
from . import events as evts
from . import _caller_params as caller_params

def zip_bp_dfs(bp_df_dict: dict[str, pd.DataFrame]):
    """
    Concatenates multiple breakpoints dataframes into one, with a new column 'source' indicating the source of each breakpoint
    Sorts the resulting dataframe by chromosome, orientation, and positions of breakends
    """
    bp_df_dict_ = {source: bp_df.copy() for source, bp_df in bp_df_dict.items()}
    for name, bp_df in bp_df_dict_.items():
        bp_df.loc[:, 'source'] = name
    bp_df = pd.concat(bp_df_dict_.values())
    bp_df = bp_df.sort_values(by=['CHROM_bnd1', 'CHROM_bnd2', 'POS_bnd1', 'POS_bnd2'])
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
        row['POS_bnd1'] <= previous_row['POS_bnd2'] + chunk_size
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
            if breakpoint_row['POS_bnd1'] <= current_chunk[-1]['POS_bnd2'] + chunk_size:
                # Add to current chunk if within proximity threshold
                current_chunk.append(breakpoint_row)
            else:
                # Yield completed chunk and start a new one
                yield current_chunk
                current_chunk = []
        # Yield the last chunk if it exists
        if len(current_chunk) > 0:
            yield current_chunk

def dist_matrix_ref(ref_bps: pd.DataFrame, bps: pd.DataFrame, which_bnd: str) -> np.ndarray:
    """
    Calculate distance matrix between reference breakpoints and other breakpoints.
    
    This function computes the minimum distance between confidence intervals
    of breakpoints from reference data and breakpoints from other sources.
    The distance is 0 if intervals overlap, otherwise it's the minimum
    separation between the intervals.
    
    Args:
        ref_bps: DataFrame containing reference breakpoint data
        bps: DataFrame containing breakpoint data from other sources
        which_bnd: String specifying which breakend to analyze ('bnd1' or 'bnd2')
        
    Returns:
        2D numpy array where element [i,j] is the distance between
        reference breakpoint i and other breakpoint j
    """
    # Extract position and confidence interval data for reference breakpoints
    ref_bnd_pos = ref_bps[f'POS_{which_bnd}'].values
    ref_bnd_cipos_l = ref_bps[f'INFO_CIPOS_l_{which_bnd}'].values
    ref_bnd_cipos_r = ref_bps[f'INFO_CIPOS_r_{which_bnd}'].values
    
    # Extract position and confidence interval data for other breakpoints
    bnd_pos = bps[f'POS_{which_bnd}'].values
    bnd_cipos_l = bps[f'INFO_CIPOS_l_{which_bnd}'].values
    bnd_cipos_r = bps[f'INFO_CIPOS_r_{which_bnd}'].values
    
    # Initialize distance matrix with zeros (overlapping intervals)
    dists = np.zeros((len(ref_bps), len(bps)))
    
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
        insert_sizes.loc[bp_df['source'] == source] = insert_source_dispatch[caller_params.insert_seq_source[source]](bp_df[bp_df['source'] == source])
    return insert_sizes

def rel_insert_size_matrix_ref(ref_bps: pd.DataFrame, bps: pd.DataFrame, insert_size_margin: int = 10):
    """
    Calculate relative insert size matrix between reference breakpoints and other breakpoints.
    To use insert size, requires 'insert_size' column to be present in the dataframe
    """
    insert_sizes_ref = pd.Series(index=ref_bps.index)
    insert_sizes_comp = pd.Series(index=bps.index)
    for source in ref_bps['source'].unique():
        insert_sizes_ref.loc[ref_bps['source'] == source] = ref_bps[ref_bps['source'] == source]['insert_size']
    for source in bps['source'].unique():
        insert_sizes_comp.loc[bps['source'] == source] = bps[bps['source'] == source]['insert_size']
    
    # Create matrix where smaller insert sizes are divided by larger ones
    # Convert to numpy arrays for broadcasting
    ref_sizes = insert_sizes_ref.values[:, np.newaxis]  # Shape: (n_ref, 1)
    comp_sizes = insert_sizes_comp.values[np.newaxis, :]  # Shape: (1, n_comp)
    
    # Use broadcasting to create division matrices
    insert_matrix1 = np.divide(ref_sizes, comp_sizes, out=np.zeros_like(ref_sizes * comp_sizes, dtype=float), where=comp_sizes > 0)
    insert_matrix2 = np.divide(comp_sizes, ref_sizes, out=np.zeros_like(comp_sizes * ref_sizes, dtype=float), where=ref_sizes > 0)
    # For each element, ensure we divide smaller by larger
    insert_matrix = np.minimum(insert_matrix1, insert_matrix2)
    # If both insert sizes are near-zero (within insert_size_margin), set the value to 1
    insert_matrix = np.where(np.logical_and(abs(insert_matrix1) < insert_size_margin, abs(insert_matrix2) < insert_size_margin), 1, insert_matrix)
    
    return insert_matrix


def get_ref_comp_matches(zipped_bp_df: pd.DataFrame, sources_ref: list[str], sources_comp: list[str], max_dist: int = 10, insert_size_threshold: float = 0.7):
    matches = []
    # Process each chunk of breakpoints grouped by genomic region
    zipped_bp_df.loc[:, 'insert_size'] = get_insert_size(zipped_bp_df)
    for chunk in chunker(zipped_bp_df):
        # Skip chunks with only one breakpoint (no matching possible)
        if len(chunk) > 1:
            chunk_df = pd.DataFrame(chunk)
            
            # Skip chunks with only one data source (no cross-source matching)
            if len(set(chunk_df['source'])) == 1:
                continue
                
            # Separate breakpoints by data source
            match_to_df = chunk_df[chunk_df['source'].isin(sources_ref)]  # Reference dataset
            match_df = chunk_df[chunk_df['source'].isin(sources_comp)]    # Query dataset
            
            # Calculate distance matrices for both breakpoint ends
            bp1_matrix = dist_matrix_ref(match_to_df, match_df, 'bnd1')
            bp2_matrix = dist_matrix_ref(match_to_df, match_df, 'bnd2')
            
            # Create position match matrix: both breakpoints must be within max_dist
            pos_match_matrix = (bp1_matrix <= max_dist) & (bp2_matrix <= max_dist)
            
            # Calculate combined distance and find closest matches
            sum_matrix = bp1_matrix + bp2_matrix
            is_min_dist_matrix = sum_matrix == np.min(sum_matrix, axis=1, keepdims=True)
            
            # Calculate insert size similarity matrix (threshold: 90% similarity)
            insert_matrix = rel_insert_size_matrix_ref(
                match_to_df, 
                match_df
            ) > insert_size_threshold

            # Final match matrix: all conditions must be satisfied
            match_matrix = pos_match_matrix & is_min_dist_matrix & insert_matrix

            # Process matches if any are found
            if np.any(match_matrix):
                # Extract event identifiers for potential DataFrame creation
                comp_events = match_df['INFO_EVENT'].values
                ref_events = match_to_df['INFO_EVENT'].values
                
                # Optional: Create labeled DataFrame for detailed analysis
                match_matrix_df = pd.DataFrame(match_matrix, 
                                            index=ref_events, 
                                            columns=comp_events)
                # Increment total match counter
                matches.append(match_matrix_df.loc[match_matrix_df.any(axis=1), match_matrix_df.any(axis=0)])

    return matches

# The resulting match list may contain multiple matches for the same event
# For now we simply pick the first match
def matches_pick_single(matches: list[pd.DataFrame]) -> list[pd.DataFrame]:
    matches_single_only = []
    for match in matches:
        if match.shape[0] > 1 or match.shape[1] > 1:
            for i, match_row in match.iterrows():
                for j, match_status in match_row.items():
                    if match_status:
                        matches_single_only.append(pd.DataFrame({
                            j: match_status
                        }, index=[i]))
                        break
        else:
            matches_single_only.append(match)
    return matches_single_only

def matches_pick_all(matches: list[pd.DataFrame]) -> list[pd.DataFrame]:
    matches_all = []
    for match in matches:
        if match.shape[0] > 1 or match.shape[1] > 1:
            for i, match_row in match.iterrows():
                for j, match_status in match_row.items():
                    if match_status:
                        matches_all.append(pd.DataFrame({
                            j: match_status
                        }, index=[i]))
        else:
            matches_all.append(match)
    return matches_all

# We also need ot convert the matches to a dataframe
# With columns for each dataset
def matches_to_df(matches: list[pd.DataFrame]) -> pd.DataFrame:
    matches_for_df = {'match_to': [], 'match': []}
    for match in matches:
        try: 
            matches_for_df['match_to'].append(match.index[0])
            matches_for_df['match'].append(match.columns[0])
        except AttributeError:
            print(match)
            break
    return pd.DataFrame(matches_for_df)

def get_ref_comp_match_df(zipped_bp_df: pd.DataFrame, sources_ref: list[str], sources_comp: list[str], max_dist: int = 10, insert_size_threshold: float = 0.7):
    matches = get_ref_comp_matches(zipped_bp_df, sources_ref, sources_comp, max_dist, insert_size_threshold)
    matches = matches_pick_all(matches)
    return matches_to_df(matches)
