"""
Functions for handling breakpoint dataframes
Includes functions for working within format + converting from format
"""

import numpy as np
import pandas as pd
import pyranges as pr

from . import breakends as bnds
from . import events as evts
from . import utils
from . import _caller_params as caller_params

def get_insert_seq_len_from_svensemble2_len_col(bp_df: pd.DataFrame) -> pd.Series:
    """
    Extracts insert sequence length from the 'svensemble2_insert_size' column.
    """
    return bp_df['svensemble2_insert_size']

insert_source_dispatch = {
    'svlen': evts.get_insert_seq_len_from_svlen,
    'alt': bnds.get_insert_seq_len_from_alt,
    'svensemble2_col': get_insert_seq_len_from_svensemble2_len_col
}

def sort_bp_df(bp_df: pd.DataFrame, chrom_keys: dict) -> pd.DataFrame:
    """
    Sorts a breakpoint DataFrame by chromosome and position of both breakends.

    This function takes a DataFrame containing breakpoint records and sorts them first by
    chromosome and position of breakend 1, then by chromosome and position of breakend 2.
    The chromosome keys allow proper sorting of chromosome names that may not sort
    naturally (e.g. chr1, chr2, chr10).

    Args:
        bp_df: DataFrame containing breakpoint records to sort. Must have columns:
              CHROM_bnd1, POS_bnd1, CHROM_bnd2, POS_bnd2
        chrom_keys: Dictionary mapping chromosome names (str) to numeric sort keys (int).
                   For example: {'chr1': 1, 'chr2': 2, 'chrX': 23}

    Returns:
        pd.DataFrame: A new DataFrame containing the same records as bp_df but sorted by
                     chromosome and position of both breakends.
    """
    sort_columns = [
        'CHROM_bnd1', 'POS_bnd1',
        'CHROM_bnd2', 'POS_bnd2'
    ]
    return bp_df.sort_values(
        sort_columns,
        key=lambda x: utils.get_chrom_keys(x, chrom_keys=chrom_keys)
    )

def get_bnd_prs(bp_vcf_df, suffixes=None, retain_cols=None):
    """
    Extracts breakend coordinates from breakpoint VCF data.

    Creates PyRanges objects containing coordinates for each breakend pair.

    Args:
        bp_vcf_df: DataFrame with breakpoint VCF data.
                   By default contains only columns nessessary for PyRanges + ID column formed from INFO_EVENT
        suffixes: Tuple of suffixes for first/second breakends
        retain_cols: Additional columns to retain

    Returns:
        Tuple of (bnd1_coords, bnd2_coords) PyRanges objects
    """
    if suffixes is None:
        suffixes = ('_bnd1', '_bnd2')
    if retain_cols is None:
        retain_cols = []

    bnd1_coords = pd.DataFrame()
    bnd2_coords = pd.DataFrame()

    bnd1_suffix = suffixes[0]
    bnd2_suffix = suffixes[1]

    # Extract first breakend coordinates
    bnd1_coords.loc[:, 'Chromosome'] = bp_vcf_df[f'CHROM{bnd1_suffix}']
    bnd1_coords.loc[:, 'Start'] = (bp_vcf_df[f'POS{bnd1_suffix}'] -1 +
                                  bp_vcf_df[f'INFO_CIPOS_l{bnd1_suffix}']).astype(int)
    bnd1_coords.loc[:, 'End'] = (bp_vcf_df[f'POS{bnd1_suffix}'] +
                                bp_vcf_df[f'INFO_CIPOS_r{bnd1_suffix}'].astype(int))
    bnd1_coords.loc[:, 'ID'] = bp_vcf_df['INFO_EVENT']
    for col in retain_cols:
        bnd1_coords.loc[:, col] = bp_vcf_df[col]

    # Extract second breakend coordinates
    bnd2_coords.loc[:, 'Chromosome'] = bp_vcf_df[f'CHROM{bnd2_suffix}']
    bnd2_coords.loc[:, 'Start'] = (bp_vcf_df[f'POS{bnd2_suffix}'] - 1 +
                                  bp_vcf_df[f'INFO_CIPOS_l{bnd2_suffix}'].astype(int))
    bnd2_coords.loc[:, 'End'] = (bp_vcf_df[f'POS{bnd2_suffix}'] +
                                bp_vcf_df[f'INFO_CIPOS_r{bnd2_suffix}'].astype(int))
    bnd2_coords.loc[:, 'ID'] = bp_vcf_df['INFO_EVENT']
    for col in retain_cols:
        bnd2_coords.loc[:, col] = bp_vcf_df[col]

    return pr.PyRanges(bnd1_coords), pr.PyRanges(bnd2_coords)


def join_bps(bp_df1, bp_df2, retain_cols_df1=None, retain_cols_df2=None,
             report_overlap_frac=False, report_distances=False):
    """
    Joins breakpoints between two VCF DataFrames.

    Joins corresponding breakends and optionally reports overlap fractions
    and distances between matched breakpoints.

    Args:
        bp_df1, bp_df2: DataFrames to join
        retain_cols_df1, retain_cols_df2: Columns to retain from each DataFrame
        report_overlap_frac: Whether to calculate overlap fractions
        report_distances: Whether to calculate distances between breakpoints

    Returns:
        Tuple of joined DataFrames for each breakend
    """
    if retain_cols_df1 is None:
        retain_cols_df1 = []
    if retain_cols_df2 is None:
        retain_cols_df2 = []

    if report_distances:
        retain_cols_df1.extend(['POS_bnd1', 'POS_bnd2'])
        retain_cols_df2.extend(['POS_bnd1', 'POS_bnd2'])

    # Get PyRanges objects
    bnd1_pr1, bnd2_pr1 = get_bnd_prs(bp_df1, retain_cols=['bndtype'] + retain_cols_df1)
    bnd1_pr2, bnd2_pr2 = get_bnd_prs(bp_df2, retain_cols=['bndtype'] + retain_cols_df2)

    # Join breakends
    joined_bnd1 = bnd1_pr1.join(bnd1_pr2, how=None, report_overlap=True).df
    joined_bnd2 = bnd2_pr1.join(bnd2_pr2, how=None, report_overlap=True).df

    if report_overlap_frac:
        # Calculate overlap fractions
        joined_bnd1.loc[:, 'overlap_frac'] = np.maximum(
            joined_bnd1['Overlap'] / (joined_bnd1['End'] - joined_bnd1['Start']),
            joined_bnd1['Overlap'] / (joined_bnd1['End_b'] - joined_bnd1['Start_b'])
        )
        joined_bnd2.loc[:, 'overlap_frac'] = np.maximum(
            joined_bnd2['Overlap'] / (joined_bnd2['End'] - joined_bnd2['Start']),
            joined_bnd2['Overlap'] / (joined_bnd2['End_b'] - joined_bnd2['Start_b'])
        )

    if report_distances:
        joined_bnd1.loc[:, 'dist_to_joined'] = np.abs(joined_bnd1['POS_bnd1'] - joined_bnd1['POS_bnd1_b'])
        joined_bnd2.loc[:, 'dist_to_joined'] = np.abs(joined_bnd2['POS_bnd2'] - joined_bnd2['POS_bnd2_b'])

    return joined_bnd1, joined_bnd2

def find_matching_breakpoint_ids(vcf_df1, vcf_df2, overlap_cutoff=0.8):
    """
    Find IDs of matching breakpoints between two VCF dataframes.

    Args:
        vcf_df1, vcf_df2: VCF dataframes in breakpoint notation with CIPOS fields
    Returns:
        Array of IDs from vcf_df1 that have matching breakpoints in vcf_df2
    """
    joined_bnd1, joined_bnd2 = join_bps(vcf_df1, vcf_df2, report_overlap_frac=True)

    # Get IDs with >80% overlap and matching bndtypes
    bnd1_ids = joined_bnd1[
        (joined_bnd1['overlap_frac'] > overlap_cutoff) &\
        (joined_bnd1['bndtype'] == joined_bnd1['bndtype_b'])
    ]['ID'].unique()
    bnd2_ids = joined_bnd2[
        (joined_bnd2['overlap_frac'] > overlap_cutoff) &\
        (joined_bnd2['bndtype'] == joined_bnd2['bndtype_b'])
    ]['ID'].unique()

    # Return IDs matching in both breakends
    return bnd1_ids[np.isin(bnd1_ids, bnd2_ids)]

def get_bp_chroms_orientations(bp_df: pd.DataFrame):
    if 'ALT_bnd1' not in bp_df.columns or 'ALT_bnd2' not in bp_df.columns:
        raise ValueError("ALT_bnd1 and ALT_bnd2 columns must be present in the dataframe")
    return bp_df['CHROM_bnd1'] + '_' + bp_df['CHROM_bnd2'] + '_' + bnds.process_breakend_alt(bp_df['ALT_bnd1'])['relative_pos'] + bnds.process_breakend_alt(bp_df['ALT_bnd2'])['relative_pos']

def trim_cipos(bp_df: pd.DataFrame, by: int | None | tuple[int, int] = None, up_to: int | None | tuple[int, int] = None):
    """
    Trims the confidence intervals of the breakpoints in the dataframe.

    Args:
        bp_df: DataFrame containing breakpoints.
        by: If int, trims the confidence intervals by the given number of bases on both sides.
            If tuple, trims the confidence intervals by the given number of bases on the left and right sides respectively.
        up_to: Maximum length of the confidence intervals. If any CIPOS is greater than this, it is set to this value. Applied after first trimming step if present

    Returns:
        DataFrame with trimmed confidence intervals.
    """
    bp_df_ = bp_df.copy()

    if by is None:
        by = (0, 0)
    elif isinstance(by, int):
        by = (by, by)
    if up_to is None:
        up_to = (np.inf, np.inf)
    elif isinstance(up_to, int):
        up_to = (up_to, up_to)

    for bnd in ['bnd1', 'bnd2']:
        bp_df_.loc[:, f'INFO_CIPOS_l_{bnd}'] = np.clip(bp_df_[f'INFO_CIPOS_l_{bnd}'] + by[0], -up_to[0], 0)
        bp_df_.loc[:, f'INFO_CIPOS_r_{bnd}'] = np.clip(bp_df_[f'INFO_CIPOS_r_{bnd}'] - by[1], 0, up_to[1])
    return bp_df

def get_insert_size(bp_df: pd.DataFrame, source: str | None = None) -> pd.Series:
    """
    Calculates insert sequence length from ALT or SVLEN column
    Requires 'source' column to be present in the dataframe or source to be provided
    """
    bp_df_ = bp_df.copy()

    insert_sizes = pd.Series(index=bp_df.index)

    if isinstance(source, str):
        bp_df_.loc[:, 'source'] = source

    for source in bp_df_['source'].unique():
        insert_sizes.loc[bp_df_['source'] == source] = (
            insert_source_dispatch[caller_params.caller_param_dict[source]['svlen_source']](
                bp_df_[bp_df_['source'] == source]
            )
        )
    insert_sizes.loc[insert_sizes < 0] = 0
    insert_sizes = insert_sizes.fillna(0)  # Convert NAs to 0
    insert_sizes = insert_sizes.astype(int)
    return insert_sizes


def annotate_svtypes(bps_df, source):

    ins_lens = get_insert_size(bps_df, source=source)
    bnd1 = bnds.process_breakend_alt(bps_df['ALT_bnd1'])
    bnd2 = bnds.process_breakend_alt(bps_df['ALT_bnd2'])

    same_chrom_mask = bps_df['CHROM_bnd1'] == bps_df['CHROM_bnd2']
    del_ins_bnds_mask = (bnd1['relative_pos'] == 'N[') & (bnd2['relative_pos'] == ']N')
    dup_bnds_mask = (bnd1['relative_pos'] == ']N') & (bnd2['relative_pos'] == 'N[')
    inv_bnds_mask = (
        ((bnd1['relative_pos'] == 'N]') & (bnd2['relative_pos'] == 'N]')) |
        ((bnd1['relative_pos'] == '[N') & (bnd2['relative_pos'] == '[N'))
    )
    pos2_pos1_more_than_lens_mask = (bps_df['POS_bnd2'] - bps_df['POS_bnd1']) > ins_lens

    del_mask = same_chrom_mask & del_ins_bnds_mask & pos2_pos1_more_than_lens_mask
    ins_mask = same_chrom_mask & del_ins_bnds_mask & ~pos2_pos1_more_than_lens_mask
    dup_mask = same_chrom_mask & dup_bnds_mask
    inv_mask = same_chrom_mask & inv_bnds_mask

    svtypes = np.full(len(bps_df), np.nan, dtype=object)

    svtype_dict = {
        'DEL': del_mask,
        'INS': ins_mask,
        'DUP': dup_mask,
        'INV': inv_mask,
    }

    for svtype, svtype_mask in svtype_dict.items():
        svtypes[svtype_mask] = svtype

    svtypes = np.where(pd.isna(svtypes), 'OTHER', svtypes)

    return svtypes
