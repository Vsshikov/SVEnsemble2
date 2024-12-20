import numpy as np
import pandas as pd
import pyranges as pr
import re
from typing import List

from . import utils

def check_sv_format(vcf_df: pd.DataFrame) -> str | None:
    """
    Checks the format of a VCF DataFrame to determine if it contains breakend or breakpoint notation.

    Returns "bnd" if breakend notation, "bp" if breakpoint notation, "event" if event notation, or None if neither.
    """
    bnd_reqs = ['CHROM', 'POS', 'ALT', 'INFO_EVENT', 'INFO_MATEID']
    bnd_reqs_missing = []
    bnd_alt_regex = re.compile(r'(^[ATGCNatgcn\.]+[\[\]]$)|(^[\[\]][ATGCNatgcn\.]+$)')
    
    bp_reqs = ['CHROM_bnd1', 'POS_bnd1', 'ALT_bnd1', 'CHROM_bnd2', 'POS_bnd2', 'ALT_bnd2']
    bp_reqs_missing = []

    event_reqs = ['CHROM', 'POS', 'ALT']
    # event_regex = re.compile(r'[DEL|DUP|INV|INS]', re.IGNORECASE)
    event_reqs_missing = []

    for bnd_req in bnd_reqs:
        if bnd_req not in vcf_df.columns:
            bnd_reqs_missing.append(bnd_req)
    if len(bnd_reqs_missing) == 0:
        if vcf_df['ALT'].str.contains(bnd_alt_regex).all():
            return 'bnd'
    
    for bp_req in bp_reqs:
        if bp_req not in vcf_df.columns:
            bp_reqs_missing.append(bp_req)
    if len(bp_reqs_missing) == 0:
        return 'bp'
    
    for event_req in event_reqs:
        if event_req not in vcf_df.columns:
            event_reqs_missing.append(event_req)
    if len(event_reqs_missing) == 0:
        return 'event'
    
    raise ValueError(f"VCF DataFrame does not contain following required columns for breakend or breakpoint notation:\
                     \nFor bnd: {bnd_reqs_missing}\nFor bp: {bp_reqs_missing}\nFor event: {event_reqs_missing}")


def bnd_to_bp(bnd_df: pd.DataFrame, force_diff_cols: List[str] | None = None) -> pd.DataFrame:
    """
    Converts breakend notation to breakpoint notation.
    
    Takes a DataFrame with breakend records and converts them to breakpoint format
    by combining corresponding breakends and renaming columns appropriately.
    
    Args:
        bnd_df: DataFrame with breakend records
        force_diff_cols: Optional list of columns to treat as different between breakends
        
    Returns:
        DataFrame in breakpoint notation
    """
    if force_diff_cols is None:
        force_diff_cols = []

    chrom_key_dict = utils.generate_chrom_key_dict(bnd_df['CHROM'])
    print(chrom_key_dict)

    # Split into first/second breakends
    grsizes = bnd_df.groupby('INFO_EVENT').transform('size')
    sorted_bnds = bnd_df.copy()[grsizes == 2].sort_values(['INFO_EVENT', 'CHROM', 'POS'], key=lambda x: utils.get_chrom_keys(x, chrom_keys=chrom_key_dict))
    sorted_bnds = sorted_bnds.reset_index(drop=True)
    
    bnd1 = sorted_bnds.iloc[::2].reset_index(drop=True)
    bnd2 = sorted_bnds.iloc[1::2].reset_index(drop=True)
    
    # Identify columns that are same/different between pairs
    same_cols_mask = (bnd1 == bnd2).all()
    force_diff_cols_mask = bnd1.columns.isin(force_diff_cols)
    # If force_diff_cols is provided, exclude those columns from being considered as 'same' columns
    same_cols_mask = same_cols_mask & (~force_diff_cols_mask)
    same_cols = bnd1.columns[same_cols_mask]
    diff_cols = bnd1.columns[~same_cols_mask]
    
    # Rename columns to indicate which breakend they came from
    bnd1_col_rename = {
        col: f'{col}_bnd1' for col in bnd1.columns 
        if (col not in same_cols) and (not col.endswith('_bnd1'))
    }
    bnd2_col_rename = {
        col: f'{col}_bnd2' for col in bnd2.columns
        if (col not in same_cols) and (not col.endswith('_bnd2'))
    }
    
    bnd1 = bnd1.rename(columns=bnd1_col_rename)
    bnd2 = bnd2.rename(columns=bnd2_col_rename)
    
    diff_cols_bnd2 = [f'{col}_bnd2' for col in diff_cols]
    
    # Combine into breakpoint records
    breakpoints = pd.concat([bnd1, bnd2.loc[:, diff_cols_bnd2]], axis=1)
    
    return breakpoints


def sort_bnd_df(bnd_df: pd.DataFrame, chrom_keys: dict) -> pd.DataFrame:
    return bnd_df.sort_values(['CHROM_bnd1', 'POS_bnd1', 'CHROM_bnd2', 'POS_bnd2'], key=lambda x: utils.get_chrom_keys(x, chrom_keys=chrom_keys))


def convert_del_to_bnd(del_vcf_df: pd.DataFrame, add_encoding: bool = False) -> pd.DataFrame:
    """
    Converts deletion records to breakend notation.
    
    Each deletion is converted to two breakends:
    - bp1: CHROM:POS N[CHR2:END[ (encoding=1)
    - bp2: CHR2:END ]CHROM:POS]N (encoding=2)
    
    Args:
        del_vcf_df: DataFrame with deletion records
        add_encoding: Whether to add position encoding
        
    Returns:
        DataFrame containing breakend notation entries with optional bnd_pos_encoding
    """
    # Create first breakend
    bp1 = del_vcf_df.copy()
    bp1.loc[:, 'ID'] = bp1['ID'] + '_bp1'
    bp1.loc[:, 'CHROM'] = bp1['CHROM']
    bp1.loc[:, 'POS'] = bp1['POS'] - 1
    bp1.loc[:, 'INFO_MATEID'] = bp1['INFO_EVENT'] + '_bp2'
    bp1.loc[:, 'ALT'] = 'N[' + bp1['INFO_CHR2'].astype(str) + ':' + (bp1['INFO_END'].astype(int) + 1).astype(str) + '['
    if add_encoding:
        bp1.loc[:, 'bnd_pos_encoding'] = 1

    # Create second breakend
    bp2 = del_vcf_df.copy()
    bp2.loc[:, 'ID'] = bp2['ID'] + '_bp2'
    bp2.loc[:, 'CHROM'] = bp2['INFO_CHR2']
    bp2.loc[:, 'POS'] = bp2['INFO_END'].astype(int) + 1
    bp2.loc[:, 'INFO_MATEID'] = bp2['INFO_EVENT'] + '_bp1'
    bp2.loc[:, 'ALT'] = ']' + bp2['CHROM'].astype(str) + ':' + (bp2['POS'].astype(int) - 1).astype(str) + ']N'
    if add_encoding:
        bp2.loc[:, 'bnd_pos_encoding'] = 2

    return pd.concat([bp1, bp2], axis=0)


def convert_inv_to_bnd(inv_vcf_df: pd.DataFrame, add_encoding: bool = False) -> pd.DataFrame:
    """
    Converts inversion records to breakend notation.
    
    Each inversion is converted to four breakends:
    - bp1: CHROM:POS-1 N]CHR2:END] (encoding=3)
    - bp2: CHROM:POS [CHR2:END+1[N (encoding=4) 
    - bp3: CHR2:END N]CHROM:POS-1] (encoding=3)
    - bp4: CHR2:END+1 [CHROM:POS[N (encoding=4)
    
    Args:
        inv_vcf_df: DataFrame with inversion records
        add_encoding: Whether to add position encoding
        
    Returns:
        DataFrame containing breakend notation entries with optional bnd_pos_encoding
    """
    # Create first breakend
    bp1 = inv_vcf_df.copy()
    bp1.loc[:, 'ID'] = bp1['ID'] + '_bp1'
    bp1.loc[:, 'CHROM'] = bp1['CHROM'] 
    bp1.loc[:, 'POS'] = bp1['POS'] - 1
    bp1.loc[:, 'INFO_EVENT'] = bp1['INFO_EVENT'] + '_e1'
    bp1.loc[:, 'INFO_MATEID'] = bp1['INFO_EVENT'] + '_bp3'
    bp1.loc[:, 'ALT'] = 'N]' + bp1['INFO_CHR2'].astype(str) + ':' + bp1['INFO_END'].astype(str) + ']'
    if add_encoding:
        bp1.loc[:, 'bnd_pos_encoding'] = 3

    # Create second breakend
    bp2 = inv_vcf_df.copy()
    bp2.loc[:, 'ID'] = bp2['ID'] + '_bp2'
    bp2.loc[:, 'CHROM'] = bp2['CHROM']
    bp2.loc[:, 'POS'] = bp2['POS']
    bp2.loc[:, 'INFO_EVENT'] = bp2['INFO_EVENT'] + '_e2'
    bp2.loc[:, 'INFO_MATEID'] = bp2['INFO_EVENT'] + '_bp4'
    bp2.loc[:, 'ALT'] = '[' + bp2['INFO_CHR2'].astype(str) + ':' + (bp2['INFO_END'].astype(int) + 1).astype(str) + '[N'
    if add_encoding:
        bp2.loc[:, 'bnd_pos_encoding'] = 4

    # Create third breakend
    bp3 = inv_vcf_df.copy()
    bp3.loc[:, 'ID'] = bp3['ID'] + '_bp3'
    bp3.loc[:, 'CHROM'] = bp3['INFO_CHR2']
    bp3.loc[:, 'POS'] = bp3['INFO_END']
    bp3.loc[:, 'INFO_EVENT'] = bp3['INFO_EVENT'] + '_e1'
    bp3.loc[:, 'INFO_MATEID'] = bp3['INFO_EVENT'] + '_bp1'
    bp3.loc[:, 'ALT'] = 'N]' + bp3['CHROM'].astype(str) + ':' + (bp3['POS'].astype(int) - 1).astype(str) + ']'
    if add_encoding:
        bp3.loc[:, 'bnd_pos_encoding'] = 3

    # Create fourth breakend
    bp4 = inv_vcf_df.copy()
    bp4.loc[:, 'ID'] = bp4['ID'] + '_bp4'
    bp4.loc[:, 'CHROM'] = bp4['INFO_CHR2']
    bp4.loc[:, 'POS'] = bp4['INFO_END'].astype(int) + 1
    bp4.loc[:, 'INFO_EVENT'] = bp4['INFO_EVENT'] + '_e2'
    bp4.loc[:, 'INFO_MATEID'] = bp4['INFO_EVENT'] + '_bp2'
    bp4.loc[:, 'ALT'] = '[' + bp4['CHROM'].astype(str) + ':' + bp4['POS'].astype(str) + '[N'
    if add_encoding:
        bp4.loc[:, 'bnd_pos_encoding'] = 4

    return pd.concat([bp1, bp2, bp3, bp4], axis=0)

def convert_dup_to_bnd(dup_vcf_df: pd.DataFrame, add_encoding: bool = False) -> pd.DataFrame:
    """
    Converts duplication records to breakend notation.
    
    Each duplication is converted to two breakends:
    - bp1: CHR2:END N[CHROM:POS[ (encoding=1)
    - bp2: CHROM:POS ]CHR2:END]N (encoding=2)
    
    Args:
        dup_vcf_df: DataFrame with duplication records
        add_encoding: Whether to add position encoding
        
    Returns:
        DataFrame containing breakend notation entries with optional bnd_pos_encoding
    """
    # Create first breakend
    bp1 = dup_vcf_df.copy()
    bp1.loc[:, 'ID'] = bp1['ID'] + '_bp1'
    bp1.loc[:, 'CHROM'] = bp1['INFO_CHR2']
    bp1.loc[:, 'POS'] = bp1['INFO_END']
    bp1.loc[:, 'INFO_MATEID'] = bp1['INFO_EVENT'] + '_bp2'
    bp1.loc[:, 'ALT'] = 'N[' + bp1['CHROM'].astype(str) + ':' + bp1['POS'].astype(str) + '['
    if add_encoding:
        bp1.loc[:, 'bnd_pos_encoding'] = 1

    # Create second breakend
    bp2 = dup_vcf_df.copy()
    bp2.loc[:, 'ID'] = bp2['ID'] + '_bp2'
    bp2.loc[:, 'CHROM'] = bp2['CHROM']
    bp2.loc[:, 'POS'] = bp2['POS']
    bp2.loc[:, 'INFO_MATEID'] = bp2['INFO_EVENT'] + '_bp1'
    bp2.loc[:, 'ALT'] = ']' + bp2['INFO_CHR2'].astype(str) + ':' + bp2['INFO_END'].astype(str) + ']N'
    if add_encoding:
        bp2.loc[:, 'bnd_pos_encoding'] = 2

    return pd.concat([bp1, bp2], axis=0)

def convert_ins_to_bnd(ins_vcf_df: pd.DataFrame, add_encoding: bool = False) -> pd.DataFrame:
    """
    Converts insertion records to breakend notation.
    
    Each insertion is converted to two breakends:
    - bp1: CHROM:POS N[CHROM:POS[ (encoding=1)
    - bp2: CHROM:POS ]CHROM:POS]N (encoding=2)
    
    Args:
        ins_vcf_df: DataFrame with insertion records
        add_encoding: Whether to add position encoding
        
    Returns:
        DataFrame containing breakend notation entries with optional bnd_pos_encoding
    """
    # Create first breakend
    bp1 = ins_vcf_df.copy()
    bp1.loc[:, 'ID'] = bp1['ID'] + '_bp1'
    bp1.loc[:, 'CHROM'] = bp1['CHROM']
    bp1.loc[:, 'POS'] = bp1['POS'] - 1
    bp1.loc[:, 'INFO_MATEID'] = bp1['INFO_EVENT'] + '_bp2'
    bp1.loc[:, 'ALT'] = 'N[' + bp1['INFO_CHR2'].astype(str) + ':' + (bp1['INFO_END'].astype(int) + 1).astype(str) + '['
    if add_encoding:
        bp1.loc[:, 'bnd_pos_encoding'] = 1

    # Create second breakend
    bp2 = ins_vcf_df.copy()
    bp2.loc[:, 'ID'] = bp2['ID'] + '_bp2'
    bp2.loc[:, 'CHROM'] = bp2['INFO_CHR2']
    bp2.loc[:, 'POS'] = bp2['INFO_END'].astype(int) + 1
    bp2.loc[:, 'INFO_MATEID'] = bp2['INFO_EVENT'] + '_bp1'
    bp2.loc[:, 'ALT'] = ']' + bp2['CHROM'].astype(str) + ':' + (bp2['POS'].astype(int) - 1).astype(str) + ']N'
    if add_encoding:
        bp2.loc[:, 'bnd_pos_encoding'] = 2

    return pd.concat([bp1, bp2], axis=0)

def event_to_bnd(vcf_df: pd.DataFrame, add_encoding: bool = False) -> pd.DataFrame:
    """
    Converts structural variant events to breakend notation.
    
    Handles DEL, DUP, INV and INS event types by converting each to appropriate
    breakend representation.
    
    Args:
        vcf_df: DataFrame with SV events
        add_encoding: Whether to add position encoding
        
    Returns:
        DataFrame with breakend records
    """
    vcf_df_copy = vcf_df.copy()
    vcf_df_copy = vcf_df_copy[~vcf_df_copy['CHROM'].isna()]
    vcf_df_copy = vcf_df_copy.sort_values(['CHROM', 'POS'], key=utils.get_chrom_keys)
    
    # Split by SV type
    del_df = vcf_df_copy[vcf_df_copy['INFO_SVTYPE'] == 'DEL']
    dup_df = vcf_df_copy[vcf_df_copy['INFO_SVTYPE'] == 'DUP']
    inv_df = vcf_df_copy[vcf_df_copy['INFO_SVTYPE'] == 'INV']
    ins_df = vcf_df_copy[vcf_df_copy['INFO_SVTYPE'] == 'INS']

    # Convert each type
    bnd_dfs = []
    
    if len(del_df) > 0:
        bnd_dfs.append(convert_del_to_bnd(del_df, add_encoding))
    
    if len(dup_df) > 0:
        bnd_dfs.append(convert_dup_to_bnd(dup_df, add_encoding))
        
    if len(inv_df) > 0:
        bnd_dfs.append(convert_inv_to_bnd(inv_df, add_encoding))
        
    if len(ins_df) > 0:
        bnd_dfs.append(convert_ins_to_bnd(ins_df, add_encoding))

    if not bnd_dfs:
        return pd.DataFrame()
        
    return pd.concat(bnd_dfs, axis=0)

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
                                  bp_vcf_df[f'INFO_CIPOS_l{bnd1_suffix}'])
    bnd1_coords.loc[:, 'End'] = (bp_vcf_df[f'POS{bnd1_suffix}'] + 
                                bp_vcf_df[f'INFO_CIPOS_r{bnd1_suffix}'])
    bnd1_coords.loc[:, 'ID'] = bp_vcf_df['INFO_EVENT']
    for col in retain_cols:
        bnd1_coords.loc[:, col] = bp_vcf_df[col]

    # Extract second breakend coordinates
    bnd2_coords.loc[:, 'Chromosome'] = bp_vcf_df[f'CHROM{bnd2_suffix}']
    bnd2_coords.loc[:, 'Start'] = (bp_vcf_df[f'POS{bnd2_suffix}'] - 1 + 
                                  bp_vcf_df[f'INFO_CIPOS_l{bnd2_suffix}'])
    bnd2_coords.loc[:, 'End'] = (bp_vcf_df[f'POS{bnd2_suffix}'] + 
                                bp_vcf_df[f'INFO_CIPOS_r{bnd2_suffix}'])
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

def encode_bnd_alts(alt_col):
    """Encode breakend ALT fields into numeric categories based on their pattern.
    
    Args:
        alt_col: pandas Series containing ALT fields from VCF
        
    Returns:
        pandas Series with numeric encodings:
        - 1: N[PARTNER[ (partner extending after breakpoint)
        - 2: ]PARTNER]N (partner extending before breakpoint) 
        - 3: N]PARTNER] (reverse complement of partner extending after breakpoint)
        - 4: [PARTNER[N (reverse complement of partner extending before breakpoint)
        - 0: No match
    """
    partner_pattern_right = r'\[\.?c?h?r?[0-9XY]{0,2}\:?[0-9]{0,}\['
    partner_pattern_left = r'\]\.?c?h?r?[0-9XY]{0,2}\:?[0-9]{0,}\]'
    nucl_pattern = r'[ATGCNatgcn]+'

    bnd_regex = re.compile(rf'({nucl_pattern}{partner_pattern_right})|({partner_pattern_left}{nucl_pattern})|({nucl_pattern}{partner_pattern_left})|({partner_pattern_right}{nucl_pattern})')

    encoding_df = alt_col.str.extract(bnd_regex)
    encoding_df.columns = [1, 2, 3, 4]
    encoding_df[~encoding_df.isna()] = True
    
    encodings = encoding_df.idxmax(axis=1)
    encodings[encodings.isna()] = 0
    encodings = encodings.astype(int)
    
    return encodings


def annotate_deletions(bps_df):
    """
    Annotate structural variants as deletions based on breakpoint pairs.
    
    This function identifies deletion variants by comparing the length of the deleted
    reference sequence to any inserted sequence at the breakpoints. A variant is classified
    as a deletion if:
    1. It is an intrachromosomal breakpoint pair with forward-forward orientation
    2. The length of deleted reference sequence exceeds the length of any inserted sequences
    
    Args:
        bps_df: DataFrame containing breakpoint pairs with columns:
            - CHROM_bnd1, CHROM_bnd2: Chromosomes for each breakpoint
            - POS_bnd1, POS_bnd2: Positions for each breakpoint  
            - ALT_bnd1, ALT_bnd2: ALT fields containing inserted sequences
            - bndtype: Orientation encoding of the breakpoint pair
    
    Returns:
        pd.DataFrame: DataFrame containing variant type annotations and lengths:
            - INFO_SVTYPE: 'DEL' for deletions, 'not_implemented' for other variant types
            - INFO_SVLEN: Length of the structural variant
    """
    # Initialize all variants as not implemented
    variant_types = pd.Series('not_implemented', index=bps_df.index)
    
    # Calculate length of deleted reference sequence between breakpoints
    deletion_length = bps_df['POS_bnd2'] - bps_df['POS_bnd1'] - 1

    # Filter for intrachromosomal breakpoint pairs with forward-forward orientation
    del_or_ins_mask = (
        (bps_df['bndtype'] == '12') & 
        (bps_df['CHROM_bnd1'] == bps_df['CHROM_bnd2'])
    )
    
    # Extract inserted sequences from breakpoint ALTs using regex
    # Format is: N[chr:pos[ for bnd1 and ]chr:pos]N for bnd2
    # where N is the inserted sequence
    inserted_seq_bnd1 = bps_df['ALT_bnd1'].str.extract(r'([ATGCNatgcn]+)\[')
    inserted_seq_bnd2 = bps_df['ALT_bnd2'].str.extract(r'(\][ATGCNatgcn]+$)')
    
    # Calculate total insertion length (-1 to account for anchor base)
    insertion_length_bnd1 = inserted_seq_bnd1[0].str.len() - 1  
    insertion_length_bnd2 = inserted_seq_bnd2[0].str.len() - 1
    total_insertion_length = insertion_length_bnd1 + insertion_length_bnd2


    # Mark as deletion where deletion length exceeds insertion length
    deletion_mask = del_or_ins_mask & (
        bps_df.index.isin(
            bps_df[deletion_length > total_insertion_length].index
        )
    )
    info_svlen = np.maximum(deletion_length, total_insertion_length)
    
    # Set deletion variants
    variant_types[deletion_mask] = 'DEL'
    
    # Create output dataframe
    result_df = pd.DataFrame({
        'INFO_SVTYPE': variant_types,
        'INFO_SVLEN': info_svlen
    })
    
    return result_df
