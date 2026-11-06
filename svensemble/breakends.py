"""
Functions for handling breakend dataframes
Includes functions for working within format + converting from format
"""

import numpy as np
import pandas as pd
import re
from typing import List

from . import utils
from ._constants import BND1, BND2, SAMPLE_ID_COL

_BND_PATTERN = re.compile(
        r'(?P<inserted_seq1>[ACGTN]+)?'  # Optional inserted sequence before breakend
        r'(?P<orientation1>\]|\[)?'       # Breakend orientation (] or [)
        r'(?P<chr>[^:]+)?'               # Mate chromosome
        r'(?P<sep>:|\.)+'                # Separator (: for mate, . for no mate)
        r'(?P<pos>\d+)?'                 # Mate position
        r'(?P<orientation2>\]|\[)?'       # Closing orientation bracket
        r'(?P<inserted_seq2>[ACGTN]+)?'  # Optional inserted sequence after breakend
    )

_BND_ALT_REGRESS = re.compile(r'(^[ATGCNatgcn\.]+[\[\]]$)|(^[\[\]][ATGCNatgcn\.]+$)')

def check_sv_format(vcf_df: pd.DataFrame) -> str | None:
    """
    Checks the format of a VCF DataFrame to determine if it contains breakend or breakpoint notation.

    Returns "bnd" if breakend notation, "bp" if breakpoint notation, "event" if event notation, or None if neither.
    """
    bnd_reqs = ['CHROM', 'POS', 'ALT', 'INFO_EVENT', 'INFO_MATEID']
    bnd_reqs_missing = []
    bnd_alt_regex = _BND_ALT_REGRESS

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


def bnd_to_bp(
    bnd_df: pd.DataFrame,
    force_diff_cols: List[str] | None = None
) -> pd.DataFrame:
    """
    Convert breakend notation to breakpoint notation.

    Takes a DataFrame with breakend records and converts them to breakpoint format
    by combining corresponding breakends and renaming columns appropriately.

    Args:
        bnd_df: DataFrame containing breakend records. Must have columns:
            - CHROM: Chromosome name
            - POS: Position
            - INFO_EVENT: Event ID linking breakend pairs
            - SAMPLE_ID_COL (optional): with it, breakends pair within one
              sample, so a joint-called record yields one breakpoint per sample
            - Other columns will be preserved and renamed with _bnd1/_bnd2 suffixes
        force_diff_cols: Optional list of column names to treat as different between
            breakend pairs, even if values are identical. These columns will always
            get _bnd1/_bnd2 suffixes regardless of whether their values match.

    Returns:
        pd.DataFrame: DataFrame in breakpoint notation with:
            - Original columns split into _bnd1/_bnd2 variants if values differ
                between breakend pairs or if column is in force_diff_cols
            - Columns identical between pairs preserved without suffix
            - Rows sorted by chromosome and position
    """
    if force_diff_cols is None:
        _force_diff_cols = []
    else:
        _force_diff_cols = force_diff_cols.copy()
    # CHROM, POS fields must always be considered different
    _force_diff_cols.extend(['CHROM', 'POS'])

    # Generate chromosome keys for sorting
    chrom_key_dict = utils.generate_chrom_key_dict(bnd_df['CHROM'])

    # Split into first/second breakends; a pair shares its event and sample
    pair_keys = ['INFO_EVENT'] + ([SAMPLE_ID_COL] if SAMPLE_ID_COL in bnd_df.columns else [])
    grsizes = bnd_df.groupby(pair_keys)['INFO_EVENT'].transform('size')
    sorted_bnds = (bnd_df[grsizes == 2]
                   .sort_values(pair_keys + ['CHROM', 'POS'],
                              key=lambda x: utils.get_chrom_keys(x, chrom_keys=chrom_key_dict))
                   .reset_index(drop=True))

    bnd1 = sorted_bnds.iloc[::2].reset_index(drop=True)
    bnd2 = sorted_bnds.iloc[1::2].reset_index(drop=True)

    # Identify columns that are same/different between pairs
    same_cols_mask = ((bnd1 == bnd2) | (bnd1.isna() & bnd2.isna())).all()

    # If force_diff_cols is provided, exclude those columns from being considered as 'same' columns
    force_diff_cols_mask = bnd1.columns.isin(_force_diff_cols)
    same_cols_mask = same_cols_mask & (~force_diff_cols_mask)

    same_cols = bnd1.columns[same_cols_mask]
    diff_cols = bnd1.columns[~same_cols_mask]
    if SAMPLE_ID_COL in bnd1.columns and len(bnd1) and SAMPLE_ID_COL not in same_cols:
        raise AssertionError(f"Breakend pairs disagree on {SAMPLE_ID_COL}")

    # Rename columns to indicate which breakend they came from
    bnd1_col_rename = {
        col: f'{col}_{BND1}' for col in bnd1.columns
        if (col not in same_cols) and (not col.endswith(f'_{BND1}'))
    }
    bnd2_col_rename = {
        col: f'{col}_{BND2}' for col in bnd2.columns
        if (col not in same_cols) and (not col.endswith(f'_{BND2}'))
    }

    bnd1 = bnd1.rename(columns=bnd1_col_rename)
    bnd2 = bnd2.rename(columns=bnd2_col_rename)

    diff_cols_bnd2 = [f'{col}_{BND2}' for col in diff_cols]

    # Combine into breakpoint records
    breakpoints = pd.concat([bnd1, bnd2.loc[:, diff_cols_bnd2]], axis=1)

    return breakpoints

def sort_bnd_df(bnd_df: pd.DataFrame, chrom_keys: dict) -> pd.DataFrame:
    """
    Sorts a breakend DataFrame by chromosome and position.

    This function takes a DataFrame containing breakend records and sorts them first by
    chromosome (using the provided chromosome key mapping) and then by position within
    each chromosome. The chromosome keys allow proper sorting of chromosome names that
    may not sort naturally (e.g. chr1, chr2, chr10).

    Args:
        bnd_df: DataFrame containing breakend records to sort. Must have 'CHROM' and 'POS' columns.
        chrom_keys: Dictionary mapping chromosome names (str) to numeric sort keys (int).
                   For example: {'chr1': 1, 'chr2': 2, 'chrX': 23}

    Returns:
        pd.DataFrame: A new DataFrame containing the same records as bnd_df but sorted by
                     chromosome and position.
    """
    return bnd_df.sort_values(
        ['CHROM', 'POS'],
        key=lambda x: utils.get_chrom_keys(x, chrom_keys=chrom_keys)
    )

def process_breakend_alt(alt_series: pd.Series) -> pd.DataFrame:
    """Process breakend ALT field from VCF into structured format.

    Takes a series of ALT fields from a VCF file containing breakend records and parses them
    into a structured DataFrame with breakend orientation and mate information.

    Args:
        alt_series: Series containing ALT field values from VCF breakend records

    Returns:
        DataFrame with parsed breakend information containing columns:
        - orientation (str): Breakend orientation ([ or ])
        - mate_chr (str): Chromosome of mate breakend
        - mate_pos (int): Position of mate breakend
        - relative_pos (str): Position relative to breakend (L/R)
        - insert_seq (str): Any inserted sequence at breakpoint
        - no_mate (bool): Whether breakend has a mate
    """
    # Use VCF missing-value sentinel "." for NaN entries (handles mixed-type CSV columns)
    # so that no_mate is set correctly for records with no ALT field
    alt_series = alt_series.fillna('.').astype(str)

    # Regex pattern to parse breakend ALT field format

    # Extract components using regex
    processed_alt = alt_series.str.extract(_BND_PATTERN)

    # Determine relative position of the partner breakend


    # Combine inserted sequences
    processed_alt.loc[:, 'insert_seq'] = (
        processed_alt['inserted_seq1'].fillna('').str[1:] +
        processed_alt['inserted_seq2'].fillna('').str[:-1]
    )

    # Mark records with no mate
    processed_alt.loc[:, 'no_mate'] = processed_alt['sep'] == '.'

    # Clean up and rename columns
    processed_alt = processed_alt.rename(
        columns={
            'orientation1': 'orientation',
            'chr': 'mate_chr',
            'pos': 'mate_pos'
        }
    )

    processed_alt.loc[:, 'relative_pos'] = np.where(
        processed_alt['inserted_seq1'].isna(),
        processed_alt['orientation'] + 'N',  # Left if no leading sequence
        'N' + processed_alt['orientation']  # Right if has leading sequence
    )

    processed_alt = processed_alt.drop(
        columns=['orientation', 'inserted_seq1', 'inserted_seq2', 'orientation2', 'sep']
    )

    return processed_alt

def get_insert_seq_len_from_alt(bp_df: pd.DataFrame):
    """
    Calculates insert sequence length from ALT columns of bps dataframe
    Requires ALT column to be present in the dataframe and for insert sequence to be present in it
    """
    processed_alts = process_breakend_alt(bp_df['ALT_bnd1']), process_breakend_alt(bp_df['ALT_bnd2'])
    insert_seq_lens = processed_alts[0]['insert_seq'].str.len(), processed_alts[1]['insert_seq'].str.len()
    if any(insert_seq_lens[0] != insert_seq_lens[1]):
        raise ValueError("Insert sequence length mismatch between breakends in single breakpoint")
    return insert_seq_lens[0]
