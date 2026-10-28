"""
Functions for handling breakpoint dataframes
Includes functions for working within format + converting from format
"""

from collections import defaultdict

import numpy as np
import pandas as pd

from . import breakends as bnds
from . import events as evts
from . import utils
from . import _caller_params as caller_params

def _get_annotated_svtypes(bp_df: pd.DataFrame, caller_params: dict|None = None) -> pd.DataFrame:
    """
    Get annotated SV types for the breakpoint dataframe.
    """
    bp_df_ = bp_df.copy()

    return annotate_svtypes(bp_df_, caller_params['source'])

def _get_annotator_col(bp_df: pd.DataFrame, col_name: str) -> pd.Series:
    """
    Get an annotator column (ANNOT_<col_name>, see vcfparse) for the breakpoint dataframe.

    bnd_to_bp splits the column into _bnd1/_bnd2 when the two breakends of any
    breakpoint disagree; bnd1 is then used, falling back to bnd2. Returns an
    all-NaN column when the callset's VCF was not annotated.
    """
    col = f'ANNOT_{col_name}'
    if col in bp_df.columns:
        return bp_df[col]
    if f'{col}_bnd1' in bp_df.columns:
        return bp_df[f'{col}_bnd1'].fillna(bp_df[f'{col}_bnd2'])
    return pd.Series(np.nan, index=bp_df.index)


_custom_col_dispatch = {
    'SVTYPE': _get_annotated_svtypes,
    # duphold
    'DHFC': lambda bp_df, caller_params: _get_annotator_col(bp_df, 'DHFC'),
    'DHBFC': lambda bp_df, caller_params: _get_annotator_col(bp_df, 'DHBFC'),
    'DHFFC': lambda bp_df, caller_params: _get_annotator_col(bp_df, 'DHFFC'),
    'GCF': lambda bp_df, caller_params: _get_annotator_col(bp_df, 'GCF'),
}

def common_bps_preprocess(bp_df, caller_params: dict|None = None) -> pd.DataFrame:
    """
    Common preprocessing steps run after caller-specific preprocessing for all breakpoint dataframes.
    Requires a breakpoints dataframe and runs preprocessing steps according to config of the caller
    """

    bp_df_ = bp_df.copy()

    # Adding custom prediction columns
    for custom_col_name in caller_params['custom_cols_pred']:
        if custom_col_name not in _custom_col_dispatch.keys():
            raise KeyError(f"Custom column {custom_col_name} specified in callset parameters but not supported.")
        bp_df_.loc[:, f'CUSTOM_{custom_col_name}'] = _custom_col_dispatch[custom_col_name](bp_df_, caller_params=caller_params)

    return bp_df_

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
    return bp_df_

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
            insert_source_dispatch[caller_params.CALLSET_PARAM_DICT[source]['svlen_source']](
                bp_df_[bp_df_['source'] == source]
            )
        )
    insert_sizes.loc[insert_sizes < 0] = 0
    insert_sizes = insert_sizes.fillna(0)  # Convert NAs to 0
    insert_sizes = insert_sizes.astype(int)
    return insert_sizes


def annotate_svtypes(bps_df, source, insertion_length_threshold=0.7):
    """
    Annotate structural variant types for each breakpoint-pair record in a DataFrame.

    This function infers canonical SV types (DEL, INS, DUP, INV, or OTHER) from
    paired breakend orientation and chromosomal context.

    Classification of complex indels (events with both a reference gap and assembled
    insertion) follows the StructuralVariantAnnotation approach: an event is classified
    as INS when ins_lens >= abs(net_event_length) * insertion_length_threshold,
    where net_event_length is ins_lens + 1 - pos_diff for DEL-like orientations
    and ins_lens + 1 + pos_diff for DUP-like orientations.

    Args:
        bps_df (pd.DataFrame): DataFrame with breakpoint-pair records.
        source (str): SV caller/source string, required for insert size calculation.
        insertion_length_threshold (float): Fraction of the absolute net event length
            that the insertion must reach to be classified as INS rather than DEL/DUP.
            Default 0.7 matches the GRIDSS simple-event-annotation.R script.

    Returns:
        pd.Series: Series of SV type strings for each record.
    """
    ins_lens = get_insert_size(bps_df, source=source)
    bnd1 = bnds.process_breakend_alt(bps_df['ALT_bnd1'])
    bnd2 = bnds.process_breakend_alt(bps_df['ALT_bnd2'])

    same_chrom_mask = bps_df["CHROM_bnd1"] == bps_df["CHROM_bnd2"]

    del_ins_bnds_mask = (
        (bnd1["relative_pos"] == "N[") & (bnd2["relative_pos"] == "]N")
    )
    dup_bnds_mask = (
        (bnd1["relative_pos"] == "]N") & (bnd2["relative_pos"] == "N[")
    )
    inv_bnds_mask = (
        ((bnd1["relative_pos"] == "N]") & (bnd2["relative_pos"] == "N]"))
        | ((bnd1["relative_pos"] == "[N") & (bnd2["relative_pos"] == "[N"))
    )
    pos_diff = bps_df["POS_bnd2"] - bps_df["POS_bnd1"]

    # StructuralVariantAnnotation-style net event lengths:
    #   DEL orientation: insLen + 1 - pos_diff  (negative = net deletion)
    #   DUP orientation: insLen + 1 + pos_diff  (always positive)
    del_net_event_length = ins_lens + 1 - pos_diff
    dup_net_event_length = ins_lens + 1 + pos_diff

    # Classify as INS when insertion fills a significant fraction of the net change
    ins_over_del_mask = ins_lens >= np.abs(del_net_event_length) * insertion_length_threshold
    ins_over_dup_mask = ins_lens >= np.abs(dup_net_event_length) * insertion_length_threshold

    del_mask = (
        same_chrom_mask & del_ins_bnds_mask & ~ins_over_del_mask
    )
    ins_mask = (
        (same_chrom_mask & del_ins_bnds_mask & ins_over_del_mask)
        | (same_chrom_mask & dup_bnds_mask & ins_over_dup_mask)
    )
    dup_mask = same_chrom_mask & dup_bnds_mask & ~ins_over_dup_mask
    inv_mask = same_chrom_mask & inv_bnds_mask

    svtypes = np.full(len(bps_df), np.nan, dtype=object)

    svtype_dict = {
        "DEL": del_mask,
        "INS": ins_mask,
        "DUP": dup_mask,
        "INV": inv_mask,
    }

    for svtype, mask in svtype_dict.items():
        svtypes[mask] = svtype

    svtypes = pd.Series(np.where(pd.isna(svtypes), "OTHER", svtypes))
    svtypes.index = bps_df.index

    return svtypes

def get_del_size(bps_df: pd.DataFrame, *args, **kwargs) -> pd.Series:
    """
    Calculates deletion size from POS values in bps_df
    Reuqires INFO_INFERRED_SVTYPE, to be present in the dataframe in addition to normal bps_df columns
    """
    bps_df_ = bps_df.copy()

    ins_sizes = get_insert_size(bps_df_, source=kwargs['source'])
    del_size = bps_df_[bps_df_['INFO_INFERRED_SVTYPE'] == 'DEL']['POS_bnd2'] - bps_df_[bps_df_['INFO_INFERRED_SVTYPE'] == 'DEL']['POS_bnd1'] - 1 - ins_sizes

    del_size.loc[del_size.isna()] = 0
    del_size.loc[del_size < 0] = 0
    del_size = del_size.astype(int)

    return del_size

def get_dup_size(bps_df: pd.DataFrame, *args, **kwargs) -> pd.Series:
    """
    Calculates duplication size from POS values in bps_df
    """
    bps_df_ = bps_df.copy()
    dup_size = abs(bps_df_['POS_bnd1'] - bps_df_['POS_bnd2'])
    dup_size = dup_size.astype(int)
    return dup_size

def get_default_size(bps_df: pd.DataFrame, *args, **kwargs) -> pd.Series:
    """
    Function that is called for SV types that are not DEL, DUP or INS when asked to get the size of the SV
    Simply returns a vector of 1s
    """
    return pd.Series(index=bps_df.index, data=1)

def get_sv_sizes(bps_df: pd.DataFrame, source: str | None = None) -> pd.Series:

    bps_df_ = bps_df.copy()

    try:
        inferred_svtypes = bps_df_['INFO_INFERRED_SVTYPE']
    except KeyError:
        inferred_svtypes = annotate_svtypes(bps_df_, source)
        bps_df_.loc[:, 'INFO_INFERRED_SVTYPE'] = inferred_svtypes

    sv_size_dispatch = {
        'DEL': get_del_size,
        'INS': get_insert_size,
        'DUP': get_dup_size,
    }

    sv_sizes = pd.Series(index=bps_df_.index)
    for svtype in np.unique(inferred_svtypes):
        if svtype in sv_size_dispatch:
            sv_sizes.loc[inferred_svtypes == svtype] = sv_size_dispatch[svtype](bps_df_[inferred_svtypes == svtype], source=source)
        else:
            sv_sizes.loc[inferred_svtypes == svtype] = get_default_size(bps_df_[inferred_svtypes == svtype], source=source)


    return sv_sizes


def _get_svlen_svtype_filter_mask(
    bps_df: pd.DataFrame,
    source: str,
    svlens: tuple[int, int] | None = None,
    svtypes: list[str] | str | None = None
) -> pd.Series:
    """
    Get mask of breakpoint DataFrame for given SVLEN and SVTYPE ranges.

    Args:
        bps_df (pd.DataFrame): Breakpoint DataFrame.
        source (str): Source/caller name, needed for annotation.
        svlens (tuple, optional): Tuple of (min, max) length. Defaults to (0, np.inf).
        svtypes (list[str], optional): List of SV types to include. Defaults to ['DEL', 'DUP', 'INV', 'TRA'].

    Returns:
        pd.Series: Mask of breakpoint DataFrame for given SVLEN and SVTYPE ranges.
    """
    if svlens is None:
        svlens = caller_params.DEFAULT_SV_LENS
    elif not isinstance(svlens, tuple):
        raise ValueError('svlens argument must be a tuple of (min, max) lengths or None')
    svlen_min, svlen_max = svlens

    if svtypes is None:
        svtypes = caller_params.DEFAULT_SV_TYPES
    elif isinstance(svtypes, str):
        svtypes = [svtypes]
    elif isinstance(svtypes, list):
        svtypes = svtypes
    else:
        raise ValueError(f'Invalid type for svtypes: {type(svtypes)}')

    svtypes_col = annotate_svtypes(bps_df, source)

    svtype_mask = svtypes_col.isin(svtypes)

    sv_sizes = get_sv_sizes(bps_df, source)
    ignore_svlen_svtypes = [no_size_svtype for no_size_svtype in ['INV', 'BND'] if no_size_svtype in svtypes]
    svlen_mask = ((sv_sizes >= svlen_min) & (sv_sizes <= svlen_max)) | svtypes_col.isin(ignore_svlen_svtypes)

    return svtype_mask & svlen_mask


def svlen_svtype_filter(
    bps_df: pd.DataFrame,
    source: str,
    svlens: tuple[int, int] | None = None,
    svtypes: list[str] | None = None
) -> pd.DataFrame:
    """
    Filter a breakpoint DataFrame for given SVLEN and SVTYPE ranges.

    Args:
        bps_df (pd.DataFrame): Breakpoint DataFrame.
        source (str): Source/caller name, needed for annotation.
        svlens (tuple, optional): Tuple of (min, max) length. Defaults to (0, np.inf).
        svtype (list[str], optional): List of SV types to include. Defaults to ['DEL', 'DUP', 'INV', 'INS', 'TRA'].

    Returns:
        pd.DataFrame: Filtered breakpoint DataFrame.
    """
    svlen_svtype_mask = _get_svlen_svtype_filter_mask(bps_df, source, svlens=svlens, svtypes=svtypes)

    return bps_df.loc[svlen_svtype_mask].copy()
