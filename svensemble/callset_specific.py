from collections.abc import Callable
from datetime import datetime
from typing import Union, List, Literal, Dict, NotRequired, TypedDict

import pandas as pd
import numpy as np

from . import breakends as bnds
from . import events as evts
from . import utils as sv_utils
from . import _caller_params
from ._constants import BND1, BND2, CI_LEFT, CI_RIGHT, CUSTOM_PREFIX

def gridss_preprocess(
    gridss_callset: pd.DataFrame,
    verbose: bool = False,
    caller_params: dict|None = None
) -> pd.DataFrame:
    """Preprocess GRIDSS callset by splitting CIPOS values and cleaning columns.

    Args:
        gridss_callset: DataFrame containing GRIDSS variant calls with INFO_CIPOS field
        verbose: If True, print progress messages during processing
        caller_params: Caller-specific parameters

    Returns:
        Processed DataFrame with split CIPOS columns

    Raises:
        KeyError: If INFO_CIPOS column is missing from input DataFrame
    """
    if caller_params is None:
        caller_params = _caller_params.GRIDSS_PARAMS

    if verbose:
        print("Preprocessing GRIDSS callset...")

    gridss_callset_ = gridss_callset.copy()

    gridss_callset_.loc[:, ['INFO_CIPOS_l', 'INFO_CIPOS_r']] = (
        sv_utils.split_cipos(gridss_callset_['INFO_CIPOS'])
    )
    gridss_callset_ = gridss_callset_.drop(columns=['INFO_CIPOS'])

    return gridss_callset_


def gridss_post_bps(bp_df: pd.DataFrame) -> pd.DataFrame:
    """Post-process GRIDSS breakpoints after bnd_to_bp conversion.

    1. Unifies per-breakend QUAL into a single column (min of the two).
    2. Unifies per-breakend FILTER_PASS into a single column (AND logic:
       both breakends must pass) and drops the per-breakend columns.

    Args:
        bp_df: DataFrame of GRIDSS breakpoints from bnd_to_bp

    Returns:
        DataFrame with unified QUAL and FILTER_PASS columns
    """
    if 'QUAL' not in bp_df.columns:
        try:
            bp_df.loc[:, 'QUAL'] = np.min(
                [bp_df['QUAL_bnd1'], bp_df['QUAL_bnd2']], axis=0
            )
        except KeyError:
            raise KeyError('No QUAL column found in GRIDSS breakpoints')

    if 'FILTER_PASS' not in bp_df.columns and \
            'FILTER_PASS_bnd1' in bp_df.columns and \
            'FILTER_PASS_bnd2' in bp_df.columns:
        bp_df.loc[:, 'FILTER_PASS'] = (
            bp_df['FILTER_PASS_bnd1'].astype(bool) &
            bp_df['FILTER_PASS_bnd2'].astype(bool)
        )
        bp_df = bp_df.drop(columns=['FILTER_PASS_bnd1', 'FILTER_PASS_bnd2'])

    return bp_df


def gnomad_split_evidence(
    evidence_series: pd.Series,
    evidence_types: list[str] | None = None,
    caller_params: dict|None = None
) -> pd.DataFrame:
    """Split gnomAD evidence string into boolean columns for each evidence type.

    Args:
        evidence_series: Series containing evidence strings (e.g. 'SR,PE')
        evidence_types: List of evidence types to check for. Defaults to ['SR','PE','RD']

    Returns:
        DataFrame with boolean columns for each evidence type indicating presence
    """
    if caller_params is None:
        caller_params = _caller_params.GNOMAD_PARAMS
    if evidence_types is None:
        evidence_types = caller_params['method_cipos'].keys()

    evidence_df = pd.DataFrame(index=evidence_series.index, columns=evidence_types)
    for evidence_type in evidence_types:
        evidence_df[evidence_type] = evidence_series.apply(lambda x: evidence_type in x)
    return evidence_df


def gnomad_get_best_evidence(
    evidence_series: pd.Series,
    evidence_types: list[str] | None = None,
    caller_params: dict|None = None
) -> pd.Series:
    """Get the highest priority evidence type for each variant.

    Args:
        evidence_series: Series containing evidence strings
        evidence_types: List of evidence types in priority order. Defaults to ['SR','PE','RD']

    Returns:
        Series containing the highest priority evidence type for each variant
    """
    if caller_params is None:
        caller_params = _caller_params.GNOMAD_PARAMS
    if evidence_types is None:
        evidence_types = caller_params['method_cipos'].keys()

    evidence_df = gnomad_split_evidence(evidence_series, evidence_types)
    return evidence_df.idxmax(axis=1)


def gnomad_get_cipos(
    evidence_series: pd.Series,
    evidence_types: list[str] | None = None,
    cipos_dict: dict[str, tuple[int, int]] | None = None,
    caller_params: dict|None = None
) -> pd.DataFrame:
    """Get confidence interval positions based on evidence type.

    Args:
        evidence_series: Series containing evidence strings
        evidence_types: List of evidence types in priority order. Defaults to ['SR','PE','RD']
        cipos_dict: Dictionary mapping evidence types to (left,right) CI sizes.
                   Defaults to values from _caller_params.GNOMAD_PARAMS['method_cipos']

    Returns:
        DataFrame with INFO_CIPOS_l and INFO_CIPOS_r columns containing CI sizes
    """
    if caller_params is None:
        caller_params = _caller_params.GNOMAD_PARAMS
    if evidence_types is None:
        evidence_types = caller_params['method_cipos'].keys()
    if cipos_dict is None:
        cipos_dict = caller_params['method_cipos']

    evidence_df = gnomad_get_best_evidence(evidence_series, evidence_types)
    cipos_sizes = evidence_df.map(cipos_dict)
    return pd.DataFrame(
        index=evidence_df.index,
        data={
            'INFO_CIPOS_l': -cipos_sizes,
            'INFO_CIPOS_r': cipos_sizes
        }
    )


def gnomad_preprocess(
    gnomad_callset: pd.DataFrame,
    evidence_cipos: Union[Dict[str, int], Literal['default']] = 'default',
    svtypes_retain: Union[str, List[str]] = 'simple',
    filter_pass_only: bool = False,
    verbose: bool = False,
    caller_params: dict|None = None
) -> pd.DataFrame:
    """Preprocess gnomAD structural variant callset.

    Processes a gnomAD SV callset by:
    1. Adding confidence interval sizes based on evidence type
    2. Filtering to specified SV types
    3. Optionally filtering to PASS variants only

    Args:
        gnomad_callset: DataFrame containing gnomAD SV calls
        evidence_cipos: Dict mapping evidence types to CI sizes in bp, or 'default'
            Default maps SR->10bp, PE->100bp, RD->500bp
        svtypes_retain: Either 'simple' for basic SV types (DEL,DUP,INV,INS),
            or list of specific SV types to include
        filter_pass_only: If True, keep only variants with FILTER=PASS
        verbose: If True, print processing information
        caller_params: Caller-specific parameters

    Returns:
        Processed DataFrame

    Raises:
        NotImplementedError: If unsupported SV types requested
    """
    if verbose:
        print("Preprocessing gnomAD callset...")
    if caller_params is None:
        caller_params = _caller_params.GNOMAD_PARAMS

    svtypes_supported = caller_params['supported_svtypes']
    if verbose:
        print(f"SV types currently supported: {svtypes_supported}")

    if evidence_cipos == 'default':
        evidence_cipos = caller_params['method_cipos']
    if verbose:
        print(f"Using default evidence CIPOS: {evidence_cipos}")

    if svtypes_retain == 'simple':
        svtypes_retain = svtypes_supported
        if verbose:
            print(f"Using default viable SV types: {svtypes_retain}")
    elif all(svtype in svtypes_supported for svtype in svtypes_retain):
        if verbose:
            print(f"Using specified viable SV types: {svtypes_retain}")
    else:
        raise NotImplementedError(f"Supported viable SV types: {svtypes_supported}")

    gnomad_callset_ = gnomad_callset.copy()

    gnomad_callset_ = gnomad_callset_[
        gnomad_callset_['INFO_SVTYPE'].isin(svtypes_retain)
    ]

    gnomad_callset_.loc[
        gnomad_callset_['INFO_CHR2'].isna(),
        'INFO_CHR2'
    ] = gnomad_callset_['CHROM']

    if filter_pass_only:
        if verbose:
            print("Filtering to PASS variants only")
        gnomad_callset_ = gnomad_callset_[
            gnomad_callset_['FILTER_PASS'].astype(bool)
        ]

    if verbose:
        print("Setting confidence intervals based on evidence type")
    gnomad_callset_.loc[:, ['INFO_CIPOS_l', 'INFO_CIPOS_r']] = gnomad_get_cipos(
        gnomad_get_best_evidence(gnomad_callset_['INFO_EVIDENCE']),
        evidence_types=evidence_cipos.keys(),
        cipos_dict=evidence_cipos
    )
    gnomad_callset_.loc[:, 'INFO_CIEND_l'] = gnomad_callset_['INFO_CIPOS_l']
    gnomad_callset_.loc[:, 'INFO_CIEND_r'] = gnomad_callset_['INFO_CIPOS_r']

    gnomad_callset_.loc[:, 'INFO_EVENT'] = gnomad_callset_['ID']

    return gnomad_callset_


def gnomad_event_to_bnd(
    gnomad_callset: pd.DataFrame,
    caller_params: dict|None = None
) -> pd.DataFrame:
    """Convert gnomAD event-based SV records to breakend (BND) format.

    Args:
        gnomad_callset: DataFrame containing gnomAD structural variants
        caller_params: Caller-specific parameters

    Returns:
        DataFrame with events converted to breakend format
    """
    if caller_params is None:
        caller_params = _caller_params.GNOMAD_PARAMS

    return evts.event_to_bnd(gnomad_callset.copy())

# MANTA
# Manta callset includes both event and bnd entries
# For preprocessing they are handled separately
# Part of the preprocessing is common between bnds and evts
# Thus 3 functions are used to preprocess manta callset:
# - manta_preprocess_both: common preprocessing for both event and bnd entries
# - manta_preprocess_evts_only: preprocessing for event entries only
# - manta_preprocess_bnd_only: preprocessing for bnd entries only

def manta_preprocess_common(manta_df: pd.DataFrame, caller_params: dict|None = None) -> pd.DataFrame:
    """
    Common preprocessing steps for both event and bnd entries.

    Args:
        manta_df: Manta DataFrame
        caller_params: Caller-specific parameters

    Returns:
        Preprocessed Manta DataFrame
    """
    if caller_params is None:
        caller_params = _caller_params.MANTA_CANDIDATE_PARAMS

    manta_df_ = manta_df.copy()

    # When SVLEN is not applicable, it is set to None.
    # We get rid of NaNs
    manta_df_.loc[:, 'INFO_SVLEN'] = manta_df_.loc[:, 'INFO_SVLEN'].astype(float)
    manta_df_.loc[:, 'INFO_SVLEN'] = manta_df_.loc[:, 'INFO_SVLEN'].fillna(0)
    manta_df_.loc[:, 'INFO_SVLEN'] = manta_df_.loc[:, 'INFO_SVLEN'].astype(int)

    # INFO_HOMLEN is a tuple initially. Manta is not supposed to have multiple HOMLEN values
    # So pick the first value of the tuple
    manta_df_.loc[:, 'INFO_HOMLEN'] = manta_df_.loc[:, 'INFO_HOMLEN'].str[0]

    # Set INFO_CHR2 to CHROM
    manta_df_.loc[:, 'INFO_CHR2'] = manta_df_.loc[:, 'CHROM']

    # Drop INFO_POS2 as it does not contain information
    manta_df_ = manta_df_.drop(columns=['INFO_POS2'])

    if caller_params.get('format_cols'):
        # SAMPLE_SR and SAMPLE_PE are both columns of tuples with 2 values
        # First value corresponds to evidence of REF, second to evidence of ALT
        # We want to split these into 2 columns:
        # - SAMPLE_SR_REF
        # - SAMPLE_SR_ALT
        # - SAMPLE_PE_REF
        # - SAMPLE_PE_ALT
        #
        # Since some variants may lack SAMPLE_SR or SAMPLE_PE, we need to handle these cases
        # Use default tuple (0,0) for these cases
        default_tuple = (0,0)

        manta_df_.loc[:, 'SAMPLE_SR_REF'] = manta_df_.loc[:, 'SAMPLE_SR'].str[0].fillna(default_tuple[0])
        manta_df_.loc[:, 'SAMPLE_SR_ALT'] = manta_df_.loc[:, 'SAMPLE_SR'].str[1].fillna(default_tuple[1])
        manta_df_.loc[:, 'SAMPLE_PR_REF'] = manta_df_.loc[:, 'SAMPLE_PR'].str[0].fillna(default_tuple[0])
        manta_df_.loc[:, 'SAMPLE_PR_ALT'] = manta_df_.loc[:, 'SAMPLE_PR'].str[1].fillna(default_tuple[1])

        cols_to_drop = ['SAMPLE_SR', 'SAMPLE_PR']
        manta_df_ = manta_df_.drop(columns=cols_to_drop)

    return manta_df_

def manta_preprocess_evts_only(manta_df: pd.DataFrame, caller_params: dict|None = None) -> pd.DataFrame:
    """
    Preprocessing for Manta event entries only.
    """
    manta_df_ = manta_df.copy()
    # Ensure correct CIPOS_l/CIPOS_r and CIEND_l/CIEND_r values. Drop CIPOS and CIEND columns.
    manta_df_.loc[manta_df_['INFO_CIEND'].isna(), 'INFO_CIEND'] = manta_df_.loc[manta_df_['INFO_CIEND'].isna(), 'INFO_CIPOS']
    manta_df_.loc[:, ['INFO_CIPOS_l', 'INFO_CIPOS_r']] = (
        sv_utils.split_cipos(manta_df_['INFO_CIPOS'])
    )
    manta_df_.loc[:, ['INFO_CIEND_l', 'INFO_CIEND_r']] = (
        sv_utils.split_ciend(manta_df_['INFO_CIEND'])
    )
    cols_to_drop = ['INFO_CIPOS', 'INFO_CIEND']
    manta_df_ = manta_df_.drop(columns=cols_to_drop)

    # Set INFO_EVENT to ID
    manta_df_.loc[:, 'INFO_EVENT'] = manta_df_.loc[:, 'ID']

    # INFO_SVLEN is a tuple initially. Manta is not supposed to have multiple SVLEN values
    # So pick the first value of the tuple
    manta_df_.loc[:, 'INFO_SVLEN'] = manta_df_.loc[:, 'INFO_SVLEN'].str[0]
    # Manta gives negative values for deletions. Convert to positive
    manta_df_.loc[manta_df_['INFO_SVTYPE'] == 'DEL', 'INFO_SVLEN'] = -manta_df_.loc[manta_df_['INFO_SVTYPE'] == 'DEL', 'INFO_SVLEN']

    return manta_preprocess_common(manta_df_, caller_params)

def manta_preprocess_bnds_only(manta_df: pd.DataFrame, caller_params: dict|None = None) -> pd.DataFrame:
    """
    Preprocessing for Manta BND entries only.

    Manta BND records do not carry an EVENT INFO field; instead each record has
    a MATEID pointing to its partner.  A shared INFO_EVENT is derived as
    min(ID, MATEID) so that both mates in a pair produce the same key, which
    is required by bnd_to_bp for correct pairing.
    """
    manta_df_ = manta_df.copy()

    # Derive shared pairing key from ID and MATEID.
    # INFO_MATEID is parsed as a one-element tuple; extract the scalar first.
    mateid_first = manta_df_['INFO_MATEID'].apply(
        lambda x: x[0] if isinstance(x, (list, tuple)) else x
    )
    manta_df_.loc[:, 'INFO_EVENT'] = pd.concat(
        [manta_df_['ID'], mateid_first], axis=1
    ).min(axis=1)

    manta_df_.loc[:, ['INFO_CIPOS_l', 'INFO_CIPOS_r']] = sv_utils.split_cipos(manta_df_['INFO_CIPOS'])

    # INFO_CIEND is not meaningful for BND records (typically None); drop without splitting.
    cols_to_drop = ['INFO_CIPOS', 'INFO_CIEND']
    manta_df_ = manta_df_.drop(columns=cols_to_drop, errors='ignore')
    return manta_preprocess_common(manta_df_, caller_params)


def manta_post_bps(bp_df: pd.DataFrame) -> pd.DataFrame:
    """Post-process Manta breakpoints after bnd_to_bp conversion.

    Collapses per-breakend pair-count columns into single columns using the
    mean of both breakends, then drops the per-breakend originals.
    Columns that were already identical between breakends (and thus kept as a
    single column by bnd_to_bp) are left untouched.
    """
    _PAIR_COUNT_COLS = [
        'INFO_PAIR_COUNT',
        'INFO_BND_PAIR_COUNT',
        'INFO_UPSTREAM_PAIR_COUNT',
        'INFO_DOWNSTREAM_PAIR_COUNT',
    ]
    for col in _PAIR_COUNT_COLS:
        bnd1 = f'{col}_{BND1}'
        bnd2 = f'{col}_{BND2}'
        if col not in bp_df.columns and bnd1 in bp_df.columns and bnd2 in bp_df.columns:
            bp_df.loc[:, col] = bp_df[[bnd1, bnd2]].astype(float).mean(axis=1)
            bp_df = bp_df.drop(columns=[bnd1, bnd2])

    return bp_df


def delly_preprocess_common(delly_df: pd.DataFrame, caller_params: dict|None = None) -> pd.DataFrame:
    """
    Common preprocessing steps for both event and bnd entries in Delly callset dataframe.
    """
    delly_df_ = delly_df.copy()

    # When SVLEN is not applicable, it is set to None.
    # We get rid of NaNs
    delly_df_.loc[:, 'INFO_SVLEN'] = delly_df_.loc[:, 'INFO_SVLEN'].astype(float)
    delly_df_.loc[:, 'INFO_SVLEN'] = delly_df_.loc[:, 'INFO_SVLEN'].fillna('0')
    delly_df_.loc[:, 'INFO_SVLEN'] = delly_df_.loc[:, 'INFO_SVLEN'].astype(int)

    return delly_df_


def delly_preprocess_evts_only(delly_df: pd.DataFrame, caller_params: dict|None = None) -> pd.DataFrame:
    """
    Preprocessing for Delly event entries only.
    """
    delly_df_ = delly_df.copy()
    delly_df_.loc[:, ['INFO_CIPOS_l', 'INFO_CIPOS_r']] = sv_utils.split_cipos(delly_df_['INFO_CIPOS'])
    delly_df_.loc[:, ['INFO_CIEND_l', 'INFO_CIEND_r']] = sv_utils.split_ciend(delly_df_['INFO_CIEND'])
    cols_to_drop = ['INFO_CIPOS', 'INFO_CIEND']
    delly_df_ = delly_df_.drop(columns=cols_to_drop)

    delly_df_.loc[:, 'INFO_SVLEN'] = delly_df_.loc[:, 'INFO_SVLEN'].astype(float)
    delly_df_.loc[:, 'INFO_SVLEN'] = delly_df_.loc[:, 'INFO_SVLEN'].fillna('0')
    delly_df_.loc[:, 'INFO_SVLEN'] = delly_df_.loc[:, 'INFO_SVLEN'].astype(int)

    delly_df_.loc[:, 'INFO_EVENT'] = delly_df_.loc[:, 'ID']

    no_chr2_mask = delly_df_['INFO_CHR2'].isna()
    delly_df_.loc[no_chr2_mask, 'INFO_CHR2'] = delly_df_.loc[no_chr2_mask, 'CHROM']

    if np.any(delly_df_['INFO_END'].isna()):
        raise ValueError(f"INFO_END is missing for some entries: {delly_df_['INFO_END'].isna().sum()} out of {delly_df_.shape[0]} entries")


    return delly_preprocess_common(delly_df_, caller_params)


# CT -> (ALT of the record's own breakend, ALT of its mate), in VCF breakend
# notation: b = anchor base, c:p = the partner breakend.
_DELLY_BND_ALTS = {
    '3to5': ('{b}[{c}:{p}[', ']{c}:{p}]{b}'),
    '5to3': (']{c}:{p}]{b}', '{b}[{c}:{p}['),
    '3to3': ('{b}]{c}:{p}]', '{b}]{c}:{p}]'),
    '5to5': ('[{c}:{p}[{b}', '[{c}:{p}[{b}'),
}
# CT as seen from the mate breakend.
_DELLY_MATE_CT = {'3to5': '5to3', '5to3': '3to5', '3to3': '3to3', '5to5': '5to5'}


def delly_split_bnds(delly_df: pd.DataFrame) -> pd.DataFrame:
    """Give each Delly BND/TRA record its mate breakend record.

    Delly writes a translocation as one record; the partner breakend exists
    only as INFO CHR2/POS2 (END for TRA) and the connection type CT. This
    returns two rows per record, ID-suffixed '_2nd' (the original) and '_1st'
    (the mate), so that stripping the 4-character suffix yields a shared
    event ID. Same layout as the former delly2bnd script, without a reference:

    - Original: ALT rebuilt from CT, anchored on its REF base; INFO_POS2 set
      (from END for TRA).
    - Mate: CHROM/POS = the partner; CHR2/POS2 = the original's CHROM/POS;
      END = POS + 1; CT seen from the mate (3to5 <-> 5to3); CIPOS and CIEND
      swapped; REF and ALT anchor 'N', since no reference base is fetched.
      Every other column is copied.

    Raises:
        ValueError: If any record's INFO_CT is missing or not one of
            3to5, 5to3, 3to3, 5to5.
    """
    ct = delly_df['INFO_CT']
    bad_ct = ct[~ct.isin(list(_DELLY_BND_ALTS))]
    if len(bad_ct) > 0:
        raise ValueError(
            f"Unexpected Delly INFO_CT for {len(bad_ct)} BND/TRA records: "
            f"{sorted(map(str, bad_ct.unique()))}; expected {list(_DELLY_BND_ALTS)}"
        )

    chrom = delly_df['CHROM'].astype(str)
    pos = delly_df['POS'].astype(int)
    chrom2 = delly_df['INFO_CHR2'].astype(str)
    is_tra = delly_df['INFO_SVTYPE'] == 'TRA'
    pos2 = delly_df['INFO_END'].where(is_tra, delly_df['INFO_POS2']).astype(int)
    anchor = delly_df['REF'].astype(str).str[0].str.upper()

    own = delly_df.copy()
    own['ID'] = delly_df['ID'].astype(str) + '_2nd'
    own['ALT'] = [
        _DELLY_BND_ALTS[t][0].format(b=b, c=c, p=p)
        for t, b, c, p in zip(ct, anchor, chrom2, pos2)
    ]
    own['INFO_POS2'] = pos2

    mate = delly_df.copy()
    mate['ID'] = delly_df['ID'].astype(str) + '_1st'
    mate['CHROM'] = chrom2
    mate['POS'] = pos2
    mate['REF'] = 'N'
    mate['ALT'] = [
        _DELLY_BND_ALTS[t][1].format(b='N', c=c, p=p)
        for t, c, p in zip(ct, chrom, pos)
    ]
    mate['INFO_CHR2'] = chrom
    mate['INFO_POS2'] = pos
    mate['INFO_END'] = pos2 + 1
    mate['INFO_CT'] = ct.map(_DELLY_MATE_CT)
    mate['INFO_CIPOS'] = delly_df['INFO_CIEND']
    mate['INFO_CIEND'] = delly_df['INFO_CIPOS']

    return pd.concat([own, mate], axis=0, ignore_index=True)


def delly_preprocess_bnds_only(delly_df: pd.DataFrame, caller_params: dict|None = None) -> pd.DataFrame:
    """
    Preprocessing for Delly BND/TRA entries only: adds each record's mate
    (delly_split_bnds), then derives INFO_EVENT, CIPOS bounds and INFO_END.
    """
    if delly_df.empty:
        return delly_df

    delly_df_ = delly_split_bnds(delly_df)
    delly_df_.loc[:, 'INFO_EVENT'] = delly_df_.loc[:, 'ID'].astype(str).str[:-4]

    delly_df_.loc[:, ['INFO_CIPOS_l', 'INFO_CIPOS_r']] = sv_utils.split_cipos(delly_df_['INFO_CIPOS'])

    delly_df_.loc[:, 'INFO_END'] = delly_df_.loc[:, 'INFO_POS2'].astype(int)

    cols_to_drop = ['INFO_CIPOS']
    delly_df_ = delly_df_.drop(columns=cols_to_drop)

    return delly_preprocess_common(delly_df_, caller_params)


# SNIFFLES2
# Sniffles2 does not emit CIPOS/CIEND fields; breakpoint uncertainty is reported
# as STDEV_POS (position stddev) and STDEV_LEN (length stddev).
# Both are converted to symmetric integer CIPOS/CIEND intervals before the
# standard BND/event pipeline is applied.

def _sniffles_cipos_from_stdev(stdev_series: pd.Series) -> pd.DataFrame:
    """Convert a Sniffles2 STDEV series into symmetric CIPOS_l / CIPOS_r integer columns.

    Args:
        stdev_series: Series of float standard-deviation values (e.g. INFO_STDEV_POS).

    Returns:
        DataFrame with two columns: <name>_l (negative) and <name>_r (positive),
        both equal in magnitude to the rounded standard deviation.
    """
    stdev_int = stdev_series.fillna(0).apply(lambda x: int(round(x)))
    base = stdev_series.name.replace('INFO_STDEV_POS', 'INFO_CIPOS').replace('INFO_STDEV_LEN', 'INFO_CIEND')
    return pd.DataFrame(
        {f'{base}_{CI_LEFT}': -stdev_int, f'{base}_{CI_RIGHT}': stdev_int},
        index=stdev_series.index
    )


def sniffles_preprocess_common(
    sniffles_df: pd.DataFrame,
    caller_params: dict | None = None
) -> pd.DataFrame:
    """Common preprocessing for all Sniffles2 entries (BND and non-BND).

    Applies steps shared by both BND and event preprocessing:
    1. Sets INFO_EVENT to the variant ID
    2. Converts INFO_STDEV_POS to symmetric INFO_CIPOS_l / INFO_CIPOS_r integers
    3. Converts negative SVLEN for DEL records to positive values

    Args:
        sniffles_df: DataFrame of Sniffles2 variant calls
        caller_params: Caller-specific parameters (defaults to SNIFFLES_PARAMS)

    Returns:
        Preprocessed copy of the input DataFrame
    """
    if caller_params is None:
        caller_params = _caller_params.SNIFFLES_PARAMS

    sniffles_df_ = sniffles_df.copy()

    sniffles_df_.loc[:, 'INFO_EVENT'] = sniffles_df_['ID']

    sniffles_df_.loc[:, ['INFO_CIPOS_l', 'INFO_CIPOS_r']] = _sniffles_cipos_from_stdev(
        sniffles_df_['INFO_STDEV_POS']
    )

    # Coverage field includes 5 separate metrics in different parts of the SV
    cov_cols = ['INFO_COVERAGE_1', 'INFO_COVERAGE_2', 'INFO_COVERAGE_3', 'INFO_COVERAGE_4', 'INFO_COVERAGE_5']
    if len(sniffles_df_) > 0:
        cov_expanded = pd.DataFrame(
            sniffles_df_['INFO_COVERAGE'].tolist(),
            index=sniffles_df_.index,
            columns=cov_cols,
            dtype=float,
        )
        sniffles_df_.loc[:, cov_cols] = cov_expanded.values
    else:
        for col in cov_cols:
            sniffles_df_[col] = pd.Series(dtype=float)
    sniffles_df_ = sniffles_df_.drop(columns=['INFO_COVERAGE'])

    # Sniffles2 reports negative SVLEN for deletions; convert to positive for consistency.
    # SVLEN is missing for BND/INV records; coerce NaN to 0 so the column stays int.
    sniffles_df_.loc[:, 'INFO_SVLEN'] = sniffles_df_.loc[:, 'INFO_SVLEN'].astype(float).fillna(0)
    del_mask = sniffles_df_['INFO_SVTYPE'] == 'DEL'
    sniffles_df_.loc[del_mask, 'INFO_SVLEN'] = sniffles_df_.loc[del_mask, 'INFO_SVLEN'].abs()
    sniffles_df_.loc[:, 'INFO_SVLEN'] = sniffles_df_.loc[:, 'INFO_SVLEN'].astype(int)

    return sniffles_df_


def sniffles_preprocess_evts_only(
    sniffles_df: pd.DataFrame,
    caller_params: dict | None = None
) -> pd.DataFrame:
    """Preprocessing for Sniffles2 non-BND event entries (DEL, INS, DUP, INV).

    In addition to the common steps, this function:
    1. Converts INFO_STDEV_LEN to symmetric INFO_CIEND_l / INFO_CIEND_r integers
    2. Sets INFO_CHR2 to CHROM (intrachromosomal events share start/end chromosome)

    Args:
        sniffles_df: DataFrame of Sniffles2 non-BND variant calls
        caller_params: Caller-specific parameters (defaults to SNIFFLES_PARAMS)

    Returns:
        Preprocessed copy of the input DataFrame
    """
    if caller_params is None:
        caller_params = _caller_params.SNIFFLES_PARAMS

    sniffles_df_ = sniffles_df.copy()

    sniffles_df_.loc[:, ['INFO_CIEND_l', 'INFO_CIEND_r']] = _sniffles_cipos_from_stdev(
        sniffles_df_['INFO_STDEV_LEN']
    )

    # Intrachromosomal events share start and end chromosome
    sniffles_df_.loc[:, 'INFO_CHR2'] = sniffles_df_['CHROM']

    return sniffles_preprocess_common(sniffles_df_, caller_params)


def sniffles_preprocess_bnds_only(
    sniffles_df: pd.DataFrame,
    caller_params: dict | None = None
) -> pd.DataFrame:
    """Preprocessing for Sniffles2 BND entries.

    INFO_CHR2 is already populated from the VCF for BND records; only the
    common preprocessing steps are applied.

    Args:
        sniffles_df: DataFrame of Sniffles2 BND variant calls
        caller_params: Caller-specific parameters (defaults to SNIFFLES_PARAMS)

    Returns:
        Preprocessed copy of the input DataFrame
    """
    if caller_params is None:
        caller_params = _caller_params.SNIFFLES_PARAMS

    return sniffles_preprocess_common(sniffles_df, caller_params)


# SURVIVOR

def survivor_preprocess_evts(
    survivor_df: pd.DataFrame,
    caller_params: dict | None = None
) -> pd.DataFrame:
    """Preprocess SURVIVOR event entries.

    Splits CIPOS/CIEND into left/right columns, drops the originals,
    and sets INFO_EVENT to the variant ID.

    Args:
        survivor_df: DataFrame of SURVIVOR variant calls
        caller_params: Caller-specific parameters (defaults to SURVIVOR_PARAMS)

    Returns:
        Preprocessed copy of the input DataFrame
    """
    if caller_params is None:
        caller_params = _caller_params.SURVIVOR_PARAMS

    df = survivor_df.copy()
    df.loc[:, ['INFO_CIPOS_l', 'INFO_CIPOS_r']] = sv_utils.split_cipos(df['INFO_CIPOS'])
    df.loc[:, ['INFO_CIEND_l', 'INFO_CIEND_r']] = sv_utils.split_ciend(df['INFO_CIEND'])
    df = df.drop(columns=['INFO_CIPOS', 'INFO_CIEND'])
    df.loc[:, 'INFO_EVENT'] = df['ID']
    return df


# SEVERUS

_SEVERUS_IMPRECISE_CI = 50

def severus_preprocess_common(
    severus_df: pd.DataFrame,
    caller_params: dict | None = None
) -> pd.DataFrame:
    """Common preprocessing for all Severus entries (BND and non-BND).

    1. Sets INFO_EVENT to variant ID
    2. Infers symmetric CIPOS from PRECISE/IMPRECISE flags
    3. Converts negative SVLEN for DEL records to positive
    4. Splits SAMPLE_hVAF 3-element tuple into three scalar columns

    Args:
        severus_df: DataFrame of Severus variant calls
        caller_params: Caller-specific parameters (defaults to SEVERUS_PARAMS)

    Returns:
        Preprocessed copy of the input DataFrame
    """
    if caller_params is None:
        caller_params = _caller_params.SEVERUS_PARAMS

    df = severus_df.copy()

    df.loc[:, 'INFO_EVENT'] = df['ID']

    precise_mask = df['INFO_PRECISE'].fillna(False).astype(bool)
    ci_val = np.where(precise_mask, 0, _SEVERUS_IMPRECISE_CI)
    df.loc[:, 'INFO_CIPOS_l'] = -ci_val
    df.loc[:, 'INFO_CIPOS_r'] = ci_val.copy()

    # SVLEN is missing for BND records; coerce NaN to 0 so the column stays int.
    df.loc[:, 'INFO_SVLEN'] = df.loc[:, 'INFO_SVLEN'].astype(float).fillna(0)
    del_mask = df['INFO_SVTYPE'] == 'DEL'
    df.loc[del_mask, 'INFO_SVLEN'] = df.loc[del_mask, 'INFO_SVLEN'].abs()
    df.loc[:, 'INFO_SVLEN'] = df.loc[:, 'INFO_SVLEN'].astype(int)

    if 'SAMPLE_hVAF' in df.columns:
        hvaf_cols = ['SAMPLE_hVAF_H0', 'SAMPLE_hVAF_H1', 'SAMPLE_hVAF_H2']
        if len(df) > 0:
            hvaf_expanded = pd.DataFrame(
                df['SAMPLE_hVAF'].tolist(),
                index=df.index,
                columns=hvaf_cols,
                dtype=float,
            )
            df.loc[:, hvaf_cols] = hvaf_expanded.values
        else:
            for col in hvaf_cols:
                df[col] = pd.Series(dtype=float)
        df = df.drop(columns=['SAMPLE_hVAF'])

    return df


def severus_preprocess_evts_only(
    severus_df: pd.DataFrame,
    caller_params: dict | None = None
) -> pd.DataFrame:
    """Preprocessing for Severus non-BND event entries (DEL, INS, DUP, INV).

    Sets INFO_CHR2, infers CIEND from precision flags (required by event_to_bnd),
    then applies common preprocessing.

    Args:
        severus_df: DataFrame of Severus non-BND variant calls
        caller_params: Caller-specific parameters (defaults to SEVERUS_PARAMS)

    Returns:
        Preprocessed copy of the input DataFrame
    """
    if caller_params is None:
        caller_params = _caller_params.SEVERUS_PARAMS

    df = severus_df.copy()

    df.loc[:, 'INFO_CHR2'] = df['CHROM']

    precise_mask = df['INFO_PRECISE'].fillna(False).astype(bool)
    ci_val = np.where(precise_mask, 0, _SEVERUS_IMPRECISE_CI)
    df.loc[:, 'INFO_CIEND_l'] = -ci_val
    df.loc[:, 'INFO_CIEND_r'] = ci_val.copy()

    return severus_preprocess_common(df, caller_params)


def severus_preprocess_bnds_only(
    severus_df: pd.DataFrame,
    caller_params: dict | None = None
) -> pd.DataFrame:
    """Preprocessing for Severus BND entries.

    BNDs go directly to bnd_to_bp (no event_to_bnd), so CIEND is not needed.
    Only the common preprocessing steps are applied.

    Args:
        severus_df: DataFrame of Severus BND variant calls
        caller_params: Caller-specific parameters (defaults to SEVERUS_PARAMS)

    Returns:
        Preprocessed copy of the input DataFrame
    """
    if caller_params is None:
        caller_params = _caller_params.SEVERUS_PARAMS

    return severus_preprocess_common(severus_df, caller_params)


# CUTESV

_CUTESV_IMPRECISE_CI = 50


def cutesv_preprocess_common(
    cutesv_df: pd.DataFrame,
    caller_params: dict | None = None
) -> pd.DataFrame:
    """Common preprocessing for all CuteSV entries (BND and non-BND).

    1. Sets INFO_EVENT to variant ID
    2. Converts negative SVLEN for DEL records to positive

    Args:
        cutesv_df: DataFrame of CuteSV variant calls
        caller_params: Caller-specific parameters (defaults to CUTESV_PARAMS)

    Returns:
        Preprocessed copy of the input DataFrame
    """
    if caller_params is None:
        caller_params = _caller_params.CUTESV_PARAMS

    df = cutesv_df.copy()

    df.loc[:, 'INFO_EVENT'] = df['ID']

    # SVLEN is missing for BND/INV records; coerce NaN to 0 so the column stays int.
    df.loc[:, 'INFO_SVLEN'] = df.loc[:, 'INFO_SVLEN'].astype(float).fillna(0)
    del_mask = df['INFO_SVTYPE'] == 'DEL'
    df.loc[del_mask, 'INFO_SVLEN'] = df.loc[del_mask, 'INFO_SVLEN'].abs()
    df.loc[:, 'INFO_SVLEN'] = df.loc[:, 'INFO_SVLEN'].astype(int)

    # INFO_AF is declared Number=A in CuteSV's VCF header, so pysam
    # returns a 1-element tuple (float,).  Unpack to scalar so it is
    # stored as a plain float and treated as continuous, not categorical.
    if 'INFO_AF' in df.columns:
        df['INFO_AF'] = df['INFO_AF'].map(
            lambda x: x[0] if isinstance(x, tuple) else x
        ).astype(float)

    return df


def cutesv_preprocess_evts_only(
    cutesv_df: pd.DataFrame,
    caller_params: dict | None = None
) -> pd.DataFrame:
    """Preprocessing for CuteSV non-BND event entries (DEL, INS, DUP, INV).

    1. Sets INFO_CHR2 to CHROM (intrachromosomal events)
    2. Splits native INFO_CIPOS and INFO_CILEN tuples
    3. For DEL/DUP/INV: computes both CIPOS and CIEND via quadrature of
       CIPOS and CILEN, since CILEN (material-length uncertainty) is
       undirected and could extend toward either breakpoint.
    4. For INS: CIPOS unchanged, CIEND = CIPOS (both breakpoints at POS)
    5. Drops raw tuple columns

    Args:
        cutesv_df: DataFrame of CuteSV non-BND variant calls
        caller_params: Caller-specific parameters (defaults to CUTESV_PARAMS)

    Returns:
        Preprocessed copy of the input DataFrame
    """
    if caller_params is None:
        caller_params = _caller_params.CUTESV_PARAMS

    df = cutesv_df.copy()

    df.loc[:, 'INFO_CHR2'] = df['CHROM']

    if len(df) > 0:
        # DUP/INV records may lack CIPOS/CILEN entirely; pysam returns
        # None which becomes NaN after DataFrame expansion.  Fill NaN
        # with 0 so the quadrature produces valid integers.
        cipos_expanded = pd.DataFrame(
            df['INFO_CIPOS'].tolist(),
            index=df.index,
            columns=['_cipos_l', '_cipos_r'],
            dtype=float,
        ).fillna(0)
        cilen_expanded = pd.DataFrame(
            df['INFO_CILEN'].tolist(),
            index=df.index,
            columns=['_cilen_l', '_cilen_r'],
            dtype=float,
        ).fillna(0)

        ins_mask = df['INFO_SVTYPE'] == 'INS'
        non_ins = ~ins_mask

        # For INS: CIPOS and CIEND are just the native CIPOS (END = POS)
        cipos_l = cipos_expanded['_cipos_l'].copy()
        cipos_r = cipos_expanded['_cipos_r'].copy()
        ciend_l = cipos_expanded['_cipos_l'].copy()
        ciend_r = cipos_expanded['_cipos_r'].copy()

        # For DEL/DUP/INV: quadrature of CIPOS and CILEN for both breakpoints
        if non_ins.any():
            quad = np.ceil(np.sqrt(
                cipos_expanded.loc[non_ins, '_cipos_l'].values ** 2
                + cilen_expanded.loc[non_ins, '_cilen_l'].values ** 2
            )).astype(int)
            cipos_l.loc[non_ins] = -quad
            ciend_l.loc[non_ins] = -quad

            quad_r = np.ceil(np.sqrt(
                cipos_expanded.loc[non_ins, '_cipos_r'].values ** 2
                + cilen_expanded.loc[non_ins, '_cilen_r'].values ** 2
            )).astype(int)
            cipos_r.loc[non_ins] = quad_r
            ciend_r.loc[non_ins] = quad_r

        df.loc[:, 'INFO_CIPOS_l'] = cipos_l.values
        df.loc[:, 'INFO_CIPOS_r'] = cipos_r.values
        df.loc[:, 'INFO_CIEND_l'] = ciend_l.values
        df.loc[:, 'INFO_CIEND_r'] = ciend_r.values
    else:
        for col in ['INFO_CIPOS_l', 'INFO_CIPOS_r', 'INFO_CIEND_l', 'INFO_CIEND_r']:
            df[col] = pd.Series(dtype=float)

    df = df.drop(columns=['INFO_CIPOS', 'INFO_CILEN'], errors='ignore')

    return cutesv_preprocess_common(df, caller_params)


def cutesv_preprocess_bnds_only(
    cutesv_df: pd.DataFrame,
    caller_params: dict | None = None
) -> pd.DataFrame:
    """Preprocessing for CuteSV BND entries.

    BNDs lack CIPOS/CILEN in the VCF; infers symmetric CIPOS from
    PRECISE/IMPRECISE flags (0 for precise, 50bp for imprecise).

    Args:
        cutesv_df: DataFrame of CuteSV BND variant calls
        caller_params: Caller-specific parameters (defaults to CUTESV_PARAMS)

    Returns:
        Preprocessed copy of the input DataFrame
    """
    if caller_params is None:
        caller_params = _caller_params.CUTESV_PARAMS

    df = cutesv_df.copy()

    precise_mask = df['INFO_PRECISE'].fillna(False).astype(bool)
    ci_val = np.where(precise_mask, 0, _CUTESV_IMPRECISE_CI)
    df.loc[:, 'INFO_CIPOS_l'] = -ci_val
    df.loc[:, 'INFO_CIPOS_r'] = ci_val.copy()

    df = df.drop(columns=['INFO_CIPOS', 'INFO_CILEN'], errors='ignore')

    return cutesv_preprocess_common(df, caller_params)


# SVABA

_SVABA_IMPRECISE_CI = 50

_SVABA_EVDNC_VALUES = ['ASSMB', 'ASDIS', 'DSCRD', 'TSI_L', 'TSI_G']


def svaba_preprocess(
    svaba_df: pd.DataFrame,
    caller_params: dict | None = None
) -> pd.DataFrame:
    """Preprocess SvABA BND entries before bnd_to_bp.

    1. Derives INFO_EVENT from the ID field (strips :1/:2 suffix)
    2. Infers symmetric CIPOS from the IMPRECISE flag
    3. Computes INFO_HOMLEN from INFO_HOMSEQ length
    4. One-hot encodes EVDNC into boolean CUSTOM_ columns
    5. Splits SAMPLE_PL tuple into three scalar columns

    Args:
        svaba_df: DataFrame of SvABA BND variant calls
        caller_params: Caller-specific parameters (defaults to SVABA_PARAMS)

    Returns:
        Preprocessed copy of the input DataFrame
    """
    if caller_params is None:
        caller_params = _caller_params.SVABA_PARAMS

    df = svaba_df.copy()

    df.loc[:, 'INFO_EVENT'] = df['ID'].str.rsplit(':', n=1).str[0]

    imprecise_mask = df['INFO_IMPRECISE'].fillna(False).astype(bool)
    ci_val = np.where(imprecise_mask, _SVABA_IMPRECISE_CI, 0)
    df.loc[:, 'INFO_CIPOS_l'] = -ci_val
    df.loc[:, 'INFO_CIPOS_r'] = ci_val.copy()

    if 'INFO_HOMSEQ' in df.columns:
        df.loc[:, 'INFO_HOMLEN'] = df['INFO_HOMSEQ'].apply(
            lambda x: len(x) if isinstance(x, str) else 0
        )
        df = df.drop(columns=['INFO_HOMSEQ'])

    if 'INFO_EVDNC' in df.columns:
        for val in _SVABA_EVDNC_VALUES:
            df.loc[:, f'{CUSTOM_PREFIX}EVDNC_{val}'] = (df['INFO_EVDNC'] == val).astype(int)
        df = df.drop(columns=['INFO_EVDNC'])

    if 'SAMPLE_PL' in df.columns:
        pl_cols = ['SAMPLE_PL_0', 'SAMPLE_PL_1', 'SAMPLE_PL_2']
        if len(df) > 0:
            pl_expanded = pd.DataFrame(
                df['SAMPLE_PL'].tolist(),
                index=df.index,
                columns=pl_cols,
                dtype=float,
            )
            df.loc[:, pl_cols] = pl_expanded.values
        else:
            for col in pl_cols:
                df[col] = pd.Series(dtype=float)
        df = df.drop(columns=['SAMPLE_PL'])

    return df


def svaba_post_bps(bp_df: pd.DataFrame) -> pd.DataFrame:
    """Post-process SvABA breakpoints after bnd_to_bp conversion.

    1. Unifies per-breakend QUAL into a single column (min of the two).
    2. Unifies per-breakend FILTER_PASS into a single column (AND logic).

    Args:
        bp_df: DataFrame of SvABA breakpoints from bnd_to_bp

    Returns:
        DataFrame with unified QUAL and FILTER_PASS columns
    """
    if 'QUAL' not in bp_df.columns:
        try:
            bp_df.loc[:, 'QUAL'] = np.min(
                [bp_df['QUAL_bnd1'], bp_df['QUAL_bnd2']], axis=0
            )
        except KeyError:
            raise KeyError('No QUAL column found in SvABA breakpoints')

    if 'FILTER_PASS' not in bp_df.columns and \
            'FILTER_PASS_bnd1' in bp_df.columns and \
            'FILTER_PASS_bnd2' in bp_df.columns:
        bp_df.loc[:, 'FILTER_PASS'] = (
            bp_df['FILTER_PASS_bnd1'].astype(bool) &
            bp_df['FILTER_PASS_bnd2'].astype(bool)
        )
        bp_df = bp_df.drop(columns=['FILTER_PASS_bnd1', 'FILTER_PASS_bnd2'])

    return bp_df


# SMOOVE / LUMPY

# smoove (LUMPY) declares these INFO fields as Number=. so pysam returns them as
# 1-tuples (e.g. (20,)) rather than scalars. Verified against real output that
# none ever carry more than one value, so unwrapping via [0] is lossless. Unwrap
# before any numeric/string use (e.g. INFO_SVLEN.astype(float), qual_metric=INFO_SU).
_SMOOVE_TUPLE_INFO_FIELDS = [
    'INFO_SVLEN',
    'INFO_MATEID',
    'INFO_SU',
    'INFO_PE',
    'INFO_SR',
]


def _smoove_unwrap_tuple_info(df: pd.DataFrame) -> pd.DataFrame:
    """Unwrap smoove's Number=. INFO fields from 1-tuples to scalars.

    pysam returns Number=. fields as tuples; smoove emits a single value for
    each, so the first element is taken. Missing values (None) pass through.
    """
    for col in _SMOOVE_TUPLE_INFO_FIELDS:
        if col in df.columns:
            df.loc[:, col] = df[col].map(
                lambda x: x[0] if isinstance(x, (tuple, list)) else x
            )
    return df


def smoove_preprocess_common(
    smoove_df: pd.DataFrame,
    caller_params: dict | None = None
) -> pd.DataFrame:
    """Common preprocessing for all smoove entries (BND and non-BND).

    1. Unwraps Number=. INFO fields (SVLEN, MATEID, SU, PE, SR) from 1-tuples
       to scalars.
    2. Casts support counts (SU, PE, SR) to numeric.
    3. Converts negative SVLEN for DEL records to positive; coerces missing
       SVLEN (BND) to 0 so the column stays int.

    Args:
        smoove_df: DataFrame of smoove variant calls
        caller_params: Caller-specific parameters (defaults to SMOOVE_PARAMS)

    Returns:
        Preprocessed copy of the input DataFrame
    """
    if caller_params is None:
        caller_params = _caller_params.SMOOVE_PARAMS

    df = smoove_df.copy()

    df = _smoove_unwrap_tuple_info(df)

    for col in ['INFO_SU', 'INFO_PE', 'INFO_SR']:
        if col in df.columns:
            df.loc[:, col] = pd.to_numeric(df[col], errors='coerce')

    # smoove reports negative SVLEN for deletions; convert to positive for
    # consistency. SVLEN is absent for BND records; coerce NaN to 0 so the
    # column stays int.
    df.loc[:, 'INFO_SVLEN'] = df.loc[:, 'INFO_SVLEN'].astype(float).fillna(0)
    del_mask = df['INFO_SVTYPE'] == 'DEL'
    df.loc[del_mask, 'INFO_SVLEN'] = df.loc[del_mask, 'INFO_SVLEN'].abs()
    df.loc[:, 'INFO_SVLEN'] = df.loc[:, 'INFO_SVLEN'].astype(int)

    return df


def smoove_preprocess_evts_only(
    smoove_df: pd.DataFrame,
    caller_params: dict | None = None
) -> pd.DataFrame:
    """Preprocessing for smoove non-BND event entries (DEL, DUP, INV).

    1. Sets INFO_CHR2 to CHROM (smoove events are intrachromosomal and smoove
       emits no CHR2 field).
    2. Splits native INFO_CIPOS / INFO_CIEND tuples into _l / _r integers.
    3. Sets INFO_EVENT to the variant ID (event_to_bnd pairs breakends by it).

    Args:
        smoove_df: DataFrame of smoove non-BND variant calls
        caller_params: Caller-specific parameters (defaults to SMOOVE_PARAMS)

    Returns:
        Preprocessed copy of the input DataFrame
    """
    if caller_params is None:
        caller_params = _caller_params.SMOOVE_PARAMS

    df = smoove_df.copy()

    df.loc[:, 'INFO_CHR2'] = df['CHROM']

    df.loc[:, ['INFO_CIPOS_l', 'INFO_CIPOS_r']] = sv_utils.split_cipos(df['INFO_CIPOS'])
    df.loc[:, ['INFO_CIEND_l', 'INFO_CIEND_r']] = sv_utils.split_ciend(df['INFO_CIEND'])
    df = df.drop(columns=['INFO_CIPOS', 'INFO_CIEND'])

    df.loc[:, 'INFO_EVENT'] = df['ID']

    return smoove_preprocess_common(df, caller_params)


def smoove_preprocess_bnds_only(
    smoove_df: pd.DataFrame,
    caller_params: dict | None = None
) -> pd.DataFrame:
    """Preprocessing for smoove BND entries.

    BNDs go straight to bnd_to_bp (no event_to_bnd), which pairs breakends by
    INFO_EVENT. smoove BND IDs are '<event>_1' / '<event>_2' sharing a common
    numeric EVENT, so INFO_EVENT is derived from the ID by stripping the mate
    suffix. Each breakend's own positional uncertainty is its INFO_CIPOS.

    Args:
        smoove_df: DataFrame of smoove BND variant calls
        caller_params: Caller-specific parameters (defaults to SMOOVE_PARAMS)

    Returns:
        Preprocessed copy of the input DataFrame
    """
    if caller_params is None:
        caller_params = _caller_params.SMOOVE_PARAMS

    df = smoove_df.copy()

    df.loc[:, ['INFO_CIPOS_l', 'INFO_CIPOS_r']] = sv_utils.split_cipos(df['INFO_CIPOS'])
    df = df.drop(columns=['INFO_CIPOS', 'INFO_CIEND'], errors='ignore')

    df.loc[:, 'INFO_EVENT'] = df['ID'].str.rsplit('_', n=1).str[0]

    return smoove_preprocess_common(df, caller_params)


# ---------------------------------------------------------------------------
# Recipe registry for vcf_df_to_raw_bps.
#
# Keyed by the `source` field of the caller's CALLSET_PARAM_DICT entry, so
# variants of a caller that share a source (e.g. svaba and svaba_H) share one
# recipe. Per-variant differences live in the CALLSET_PARAM_DICT entries.
# ---------------------------------------------------------------------------

class ParseRecipe(TypedDict):
    """How vcf_df_to_raw_bps turns one caller's parsed VCF into raw breakpoints.

    Steps run in this order; an omitted step is skipped.
      preprocess_all  - whole DataFrame, before the BND/event split.
      bnd_svtypes     - split rule: None = all rows are BNDs (no event_to_bnd);
                        [] = all rows are events; list = rows whose
                        INFO_SVTYPE is listed are BNDs, the rest are events.
      preprocess_bnds - the BND rows (Delly's mate-record split happens here).
      preprocess_evts - the event rows, before event_to_bnd.
      force_diff_cols - passed to bnd_to_bp: columns always suffixed _bnd1/_bnd2.
      post_bps        - the breakpoint DataFrame after bnd_to_bp.

    The preprocess_* steps are called as f(df, caller_params=...);
    post_bps as f(bp_df).
    """
    bnd_svtypes: list[str] | None
    preprocess_all: NotRequired[Callable[..., pd.DataFrame]]
    preprocess_bnds: NotRequired[Callable[..., pd.DataFrame]]
    preprocess_evts: NotRequired[Callable[..., pd.DataFrame]]
    force_diff_cols: NotRequired[list[str]]
    post_bps: NotRequired[Callable[[pd.DataFrame], pd.DataFrame]]


_CIPOS_LR = ['INFO_CIPOS_l', 'INFO_CIPOS_r']

parse_recipes: dict[str, ParseRecipe] = {
    'gridss': ParseRecipe(
        bnd_svtypes=None,
        preprocess_all=gridss_preprocess,
        post_bps=gridss_post_bps,
    ),
    'manta_candidate': ParseRecipe(
        bnd_svtypes=['BND', 'INV'],
        preprocess_bnds=manta_preprocess_bnds_only,
        preprocess_evts=manta_preprocess_evts_only,
        force_diff_cols=_CIPOS_LR,
        post_bps=manta_post_bps,
    ),
    'manta_diploid': ParseRecipe(
        bnd_svtypes=['BND', 'INV'],
        preprocess_bnds=manta_preprocess_bnds_only,
        preprocess_evts=manta_preprocess_evts_only,
        force_diff_cols=_CIPOS_LR,
        post_bps=manta_post_bps,
    ),
    'delly': ParseRecipe(
        bnd_svtypes=['BND', 'TRA'],
        preprocess_bnds=delly_preprocess_bnds_only,
        preprocess_evts=delly_preprocess_evts_only,
        force_diff_cols=_CIPOS_LR,
    ),
    'sniffles': ParseRecipe(
        bnd_svtypes=['BND'],
        preprocess_bnds=sniffles_preprocess_bnds_only,
        preprocess_evts=sniffles_preprocess_evts_only,
        force_diff_cols=_CIPOS_LR,
    ),
    'sniffles_207': ParseRecipe(
        bnd_svtypes=['BND'],
        preprocess_bnds=sniffles_preprocess_bnds_only,
        preprocess_evts=sniffles_preprocess_evts_only,
        force_diff_cols=_CIPOS_LR,
    ),
    'survivor_short': ParseRecipe(
        bnd_svtypes=[],
        preprocess_evts=survivor_preprocess_evts,
    ),
    'survivor_long': ParseRecipe(
        bnd_svtypes=[],
        preprocess_evts=survivor_preprocess_evts,
    ),
    'survivor_all': ParseRecipe(
        bnd_svtypes=[],
        preprocess_evts=survivor_preprocess_evts,
    ),
    'severus': ParseRecipe(
        bnd_svtypes=['BND'],
        preprocess_bnds=severus_preprocess_bnds_only,
        preprocess_evts=severus_preprocess_evts_only,
        force_diff_cols=_CIPOS_LR,
    ),
    'cutesv': ParseRecipe(
        bnd_svtypes=['BND'],
        preprocess_bnds=cutesv_preprocess_bnds_only,
        preprocess_evts=cutesv_preprocess_evts_only,
        force_diff_cols=_CIPOS_LR,
    ),
    'svaba': ParseRecipe(
        bnd_svtypes=None,
        preprocess_all=svaba_preprocess,
        force_diff_cols=_CIPOS_LR,
        post_bps=svaba_post_bps,
    ),
    'smoove': ParseRecipe(
        bnd_svtypes=['BND'],
        preprocess_bnds=smoove_preprocess_bnds_only,
        preprocess_evts=smoove_preprocess_evts_only,
        force_diff_cols=_CIPOS_LR,
    ),
}


def _log(msg: str) -> None:
    """Print a message with an ISO 8601 timestamp prefix."""
    print(f"{datetime.now().astimezone().isoformat()} {msg}")


def vcf_df_to_raw_bps(
    vcf_df: pd.DataFrame,
    caller_params: dict,
    verbose: bool = False,
) -> pd.DataFrame:
    """Turn one caller's parsed VCF DataFrame into raw breakpoints.

    The recipe is parse_recipes[caller_params['source']]; see ParseRecipe for
    the steps. *vcf_df* is not modified.

    Args:
        vcf_df: The caller's VCF as parsed by vcfparse.parse_vcf_to_df.
        caller_params: The caller's CALLSET_PARAM_DICT entry.
        verbose: If True, print progress messages.

    Returns:
        DataFrame of breakpoints.

    Raises:
        KeyError: If no recipe exists for caller_params['source'].
    """
    source = caller_params['source']
    try:
        recipe = parse_recipes[source]
    except KeyError:
        raise KeyError(
            f"No parse recipe for source {source!r}; known: {sorted(parse_recipes)}"
        ) from None

    df = vcf_df.copy()

    if preprocess_all := recipe.get('preprocess_all'):
        if verbose:
            _log(f"Preprocessing all {source} variants")
        df = preprocess_all(df, caller_params=caller_params)

    bnd_svtypes = recipe['bnd_svtypes']

    if bnd_svtypes is None:
        # Entire VCF is BND -- go straight to bnd_to_bp
        bnd_df = df
    else:
        if bnd_svtypes:
            bnd_mask = df['INFO_SVTYPE'].isin(bnd_svtypes)
            bnd_df = df[bnd_mask]
            evt_df = df[~bnd_mask]
        else:
            bnd_df = pd.DataFrame(columns=df.columns)
            evt_df = df

        preprocess_bnds = recipe.get('preprocess_bnds')
        if preprocess_bnds and len(bnd_df) > 0:
            if verbose:
                _log(f"Preprocessing {source} BND variants")
            bnd_df = preprocess_bnds(bnd_df, caller_params=caller_params)

        preprocess_evts = recipe.get('preprocess_evts')
        if preprocess_evts and len(evt_df) > 0:
            if verbose:
                _log(f"Preprocessing {source} event variants")
            evt_df = preprocess_evts(evt_df, caller_params=caller_params)

        if len(evt_df) > 0:
            if verbose:
                _log(f"Converting {source} events to breakends")
            evt_bnd_df = evts.event_to_bnd(evt_df, bnd2_pos=caller_params['bnd2_pos'])
            bnd_df = pd.concat([bnd_df, evt_bnd_df], axis=0) if len(bnd_df) > 0 else evt_bnd_df

    if verbose:
        _log(f"Converting {source} breakends to breakpoints")
    force_diff = recipe.get('force_diff_cols')
    bp_df = bnds.bnd_to_bp(bnd_df, force_diff_cols=force_diff) if force_diff else bnds.bnd_to_bp(bnd_df)

    if post_bps := recipe.get('post_bps'):
        bp_df = post_bps(bp_df)

    return bp_df


def gnomad_process(
    df: pd.DataFrame,
    caller_params: dict,
    svtypes_retain: Union[str, List[str]] = 'simple',
    verbose: bool = False,
) -> pd.DataFrame:
    """Top-level processing for a gnomAD-family ground-truth callset.

    Takes an unprocessed DataFrame (a parsed gnomAD VCF, optionally already
    record-level filtered by the caller) and returns a breakpoint DataFrame
    with CI/EVENT columns populated and events collapsed to bp form.

    Composes `gnomad_preprocess` -> `gnomad_event_to_bnd` -> `bnds.bnd_to_bp`
    with the CIPOS columns marked as force_diff so paired breakends preserve
    their per-side CI.
    """
    df = gnomad_preprocess(
        df,
        svtypes_retain=svtypes_retain,
        filter_pass_only=False,
        caller_params=caller_params,
        verbose=verbose,
    )
    if verbose:
        print("Converting events to breakends")
    bnd_df = gnomad_event_to_bnd(df, caller_params=caller_params)
    if verbose:
        print("Converting breakends to breakpoints")
    bp_df = bnds.bnd_to_bp(bnd_df, force_diff_cols=['INFO_CIPOS_l', 'INFO_CIPOS_r'])
    return bp_df


# ---------------------------------------------------------------------------
# ground_truth_recipes: registry of processing recipes per ground-truth family.
#
# Sibling of parse_recipes. Keyed by *family* name (matching the `source`
# field of the corresponding entries in _caller_params.CALLSET_PARAM_DICT), so
# a single recipe serves every variant of a family (e.g. gnomad, gnomad_pass,
# gnomad_filtered all have source='gnomad' and share the 'gnomad' recipe).
#
# Adding a new ground-truth family: register a recipe here plus one or more
# CALLSET_PARAM_DICT entries in _caller_params.py whose `source` matches this
# key. Per-variant differences (CI dictionaries, filter defaults, etc.) live
# in the CALLSET_PARAM_DICT entries, not in the recipe.
#
# Keys per entry:
#   process - callable(df, caller_params, svtypes_retain='simple', verbose=False)
#             takes the parsed+filtered VCF DataFrame and returns the fully
#             processed breakpoint DataFrame (preprocess + event_to_bnd +
#             bnd_to_bp collapsed into one step).
# ---------------------------------------------------------------------------

ground_truth_recipes: dict[str, dict] = {
    'gnomad': {
        'process': gnomad_process,
    },
}