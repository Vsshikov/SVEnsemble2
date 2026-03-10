import pandas as pd
import numpy as np
from typing import Union, List, Literal, Dict

from . import events as evts
from . import utils as sv_utils
from . import _caller_params

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
        caller_params = _caller_params.gridss_params

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
        caller_params = _caller_params.gnomad_params
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
        caller_params = _caller_params.gnomad_params
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
                   Defaults to values from caller_params.gnomad_method_cipos

    Returns:
        DataFrame with INFO_CIPOS_l and INFO_CIPOS_r columns containing CI sizes
    """
    if caller_params is None:
        caller_params = _caller_params.gnomad_params
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
        caller_params = _caller_params.gnomad_params

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
        caller_params = _caller_params.gnomad_params

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
        caller_params = _caller_params.manta_candidate_params

    manta_df_ = manta_df.copy()

    # When SVLEN is not applicable, it is set to None.
    # We get rid of NaNs
    manta_df_.loc[:, 'INFO_SVLEN'] = manta_df_.loc[:, 'INFO_SVLEN'].astype(float)
    manta_df_.loc[:, 'INFO_SVLEN'] = manta_df_.loc[:, 'INFO_SVLEN'].fillna('0')
    manta_df_.loc[:, 'INFO_SVLEN'] = manta_df_.loc[:, 'INFO_SVLEN'].astype(int)

    # INFO_HOMLEN is a tuple initially. Manta is not supposed to have multiple HOMLEN values
    # So pick the first value of the tuple
    manta_df_.loc[:, 'INFO_HOMLEN'] = manta_df_.loc[:, 'INFO_HOMLEN'].str[0]

    # Set INFO_CHR2 to CHROM
    manta_df_.loc[:, 'INFO_CHR2'] = manta_df_.loc[:, 'CHROM']

    # Drop INFO_POS2 as it does not contain information
    manta_df_ = manta_df_.drop(columns=['INFO_POS2'])

    if caller_params['parse_format']:
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
    """
    manta_df_ = manta_df.copy()
    manta_df_.loc[:, ['INFO_CIPOS_l', 'INFO_CIPOS_r']] = sv_utils.split_cipos(manta_df_['INFO_CIPOS'])

    cols_to_drop = ['INFO_CIPOS', 'INFO_CIEND']
    manta_df_ = manta_df_.drop(columns=cols_to_drop)
    return manta_preprocess_common(manta_df_, caller_params)


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

def delly_preprocess_bnds_only(delly_df: pd.DataFrame, caller_params: dict|None = None) -> pd.DataFrame:
    """
    Preprocessing for Delly BND entries only.
    """
    delly_df_ = delly_df.copy()
    delly_df_.loc[:, 'INFO_EVENT'] = delly_df_.loc[:, 'ID'].str[:-4]

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
        {f'{base}_l': -stdev_int, f'{base}_r': stdev_int},
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
        caller_params: Caller-specific parameters (defaults to sniffles_params)

    Returns:
        Preprocessed copy of the input DataFrame
    """
    if caller_params is None:
        caller_params = _caller_params.sniffles_params

    sniffles_df_ = sniffles_df.copy()

    sniffles_df_.loc[:, 'INFO_EVENT'] = sniffles_df_['ID']

    sniffles_df_.loc[:, ['INFO_CIPOS_l', 'INFO_CIPOS_r']] = _sniffles_cipos_from_stdev(
        sniffles_df_['INFO_STDEV_POS']
    )

    # Sniffles2 reports negative SVLEN for deletions; convert to positive for consistency
    del_mask = sniffles_df_['INFO_SVTYPE'] == 'DEL'
    sniffles_df_.loc[del_mask, 'INFO_SVLEN'] = sniffles_df_.loc[del_mask, 'INFO_SVLEN'].abs()

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
        caller_params: Caller-specific parameters (defaults to sniffles_params)

    Returns:
        Preprocessed copy of the input DataFrame
    """
    if caller_params is None:
        caller_params = _caller_params.sniffles_params

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
        caller_params: Caller-specific parameters (defaults to sniffles_params)

    Returns:
        Preprocessed copy of the input DataFrame
    """
    if caller_params is None:
        caller_params = _caller_params.sniffles_params

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
        caller_params: Caller-specific parameters (defaults to survivor_params)

    Returns:
        Preprocessed copy of the input DataFrame
    """
    if caller_params is None:
        caller_params = _caller_params.survivor_params

    df = survivor_df.copy()
    df.loc[:, ['INFO_CIPOS_l', 'INFO_CIPOS_r']] = sv_utils.split_cipos(df['INFO_CIPOS'])
    df.loc[:, ['INFO_CIEND_l', 'INFO_CIEND_r']] = sv_utils.split_ciend(df['INFO_CIEND'])
    df = df.drop(columns=['INFO_CIPOS', 'INFO_CIEND'])
    df.loc[:, 'INFO_EVENT'] = df['ID']
    return df


# ---------------------------------------------------------------------------
# Recipe registry for the generic parse_preprocess_vcf dispatcher.
#
# Each entry maps a callset name to its preprocessing recipe:
#   bnd_svtypes   - SV types treated as BNDs for the split.
#                   None  => entire VCF is BND (no split, no event_to_bnd).
#                   []    => entire VCF is events (all go through event_to_bnd).
#                   list  => rows whose INFO_SVTYPE is in the list are BND;
#                            the rest are events.
#   preprocess_all  - applied to entire VCF before the BND/event split.
#   preprocess_bnds - applied to the BND half after the split.
#   preprocess_evts - applied to the event half after the split.
#   force_diff_cols - passed to bnd_to_bp (columns that may differ per breakend).
#   post_bps        - applied to the breakpoint DataFrame after bnd_to_bp.
# ---------------------------------------------------------------------------

parse_recipes: dict[str, dict] = {
    'gridss': {
        'bnd_svtypes': None,
        'preprocess_all': gridss_preprocess,
        'preprocess_bnds': None,
        'preprocess_evts': None,
        'force_diff_cols': None,
        'post_bps': gridss_post_bps,
    },
    'manta_candidate': {
        'bnd_svtypes': ['BND', 'INV'],
        'preprocess_all': None,
        'preprocess_bnds': manta_preprocess_bnds_only,
        'preprocess_evts': manta_preprocess_evts_only,
        'force_diff_cols': ['INFO_CIPOS_l', 'INFO_CIPOS_r'],
        'post_bps': None,
    },
    'manta_diploid': {
        'bnd_svtypes': ['BND', 'INV'],
        'preprocess_all': None,
        'preprocess_bnds': manta_preprocess_bnds_only,
        'preprocess_evts': manta_preprocess_evts_only,
        'force_diff_cols': ['INFO_CIPOS_l', 'INFO_CIPOS_r'],
        'post_bps': None,
    },
    'sniffles': {
        'bnd_svtypes': ['BND'],
        'preprocess_all': None,
        'preprocess_bnds': sniffles_preprocess_bnds_only,
        'preprocess_evts': sniffles_preprocess_evts_only,
        'force_diff_cols': ['INFO_CIPOS_l', 'INFO_CIPOS_r'],
        'post_bps': None,
    },
    'survivor': {
        'bnd_svtypes': [],
        'preprocess_all': None,
        'preprocess_bnds': None,
        'preprocess_evts': survivor_preprocess_evts,
        'force_diff_cols': None,
        'post_bps': None,
    },
}