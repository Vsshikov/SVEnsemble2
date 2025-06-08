import pandas as pd
from typing import Union, Tuple, Optional, List, Any, Literal, Dict
import copy
import pysam

from . import events as evts
from . import breakends as bnds
from . import utils as sv_utils
from . import _caller_params as caller_params

def gridss_preprocess(
    gridss_callset: pd.DataFrame,
    inplace: bool = False,
    verbose: bool = False
) -> Optional[pd.DataFrame]:
    """Preprocess GRIDSS callset by splitting CIPOS values and cleaning columns.
    
    This function processes GRIDSS variant calls by:
    1. Splitting the INFO_CIPOS field into separate left and right columns
    2. Removing the original INFO_CIPOS column to avoid redundancy
    
    Args:
        gridss_callset: DataFrame containing GRIDSS variant calls with INFO_CIPOS field
        inplace: If True, modify input DataFrame in-place. If False, return new copy.
        verbose: If True, print progress messages during processing
        
    Returns:
        If inplace=False: Processed DataFrame with split CIPOS columns
        If inplace=True: None (modifies input DataFrame in place)
        
    Raises:
        KeyError: If INFO_CIPOS column is missing from input DataFrame
    """
    if verbose:
        print("Preprocessing GRIDSS callset...")
    
    # Create working copy if not modifying in place
    if not inplace:
        gridss_callset_ = gridss_callset.copy()
    else:
        gridss_callset_ = gridss_callset

    # Split CIPOS values into left and right confidence intervals
    if verbose:
        print("Splitting CIPOS values...")
    gridss_callset_.loc[:, ['INFO_CIPOS_l', 'INFO_CIPOS_r']] = (
        sv_utils.split_cipos(gridss_callset_['INFO_CIPOS'])
    )
    
    # Remove original CIPOS column to avoid redundancy
    if verbose:
        print("Dropping INFO_CIPOS column...")
    gridss_callset_ = gridss_callset_.drop(columns=['INFO_CIPOS'])

    # Return processed DataFrame or None based on inplace parameter
    if not inplace:
        if verbose:
            print("Returning processed callset...")
        return gridss_callset_
    else:
        if verbose:
            print("Successfully modified input callset...")
        return None


def gnomad_split_evidence(
    evidence_series: pd.Series,
    evidence_types: list[str] | None = None
) -> pd.DataFrame:
    """Split gnomAD evidence string into boolean columns for each evidence type.
    
    Args:
        evidence_series: Series containing evidence strings (e.g. 'SR,PE')
        evidence_types: List of evidence types to check for. Defaults to ['SR','PE','RD']
        
    Returns:
        DataFrame with boolean columns for each evidence type indicating presence
    """
    if evidence_types is None:
        evidence_types = ['SR', 'PE', 'RD']
        
    evidence_df = pd.DataFrame(index=evidence_series.index, columns=evidence_types)
    for evidence_type in evidence_types:
        evidence_df[evidence_type] = evidence_series.apply(lambda x: evidence_type in x)
    return evidence_df


def gnomad_get_best_evidence(
    evidence_series: pd.Series,
    evidence_types: list[str] | None = None
) -> pd.Series:
    """Get the highest priority evidence type for each variant.
    
    Args:
        evidence_series: Series containing evidence strings
        evidence_types: List of evidence types in priority order. Defaults to ['SR','PE','RD']
        
    Returns:
        Series containing the highest priority evidence type for each variant
    """
    if evidence_types is None:
        evidence_types = ['SR', 'PE', 'RD']
        
    evidence_df = gnomad_split_evidence(evidence_series, evidence_types)
    return evidence_df.idxmax(axis=1)


def gnomad_get_cipos(
    evidence_series: pd.Series,
    evidence_types: list[str] | None = None,
    cipos_dict: dict[str, tuple[int, int]] | None = None
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
    if evidence_types is None:
        evidence_types = ['SR', 'PE', 'RD']
    if cipos_dict is None:
        cipos_dict = caller_params.gnomad_method_cipos
        
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
    inplace: bool = False,
    verbose: bool = False
) -> pd.DataFrame | None:
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
        inplace: If True, modify input DataFrame in place
        verbose: If True, print processing information
        
    Returns:
        If inplace=False: Processed DataFrame
        If inplace=True: None (modifies input in place)
        
    Raises:
        NotImplementedError: If unsupported SV types requested
    """
    if verbose:
        print("Preprocessing gnomAD callset...")
    # Define supported SV types
    svtypes_supported = caller_params.gnomad_supported_svtypes
    if verbose:
        print(f"SV types currently supported: {svtypes_supported}")
    
    # Define evidence CI sizes
    if evidence_cipos == 'default':
        evidence_cipos = caller_params.gnomad_method_cipos
    if verbose:
        print(f"Using default evidence CIPOS: {evidence_cipos}")
    
    # Validate and set viable SV types
    if svtypes_retain == 'simple':
        svtypes_retain = svtypes_supported
        if verbose:
            print(f"Using default viable SV types: {svtypes_retain}")
    elif all(svtype in svtypes_supported for svtype in svtypes_retain):
        svtypes_retain = svtypes_retain
        if verbose:
            print(f"Using specified viable SV types: {svtypes_retain}")
    else:
        raise NotImplementedError(f"Supported viable SV types: {svtypes_supported}")

    # Create copies if not modifying in place
    if not inplace:
        gnomad_callset_ = gnomad_callset.copy()
    else:
        gnomad_callset_ = gnomad_callset

    # Filter to desired SV types
    if verbose:
        print(f"Filtering to specified SV types: {svtypes_retain}")
    gnomad_callset_ = gnomad_callset_[
        gnomad_callset_['INFO_SVTYPE'].isin(svtypes_retain)
    ]

    # Some variants may lack CHR2 field
    # Set it to CHROM for them
    gnomad_callset_.loc[
        gnomad_callset_['INFO_CHR2'].isna(),
        'INFO_CHR2'
    ] = gnomad_callset_['CHROM']
    
    # Apply FILTER=PASS filter if requested
    if filter_pass_only:
        if verbose:
            print("Filtering to PASS variants only")
        gnomad_callset_ = gnomad_callset_[
            gnomad_callset_['FILTER_PASS'].astype(bool)
        ]

    # Set confidence intervals based on evidence type
    if verbose:
        print("Setting confidence intervals based on evidence type")
    gnomad_callset_.loc[:, ['INFO_CIPOS_l', 'INFO_CIPOS_r']] = gnomad_get_cipos(
        gnomad_get_best_evidence(gnomad_callset_['INFO_EVIDENCE']),
        evidence_types=evidence_cipos.keys(),
        cipos_dict=evidence_cipos
    )

    # Set EVENT field to match variant ID
    if verbose:
        print("Setting EVENT field to match variant ID")
    gnomad_callset_.loc[:, 'INFO_EVENT'] = gnomad_callset_['ID']

    if not inplace:
        if verbose:
            print("Returning processed callset")
        return gnomad_callset_
    else:
        if verbose:
            print("Successfully modified input callset")
        gnomad_callset = gnomad_callset_
        return None
    

def gnomad_event_to_bnd(
    gnomad_callset: pd.DataFrame,
    inplace: bool = False
) -> Optional[pd.DataFrame]:
    """Convert gnomAD event-based SV records to breakend (BND) format.
    
    Args:
        gnomad_callset: DataFrame containing gnomAD structural variants
        inplace: If True, modify input DataFrame in place. If False, return a copy
        add_encoding: If True, add BND encoding information to the output
    
    Returns:
        Modified DataFrame if inplace=False, None if inplace=True
    """
    if not inplace:
        gnomad_callset_local = gnomad_callset.copy()
    else:
        gnomad_callset_local = gnomad_callset

    # Convert events to breakend format
    gnomad_callset_local = evts.event_to_bnd(
        gnomad_callset_local
    )

    if not inplace:
        return gnomad_callset_local
    else:
        gnomad_callset = gnomad_callset_local
        return None
