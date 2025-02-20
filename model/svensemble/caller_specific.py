import pandas as pd
from typing import Union, Tuple, Optional, List, Any, Literal, Dict
import copy


from . import bndbpconv as bndc
from . import utils as sv_utils

def gridss_preprocess(
    gridss_callset: pd.DataFrame,
    gridss_meta: dict,
    inplace: bool = False
) -> Union[Tuple[pd.DataFrame, dict], None]:
    """Preprocess GRIDSS callset by splitting CIPOS values and converting column types.
    
    Args:
        gridss_callset: DataFrame containing GRIDSS variant calls
        gridss_meta: Dictionary containing GRIDSS metadata
        inplace: If True, modify input DataFrame in-place. If False, return new copy.
        
    Returns:
        If inplace=False, returns tuple of (processed_callset, processed_meta)
        If inplace=True, returns None after modifying inputs in-place
    """
    if not inplace:
        gridss_callset_local = gridss_callset.copy()
        gridss_meta_local = copy.deepcopy(gridss_meta)
    else:
        gridss_callset_local = gridss_callset
        gridss_meta_local = gridss_meta

    # Split CIPOS string into left and right components
    gridss_callset_cipos = (gridss_callset_local['INFO_CIPOS']
                           .str.split(',', expand=True)
                           .fillna('0')
                           .astype(int))
    
    gridss_callset_ciposl = gridss_callset_cipos.loc[:, 0]
    gridss_callset_ciposr = gridss_callset_cipos.loc[:, 1]

    # Add separated CIPOS columns
    gridss_callset_local, gridss_meta_local = sv_utils.add_vcf_col(
        gridss_callset_local, 
        gridss_meta_local,
        'INFO_CIPOS_l',
        gridss_callset_ciposl,
        description='Left CIPOS'
    )
    
    gridss_callset_local, gridss_meta_local = sv_utils.add_vcf_col(
        gridss_callset_local,
        gridss_meta_local, 
        'INFO_CIPOS_r',
        gridss_callset_ciposr,
        description='Right CIPOS'
    )

    # Clean up and standardize columns
    sv_utils.delete_mult_val_cols(gridss_callset_local, gridss_meta_local, vocal=True)
    sv_utils.convert_col_types(gridss_callset_local, gridss_meta_local)
    
    gridss_callset_local = gridss_callset_local.reset_index(drop=True)

    if not inplace:
        return gridss_callset_local, gridss_meta_local
    else:
        gridss_callset = gridss_callset_local
        gridss_meta = gridss_meta_local
        return None


def gridss_bnd_to_bp(
    gridss_callset: pd.DataFrame,
    inplace: bool = False,
    add_encoding: bool = True
) -> pd.DataFrame:
    """Convert GRIDSS breakend format to breakpoint format.
    
    Takes a DataFrame of GRIDSS breakend calls and converts them to breakpoints,
    optionally adding orientation encoding information.
    
    Args:
        gridss_callset: DataFrame containing GRIDSS breakend calls
        inplace: If True, modify input DataFrame in place. If False, create copy.
        add_encoding: If True, add breakend orientation encoding columns
        
    Returns:
        DataFrame with breakpoint pairs containing:
            - Original fields split into bnd1/bnd2 columns
            - bnd_pos_encoding_bnd1/2: Encoding of breakpoint orientations
            - bndtype: Combined orientation encoding for the pair
            
    Raises:
        KeyError: If required CIPOS columns are missing from input DataFrame
    """
    # Verify required columns exist
    try:
        gridss_callset[['INFO_CIPOS_l', 'INFO_CIPOS_r']]
    except KeyError:
        raise KeyError(
            "GRIDSS callset must contain INFO_CIPOS_l and INFO_CIPOS_r columns. "
            "Run gridss_preprocess() to add them first."
        )

    # Create copy if not modifying in place
    if not inplace:
        gridss_callset_local = gridss_callset.copy()
    else:
        gridss_callset_local = gridss_callset

    # Convert breakends to breakpoints
    gridss_callset_local = bndc.bnd_to_bp(gridss_callset_local)

    # Add orientation encoding if requested
    if add_encoding:
        gridss_callset_local = sv_utils.add_bnd_encoding(gridss_callset_local, inplace=False)
    
    if not inplace:
        return gridss_callset_local
    else:
        gridss_callset = gridss_callset_local
        return None

    
def gnomad_preprocess(
    gnomad_callset: pd.DataFrame,
    gnomad_meta: pd.DataFrame,
    evidence_ci_sizes: Union[Dict[str, int], Literal['default']] = 'default',
    viable_svtypes: Union[str, List[str]] = 'simple',
    filter_pass_only: bool = True,
    inplace: bool = False
) -> Optional[Tuple[pd.DataFrame, pd.DataFrame]]:
    """Preprocess gnomAD structural variant callset.
    
    Processes a gnomAD SV callset by:
    1. Adding confidence interval sizes based on evidence type
    2. Filtering to specified SV types
    3. Optionally filtering to PASS variants only
    4. Processing evidence and metadata columns
    5. Setting EVENT field to match variant ID
    
    Args:
        gnomad_callset: DataFrame containing gnomAD SV calls
        gnomad_meta: DataFrame containing metadata for gnomAD calls
        evidence_ci_sizes: Dict mapping evidence types to CI sizes in bp, or 'default'
            Default maps SR->10bp, PE->100bp, RD->500bp
        viable_svtypes: Either 'simple' for basic SV types (DEL,DUP,INV,INS),
            or list of specific SV types to include
        filter_pass_only: If True, keep only variants with FILTER=PASS
        inplace: If True, modify input DataFrames in place
        
    Returns:
        If inplace=False: Tuple of (processed_callset, processed_meta)
        If inplace=True: None (modifies inputs in place)
        
    Raises:
        NotImplementedError: If unsupported SV types requested
    """
    # Define supported SV types
    svtypes_supported = ['DEL', 'DUP', 'INV', 'INS']
    
    # Validate and set viable SV types
    if viable_svtypes == 'simple':
        viable_svtypes = svtypes_supported
    elif all(svtype in svtypes_supported for svtype in viable_svtypes):
        viable_svtypes = viable_svtypes
    else:
        raise NotImplementedError(f"Supported viable SV types: {svtypes_supported}")
    
    # Set default confidence interval sizes if not provided
    if evidence_ci_sizes == 'default':
        evidence_ci_sizes = {
            'SR': 10,   # Split reads - highest resolution
            'PE': 100,  # Paired-end reads - medium resolution  
            'RD': 500   # Read depth - lowest resolution
        }

    # Create copies if not modifying in place
    if not inplace:
        gnomad_callset_local = gnomad_callset.copy()
        gnomad_meta_local = copy.deepcopy(gnomad_meta)
    else:
        gnomad_callset_local = gnomad_callset
        gnomad_meta_local = gnomad_meta

    # Filter to desired SV types
    gnomad_callset_local = gnomad_callset_local[
        gnomad_callset_local['INFO_SVTYPE'].isin(viable_svtypes)
    ]
    
    # Apply FILTER=PASS filter if requested
    if filter_pass_only:
        gnomad_callset_local = gnomad_callset_local[
            gnomad_callset_local['FILTER_PASS'].astype(bool)
        ]

    # Process evidence column
    sv_utils.split_evidence_col(gnomad_callset_local, gnomad_meta_local)

    # Set confidence intervals based on evidence type
    match_margins = gnomad_callset_local['INFO_EVIDENCE_BEST'].map(evidence_ci_sizes)
    gnomad_callset_local.loc[:, 'INFO_CIPOS_l'] = -match_margins
    gnomad_callset_local.loc[:, 'INFO_CIPOS_r'] = match_margins

    sv_utils.delete_mult_val_cols(
        gnomad_callset_local,
        gnomad_meta_local,
        exceptions=['INFO_EVIDENCE'],
        vocal=True
    )

    sv_utils.convert_col_types(gnomad_callset_local, gnomad_meta_local)

    # Set EVENT field to match variant ID
    gnomad_callset_local.loc[:, 'INFO_EVENT'] = gnomad_callset_local['ID']

    if not inplace:
        return gnomad_callset_local, gnomad_meta_local
    else:
        gnomad_callset = gnomad_callset_local
        gnomad_meta = gnomad_meta_local
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
    gnomad_callset_local = bndc.event_to_bnd(
        gnomad_callset_local
    )

    if not inplace:
        return gnomad_callset_local
    else:
        gnomad_callset = gnomad_callset_local
        return None

def gnomad_bnd_to_bp(
    gnomad_callset_bnds: pd.DataFrame,
    inplace: bool = False,
    add_encoding: bool = True
) -> Optional[pd.DataFrame]:
    """Convert gnomAD breakend (BND) records to breakpoint (BP) format.
    
    Args:
        gnomad_callset_bnds: DataFrame containing gnomAD variants in BND format
        inplace: If True, modify input DataFrame in place. If False, return a copy
    
    Returns:
        Modified DataFrame if inplace=False, None if inplace=True
    """
    if not inplace:
        gnomad_callset_local = gnomad_callset_bnds.copy()
    else:
        gnomad_callset_local = gnomad_callset_bnds

    # Convert breakends to breakpoint format
    gnomad_callset_local = bndc.bnd_to_bp(
        gnomad_callset_local,
        force_diff_cols=['INFO_CIPOS_l', 'INFO_CIPOS_r']
    )

    # Optionally add breakend encoding information
    if add_encoding:
        gnomad_callset_local = sv_utils.add_bnd_encoding(
            gnomad_callset_local,
            inplace=False
        )

    if not inplace:
        return gnomad_callset_local
    else:
        gnomad_callset_bnds = gnomad_callset_local
        return None
    
def gnomad_event_to_bp(
    gnomad_callset: pd.DataFrame,
    inplace: bool = False
) -> Optional[pd.DataFrame]:
    """Convert gnomAD event format to breakpoint (BP) format via breakend (BND) format.
    
    This function performs a two-step conversion:
    1. Event format -> Breakend (BND) format
    2. Breakend format -> Breakpoint (BP) format
    
    Args:
        gnomad_callset: DataFrame containing gnomAD variants in event format
        inplace: If True, modify input DataFrame in place. If False, return a copy
        
    Returns:
        Modified DataFrame if inplace=False, None if inplace=True
    """
    gnomad_callset = gnomad_event_to_bnd(gnomad_callset, inplace=inplace)
    gnomad_callset = gnomad_bnd_to_bp(gnomad_callset, inplace=inplace)
    return gnomad_callset


# Dispatch dictionaries for preprocessing and format conversion functions
dispatch_preprocess = {
    'gnomad': gnomad_preprocess,
    'gridss': gridss_preprocess
}

dispatch_to_bp = {
    'gnomad': gnomad_event_to_bp,
    'gridss': gridss_bnd_to_bp
}


def callset_to_bp(
    callset: pd.DataFrame,
    meta: dict[str, Any],
    source: Literal['gnomad', 'gridss']
) -> pd.DataFrame:
    """Convert a variant callset to breakpoint (BP) format.
    
    This function handles the conversion of variant calls from different sources
    into a standardized breakpoint format by:
    1. Preprocessing the data using source-specific functions
    2. Converting to breakpoint format using source-specific converters
    
    Args:
        callset: DataFrame containing variant calls
        meta: Dictionary containing VCF metadata
        source: Variant caller name ('gnomad' or 'gridss' currently supported)
        
    Returns:
        DataFrame in breakpoint format
        
    Raises:
        KeyError: If source is not found in dispatch dictionaries
    """
    callset_bp = dispatch_to_bp[source](
        dispatch_preprocess[source](callset, meta)[0]
    )
    return callset_bp
