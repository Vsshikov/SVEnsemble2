from . import bndbpconv as bndc

def gridss_to_bp(gridss_callset):
    """Convert GRIDSS breakend format to breakpoint pairs
    
    Args:
        gridss_callset_raw: DataFrame with GRIDSS breakend calls
        
    Returns:
        DataFrame with breakpoint pairs including:
        - Original fields split into bnd1/bnd2
        - bnd_pos_encoding: Encoding of breakpoint orientations 
        - bndtype: Combined orientation encoding for the pair
    """
    gridss_callset_bps = gridss_callset.copy()
    # Add breakend position encoding
    gridss_callset_bps.loc[:, 'bnd_pos_encoding'] = bndc.encode_bnd_alts(gridss_callset_bps['ALT'])
    
    # Split CIPOS into left/right bounds
    gridss_callset_cipos = gridss_callset_bps['INFO_CIPOS'].str.split(',', expand=True).fillna('0').astype(int)
    gridss_callset_ciposl = gridss_callset_cipos.loc[:, 0]
    gridss_callset_ciposr = gridss_callset_cipos.loc[:, 1]
    
    # Add as separate columns and drop original
    gridss_callset_bps.loc[:, 'INFO_CIPOS_l'] = gridss_callset_ciposl
    gridss_callset_bps.loc[:, 'INFO_CIPOS_r'] = gridss_callset_ciposr
    gridss_callset_bps = gridss_callset_bps.drop(columns=['INFO_CIPOS'])
    
    # Convert breakends to breakpoints
    gridss_callset_bps = bndc.bnd_to_bp(gridss_callset_bps)
    gridss_callset_bps.loc[:, 'bndtype'] = gridss_callset_bps['bnd_pos_encoding_bnd1'].astype(str) + gridss_callset_bps['bnd_pos_encoding_bnd2'].astype(str)
    
    return gridss_callset_bps
