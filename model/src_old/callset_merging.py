"""
This module contains functions for merging variant callsets from different sources.

The callsets are presumed to be in breakpoint (BP) notation. See `caller_specific.py` 
and `bndbpconv.py` for more details.
"""

import copy
import importlib
import re
import os

import numpy as np
import pandas as pd
import matplotlib as mpl
import pyranges as pr
from scipy.spatial import cKDTree

from ..svensemble import vcfparse as vcfp
from ..svensemble import bndbpconv as bndc
from ..svensemble import utils as sv_utils


def preprocess_bp_df_for_kdtree(bp_df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess breakpoint dataframe for KD-tree by sorting chromosomes."""
    chrom_keys = sv_utils.generate_chrom_key_dict(
        pd.concat([bp_df['CHROM_bnd1'], bp_df['CHROM_bnd2']])
    )
    return bndc.sort_bnd_df(bp_df, chrom_keys=chrom_keys)


def get_chrom_bp_type(bp_df: pd.DataFrame) -> pd.Series:
    """
    Get unique identifier for chromosome position of a breakpoint + its type.
    
    Args:
        bp_df: Breakpoint dataframe containing CHROM_bnd1, CHROM_bnd2 and bp_type columns
        
    Returns:
        Series containing concatenated chromosome and breakpoint type identifiers
    """
    return bp_df['CHROM_bnd1'] + '_' + bp_df['CHROM_bnd2'] + '_' + bp_df['bp_type']


def get_kdtforest(bp_df: pd.DataFrame, unique_chrom_bp_types: np.ndarray) -> dict:
    """Build KD-tree forest from breakpoint positions grouped by chromosome/type."""
    return {
        chrom_bp_type: cKDTree(
            bp_df[bp_df['chrom_bp_type'] == chrom_bp_type][['POS_bnd1', 'POS_bnd2']].to_numpy()
        )
        for chrom_bp_type in unique_chrom_bp_types
    }


def kdtforest_query(
    kdtforest: dict,
    bp_df_query: pd.DataFrame,
    unique_chrom_bp_types_ref: np.ndarray
) -> dict:
    """Query KD-tree forest with breakpoint positions."""
    return {
        chrom_bp_type: kdtforest[chrom_bp_type].query(
            bp_df_query[bp_df_query['chrom_bp_type'] == chrom_bp_type][['POS_bnd1', 'POS_bnd2']].to_numpy()
        )
        for chrom_bp_type in unique_chrom_bp_types_ref
    }


def match_nearest_query(
    bp_df_query: pd.DataFrame,
    bp_df_ref: pd.DataFrame,
    ref_closest: dict,
    unique_chrom_bp_types_query: np.ndarray,
    chrom_bp_types_ref: pd.Series,
    caller_name: str = 'defaultcaller'
) -> pd.DataFrame:
    """
    Match query breakpoints to their nearest reference breakpoints.
    
    Args:
        bp_df_query: Query breakpoint dataframe
        bp_df_ref: Reference breakpoint dataframe
        ref_closest: Dictionary of closest reference matches from KD-tree query
        unique_chrom_bp_types_query: Unique chromosome/breakpoint type combinations in query
        chrom_bp_types_ref: Chromosome/breakpoint types in reference
        caller_name: Name of variant caller to use in column suffixes
        
    Returns:
        DataFrame containing matched query and reference breakpoints
    """
    query_matches = pd.DataFrame()
    ref_matches = pd.DataFrame()
    unmatched_chrom_bp_types = []

    for chrom_bp_type in unique_chrom_bp_types_query:
        # Filter reference breakpoints by chromosomes and breakpoint type
        filtered_bp_df_query = bp_df_query[
            bp_df_query['chrom_bp_type'] == chrom_bp_type
        ]
        filtered_bp_df_ref = bp_df_ref[
            bp_df_ref['chrom_bp_type'] == chrom_bp_type
        ]
        
        if filtered_bp_df_ref.shape[0] == 0:
            unmatched_chrom_bp_types.append(chrom_bp_type)
            continue
            
        # Reset index to ensure proper indexing when selecting matches
        filtered_bp_df_ref = filtered_bp_df_ref.reset_index(drop=True)
        
        # Get indices of closest reference breakpoints
        matched_indices = ref_closest[chrom_bp_type][1]
        
        # Select matching reference breakpoints and add to results
        matched_refs = filtered_bp_df_ref.iloc[matched_indices]
        ref_matches = pd.concat([ref_matches, matched_refs])
        query_matches = pd.concat([query_matches, filtered_bp_df_query])

    ref_matches = ref_matches.reset_index(drop=True).add_suffix('_ref')
    query_matches = query_matches.reset_index(drop=True).add_suffix(f'_{caller_name}')

    unmatched_calls = (
        bp_df_query[bp_df_query['chrom_bp_type'].isin(unmatched_chrom_bp_types)]
        .reset_index(drop=True)
        .add_suffix(f'_{caller_name}')
    )

    return pd.concat(
        [pd.concat([query_matches, ref_matches], axis=1), unmatched_calls],
        axis=0
    ).reset_index(drop=True)


def match_nearest_breakpoints(
    bp_df_query: pd.DataFrame,
    bp_df_ref: pd.DataFrame,
    k: int = 1,
    caller_name: str = 'defaultcaller'
) -> pd.DataFrame:
    """
    Match breakpoints between query and reference callsets using KD-tree nearest neighbor search.
    
    Args:
        bp_df_query: Query breakpoint dataframe
        bp_df_ref: Reference breakpoint dataframe  
        k: Number of nearest neighbors to find
        caller_name: Name of variant caller to use in column suffixes
        
    Returns:
        DataFrame containing matched breakpoints
    """
    bp_df_query_preprocessed = preprocess_bp_df_for_kdtree(bp_df_query)
    bp_df_ref_preprocessed = preprocess_bp_df_for_kdtree(bp_df_ref)

    chrom_bp_types_ref = get_chrom_bp_type(bp_df_ref_preprocessed)
    chrom_bp_types_query = get_chrom_bp_type(bp_df_query_preprocessed)

    bp_df_query_preprocessed.loc[:, 'chrom_bp_type'] = chrom_bp_types_query
    bp_df_ref_preprocessed.loc[:, 'chrom_bp_type'] = chrom_bp_types_ref

    unique_chrom_bp_types_ref = chrom_bp_types_ref.unique()
    unique_chrom_bp_types_query = chrom_bp_types_query.unique()

    kdtforest = get_kdtforest(bp_df_ref_preprocessed, unique_chrom_bp_types_ref)
    ref_nearest_to_query = kdtforest_query(
        kdtforest,
        bp_df_query_preprocessed,
        unique_chrom_bp_types_ref
    )

    return match_nearest_query(
        bp_df_query_preprocessed,
        bp_df_ref_preprocessed,
        ref_nearest_to_query,
        unique_chrom_bp_types_query,
        unique_chrom_bp_types_ref,
        caller_name
    )


def get_bp_window_matches(
    matched_bp_df: pd.DataFrame,
    caller_name: str = 'defaultcaller'
) -> pd.Series:
    """
    Get boolean mask of breakpoints that match within confidence intervals.
    
    Args:
        matched_bp_df: DataFrame of matched breakpoints
        caller_name: Name of variant caller used in column suffixes
        
    Returns:
        Boolean Series indicating which breakpoints match within confidence intervals
    """
    bnd1_left_boundary = (
        matched_bp_df['POS_bnd1_ref'] +
        matched_bp_df['INFO_CIPOS_l_bnd1_ref'] +
        matched_bp_df[f'INFO_CIPOS_l_bnd1_{caller_name}']
    )
    bnd1_right_boundary = (
        matched_bp_df['POS_bnd1_ref'] +
        matched_bp_df['INFO_CIPOS_r_bnd1_ref'] +
        matched_bp_df[f'INFO_CIPOS_r_bnd1_{caller_name}']
    )
    bnd2_left_boundary = (
        matched_bp_df['POS_bnd2_ref'] +
        matched_bp_df['INFO_CIPOS_l_bnd2_ref'] +
        matched_bp_df[f'INFO_CIPOS_l_bnd2_{caller_name}']
    )
    bnd2_right_boundary = (
        matched_bp_df['POS_bnd2_ref'] +
        matched_bp_df['INFO_CIPOS_r_bnd2_ref'] +
        matched_bp_df[f'INFO_CIPOS_r_bnd2_{caller_name}']
    )

    ref_match_bnd1_mask = (
        (matched_bp_df[f'POS_bnd1_{caller_name}'] >= bnd1_left_boundary) &
        (matched_bp_df[f'POS_bnd1_{caller_name}'] <= bnd1_right_boundary)
    )
    ref_match_bnd2_mask = (
        (matched_bp_df[f'POS_bnd2_{caller_name}'] >= bnd2_left_boundary) &
        (matched_bp_df[f'POS_bnd2_{caller_name}'] <= bnd2_right_boundary)
    )

    return ref_match_bnd1_mask & ref_match_bnd2_mask
