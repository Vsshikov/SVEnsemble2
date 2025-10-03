"""
Functions to handle dataframes with multiple matched breakpoint sets
"""
import pandas as pd
import numpy as np

from . import _caller_params as caller_params
from . import chunk_match

def get_combined_insert_size(matchedbps_df: pd.DataFrame, matchedbps_callsets: list[str], inplace: bool = False) -> pd.DataFrame:
    """
    Compute the max insert size for each callset and aggregate into a single column.

    Args:
        matchedbps_df (pd.DataFrame): DataFrame containing matched breakpoints.
        callset_config_dict (dict): Dictionary of callset configurations.

    Returns:
        pd.DataFrame: DataFrame with an added 'svensemble_insert_size' column.
    """
    if not inplace:
        matchedbps_df = matchedbps_df.copy()

    for callset in matchedbps_callsets:
        matchedbps_no_prefix = matchedbps_df.rename(
            columns=lambda col: col[len(f"{callset}_"):] if col.startswith(f"{callset}_") else col
        )
        matchedbps_df[f"{callset}_insert_size"] = chunk_match.get_insert_size(
            pd.concat(
                [
                    matchedbps_no_prefix,
                    pd.DataFrame({"source": [callset] * len(matchedbps_no_prefix)})
                ],
                axis=1
            )
        )

    insert_size_cols = [f"{callset}_insert_size" for callset in matchedbps_callsets]
    insert_sizes = matchedbps_df[insert_size_cols].copy()

    for callset in matchedbps_callsets:
        info_event_col = f"{callset}_INFO_EVENT"
        insert_size_col = f"{callset}_insert_size"
        mask = matchedbps_df[info_event_col].isna()
        insert_sizes.loc[mask, insert_size_col] = np.nan

    if inplace:
        matchedbps_df.loc[:, "svensemble_insert_size"] = np.round(
            np.round(np.nanmax(insert_sizes, axis=1)).astype(int)
        )
        return matchedbps_df
    else:
        return np.round(np.nanmax(insert_sizes, axis=1)).astype(int)

def matchedbps_to_bps(
    matched_bps_df: pd.DataFrame,
    matched_bps_callsets: list[str],
    extra_cols: list[str] | None = None
) -> pd.DataFrame:
    """
    Convert a matched breakpoints DataFrame to a unified format with one coordinate pair per breakpoint.

    This function merges coordinates and confidence intervals from multiple callsets,
    producing a DataFrame with consensus positions and intervals for each breakpoint.

    Args:
        matched_bps_df (pd.DataFrame): DataFrame containing matched breakpoints from multiple callsets.
        matched_bps_callsets (list[str]): List of callsets to process.
        extra_cols (list[str] | None): Optional list of additional columns to retain in the output.

    Returns:
        pd.DataFrame: DataFrame with unified coordinates and confidence intervals per breakpoint.

    Notes:
        - For each breakpoint end (bnd1, bnd2), the function computes the mean position,
          and the leftmost/rightmost confidence interval bounds across all callsets.
        - Chromosome is taken from the first non-null value among callsets for each breakpoint end.
    """
    callsets = matched_bps_callsets
    bnds = ["1", "2"]

    # Collect relevant columns for merging
    relevant_cols = (
        [f"{callset}_INFO_EVENT" for callset in callsets] +
        [f"{callset}_POS_bnd{bnd}" for callset in callsets for bnd in bnds] +
        [f"{callset}_CHROM_bnd{bnd}" for callset in callsets for bnd in bnds] +
        [f"{callset}_INFO_CIPOS_l_bnd{bnd}" for callset in callsets for bnd in bnds] +
        [f"{callset}_INFO_CIPOS_r_bnd{bnd}" for callset in callsets for bnd in bnds] +
        [f"{callset}_ALT_bnd{bnd}" for callset in callsets for bnd in bnds]
    )
    if extra_cols:
        relevant_cols.extend(extra_cols)

    # Copy relevant columns for processing
    merged_coords_matched_bps = matched_bps_df[relevant_cols].copy()
    coords_with_cipos = pd.DataFrame(index=merged_coords_matched_bps.index)

    # Dictionaries for direction handling
    dir_dict = {"l": "leftmost", "r": "rightmost"}
    dir_np_funcs = {"l": np.nanmin, "r": np.nanmax}

    for bnd in bnds:
        pos_cols = []
        # Compute leftmost/rightmost positions for each direction and callset
        for direction in ["l", "r"]:
            for callset in callsets:
                col_pos = f"{callset}_POS_bnd{bnd}"
                pos_cols.append(col_pos)
                col_cipos = f"{callset}_INFO_CIPOS_{direction}_bnd{bnd}"
                coords_with_cipos[f"{callset}_POS_{direction}_bnd{bnd}"] = (
                    merged_coords_matched_bps[col_pos] +
                    merged_coords_matched_bps[col_cipos]
                )
            # Aggregate leftmost/rightmost positions across callsets
            cipos_cols = coords_with_cipos.columns.str.contains(f"POS_{direction}_bnd{bnd}")
            coords_with_cipos.loc[:, f"POS_{direction}_bnd{bnd}_{dir_dict[direction]}"] = (
                np.round(
                    dir_np_funcs[direction](
                        coords_with_cipos.loc[:, cipos_cols], axis=1
                    )
                ).astype(int)
            )

        # Compute consensus position as mean of all callset positions
        merged_coords_matched_bps.loc[:, f"POS_bnd{bnd}"] = np.round(
            np.nanmean(merged_coords_matched_bps.loc[:, pos_cols], axis=1)
        ).astype(int)

        # Compute confidence intervals relative to consensus position
        for direction in ["l", "r"]:
            merged_coords_matched_bps.loc[:, f"INFO_CIPOS_{direction}_bnd{bnd}"] = (
                coords_with_cipos.loc[:, f"POS_{direction}_bnd{bnd}_{dir_dict[direction]}"]
                - merged_coords_matched_bps.loc[:, f"POS_bnd{bnd}"]
            )

        # Assign chromosome from the first non-null value among callsets
        chrom_cols = [f"{callset}_CHROM_bnd{bnd}" for callset in callsets]
        merged_coords_matched_bps.loc[:, f"CHROM_bnd{bnd}"] = (
            merged_coords_matched_bps.loc[:, chrom_cols]
            .apply(lambda x: x.dropna().iloc[0], axis=1)
        )

        # Using caller priorty list from caller_params, assign ALT_bnd1 and ALT_bnd2
        merged_coords_matched_bps.loc[:, f"ALT_bnd{bnd}"] = np.nan
        for callset_name in caller_params.caller_alt_priority:
            # Skip if callset not in the data
            if callset_name not in callsets:
                continue
            missing_alts = merged_coords_matched_bps.loc[:, f"ALT_bnd{bnd}"].isna()
            if missing_alts.any():
                merged_coords_matched_bps.loc[missing_alts, f"ALT_bnd{bnd}"] = merged_coords_matched_bps.loc[missing_alts, f"{callset_name}_ALT_bnd{bnd}"]


    # Define columns to return in the final DataFrame
    return_cols = (
        [f"CHROM_bnd{bnd}" for bnd in bnds] +
        [f"POS_bnd{bnd}" for bnd in bnds] +
        [f"INFO_CIPOS_{direction}_bnd{bnd}" for bnd in bnds for direction in ["l", "r"]] +
        [f"ALT_bnd{bnd}" for bnd in bnds] +
        [f"{callset}_INFO_EVENT" for callset in callsets]
    )
    if extra_cols:
        return_cols.extend(extra_cols)

    return merged_coords_matched_bps.loc[:, return_cols]


def preprocess_matchedbps_for_pu(
    joint_bp_df: pd.DataFrame,
    sample_config: dict,
    reference_sample_name: str,
    training_set_name: str,
    verbose: bool = False
) -> pd.DataFrame:
    """
    Preprocess the joined breakpoint DataFrame for PU learning.

    For the specified training set, select relevant columns, filter rows, convert boolean-like
    columns, drop constant columns, and add a 'ref_match' column indicating whether
    the breakpoint matches the reference.

    Args:
        joint_bp_df (pd.DataFrame): DataFrame containing joined breakpoints.
        sample_config (dict): Sample configuration dictionary (currently unused).
        verbose (bool): If True, print information about dropped columns.

    Returns:
        pd.DataFrame: Preprocessed DataFrame.
    """
    # Set INFO_EVENT columns as index for easier filtering
    idx_cols = [col for col in joint_bp_df.columns if col.endswith('_INFO_EVENT')]
    joint_bp_df = joint_bp_df.set_index(idx_cols).sort_index()

    training_cols: list[str] = []
    cols_to_retain: list[str] = []

    # Collect all relevant columns for this training set
    for callset in sample_config['training_sets'][training_set_name]:
        training_cols.append(f'{callset}_INFO_EVENT')
        training_cols.extend(
            f'{callset}_FILTER_{col}'
            for col in caller_params.caller_param_dict[callset]['filter_cols']
        )
        training_cols.extend(
            f'{callset}_INFO_{col}'
            for col in caller_params.caller_param_dict[callset]['info_cols_pred']
        )
        training_cols.extend(
            f'{callset}_SAMPLE_{col}'
            for col in caller_params.caller_param_dict[callset]['format_cols']
        )
    # Find all columns in joint_bp_df that start with any of the training_col names
    for col in training_cols:
        cols_to_retain.extend(
            [c for c in joint_bp_df.columns if c.startswith(col)]
        )
    # Remove duplicates while preserving order
    seen = set()
    cols_to_retain = [x for x in cols_to_retain if not (x in seen or seen.add(x))]

    # Subset the DataFrame to relevant columns
    training_df = joint_bp_df.loc[:, cols_to_retain]

    # Retain rows where at least one callset's INFO_EVENT index level is not null
    info_event_cols = [f'{callset}_INFO_EVENT' for callset in sample_config['training_sets'][training_set_name]]
    if isinstance(training_df.index, pd.MultiIndex):
        info_event_level_idxs = [
            training_df.index.names.index(col)
            for col in info_event_cols if col in training_df.index.names
        ]
        if info_event_level_idxs:
            mask = (
                training_df.index.to_frame(index=False)
                .iloc[:, info_event_level_idxs]
                .notna()
                .any(axis=1)
            )
            training_df = training_df[mask.values]
    # else: If not MultiIndex, skip filtering

    # Add a 'ref_match' column: True if '{reference_sample_name}_INFO_EVENT' is not null, else False
    if reference_sample_name is not None:
        if f'{reference_sample_name}_INFO_EVENT' in training_df.index.names:
            training_df = training_df.copy()
            training_df['ref_match'] = training_df.index.get_level_values(f'{reference_sample_name}_INFO_EVENT').notna()
        else:
            raise ValueError(f"'{reference_sample_name}_INFO_EVENT' not found in joint_bp_df index.")

    # Convert columns with boolean-like strings to proper bool dtype
    for column in training_df.columns:
        if training_df[column].dtype == 'object':
            try:
                # Check if column contains True/False strings
                if training_df[column].astype(str).str.contains(r'^(True|False)$', na=False).any():
                    vals = (
                        training_df[column]
                        .map({'True': True, 'False': False})
                        .fillna(False)
                        .astype(bool)
                    )
                    training_df.iloc[:, training_df.columns.get_loc(column)] = vals
            except (AttributeError, TypeError) as err:
                if verbose:
                    print(f"Column '{column}' does not support string operations: {err}")
                continue

    # Check if 'ref_match' is constant
    n_unique = training_df['ref_match'].nunique()
    if n_unique == 1:
        if training_df['ref_match'].iloc[0]:
            raise ValueError("All calls are matched to reference; not a valid training set.")
        else:
            raise ValueError("No calls are matched to reference; not a valid training set.")

    # Drop columns with only 1 unique value
    nunique = training_df.nunique()
    constant_cols = nunique[nunique == 1].index.tolist()
    if constant_cols and verbose:
        print(f"Dropping {len(constant_cols)} constant columns: {constant_cols}")
    training_df = training_df.drop(columns=constant_cols)

    return training_df


def get_X_y_from_training_df(
    training_df: pd.DataFrame,
    match_col: str = 'ref_match',
    verbose: bool = False
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Prepare feature matrix X and target vector y from a DataFrame with matched breakpoints.

    This function:
        - Drops the match column (default: 'ref_match') from the input DataFrame.
        - Applies one-hot encoding to categorical columns.
        - Fills missing values with 0.
        - Returns the resulting feature matrix X and the target vector y (the match column).

    Args:
        training_df (pd.DataFrame): DataFrame containing features and a match column.
        match_col (str): Name of the column indicating reference match (target).
        verbose (bool): If True, print information about dropped columns.

    Returns:
        X (pd.DataFrame): Feature matrix after one-hot encoding and NaN filling.
        y (pd.Series): Target vector (reference match indicator).
    """
    training_df_copy = training_df.copy()
    X = pd.get_dummies(training_df_copy.drop(columns=[match_col]))
    X = X.fillna(0)

    y = training_df_copy[match_col]

    return X, y