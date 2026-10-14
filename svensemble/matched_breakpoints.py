"""
Functions to handle dataframes with multiple matched breakpoint sets
Some of the functions related to SVEnsemble2 metrics are also in this module.
"""
import pandas as pd
import numpy as np

from . import _caller_params as caller_params
from . import breakpoints as bps

def generate_svensemble2_ids(matchedbps_df: pd.DataFrame, callsets: list[str], prefix: str = 'svensemble2_') -> pd.Series:
    """
    Generate svensemble IDs for a matched breakpoints DataFrame.
    """
    svensemble2_id = pd.Series(index=matchedbps_df.index, dtype=str)
    svensemble2_id = matchedbps_df[[f'{callset}_INFO_EVENT' for callset in callsets]].astype(str).agg('|'.join, axis=1)
    svensemble2_id = prefix + pd.factorize(svensemble2_id)[0].astype(str)
    return svensemble2_id

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
        matchedbps_df[f"{callset}_insert_size"] = bps.get_insert_size(
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
        def _first_chrom(row):
            vals = row.dropna()
            if vals.empty:
                raise ValueError(
                    f"All callsets have NaN for CHROM_bnd{bnd} "
                    f"at index {row.name!r}; cannot determine chromosome"
                )
            return vals.iloc[0]
        merged_coords_matched_bps.loc[:, f"CHROM_bnd{bnd}"] = (
            merged_coords_matched_bps.loc[:, chrom_cols]
            .apply(_first_chrom, axis=1)
        )

        # Using callset priorty list from caller_params, assign ALT_bnd1 and ALT_bnd2
        merged_coords_matched_bps.loc[:, f"ALT_bnd{bnd}"] = np.nan
        for callset_name in caller_params.callset_alt_priority:
            # Skip if callset not in the data
            if callset_name not in callsets:
                continue
            missing_alts = merged_coords_matched_bps.loc[:, f"ALT_bnd{bnd}"].isna()
            if missing_alts.any():
                merged_coords_matched_bps[f"ALT_bnd{bnd}"] = merged_coords_matched_bps[f"ALT_bnd{bnd}"].astype(object)
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
    workflow_config: dict,
    match_sample_name: str,
    training_set_name: str,
    add_match_ids_to_index: bool = False,
    ignore_match_errors: bool = False,
    verbose: bool = False
) -> pd.DataFrame:
    """
    Preprocess the joined breakpoint DataFrame for PU learning.

    For the specified training set, select relevant columns, filter rows, convert boolean-like
    columns, drop constant columns, and add a 'ref_match' column indicating whether
    the breakpoint matches the reference.

    Args:
        joint_bp_df (pd.DataFrame): DataFrame containing joined breakpoints.
        workflow_config (dict): Workflow config; supplies the training set's
            callers, their callset params and the model's training_params.
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
    for callset_name in workflow_config['training_sets'][training_set_name]:
        callset_params = workflow_config['callsets'][callset_name]['params']
        training_cols.append(f'{callset_name}_INFO_EVENT')
        training_cols.extend(
            f'{callset_name}_FILTER_{col}'
            for col in callset_params['filter_cols']
        )
        training_cols.extend(
            f'{callset_name}_INFO_{col}'
            for col in callset_params['info_cols_pred']
        )
        training_cols.extend(
            f'{callset_name}_SAMPLE_{col}'
            for col in callset_params['format_cols']
        )
        training_cols.extend(
            f'{callset_name}_CUSTOM_{col}'
            for col in callset_params['custom_cols_pred']
        )
        # Quality metric column may not be included in the training_cols list, so add it if it exists and is not already in training_cols
        if (callset_params['qual_metric'] not in training_cols) and (callset_params['qual_metric'] is not None):
            training_cols.append(f'{callset_name}_{callset_params["qual_metric"]}')

    if workflow_config['model'].get('training_params', {}).get('add_topology', False):
        training_cols.extend(col for col in joint_bp_df.columns if col.startswith('svensembletopology_'))


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
    info_event_cols = [f'{callset}_INFO_EVENT' for callset in workflow_config['training_sets'][training_set_name]]
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

    # Add a 'ref_match' column: True if '{match_sample_name}_INFO_EVENT' is not null, else False
    if match_sample_name is not None:
        if f'{match_sample_name}_INFO_EVENT' in training_df.index.names:
            training_df['ref_match'] = training_df.index.get_level_values(f'{match_sample_name}_INFO_EVENT').notna()
        else:
            raise ValueError(f"'{match_sample_name}_INFO_EVENT' not found in joint_bp_df index.")

    # Convert columns with boolean-like strings to proper bool dtype
    for column in training_df.columns:
        if training_df[column].dtype == 'object':
            try:
                # Check if column contains True/False strings
                if training_df[column].astype(str).isin(['True', 'False']).any():
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
    if (n_unique == 1) and not ignore_match_errors:
        if training_df['ref_match'].iloc[0]:
            raise ValueError("All calls are matched to reference; not a valid training set.")
        else:
            raise ValueError("No calls are matched to reference; not a valid training set.")

    nunique = training_df.nunique()
    constant_cols = nunique[nunique == 1].index.tolist()
    # Never drop the match column even if it's constant
    if 'ref_match' in constant_cols:
        constant_cols.remove('ref_match')
    if constant_cols and verbose:
        print(f"Dropping {len(constant_cols)} constant columns: {constant_cols}")
    training_df = training_df.drop(columns=constant_cols)

    # Drop the match_sample_INFO_EVENT index from multiindex if specified by add_match_ids_to_index=False
    if not add_match_ids_to_index:
        training_df.index = training_df.index.droplevel(f'{match_sample_name}_INFO_EVENT')

    return training_df


def get_X_y_from_training_df(
    training_df: pd.DataFrame,
    match_col: str = 'ref_match',
    fillna: float | None = 0,
    verbose: bool = False
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Prepare feature matrix X and target vector y from a DataFrame with matched breakpoints.

    This function:
        - Drops the match column (default: 'ref_match') from the input DataFrame.
        - Applies one-hot encoding to categorical columns.
        - Optionally fills missing values (controlled by fillna).
        - Returns the resulting feature matrix X and the target vector y (the match column).

    Args:
        training_df (pd.DataFrame): DataFrame containing features and a match column.
        match_col (str): Name of the column indicating reference match (target).
        fillna (int | float | None): Value to fill NaN with. Set to None to
            preserve NaN (e.g. for models with native missing-value support).
        verbose (bool): If True, print information about dropped columns.

    Returns:
        X (pd.DataFrame): Feature matrix after one-hot encoding and optional NaN filling.
        y (pd.Series): Target vector (reference match indicator).
    """
    training_df_copy = training_df.copy()
    X = pd.get_dummies(training_df_copy.drop(columns=[match_col]))

    if fillna is not None:
        X = X.fillna(fillna)

    for column in X.columns:
        if fillna is not None:
            try:
                if np.allclose(X[column], X[column].round()):
                    X[column] = X[column].astype(int)
                else:
                    X[column] = X[column].astype(float)
            except (ValueError, pd.errors.IntCastingNaNError):
                X[column] = X[column].astype(float)
        else:
            X[column] = X[column].astype(float)

    y = training_df_copy[match_col]

    return X, y

def get_svensemble2_calls(
    matchedbps_df: pd.DataFrame,
    pred_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Summarizes SVEnsemble2 calls by merging matched breakpoint and prediction DataFrames,
    computing combined insert sizes, generating unique SVEnsemble2 IDs, and formatting results.

    Args:
        matchedbps_df (pd.DataFrame): DataFrame of matched breakpoints.
        pred_df (pd.DataFrame): DataFrame containing model predictions and INFO_EVENT keys.

    Returns:
        pd.DataFrame: Unified DataFrame of SVEnsemble2 summary calls with consensus coordinates, insert size, unique ID, and predictions.
    """
    # Create a working copy of the matched breakpoints to avoid modifying the input DataFrame
    matchedbps_df_copy = matchedbps_df.copy()

    # Infer callset list based on prefixes of INFO_EVENT columns in pred_df
    callsets = [col.replace('_INFO_EVENT', '') for col in pred_df.columns if 'INFO_EVENT' in col]

    # Merge the predictions into the matched breakpoints data using the INFO_EVENT columns as keys
    matchedbps_df_copy = matchedbps_df_copy.merge(
        pred_df,
        on=[f'{callset}_INFO_EVENT' for callset in callsets]
    )

    # Compute the combined (aggregate/max) insert size across all callsets and store in a new column
    matchedbps_df_copy['svensemble2_insert_size'] = get_combined_insert_size(
        matchedbps_df_copy, callsets
    )

    # Generate a unique identifier for each SV by concatenating INFO_EVENT fields from all callsets
    # This is useful for matching SVs across different sets and preserving provenance
    matchedbps_df_copy.loc[:, 'INFO_EVENT'] = generate_svensemble2_ids(
        matchedbps_df_copy, callsets
    )

    # Infer the training_set_name and model_name from the single prediction column in pred_df
    # Expect column like "{training_set_name}_{model_name}_pred"
    prediction_cols = [col for col in pred_df.columns if col.endswith('_pred')]
    if len(prediction_cols) != 1:
        raise ValueError("There should be exactly one prediction column in pred_df ending with '_pred'")
    pred_col = prediction_cols[0]
    # Split out training_set_name and model_name
    name_parts = pred_col.rsplit('_', 2)
    if len(name_parts) == 3:
        training_set_name, model_name, _ = name_parts
    else:
        raise ValueError(f"Prediction column name does not match expected pattern: '{pred_col}'")


    # Convert to a SVEnsemble2 "bps" format, retaining coordinate/ALT consensus, insert size, unique ID, and predictions column
    return matchedbps_to_bps(
        matchedbps_df_copy,
        callsets,
        extra_cols=[
            'svensemble2_insert_size',
            'INFO_EVENT',
            f"{training_set_name}_{model_name}_pred"
        ]
    )