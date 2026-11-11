"""
Functions to handle dataframes with multiple matched breakpoint sets
Some of the functions related to SVEnsemble2 metrics are also in this module.
"""
import re

import pandas as pd
import numpy as np

from . import _caller_params as caller_params
from . import breakpoints as bps
from . import vcfwrite
from .annotate import pred_column
from ._constants import (
    BNDS,
    CI_LEFT,
    CI_RIGHT,
    CI_SIDES,
    CUSTOM_PREFIX,
    FILTER_PREFIX,
    INFO_PREFIX,
    PRED_SUFFIX,
    SAMPLE_ID_COL,
    SAMPLE_PREFIX,
    TOPOLOGY_PREFIX,
)

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

    # Collect relevant columns for merging
    relevant_cols = (
        [f"{callset}_INFO_EVENT" for callset in callsets] +
        [f"{callset}_POS_{bnd_label}" for callset in callsets for bnd_label in BNDS] +
        [f"{callset}_CHROM_{bnd_label}" for callset in callsets for bnd_label in BNDS] +
        [f"{callset}_INFO_CIPOS_l_{bnd_label}" for callset in callsets for bnd_label in BNDS] +
        [f"{callset}_INFO_CIPOS_r_{bnd_label}" for callset in callsets for bnd_label in BNDS] +
        [f"{callset}_ALT_{bnd_label}" for callset in callsets for bnd_label in BNDS]
    )
    if extra_cols:
        relevant_cols.extend(extra_cols)

    # Copy relevant columns for processing
    merged_coords_matched_bps = matched_bps_df[relevant_cols].copy()
    coords_with_cipos = pd.DataFrame(index=merged_coords_matched_bps.index)

    # Dictionaries for direction handling
    dir_dict = {CI_LEFT: "leftmost", CI_RIGHT: "rightmost"}
    dir_np_funcs = {CI_LEFT: np.nanmin, CI_RIGHT: np.nanmax}

    for bnd_label in BNDS:
        pos_cols = []
        # Compute leftmost/rightmost positions for each direction and callset
        for direction in CI_SIDES:
            for callset in callsets:
                col_pos = f"{callset}_POS_{bnd_label}"
                pos_cols.append(col_pos)
                col_cipos = f"{callset}_INFO_CIPOS_{direction}_{bnd_label}"
                coords_with_cipos[f"{callset}_POS_{direction}_{bnd_label}"] = (
                    merged_coords_matched_bps[col_pos] +
                    merged_coords_matched_bps[col_cipos]
                )
            # Aggregate leftmost/rightmost positions across callsets
            cipos_cols = coords_with_cipos.columns.str.contains(f"POS_{direction}_{bnd_label}")
            coords_with_cipos.loc[:, f"POS_{direction}_{bnd_label}_{dir_dict[direction]}"] = (
                np.round(
                    dir_np_funcs[direction](
                        coords_with_cipos.loc[:, cipos_cols], axis=1
                    )
                ).astype(int)
            )

        # Compute consensus position as mean of all callset positions
        merged_coords_matched_bps.loc[:, f"POS_{bnd_label}"] = np.round(
            np.nanmean(merged_coords_matched_bps.loc[:, pos_cols], axis=1)
        ).astype(int)

        # Compute confidence intervals relative to consensus position
        for direction in CI_SIDES:
            merged_coords_matched_bps.loc[:, f"INFO_CIPOS_{direction}_{bnd_label}"] = (
                coords_with_cipos.loc[:, f"POS_{direction}_{bnd_label}_{dir_dict[direction]}"]
                - merged_coords_matched_bps.loc[:, f"POS_{bnd_label}"]
            )

        # Assign chromosome from the first non-null value among callsets
        chrom_cols = [f"{callset}_CHROM_{bnd_label}" for callset in callsets]
        def _first_chrom(row):
            vals = row.dropna()
            if vals.empty:
                raise ValueError(
                    f"All callsets have NaN for CHROM_{bnd_label} "
                    f"at index {row.name!r}; cannot determine chromosome"
                )
            return vals.iloc[0]
        merged_coords_matched_bps.loc[:, f"CHROM_{bnd_label}"] = (
            merged_coords_matched_bps.loc[:, chrom_cols]
            .apply(_first_chrom, axis=1)
        )

        # Using callset priorty list from caller_params, assign ALT_bnd1 and ALT_bnd2
        merged_coords_matched_bps.loc[:, f"ALT_{bnd_label}"] = np.nan
        for callset_name in caller_params.CALLSET_ALT_PRIORITY:
            # Skip if callset not in the data
            if callset_name not in callsets:
                continue
            missing_alts = merged_coords_matched_bps.loc[:, f"ALT_{bnd_label}"].isna()
            if missing_alts.any():
                merged_coords_matched_bps[f"ALT_{bnd_label}"] = merged_coords_matched_bps[f"ALT_{bnd_label}"].astype(object)
                merged_coords_matched_bps.loc[missing_alts, f"ALT_{bnd_label}"] = merged_coords_matched_bps.loc[missing_alts, f"{callset_name}_ALT_{bnd_label}"]


    # Define columns to return in the final DataFrame
    return_cols = (
        [f"CHROM_{bnd_label}" for bnd_label in BNDS] +
        [f"POS_{bnd_label}" for bnd_label in BNDS] +
        [f"INFO_CIPOS_{direction}_{bnd_label}" for bnd_label in BNDS for direction in CI_SIDES] +
        [f"ALT_{bnd_label}" for bnd_label in BNDS] +
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
    # Set INFO_EVENT columns (and the sample, last) as index for easier
    # filtering; as an index level the sample can never become a feature
    idx_cols = [col for col in joint_bp_df.columns if col.endswith('_INFO_EVENT')]
    if SAMPLE_ID_COL in joint_bp_df.columns:
        idx_cols.append(SAMPLE_ID_COL)
    joint_bp_df = joint_bp_df.set_index(idx_cols).sort_index()

    training_cols: list[str] = []
    cols_to_retain: list[str] = []

    # Collect all relevant columns for this training set
    for callset_name in workflow_config['training_sets'][training_set_name]:
        callset_params = workflow_config['callsets'][callset_name]['params']
        training_cols.append(f'{callset_name}_INFO_EVENT')
        training_cols.extend(
            f'{callset_name}_{FILTER_PREFIX}{col}'
            for col in callset_params['filter_cols']
        )
        training_cols.extend(
            f'{callset_name}_{INFO_PREFIX}{col}'
            for col in callset_params['info_cols_pred']
        )
        training_cols.extend(
            f'{callset_name}_{SAMPLE_PREFIX}{col}'
            for col in callset_params['format_cols']
        )
        training_cols.extend(
            f'{callset_name}_{CUSTOM_PREFIX}{col}'
            for col in callset_params['custom_cols_pred']
        )
        # Quality metric column may not be included in the training_cols list, so add it if it exists and is not already in training_cols
        if (callset_params['qual_metric'] not in training_cols) and (callset_params['qual_metric'] is not None):
            training_cols.append(f'{callset_name}_{callset_params["qual_metric"]}')

    if workflow_config['model'].get('training_params', {}).get('add_topology', False):
        training_cols.extend(col for col in joint_bp_df.columns if col.startswith(TOPOLOGY_PREFIX))


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
    pred_df: pd.DataFrame,
    common_id_name: str = 'svensemble_id',
) -> pd.DataFrame:
    """
    Summarizes SVEnsemble2 calls by merging matched breakpoint and prediction DataFrames,
    computing combined insert sizes, generating unique SVEnsemble2 IDs, and formatting results.

    Args:
        matchedbps_df (pd.DataFrame): DataFrame of matched breakpoints.
        pred_df (pd.DataFrame): DataFrame containing model predictions and INFO_EVENT keys.
        common_id_name (str): SV ID column (join_calls with add_common_id); kept in
            the calls when matchedbps_df has it. Defaults to 'svensemble_id'.

    Returns:
        pd.DataFrame: Unified DataFrame of SVEnsemble2 summary calls with consensus coordinates, insert size, unique ID, and predictions.
    """
    # Create a working copy of the matched breakpoints to avoid modifying the input DataFrame
    matchedbps_df_copy = matchedbps_df.copy()

    # Infer callset list based on prefixes of INFO_EVENT columns in pred_df
    callsets = [col.replace('_INFO_EVENT', '') for col in pred_df.columns if 'INFO_EVENT' in col]

    # Merge the predictions into the matched breakpoints data on the
    # INFO_EVENT columns, plus the sample when the predictions are per sample
    merge_keys = [f'{callset}_INFO_EVENT' for callset in callsets]
    if SAMPLE_ID_COL in pred_df.columns:
        merge_keys.append(SAMPLE_ID_COL)
        matchedbps_df_copy[SAMPLE_ID_COL] = matchedbps_df_copy[SAMPLE_ID_COL].astype(str)
        pred_df = pred_df.assign(**{SAMPLE_ID_COL: pred_df[SAMPLE_ID_COL].astype(str)})
    matchedbps_df_copy = matchedbps_df_copy.merge(pred_df, on=merge_keys)

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
    prediction_cols = [col for col in pred_df.columns if col.endswith(PRED_SUFFIX)]
    if len(prediction_cols) != 1:
        raise ValueError(f"There should be exactly one prediction column in pred_df ending with '{PRED_SUFFIX}'")
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
            *([SAMPLE_ID_COL] if SAMPLE_ID_COL in matchedbps_df_copy.columns else []),
            *([common_id_name] if common_id_name in matchedbps_df_copy.columns else []),
            f"{training_set_name}_{model_name}{PRED_SUFFIX}"
        ]
    )

_MATE_IN_ALT = re.compile(r'([\[\]])([^\[\]:]+):(\d+)([\[\]])')


def _set_alt_mate_pos(alt: pd.Series, mate_pos: pd.Series) -> pd.Series:
    """BND ALT strings with the mate position replaced; other ALTs unchanged."""
    return pd.Series(
        [
            _MATE_IN_ALT.sub(lambda m, p=p: f'{m[1]}{m[2]}:{int(p)}{m[4]}', a) if isinstance(a, str) else a
            for a, p in zip(alt, mate_pos)
        ],
        index=alt.index,
    )


def _harmonise_samples(
    df: pd.DataFrame, how: str = 'mean', common_id_name: str = 'svensemble_id',
) -> pd.DataFrame:
    """Give every row of one SV the same site across the samples it appears in.

    Groups the rows by their SV ID (the common_id_name column, from join_calls
    with add_common_id). For each SV with rows in more than one sample, the site columns
    (vcfwrite.site_cols: coordinates, CIs, ALT, INFO_INFERRED_SVTYPE,
    INFO_SVLEN) of all its rows are overwritten with one value set; every
    other column, and the rows themselves, are unchanged. A frame with at
    most one sample is returned as is.

    Args:
        df: matched_bps or calls rows with SAMPLE_ID_COL and common_id_name.
        how: 'mean' -- positions are the rounded mean, CIs the envelope of
            the rows' intervals around the new positions, INFO_SVLEN the
            mean; text columns (CHROM, ALT, INFO_INFERRED_SVTYPE) come from
            the first row in sample order, with BND ALT mate positions
            updated. 'max_score' -- all site columns are copied from the
            SV's highest-scoring row (needs the '*_pred' column).
        common_id_name: Column holding the SV IDs. Defaults to 'svensemble_id'.
    """
    if how not in ('mean', 'max_score'):
        raise ValueError(f"how must be 'mean' or 'max_score', got {how!r}")
    if SAMPLE_ID_COL not in df.columns or df[SAMPLE_ID_COL].nunique() <= 1:
        return df

    sv = df[common_id_name].astype(str)
    shared = (sv.groupby(sv).transform('size') > 1).to_numpy()
    if not shared.any():
        return df
    if pd.DataFrame({'sv': sv, 'sample': df[SAMPLE_ID_COL].astype(str)}).duplicated().any():
        raise ValueError("An SV has more than one row for one sample; prune the ID graph first")

    cols = vcfwrite.site_cols(df, include_event=False)
    rows = df.loc[shared]
    groups = sv[shared]
    out = df.copy()

    if how == 'max_score':
        pred = pred_column(df)
        best = rows[pred].astype(float).groupby(groups).idxmax()
        site = df.loc[best.to_numpy(), cols].set_axis(best.index)
        out.loc[shared, cols] = site.reindex(groups.to_numpy()).to_numpy()
        return out

    ordered = rows.assign(_sv=groups).sort_values(SAMPLE_ID_COL, kind='stable')
    site = ordered.groupby('_sv')[cols].first()
    for bnd in BNDS:
        pos_col = f'POS_{bnd}'
        mean_pos = rows[pos_col].astype(float).groupby(groups).mean().round().astype(np.int64)
        for side, agg in ((CI_LEFT, 'min'), (CI_RIGHT, 'max')):
            ci_col = f'INFO_CIPOS_{side}_{bnd}'
            if ci_col in cols:
                edge = (rows[pos_col] + rows[ci_col]).groupby(groups).agg(agg)
                site[ci_col] = (edge - mean_pos).astype(np.int64)
        site[pos_col] = mean_pos
    if 'INFO_SVLEN' in cols:
        site['INFO_SVLEN'] = rows['INFO_SVLEN'].astype(float).groupby(groups).mean().round()
    for bnd, mate in ((BNDS[0], BNDS[1]), (BNDS[1], BNDS[0])):
        alt_col = f'ALT_{bnd}'
        if alt_col in cols:
            site[alt_col] = _set_alt_mate_pos(site[alt_col], site[f'POS_{mate}'])

    for col in cols:
        values = site[col].reindex(groups.to_numpy()).to_numpy()
        if pd.api.types.is_integer_dtype(out[col].dtype) and not pd.isna(values).any():
            values = values.astype(out[col].dtype)
        out.loc[shared, col] = values
    return out
