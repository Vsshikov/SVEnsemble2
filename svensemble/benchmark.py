"""
Benchmark a callset or SVEnsemble2 predictions against a validation set.

run_single_callset_benchmark backs `svensemble benchmark-callset`. Validation
sets are produced by validation_sets.preprocess_validation_set; curve plotting
lives in plotting.
"""

import pandas as pd
import numpy as np
from typing import List
from pathlib import Path

from . import _caller_params as caller_params
from . import chunk_match as chunk_match
from . import breakpoints as bps


def get_truth_count(
    val_set_path: str | Path,
    val_set_name: str,
    svtypes: list[str] | None = None,
    svlens: tuple[float, float] = (50, np.inf),
) -> int:
    """Return the number of truth variants that survive the svtype/svlen filter.

    Args:
        val_set_path: Path to the validation breakpoints CSV.
        val_set_name: Validation set name (callset_param_dict key, used for annotation).
        svtypes: List of SV types to include. Defaults to caller_params.default_sv_types.
        svlens: (min, max) SVLEN range. Defaults to (50, inf).

    Returns:
        Integer count of truth variants passing the filter.
    """
    truth = pd.read_csv(val_set_path)
    truth_filtered = bps.svlen_svtype_filter(truth, val_set_name, svtypes=svtypes, svlens=svlens)
    return len(truth_filtered)


def validate_basic_callset(callset_bps, val_set, val_name, callset_name, val_match_margin=300, val_match_insert_delta_threshold=0.5, additional_cols: None | List[str] = None, verbose=True):
    """
    Given dataframes of a callset and a validation set, match them and return a dataframe with the following columns:
    INFO_EVENT, {qual_metric}, is_in_val, INFO_INFERRED_SVTYPE, {FILTER_PASS}

    qual_metric column is dependent on the callset and is read from callset_param_dict[callset_name]
    FILTER_PASS column is returned only if it is present in the callset
    val_name is the validation set's callset_param_dict key
    """
    qual_metric = caller_params.callset_param_dict[callset_name]['qual_metric']
    callset_bps_ = callset_bps.copy()
    val_set_ = val_set.copy()

    # Caller-specific operations go here
    if callset_name == 'manta_candidate':
        callset_bps_.loc[:, 'INFO_TOTAL_PAIR_COUNT'] = \
            callset_bps_['INFO_UPSTREAM_PAIR_COUNT'] + \
            callset_bps_['INFO_DOWNSTREAM_PAIR_COUNT'] + \
            callset_bps_['INFO_PAIR_COUNT']

    # INFO_INFERRED_SVTYPE is required downstream for svtype-stratified filtering.
    # Compute it from breakpoint geometry when the callset was not pre-annotated.
    if 'INFO_INFERRED_SVTYPE' not in callset_bps_.columns:
        callset_bps_.loc[:, 'INFO_INFERRED_SVTYPE'] = bps.annotate_svtypes(callset_bps_, callset_name)

    val_key = val_name
    if val_key == callset_name:
        i = 0
        val_key = f'{val_key}_{i}'
        if val_key not in caller_params.callset_param_dict:
            caller_params.callset_param_dict[val_key] = caller_params.callset_param_dict[val_name]
        else:
            while val_key in caller_params.callset_param_dict:
                i += 1
                val_key = val_key.replace(f'_{i-1}', f'_{i}')
            caller_params.callset_param_dict[val_key] = caller_params.callset_param_dict[val_name]
            if i > 1000000:
                raise ValueError(f'Too many temporary validation set names generated for {val_key}')

    matched_ids = chunk_match.inner_join_bps(
        bp_df_dict={
            callset_name: callset_bps_,
            val_key: val_set_
        },
        max_dist=val_match_margin,
        insert_delta_threshold=val_match_insert_delta_threshold,
        ids_only=True,
        ignore_size=False
    )
    callset_bps_.loc[:, 'is_in_val'] = callset_bps_['INFO_EVENT'].astype(str).isin(matched_ids[f'{callset_name}_INFO_EVENT'])
    cols_to_keep = ['INFO_EVENT', qual_metric, 'is_in_val']
    if 'FILTER_PASS' in callset_bps_.columns:
        cols_to_keep.append('FILTER_PASS')
    if 'INFO_INFERRED_SVTYPE' in callset_bps_.columns:
        cols_to_keep.append('INFO_INFERRED_SVTYPE')
    if isinstance(additional_cols, list):
        cols_to_keep.extend([col for col in additional_cols if ((col not in cols_to_keep) and (col in callset_bps_.columns))])

    return callset_bps_.loc[callset_bps_['INFO_EVENT'].notna(), cols_to_keep].rename(columns={qual_metric: 'qual_metric'})



def validate_svensemble2_predictions(svensemble2_bps, val_set, val_name, training_set_name=None, model_name=None, pred_col=None, val_match_margin=300, val_match_insert_delta_threshold=0.5, additional_cols: None | List[str] = None, verbose=True):
    svens_calls_ = svensemble2_bps.copy()
    val_set_ = val_set.copy()

    # Resolve the single prediction column. Callers may pass it explicitly,
    # supply training_set_name + model_name to reconstruct it, or rely on
    # auto-detection of the unique '*_pred' column written by get_svensemble2_calls.
    if pred_col is None:
        if training_set_name is not None and model_name is not None:
            pred_col = f'{training_set_name}_{model_name}_pred'
        else:
            pred_candidates = [c for c in svens_calls_.columns if c.endswith('_pred')]
            if len(pred_candidates) != 1:
                raise ValueError(
                    f"Expected exactly one '*_pred' column in svensemble2 calls, found {pred_candidates}"
                )
            pred_col = pred_candidates[0]

    # INFO_INFERRED_SVTYPE is required downstream for svtype-stratified filtering.
    # Compute it from breakpoint geometry when the calls were not pre-annotated.
    if 'INFO_INFERRED_SVTYPE' not in svens_calls_.columns:
        svens_calls_.loc[:, 'INFO_INFERRED_SVTYPE'] = bps.annotate_svtypes(svens_calls_, 'svensemble2')

    val_key = val_name
    if val_key == 'svensemble2':
        i = 0
        val_key = f'{val_key}_val_{i}'
        if val_key not in caller_params.callset_param_dict:
            caller_params.callset_param_dict[val_key] = caller_params.callset_param_dict[val_name]
        else:
            while val_key in caller_params.callset_param_dict:
                i += 1
                val_key = val_key.replace(f'_val_{i-1}', f'_val_{i}')
            caller_params.callset_param_dict[val_key] = caller_params.callset_param_dict[val_name]
            if i > 1000000:
                raise ValueError(f'Too many temporary validation set names generated for {val_key}')

    matched_ids = chunk_match.inner_join_bps(
        bp_df_dict={
            'svensemble2': svens_calls_,
            val_key: val_set_
        },
        max_dist=val_match_margin,
        insert_delta_threshold=val_match_insert_delta_threshold,
        ids_only=True,
        ignore_size=False
    )

    svens_calls_.loc[:, 'is_in_val'] = svens_calls_['INFO_EVENT'].astype(str).isin(matched_ids['svensemble2_INFO_EVENT'])
    cols_to_keep = ['INFO_EVENT', pred_col, 'is_in_val']
    if 'INFO_INFERRED_SVTYPE' in svens_calls_.columns:
        cols_to_keep.append('INFO_INFERRED_SVTYPE')
    if isinstance(additional_cols, list):
        cols_to_keep.extend([col for col in additional_cols if ((col not in cols_to_keep) and (col in svens_calls_.columns))])
    val_df = svens_calls_[cols_to_keep]
    val_df = val_df[val_df[pred_col].notna()].rename(columns={pred_col: 'qual_metric'})

    return val_df


def run_single_callset_benchmark(
    callset_bps_path: Path,
    callset_name: str,
    val_bps_path: Path,
    val_name: str,
    output_dir: Path | None = None,
    svtypes: list[str] | None = None,
    svlen_min: int | None = None,
    svlen_max: int | None = None,
    verbose: bool = False,
):
    """Benchmark one callset against a validation set and write a single CSV.

    Dispatch is by callset_name: names prefixed svensemble2 are ensemble
    prediction sets (validated via validate_svensemble2_predictions); all
    other names are basic caller callsets (validated via
    validate_basic_callset, with qual_metric resolved from
    callset_param_dict[callset_name]). This unifies individual callers
    (gridss, delly, survivor_*, ...) and ensemble predictions behind one entry
    point.

    Optional SV-length and SV-type filtering is each applied independently. When
    any filter is set it is applied to BOTH the callset and the validation set
    so metrics are computed within the category.

    Args:
        callset_bps_path (Path): Path to the callset breakpoints CSV.
        callset_name (str): Callset identifier. Names prefixed svensemble2
            are treated as ensemble prediction sets; all others are basic
            caller callsets resolved against callset_param_dict.
        val_bps_path (Path): Path to the validation set breakpoints CSV.
        val_name (str): Name of the validation set.
        output_dir (Path, optional): Directory to write
            {callset_name}.benchmark.csv into. If None, the validation
            dataframe is returned instead. Default: None.
        svtypes (list[str], optional): INFO_INFERRED_SVTYPE values to keep.
            None keeps all types. Default: None.
        svlen_min (int, optional): Inclusive lower SV-length bound. None means
            0. Default: None.
        svlen_max (int, optional): Inclusive upper SV-length bound. None means
            unbounded. Default: None.
        verbose (bool, optional): Whether to print processing info.
            Default: False.

    Returns:
        pandas.DataFrame or None:
        - The validation dataframe when output_dir is None;
        - None after writing {callset_name}.benchmark.csv to output_dir when output_dir is not None.
    """
    val_set = pd.read_csv(val_bps_path)
    callset_bps = pd.read_csv(callset_bps_path, low_memory=False)

    source = 'svensemble2' if callset_name.startswith('svensemble2') else callset_name

    if svtypes is not None or svlen_min is not None or svlen_max is not None:
        svlens = (
            svlen_min if svlen_min is not None else 0,
            svlen_max if svlen_max is not None else np.inf,
        )
        callset_bps = bps.svlen_svtype_filter(callset_bps, source, svtypes=svtypes, svlens=svlens)
        val_set = bps.svlen_svtype_filter(val_set, val_name, svtypes=svtypes, svlens=svlens)
        if verbose:
            print(f"Filtered {callset_name} and {val_name} to svtypes={svtypes}, svlens={svlens}")

    if callset_name.startswith('svensemble2'):
        val_df = validate_svensemble2_predictions(
            svensemble2_bps=callset_bps,
            val_set=val_set,
            val_name=val_name,
            verbose=verbose,
        )
    else:
        val_df = validate_basic_callset(
            callset_bps=callset_bps,
            val_set=val_set,
            val_name=val_name,
            callset_name=callset_name,
            verbose=verbose,
        )

    if verbose:
        match_count = val_df["is_in_val"].sum()
        total_count = len(val_df)
        print(
            f"{callset_name} callset processed against {val_name} "
            f"({match_count} matches out of {total_count} predictions)"
        )

    if output_dir is None:
        return val_df

    output_dir.mkdir(parents=True, exist_ok=True)
    val_df.to_csv(output_dir / f'{callset_name}.benchmark.csv', index=False)
