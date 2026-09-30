"""
Python-API entry point for the SVEnsemble2 pipeline.

Exposes `svensemble2(...)` and per-stage helpers (`parse_preprocess_raw`,
`parse_preprocess_model_specific`, `match_sample_breakpoints`,
`pu_train_sample`, `pu_predict_sample`, ...). The command-line surface lives
in svensemble.cli.
"""

import os
import pickle
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from pysam import VariantFile


def _read_csv_with_retry(path: Path, retries: int = 3, **kwargs) -> pd.DataFrame:
    """Read a CSV with retry on transient I/O errors (e.g. NFS attribute caching)."""
    for attempt in range(retries):
        try:
            return pd.read_csv(path, **kwargs)
        except (pd.errors.EmptyDataError, pd.errors.ParserError) as exc:
            if attempt == retries - 1:
                raise
            wait = 2 ** attempt
            _log(f"Transient read error on {path} (attempt {attempt + 1}/{retries}): {exc}. Retrying in {wait}s...")
            time.sleep(wait)
    return pd.read_csv(path, **kwargs)  # unreachable, satisfies type checker


def _log(msg: str) -> None:
    """Print a message with an ISO 8601 timestamp prefix."""
    print(f"{datetime.now().astimezone().isoformat()} {msg}")

from . import vcfparse as vcfp  # noqa: E402
from . import vcfwrite as vcfw  # noqa: E402
from . import breakends as bnds  # noqa: E402
from . import breakpoints as bps  # noqa: E402
from . import events as evts  # noqa: E402
from . import callset_specific as callset_specific  # noqa: E402
from . import utils as sv_utils  # noqa: E402
from . import chunk_match as chunk_match  # noqa: E402
from . import matched_breakpoints as mbps  # noqa: E402
from . import models as models  # noqa: E402
from . import delly2bnd as delly2bnd  # noqa: E402


def parse_preprocess_gnomad(
    sample_name: str,
    workflow_config: dict,
    verbose: bool = False
) -> pd.DataFrame:
    """
    Read and preprocess the gnomAD dataset for a reference sample.

    Args:
        sample_name (str): Name of the sample.
        workflow_config (dict): Configuration dictionary for the sample.
        verbose (bool): If True, print progress messages.
    """
    # Check if the sample is a reference sample
    if not workflow_config['is_reference']:
        raise ValueError(
            f"Sample {sample_name} is not a reference sample, "
            "cannot be preprocessed as reference"
        )

    # Check if the reference sample contains only 1 "callset"
    if len(workflow_config['callsets']) > 1:
        raise ValueError(
            f"Reference sample {sample_name} contains multiple callsets: "
            f"{list(workflow_config['callsets'].keys())}, must contain 1"
        )

    callset_key = next(iter(workflow_config['callsets']))
    callset = workflow_config['callsets'][callset_key]

    # Preprocess the reference sample
    if callset['source'] != 'gnomad':
        raise ValueError(
            f"Attempting to run preprocess_gnomad_sample for {sample_name} as reference, "
            "but it is not a gnomad sample"
        )

    gnomad_path = callset['path']
    gnomad_params = callset['params']

    if verbose:
        _log(f"Loading gnomad file {gnomad_path}")

    gnomad_vcf = VariantFile(gnomad_path)
    gnomad_df, gnomad_meta = vcfp.parse_vcf_to_df(
        vcf_file=gnomad_vcf,
        caller_params=gnomad_params
    )

    if verbose:
        _log(f"Preprocessing {gnomad_path}")

    gnomad_df = callset_specific.gnomad_preprocess(
        gnomad_df,
        caller_params=gnomad_params,
        verbose=verbose
    )

    if verbose:
        _log(f"Converting variants in {gnomad_path} to breakends")

    gnomad_bnd_df = callset_specific.gnomad_event_to_bnd(gnomad_df)

    if verbose:
        _log(f"Converting breakends in {gnomad_path} to breakpoints")

    gnomad_bp_df = bnds.bnd_to_bp(
        gnomad_bnd_df,
        force_diff_cols=['INFO_CIPOS_l', 'INFO_CIPOS_r']
    )

    return gnomad_bp_df


def parse_preprocess_delly(
    sample_name: str,
    workflow_config: dict,
    verbose: bool = False
) -> pd.DataFrame:
    """Read and preprocess the Delly dataset for a sample.

    Delly requires a custom parser because BND/TRA records are first
    converted to a standard BND VCF via delly2bnd, then parsed separately
    from the event records.

    Args:
        sample_name: Name of the sample.
        workflow_config: Configuration dictionary for the sample.
        verbose: If True, print progress messages.
    """
    callset = workflow_config['callsets']['delly']
    if 'ref_path' not in workflow_config:
        raise ValueError("Reference path not found in sample config")
    temp_dir = workflow_config['workflow_dir'] / 'temp'
    ref_path = workflow_config['ref_path']

    if not (temp_dir / 'delly_bnd.tmp.vcf').exists():
        temp_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        _log("Converting DELLY BND/TRA records via delly2bnd")
    delly2bnd.delly2bnd_noargparse(
        vcf=callset['path'],
        ref=ref_path,
        out=temp_dir / 'delly_bnd.tmp.vcf'
    )

    if verbose:
        _log(f"Loading DELLY BND VCF from {temp_dir / 'delly_bnd.tmp.vcf'}")
    delly_bnd_only_vcf = VariantFile(temp_dir / 'delly_bnd.tmp.vcf')
    delly_bnd_only_df, _ = vcfp.parse_vcf_to_df(
        vcf_file=delly_bnd_only_vcf,
        caller_params=callset['params']
    )
    delly_bnd_only_df = callset_specific.delly_preprocess_bnds_only(
        delly_bnd_only_df, caller_params=callset['params']
    )
    os.remove(temp_dir / 'delly_bnd.tmp.vcf')

    if verbose:
        _log(f"Loading DELLY event VCF from {callset['path']}")
    delly_vcf = VariantFile(callset['path'])
    delly_df, _ = vcfp.parse_vcf_to_df(
        vcf_file=delly_vcf,
        caller_params=callset['params']
    )
    delly_evt_only_df = delly_df[~delly_df['INFO_SVTYPE'].isin(['BND', 'TRA'])]
    delly_evt_only_df = callset_specific.delly_preprocess_evts_only(
        delly_evt_only_df, caller_params=callset['params']
    )
    if verbose:
        _log("Converting DELLY events to breakends")
    delly_evt_bnd_df = evts.event_to_bnd(
        delly_evt_only_df, bnd2_pos=callset['params']['bnd2_pos']
    )

    delly_bnd_df = pd.concat([delly_evt_bnd_df, delly_bnd_only_df], axis=0)

    if verbose:
        _log("Converting DELLY breakends to breakpoints")
    delly_bp_df = bnds.bnd_to_bp(
        delly_bnd_df, force_diff_cols=['INFO_CIPOS_l', 'INFO_CIPOS_r']
    )
    return delly_bp_df


def parse_preprocess_vcf(
    callset_name: str,
    callset: dict,
    verbose: bool = False
) -> pd.DataFrame:
    """Generic VCF-to-breakpoints preprocessor driven by callset_specific.parse_recipes.

    Handles three VCF layouts via the recipe's bnd_svtypes field:
      None  - entire VCF is BND (no event_to_bnd step)
      []    - entire VCF is events (all rows go through event_to_bnd)
      list  - rows with INFO_SVTYPE in the list are BND; the rest are events

    Args:
        callset_name: Name of the callset (must be a key in parse_recipes).
        callset: Callset dict containing 'path' and 'params'.
        verbose: If True, print progress messages.

    Returns:
        DataFrame of breakpoints.
    """
    recipe = callset_specific.parse_recipes[callset_name]

    if verbose:
        _log(f"Loading {callset_name} VCF from {callset['path']}")
    vcf = VariantFile(callset['path'])
    df, _ = vcfp.parse_vcf_to_df(vcf_file=vcf, caller_params=callset['params'])

    if recipe['preprocess_all']:
        if verbose:
            _log(f"Preprocessing all {callset_name} variants")
        df = recipe['preprocess_all'](df, caller_params=callset['params'])

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

        if recipe['preprocess_bnds'] and len(bnd_df) > 0:
            if verbose:
                _log(f"Preprocessing {callset_name} BND variants")
            bnd_df = recipe['preprocess_bnds'](bnd_df, caller_params=callset['params'])

        if recipe['preprocess_evts'] and len(evt_df) > 0:
            if verbose:
                _log(f"Preprocessing {callset_name} event variants")
            evt_df = recipe['preprocess_evts'](evt_df, caller_params=callset['params'])

        if len(evt_df) > 0:
            if verbose:
                _log(f"Converting {callset_name} events to breakends")
            evt_bnd_df = evts.event_to_bnd(evt_df, bnd2_pos=callset['params']['bnd2_pos'])
            bnd_df = pd.concat([bnd_df, evt_bnd_df], axis=0) if len(bnd_df) > 0 else evt_bnd_df

    if verbose:
        _log(f"Converting {callset_name} breakends to breakpoints")
    force_diff = recipe['force_diff_cols']
    bp_df = bnds.bnd_to_bp(bnd_df, force_diff_cols=force_diff) if force_diff else bnds.bnd_to_bp(bnd_df)

    if recipe['post_bps']:
        bp_df = recipe['post_bps'](bp_df)

    return bp_df


# Maps derived callset names to (parent_callset, filter_function).
# Filter functions take a DataFrame and return a filtered DataFrame.
_derived_callset_config = {
    'gridss_H': {
        'parent': 'gridss',
        'filter': lambda df: df[df['FILTER_PASS'].astype(bool)]
    },
    'sniffles_H': {
        'parent': 'sniffles',
        'filter': lambda df: df[df['FILTER_PASS'].astype(bool)]
    },
    'sniffles_207_H': {
        'parent': 'sniffles_207',
        'filter': lambda df: df[df['FILTER_PASS'].astype(bool)]
    },
    'severus_H': {
        'parent': 'severus',
        'filter': lambda df: df[df['FILTER_PASS'].astype(bool)]
    },
    'cutesv_H': {
        'parent': 'cutesv',
        'filter': lambda df: df[df['FILTER_PASS'].astype(bool)]
    },
}


def parse_preprocess_raw(
    sample_name: str,
    workflow_config: dict,
    verbose: bool = False
) -> None:
    """
    Model-independent stage: parse caller VCFs into breakpoints and write
    raw (pre-common_bps_preprocess) CSVs to a shared directory.
    Derived callsets (e.g. gridss_H, sniffles_H) are produced in a second
    pass after their parent callsets have been written.

    Output directory: workflow_config['callset_bps_dir']
    """
    _custom_parsers = {
        "gnomad": parse_preprocess_gnomad,
        "delly": parse_preprocess_delly,
    }

    callset_bps_dir = workflow_config['callset_bps_dir']
    if not callset_bps_dir.exists():
        callset_bps_dir.mkdir(parents=True, exist_ok=True)

    # Pass 1: VCF-based callers
    for callset_name in workflow_config["callsets"].keys():
        if callset_name in _derived_callset_config:
            continue
        if callset_name not in _custom_parsers and \
                callset_name not in callset_specific.parse_recipes:
            raise ValueError(f"Callset '{callset_name}' has no parser registered "
                             "(not in callset_specific.parse_recipes or custom parsers)")
        output_path = callset_bps_dir / f'{callset_name}.raw_bps.csv'
        if output_path.exists():
            if verbose:
                _log(f"Raw breakpoints already exist at {output_path}, skipping")
            continue
        vcf_path = Path(workflow_config['callsets'][callset_name]['path'])
        if not vcf_path.exists():
            raise FileNotFoundError(
                f"VCF not found for '{callset_name}' at {vcf_path}. "
                "Run the caller for it first or remove it from SVE_CALLSETS "
                "amd the --callsets argument."
            )
        if verbose:
            _log(f"Parsing callset '{callset_name}' into raw breakpoints")
        if callset_name in _custom_parsers:
            bp_df = _custom_parsers[callset_name](sample_name, workflow_config, verbose)
        else:
            bp_df = parse_preprocess_vcf(
                callset_name,
                workflow_config['callsets'][callset_name],
                verbose
            )
        bp_df.to_csv(output_path, index=False)
        if verbose:
            _log(f"Saved raw breakpoints to {output_path}")

    # Pass 2: Derived callsets (filtered from parent raw_bps)
    for callset_name in workflow_config["callsets"].keys():
        if callset_name not in _derived_callset_config:
            continue
        config = _derived_callset_config[callset_name]
        output_path = callset_bps_dir / f'{callset_name}.raw_bps.csv'
        if output_path.exists():
            if verbose:
                _log(f"Raw breakpoints already exist at {output_path}, skipping")
            continue
        parent_path = callset_bps_dir / f'{config["parent"]}.raw_bps.csv'
        if not parent_path.exists():
            raise FileNotFoundError(
                f"Parent callset '{config['parent']}' raw breakpoints not found "
                f"at {parent_path}. Cannot derive '{callset_name}'."
            )
        if verbose:
            _log(f"Deriving '{callset_name}' from parent '{config['parent']}'")
        parent_df = pd.read_csv(parent_path, low_memory=False)
        bp_df = config['filter'](parent_df)
        if verbose:
            _log(f"Filtered {len(parent_df)} -> {len(bp_df)} breakpoints for '{callset_name}'")
        bp_df.to_csv(output_path, index=False)
        if verbose:
            _log(f"Saved raw breakpoints to {output_path}")


def parse_preprocess_model_specific(
    sample_name: str,
    workflow_config: dict,
    verbose: bool = False
) -> None:
    """
    Model-specific stage: read raw breakpoints from the shared callset_bps
    directory, apply common_bps_preprocess (which may add model-specific
    CUSTOM_* columns), and write processed CSVs to the workflow directory.
    """
    callset_bps_dir = workflow_config['callset_bps_dir']

    if not workflow_config['workflow_dir'].exists():
        workflow_config['workflow_dir'].mkdir(parents=True, exist_ok=True)

    for callset_name in workflow_config["callsets"].keys():
        raw_path = callset_bps_dir / f'{callset_name}.raw_bps.csv'
        if not raw_path.exists():
            raise FileNotFoundError(
                f"Raw breakpoints not found at {raw_path}. "
                "Run parse_preprocess_raw first."
            )
        if verbose:
            _log(f"Loading raw breakpoints for '{callset_name}' from {raw_path}")
        bp_df = pd.read_csv(raw_path, low_memory=False)
        bp_df = bps.common_bps_preprocess(bp_df, workflow_config['callsets'][callset_name]['params'])
        bp_df.to_csv(workflow_config['workflow_dir'] / f'{callset_name}.bps.csv', index=False)
        vcfw.write_vcf(bp_df, workflow_config['workflow_dir'] / f'{callset_name}.vcf', source=callset_name, sample_name=sample_name)
        if verbose:
            _log(f"Saved processed breakpoints for '{callset_name}' to {workflow_config['workflow_dir']}")


def match_sample_breakpoints(
    sample_name: str,
    workflow_config: dict,
    ground_truth_config: dict,
    max_dist: int = 200,
    insert_delta_threshold: float = 0.7,
    verbose: bool = False
) -> None:
    """
    For a given sample, load available breakpoint DataFrames from supported callsets,
    add the reference sample breakpoints, perform outer join matching for each training_set specified for sample, and save results.
    """
    if verbose:
        _log("Matching breakpoints for sample...")

    bp_df_dict = {}

    # For each callset in the sample, load the breakpoints into bp_df_dict
    for callset_name, callset_config in workflow_config['callsets'].items():
        if verbose:
            _log(f"Loading breakpoints for callset '{callset_name}'...")
        bp_df = pd.read_csv(workflow_config['workflow_dir'] / f'{callset_name}.bps.csv', low_memory=False)
        if verbose:
            _log(f"Loaded {len(bp_df)} breakpoints for '{callset_name}'.")

        bp_df_dict[callset_name] = bp_df

    # Always add reference breakpoints
    # The reference at the moment should always have a single callset
    # Check if true
    if len(ground_truth_config['callsets']) > 1:
        raise ValueError("Reference sample should have a single callset")

    ground_truth_callset_name = list(ground_truth_config['callsets'].keys())[0]
    if verbose:
        _log(f"Loading reference breakpoints for callset '{ground_truth_callset_name}'...")
    ref_bp_df = pd.read_csv(ground_truth_config['callsets'][ground_truth_callset_name]['svensemble_dir'] / 'bps.csv', low_memory=False)

    if verbose:
        if ref_bp_df is not None:
            _log(f"Loaded {len(ref_bp_df)} reference breakpoints for '{ground_truth_callset_name}'.")
        else:
            raise FileNotFoundError(f"No reference breakpoints found for '{ground_truth_callset_name}'.")

    bp_df_dict[ground_truth_callset_name] = ref_bp_df

    # Ensure output directory exists
    output_dir = workflow_config['workflow_dir']
    if not output_dir.exists():
        if verbose:
            _log(f"Creating output directory: {output_dir}")
        output_dir.mkdir(parents=True)


    # Perform breakpoint matching for each training set specified for sample using chunk_match
    for training_set_name, training_set_callsets in workflow_config['training_sets'].items():
        if verbose:
            _log(f"Performing outer join matching of breakpoints for training set: {training_set_name}...")
        matched_bp_df = chunk_match.outer_join_bps(
            bp_df_dict={
                callset_name: bp_df_dict[callset_name] for callset_name in training_set_callsets + [ground_truth_callset_name]
            },
            ref_callset=ground_truth_callset_name,
            insert_delta_threshold=insert_delta_threshold,
            max_dist=max_dist
        )
        if verbose:
            _log(f"Matching complete. Matched breakpoints: {len(matched_bp_df)}")
        # Save matched breakpoints to CSV
        output_csv_path = output_dir / 'matched_bps.csv'
        if verbose:
            _log(f"Saving matched breakpoints to: {output_csv_path}")
        matched_bp_df.to_csv(output_csv_path, index=False)

def pu_train_sample(
    sample_name: str,
    workflow_config: dict,
    ground_truth_config: dict,
    save_feature_matrix: bool = False,
    verbose: bool = False
) -> None:
    """
    Train a Positive-Unlabeled (PU) learning model for a given sample.
    """
    # Load matched breakpoints DataFrame for each training set

    if len(workflow_config['training_sets']) > 1:
        raise ValueError("Only one training set per workflow is supported")

    training_set_name, training_set_callsets = list(workflow_config['training_sets'].items())[0]
    joint_bp_path = workflow_config['workflow_dir'] / 'matched_bps.csv'
    if verbose:
        _log(f"Loading joint breakpoints from: {joint_bp_path}")
    try:
        joint_bp_df = _read_csv_with_retry(joint_bp_path, low_memory=False)
    except FileNotFoundError:
        if verbose:
            _log(f"Joint breakpoints not found at {joint_bp_path}, loading from {workflow_config['model']['matched_bps_path']}")
        joint_bp_df = _read_csv_with_retry(workflow_config['model']['matched_bps_path'], low_memory=False)

    # Preprocess for PU learning
    if verbose:
        _log("Preprocessing sample for PU learning...")
        _log(f"Reference name: {ground_truth_config['callsets'][list(ground_truth_config['callsets'].keys())[0]]['source']}")

    X, rf_fit = models.train_model(
        joint_bp_df,
        workflow_config,
        ground_truth_config,
        training_set_name,
        verbose=verbose
    )
    if save_feature_matrix:
        # Save the feature matrix to disk
        output_path = workflow_config['workflow_dir'] / 'feature_matrix.csv'
        if verbose:
            _log(f"Saving feature matrix to: {output_path}")
        pd.DataFrame(X).to_csv(output_path, index=True)
    if rf_fit is not None:
        # Save the model to disk
        output_path = workflow_config['workflow_dir'] / 'model.pkl'
        if verbose:
            _log(f"Saving PU model to: {output_path}")
        with open(output_path, 'wb') as f:
            pickle.dump(rf_fit, f)
    else:
        raise ValueError("No PU model trained for training set")
    return rf_fit

def pu_predict_sample(
    sample_name: str,
    workflow_config: dict,
    verbose: bool = False
) -> None:
    """
    Predict the probability of a breakpoint being a true SV for a given sample.
    """
    if len(workflow_config['training_sets']) > 1:
        raise ValueError("Only one training set per workflow is supported")
    if not (workflow_config['model']['feature_matrix_path']).exists():
        raise FileNotFoundError(f"Feature matrix not found at {workflow_config['model']['feature_matrix_path']}")
    X = _read_csv_with_retry(workflow_config['model']['feature_matrix_path'], low_memory=False)
    X.set_index([f'{callset_name}_INFO_EVENT' for callset_name in workflow_config['callsets'].keys()], inplace=True)
    if not (workflow_config['workflow_dir'] / 'model.pkl').exists():
        raise FileNotFoundError(f"Model dictionary not found at {workflow_config['workflow_dir'] / 'model.pkl'}")
    if verbose:
        _log(f"Loading model from: {workflow_config['workflow_dir'] / 'model.pkl'}")

    with open(workflow_config['workflow_dir'] / 'model.pkl', 'rb') as f:
        model_fit = pickle.load(f)

    if verbose:
        _log(f"Predicting probabilities for {len(X)} breakpoints")

    fillna = workflow_config['model']['training_params'].get('fillna', 0)
    fill_val = fillna if fillna is not None else np.nan
    X = X.reindex(columns=model_fit.feature_names_in_, fill_value=fill_val)

    y_pred = model_fit.predict_proba(X)[:,1]
    if verbose:
        _log(f"Saving predictions to: {workflow_config['workflow_dir'] / 'predictions.csv'}")
    y_pred = pd.DataFrame(y_pred, columns=[f'{list(workflow_config['training_sets'].keys())[0]}_{workflow_config['model']['name']}_pred'], index=X.index).reset_index(drop=False)

    # Load the matched breakpoints DataFrame (prefer local workflow dir,
    # fall back to the configured shared path from another model).
    local_matched_bps = workflow_config['workflow_dir'] / 'matched_bps.csv'
    configured_matched_bps = workflow_config['model']['matched_bps_path']
    if local_matched_bps.exists():
        matched_bp_df = _read_csv_with_retry(local_matched_bps, low_memory=False)
    elif configured_matched_bps.exists():
        matched_bp_df = _read_csv_with_retry(configured_matched_bps, low_memory=False)
    else:
        raise FileNotFoundError(
            f"Matched breakpoints not found at {local_matched_bps} or {configured_matched_bps}"
        )

    svensemble2_calls_df = mbps.get_svensemble2_calls(
        matched_bp_df,
        y_pred
    )
    svensemble2_calls_df.to_csv(workflow_config['workflow_dir'] / 'svensemble2_calls.csv', index=True)
    vcfw.write_vcf(svensemble2_calls_df, workflow_config['workflow_dir'] / 'svensemble2_calls.vcf', source='svensemble2', sample_name=sample_name)
