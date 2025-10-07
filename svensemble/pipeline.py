#! /usr/bin/env python3

#SBATCH --job-name=svensemble2
#SBATCH --output=svensemble2.out
#SBATCH --error=svensemble2.err
#SBATCH --time=10:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G


"""
A top-level script to run the SVEnsemble pipeline.
This module provides preprocessing and matching utilities for SV callsets.
"""

import os
import sys
import pickle
from pathlib import Path

import pandas as pd
import sklearn.model_selection as skm
from pysam import VariantFile

# Set working directory and sys.path for local svensemble module
WORKDIR = "/vast/ai_projects/Papenfuss_SVEnsemble2"
os.chdir(WORKDIR)
if WORKDIR not in sys.path:
    sys.path.insert(0, WORKDIR)

import svensemble.vcfparse as vcfp  # noqa: E402
import svensemble.breakends as bnds  # noqa: E402
import svensemble.events as evts  # noqa: E402
import svensemble.caller_specific as caller_specific  # noqa: E402
import svensemble.utils as sv_utils  # noqa: E402
import svensemble.chunk_match as chunk_match  # noqa: E402
import svensemble.matched_breakpoints as mbps  # noqa: E402
import svensemble.models as models  # noqa: E402
import svensemble.delly2bnd as delly2bnd  # noqa: E402


default_ref_path = Path("/home/users/allstaff/shikov.v/projects/Papenfuss_SVEnsemble2/data/hg38/downloads/hg38.no_decoys.fa")


def parse_preprocess_gnomad(
    sample_name: str,
    sample_config: dict,
    verbose: bool = False
) -> None:
    """
    Read and preprocess the gnomAD dataset for a reference sample.

    Args:
        sample_name (str): Name of the sample.
        sample_config (dict): Configuration dictionary for the sample.
        verbose (bool): If True, print progress messages.
    """
    # Check if the sample is a reference sample
    if not sample_config['is_reference']:
        raise ValueError(
            f"Sample {sample_name} is not a reference sample, "
            "cannot be preprocessed as reference"
        )

    # Check if the reference sample contains only 1 "callset"
    if len(sample_config['callsets']) > 1:
        raise ValueError(
            f"Reference sample {sample_name} contains multiple callsets: "
            f"{list(sample_config['callsets'].keys())}, must contain 1"
        )

    callset_key = next(iter(sample_config['callsets']))
    callset = sample_config['callsets'][callset_key]

    # Preprocess the reference sample
    if callset['source'] != 'gnomad':
        raise ValueError(
            f"Attempting to run preprocess_gnomad_sample for {sample_name} as reference, "
            "but it is not a gnomad sample"
        )

    gnomad_path = callset['path']
    gnomad_params = callset['params']

    if verbose:
        print(f"Loading gnomad file {gnomad_path}")

    gnomad_vcf = VariantFile(gnomad_path)
    gnomad_df, gnomad_meta = vcfp.parse_vcf_to_df(
        vcf_file=gnomad_vcf,
        caller_params=gnomad_params
    )

    if verbose:
        print(f"Preprocessing {gnomad_path}")

    gnomad_df = caller_specific.gnomad_preprocess(
        gnomad_df,
        caller_params=gnomad_params,
        verbose=verbose
    )

    if verbose:
        print(f"Converting variants in {gnomad_path} to breakends")

    gnomad_bnd_df = caller_specific.gnomad_event_to_bnd(gnomad_df)

    if verbose:
        print(f"Converting breakends in {gnomad_path} to breakpoints")

    gnomad_bp_df = bnds.bnd_to_bp(
        gnomad_bnd_df,
        force_diff_cols=['INFO_CIPOS_l', 'INFO_CIPOS_r']
    )

    output_dir = sample_config['sample_dir'] / 'svensemble2'
    output_path = output_dir / 'gnomad_bps.csv'

    if verbose:
        print(f"Saving breakpoints in {output_path}")

    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)

    gnomad_bp_df.to_csv(output_path)

    if verbose:
        print(f"Saved breakpoints in {output_path}")

def parse_preprocess_gridss(
    sample_name: str,
    sample_config: dict,
    verbose: bool = False
) -> None:
    """
    Read and preprocess the gridss dataset for a sample.

    Args:
        sample_name (str): Name of the sample.
        sample_config (dict): Configuration dictionary for the sample.
        verbose (bool): If True, print progress messages.
    """
    def _process_gridss_callset(callset: dict, output_dir: Path, verbose: bool = False) -> None:
        """
        Internal function to process a gridss callset and save breakpoints.

        Args:
            callset (dict): Callset dictionary containing 'path' and 'params'.
            output_dir (Path): Directory to save the output CSV.
            verbose (bool): If True, print progress messages.
        """
        if verbose:
            print(f"Loading GRIDSS VCF from {callset['path']}")
        gridss_vcf = VariantFile(callset['path'])
        gridss_df, gridss_meta = vcfp.parse_vcf_to_df(
            vcf_file=gridss_vcf,
            caller_params=callset['params']
        )
        if verbose:
            print("Preprocessing GRIDSS variants")
        gridss_df = caller_specific.gridss_preprocess(
            gridss_df,
            caller_params=callset['params'],
            verbose=verbose
        )
        if verbose:
            print("Converting GRIDSS breakends to breakpoints")
        gridss_bp_df = bnds.bnd_to_bp(gridss_df)

        # A very small proportion of breakpoints will have different QUAL values for the two breakends
        # If so, the bps_df will have two QUAL columns for each breakend
        # For now we drop those breakpoints
        if 'QUAL' not in gridss_bp_df.columns:
            try:
                gridss_bp_df = gridss_bp_df[gridss_bp_df['QUAL_bnd1'] == gridss_bp_df['QUAL_bnd2']]
                gridss_bp_df.loc[:, 'QUAL'] = gridss_bp_df['QUAL_bnd1']
            except KeyError:
                raise KeyError('No QUAL column found in GRIDSS breakpoints')

        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / 'gridss_bps.csv'
        if verbose:
            print(f"Saving breakpoints to {output_path}")
        gridss_bp_df.to_csv(output_path)
        if verbose:
            print(f"Saved breakpoints in {output_path}")


    if 'gridss' in sample_config.get('callsets', {}):
        output_dir = sample_config['callsets']['gridss']['svensemble_dir']
        _process_gridss_callset(
            callset=sample_config['callsets']['gridss'],
            output_dir=output_dir,
            verbose=verbose
        )

def parse_preprocess_manta_candidate(
    sample_name: str,
    sample_config: dict,
    verbose: bool = False
) -> None:
    """
    Read and preprocess the manta candidate dataset for a sample.

    Args:
        sample_name (str): Name of the sample.
        sample_config (dict): Configuration dictionary for the sample.
        verbose (bool): If True, print progress messages.
    """
    def _process_manta_candidate_callset(callset: dict, output_dir: Path, verbose: bool = False) -> None:
        """
        Internal function to process a manta candidate callset and save breakpoints.

        Args:
            callset (dict): Callset dictionary containing 'path' and 'params'.
            output_dir (Path): Directory to save the output CSV.
            verbose (bool): If True, print progress messages.
        """
        if verbose:
            print(f"Loading Manta candidate VCF from {callset['path']}")
        manta_candidate_vcf = VariantFile(callset['path'])
        manta_candidate_df, manta_candidate_meta = vcfp.parse_vcf_to_df(
            vcf_file=manta_candidate_vcf,
            caller_params=callset['params']
        )
        if verbose:
            print("Splitting Manta candidate variants into BND/INV and event-only dataframes")
        manta_candidate_bnd_only_df = manta_candidate_df[manta_candidate_df['INFO_SVTYPE'].isin(['BND', 'INV'])]
        manta_candidate_evt_only_df = manta_candidate_df[~manta_candidate_df['INFO_SVTYPE'].isin(['BND', 'INV'])]
        if verbose:
            print("Preprocessing Manta BND/INV variants")
        manta_candidate_bnd_only_df = caller_specific.manta_preprocess_bnds_only(
            manta_candidate_bnd_only_df,
            caller_params=callset['params']
        )
        if verbose:
            print("Preprocessing Manta event-only variants")
        manta_candidate_evt_only_df = caller_specific.manta_preprocess_evts_only(
            manta_candidate_evt_only_df,
            caller_params=callset['params']
        )
        if verbose:
            print("Converting Manta event-only variants to breakends")
        manta_candidate_evt_bnd_df = evts.event_to_bnd(manta_candidate_evt_only_df)
        manta_candidate_df = pd.concat([manta_candidate_bnd_only_df, manta_candidate_evt_bnd_df], axis=0)
        if verbose:
            print("Converting Manta breakends to breakpoints")
        manta_candidate_bp_df = bnds.bnd_to_bp(
            manta_candidate_df,
            force_diff_cols=['INFO_CIPOS_l', 'INFO_CIPOS_r']
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / 'manta_candidate_bps.csv'
        if verbose:
            print(f"Saving breakpoints to {output_path}")
        manta_candidate_bp_df.to_csv(output_path)
        if verbose:
            print(f"Saved breakpoints in {output_path}")

    if 'manta_candidate' in sample_config.get('callsets', {}):
        output_dir = sample_config['callsets']['manta_candidate']['svensemble_dir']
        _process_manta_candidate_callset(
            callset=sample_config['callsets']['manta_candidate'],
            output_dir=output_dir,
            verbose=verbose
        )

def parse_preprocess_manta_diploid(
    sample_name: str,
    sample_config: dict,
    verbose: bool = False
) -> None:
    """
    Read and preprocess the manta diploid dataset for a sample.

    Args:
        sample_name (str): Name of the sample.
        sample_config (dict): Configuration dictionary for the sample.
        verbose (bool): If True, print progress messages.
    """
    def _process_manta_diploid_callset(
        callset: dict,
        output_dir,
        verbose: bool = False
    ) -> None:
        """
        Internal function to process a manta diploid callset and save breakpoints.

        Args:
            callset (dict): Callset dictionary containing 'path' and 'params'.
            output_dir (Path): Directory to save the output CSV.
            verbose (bool): If True, print progress messages.
        """
        if verbose:
            print(f"Loading Manta diploid VCF from {callset['path']}")
        manta_diploid_vcf = VariantFile(callset['path'])
        manta_diploid_df, manta_diploid_meta = vcfp.parse_vcf_to_df(
            vcf_file=manta_diploid_vcf,
            caller_params=callset['params']
        )
        if verbose:
            print("Splitting Manta diploid variants into BND/INV and event-only dataframes")
        manta_bnd_only_df = manta_diploid_df[manta_diploid_df['INFO_SVTYPE'].isin(['BND', 'INV'])]
        manta_evt_only_df = manta_diploid_df[~manta_diploid_df['INFO_SVTYPE'].isin(['BND', 'INV'])]
        if verbose:
            print("Preprocessing Manta diploid BND/INV variants")
        manta_bnd_only_df = caller_specific.manta_preprocess_bnds_only(
            manta_bnd_only_df,
            caller_params=callset['params']
        )
        if verbose:
            print("Preprocessing Manta diploid event-only variants")
        manta_evt_only_df = caller_specific.manta_preprocess_evts_only(
            manta_evt_only_df,
            caller_params=callset['params']
        )
        if verbose:
            print("Converting Manta diploid event-only variants to breakends")
        manta_evt_bnd_df = evts.event_to_bnd(manta_evt_only_df)
        manta_diploid_df = pd.concat([manta_bnd_only_df, manta_evt_bnd_df], axis=0)
        if verbose:
            print("Converting Manta diploid breakends to breakpoints")
        manta_diploid_bp_df = bnds.bnd_to_bp(
            manta_diploid_df,
            force_diff_cols=['INFO_CIPOS_l', 'INFO_CIPOS_r']
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / 'manta_diploid_bps.csv'
        if verbose:
            print(f"Saving breakpoints to {output_path}")
        manta_diploid_bp_df.to_csv(output_path)
        if verbose:
            print(f"Saved breakpoints in {output_path}")

    if 'manta_diploid' in sample_config.get('callsets', {}):
        output_dir = sample_config['callsets']['manta_diploid']['svensemble_dir']
        _process_manta_diploid_callset(
            callset=sample_config['callsets']['manta_diploid'],
            output_dir=output_dir,
            verbose=verbose
        )

def parse_preprocess_delly(
    sample_name: str,
    sample_config: dict,
    verbose: bool = False
) -> None:
    """
    Read and preprocess the delly dataset for a sample.
    """

    def _process_delly_callset(callset: dict, ref_path: Path, output_dir: Path, verbose: bool = False) -> None:
        """
        Internal function to process a delly callset and save breakpoints.
        """
        if verbose:
            print(f"Creating breakpoints from DELLY BND/TRA calls and saving to {output_dir / 'delly_bps.csv'}")

        if not (output_dir / 'delly_bnd.vcf').exists():
            output_dir.mkdir(parents=True, exist_ok=True)

        delly2bnd.delly2bnd_noargparse(
            vcf=callset['path'],
            ref=ref_path,
            out=output_dir / 'delly_bnd.vcf'
        )

        if verbose:
            print(f"Loading DELLY VCF from {callset['path']}")
        delly_bnd_only_vcf = VariantFile(output_dir / 'delly_bnd.vcf')
        delly_bnd_only_df, delly_bnd_only_meta = vcfp.parse_vcf_to_df(
            vcf_file=delly_bnd_only_vcf,
            caller_params=callset['params']
        )
        delly_bnd_only_df = caller_specific.delly_preprocess_bnds_only(delly_bnd_only_df, caller_params=callset['params'])


        delly_vcf = VariantFile(callset['path'])
        delly_df, delly_meta = vcfp.parse_vcf_to_df(
            vcf_file=delly_vcf,
            caller_params=callset['params']
        )
        delly_evt_only_df = delly_df[~delly_df['INFO_SVTYPE'].isin(['BND', 'TRA'])]
        delly_evt_only_df = caller_specific.delly_preprocess_evts_only(delly_evt_only_df, caller_params=callset['params'])
        delly_evt_bnd_df = evts.event_to_bnd(delly_evt_only_df)

        delly_bnd_df = pd.concat([delly_evt_bnd_df, delly_bnd_only_df], axis=0)

        delly_bp_df = bnds.bnd_to_bp(delly_bnd_df, force_diff_cols=['INFO_CIPOS_l', 'INFO_CIPOS_r'])

        output_path = output_dir / 'delly_bps.csv'
        if verbose:
            print(f"Saving breakpoints to {output_path}")
        delly_bp_df.to_csv(output_path)
        if verbose:
            print(f"Saved breakpoints in {output_path}")

    if 'delly' in sample_config.get('callsets', {}):
        if 'ref_path' not in sample_config:
            raise ValueError("Reference path not found in sample config")
        output_dir = sample_config['callsets']['delly']['svensemble_dir']
        _process_delly_callset(
            callset=sample_config['callsets']['delly'],
            ref_path=sample_config['ref_path'],
            output_dir=output_dir,
            verbose=verbose
        )




def parse_preprocess_sample(
    sample_name: str,
    sample_config: dict,
    verbose: bool = False
) -> None:
    """
    Read and preprocess the sample.

    Args:
        sample_name (str): Name of the sample.
        sample_config (dict): Configuration dictionary for the sample.
        verbose (bool): If True, print progress messages.
    """
    caller_dispatch = {
        "gnomad": parse_preprocess_gnomad,
        "gridss": parse_preprocess_gridss,
        "manta_diploid": parse_preprocess_manta_diploid,
        "manta_candidate": parse_preprocess_manta_candidate,
        "delly": parse_preprocess_delly,
    }

    for callset_name in sample_config["callsets"].keys():
        if callset_name in caller_dispatch.keys():
            caller_dispatch[callset_name](sample_name, sample_config, verbose)
        else:
            raise ValueError(f"Callset {callset_name} not supported")


def load_gridss_breakpoints(sample_config: dict) -> pd.DataFrame | None:
    """Load GRIDSS breakpoints if available."""
    if 'gridss' in sample_config.get('callsets', {}):
        gridss_bp_path = (
            sample_config['sample_dir'] /
            'sv_calling' / 'gridss' / 'svensemble2' / 'gridss_bps.csv'
        )
        return pd.read_csv(gridss_bp_path)
    return None


def load_manta_candidate_breakpoints(sample_config: dict) -> pd.DataFrame | None:
    """Load Manta candidate breakpoints if available."""
    if 'manta_candidate' in sample_config.get('callsets', {}):
        manta_candidate_bp_path = (
            sample_config['sample_dir'] /
            'sv_calling' / 'manta' / 'svensemble2' / 'manta_candidate_bps.csv'
        )
        return pd.read_csv(manta_candidate_bp_path)
    return None


def load_manta_diploid_breakpoints(sample_config: dict) -> pd.DataFrame | None:
    """Load Manta diploid breakpoints if available."""
    if 'manta_diploid' in sample_config.get('callsets', {}):
        manta_diploid_bp_path = (
            sample_config['sample_dir'] /
            'sv_calling' / 'manta' / 'svensemble2' / 'manta_diploid_bps.csv'
        )
        return pd.read_csv(manta_diploid_bp_path)
    return None


def load_delly_breakpoints(sample_config: dict) -> pd.DataFrame | None:
    """Load DELLY breakpoints if available."""
    if 'delly' in sample_config.get('callsets', {}):
        delly_bp_path = (
            sample_config['sample_dir'] /
            'sv_calling' / 'delly' / 'svensemble2' / 'delly_bps.csv'
        )
        return pd.read_csv(delly_bp_path)
    return None

def load_gnomad_breakpoints(reference_sample_config: dict) -> pd.DataFrame:
    """Always load gnomAD reference breakpoints."""
    gnomad_bp_path = (
        reference_sample_config['sample_dir'] /
        'svensemble2' / 'gnomad_bps.csv'
    )
    return pd.read_csv(gnomad_bp_path)


def match_sample_breakpoints(
    sample_config: dict,
    reference_sample_config: dict,
    verbose: bool = False
) -> None:
    """
    For a given sample, load available breakpoint DataFrames from supported callsets,
    add the reference sample breakpoints, perform outer join matching for each training_set sprecified for sample, and save results.
    """
    if verbose:
        print("Matching breakpoints for sample...")

    bp_df_dict = {}

    # For each callset in the sample, load the breakpoints into bp_df_dict
    for callset_name, callset_config in sample_config['callsets'].items():
        if verbose:
            print(f"Loading breakpoints for callset '{callset_name}'...")
        bp_df = pd.read_csv(callset_config['svensemble_dir'] / f'{callset_name}_bps.csv', low_memory=False)
        if verbose:
            print(f"Loaded {len(bp_df)} breakpoints for '{callset_name}'.")

        bp_df_dict[callset_name] = bp_df

    # Always add reference breakpoints
    # The reference at the moment should always have a single callset
    # Check if true
    if len(reference_sample_config['callsets']) > 1:
        raise ValueError("Reference sample should have a single callset")

    reference_callset_name = list(reference_sample_config['callsets'].keys())[0]
    if verbose:
        print(f"Loading reference breakpoints for callset '{reference_callset_name}'...")
    ref_bp_df = pd.read_csv(reference_sample_config['callsets'][reference_callset_name]['svensemble_dir'] / f'{reference_callset_name}_bps.csv', low_memory=False)

    if verbose:
        if ref_bp_df is not None:
            print(f"Loaded {len(ref_bp_df)} reference breakpoints for '{reference_callset_name}'.")
        else:
            raise FileNotFoundError(f"No reference breakpoints found for '{reference_callset_name}'.")

    bp_df_dict[reference_callset_name] = ref_bp_df

    # Ensure output directory exists
    output_dir = sample_config['sample_dir'] / 'sv_calling' / 'svensemble2'
    if not output_dir.exists():
        if verbose:
            print(f"Creating output directory: {output_dir}")
        output_dir.mkdir(parents=True)


    # Perform breakpoint matching for each training set specified for sample using chunk_match
    for training_set_name, training_set_callsets in sample_config['training_sets'].items():
        if verbose:
            print(f"Performing outer join matching of breakpoints for training set: {training_set_name}...")
        matched_bp_df = chunk_match.outer_join_bps(
            bp_df_dict={
                callset_name: bp_df_dict[callset_name] for callset_name in training_set_callsets + [reference_callset_name]
            },
            ref_callset=reference_callset_name,
            insert_size_threshold=0.7,
            max_dist=200
        )
        if verbose:
            print(f"Matching complete. Matched breakpoints: {len(matched_bp_df)}")
        # Save matched breakpoints to CSV
        output_csv_path = output_dir / f'matched_bps_{training_set_name}.csv'
        if verbose:
            print(f"Saving matched breakpoints to: {output_csv_path}")
        matched_bp_df.to_csv(output_csv_path, index=False)



def pu_train_predict_sample(
    sample_name: str,
    sample_config: dict,
    ground_truth_sample_config: dict,
    verbose: bool = False
) -> None:
    """
    Train and save a Positive-Unlabeled (PU) learning model for a given sample.

    This function loads the joined breakpoint DataFrame for the sample, preprocesses it
    for PU learning, splits it into training and validation sets, trains a PU model,
    and saves the resulting model and predictions.

    Args:
        sample_name (str): Name of the sample.
        sample_config (dict): Configuration dictionary for the sample.
        reference_sample_name (str): Name of the reference sample.
        verbose (bool, optional): If True, print progress messages. Defaults to False.

    Returns:
        None
    """
    # Load matched breakpoints DataFrame for each training set
    unbiased_dict = {}
    for training_set_name, training_set_callsets in sample_config['training_sets'].items():
        joint_bp_path = (
            sample_config['sample_dir'] /
            'sv_calling' /
            'svensemble2' /
            f'matched_bps_{training_set_name}.csv'
        )
        if verbose:
            print(f"Loading joint breakpoints from: {joint_bp_path}")
        joint_bp_df = pd.read_csv(joint_bp_path, low_memory=False)

        # Preprocess for PU learning
        if verbose:
            print("Preprocessing sample for PU learning...")
            print(f"Reference name: {ground_truth_sample_config['callsets'][list(ground_truth_sample_config['callsets'].keys())[0]]['source']}")

        training_set_df = mbps.preprocess_matchedbps_for_pu(
            joint_bp_df,
            sample_config,
            ground_truth_sample_config['callsets'][list(ground_truth_sample_config['callsets'].keys())[0]]['source'],
            training_set_name,
            verbose=verbose
        )
        del joint_bp_df

        X, y = mbps.get_X_y_from_training_df(
            training_set_df,
            match_col='ref_match'
        )
        X_train, X_val, y_train, y_val = skm.train_test_split(
            X, y, train_size=0.8, random_state=0
        )
        training_set_splits = {
            'X': X,
            'X_train': X_train,
            'X_val': X_val,
            'y': y,
            'y_train': y_train,
            'y_val': y_val,
            'X_idx': training_set_df.index
        }
        del training_set_df

        if verbose:
            print(f"Training PU model for training set: {training_set_name}")
        rf_fit, y_pred = models.train_unbiased_rf_for_split(training_set_splits)
        # Ensure predictions are indexed like the input
        y_pred = pd.DataFrame(y_pred, index=training_set_splits['X_idx'])
        unbiased_dict[training_set_name] = {
            'RF_fit': rf_fit,
            'y': y_pred
        }

    # Save the unbiased_dict to disk
    output_path = (
        sample_config['sample_dir'] /
        'sv_calling' /
        'svensemble2' /
        'unbiased_dict.pkl'
    )
    if verbose:
        print(f"Saving PU model results to: {output_path}")
    with open(output_path, 'wb') as f:
        pickle.dump(unbiased_dict, f)


def svensemble2(sample_names: list[str], ground_truth_sample_name: str, ref_path: Path, verbose: bool = False, preprocess: bool = True, match: bool = True, train: bool = True) -> dict:
    """
    Accepts a list of sample names and a data directory.
    For each name, reads the outputs of supported callers from the specified directory,
    converts all outputs into breakpoints, matches them to each other and to the reference,
    and trains a PU model to predict the posterior probability of a breakpoint being a true SV.

    Saves:
        - dataframes with matched breakpoints for each sample in the data directory as svensemble2/matched_breakpoints.pkl
        - the trained models and the predictions for each sample in the data directory as svensemble2/unbiased_dict.pkl

    Args:
        sample_names: list of sample names to process
        ground_truth_sample_name: name of the ground truth sample
        ref_path: path to the reference genome
        data_dir: path to the data directory
        verbose: if True, print progress messages
    """
    # Create a config dict for the samples using the sample_names
    samples = {}
    for sample_name in sample_names:
        samples.update(sv_utils.create_config_dict(sample_name, ref_path=ref_path, is_ground_truth=False))
    # Also create a config dict for the ground truth sample
    samples.update(sv_utils.create_config_dict(ground_truth_sample_name, ref_path=ref_path, is_ground_truth=True))

    if preprocess:
        # Parse and preprocess the samples
        for sample_name, sample_config in samples.items():
            parse_preprocess_sample(sample_name, sample_config, verbose)

    if match:
        # Match the breakpoints for each sample
        for sample_name, sample_config in samples.items():
            if sample_name != ground_truth_sample_name:
                match_sample_breakpoints(sample_config, samples[ground_truth_sample_name], verbose)

    if train:
        # Train the PU models for each sample
        for sample_name, sample_config in samples.items():
            if sample_name != ground_truth_sample_name:
                pu_train_predict_sample(sample_name, sample_config, samples[ground_truth_sample_name], verbose)


if __name__ == '__main__':
    import argparse

    # Create the argument parser for command-line execution
    parser = argparse.ArgumentParser(
        description="Run the SVEnsemble2 pipeline for structural variant analysis."
    )
    parser.add_argument(
        '--sample_name',
        type=str,
        required=True,
        help="Name of the sample to process."
    )
    parser.add_argument(
        '--ground_truth_sample_name',
        type=str,
        required=True,
        help="Name of the reference sample."
    )
    parser.add_argument(
        '--ref_path',
        type=str,
        required=True,
        help="Path to the reference genome."
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help="Enable verbose output for progress messages."
    )
    parser.add_argument(
        '--preprocess',
        action='store_true',
        default=False,
        help="If set, perform preprocessing of input samples."
    )
    parser.add_argument(
        '--match',
        action='store_true',
        default=False,
        help="If set, perform breakpoint matching."
    )
    parser.add_argument(
        '--train',
        action='store_true',
        default=False,
        help="If set, train the PU model."
    )
    args = parser.parse_args()

    # Run the main SVEnsemble2 pipeline with the provided arguments
    svensemble2(
        sample_names=[args.sample_name],
        ground_truth_sample_name=args.ground_truth_sample_name,
        ref_path=args.ref_path,
        verbose=args.verbose,
        preprocess=args.preprocess,
        match=args.match,
        train=args.train
    )