"""
General-purpose CLI for SVEnsemble2.

The single command-line entry point for the package. Provides:
  * per-stage subcommands (preprocess-raw, preprocess, match, train, predict,
    ground-truth, benchmark-callset, shap) that accept explicit file paths.
    These are the units invoked by Nextflow modules.
  * a `run` subcommand that chains all stages against explicit full input
    paths and an explicit workdir. This is the standalone entry point for
    running the pipeline without a workflow manager -- it does not expect
    pre-staged files, files are copied/symlinked into the workdir as needed.
"""

import argparse
import sys
from argparse import Namespace
from pathlib import Path

from . import __version__


def main():
    parser = argparse.ArgumentParser(
        prog="svensemble",
        description="SVEnsemble2: structural variant ensemble calling toolkit",
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}"
    )
    subparsers = parser.add_subparsers(dest="subcommand")

    _register_preprocess_raw(subparsers)
    _register_preprocess(subparsers)
    _register_match(subparsers)
    _register_train(subparsers)
    _register_predict(subparsers)
    _register_benchmark_callset(subparsers)
    _register_shap(subparsers)
    _register_ground_truth(subparsers)
    _register_run(subparsers)
    _register_model_id(subparsers)

    args = parser.parse_args()
    if args.subcommand is None:
        parser.print_help()
        sys.exit(0)

    args.func(args)


# ---------------------------------------------------------------------------
# Shared argument helpers
# ---------------------------------------------------------------------------

def _add_verbose_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output.")


def _add_model_config_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--model_config", type=Path, default=None,
        help="Path to a model config override file (JSON/YAML, self-contained "
             "with 'base'). Branch-2 model input.",
    )


def _load_model_config(args) -> dict | None:
    """Load the --model_config override file into a dict, if provided."""
    path = getattr(args, "model_config", None)
    if path is None:
        return None
    from .config_io import load_config
    return load_config(path)


# ---------------------------------------------------------------------------
# Subcommand: preprocess-raw
# ---------------------------------------------------------------------------
#
# Parses a per-caller VCF into a raw breakpoints CSV. Ground-truth VCFs
# (gnomAD, gnomAD_pass, and any future ground-truth sets) are NOT routed
# through this subcommand; use `ground-truth` instead. The dispatch table
# below only covers actual per-sample SV callers.
#

def _register_preprocess_raw(subparsers):
    p = subparsers.add_parser(
        "preprocess-raw",
        help="Parse caller VCF into raw breakpoints CSV.",
    )
    p.add_argument("--vcf", type=Path, required=True, help="Input VCF file path.")
    p.add_argument("--caller", type=str, required=True, help="Caller name (e.g. gridss, manta_diploid, delly).")
    p.add_argument("--ref", type=Path, required=True, help="Reference FASTA path.")
    p.add_argument("--output", type=Path, required=True, help="Output raw_bps.csv path.")
    _add_verbose_arg(p)
    p.set_defaults(func=_cmd_preprocess_raw)


def _cmd_preprocess_raw(args):
    from . import pipeline
    from . import _config as sv_config

    workflow_config = sv_config.build_workflow_config(
        sample_name="sample",
        callers=[args.caller],
        model_name=None,
        ref_path=args.ref,
    )

    callset_name = args.caller
    callset = workflow_config["callsets"][callset_name]
    callset["path"] = args.vcf

    if callset_name == "delly":
        bp_df = pipeline.parse_preprocess_delly(
            sample_name="sample",
            workflow_config=workflow_config,
            verbose=args.verbose,
        )
    else:
        bp_df = pipeline.parse_preprocess_vcf(
            callset_name=callset_name,
            callset=callset,
            verbose=args.verbose,
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    bp_df.to_csv(args.output, index=False)

    if args.verbose:
        print(f"Wrote {len(bp_df)} raw breakpoints to {args.output}")


# ---------------------------------------------------------------------------
# Subcommand: preprocess
# ---------------------------------------------------------------------------

def _register_preprocess(subparsers):
    p = subparsers.add_parser(
        "preprocess",
        help="Apply model-specific common preprocessing to raw breakpoints.",
    )
    p.add_argument("--raw_bps", type=Path, required=True, help="Input raw_bps.csv path.")
    p.add_argument("--caller", type=str, required=True, help="Caller name.")
    p.add_argument("--model", type=str, default=None, help="Model name (for custom_cols).")
    p.add_argument("--sample", type=str, required=True, help="Sample name.")
    p.add_argument("--output", type=Path, required=True, help="Output directory for bps.csv and .vcf.")
    _add_model_config_arg(p)
    _add_verbose_arg(p)
    p.set_defaults(func=_cmd_preprocess)


def _cmd_preprocess(args):
    import pandas as pd
    from . import breakpoints as bps_mod
    from . import _config as sv_config
    from . import vcfwrite as vcfw

    workflow_config = sv_config.build_workflow_config(
        sample_name=args.sample,
        callers=[args.caller],
        model_name=args.model,
        model_config=_load_model_config(args),
    )

    callset = workflow_config["callsets"][args.caller]
    bp_df = pd.read_csv(args.raw_bps, low_memory=False)
    bp_df = bps_mod.common_bps_preprocess(bp_df, callset["params"])

    args.output.mkdir(parents=True, exist_ok=True)
    bps_path = args.output / f"{args.caller}.bps.csv"
    vcf_path = args.output / f"{args.caller}.vcf"

    bp_df.to_csv(bps_path, index=False)
    vcfw.write_vcf(bp_df, vcf_path, source=args.caller, sample_name=args.sample)

    if args.verbose:
        print(f"Wrote {len(bp_df)} processed breakpoints to {bps_path}")


# ---------------------------------------------------------------------------
# Subcommand: match
# ---------------------------------------------------------------------------

def _register_match(subparsers):
    p = subparsers.add_parser(
        "match",
        help="Outer-join breakpoint matching across callers + ground truth.",
    )
    p.add_argument("--bps", type=Path, nargs="+", required=True, help="Processed bps.csv files (one per caller).")
    p.add_argument("--ground_truth_bps", type=Path, required=True, help="Ground truth breakpoints CSV.")
    p.add_argument("--callers", type=str, nargs="+", required=True, help="Caller names (same order as --bps).")
    p.add_argument("--sample", type=str, required=True, help="Sample name.")
    p.add_argument("--model", type=str, default=None, help="Model name (for custom_cols).")
    p.add_argument("--ground_truth_name", type=str, default="gnomad", help="Ground truth set name.")
    p.add_argument("--max_dist", type=int, default=100, help="Maximum distance for matching.")
    p.add_argument("--insert_delta_threshold", type=float, default=0.7, help="Insert delta threshold.")
    p.add_argument("--output", type=Path, required=True, help="Output matched_bps.csv path.")
    _add_model_config_arg(p)
    _add_verbose_arg(p)
    p.set_defaults(func=_cmd_match)


def _cmd_match(args):
    import pandas as pd
    from . import chunk_match
    from . import _config as sv_config

    workflow_config = sv_config.build_workflow_config(
        sample_name=args.sample,
        callers=args.callers,
        model_name=args.model,
        model_config=_load_model_config(args),
        ground_truth_name=args.ground_truth_name,
        ground_truth_bps_path=args.ground_truth_bps,
    )

    if len(args.bps) != len(args.callers):
        raise ValueError(f"Number of --bps files ({len(args.bps)}) must match --callers ({len(args.callers)})")

    bp_df_dict = {}
    for callset, bps_path in zip(args.callers, args.bps):
        bp_df_dict[callset] = pd.read_csv(bps_path, low_memory=False)

    ground_truth_df = pd.read_csv(args.ground_truth_bps, low_memory=False)
    bp_df_dict[args.ground_truth_name] = ground_truth_df

    if args.verbose:
        print(f"Matching breakpoints: {list(bp_df_dict.keys())}")

    matched_bp_df = chunk_match.outer_join_bps(
        bp_df_dict=bp_df_dict,
        ref_callset=args.ground_truth_name,
        insert_delta_threshold=args.insert_delta_threshold,
        max_dist=args.max_dist,
        workflow_config=workflow_config
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    matched_bp_df.to_csv(args.output, index=False)

    if args.verbose:
        print(f"Wrote {len(matched_bp_df)} matched breakpoints to {args.output}")


# ---------------------------------------------------------------------------
# Subcommand: train
# ---------------------------------------------------------------------------

def _register_train(subparsers):
    p = subparsers.add_parser(
        "train",
        help="Train PU learning model from matched breakpoints.",
    )
    p.add_argument("--matched_bps", type=Path, default=None, help="Matched breakpoints CSV. Required unless --train_samples/--train_matched_bps are given.")
    p.add_argument("--model", type=str, required=True, help="Base preset / model name (e.g. unbiased_rf, unbiased_xgb, consensus). Human label when --model_config is given.")
    p.add_argument("--training_set", type=str, required=True, help="Training set name.")
    p.add_argument("--callers", type=str, nargs="+", required=True, help="Caller names in the matched_bps.")
    p.add_argument("--ground_truth_name", type=str, default="gnomad", help="Ground truth name.")
    p.add_argument("--sample", type=str, required=True, help="Sample name.")
    p.add_argument("--output_dir", type=Path, required=True, help="Output directory for model.pkl and feature_matrix.csv.")
    p.add_argument(
        "--train_samples", type=str, nargs="+", default=None,
        help="Sample names forming the pooled training set for concat/cross-training "
             "(model must have training_params['concat']=True in _model_configs.py). "
             "Paired positionally with --train_matched_bps; mutually exclusive with "
             "--matched_bps. Include --sample's own name here for leave-in (concat) "
             "training; omit it for leave-out (cross) training.",
    )
    p.add_argument(
        "--train_matched_bps", type=Path, nargs="+", default=None,
        help="Matched-breakpoints CSV per --train_samples entry (same order/length).",
    )
    _add_model_config_arg(p)
    _add_verbose_arg(p)
    p.set_defaults(func=_cmd_train)


def _cmd_train(args):
    import pickle
    import pandas as pd
    from . import models
    from . import _config as sv_config

    workflow_config = sv_config.build_workflow_config(
        sample_name=args.sample,
        callers=args.callers,
        model_name=args.model,
        training_set=args.training_set,
        model_config=_load_model_config(args),
        ground_truth_name=args.ground_truth_name,
    )

    pooled = args.train_samples is not None or args.train_matched_bps is not None
    if pooled and args.matched_bps is not None:
        raise ValueError("--matched_bps cannot be combined with --train_samples/--train_matched_bps")
    if pooled:
        if args.train_samples is None or args.train_matched_bps is None:
            raise ValueError("--train_samples and --train_matched_bps must be provided together")
        if len(args.train_samples) != len(args.train_matched_bps):
            raise ValueError("--train_samples and --train_matched_bps must have the same length")
        matched_bps_sources = dict(zip(args.train_samples, args.train_matched_bps))
        joint_bp_df = None
    elif args.matched_bps is not None:
        matched_bps_sources = None
        joint_bp_df = pd.read_csv(args.matched_bps, low_memory=False)
    else:
        raise ValueError("Either --matched_bps or --train_samples/--train_matched_bps must be provided")

    X, model_fit = models.train_model(
        joint_bp_df,
        workflow_config,
        args.training_set,
        verbose=args.verbose,
        matched_bps_sources=matched_bps_sources,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    model_path = args.output_dir / "model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model_fit, f)

    fm_path = args.output_dir / "feature_matrix.csv"
    X.to_csv(fm_path, index=True)

    if args.verbose:
        print(f"Saved model to {model_path}")
        print(f"Saved feature matrix ({X.shape}) to {fm_path}")


# ---------------------------------------------------------------------------
# Subcommand: predict
# ---------------------------------------------------------------------------

def _register_predict(subparsers):
    p = subparsers.add_parser(
        "predict",
        help="Run model prediction on feature matrix.",
    )
    p.add_argument("--model_pkl", type=Path, required=True, help="Trained model pickle.")
    p.add_argument("--feature_matrix", type=Path, required=True, help="Feature matrix CSV.")
    p.add_argument("--matched_bps", type=Path, required=True, help="Matched breakpoints CSV (for call reconstruction).")
    p.add_argument("--sample", type=str, required=True, help="Sample name.")
    p.add_argument("--callers", type=str, nargs="+", required=True, help="Caller names.")
    p.add_argument("--training_set", type=str, required=True, help="Training set name.")
    p.add_argument("--model", type=str, required=True, help="Model name.")
    p.add_argument("--output_dir", type=Path, required=True, help="Output directory.")
    _add_model_config_arg(p)
    _add_verbose_arg(p)
    p.set_defaults(func=_cmd_predict)


def _cmd_predict(args):
    import pickle
    import numpy as np
    import pandas as pd
    from . import matched_breakpoints as mbps
    from . import vcfwrite as vcfw
    from . import _config as sv_config

    workflow_config = sv_config.build_workflow_config(
        sample_name=args.sample,
        callers=args.callers,
        model_name=args.model,
        training_set=args.training_set,
        model_config=_load_model_config(args),
    )

    with open(args.model_pkl, "rb") as f:
        model_fit = pickle.load(f)

    X = pd.read_csv(args.feature_matrix, low_memory=False)
    index_cols = [c for c in X.columns if c.endswith("_INFO_EVENT")]
    if not index_cols:
        raise ValueError(
            f"Feature matrix at {args.feature_matrix} has no '<callset>_INFO_EVENT' index "
            f"columns. These are required for reconstructing calls and should have been "
            f"written by the training step; their absence indicates a broken or truncated "
            f"feature matrix. Found columns: {list(X.columns)[:20]}{'...' if len(X.columns) > 20 else ''}"
        )
    X.set_index(index_cols, inplace=True)

    training_params = workflow_config["model"]["training_params"]
    fillna = training_params.get("fillna", 0)
    fill_val = fillna if fillna is not None else np.nan
    X = X.reindex(columns=model_fit.feature_names_in_, fill_value=fill_val)

    y_pred = model_fit.predict_proba(X)[:, 1]
    pred_col = f"{args.training_set}_{workflow_config['model_id']}_pred"
    y_pred_df = pd.DataFrame(y_pred, columns=[pred_col], index=X.index).reset_index(drop=False)

    matched_bp_df = pd.read_csv(args.matched_bps, low_memory=False)
    svensemble2_calls_df = mbps.get_svensemble2_calls(matched_bp_df, y_pred_df)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    calls_csv = args.output_dir / "svensemble2_calls.csv"
    calls_vcf = args.output_dir / "svensemble2_calls.vcf"

    svensemble2_calls_df.to_csv(calls_csv, index=True)
    vcfw.write_vcf(svensemble2_calls_df, calls_vcf, source="svensemble2", sample_name=args.sample)

    if args.verbose:
        print(f"Wrote {len(svensemble2_calls_df)} calls to {calls_csv}")


# ---------------------------------------------------------------------------
# Subcommand: benchmark-callset
# ---------------------------------------------------------------------------

def _register_benchmark_callset(subparsers):
    p = subparsers.add_parser(
        "benchmark-callset",
        help="Benchmark a single callset (individual caller or svensemble2 ensemble) against a validation set.",
    )
    p.add_argument("--callset_bps", type=Path, required=True, help="Callset bps CSV (caller bps.csv or svensemble2 calls CSV).")
    p.add_argument("--caller", type=str, required=True, help="Callset name; 'svensemble2'-prefixed names are treated as ensemble predictions.")
    p.add_argument("--val_bps", type=Path, required=True, help="Validation breakpoints CSV.")
    p.add_argument("--val_name", type=str, required=True, help="Validation set name.")
    p.add_argument("--output_dir", type=Path, required=True, help="Output directory for the benchmark CSV.")
    p.add_argument("--svtypes", type=str, default=None, help="Optional '+'-separated SV types to keep (e.g. DEL+INS). Default: all types.")
    p.add_argument("--svlen_min", type=int, default=None, help="Optional inclusive minimum SV length. Default: no lower bound.")
    p.add_argument("--svlen_max", type=int, default=None, help="Optional inclusive maximum SV length. Default: no upper bound.")
    _add_verbose_arg(p)
    p.set_defaults(func=_cmd_benchmark_callset)


def _cmd_benchmark_callset(args):
    from . import benchmark as bench_mod

    args.output_dir.mkdir(parents=True, exist_ok=True)

    svtypes = args.svtypes.split("+") if args.svtypes else None

    bench_mod.run_single_callset_benchmark(
        callset_bps_path=args.callset_bps,
        callset_name=args.caller,
        val_bps_path=args.val_bps,
        val_name=args.val_name,
        output_dir=args.output_dir,
        svtypes=svtypes,
        svlen_min=args.svlen_min,
        svlen_max=args.svlen_max,
        verbose=args.verbose,
    )


# ---------------------------------------------------------------------------
# Subcommand: shap
# ---------------------------------------------------------------------------

def _register_shap(subparsers):
    p = subparsers.add_parser(
        "shap",
        help="Compute SHAP values for model interpretation.",
    )
    p.add_argument("--model_pkl", type=Path, required=True, help="Trained model pickle.")
    p.add_argument("--feature_matrix", type=Path, required=True, help="Feature matrix CSV.")
    p.add_argument("--sample", type=str, required=True, help="Sample name.")
    p.add_argument("--model", type=str, required=True, help="Model name.")
    p.add_argument("--training_set", type=str, required=True, help="Training set name.")
    p.add_argument("--output_dir", type=Path, required=True, help="Output directory for SHAP results.")
    _add_model_config_arg(p)
    _add_verbose_arg(p)
    p.set_defaults(func=_cmd_shap)


def _cmd_shap(args):
    from . import sv_shap

    model_config = _load_model_config(args)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    sv_shap.run_shap(
        model_pkl=args.model_pkl,
        feature_matrix=args.feature_matrix,
        sample_name=args.sample,
        model_name=args.model,
        training_set=args.training_set,
        output_dir=args.output_dir,
        model_config=model_config,
        verbose=args.verbose,
    )


# ---------------------------------------------------------------------------
# Subcommand: ground-truth
# ---------------------------------------------------------------------------
#
# All ground-truth preprocessing routes through this subcommand, regardless of
# source (gnomAD, gnomAD_pass, or any future set). The `ground_truth_name`
# argument keys two registries:
#   * callset_specific.ground_truth_recipes  -- preprocessing / event-to-bnd
#                                               step functions for the source
#   * _caller_params.callset_param_dict      -- VCF parsing parameters
# Adding a new ground truth is a matter of registering an entry in each of
# these; no new subcommand is required.
#

def _register_ground_truth(subparsers):
    p = subparsers.add_parser(
        "ground-truth",
        help="Preprocess ground truth VCF into filtered breakpoints CSV.",
    )
    p.add_argument("--vcf_path", type=Path, required=True, help="Ground truth VCF file.")
    p.add_argument("--output_path", type=Path, required=True, help="Output bps.csv path.")
    p.add_argument("--ground_truth_name", type=str, default="gnomad", help="Ground truth dataset name (must exist in callset_specific.ground_truth_recipes and _caller_params.callset_param_dict).")
    p.add_argument("--filters", type=str, default=None, help="Comma-separated FILTER values to retain. When unset, resolved from callset_param_dict[ground_truth_name]['filter_defaults']['filters']; if that is unset too, no FILTER filter is applied.")
    p.add_argument("--evidence", type=str, default=None, help="Comma-separated EVIDENCE types to require (e.g. SR,PE). When unset, resolved from filter_defaults['evidence']; else no filter.")
    p.add_argument("--svtypes", type=str, default=None, help="Comma-separated SV types to retain. When unset, resolved from filter_defaults['svtypes'], falling back to the recipe's supported_svtypes.")
    p.add_argument("--algorithms", type=str, default=None, help="Comma-separated ALGORITHMS to require (e.g. manta,depth). When unset, resolved from filter_defaults['algorithms']; else no filter.")
    p.add_argument("--min_af", type=float, default=None, help="Minimum allele frequency (inclusive). When unset, resolved from filter_defaults['min_af']; else no filter.")
    p.add_argument("--max_af", type=float, default=None, help="Maximum allele frequency (inclusive). When unset, resolved from filter_defaults['max_af']; else no filter.")
    p.add_argument("--svlen_lower", type=int, default=50, help="Inclusive minimum SV length (default 50).")
    p.add_argument("--svlen_upper", type=float, default=None, help="Inclusive maximum SV length. Default: no upper bound.")
    p.add_argument("--bothsides_support_only", action="store_true", default=False, help="Keep only variants with BOTHSIDES_SUPPORT flag.")
    p.add_argument("--no_canonical_chroms", action="store_true", default=False, help="Disable restriction to canonical chromosomes (chr1-22, X, Y, M).")
    p.add_argument("--force_rerun", action="store_true", default=False, help="Overwrite existing output.")
    p.add_argument("--no_manifest", action="store_true", default=False, help="Skip writing manifest.txt alongside output.")
    _add_verbose_arg(p)
    p.set_defaults(func=_cmd_ground_truth)


def _cmd_ground_truth(args):
    import numpy as np
    from . import ground_truth as gt_mod

    filters = args.filters.split(",") if args.filters else None
    evidence = args.evidence.split(",") if args.evidence else None
    svtypes = args.svtypes.split(",") if args.svtypes else None
    algorithms = args.algorithms.split(",") if args.algorithms else None

    upper = args.svlen_upper if args.svlen_upper is not None else np.inf
    svlens = (args.svlen_lower, upper)

    gt_mod.preprocess_ground_truth(
        vcf_path=args.vcf_path,
        output_path=args.output_path,
        ground_truth_name=args.ground_truth_name,
        filters=filters,
        evidence=evidence,
        svtypes=svtypes,
        algorithms=algorithms,
        min_af=args.min_af,
        max_af=args.max_af,
        svlens=svlens,
        bothsides_support_only=args.bothsides_support_only,
        canonical_chroms_only=not args.no_canonical_chroms,
        force_rerun=args.force_rerun,
        write_manifest=not args.no_manifest,
        verbose=args.verbose,
    )


# ---------------------------------------------------------------------------
# Subcommand: run (end-to-end standalone pipeline)
# ---------------------------------------------------------------------------
#
# Chains the per-stage subcommands in a single process. Accepts full input
# paths and an explicit `--workdir`; does not expect files to be pre-staged.
# Intermediate outputs land under workdir/{preprocess_raw,preprocess,ground_truth,match,train}
# and the final calls under `--output_dir` (defaults to workdir/predict).
#
# For workflow-manager use, invoke the per-stage subcommands directly instead
# of `run` -- Nextflow needs each stage as its own process for caching and
# parallelism.
#

def _register_run(subparsers):
    p = subparsers.add_parser(
        "run",
        help="End-to-end standalone run: preprocess-raw -> ... -> predict. Takes full input paths + workdir.",
    )
    p.add_argument("--sample", type=str, required=True, help="Sample name (used as identifier only).")
    p.add_argument("--workdir", type=Path, required=True, help="Working directory. Created if missing. Intermediate outputs land under this path.")
    p.add_argument("--ref", type=Path, required=True, help="Reference FASTA path.")
    p.add_argument("--callers", type=str, nargs="+", required=True, help="Caller names (same order as --caller_vcfs).")
    p.add_argument("--caller_vcfs", type=Path, nargs="+", required=True, help="Full paths to caller VCFs (same order as --callers).")
    grp = p.add_mutually_exclusive_group(required=True)
    grp.add_argument("--ground_truth_vcf", type=Path, default=None, help="Full path to a ground truth VCF (run through the ground-truth stage).")
    grp.add_argument("--ground_truth_bps", type=Path, default=None, help="Full path to a pre-computed ground truth bps.csv (skip the ground-truth stage).")
    p.add_argument("--ground_truth_name", type=str, default="gnomad", help="Ground truth dataset name.")
    p.add_argument("--model", type=str, required=True, help="Model name.")
    p.add_argument("--training_set", type=str, required=True, help="Training set name.")
    p.add_argument("--max_dist", type=int, default=100, help="Match: maximum distance for breakpoint matching.")
    p.add_argument("--insert_delta_threshold", type=float, default=0.7, help="Match: insert delta threshold.")
    p.add_argument("--output_dir", type=Path, default=None, help="Final output directory (defaults to WORKDIR/predict).")
    p.add_argument("--ground_truth_filters", type=str, default=None, help="Ground-truth: comma-separated FILTER values.")
    p.add_argument("--ground_truth_svtypes", type=str, default=None, help="Ground-truth: comma-separated SV types.")
    p.add_argument("--ground_truth_svlen_lower", type=int, default=50, help="Ground-truth: minimum SV length.")
    p.add_argument("--ground_truth_svlen_upper", type=float, default=None, help="Ground-truth: maximum SV length.")
    _add_model_config_arg(p)
    _add_verbose_arg(p)
    p.set_defaults(func=_cmd_run)


def _cmd_run(args):
    if len(args.callers) != len(args.caller_vcfs):
        raise ValueError(
            f"--callers ({len(args.callers)}) and --caller_vcfs ({len(args.caller_vcfs)}) must be the same length."
        )

    workdir = args.workdir.resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    output_dir = (args.output_dir or workdir / "predict").resolve()

    if args.verbose:
        print(f"[run] workdir: {workdir}")
        print(f"[run] output_dir: {output_dir}")

    raw_dir = workdir / "preprocess_raw"
    pre_dir = workdir / "preprocess"
    gt_dir = workdir / "ground_truth"
    match_dir = workdir / "match"
    train_dir = workdir / "train"
    for d in (raw_dir, pre_dir, gt_dir, match_dir, train_dir, output_dir):
        d.mkdir(parents=True, exist_ok=True)

    # Stage 1: preprocess-raw for each caller.
    raw_bps_paths = []
    for callset, vcf in zip(args.callers, args.caller_vcfs):
        raw_out = raw_dir / f"{callset}.raw_bps.csv"
        _cmd_preprocess_raw(Namespace(
            vcf=Path(vcf).resolve(),
            caller=callset,
            ref=Path(args.ref).resolve(),
            output=raw_out,
            verbose=args.verbose,
        ))
        raw_bps_paths.append(raw_out)

    # Stage 2: preprocess for each caller.
    bps_paths = []
    for callset, raw_out in zip(args.callers, raw_bps_paths):
        callset_pre_dir = pre_dir / callset
        _cmd_preprocess(Namespace(
            raw_bps=raw_out,
            caller=callset,
            model=args.model,
            model_config=args.model_config,
            sample=args.sample,
            output=callset_pre_dir,
            verbose=args.verbose,
        ))
        bps_paths.append(callset_pre_dir / f"{callset}.bps.csv")

    # Stage 3: ground truth (or symlink if a bps CSV was provided).
    ground_truth_bps = gt_dir / f"{args.ground_truth_name}.bps.csv"
    if args.ground_truth_vcf is not None:
        _cmd_ground_truth(Namespace(
            vcf_path=Path(args.ground_truth_vcf).resolve(),
            output_path=ground_truth_bps,
            ground_truth_name=args.ground_truth_name,
            filters=args.ground_truth_filters,
            evidence=None,
            svtypes=args.ground_truth_svtypes,
            algorithms=None,
            min_af=None,
            max_af=None,
            svlen_lower=args.ground_truth_svlen_lower,
            svlen_upper=args.ground_truth_svlen_upper,
            bothsides_support_only=False,
            no_canonical_chroms=False,
            force_rerun=False,
            no_manifest=False,
            verbose=args.verbose,
        ))
    else:
        src = Path(args.ground_truth_bps).resolve()
        if ground_truth_bps.exists() or ground_truth_bps.is_symlink():
            ground_truth_bps.unlink()
        ground_truth_bps.symlink_to(src)
        if args.verbose:
            print(f"[run] symlinked ground truth bps: {ground_truth_bps} -> {src}")

    # Stage 4: match.
    matched_bps = match_dir / "matched_bps.csv"
    _cmd_match(Namespace(
        sample=args.sample,
        model=args.model,
        model_config=args.model_config,
        bps=bps_paths,
        ground_truth_bps=ground_truth_bps,
        callers=args.callers,
        ground_truth_name=args.ground_truth_name,
        max_dist=args.max_dist,
        insert_delta_threshold=args.insert_delta_threshold,
        output=matched_bps,
        verbose=args.verbose,
    ))

    # Stage 5: train.
    _cmd_train(Namespace(
        matched_bps=matched_bps,
        model=args.model,
        model_config=args.model_config,
        training_set=args.training_set,
        callers=args.callers,
        ground_truth_name=args.ground_truth_name,
        sample=args.sample,
        output_dir=train_dir,
        train_samples=None, train_matched_bps=None,
        verbose=args.verbose,
    ))

    # Stage 6: predict.
    _cmd_predict(Namespace(
        model_pkl=train_dir / "model.pkl",
        feature_matrix=train_dir / "feature_matrix.csv",
        matched_bps=matched_bps,
        sample=args.sample,
        callers=args.callers,
        training_set=args.training_set,
        model=args.model,
        model_config=args.model_config,
        output_dir=output_dir,
        verbose=args.verbose,
    ))

    if args.verbose:
        print(f"[run] done. calls -> {output_dir / 'svensemble2_calls.csv'}")


# ---------------------------------------------------------------------------
# Subcommand: model-id
# ---------------------------------------------------------------------------

def _register_model_id(subparsers):
    p = subparsers.add_parser(
        "model-id",
        help="Print the reconstructible model_id for a model (preset name or --model_config).",
    )
    p.add_argument("--model", type=str, default=None, help="Preset / base name.")
    _add_model_config_arg(p)
    p.set_defaults(func=_cmd_model_id)


def _cmd_model_id(args):
    from . import _config as sv_config

    model_config = _load_model_config(args)
    if model_config is not None:
        resolved = sv_config.resolve_model_config(config=model_config)
    elif args.model:
        resolved = sv_config.resolve_model_config(name=args.model)
    else:
        raise SystemExit("Provide --model <preset> or --model_config <file>.")
    print(sv_config.model_id(resolved))
