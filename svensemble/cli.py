"""
General-purpose CLI for SVEnsemble2.

The single command-line entry point for the package. Provides:
  * per-stage subcommands (preprocess-raw, preprocess, match, train, predict,
    ground-truth, benchmark-callset, shap) that accept explicit file paths.
    These are the units invoked by Nextflow modules.
  * utilities: validation-set (preprocess a raw truth set into breakpoints)
    and model-id (print the hashed id for a preset or model config).
  * a `run` subcommand that chains all stages against explicit full input
    paths and an explicit workdir. This is the standalone entry point for
    running the pipeline without a workflow manager -- it does not expect
    pre-staged files, files are copied/symlinked into the workdir as needed.
"""

import argparse
import sys
from pathlib import Path

from . import __version__
from . import api
from . import io as sv_io


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
    _register_validation_set(subparsers)
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
    return sv_io.read_model_config(path)


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
    bp_df = api.preprocess_raw(args.vcf, args.caller, args.ref, verbose=args.verbose)
    sv_io.write_bps(bp_df, args.output)

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
    bp_df = api.preprocess(
        sv_io.read_bps(args.raw_bps), args.caller,
        model=args.model, model_config=_load_model_config(args),
    )
    bps_path, _ = sv_io.write_callset(bp_df, args.output, args.caller, args.sample)

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
    if len(args.bps) != len(args.callers):
        raise ValueError(f"Number of --bps files ({len(args.bps)}) must match --callers ({len(args.callers)})")

    caller_bps = {callset: sv_io.read_bps(path) for callset, path in zip(args.callers, args.bps)}
    matched_bp_df = api.match(
        caller_bps,
        sv_io.read_bps(args.ground_truth_bps),
        ground_truth_name=args.ground_truth_name,
        model=args.model,
        model_config=_load_model_config(args),
        max_dist=args.max_dist,
        insert_delta_threshold=args.insert_delta_threshold,
        verbose=args.verbose,
    )
    sv_io.write_bps(matched_bp_df, args.output)

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
    pooled = args.train_samples is not None or args.train_matched_bps is not None
    if pooled and args.matched_bps is not None:
        raise ValueError("--matched_bps cannot be combined with --train_samples/--train_matched_bps")
    if pooled:
        if args.train_samples is None or args.train_matched_bps is None:
            raise ValueError("--train_samples and --train_matched_bps must be provided together")
        if len(args.train_samples) != len(args.train_matched_bps):
            raise ValueError("--train_samples and --train_matched_bps must have the same length")
        matched_bps = dict(zip(args.train_samples, args.train_matched_bps))
    elif args.matched_bps is not None:
        matched_bps = sv_io.read_bps(args.matched_bps)
    else:
        raise ValueError("Either --matched_bps or --train_samples/--train_matched_bps must be provided")

    X, model_fit = api.train(
        matched_bps,
        args.callers,
        args.model,
        args.training_set,
        model_config=_load_model_config(args),
        ground_truth_name=args.ground_truth_name,
        verbose=args.verbose,
    )
    model_path = sv_io.write_model(model_fit, args.output_dir / sv_io.MODEL)
    fm_path = sv_io.write_feature_matrix(X, args.output_dir / sv_io.FEATURE_MATRIX)

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
    svensemble2_calls_df = api.predict(
        sv_io.read_model(args.model_pkl),
        sv_io.read_feature_matrix(args.feature_matrix),
        sv_io.read_bps(args.matched_bps),
        args.model,
        args.training_set,
        model_config=_load_model_config(args),
    )
    calls_csv, _ = sv_io.write_calls(svensemble2_calls_df, args.output_dir, args.sample)

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
    svtypes = args.svtypes.split("+") if args.svtypes else None

    api.benchmark_callset(
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
    p.add_argument("--use_gpu", action="store_true", default=False, help="Use GPUTreeExplainer (requires CUDA).")
    p.add_argument("--max_samples", type=int, default=None, help="Subsample the feature matrix to at most N rows.")
    p.add_argument("--top_n", type=int, default=20, help="Number of top features in the summary CSV (default 20).")
    _add_model_config_arg(p)
    _add_verbose_arg(p)
    p.set_defaults(func=_cmd_shap)


def _cmd_shap(args):
    api.shap(
        model_pkl=args.model_pkl,
        feature_matrix=args.feature_matrix,
        sample_name=args.sample,
        model_name=args.model,
        training_set=args.training_set,
        output_dir=args.output_dir,
        model_config=_load_model_config(args),
        use_gpu=args.use_gpu,
        max_samples=args.max_samples,
        top_n=args.top_n,
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
#   * _caller_params.CALLSET_PARAM_DICT      -- VCF parsing parameters
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
    p.add_argument("--filters", type=str, default=None, help="Comma-separated FILTER values to retain. When unset, resolved from CALLSET_PARAM_DICT[ground_truth_name]['filter_defaults']['filters']; if that is unset too, no FILTER filter is applied.")
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

    filters = args.filters.split(",") if args.filters else None
    evidence = args.evidence.split(",") if args.evidence else None
    svtypes = args.svtypes.split(",") if args.svtypes else None
    algorithms = args.algorithms.split(",") if args.algorithms else None

    upper = args.svlen_upper if args.svlen_upper is not None else np.inf
    svlens = (args.svlen_lower, upper)

    api.preprocess_ground_truth(
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
# Subcommand: validation-set
# ---------------------------------------------------------------------------
#
# Converts a published truth set into the breakpoints CSV consumed by
# `benchmark-callset --val_bps`. --name must match the later --val_name.
#

def _register_validation_set(subparsers):
    from .validation_sets import VALIDATION_SET_NAMES

    p = subparsers.add_parser(
        "validation-set",
        help="Preprocess a published validation set into a breakpoints CSV for benchmark-callset.",
    )
    p.add_argument("--name", type=str, required=True, choices=VALIDATION_SET_NAMES, help="Validation set name; pass the same value as benchmark-callset --val_name.")
    p.add_argument("--raw_path", type=Path, required=True, help="Published truth file (VCF, BED or CSV depending on the set).")
    p.add_argument("--output", type=Path, required=True, help="Output breakpoints CSV path.")
    p.add_argument("--force_rerun", action="store_true", default=False, help="Overwrite existing output.")
    _add_verbose_arg(p)
    p.set_defaults(func=_cmd_validation_set)


def _cmd_validation_set(args):
    out = api.preprocess_validation_set(
        name=args.name,
        raw_path=args.raw_path,
        output_path=args.output,
        force_rerun=args.force_rerun,
    )
    if args.verbose:
        print(f"Wrote {args.name} validation breakpoints to {out}")


# ---------------------------------------------------------------------------
# Subcommand: run (end-to-end standalone pipeline)
# ---------------------------------------------------------------------------
#
# Chains every step in a single process via api.run. Accepts full input
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
    api.run(
        sample=args.sample,
        workdir=args.workdir,
        ref=args.ref,
        callers=args.callers,
        caller_vcfs=args.caller_vcfs,
        model=args.model,
        training_set=args.training_set,
        ground_truth_vcf=args.ground_truth_vcf,
        ground_truth_bps=args.ground_truth_bps,
        ground_truth_name=args.ground_truth_name,
        model_config=_load_model_config(args),
        max_dist=args.max_dist,
        insert_delta_threshold=args.insert_delta_threshold,
        output_dir=args.output_dir,
        ground_truth_filters=args.ground_truth_filters.split(",") if args.ground_truth_filters else None,
        ground_truth_svtypes=args.ground_truth_svtypes.split(",") if args.ground_truth_svtypes else None,
        ground_truth_svlen_lower=args.ground_truth_svlen_lower,
        ground_truth_svlen_upper=args.ground_truth_svlen_upper,
        verbose=args.verbose,
    )


# ---------------------------------------------------------------------------
# Subcommand: model-id
# ---------------------------------------------------------------------------

def _register_model_id(subparsers):
    p = subparsers.add_parser(
        "model-id",
        help="Print the reconstructible model_id for a model (preset name or --model_config).",
    )
    grp = p.add_mutually_exclusive_group()
    grp.add_argument("--model", type=str, default=None, help="Preset / base name.")
    _add_model_config_arg(grp)
    grp.add_argument("--all", action="store_true", default=False, help="Print 'preset<TAB>model_id' for every preset.")
    p.set_defaults(func=_cmd_model_id)


def _cmd_model_id(args):
    if args.all:
        for name, mid in api.list_preset_ids().items():
            print(f"{name}\t{mid}")
        return
    model_config = _load_model_config(args)
    if model_config is None and not args.model:
        raise SystemExit("Provide --model <preset> or --model_config <file>.")
    print(api.model_id(model=args.model, model_config=model_config))
