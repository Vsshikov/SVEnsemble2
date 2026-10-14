"""
SVEnsemble2 Python API.

Each pipeline step takes DataFrames and plain values and returns DataFrames
or a fitted model; apart from preprocess_raw parsing its VCF, none of them
read or write files. Pair them with ``svensemble.io`` to load inputs and
save outputs:

    raw = api.preprocess_raw(vcf, "manta_diploid", ref)
    bps = api.preprocess(raw, "manta_diploid", model="unbiased_xgb")
    matched = api.match({"manta_diploid": bps}, io.read_bps(gt_csv))
    X, model_fit = api.train(matched, ["manta_diploid"], "unbiased_xgb", "manta_diploid")
    calls = api.predict(model_fit, X, matched, "unbiased_xgb", "manta_diploid")

Ground truth, validation sets, benchmarking and SHAP are file-to-file and
are exposed here unchanged. ``run`` chains every step through io into a
workdir, as ``svensemble run`` does.
"""

import copy
from pathlib import Path

import numpy as np
import pandas as pd

from . import _config as sv_config
from . import breakpoints as bps_mod
from . import chunk_match
from . import io as sv_io
from . import matched_breakpoints as mbps
from . import models
from . import pipeline
from ._config import list_preset_ids
from .benchmark import run_single_callset_benchmark as benchmark_callset
from .ground_truth import preprocess_ground_truth
from .validation_sets import preprocess_validation_set

__all__ = [
    "generate_config", "model_id", "list_preset_ids",
    "preprocess_raw", "preprocess", "match", "train", "predict",
    "preprocess_ground_truth", "preprocess_validation_set", "benchmark_callset", "shap",
    "run",
]


def _workflow_config(**kwargs) -> dict:
    """build_workflow_config for steps that never use the sample name."""
    return sv_config.build_workflow_config(sample_name="sample", **kwargs)


def _resolve_model(model: str | None, model_config: dict | None) -> dict:
    """Resolve a model config override if given, else the named preset."""
    if model_config is not None:
        return sv_config.resolve_model_config(config=model_config)
    return sv_config.resolve_model_config(name=model)


# ---------------------------------------------------------------------------
# Model configs
# ---------------------------------------------------------------------------

def generate_config(
    base: str,
    training_params: dict | None = None,
    name: str | None = None,
    description: str | None = None,
    output: Path | None = None,
) -> dict:
    """Build a model config override, the form ``--model_config`` takes.

    The result holds *base* plus only the given keys; *training_params* are
    deep-merged onto the base preset's when the config is resolved. It is
    validated by resolving it, so an unknown base or training function raises
    ValueError here rather than at training time.

    Args:
        base: Preset to start from (a key of _model_configs.model_configs).
        training_params: Overrides for the base preset's training_params.
        name, description: Human labels; they do not affect the model_id.
        output: If given, write the config here (JSON, or YAML by suffix),
            tag-encoded so Paths, tuples and inf survive the round trip
            through io.read_model_config.
    """
    cfg = {"base": base}
    if name is not None:
        cfg["name"] = name
    if description is not None:
        cfg["description"] = description
    if training_params:
        cfg["training_params"] = copy.deepcopy(training_params)
    sv_config.resolve_model_config(config=cfg)
    if output is not None:
        sv_io.write_config(sv_io.dict_json_encode(cfg), output)
    return cfg


def model_id(model: str | None = None, model_config: dict | None = None) -> str:
    """The reconstructible ``{base}-{hash8}`` id for a preset or a config override."""
    if model is None and model_config is None:
        raise ValueError("Provide model (a preset name) or model_config.")
    return sv_config.model_id(_resolve_model(model, model_config))


# ---------------------------------------------------------------------------
# Pipeline steps
# ---------------------------------------------------------------------------

def preprocess_raw(vcf: Path, caller: str, ref: Path, verbose: bool = False) -> pd.DataFrame:
    """Parse one caller's VCF into raw breakpoints.

    Args:
        vcf: Caller VCF/BCF path.
        caller: Caller name (a key of callset_specific.parse_recipes, or 'delly').
        ref: Reference FASTA; only Delly uses it, to resolve BND records.
        verbose: Print progress messages.
    """
    workflow_config = _workflow_config(callers=[caller], ref_path=ref)
    callset = workflow_config["callsets"][caller]
    callset["path"] = vcf
    if caller == "delly":
        return pipeline.parse_preprocess_delly(
            sample_name="sample", workflow_config=workflow_config, verbose=verbose,
        )
    return pipeline.parse_preprocess_vcf(callset_name=caller, callset=callset, verbose=verbose)


def preprocess(
    raw_bps: pd.DataFrame,
    caller: str,
    model: str | None = None,
    model_config: dict | None = None,
) -> pd.DataFrame:
    """Apply the common breakpoint preprocessing for *caller*.

    The model only matters through its ``custom_cols`` training param, which
    adds caller columns to keep.
    """
    workflow_config = _workflow_config(callers=[caller], model_name=model, model_config=model_config)
    return bps_mod.common_bps_preprocess(raw_bps, workflow_config["callsets"][caller]["params"])


def match(
    caller_bps: dict[str, pd.DataFrame],
    ground_truth_bps: pd.DataFrame,
    ground_truth_name: str = "gnomad",
    model: str | None = None,
    model_config: dict | None = None,
    max_dist: int = 100,
    insert_delta_threshold: float = 0.7,
    verbose: bool = False,
) -> pd.DataFrame:
    """Outer-join the callers' breakpoints with the ground truth.

    Args:
        caller_bps: Processed breakpoints per caller name; not modified.
        ground_truth_bps: Ground truth breakpoints (from preprocess_ground_truth).
        ground_truth_name: Ground truth set name; becomes its column prefix.
        model, model_config: Only the ``add_topology`` training param is used.
        max_dist: Maximum breakpoint distance for a match.
        insert_delta_threshold: Insert-size similarity threshold for a match.
        verbose: Print the callsets being matched.
    """
    workflow_config = _workflow_config(
        callers=list(caller_bps), model_name=model, model_config=model_config,
        ground_truth_name=ground_truth_name,
    )
    bp_df_dict = {**caller_bps, ground_truth_name: ground_truth_bps}
    if verbose:
        print(f"Matching breakpoints: {list(bp_df_dict.keys())}")
    return chunk_match.outer_join_bps(
        bp_df_dict=bp_df_dict,
        ref_callset=ground_truth_name,
        insert_delta_threshold=insert_delta_threshold,
        max_dist=max_dist,
        workflow_config=workflow_config,
    )


def train(
    matched_bps: pd.DataFrame | dict[str, pd.DataFrame | Path],
    callers: list[str],
    model: str,
    training_set: str,
    model_config: dict | None = None,
    ground_truth_name: str = "gnomad",
    verbose: bool = False,
) -> tuple[pd.DataFrame, object]:
    """Train a model on matched breakpoints.

    Args:
        matched_bps: One sample's matched breakpoints, or for pooled (concat)
            models a mapping of sample name to matched breakpoints DataFrame
            or CSV path.
        callers: Caller names present in the matched breakpoints.
        model: Preset name; a human label when model_config is given.
        training_set: Training set name; a preset name selects its callers.
        model_config: Optional model config override (with 'base').
        ground_truth_name: Ground truth set used as the positive labels.
        verbose: Print training progress.

    Returns:
        (X, model_fit): the feature matrix, indexed by ``<callset>_INFO_EVENT``,
        and the fitted model.
    """
    workflow_config = _workflow_config(
        callers=callers, model_name=model, training_set=training_set,
        model_config=model_config, ground_truth_name=ground_truth_name,
    )
    if isinstance(matched_bps, pd.DataFrame):
        joint_bp_df, matched_bps_sources = matched_bps, None
    else:
        joint_bp_df, matched_bps_sources = None, dict(matched_bps)
    return models.train_model(
        joint_bp_df,
        workflow_config,
        training_set,
        verbose=verbose,
        matched_bps_sources=matched_bps_sources,
    )


def predict(
    model_fit,
    feature_matrix: pd.DataFrame,
    matched_bps: pd.DataFrame,
    model: str,
    training_set: str,
    model_config: dict | None = None,
) -> pd.DataFrame:
    """Score a feature matrix and reconstruct the ensemble calls.

    Args:
        model_fit: Fitted model from train.
        feature_matrix: Features indexed by ``<callset>_INFO_EVENT``, as train
            returns and io.read_feature_matrix restores. Columns the model was
            not trained on are dropped and missing ones filled with the
            model's ``fillna`` value.
        matched_bps: The matched breakpoints the features were built from.
        model: Preset name; a human label when model_config is given.
        training_set: Training set name; part of the prediction column name.
        model_config: Optional model config override (with 'base').
    """
    resolved = _resolve_model(model, model_config)
    fillna = resolved["training_params"].get("fillna", 0)
    fill_val = fillna if fillna is not None else np.nan
    X = feature_matrix.reindex(columns=model_fit.feature_names_in_, fill_value=fill_val)

    y_pred = model_fit.predict_proba(X)[:, 1]
    pred_col = f"{training_set}_{sv_config.model_id(resolved)}_pred"
    y_pred_df = pd.DataFrame(y_pred, columns=[pred_col], index=X.index).reset_index(drop=False)
    return mbps.get_svensemble2_calls(matched_bps, y_pred_df)


# ---------------------------------------------------------------------------
# File-to-file steps
# ---------------------------------------------------------------------------

def shap(
    model_pkl: Path,
    feature_matrix: Path,
    sample_name: str,
    model_name: str,
    training_set: str,
    output_dir: Path,
    model_config: dict | None = None,
    use_gpu: bool = False,
    max_samples: int | None = None,
    top_n: int = 20,
    plot: bool = False,
    verbose: bool = False,
) -> Path:
    """SHAP analysis of a trained model; see sv_shap.run_shap.

    Needs the optional ``shap`` dependency (``pip install svensemble[shap]``).
    """
    from . import sv_shap

    return sv_shap.run_shap(
        model_pkl=model_pkl,
        feature_matrix=feature_matrix,
        sample_name=sample_name,
        model_name=model_name,
        training_set=training_set,
        output_dir=output_dir,
        model_config=model_config,
        use_gpu=use_gpu,
        max_samples=max_samples,
        top_n=top_n,
        plot=plot,
        verbose=verbose,
    )


# ---------------------------------------------------------------------------
# End-to-end run
# ---------------------------------------------------------------------------

def run(
    sample: str,
    workdir: Path,
    ref: Path,
    callers: list[str],
    caller_vcfs: list[Path],
    model: str,
    training_set: str,
    ground_truth_vcf: Path | None = None,
    ground_truth_bps: Path | None = None,
    ground_truth_name: str = "gnomad",
    model_config: dict | None = None,
    max_dist: int = 100,
    insert_delta_threshold: float = 0.7,
    output_dir: Path | None = None,
    ground_truth_filters: list[str] | None = None,
    ground_truth_svtypes: list[str] | None = None,
    ground_truth_svlen_lower: int = 50,
    ground_truth_svlen_upper: float | None = None,
    verbose: bool = False,
) -> Path:
    """Run every step for one sample and return the path to the calls CSV.

    Intermediate outputs land under ``workdir/{preprocess_raw, preprocess/<caller>,
    ground_truth, match, train}`` and the calls under *output_dir* (default
    ``workdir/predict``), with the same filenames the per-stage subcommands
    write. Each artifact is read back from its file before the next step, so
    the results equal running the subcommands one by one, as Nextflow does.

    Args:
        sample: Sample name; used only in the VCF outputs.
        callers, caller_vcfs: Caller names and their VCFs, paired by position.
        ground_truth_vcf, ground_truth_bps: Exactly one. A VCF goes through
            preprocess_ground_truth (with the ground_truth_* filters); a
            preprocessed breakpoints CSV is symlinked into the workdir.
        model_config: Optional decoded model config override (with 'base').

    Remaining arguments are passed to the step of the same name.
    """
    if len(callers) != len(caller_vcfs):
        raise ValueError(
            f"callers ({len(callers)}) and caller_vcfs ({len(caller_vcfs)}) must be the same length."
        )
    if (ground_truth_vcf is None) == (ground_truth_bps is None):
        raise ValueError("Provide exactly one of ground_truth_vcf or ground_truth_bps.")

    workdir = Path(workdir).resolve()
    output_dir = Path(output_dir or workdir / "predict").resolve()
    raw_dir = workdir / "preprocess_raw"
    pre_dir = workdir / "preprocess"
    gt_dir = workdir / "ground_truth"
    match_dir = workdir / "match"
    train_dir = workdir / "train"
    for d in (raw_dir, pre_dir, gt_dir, match_dir, train_dir, output_dir):
        d.mkdir(parents=True, exist_ok=True)
    if verbose:
        print(f"[run] workdir: {workdir}")
        print(f"[run] output_dir: {output_dir}")

    caller_bps = {}
    for caller, vcf in zip(callers, caller_vcfs):
        raw_path = sv_io.write_bps(
            preprocess_raw(Path(vcf).resolve(), caller, Path(ref).resolve(), verbose=verbose),
            raw_dir / sv_io.RAW_BPS.format(caller=caller),
        )
        bps_path, _ = sv_io.write_callset(
            preprocess(sv_io.read_bps(raw_path), caller, model=model, model_config=model_config),
            pre_dir / caller, caller, sample,
        )
        caller_bps[caller] = sv_io.read_bps(bps_path)
        if verbose:
            print(f"[run] {caller}: {len(caller_bps[caller])} breakpoints -> {bps_path}")

    gt_path = gt_dir / sv_io.CALLSET_BPS.format(caller=ground_truth_name)
    if ground_truth_vcf is not None:
        preprocess_ground_truth(
            vcf_path=Path(ground_truth_vcf).resolve(),
            output_path=gt_path,
            ground_truth_name=ground_truth_name,
            filters=ground_truth_filters,
            svtypes=ground_truth_svtypes,
            svlens=(ground_truth_svlen_lower,
                    ground_truth_svlen_upper if ground_truth_svlen_upper is not None else np.inf),
            verbose=verbose,
        )
    else:
        src = Path(ground_truth_bps).resolve()
        if gt_path.exists() or gt_path.is_symlink():
            gt_path.unlink()
        gt_path.symlink_to(src)
        if verbose:
            print(f"[run] symlinked ground truth bps: {gt_path} -> {src}")

    matched_path = sv_io.write_bps(
        match(
            caller_bps, sv_io.read_bps(gt_path), ground_truth_name=ground_truth_name,
            model=model, model_config=model_config, max_dist=max_dist,
            insert_delta_threshold=insert_delta_threshold, verbose=verbose,
        ),
        match_dir / sv_io.MATCHED_BPS,
    )
    matched_bps = sv_io.read_bps(matched_path)

    X, model_fit = train(
        matched_bps, callers, model, training_set, model_config=model_config,
        ground_truth_name=ground_truth_name, verbose=verbose,
    )
    sv_io.write_model(model_fit, train_dir / sv_io.MODEL)
    fm_path = sv_io.write_feature_matrix(X, train_dir / sv_io.FEATURE_MATRIX)

    calls = predict(
        model_fit, sv_io.read_feature_matrix(fm_path), matched_bps, model, training_set,
        model_config=model_config,
    )
    calls_csv, _ = sv_io.write_calls(calls, output_dir, sample)
    if verbose:
        print(f"[run] done. calls -> {calls_csv}")
    return calls_csv
