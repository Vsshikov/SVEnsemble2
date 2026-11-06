"""
SVEnsemble2 Python API.

Each pipeline step takes DataFrames and plain values and returns DataFrames
or a fitted model; none of them read or write files. Pair them with
``svensemble.io`` to load inputs and save outputs:

    raw = api.preprocess_raw(io.read_vcf(vcf, "manta_diploid"), "manta_diploid")
    bps = api.preprocess(raw, "manta_diploid", model="unbiased_xgb")
    matched = api.match({"manta_diploid": bps}, io.read_bps(gt_csv))
    X, model_fit = api.train(matched, ["manta_diploid"], "unbiased_xgb", "manta_diploid")
    calls = api.predict(model_fit, X, matched, "unbiased_xgb", "manta_diploid")

Ground truth, validation sets, benchmarking, SHAP and annotation are
file-to-file: they take input paths and an ``output_dir`` and write the
fixed filenames named in ``svensemble.io``. Argument names match the CLI
flags. ``annotate`` also takes the calls DataFrame and orchestrates
``svensemble.io`` (reading and writing) around the pure ``svensemble.annotate``.

``run`` chains every step for one sample, as ``svensemble run`` does. It
takes input paths, metadata and an ``output_dir``; intermediates go to a
private scratch directory (a tmpdir under ``$TMPDIR`` unless ``workdir`` is
given) and only the calls plus the artifacts named in ``save`` are written to
``output_dir``. A provenance record (io.RUN_PROVENANCE) is written to both.
"""

import copy
import shutil
import sys
import tempfile
import time
from collections.abc import Collection, Mapping, Sequence
from contextlib import ExitStack
from pathlib import Path

import numpy as np
import pandas as pd

from . import _config as sv_config
from . import _provenance
from . import annotate as sv_annotate
from . import breakpoints as bps_mod
from . import callset_specific
from . import chunk_match
from . import io as sv_io
from . import matched_breakpoints as mbps
from . import models
from ._caller_params import CALLSET_PARAM_DICT
from ._config import list_preset_ids
from ._constants import PRED_SUFFIX, SAMPLE_ID_COL
from .benchmark import run_single_callset_benchmark as benchmark_callset
from .ground_truth import preprocess_ground_truth
from .utils import atomic_path
from .validation_sets import preprocess_validation_set

__all__ = [
    "generate_config", "model_id", "list_preset_ids",
    "preprocess_raw", "preprocess", "match", "train", "predict",
    "preprocess_ground_truth", "preprocess_validation_set", "benchmark_callset", "shap",
    "annotate", "check_sample_sets", "run", "SAVE_ARTIFACTS",
]


def _workflow_config_anonymous_sample(**kwargs) -> dict:
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
        base: Preset to start from (a key of _model_configs.MODEL_CONFIGS).
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

def preprocess_raw(vcf_df: pd.DataFrame, caller: str, verbose: bool = False) -> pd.DataFrame:
    """Turn one caller's parsed VCF into raw breakpoints.

    Args:
        vcf_df: The caller's VCF as parsed by io.read_vcf; not modified.
        caller: Caller name (a key of _caller_params.CALLSET_PARAM_DICT whose
            source has a callset_specific.parse_recipes entry).
        verbose: Print progress messages.
    """
    if caller not in CALLSET_PARAM_DICT:
        raise ValueError(f"Unknown caller {caller!r}: no CALLSET_PARAM_DICT entry.")
    params = copy.deepcopy(CALLSET_PARAM_DICT[caller])
    if params.get("source") not in callset_specific.parse_recipes:
        raise ValueError(
            f"Caller {caller!r} (source {params.get('source')!r}) has no parse recipe; "
            f"known sources: {sorted(callset_specific.parse_recipes)}."
        )
    return callset_specific.vcf_df_to_raw_bps(vcf_df, params, verbose=verbose)


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
    workflow_config = _workflow_config_anonymous_sample(callers=[caller], model_name=model, model_config=model_config)
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
    workflow_config = _workflow_config_anonymous_sample(
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
    training_set: str | None = None,
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
        training_set: Training set name. Defaults to *callers* sorted and
            joined with '+', training on all of them. A TRAINING_SET_PRESETS
            name instead trains on that preset's callers, e.g. a subset.
        model_config: Optional model config override (with 'base').
        ground_truth_name: Ground truth set used as the positive labels.
        verbose: Print training progress.

    Returns:
        (X, model_fit): the feature matrix, indexed by ``<callset>_INFO_EVENT``,
        and the fitted model.
    """
    if training_set is None:
        # Sorted, like the default name, so the feature column order does not
        # depend on the order the callers were given in.
        callers = sorted(callers)
    training_set = sv_config.resolve_training_set(training_set, callers)
    workflow_config = _workflow_config_anonymous_sample(
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
    training_set: str | None = None,
    model_config: dict | None = None,
    callers: list[str] | None = None,
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
            Defaults to *callers* sorted and joined with '+', as in train.
        model_config: Optional model config override (with 'base').
        callers: Caller names; required only when training_set is not given.
    """
    training_set = sv_config.resolve_training_set(training_set, callers)
    resolved = _resolve_model(model, model_config)
    fillna = resolved["training_params"].get("fillna", 0)
    fill_val = fillna if fillna is not None else np.nan
    X = feature_matrix.reindex(columns=model_fit.feature_names_in_, fill_value=fill_val)

    y_pred = model_fit.predict_proba(X)[:, 1]
    pred_col = f"{training_set}_{sv_config.model_id(resolved)}{PRED_SUFFIX}"
    y_pred_df = pd.DataFrame(y_pred, columns=[pred_col], index=X.index).reset_index(drop=False)
    return mbps.get_svensemble2_calls(matched_bps, y_pred_df)


# ---------------------------------------------------------------------------
# File-to-file steps
# ---------------------------------------------------------------------------

def shap(
    model_pkl: Path,
    feature_matrix: Path,
    sample: str,
    model: str,
    training_set: str,
    output_dir: Path,
    model_config: dict | None = None,
    use_gpu: bool = False,
    max_samples: int | None = None,
    top_n: int = 20,
    verbose: bool = False,
) -> Path:
    """SHAP analysis of a trained model; see sv_shap.run_shap.

    Needs the optional ``shap`` dependency (``pip install svensemble[shap]``).
    """
    from . import sv_shap

    return sv_shap.run_shap(
        model_pkl=model_pkl,
        feature_matrix=feature_matrix,
        sample=sample,
        model=model,
        training_set=training_set,
        output_dir=output_dir,
        model_config=model_config,
        use_gpu=use_gpu,
        max_samples=max_samples,
        top_n=top_n,
        verbose=verbose,
    )


# ---------------------------------------------------------------------------
# Caller VCF inputs: sample checks and annotation
# ---------------------------------------------------------------------------

CallerVcfs = Mapping[str, Path | Sequence[Path]]
"""Caller name -> one VCF/BCF path, or several (e.g. one per sample)."""

_NA_LIKE_SAMPLES = frozenset({"", "NA", "N/A", "NaN", "nan", "None", "null", "NULL", "<NA>", "*"})


def _as_path_tuples(caller_vcfs: CallerVcfs) -> dict[str, tuple[Path, ...]]:
    """caller_vcfs with every value as a tuple of Paths."""
    return {
        caller: (Path(vcfs),) if isinstance(vcfs, (str, Path)) else tuple(Path(v) for v in vcfs)
        for caller, vcfs in caller_vcfs.items()
    }


def _compare_sample_sets(sets_by_caller: Mapping[str, Collection[str]]) -> tuple[str, ...]:
    """The one sample set shared by every caller that has samples.

    Raises if callers differ, or if a name is one pandas would read back from
    CSV as NaN (or the ground-truth placeholder '*'). Callers with no samples
    are skipped. Returns the samples sorted.
    """
    with_samples = {caller: frozenset(s) for caller, s in sets_by_caller.items() if s}
    bad = sorted({s for samples in with_samples.values() for s in samples} & _NA_LIKE_SAMPLES)
    if bad:
        raise ValueError(f"Sample names {bad} are reserved or read back as missing values; rename them")
    distinct = set(with_samples.values())
    if len(distinct) > 1:
        detail = "; ".join(f"{caller}: {sorted(s)}" for caller, s in with_samples.items())
        raise ValueError(f"Caller VCFs must share one sample set, got {detail}")
    return tuple(sorted(next(iter(distinct)))) if distinct else ()


def check_sample_sets(caller_vcfs: CallerVcfs) -> tuple[str, ...]:
    """Samples shared by all caller VCFs, read from their #CHROM lines.

    A caller's samples are the union over its files; a sample in two files of
    one caller raises. Then _compare_sample_sets applies across callers.
    """
    sets_by_caller = {}
    for caller, paths in _as_path_tuples(caller_vcfs).items():
        per_file = {path: sv_io.parse_samples_from_vcf(path) for path in paths}
        seen: dict[str, Path] = {}
        for path, samples in per_file.items():
            for sample in samples:
                if sample in seen:
                    raise ValueError(f"{caller}: sample {sample!r} in both {seen[sample]} and {path}")
                seen[sample] = path
        sets_by_caller[caller] = set(seen)
    return _compare_sample_sets(sets_by_caller)


def _calls_with_samples(calls: pd.DataFrame, samples: tuple[str, ...]) -> pd.DataFrame:
    """Calls carrying SAMPLE_ID_COL, checked against the VCFs' samples.

    Calls written before the sample key existed have no SAMPLE_ID_COL; they
    come from single-sample runs, so they are accepted only when the VCFs have
    exactly one sample, which they are then assigned to.
    """
    if SAMPLE_ID_COL not in calls.columns:
        if len(samples) != 1:
            raise ValueError(
                f"Calls have no {SAMPLE_ID_COL} column, so they can only annotate single-sample "
                f"VCFs; the VCFs have samples {list(samples)}"
            )
        return calls.assign(**{SAMPLE_ID_COL: samples[0]})
    extra = sorted(set(calls[SAMPLE_ID_COL].dropna().astype(str)) - set(samples))
    if extra:
        raise ValueError(f"Calls have samples {extra} that are not in the caller VCFs {list(samples)}")
    return calls


def _check_annotatable(
    paths: Mapping[str, tuple[Path, ...]], output_dir: Path,
) -> tuple[tuple[str, ...], dict[str, tuple[Path, ...]]]:
    """(run samples, output paths) after the checks that need only the VCFs.

    Every caller needs an event-key rule, no input may be sites-only, sample
    sets must agree (check_sample_sets) and output names must not collide.
    """
    for caller in paths:
        sv_annotate.event_key_fn(caller)
    samples = check_sample_sets(paths)
    for caller, vcfs in paths.items():
        for vcf in vcfs:
            if not sv_io.parse_samples_from_vcf(vcf):
                raise ValueError(f"{caller}: {vcf} is sites-only; a FORMAT field cannot be added")
    out_paths = {
        caller: tuple(Path(output_dir) / sv_io.annotated_vcf_name(caller, p) for p in vcfs)
        for caller, vcfs in paths.items()
    }
    flat = [p for ps in out_paths.values() for p in ps]
    if len(set(flat)) != len(flat):
        raise ValueError(f"Annotated output names collide: {sorted(map(str, flat))}")
    return samples, out_paths


def annotate(calls: pd.DataFrame, caller_vcfs: CallerVcfs, output_dir: Path) -> dict[str, tuple[Path, ...]]:
    """Copies of the caller VCFs/BCFs with the ensemble score as FORMAT SVENSSC.

    Records are edited as text or BCF bytes, never re-serialised, so each
    output differs from its input only by the SVENSSC header line and field.
    Everything is validated before anything is written, and the outputs are
    all-or-nothing: if any file fails, none appears.

    Args:
        calls: Ensemble calls as predict returns them (or io.read_calls).
        caller_vcfs: Caller name -> VCF/BCF path(s), as for run.
        output_dir: Where io.ANNOTATED_VCF files go (created if missing).

    Returns:
        Caller name -> annotated paths, in input order.
    """
    paths = _as_path_tuples(caller_vcfs)
    sv_annotate.pred_column(calls)
    for caller in paths:
        if f"{caller}_INFO_EVENT" not in calls.columns:
            raise KeyError(f"Calls have no {caller}_INFO_EVENT column; was {caller!r} part of this model?")
    samples, out_paths = _check_annotatable(paths, output_dir)
    calls = _calls_with_samples(calls, samples)

    with ExitStack() as stack:
        for caller, vcfs in paths.items():
            for vcf, out in zip(vcfs, out_paths[caller]):
                tmp = stack.enter_context(atomic_path(out))
                with sv_io.open_variant_file(vcf) as (kind, compressed, header, body):
                    codec = sv_annotate.annotate_bcf if kind == "bcf" else sv_annotate.annotate_text
                    sv_io.write_annotated(codec(header, body, caller, calls), tmp, compressed)
    return out_paths


# ---------------------------------------------------------------------------
# End-to-end run
# ---------------------------------------------------------------------------

SAVE_ARTIFACTS = frozenset({"model", "feature_matrix", "matched_bps", "ground_truth", "callsets"})
"""Intermediates ``run`` can publish to output_dir besides the calls."""


def run(
    sample: str,
    caller_vcfs: CallerVcfs,
    ref: Path,
    model: str,
    output_dir: Path,
    *,
    training_set: str | None = None,
    ground_truth_vcf: Path | None = None,
    ground_truth_bps: Path | None = None,
    ground_truth_name: str = "gnomad",
    force_preprocess_ground_truth: bool = False,
    model_config: dict | None = None,
    max_dist: int = 100,
    insert_delta_threshold: float = 0.7,
    save: Collection[str] = (),
    workdir: Path | None = None,
    keep_workdir: bool = False,
    annotate_vcfs: bool = False,
    command: str | None = None,
    verbose: bool = False,
) -> dict[str, Path]:
    """Run every step for one sample; return the paths written to output_dir.

    Inputs are paths, metadata are plain values, and output_dir receives the
    calls (io.CALLS_CSV, io.CALLS_VCF) plus whatever *save* names, under their
    io filenames. Everything else is an intermediate written to scratch:

    * ``workdir`` given -- scratch is that directory (created if missing) and
      is never deleted; intermediates land under
      ``{preprocess_raw, preprocess/<caller>, ground_truth, match, train}``.
    * ``workdir`` None -- scratch is a new private directory under ``$TMPDIR``
      (``tempfile.mkdtemp``), deleted after a successful run unless
      *keep_workdir*. After a failure it is kept and its path printed to
      stderr.

    Each artifact is read back from its file before the next step, so the
    results equal running the stage subcommands one by one, as Nextflow does.
    A provenance record (io.RUN_PROVENANCE) is written to scratch and to
    output_dir at the start and rewritten at the end.

    Args:
        sample: Sample name; used in the VCF outputs.
        caller_vcfs: Caller name -> caller VCF, or several (e.g. one per
            sample); each caller's files are parsed and concatenated. Sample
            names must agree across callers (check_sample_sets).
        ref: Reference FASTA. Reserved for VCF writing; currently unused
            apart from the provenance record.
        model, training_set, model_config: As for train/predict.
        output_dir: Where the calls and saved artifacts go (created if missing).
            One run per output_dir: filenames are fixed and are overwritten.
        training_set: As in train; defaults to *callers* sorted and joined with '+'.
        ground_truth_vcf, ground_truth_bps: Exactly one. A VCF goes through
            preprocess_ground_truth with the registry's filter_defaults (for
            other filters run ``svensemble ground-truth`` and pass the bps).
            An existing ground-truth output in its directory is reused unless
            *force_preprocess_ground_truth*; with a tmpdir and without
            ``"ground_truth"`` in *save* it is rebuilt every run. A bps CSV
            is read in place.
        max_dist, insert_delta_threshold: Passed to match.
        save: Subset of SAVE_ARTIFACTS to write to output_dir.
            ``"ground_truth"`` requires ground_truth_vcf.
        annotate_vcfs: Also write each caller VCF annotated with the ensemble
            score (see annotate) to output_dir. Inputs are checked for this
            before any other work.
        command: The invoking command line, recorded in the provenance record
            (the CLI passes it; API callers usually leave it None).
        verbose: Print progress messages.

    Returns:
        ``calls_csv`` and ``calls_vcf``, plus for each saved artifact
        ``model``, ``feature_matrix``, ``matched_bps``, ``ground_truth`` and
        ``<caller>.bps`` / ``<caller>.vcf``; with *annotate_vcfs*,
        ``<caller>.annotated_vcf`` (``<caller>.annotated_vcf.<k>`` when the
        caller has several files).
    """
    caller_vcfs = {
        caller: tuple(p.resolve() for p in paths)
        for caller, paths in _as_path_tuples(caller_vcfs).items()
    }
    if not caller_vcfs:
        raise ValueError("caller_vcfs is empty; give at least one caller VCF.")
    save = frozenset(save)
    unknown = save - SAVE_ARTIFACTS
    if unknown:
        raise ValueError(f"Unknown save artifacts {sorted(unknown)}; choose from {sorted(SAVE_ARTIFACTS)}.")
    if (ground_truth_vcf is None) == (ground_truth_bps is None):
        raise ValueError("Provide exactly one of ground_truth_vcf or ground_truth_bps.")
    if "ground_truth" in save and ground_truth_bps is not None:
        raise ValueError("save='ground_truth' needs ground_truth_vcf; ground_truth_bps is already a file you own.")

    ref = Path(ref).resolve()
    output_dir = Path(output_dir).resolve()
    if annotate_vcfs:
        samples, _ = _check_annotatable(caller_vcfs, output_dir)
    else:
        samples = check_sample_sets(caller_vcfs)
    ground_truth_vcf = Path(ground_truth_vcf).resolve() if ground_truth_vcf is not None else None
    ground_truth_bps = Path(ground_truth_bps).resolve() if ground_truth_bps is not None else None
    resolved_model = _resolve_model(model, model_config)

    owned = workdir is None
    scratch = Path(tempfile.mkdtemp(prefix="svensemble_run_")) if owned else Path(workdir).resolve()
    record = _provenance.start(
        command=command,
        parameters={
            "sample": sample,
            "caller_vcfs": caller_vcfs,
            "ref": ref,
            "model": model,
            "training_set": training_set,
            "output_dir": output_dir,
            "ground_truth_vcf": ground_truth_vcf,
            "ground_truth_bps": ground_truth_bps,
            "ground_truth_name": ground_truth_name,
            "force_preprocess_ground_truth": force_preprocess_ground_truth,
            "max_dist": max_dist,
            "insert_delta_threshold": insert_delta_threshold,
            "save": sorted(save),
            "workdir": scratch,
            "workdir_is_tmp": owned,
            "keep_workdir": keep_workdir,
            "annotate_vcfs": annotate_vcfs,
            "samples": list(samples),
        },
        inputs={
            "caller_vcfs": {c: [_provenance.file_info(v) for v in vs] for c, vs in caller_vcfs.items()},
            "ref": _provenance.file_info(ref),
            "ground_truth_vcf": _provenance.file_info(ground_truth_vcf),
            "ground_truth_bps": _provenance.file_info(ground_truth_bps),
        },
        model={"model_id": sv_config.model_id(resolved_model), "config": resolved_model},
    )
    if verbose:
        print(f"[run] scratch: {scratch}{' (tmpdir)' if owned else ''}")
        print(f"[run] output_dir: {output_dir}")

    t0 = time.monotonic()
    error, outputs = None, None
    try:
        _provenance.write(record, [scratch, output_dir])
        outputs = _run_in(
            scratch, output_dir,
            sample=sample, caller_vcfs=caller_vcfs, ref=ref, model=model,
            training_set=training_set, ground_truth_vcf=ground_truth_vcf,
            ground_truth_bps=ground_truth_bps, ground_truth_name=ground_truth_name,
            force_preprocess_ground_truth=force_preprocess_ground_truth,
            model_config=model_config, max_dist=max_dist,
            insert_delta_threshold=insert_delta_threshold, save=save,
            annotate_vcfs=annotate_vcfs, verbose=verbose,
        )
        return outputs
    except BaseException as e:
        error = e
        raise
    finally:
        delete = owned and error is None and not keep_workdir
        _provenance.finish(record, error=error, outputs=outputs,
                           start_monotonic=t0, end_monotonic=time.monotonic())
        try:
            _provenance.write(record, [output_dir] if delete else [scratch, output_dir])
        except OSError as e:
            print(f"[run] could not write provenance record: {e}", file=sys.stderr)
        if delete:
            shutil.rmtree(scratch, ignore_errors=True)
        elif owned:
            print(f"[run] scratch kept at {scratch}", file=sys.stderr)


def _run_in(
    scratch: Path,
    output_dir: Path,
    *,
    sample: str,
    caller_vcfs: dict[str, tuple[Path, ...]],
    ref: Path,
    model: str,
    training_set: str,
    ground_truth_vcf: Path | None,
    ground_truth_bps: Path | None,
    ground_truth_name: str,
    force_preprocess_ground_truth: bool,
    model_config: dict | None,
    max_dist: int,
    insert_delta_threshold: float,
    save: frozenset[str],
    annotate_vcfs: bool,
    verbose: bool,
) -> dict[str, Path]:
    """The body of run: every step, reading each artifact back from its file.

    An artifact named in *save* is written straight to output_dir; the rest go
    to their scratch subdirectory. The calls always go to output_dir.
    """
    def artifact_dir(artifact: str, subdir: str) -> Path:
        return output_dir if artifact in save else scratch / subdir

    published = {}
    caller_bps = {}
    for caller, vcfs in caller_vcfs.items():
        vcf_df = pd.concat([sv_io.read_vcf(vcf, caller) for vcf in vcfs], ignore_index=True)
        raw_path = sv_io.write_bps(
            preprocess_raw(vcf_df, caller, verbose=verbose),
            scratch / "preprocess_raw" / sv_io.RAW_BPS.format(caller=caller),
        )
        bps_path, vcf_path = sv_io.write_callset(
            preprocess(sv_io.read_bps(raw_path), caller, model=model, model_config=model_config),
            artifact_dir("callsets", f"preprocess/{caller}"), caller, sample,
        )
        if "callsets" in save:
            published[f"{caller}.bps"] = bps_path
            published[f"{caller}.vcf"] = vcf_path
        caller_bps[caller] = sv_io.read_bps(bps_path)
        if verbose:
            print(f"[run] {caller}: {len(caller_bps[caller])} breakpoints -> {bps_path}")

    if ground_truth_vcf is not None:
        gt_path = preprocess_ground_truth(
            vcf=ground_truth_vcf,
            output_dir=artifact_dir("ground_truth", "ground_truth"),
            ground_truth_name=ground_truth_name,
            overwrite=force_preprocess_ground_truth,
            verbose=verbose,
        )
        if "ground_truth" in save:
            published["ground_truth"] = gt_path
    else:
        gt_path = ground_truth_bps

    matched_path = sv_io.write_bps(
        match(
            caller_bps, sv_io.read_bps(gt_path), ground_truth_name=ground_truth_name,
            model=model, model_config=model_config, max_dist=max_dist,
            insert_delta_threshold=insert_delta_threshold, verbose=verbose,
        ),
        artifact_dir("matched_bps", "match") / sv_io.MATCHED_BPS,
    )
    matched_bps = sv_io.read_bps(matched_path)

    X, model_fit = train(
        matched_bps, list(caller_vcfs), model, training_set, model_config=model_config,
        ground_truth_name=ground_truth_name, verbose=verbose,
    )
    model_path = sv_io.write_model(model_fit, artifact_dir("model", "train") / sv_io.MODEL)
    fm_path = sv_io.write_feature_matrix(X, artifact_dir("feature_matrix", "train") / sv_io.FEATURE_MATRIX)

    calls = predict(
        model_fit, sv_io.read_feature_matrix(fm_path), matched_bps, model, training_set,
        model_config=model_config, callers=list(caller_vcfs),
    )
    calls_csv, calls_vcf = sv_io.write_calls(calls, output_dir, sample)
    if verbose:
        print(f"[run] done. calls -> {calls_csv}")

    outputs = {"calls_csv": calls_csv, "calls_vcf": calls_vcf, **published}
    if annotate_vcfs:
        for caller, paths in annotate(calls, caller_vcfs, output_dir).items():
            if len(paths) == 1:
                outputs[f"{caller}.annotated_vcf"] = paths[0]
            else:
                outputs.update({f"{caller}.annotated_vcf.{k}": path for k, path in enumerate(paths)})
    for artifact, path in (("matched_bps", matched_path), ("model", model_path),
                           ("feature_matrix", fm_path)):
        if artifact in save:
            outputs[artifact] = path
    return outputs
