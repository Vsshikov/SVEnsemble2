import copy
import hashlib
import math
from pathlib import Path
from typing import TypedDict, NotRequired

from .io import canonical_json
from ._model_configs import MODEL_CONFIGS, DEFAULT_MODEL_NAME, TrainingParams, ModelConfig
from .utils import update
from ._caller_params import TRAINING_SET_PRESETS, CALLSET_PARAM_DICT
# ---------------------------------------------------------------------------
# Model config resolution + reconstructible identity (model_id)
#
# _model_configs.py holds pure preset data; the two-branch resolver, the hasher
# and model_id live here (they need the deep-merge `update` and io
# canonicalisation, and _model_configs must never import from utils).
# ---------------------------------------------------------------------------

_MODEL_ID_HASH_VERSION = "v2"

# ---------------------------------------------------------------------------
# TypedDict schemas
#
# These describe the shape of the nested config dicts that flow through the
# pipeline.  They add IDE autocomplete and static type-checking without
# changing any runtime behaviour (every instance is still a plain dict).
# ---------------------------------------------------------------------------

class ResolvedModelConfig(ModelConfig):
    """A resolved model config with derived keys."""
    model_id: str
    matched_bps_path: Path
    feature_matrix_path: Path

class CallsetParams(TypedDict, total=False):
    """Per-callset VCF parsing and feature parameters (from _caller_params)."""
    base_cols: list[str]
    filter_cols: list[str]
    info_cols_mandatory: list[str]
    info_cols_pred: list[str]
    custom_cols_pred: list[str]
    parse_format: bool
    format_cols: list[str]
    is_reference: bool
    qual_metric: str | None
    svlen_source: str
    bnd2_pos: str
    svlens: tuple[int | float, int | float]
    annotator_cols: dict[str, dict[str, tuple[str, ...]]]
    source: str
    # Ground-truth-specific optional keys
    filter_defaults: dict
    supported_svtypes: list[str]


class CallsetConfig(TypedDict):
    """One entry in WorkflowConfig['callsets']."""
    source: str
    path: Path
    params: CallsetParams


class GroundTruthConfig(TypedDict):
    """Ground truth sub-dict in WorkflowConfig."""
    name: str
    params: CallsetParams
    vcf_path: NotRequired[Path]
    bps_path: NotRequired[Path]


class WorkflowConfig(TypedDict):
    """Top-level config dict consumed by pipeline, models, and CLI."""
    sample_name: str
    callsets: dict[str, CallsetConfig]
    model: ResolvedModelConfig
    model_id: str
    training_sets: dict[str, list[str]]
    workflow_dir: Path
    callset_bps_dir: Path
    is_reference: bool
    ref_path: NotRequired[Path]
    ground_truth: NotRequired[GroundTruthConfig]

def _tag_constructor(tag_dict):
    """
    Generates a reproducible tag string using an arbitrary dict of format {'tag_name': 'tag_value'}.

    Matches benchmarkFilterTag in nf-core-svensemble (workflows/svensemble.nf), which
    names the benchmark filter directories, e.g. 'svlenmin-50.svtypes-DEL+INS'.
    None, +-inf and empty values are dropped; lists and '+'-joined strings are
    sorted and '+'-joined; integral floats print as ints. Returns 'unfiltered'
    when no tokens remain.
    """
    tokens = []
    for k, v in tag_dict.items():
        if isinstance(v, str) and '+' in v:
            v = v.split('+')
        if isinstance(v, (list, tuple, set, frozenset)):
            v = '+'.join(sorted(v))
        if isinstance(v, float):
            if math.isinf(v):
                continue
            if v.is_integer():
                v = int(v)
        if v is None or v == '':
            continue
        tokens.append(f"{k}-{v}")
    return '.'.join(sorted(tokens)) or 'unfiltered'

TAG_DICT_PRESET_BENCHMARK = {
    'svlenmin': None,
    'svlenmax': None,
    'svtypes': None,
}


def resolve_model_config(name: str | None = None, config: dict | None = None) -> ModelConfig:
    """Resolve a full model config (two-branch).

    Branch 1 -- ``resolve_model_config(name="unbiased_xgb")``: return the named
    preset from ``MODEL_CONFIGS`` (enriched, deep-copied).

    Branch 2 -- ``resolve_model_config(config={"base": "unbiased_xgb", ...})``:
    take the ``base`` preset and deep-merge the remaining keys of ``config``
    (``training_params`` overrides, plus optional ``name``/``description``
    labels) on top of it. ``base`` is preserved from the override.

    The resolved config's ``training_params`` fully determine the reconstructible
    ``model_id`` (see :func:`model_id`); ``name``/``description``/``base`` are
    human labels and do not affect identity.
    """
    if config is not None:
        base = config.get("base")
        if not base:
            raise ValueError("Model config override must specify a 'base' preset.")
        if base not in MODEL_CONFIGS:
            raise ValueError(
                f"Unknown base preset '{base}'. Known presets: {sorted(MODEL_CONFIGS)}"
            )
        resolved = copy.deepcopy(MODEL_CONFIGS[base])
        overrides = {k: v for k, v in config.items() if k != "base"}
        resolved = update(resolved, copy.deepcopy(overrides))
        resolved["base"] = base
        resolved.setdefault("name", base)
        return _validate_model_config(resolved)
    if name is not None:
        if name not in MODEL_CONFIGS:
            raise ValueError(
                f"Unknown model preset '{name}'. Known presets: {sorted(MODEL_CONFIGS)}"
            )
        return _validate_model_config(copy.deepcopy(MODEL_CONFIGS[name]))
    raise ValueError("resolve_model_config requires either name= or config=.")


def _validate_model_config(resolved: dict) -> dict:
    """Sanity-check a resolved model config; return it unchanged on success."""
    fn = resolved.get("training_params", {}).get("function")
    # Deferred import: models imports from this package, so import lazily to
    # avoid an import cycle at module load time.
    from .models import model_dispatch
    if fn not in model_dispatch:
        raise ValueError(
            f"Model config 'function' {fn!r} is not a known training function "
            f"(model_dispatch keys: {sorted(model_dispatch)})."
        )
    return resolved


def model_config_hash(resolved_cfg: dict, length: int = 8) -> str:
    """Reconstructible content hash of a resolved model config.

    Hashes ONLY the behavioural ``training_params``; human labels
    (name/description/base) and derived keys (model_id, *_path) are ignored, so
    the digest is stable and pollution-free even when passed a full
    workflow_config['model'] sub-dict. Canonicalised for cross-process stability.
    """
    payload = resolved_cfg.get("training_params", {})
    canonical = canonical_json(payload)
    digest = hashlib.blake2b(
        f"{_MODEL_ID_HASH_VERSION}:{canonical}".encode(), digest_size=16
    ).hexdigest()
    return digest[:length]


def model_id(resolved_cfg: dict) -> str:
    """Reconstructible model identity: ``{base}-{hash8}``."""
    return f"{resolved_cfg['base']}-{model_config_hash(resolved_cfg)}"


def list_preset_ids() -> dict:
    """Map every preset name -> its model_id. Display convenience for humans."""
    return {name: model_id(resolve_model_config(name=name)) for name in MODEL_CONFIGS}


def build_ground_truth_config(
    name: str,
    vcf_path: Path | None = None,
    bps_path: Path | None = None,
) -> GroundTruthConfig:
    """Build a standalone ground truth config dict.

    Used on its own for the ground truth preprocessing stage, and embedded
    into workflow configs under the 'ground_truth' key for match/train stages.

    Args:
        name: Ground truth dataset name (must exist in CALLSET_PARAM_DICT,
            e.g. 'gnomad', 'gnomad_pass').
        vcf_path: Optional path to the original ground truth VCF.
        bps_path: Optional path to the pre-processed ground truth bps CSV.

    Returns:
        Dict with 'name', 'params', and optionally 'vcf_path'/'bps_path'.

    Raises:
        ValueError: If name is not found in CALLSET_PARAM_DICT.
    """
    params = CALLSET_PARAM_DICT.get(name)
    if params is None:
        raise ValueError(
            f"Unknown ground truth name: {name!r}. "
            f"Available: {list(CALLSET_PARAM_DICT.keys())}"
        )
    cfg = GroundTruthConfig(
        name=name,
        params=copy.deepcopy(params),
        vcf_path=Path(vcf_path) if vcf_path is not None else None,
        bps_path=Path(bps_path) if bps_path is not None else None,
    )
    return cfg

def build_workflow_config(
    sample_name: str,
    callers: list[str] | None = None,
    model_name: str | None = None,
    training_set: str | None = None,
    ref_path: Path | None = None,
    model_config: dict | None = None,
    ground_truth_name: str | None = None,
    ground_truth_vcf_path: Path | None = None,
    ground_truth_bps_path: Path | None = None,
) -> WorkflowConfig:
    """
    Build a workflow_config dict for pipeline subcommands.

    Paths are all Path('.') (or bare filenames) -- the caller is expected
    to have staged files into the working directory (Nextflow work dir,
    or the user's cwd for standalone CLI invocations).

    training_set is treated as a free-form name: if it matches a preset in
    TRAINING_SET_PRESETS, the preset's caller list is used; otherwise the
    provided `callers` list is stored under that name.

    Args:
        sample_name: Sample identifier (str). Stored under 'sample_name'.
        callers: List of callset names present in this sample's matched_bps.
        model_name: Key in MODEL_CONFIGS; controls custom_cols_pred injection.
        training_set: Name to key the training_sets dict; matched against
            TRAINING_SET_PRESETS before falling back to `callers`.
        ref_path: Optional reference FASTA, stored verbatim under 'ref_path'.
        model_config: Optional freeform model config dict override.
        ground_truth_name: Ground truth dataset name. When provided, a
            'ground_truth' sub-dict (from build_ground_truth_config) is
            included in the returned config.
        ground_truth_vcf_path: Optional path to the ground truth VCF,
            stored in ground_truth config if ground_truth_name is set.
        ground_truth_bps_path: Optional path to the processed ground truth
            bps CSV, stored in ground_truth config if ground_truth_name is set.

    Returns:
        Dict with the shape consumed by pipeline.parse_preprocess_delly,
        pipeline.parse_preprocess_vcf, breakpoints.common_bps_preprocess and
        models.train_model.
    """
    if callers is None:
        callers = []

    # Two-branch model resolution: a freeform override (Branch 2, self-contained
    # with `base`) wins over a preset name (Branch 1); otherwise the default.
    if model_config is not None:
        model_cfg = resolve_model_config(config=model_config)
    else:
        model_cfg = resolve_model_config(name=model_name or DEFAULT_MODEL_NAME)
    mid = model_id(model_cfg)
    training_params = model_cfg['training_params']
    custom_cols = training_params.get('custom_cols', [])

    if training_set and training_set in TRAINING_SET_PRESETS:
        resolved_callers = TRAINING_SET_PRESETS[training_set]
    else:
        resolved_callers = callers

    callsets = {}
    for callset_name in resolved_callers:
        params = copy.deepcopy(CALLSET_PARAM_DICT.get(callset_name, {}))
        if custom_cols:
            params['custom_cols_pred'] = custom_cols
        callsets[callset_name] = {
            'source': callset_name,
            'path': Path(f'{callset_name}.vcf'),
            'params': params,
        }

    training_sets = {}
    if training_set:
        training_sets[training_set] = resolved_callers

    workflow_dir = Path('.')

    config = WorkflowConfig(
        sample_name=sample_name,
        callsets=callsets,
        callset_bps_dir=workflow_dir,
        workflow_dir=workflow_dir,
        is_reference=False,
        training_sets=training_sets,
        model=ResolvedModelConfig(
            **model_cfg,
            model_id=mid,
            matched_bps_path=workflow_dir / 'matched_bps.csv',
            feature_matrix_path=workflow_dir / 'feature_matrix.csv',
        ),
        model_id=mid,
    )

    if ref_path is not None:
        config['ref_path'] = ref_path

    if ground_truth_name is not None:
        config['ground_truth'] = build_ground_truth_config(
            ground_truth_name,
            vcf_path=ground_truth_vcf_path,
            bps_path=ground_truth_bps_path,
        )

    return config
