"""
Utility functions for SVEnsemble
"""
import os
import copy
import hashlib
import json
import pandas as pd
from pathlib import Path
from collections import abc
import itertools
from ._caller_params import callset_param_dict, callset_path_configs, training_set_presets, default_reference_set, default_training_set
from ._sample_params import sample_val_sets, sample_val_set_template, default_model_name
from ._model_configs import model_configs, BASE_MODELS
from . import config_io
from typing import Dict, Any

def get_base_dir() -> Path:
    base = os.environ.get("SVENSEMBLE_BASE_DIR")
    if base:
        return Path(base)
    # Fallback: assume we're running from the project root
    return Path.cwd()

BASE_DIR = get_base_dir()
DATA_DIR = BASE_DIR / "data"

def update(d, u):
    for k, v in u.items():
        if isinstance(v, abc.Mapping):
            d[k] = update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


# ---------------------------------------------------------------------------
# Model config resolution + reconstructible identity (model_id)
#
# _model_configs.py holds pure preset data; the two-branch resolver, the hasher
# and model_id live here (they need the deep-merge `update` and config_io
# canonicalisation, and _model_configs must never import from utils).
# ---------------------------------------------------------------------------

_MODEL_ID_HASH_VERSION = "v1"


def resolve_model_config(name: str | None = None, config: dict | None = None) -> dict:
    """Resolve a full model config (two-branch).

    Branch 1 -- ``resolve_model_config(name="unbiased_xgb")``: return the named
    preset from ``model_configs`` (enriched, deep-copied).

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
        if base not in model_configs:
            raise ValueError(
                f"Unknown base preset '{base}'. Known presets: {sorted(model_configs)}"
            )
        resolved = copy.deepcopy(model_configs[base])
        overrides = {k: v for k, v in config.items() if k != "base"}
        resolved = update(resolved, copy.deepcopy(overrides))
        resolved["base"] = base
        resolved.setdefault("name", base)
        return _validate_model_config(resolved)
    if name is not None:
        if name not in model_configs:
            raise ValueError(
                f"Unknown model preset '{name}'. Known presets: {sorted(model_configs)}"
            )
        return _validate_model_config(copy.deepcopy(model_configs[name]))
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
    canonical = json.dumps(
        config_io._sanitize_for_json(payload), sort_keys=True, separators=(",", ":")
    )
    digest = hashlib.blake2b(
        f"{_MODEL_ID_HASH_VERSION}:{canonical}".encode(), digest_size=16
    ).hexdigest()
    return digest[:length]


def model_id(resolved_cfg: dict) -> str:
    """Reconstructible model identity: ``{base}-{hash8}``."""
    return f"{resolved_cfg['base']}-{model_config_hash(resolved_cfg)}"


def list_preset_ids() -> dict:
    """Map every preset name -> its model_id. Display convenience for humans."""
    return {name: model_id(resolve_model_config(name=name)) for name in model_configs}


def model_results_dir(
    base_dir,
    sample_name: str,
    model,
    training_set,
    ground_truth_name: str | None = None,
) -> Path:
    """Reconstruct a model's results directory from its config (fetch-by-config).

    ``model`` may be a preset name (str), a freeform override dict (with
    ``base``), or an already-resolved config. Mirrors the data-tree layout built
    by :func:`create_config_dict`, so a config + sample + training_set is enough
    to locate results in a Jupyter session without knowing the hash up front.
    """
    base_dir = Path(base_dir)
    if isinstance(model, str):
        resolved = resolve_model_config(name=model)
    elif isinstance(model, dict) and 'base' in model and 'training_params' in model:
        resolved = model
    else:
        resolved = resolve_model_config(config=model)
    mid = model_id(resolved)

    if isinstance(training_set, (list, tuple)):
        ts_name = '+'.join(training_set)
    elif training_set in training_set_presets:
        ts_name = training_set
    elif training_set in (None, 'default'):
        ts_name = default_training_set
    else:
        ts_name = training_set

    gt_suffix = ''
    if ground_truth_name and ground_truth_name != default_reference_set:
        gt_suffix = f'.gt_{ground_truth_name}'

    return base_dir / Path(
        f'data/{sample_name}/sv_calling/svensemble2/'
        f'{sample_name}.{mid}.{ts_name}{gt_suffix}'
    )


def list_models(base_dir, sample_name: str) -> list[dict]:
    """Discover existing svensemble2 result dirs for a sample.

    Returns one dict per directory with its path and, when a ``config_out.json``
    is present, the recorded ``model_id`` and human ``name`` -- the reverse of
    :func:`model_results_dir` (model_id -> human label).
    """
    base_dir = Path(base_dir)
    root = base_dir / f'data/{sample_name}/sv_calling/svensemble2'
    out: list[dict] = []
    if not root.exists():
        return out
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        info = {'dir': d, 'dirname': d.name}
        cfg_path = d / 'config_out.json'
        if cfg_path.exists():
            try:
                cfg = config_io.load_config(cfg_path)
                info['model_id'] = cfg.get('model_id')
                model_block = cfg.get('model')
                info['model_name'] = (
                    model_block.get('name') if isinstance(model_block, dict)
                    else cfg.get('model_name')
                )
            except Exception:
                pass
        out.append(info)
    return out


human_chroms = [f'chr{i}' for i in range(1, 23)] + ['chrX', 'chrY', 'chrM']

def _get_callset_dict(sample_name: str, callset_name: str, base_dir: Path | None = None) -> dict:
    """Build a callset config dict for a given callset and sample.

    Paths are resolved from callset_path_configs (_caller_params) using sample_name substitution.
    The params key always matches callset_name, so _H variants (gridss_H, sniffles_H)
    share paths with their base caller but carry separate parameter sets.
    """
    if base_dir is None:
        base_dir = BASE_DIR
    try:
        source, svensemble_dir_tmpl, path_tmpl = callset_path_configs[callset_name]
    except KeyError:
        raise KeyError(
            f'Unknown callset: {callset_name!r}. '
            f'Available: {list(callset_path_configs)}'
        )
    return {
        'source': source,
        'svensemble_dir': base_dir / Path(svensemble_dir_tmpl.format(s=sample_name)),
        'path': base_dir / Path(path_tmpl.format(s=sample_name)),
        'params': copy.deepcopy(callset_param_dict[callset_name]),
    }

def create_config_dict(
    sample_name: str,
    ref_path: Path | None = None,
    is_ground_truth: bool = False,
    training_set: str | list[str] = 'default',
    val_set: None | str | dict[str, Any] = None,
    model_name: str | None = None,
    additional_callsets: list[str] | None = None,
    callsets: list[str] | None = None,
    ground_truth_name: str | None = None,
    base_dir: Path | None = None
):
    """
    Based on _caller_params, create a config dictionary for the sample.

    The returned dictionary is keyed by sample_name and resolves all caller
    callset paths, parameters, training/validation sets and model configuration
    from _caller_params (and related modules) relative to base_dir.

    Two structurally different dictionaries are produced depending on
    is_ground_truth:

    - is_ground_truth=True returns a minimal "reference" config. It marks the
      sample as a reference (is_reference=True) and contains a single callset
      whose name equals sample_name (the ground-truth callset). No training
      sets, validation set or model are attached, and all other arguments
      (training_set, val_set, model_name, additional_callsets, callsets,
      ground_truth_name) are ignored.
    - is_ground_truth=False returns the full config used for training and
      prediction. It includes the resolved callsets, the selected training
      set(s), an optional validation set, the model configuration and a
      bps_path for every callset.

    When callsets is provided it is used as the exact callset list, bypassing
    the training-set-derived list and additional_callsets.

    Args:
        sample_name: Name of the sample, used as the top-level dictionary key
            and for resolving data paths.
        ref_path: Optional path to the reference genome, stored verbatim under
            'ref_path'. Only used when is_ground_truth is False.
        is_ground_truth: If True, build the minimal reference config (see above),
            otherwise build the full training/prediction config.
        training_set: Either 'default', a preset name in training_set_presets,
            or an explicit list of caller names. A list is stored under a key
            formed by joining the names with '+'. Ignored when callsets is given.
        val_set: Validation set specification. May be None (no validation set),
            'use_name' to look up the sample's entry in sample_val_sets, a key in
            sample_val_sets, or a dict matching sample_val_set_template. The
            validation set's 'name' must be a key in callset_param_dict.
        model_name: Key in model_configs. Defaults to default_model_name when
            None.
        additional_callsets: Extra caller names to append to the
            training-set-derived callset list. Ignored when callsets is given.
        callsets: Explicit list of caller names to use as the callset list,
            bypassing training_set and additional_callsets.
        ground_truth_name: Optional name of the ground-truth set. When set and
            different from default_reference_set, a '.gt_<name>' suffix is
            appended to the workflow directory path.
        base_dir: Base directory for resolving all relative data paths. Defaults
            to BASE_DIR when None.

    Returns:
        Dictionary keyed by sample_name whose value is the sample config.

        For is_ground_truth=True the value has the shape:
            {
                'sample_dir': Path('<base_dir>/data/<sample>'),
                'callset_bps_dir': Path('<base_dir>/data/<sample>/callset_bps'),
                'workflow_dir': Path('<base_dir>/data/<sample>/svensemble2'),
                'is_reference': True,
                'callsets': {
                    '<sample>': {
                        'source': ...,
                        'svensemble_dir': Path(...),
                        'path': Path(...),
                        'params': {...},
                    }
                },
            }

        For is_ground_truth=False the value has the shape:
            {
                'sample_name': '<sample>',
                'sample_dir': Path(...),
                'callset_bps_dir': Path(...),
                'workflow_dir': Path(...),
                'is_reference': False,
                'callsets': {
                    '<caller>': {
                        'source': ...,
                        'svensemble_dir': Path(...),
                        'path': Path(...),
                        'params': {...},
                        'bps_path': Path(...),
                    },
                    ...
                },
                'training_sets': {'<training_set_name>': ['<caller>', ...]},
                'ref_path': <ref_path>,
                'val_set': None | {...},
                'model': {...},
            }

    Raises:
        ValueError: If callsets or additional_callsets contain unknown caller
            names, if training_set, val_set or model_name are invalid, or if no
            config could be created.
        KeyError: If a caller name cannot be resolved in callset_path_configs
            (raised by _get_callset_dict).

    Examples:
        Ground-truth (reference) config, where extra arguments are ignored:
            cfg = create_config_dict('COLO829', is_ground_truth=True)
            # cfg['COLO829']['is_reference'] -> True
            # list(cfg['COLO829']['callsets']) -> ['COLO829']
            # 'model' in cfg['COLO829'] -> False

        Full training/prediction config using the default training set:
            cfg = create_config_dict(
                'COLO829',
                is_ground_truth=False,
                model_name='default',
            )
            # cfg['COLO829']['is_reference'] -> False
            # 'model' in cfg['COLO829'] -> True
            # every callset in cfg['COLO829']['callsets'] has a 'bps_path'

        Full config with an explicit callset list (bypasses the training set):
            cfg = create_config_dict(
                'COLO829',
                is_ground_truth=False,
                callsets=['gridss', 'manta'],
            )
            # sorted(cfg['COLO829']['callsets']) -> ['gridss', 'manta']
    """
    if base_dir is None:
        base_dir = BASE_DIR

    if callsets is not None:
        invalid = [c for c in callsets if c not in callset_param_dict]
        if invalid:
            raise ValueError(f'Unknown callsets: {invalid}')
        callset_names = list(callsets)
        _training_sets = {default_training_set: training_set_presets[default_training_set]}
    else:
        if isinstance(training_set, str):
            if training_set == 'default':
                _training_sets = {default_training_set: training_set_presets[default_training_set]}
            elif training_set in training_set_presets.keys():
                _training_sets = {training_set: training_set_presets[training_set]}
            else:
                raise ValueError(f'Invalid training set: {training_set}')
        elif isinstance(training_set, list):
            _training_sets = {f'{'+'.join(training_set)}': training_set}
        else:
            raise ValueError(f'Invalid training set: {training_set}')

        callset_names = set()
        for training_set in _training_sets.values():
            for callset_name in training_set:
                callset_names.add(callset_name)
        callset_names = list(callset_names)
        if isinstance(additional_callsets, list):
            if any(callset_name not in callset_param_dict.keys() for callset_name in additional_callsets):
                raise ValueError(f'Invalid extra callsets: {additional_callsets}')
            callset_names.extend(additional_callsets)

    # If a validation set exists for the sample, add it to the config dict.
    if (val_set == 'use_name'):
        if sample_name not in sample_val_sets.keys():
            raise ValueError(f'Validation set not found for sample: {sample_name}')
        val_set = sample_val_sets[sample_name].copy()
        if not val_set['path'].is_absolute():
            val_set['path'] = base_dir / val_set['path']
    elif val_set in list(sample_val_sets.keys()):
        val_set = sample_val_sets[val_set].copy()
        if not val_set['path'].is_absolute():
            val_set['path'] = base_dir / val_set['path']
    elif isinstance(val_set, dict):
        if all(key in val_set.keys() for key in sample_val_set_template.keys()):
            pass
        else:
            raise ValueError(f'Invalid validation set: {val_set}')
        try:
            val_set['path'] = Path(val_set['path'])
            if not val_set['path'].is_absolute():
                val_set['path'] = base_dir / val_set['path']
        except Exception as e:
            raise ValueError(f'Invalid validation set path: {val_set['path']}, error: {e}')
    elif val_set is None:
        val_set = None
    else:
        raise ValueError(f'Invalid validation set: {val_set}')

    # Val_set must have a viable callset param setup
    if val_set is not None:
        val_name = val_set.get('name')
        if not val_name or not str(val_name).strip():
            raise ValueError(
                f"Validation set has an empty or missing 'name'. "
                f"Ensure 'name' is set to a key in _caller_params.callset_param_dict. Got: {repr(val_name)}"
            )
        try:
            val_set['params'] = callset_param_dict[val_name]
        except KeyError as e:
            raise ValueError(
                f'Validation set "{val_name}" does not have a callable param setup in _caller_params.py. '
                f'Available names: {list(callset_param_dict.keys())}. Error: {e}'
            )


    if model_name is None:
        model_name = default_model_name
    if model_name not in model_configs.keys():
        raise ValueError(f'Invalid model name: {model_name}')


    if is_ground_truth:
        config_dict = {
            sample_name: {
                'sample_dir': base_dir / Path(f'data/{sample_name}'),
                'callset_bps_dir': base_dir / Path(f'data/{sample_name}/callset_bps'),
                'workflow_dir': base_dir / Path(f'data/{sample_name}/svensemble2'),
                'is_reference': True,
                'callsets': {
                    sample_name: _get_callset_dict(sample_name, sample_name, base_dir=base_dir)
                }
            }
        }
        return config_dict

    training_set_name, training_set_callers = next(iter(_training_sets.items()))

    gt_suffix = ''
    if ground_truth_name and ground_truth_name != default_reference_set:
        gt_suffix = f'.gt_{ground_truth_name}'

    # Resolve the model config and its reconstructible identity. model_id
    # ({base}-{hash8}) replaces the free-text model name in the data-tree path.
    model_cfg = resolve_model_config(name=model_name)
    mid = model_id(model_cfg)
    model_cfg['model_id'] = mid

    config_dict = {
        sample_name: {
            'sample_name': sample_name,
            'sample_dir': base_dir / Path(f'data/{sample_name}'),
            'callset_bps_dir': base_dir / Path(f'data/{sample_name}/sv_calling/callset_bps'),
            'workflow_dir': base_dir / Path(
                f'data/{sample_name}/sv_calling/svensemble2/'
                f'{sample_name}.{mid}.'
                f'{training_set_name}{gt_suffix}'
            ),
            'is_reference': False,
            'callsets': {
                callset_name: _get_callset_dict(
                    sample_name, callset_name, base_dir=base_dir
                )
                for callset_name in callset_names
            },
            'training_sets': _training_sets,
            'ref_path': ref_path,
            'val_set': val_set,
            'model': model_cfg,
            'model_id': mid,
        }
    }

    # If there are custom columns specified in the model config, add them to the config dict of each caller.
    if ('custom_cols' in config_dict[sample_name]['model']['training_params'].keys()) and len(config_dict[sample_name]['model']['training_params']['custom_cols']) > 0:
        print(f"Adding custom columns to the config dict for sample {sample_name}")
        print(f"Custom columns: {config_dict[sample_name]['model']['training_params']['custom_cols']}")
        for callset_name in callset_names:
            config_dict[sample_name]['callsets'][callset_name]['params']['custom_cols_pred'] = config_dict[sample_name]['model']['training_params']['custom_cols']

    # matched_bps / feature_matrix live in the model's own workflow_dir. Model
    # identity is now the config hash (model_id), so there is no cross-model
    # suffix sharing -- every model_id owns its own intermediates.
    workflow_dir = config_dict[sample_name]['workflow_dir']
    config_dict[sample_name]['model']['matched_bps_path'] = workflow_dir / 'matched_bps.csv'
    config_dict[sample_name]['model']['feature_matrix_path'] = workflow_dir / 'feature_matrix.csv'

    _additional_set = set(additional_callsets) if isinstance(additional_callsets, list) else set()
    for callset_name in callset_names:
        if callset_name in _additional_set:
            config_dict[sample_name]['callsets'][callset_name]['bps_path'] = config_dict[sample_name]['callset_bps_dir'] / f'{callset_name}.raw_bps.csv'
        elif config_dict[sample_name]['callsets'][callset_name]['params']['override_bps_path']:
            config_dict[sample_name]['callsets'][callset_name]['bps_path'] = config_dict[sample_name]['sample_dir'] / config_dict[sample_name]['callsets'][callset_name]['params']['bps_path']
        else:
            config_dict[sample_name]['callsets'][callset_name]['bps_path'] = (
                workflow_dir / config_dict[sample_name]['callsets'][callset_name]['params']['bps_path']
            )
    if not config_dict:
        raise ValueError(f"No config dictionary created for sample {sample_name}")

    return config_dict


def build_cli_workflow_config(
    sample_name: str,
    callers: list[str] | None = None,
    model_name: str | None = None,
    training_set: str | None = None,
    ref_path: Path | None = None,
    model_config: dict | None = None,
) -> dict:
    """
    Build a workflow_config dict for CLI/Nextflow subcommands.

    This is a flat-path sibling of create_config_dict for the case where the
    files being consumed are already staged into the current working directory
    (Nextflow work dir, or the user's cwd for standalone CLI invocations) and
    the standard data/{sample}/... directory templating does not apply.

    Compared to create_config_dict:
      - Paths are all Path('.') (or bare filenames); no data/{sample}/ prefix
        derivation, no filesystem existence checks.
      - training_set is treated as a free-form name: if it matches a preset in
        training_set_presets, the preset's caller list is used; otherwise the
        provided `callers` list is stored under that name.
      - No ground-truth suffix or validation-set logic; ground truth is passed
        to downstream stages as its own bps CSV.

    Args:
        sample_name: Sample identifier (str). Stored under 'sample_name'.
        callers: List of callset names present in this sample's matched_bps.
        model_name: Key in model_configs; controls custom_cols_pred injection.
        training_set: Name to key the training_sets dict; matched against
            training_set_presets before falling back to `callers`.
        ref_path: Optional reference FASTA, stored verbatim under 'ref_path'.

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
        model_cfg = resolve_model_config(name=model_name or default_model_name)
    mid = model_id(model_cfg)
    training_params = model_cfg['training_params']
    custom_cols = training_params.get('custom_cols', [])

    if training_set and training_set in training_set_presets:
        resolved_callers = training_set_presets[training_set]
    else:
        resolved_callers = callers

    callsets = {}
    for callset_name in resolved_callers:
        params = copy.deepcopy(callset_param_dict.get(callset_name, {}))
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

    config = {
        'sample_name': sample_name,
        'callsets': callsets,
        'callset_bps_dir': workflow_dir,
        'workflow_dir': workflow_dir,
        'is_reference': False,
        'training_sets': training_sets,
        'model': {
            **model_cfg,
            'model_id': mid,
            'matched_bps_path': workflow_dir / 'matched_bps.csv',
            'feature_matrix_path': workflow_dir / 'feature_matrix.csv',
        },
        'model_id': mid,
    }

    if ref_path is not None:
        config['ref_path'] = ref_path

    return config


def generate_chrom_key_dict(chrom_series: pd.Series) -> dict:
    """
    Generate a dictionary mapping chromosome names to numeric indices for sorting.

    Creates a mapping of standard chromosome names (chr1-22, X, Y, M) plus any
    non-standard chromosomes found in the input series to sequential indices.

    Args:
        chrom_series: Series containing chromosome names

    Returns:
        Dictionary mapping chromosome names to numeric indices
    """
    # Standard chromosome list in desired sort order
    chrom_list = ['chr' + str(i) for i in range(1, 23)] + ['chrX', 'chrY', 'chrM']

    # Add any non-standard chromosomes found in input
    nonstd_chrom_list = chrom_series[~chrom_series.isin(chrom_list)].unique().tolist()
    chrom_list += nonstd_chrom_list

    # Create mapping of chromosome names to indices
    return {chrom: idx for idx, chrom in enumerate(chrom_list)}


def get_chrom_keys(chrom_series: pd.Series, chrom_keys: dict | None = None) -> pd.Series:
    """
    Convert chromosome names to numeric keys for sorting.

    Maps chromosome names to numeric indices using either provided mapping
    or default standard chromosome order.

    Args:
        chrom_series: Series containing chromosome names
        chrom_keys: Optional dictionary mapping chromosomes to indices.
                   If None, uses standard chromosome order.

    Returns:
        Series with chromosome names converted to numeric indices.
        Returns original series if chromosome not found in mapping.
    """
    if chrom_keys is None:
        # Use standard chromosome order if no mapping provided
        chrom_list = ['chr' + str(i) for i in range(1, 23)] + ['chrX', 'chrY', 'chrM']
        chrom_keys = {chrom: idx for idx, chrom in enumerate(chrom_list)}

    return chrom_series.map(chrom_keys).fillna(chrom_series)


def split_cipos(cipos_col: pd.Series, default_cipos: tuple[int, int] = (0, 0)) -> pd.DataFrame:
    cipos_df = pd.DataFrame(cipos_col)
    cipos_df.loc[:, cipos_col.name.replace('INFO_CIPOS', 'INFO_CIPOS_l')] = cipos_col.str[0].fillna(default_cipos[0]).astype(int)
    cipos_df.loc[:, cipos_col.name.replace('INFO_CIPOS', 'INFO_CIPOS_r')] = cipos_col.str[1].fillna(default_cipos[1]).astype(int)
    return cipos_df.drop(columns=[cipos_col.name])


def split_ciend(ciend_col: pd.Series, default_ciend: tuple[int, int] = (0, 0)) -> pd.DataFrame:
    ciend_df = pd.DataFrame(ciend_col)
    ciend_df.loc[:, ciend_col.name.replace('INFO_CIEND', 'INFO_CIEND_l')] = ciend_col.str[0].fillna(default_ciend[0]).astype(int)
    ciend_df.loc[:, ciend_col.name.replace('INFO_CIEND', 'INFO_CIEND_r')] = ciend_col.str[1].fillna(default_ciend[1]).astype(int)
    return ciend_df.drop(columns=[ciend_col.name])


def create_samples_config_table(samples: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    """
    Create a configuration table (CSV) listing all sample, training set, and model name combinations.

    Args:
        samples (Dict[str, Dict[str, Any]]):
            Dictionary mapping sample names to their configuration dictionaries.
            Each configuration dictionary is expected to contain a 'training_sets' key,
            which is iterable of training set names.

    Returns:
        pandas.DataFrame:
            DataFrame with columns ['sample_name', 'training_set_name', 'model_name'].
            Each row is a combination of sample, training set, and model name ("unbiased").
    """
    sample_names: list = []
    training_set_names: list = []
    model_names: list = []

    for sample_name, sample_config in samples.items():
        for training_set_name in sample_config['training_sets']:
            sample_names.append(sample_name)
            training_set_names.append(training_set_name)
            model_names.append('unbiased_rf')

    return pd.DataFrame({
        'sample_name': sample_names,
        'training_set_name': training_set_names,
        'model_name': model_names
    })


def generate_callset_combinations(callset_names: list[str]) -> list[str]:
    """
    Generate all possible combinations - from single to pairs to triplets to ... to all of the callsets.
    Callsets are sorted alphabetically to ensure consistent ordering.
    Duplicate combinations are removed.

    Args:
        callset_names: List of callset names.

    Returns:
        List of all possible combinations of callsets.
    """
    return sorted(list(set(['+'.join(sorted(comb)) for r in range(1, len(callset_names) + 1) for comb in itertools.combinations(callset_names, r)])))
