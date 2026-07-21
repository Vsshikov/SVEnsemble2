"""
Config I/O for SVEnsemble2.

The CLI passes a config dict between pipeline stages as JSON (--config_in /
--config_out). JSON can't represent Path, np.inf/np.nan, tuples or sets, so
dump_config sanitizes those away on write. Nothing in cli.py currently reads
a stage's own fields back as Path/inf, but rehydrate_config is here for
whenever a fuller config (e.g. a workflow_config with real Path/inf fields)
needs to survive a JSON round-trip. It is opt-in and schema-driven: callers
pass the keys that need converting back, nothing is guessed from key names.
"""

import json
from pathlib import Path

import numpy as np


REQUIRED_KEYS = {"sample_name"}


def _sanitize_for_json(obj):
    """Recursively convert Path/np.inf/np.nan/tuple/set into JSON-safe equivalents."""
    if isinstance(obj, float):
        if obj != obj or obj == float("inf") or obj == float("-inf"):
            return None
        return obj
    if isinstance(obj, np.floating):
        if np.isinf(obj) or np.isnan(obj):
            return None
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return [_sanitize_for_json(v) for v in obj.tolist()]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_for_json(v) for v in obj]
    if isinstance(obj, (set, frozenset)):
        return sorted(_sanitize_for_json(v) for v in obj)
    return obj


def _rehydrate_from_json(obj, path_keys=frozenset(), inf_keys=frozenset()):
    """Recursively reverse _sanitize_for_json for the given key sets.

    Walks obj the same way _sanitize_for_json does, but only touches values
    reached through a dict key in path_keys or inf_keys - everything else is
    left as plain JSON types (this is the "explicit schema" side of the
    sanitize/rehydrate pair; see module docstring).

    Args:
        obj: JSON-decoded object (dict/list/scalar) to rehydrate in place.
        path_keys: Dict keys whose string value should become a Path.
        inf_keys: Dict keys whose value should have None turned back into
            np.inf, either directly or inside a list/tuple (e.g. a
            [min, max] bound where None marks an unbounded max).

    Returns:
        A new object with the same shape as obj, with conversions applied.
    """
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if k in path_keys and isinstance(v, str):
                out[k] = Path(v)
            elif k in inf_keys:
                if v is None:
                    out[k] = np.inf
                elif isinstance(v, (list, tuple)):
                    out[k] = [np.inf if x is None else x for x in v]
                else:
                    out[k] = v
            else:
                out[k] = _rehydrate_from_json(v, path_keys=path_keys, inf_keys=inf_keys)
        return out
    if isinstance(obj, list):
        return [_rehydrate_from_json(v, path_keys=path_keys, inf_keys=inf_keys) for v in obj]
    return obj


def dump_config(config: dict, path: Path) -> None:
    """Sanitize and write a config dict to path as JSON or YAML (by suffix).

    Args:
        config: Config dict to write. May contain Path, np.inf/np.nan,
            numpy scalars, tuples and sets - all converted to JSON-safe
            equivalents by _sanitize_for_json before writing.
        path: Output path. A ``.yaml``/``.yml`` suffix writes YAML; anything
            else writes JSON. Parent directories are created if missing.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sanitized = _sanitize_for_json(config)
    with open(path, "w") as f:
        if path.suffix.lower() in (".yaml", ".yml"):
            import yaml  # lazy: only required when YAML is actually used
            yaml.safe_dump(sanitized, f, sort_keys=False, default_flow_style=False)
        else:
            json.dump(sanitized, f, indent=2)


def load_config(path: Path) -> dict:
    """Read a config dict from a JSON or YAML file (by suffix).

    A ``.yaml``/``.yml`` suffix parses YAML; anything else parses JSON. Plain
    load - Paths stay strings and sanitized inf/nan values stay None. Use
    rehydrate_config on the result if the consumer needs those back.
    """
    path = Path(path)
    with open(path) as f:
        if path.suffix.lower() in (".yaml", ".yml"):
            import yaml  # lazy: only required when YAML is actually used
            return yaml.safe_load(f)
        return json.load(f)


def rehydrate_config(config: dict, path_keys=None, inf_keys=None) -> dict:
    """Reconstruct Path/np.inf values in a config loaded via load_config.

    Args:
        config: Dict as returned by load_config.
        path_keys: Keys whose string value should be converted to Path.
            Defaults to none (no path rehydration).
        inf_keys: Keys whose None value (bare, or inside a list/tuple)
            should be converted back to np.inf. Defaults to none.

    Returns:
        A new dict with the requested keys rehydrated; everything else is
        passed through unchanged.
    """
    return _rehydrate_from_json(
        config,
        path_keys=frozenset(path_keys or ()),
        inf_keys=frozenset(inf_keys or ()),
    )


def validate_config(config: dict) -> None:
    """Raise ValueError if config is missing any of REQUIRED_KEYS."""
    missing = REQUIRED_KEYS - set(config.keys())
    if missing:
        raise ValueError(f"Config missing required keys: {sorted(missing)}")


def merge_configs(base: dict, override: dict) -> dict:
    """Shallow-merge two config dicts, with override winning on conflict.

    Dict-valued keys present in both (e.g. match_params) are merged one
    level deep instead of being fully replaced; everything else in
    override just overwrites the value in base.
    """
    merged = base.copy()
    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = {**merged[key], **value}
        else:
            merged[key] = value
    return merged
