"""
Lossless config encoding and file I/O for SVEnsemble2.

Provides two decoupled layers:

  Encoding -- ``dict_json_encode`` / ``dict_json_decode`` convert between
  native Python objects (Path, inf, tuple, set, numpy scalars) and a
  JSON-safe tagged representation using Kombu/Celery-style envelopes:
  ``{"__type__": "<marker>", "__value__": <data>}``.

  File I/O -- ``dump_config`` / ``load_config`` read and write dicts as
  JSON or YAML (by suffix).  They perform **no** encoding or decoding;
  callers compose the layers explicitly.

  Hashing -- ``canonical_json`` encodes and serialises deterministically
  for content-addressable hashing (used by ``_config.model_config_hash``).
"""

import json
from pathlib import Path

import numpy as np


REQUIRED_KEYS = {"sample_name"}

# -- Tagged envelope (Kombu/Celery convention) ------------------------------

_DECODERS = {
    "path":      lambda v: Path(v),
    "tuple":     lambda v: tuple(v),
    "set":       lambda v: set(v),
    "frozenset": lambda v: frozenset(v),
    "inf":       lambda v: float("inf"),
    "-inf":      lambda v: float("-inf"),
    "nan":       lambda v: float("nan"),
    "ndarray":   lambda v: np.array(v),
}


def _as(t: str, v):
    """Wrap a value in a ``{"__type__": t, "__value__": v}`` envelope."""
    return {"__type__": t, "__value__": v}


def dict_json_encode(obj):
    """Recursively encode non-JSON-native types as tagged envelopes.

    Returns a structure containing only JSON-native types (dict, list,
    str, int, float, bool, None) with ``{"__type__", "__value__"}``
    wrappers for values that would otherwise be lost or mangled by
    ``json.dumps``.

    Handled types and their markers:

    =============  =========  ====================================
    Python type    Marker     Encoded ``__value__``
    =============  =========  ====================================
    ``Path``       path       ``str(obj)``
    ``tuple``      tuple      ``[encode(elem), ...]``
    ``set``        set        ``sorted([encode(elem), ...])``
    ``frozenset``  frozenset  ``sorted([encode(elem), ...])``
    ``float`` inf  inf        ``None``
    ``float`` -inf -inf       ``None``
    ``float`` NaN  nan        ``None``
    =============  =========  ====================================

    ``np.ndarray`` ndarray    ``[encode(elem), ...]`` (via tolist)
    =============  =========  ====================================

    ``np.integer`` and finite ``np.floating`` are silently promoted to
    ``int`` / ``float`` (lossless for JSON).
    """
    if isinstance(obj, float):
        if obj != obj:
            return _as("nan", None)
        if obj == float("inf"):
            return _as("inf", None)
        if obj == float("-inf"):
            return _as("-inf", None)
        return obj
    if isinstance(obj, np.floating):
        return dict_json_encode(float(obj))
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return _as("ndarray", [dict_json_encode(v) for v in obj.tolist()])
    if isinstance(obj, Path):
        return _as("path", str(obj))
    if isinstance(obj, dict):
        return {k: dict_json_encode(v) for k, v in obj.items()}
    if isinstance(obj, tuple):
        return _as("tuple", [dict_json_encode(v) for v in obj])
    if isinstance(obj, list):
        return [dict_json_encode(v) for v in obj]
    if isinstance(obj, frozenset):
        return _as("frozenset", sorted(dict_json_encode(v) for v in obj))
    if isinstance(obj, set):
        return _as("set", sorted(dict_json_encode(v) for v in obj))
    return obj


def dict_json_decode(obj):
    """Recursively decode tagged envelopes back to Python types.

    Recognises dicts whose keys are exactly ``{"__type__", "__value__"}``
    and dispatches on the ``__type__`` marker.  All other dicts and lists
    are recursed into; scalars pass through unchanged.

    Works both as a standalone recursive decoder (e.g. after
    ``yaml.safe_load``) and as ``json.loads(s, object_hook=...)``.
    """
    if isinstance(obj, dict):
        if obj.keys() == {"__type__", "__value__"}:
            marker = obj["__type__"]
            value = obj["__value__"]
            decoder = _DECODERS.get(marker)
            if decoder is None:
                return obj
            if isinstance(value, list):
                return decoder([dict_json_decode(item) for item in value])
            return decoder(value)
        return {k: dict_json_decode(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [dict_json_decode(v) for v in obj]
    return obj


# -- Canonical JSON for hashing ---------------------------------------------

def canonical_json(obj) -> str:
    """Lossless, deterministic JSON string suitable for hashing.

    Encodes *obj* with :func:`dict_json_encode`, then serialises with
    ``sort_keys=True``, compact separators, and ``allow_nan=False``
    (special floats are already tagged, so any remaining NaN/Infinity
    would indicate an encoding bug and should raise).
    """
    return json.dumps(
        dict_json_encode(obj),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


# -- File I/O (no encoding / decoding) -------------------------------------

def dump_config(config: dict, path: Path) -> None:
    """Write *config* to *path* as JSON or YAML (by suffix).

    Pure I/O -- **no** encoding is applied.  To write a lossless tagged
    representation, encode first::

        dump_config(dict_json_encode(cfg), path)

    Parent directories are created if missing.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        if path.suffix.lower() in (".yaml", ".yml"):
            import yaml
            yaml.safe_dump(config, f, sort_keys=False, default_flow_style=False)
        else:
            json.dump(config, f, indent=2)


def load_config(path: Path) -> dict:
    """Read a dict from a JSON or YAML file (by suffix).

    Pure I/O -- **no** decoding is applied.  To restore tagged values::

        cfg = dict_json_decode(load_config(path))
    """
    path = Path(path)
    with open(path) as f:
        if path.suffix.lower() in (".yaml", ".yml"):
            import yaml
            return yaml.safe_load(f)
        return json.load(f)


# -- Validation -------------------------------------------------------------

def validate_config(config: dict) -> None:
    """Raise ValueError if *config* is missing any of REQUIRED_KEYS."""
    missing = REQUIRED_KEYS - set(config.keys())
    if missing:
        raise ValueError(f"Config missing required keys: {sorted(missing)}")
