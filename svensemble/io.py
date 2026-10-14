"""
File I/O for SVEnsemble2: every pipeline filename and every read/write.

  Filenames -- the constants below name each artifact a pipeline stage
  writes. ``{caller}`` placeholders are filled with ``str.format``.

  Artifacts -- ``read_*`` / ``write_*`` for breakpoint CSVs, the feature
  matrix, the model pickle and the final calls. Writers take the object
  first and the location second, create parent directories and return the
  path(s) they wrote. Readers take only the location, and exist only for
  artifacts the pipeline reads back. A one-file artifact takes a file path;
  a CSV-plus-VCF artifact takes a directory and uses the constants.

  Encoding -- ``dict_json_encode`` / ``dict_json_decode`` convert between
  native Python objects (Path, inf, tuple, set, numpy scalars) and a
  JSON-safe tagged representation using Kombu/Celery-style envelopes:
  ``{"__type__": "<marker>", "__value__": <data>}``.

  Config files -- ``write_config`` / ``read_config`` read and write dicts
  as JSON or YAML (by suffix) with **no** encoding or decoding; callers
  compose the layers explicitly. ``read_model_config`` is the one reader
  that also decodes.

  Hashing -- ``canonical_json`` encodes and serialises deterministically
  for content-addressable hashing (used by ``_config.model_config_hash``).
"""

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from . import vcfwrite


RAW_BPS = "{caller}.raw_bps.csv"
CALLSET_BPS = "{caller}.bps.csv"
CALLSET_VCF = "{caller}.vcf"
MATCHED_BPS = "matched_bps.csv"
MODEL = "model.pkl"
FEATURE_MATRIX = "feature_matrix.csv"
CALLS_CSV = "svensemble2_calls.csv"
CALLS_VCF = "svensemble2_calls.vcf"

REQUIRED_KEYS = {"sample_name"}


# -- Pipeline artifacts -----------------------------------------------------

def read_bps(path: Path) -> pd.DataFrame:
    """Read any breakpoint CSV: raw, processed, matched or ground truth."""
    return pd.read_csv(path, low_memory=False)


def write_bps(bp_df: pd.DataFrame, path: Path) -> Path:
    """Write a breakpoint CSV without its index (e.g. RAW_BPS, MATCHED_BPS)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    bp_df.to_csv(path, index=False)
    return path


def write_callset(bp_df: pd.DataFrame, out_dir: Path, caller: str, sample: str) -> tuple[Path, Path]:
    """Write a processed callset as CALLSET_BPS plus CALLSET_VCF in *out_dir*."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    bps_path = out_dir / CALLSET_BPS.format(caller=caller)
    vcf_path = out_dir / CALLSET_VCF.format(caller=caller)
    bp_df.to_csv(bps_path, index=False)
    vcfwrite.write_vcf(bp_df, vcf_path, source=caller, sample_name=sample)
    return bps_path, vcf_path


def read_feature_matrix(path: Path) -> pd.DataFrame:
    """Read a feature matrix and restore its ``<callset>_INFO_EVENT`` index.

    The index columns map each row back to its matched breakpoints, so a
    matrix without them is rejected rather than silently re-indexed.
    """
    X = pd.read_csv(path, low_memory=False)
    index_cols = [c for c in X.columns if c.endswith("_INFO_EVENT")]
    if not index_cols:
        raise ValueError(
            f"Feature matrix at {path} has no '<callset>_INFO_EVENT' index "
            f"columns. These are required for reconstructing calls and should have been "
            f"written by the training step; their absence indicates a broken or truncated "
            f"feature matrix. Found columns: {list(X.columns)[:20]}{'...' if len(X.columns) > 20 else ''}"
        )
    return X.set_index(index_cols)


def write_feature_matrix(X: pd.DataFrame, path: Path) -> Path:
    """Write a feature matrix with its ``<callset>_INFO_EVENT`` index."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    X.to_csv(path, index=True)
    return path


def read_model(path: Path):
    """Unpickle a fitted model."""
    with open(path, "rb") as f:
        return pickle.load(f)


def write_model(model_fit, path: Path) -> Path:
    """Pickle a fitted model."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model_fit, f)
    return path


def write_calls(calls_df: pd.DataFrame, out_dir: Path, sample: str) -> tuple[Path, Path]:
    """Write ensemble calls as CALLS_CSV (with index) plus CALLS_VCF in *out_dir*."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / CALLS_CSV
    vcf_path = out_dir / CALLS_VCF
    calls_df.to_csv(csv_path, index=True)
    vcfwrite.write_vcf(calls_df, vcf_path, source="svensemble2", sample_name=sample)
    return csv_path, vcf_path

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


# -- Config files (no encoding / decoding, except read_model_config) -------

def write_config(config: dict, path: Path) -> Path:
    """Write *config* to *path* as JSON or YAML (by suffix).

    Pure I/O -- **no** encoding is applied.  To write a lossless tagged
    representation, encode first::

        write_config(dict_json_encode(cfg), path)

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
    return path


def read_config(path: Path) -> dict:
    """Read a dict from a JSON or YAML file (by suffix).

    Pure I/O -- **no** decoding is applied.  To restore tagged values::

        cfg = dict_json_decode(read_config(path))
    """
    path = Path(path)
    with open(path) as f:
        if path.suffix.lower() in (".yaml", ".yml"):
            import yaml
            return yaml.safe_load(f)
        return json.load(f)


def read_model_config(path: Path) -> dict:
    """Read a model config file and decode its tagged values.

    Plain JSON/YAML without envelopes decodes to itself, so hand-written
    configs and their model_ids are unaffected.
    """
    return dict_json_decode(read_config(path))


# -- Validation -------------------------------------------------------------

def validate_config(config: dict) -> None:
    """Raise ValueError if *config* is missing any of REQUIRED_KEYS."""
    missing = REQUIRED_KEYS - set(config.keys())
    if missing:
        raise ValueError(f"Config missing required keys: {sorted(missing)}")
