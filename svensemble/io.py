"""
File I/O for SVEnsemble2: every pipeline filename and every read/write.

  Filenames -- the constants below name each artifact a pipeline stage
  writes. ``{caller}`` placeholders are filled with ``str.format``.

  Caller VCFs -- ``read_vcf`` parses a caller's VCF into the DataFrame
  that ``api.preprocess_raw`` takes.

  Artifacts -- ``read_*`` / ``write_*`` for breakpoint CSVs, the feature
  matrix, the model pickle and the final calls. Writers take the object
  first and the location second, create parent directories and return the
  path(s) they wrote. Every write is atomic (``utils.atomic_path``): a
  failed or killed write never leaves a partial file at the final path. Readers take only the location, and exist only for
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

import copy
import gzip
import json
import pickle
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from io import TextIOWrapper
from pathlib import Path
from typing import BinaryIO, TextIO

import numpy as np
import pandas as pd
from pysam import BGZFile, VariantFile

from . import annotate as sv_annotate
from . import vcfwrite
from ._caller_params import CALLSET_PARAM_DICT
from ._constants import SAMPLE_ID_COL
from .utils import atomic_path
from .vcfparse import parse_vcf_to_df


RAW_BPS = "{caller}.raw_bps.csv"
CALLSET_BPS = "{caller}.bps.csv"
CALLSET_VCF = "{caller}.vcf"
MATCHED_BPS = "matched_bps.csv"
MODEL = "model.pkl"
FEATURE_MATRIX = "feature_matrix.csv"
CALLS_CSV = "svensemble2_calls.csv"
CALLS_VCF = "svensemble2_calls.vcf"
GROUND_TRUTH_MANIFEST = "{ground_truth_name}.manifest.txt"
VAL_BPS = "{val_set}.val_bps.csv"
BENCHMARK_CSV = "{caller}.benchmark.csv"
SHAP_SUMMARY = "shap_summary.csv"
SHAP_VALUES = "shap_values.pkl"
RUN_PROVENANCE = "svensemble_run.json"
ANNOTATED_VCF = "{caller}.{stem}.svensemble2{ext}"

REQUIRED_KEYS = {"sample_name"}


# -- Pipeline artifacts -----------------------------------------------------

def read_vcf(path: Path, caller: str, sample_names: str | list[str] | None = None) -> pd.DataFrame:
    """Parse a caller's VCF/BCF into a DataFrame (the input to api.preprocess_raw).

    Columns are chosen by the caller's CALLSET_PARAM_DICT entry.

    Args:
        path: VCF/BCF path.
        caller: Callset name (a key of _caller_params.CALLSET_PARAM_DICT).
        sample_names: Sample(s) to read, one row per (record, sample).
            Defaults to every sample in the header.
    """
    try:
        params = copy.deepcopy(CALLSET_PARAM_DICT[caller])
    except KeyError:
        raise KeyError(f"Unknown caller {caller!r}: no CALLSET_PARAM_DICT entry") from None
    with VariantFile(str(path)) as vcf:
        vcf_df, _ = parse_vcf_to_df(vcf, params, sample_names=sample_names)
    return vcf_df


def read_bps(path: Path) -> pd.DataFrame:
    """Read any breakpoint CSV: raw, processed, matched or ground truth.

    Sample columns are read as text, so digit-only sample names stay names.
    """
    cols = pd.read_csv(path, nrows=0).columns
    sample_cols = {c: str for c in cols if c.endswith(SAMPLE_ID_COL)}
    return pd.read_csv(path, low_memory=False, dtype=sample_cols)


def write_bps(bp_df: pd.DataFrame, path: Path) -> Path:
    """Write a breakpoint CSV without its index (e.g. RAW_BPS, MATCHED_BPS)."""
    path = Path(path)
    with atomic_path(path) as tmp:
        bp_df.to_csv(tmp, index=False)
    return path


def write_callset(bp_df: pd.DataFrame, out_dir: Path, caller: str, sample: str) -> tuple[Path, Path]:
    """Write a processed callset as CALLSET_BPS plus CALLSET_VCF in *out_dir*."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    bps_path = out_dir / CALLSET_BPS.format(caller=caller)
    vcf_path = out_dir / CALLSET_VCF.format(caller=caller)
    with atomic_path(bps_path) as tmp:
        bp_df.to_csv(tmp, index=False)
    vcfwrite.write_vcf(bp_df, vcf_path, source=caller, sample_name=sample)
    return bps_path, vcf_path


def read_feature_matrix(path: Path) -> pd.DataFrame:
    """Read a feature matrix and restore its ``<callset>_INFO_EVENT`` (+ sample) index.

    The index columns map each row back to its matched breakpoints, so a
    matrix without them is rejected rather than silently re-indexed.
    """
    X = pd.read_csv(path, low_memory=False, dtype={SAMPLE_ID_COL: str})
    index_cols = [c for c in X.columns if c.endswith("_INFO_EVENT")]
    if not index_cols:
        raise ValueError(
            f"Feature matrix at {path} has no '<callset>_INFO_EVENT' index "
            f"columns. These are required for reconstructing calls and should have been "
            f"written by the training step; their absence indicates a broken or truncated "
            f"feature matrix. Found columns: {list(X.columns)[:20]}{'...' if len(X.columns) > 20 else ''}"
        )
    if SAMPLE_ID_COL in X.columns:
        index_cols.append(SAMPLE_ID_COL)
    return X.set_index(index_cols)


def write_feature_matrix(X: pd.DataFrame, path: Path) -> Path:
    """Write a feature matrix with its ``<callset>_INFO_EVENT`` index."""
    path = Path(path)
    with atomic_path(path) as tmp:
        X.to_csv(tmp, index=True)
    return path


def read_model(path: Path):
    """Unpickle a fitted model."""
    with open(path, "rb") as f:
        return pickle.load(f)


def write_model(model_fit, path: Path) -> Path:
    """Pickle a fitted model."""
    path = Path(path)
    with atomic_path(path) as tmp, open(tmp, "wb") as f:
        pickle.dump(model_fit, f)
    return path


def write_calls(calls_df: pd.DataFrame, out_dir: Path, sample: str) -> tuple[Path, Path]:
    """Write ensemble calls as CALLS_CSV (with index) plus CALLS_VCF in *out_dir*.

    The VCF has one record per SV and the score as FORMAT SVENSSC for each
    sample in FORMAT_SAMPLE_ID (*sample* names the column of calls without one).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / CALLS_CSV
    vcf_path = out_dir / CALLS_VCF
    with atomic_path(csv_path) as tmp:
        calls_df.to_csv(tmp, index=True)
    vcfwrite.write_vcf(
        calls_df, vcf_path, source="svensemble2", sample_name=sample,
        score_col=sv_annotate.pred_column(calls_df),
    )
    return csv_path, vcf_path


def read_calls(path: Path) -> pd.DataFrame:
    """Read CALLS_CSV with event and sample columns as text.

    Text dtype keeps digit-only sample names and event IDs from turning into
    numbers; event IDs already float-ified upstream ('21.0') are normalised by
    annotate. NA-like sample names are rejected before any run, so default NA
    handling is safe here.
    """
    cols = pd.read_csv(path, nrows=0).columns
    text_cols = [c for c in cols if c.endswith("_INFO_EVENT") or c == SAMPLE_ID_COL]
    return pd.read_csv(path, index_col=0, low_memory=False, dtype={c: str for c in text_cols})


# -- Caller VCFs as raw text or BCF records (annotation) --------------------

_VCF_EXTS = (".vcf.gz", ".vcf.bgz", ".vcf", ".bcf")


def split_vcf_name(path: Path) -> tuple[str, str]:
    """(stem, extension) of a VCF/BCF file name, e.g. ('s1.delly', '.vcf.gz')."""
    name = Path(path).name
    for ext in _VCF_EXTS:
        if name.endswith(ext):
            return name[: -len(ext)], ext
    raise ValueError(f"{path}: expected one of {_VCF_EXTS}")


def annotated_vcf_name(caller: str, path: Path) -> str:
    """ANNOTATED_VCF for one input, keeping the input's extension."""
    stem, ext = split_vcf_name(path)
    return ANNOTATED_VCF.format(caller=caller, stem=stem, ext=ext)


def _is_gzip(path: Path) -> bool:
    with open(path, "rb") as fh:
        return fh.read(2) == b"\x1f\x8b"


def _read_text_header(fh: TextIO) -> list[str]:
    """Header lines up to and including #CHROM; fh is left at the first record."""
    lines = []
    for line in fh:
        lines.append(line)
        if line.startswith("#CHROM"):
            return lines
        if not line.startswith("##"):
            break
    raise ValueError("No #CHROM line before the first record")


@contextmanager
def open_variant_file(path: Path) -> Iterator[tuple[str, bool, list[str], Iterable]]:
    """Open a caller VCF/BCF for annotation without parsing records.

    Yields (kind, compressed, header_lines, body): kind is 'vcf' or 'bcf';
    body iterates raw text lines (endings kept, undecodable bytes preserved
    via surrogateescape) or BCF (shared, per-sample) record blocks.
    """
    path = Path(path)
    compressed = _is_gzip(path)
    raw: BinaryIO = gzip.open(path, "rb") if compressed else open(path, "rb")
    try:
        kind = "bcf" if raw.peek(3)[:3] == b"BCF" else "vcf"
        if kind == "bcf":
            yield kind, compressed, sv_annotate.read_bcf_header(raw), sv_annotate.iter_bcf_records(raw)
            return
        text = TextIOWrapper(raw, encoding="utf-8", errors="surrogateescape", newline="")
        yield kind, compressed, _read_text_header(text), text
    finally:
        raw.close()


def parse_samples_from_vcf(path: Path) -> list[str]:
    """Sample names from a VCF/BCF header (empty for a sites-only file)."""
    with open_variant_file(path) as (_kind, _compressed, header, _body):
        chrom = next(line for line in header if line.startswith("#CHROM"))
        return sv_annotate.split_line_ending(chrom)[0].split("\t")[9:]


def write_annotated(chunks: Iterable[str | bytes], path: Path, compress: bool) -> Path:
    """Write annotated text lines or BCF bytes to *path*, BGZF-compressed if *compress*.

    *path* is written directly: callers wrap it in utils.atomic_path.
    """
    opener = (lambda: BGZFile(str(path), "wb")) if compress else (lambda: open(path, "wb"))
    with opener() as out:
        for chunk in chunks:
            out.write(chunk if isinstance(chunk, bytes) else chunk.encode("utf-8", "surrogateescape"))
    return Path(path)

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
    with atomic_path(path) as tmp, open(tmp, "w") as f:
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
