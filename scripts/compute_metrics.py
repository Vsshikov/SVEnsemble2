#! /usr/bin/env python3
"""
Compute classification metrics from SVEnsemble2 benchmark CSV files.

For each benchmark CSV (produced by svensemble/benchmark.py):

  Callset-level counts
  --------------------
  For raw callers, every call is a positive prediction — the caller already
  decided what to report.  TP, FP, FN are determined directly from truth
  matching without any threshold.

  For SVEnsemble2 models, predictions are thresholded on the ``_pred``
  probability at one or more operating points (default: 0.1, 0.2, 0.3).

  Metrics at each operating point
  --------------------------------
  TP         — predicted-positive calls matching a truth variant
  FP         — predicted-positive calls NOT matching a truth variant
  FN         — n_truth - TP  (requires --n_truth or --sample_name)
  Precision  — TP / (TP + FP)
  Recall     — TP / n_truth
  F1         — harmonic mean of Precision and Recall

  Curve metrics (threshold-free, always on quality score)
  -------------------------------------------------------
  AUROC  — area under the ROC curve (callset-level, no padding)
  AUPRC  — area under the precision-recall curve (global, padded for
            missed truth variants when n_truth is known)

Usage:
    python scripts/compute_metrics.py \\
        --benchmark_csvs a.benchmark.csv,b.benchmark.csv \\
        --sample_name NA12878 \\
        --thresholds 0.1 0.2 0.3 \\
        [--output metrics.csv] [--verbose]

    # Pipe paths from find via stdin:
    find data/ -name '*.benchmark.csv' | python scripts/compute_metrics.py \\
        --benchmark_csvs_file - \\
        --sample_name NA12878
"""

import os
import sys
import warnings
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
)

WORKDIR = "/vast/ai_projects/Papenfuss_SVEnsemble2"
if WORKDIR not in sys.path:
    sys.path.insert(0, WORKDIR)
if __name__ == '__main__':
    os.chdir(WORKDIR)

import svensemble._caller_params as caller_params
import svensemble.benchmark as bench


# ---------------------------------------------------------------------------
# Column helpers
# ---------------------------------------------------------------------------

_META_COLS = frozenset({'INFO_EVENT', 'is_in_val', 'FILTER_PASS', 'INFO_INFERRED_SVTYPE'})


def resolve_score_column(df: pd.DataFrame, callset: str | None = None) -> str:
    """Return the quality/prediction score column from a benchmark CSV.

    Resolution order:

    1. If a ``_pred`` column is present (SVEnsemble2 output), return it.
    2. If *callset* is given and its ``qual_metric`` (from
       ``caller_param_dict``) exists in the DataFrame, return that.
    3. Fall back to the first column that is not a metadata column.
    """
    pred_cols = [c for c in df.columns if c.endswith('_pred')]
    if pred_cols:
        return pred_cols[0]

    if callset is not None:
        params = caller_params.caller_param_dict.get(callset)
        if params is not None:
            qual_metric = params.get('qual_metric')
            if qual_metric is not None and qual_metric in df.columns:
                return qual_metric

    score_cols = [c for c in df.columns if c not in _META_COLS]
    if not score_cols:
        raise ValueError(
            f"No score column found.  Columns present: {list(df.columns)}"
        )
    return score_cols[0]


def is_svensemble2_csv(df: pd.DataFrame) -> bool:
    """Return True when the benchmark CSV comes from an SVEnsemble2 model."""
    return any(c.endswith('_pred') for c in df.columns)


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def compute_at_threshold(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float | None,
    *,
    n_truth: int | None = None,
) -> dict:
    """Compute confusion-matrix counts and derived rates at a single threshold.

    Args:
        y_true: Binary ground-truth labels (1 = matched a truth variant).
        y_score: Quality metric or model prediction scores.
        threshold: Calls with score >= threshold are predicted positive.
            If None, **all** calls are treated as positive predictions
            (appropriate for raw callers whose callset IS their output).
        n_truth: Total truth variants for global recall / FN.

    Returns:
        Dict with keys: tp, fp, fn, precision, recall, f1.
        ``fn`` and ``recall`` are global (using n_truth) when available,
        otherwise callset-level.
    """
    y_true = np.asarray(y_true, dtype=int)

    if threshold is None:
        y_pred = np.ones_like(y_true)
    else:
        y_pred = (np.asarray(y_score, dtype=float) >= threshold).astype(int)

    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())

    if n_truth is not None:
        fn = n_truth - tp
        recall = tp / n_truth if n_truth > 0 else float('nan')
    else:
        fn_callset = int(((y_pred == 0) & (y_true == 1)).sum())
        fn = fn_callset
        recall = tp / (tp + fn) if (tp + fn) > 0 else float('nan')

    precision = tp / (tp + fp) if (tp + fp) > 0 else float('nan')
    f1 = (
        2 * precision * recall / (precision + recall)
        if (not np.isnan(precision)) and (not np.isnan(recall)) and (precision + recall) > 0
        else float('nan')
    )

    return dict(tp=tp, fp=fp, fn=fn, precision=precision, recall=recall, f1=f1)


def compute_metrics(
    df: pd.DataFrame,
    thresholds: list[float] | None = None,
    n_truth: int | None = None,
    callset: str | None = None,
) -> dict:
    """Compute all metrics for a single benchmark CSV dataframe.

    Args:
        df: Benchmark CSV with at least ``is_in_val`` and a score column.
        thresholds: Operating thresholds for SVEnsemble2 predictions.
            Raw callers ignore these (all calls are positive predictions).
        n_truth: Total truth variants for global recall / FN.
        callset: Optional callset name for qual_metric lookup.

    Returns:
        Flat dict.  Common keys: score_col, is_svensemble2, n, n_pos,
        n_truth, auroc, auprc.  Per-threshold keys are suffixed with
        ``_<threshold>`` (e.g. ``tp_0.2``, ``precision_0.2``).
    """
    if thresholds is None:
        thresholds = [0.1, 0.2, 0.3]

    df = df.copy()
    df = df.dropna(subset=['is_in_val'])

    score_col = resolve_score_column(df, callset=callset)
    y_true = df['is_in_val'].astype(int).values
    y_score = df[score_col].fillna(0.0).values
    is_sve2 = score_col.endswith('_pred')

    n_pos = int(y_true.sum())

    # -- Curve metrics (always on quality score) ----------------------------
    n_missed = max(n_truth - n_pos, 0) if n_truth is not None else 0
    if n_missed > 0:
        y_true_global = np.concatenate([y_true, np.ones(n_missed, dtype=int)])
        y_score_global = np.concatenate([y_score, np.zeros(n_missed)])
    else:
        y_true_global = y_true
        y_score_global = y_score

    # AUROC — callset-level only (no padding)
    n_pos_callset = int(y_true.sum())
    n_neg_callset = len(y_true) - n_pos_callset
    if n_pos_callset > 0 and n_neg_callset > 0:
        auroc = float(roc_auc_score(y_true, y_score))
    else:
        auroc = float('nan')

    # AUPRC — global (padded for missed truth when n_truth known)
    n_pos_global = int(y_true_global.sum())
    auprc = (
        float(average_precision_score(y_true_global, y_score_global))
        if n_pos_global > 0 else float('nan')
    )

    result = dict(
        score_col=score_col,
        is_svensemble2=is_sve2,
        n=len(df),
        n_pos=n_pos,
        n_truth=n_truth if n_truth is not None else float('nan'),
        auroc=auroc,
        auprc=auprc,
    )

    # -- Per-threshold confusion matrix -------------------------------------
    if is_sve2:
        for thr in thresholds:
            at_thr = compute_at_threshold(y_true, y_score, thr, n_truth=n_truth)
            for key, val in at_thr.items():
                result[f'{key}_{thr}'] = val
    else:
        # Raw callers: every call is a positive prediction, no threshold.
        # Values repeated in each _T column for easier comparison.
        at_callset = compute_at_threshold(y_true, y_score, None, n_truth=n_truth)
        for thr in thresholds:
            for key, val in at_callset.items():
                result[f'{key}_{thr}'] = val

    return result


# ---------------------------------------------------------------------------
# Filename parsing
# ---------------------------------------------------------------------------

def parse_benchmark_filename(path: Path) -> dict:
    """Extract metadata fields from a benchmark CSV filename.

    Benchmark filenames follow the pattern::

        {callset}.{svtypes}.{svlens}.{val_set}.benchmark.csv

    where *svtypes* is a dash-separated combination (e.g. ``DEL-INS``),
    *svlens* is a colon-separated range (e.g. ``50:inf``), and *val_set*
    may be empty (producing a ``..`` in the filename).

    Returns:
        Dict with keys: callset, svtypes (list of str), svlens_str, val_set.
    """
    name = path.name
    if name.endswith('.benchmark.csv'):
        name = name[: -len('.benchmark.csv')]
    parts = name.split('.')
    if len(parts) < 3:
        raise ValueError(
            f"Cannot parse benchmark filename: {path.name!r} "
            f"(expected at least callset.svtypes.svlens)"
        )
    val_set = parts[-1]
    svlens_str = parts[-2]
    svtypes_str = parts[-3]
    callset = '.'.join(parts[:-3])
    svtypes = svtypes_str.split('-') if svtypes_str else []
    return dict(callset=callset, svtypes=svtypes, svlens_str=svlens_str, val_set=val_set)


# ---------------------------------------------------------------------------
# Batch loading
# ---------------------------------------------------------------------------

def load_and_compute(
    csv_paths: List[Path],
    labels: List[str] | None = None,
    thresholds: list[float] | None = None,
    n_truth: int | List[int | None] | None = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """Load benchmark CSVs and compute metrics for each.

    Args:
        csv_paths: Paths to benchmark CSV files.
        labels: Optional display labels.  Defaults to file stems.
        thresholds: Operating thresholds for SVEnsemble2 (default: [0.1, 0.2, 0.3]).
            Raw callers always treat all calls as positive predictions.
        n_truth: Total truth variants; single int or per-file list.
        verbose: Print per-file progress.
    """
    if thresholds is None:
        thresholds = [0.1, 0.2, 0.3]
    if labels is None:
        labels = [p.stem for p in csv_paths]

    if isinstance(n_truth, list):
        n_truths = n_truth
        if len(n_truths) != len(csv_paths):
            raise ValueError(
                f"n_truth list has {len(n_truths)} entries but there are "
                f"{len(csv_paths)} CSV paths"
            )
    else:
        n_truths = [n_truth] * len(csv_paths)

    rows = []
    for path, label, file_n_truth in zip(csv_paths, labels, n_truths):
        if verbose:
            print(f"  Processing {path} ({label}) n_truth={file_n_truth} …")
        try:
            parsed = parse_benchmark_filename(path)
            callset = parsed['callset']
        except ValueError:
            callset = None
        try:
            df = pd.read_csv(path, low_memory=False)
            metrics = compute_metrics(
                df,
                thresholds=thresholds,
                n_truth=file_n_truth,
                callset=callset,
            )
        except Exception as exc:
            warnings.warn(f"Failed to compute metrics for '{path}': {exc}")
            continue
        rows.append({'label': label, 'path': str(path), **metrics})

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

_THRESHOLD_METRICS = ('tp', 'fp', 'fn', 'precision', 'recall', 'f1')
_FLOAT_METRICS = frozenset({'precision', 'recall', 'f1', 'auroc', 'auprc'})
_INT_METRICS = frozenset({'n', 'n_pos', 'n_truth', 'tp', 'fp', 'fn'})


def _fmt(val, metric_name: str) -> str:
    """Format a single value for display."""
    base = metric_name.rsplit('_', 1)[0] if '_' in metric_name else metric_name
    if pd.isna(val):
        return ''
    if base in _FLOAT_METRICS:
        return f'{val:.4f}'
    if base in _INT_METRICS:
        return str(int(val))
    return str(val)


def _detect_thresholds(results: pd.DataFrame) -> list[float]:
    """Infer threshold values from column names like 'tp_0.2'."""
    thresholds = set()
    for col in results.columns:
        if '_' in col:
            try:
                thresholds.add(float(col.rsplit('_', 1)[1]))
            except ValueError:
                pass
    return sorted(thresholds)


def format_results(results: pd.DataFrame) -> pd.DataFrame:
    """Return a display-ready copy of the results table.

    Columns are ordered: common info, unsuffixed callset-level metrics
    (raw callers), then threshold-grouped metrics (SVEnsemble2),
    then curve metrics.
    """
    thresholds = _detect_thresholds(results)

    common_cols = ['label', 'n', 'n_pos', 'n_truth']
    threshold_cols = []
    for thr in thresholds:
        for m in _THRESHOLD_METRICS:
            col = f'{m}_{thr}'
            if col in results.columns:
                threshold_cols.append(col)
    curve_cols = ['auroc', 'auprc']

    display_cols = [c for c in common_cols + threshold_cols + curve_cols if c in results.columns]

    rows = []
    for _, row in results.iterrows():
        formatted: dict = {}
        for col in display_cols:
            formatted[col] = _fmt(row.get(col), col)
        rows.append(formatted)

    return pd.DataFrame(rows, columns=display_cols)


def print_metrics_table(
    results: pd.DataFrame,
    verbose: bool = False,
) -> None:
    """Print a formatted metrics table to stdout."""
    thresholds = _detect_thresholds(results)
    sep = '=' * 120

    print(f'\n{sep}')
    print('METRICS SUMMARY')
    print(sep)
    print('  Raw callers: every call is a positive prediction (no threshold).')
    print('               Values repeated in each _T column for easier comparison.')
    print(f'  SVEnsemble2: thresholded on _pred probability at {thresholds}')
    print('  FN = n_truth − TP   (requires --n_truth or --sample_name)')
    print('  Recall = TP / n_truth   |   Precision = TP / (TP + FP)')
    print('  AUROC/AUPRC: computed on quality score (not thresholded)')
    print(sep)

    disp = format_results(results)
    print(disp.to_string(index=False))

    if verbose and 'is_svensemble2' in results.columns:
        print(f'\n{sep}')
        print('Score column used per callset:')
        for _, row in results.iterrows():
            kind = 'SVEnsemble2' if row.get('is_svensemble2') else 'raw caller'
            print(f"  {row['label']:<50s}  [{kind}]  score_col = {row['score_col']}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser(
        description=__doc__,
        formatter_class=__import__('argparse').RawDescriptionHelpFormatter,
    )

    # --- Input / labelling ------------------------------------------------
    csv_input = parser.add_mutually_exclusive_group(required=True)
    csv_input.add_argument(
        '--benchmark_csvs', type=str, default=None,
        help='Comma-separated list of benchmark CSV paths.'
    )
    csv_input.add_argument(
        '--benchmark_csvs_file', type=str, default=None,
        help='Path to a text file with one benchmark CSV path per line.  '
             'Use "-" to read paths from stdin (e.g. find … | … --benchmark_csvs_file -).'
    )
    parser.add_argument(
        '--labels', type=str, default=None,
        help='Comma-separated display labels, one per CSV.  Defaults to file stems.'
    )

    # --- Truth count resolution -------------------------------------------
    parser.add_argument(
        '--n_truth', type=int, default=None,
        help='Total number of truth variants (for global Recall / FN).  '
             'Provide either this or --sample_name.'
    )
    parser.add_argument(
        '--sample_name', type=str, default=None,
        help='Sample name (e.g. NA12878).  Used to resolve n_truth automatically '
             'when --n_truth is not given.'
    )
    parser.add_argument(
        '--val_set_path', type=str, default=None,
        help='Override path to truth BPS CSV (requires --val_set_name).'
    )
    parser.add_argument(
        '--val_set_name', type=str, default=None,
        help='Override validation set source name (requires --val_set_path).'
    )

    # --- SV filtering (for truth count resolution) ------------------------
    parser.add_argument(
        '--svtypes', type=str, default=None,
        help='Comma-separated SV types to consider (e.g. DEL,INS,DUP,INV).'
    )
    parser.add_argument(
        '--svlen_lower', type=int, default=caller_params.default_sv_lens[0],
        help='Minimum SVLEN filter (default: %(default)s).'
    )
    parser.add_argument(
        '--svlen_upper', type=float, default=caller_params.default_sv_lens[1],
        help='Maximum SVLEN filter (default: %(default)s).'
    )

    # --- Thresholds -------------------------------------------------------
    parser.add_argument(
        '--thresholds', type=float, nargs='+', default=[0.1, 0.2, 0.3],
        help='Operating thresholds for SVEnsemble2 predictions.  Raw callers '
             'always treat all calls as positive predictions regardless of '
             'threshold (default: 0.1 0.2 0.3).'
    )

    # --- Output -----------------------------------------------------------
    parser.add_argument(
        '--output', type=str, default=None,
        help='Optional path to write metrics as a CSV file.'
    )
    parser.add_argument(
        '--verbose', action='store_true', default=False,
        help='Print per-file progress and score column notes.'
    )

    args = parser.parse_args()

    # Parse CSV paths
    if args.benchmark_csvs is not None:
        csv_paths = [Path(p.strip()) for p in args.benchmark_csvs.split(sep=',')]
    elif args.benchmark_csvs_file == '-':
        lines = sys.stdin.read().splitlines()
        csv_paths = [
            Path(line.strip())
            for line in lines
            if line.strip() and not line.strip().startswith('#')
        ]
        if not csv_paths:
            raise ValueError("No paths received on stdin.")
    else:
        list_file = Path(args.benchmark_csvs_file)
        if not list_file.exists():
            raise FileNotFoundError(f"--benchmark_csvs_file not found: {list_file}")
        csv_paths = [
            Path(line.strip())
            for line in list_file.read_text().splitlines()
            if line.strip() and not line.strip().startswith('#')
        ]
    for p in csv_paths:
        if not p.exists():
            raise FileNotFoundError(f"Benchmark CSV not found: {p}")

    # Parse labels
    if args.labels is not None:
        labels = [s.strip() for s in args.labels.split(sep=',')]
        if len(labels) != len(csv_paths):
            raise ValueError(
                f"--labels has {len(labels)} entries but the benchmark CSV list "
                f"has {len(csv_paths)}"
            )
    else:
        labels = None

    # Parse svtypes (global override — forces the same svtypes for all files)
    svtypes_override = None
    if args.svtypes is not None and args.svtypes.lower() != 'all':
        svtypes_override = args.svtypes.split(sep=',')
        for svtype in svtypes_override:
            if svtype not in caller_params.default_sv_types:
                raise ValueError(f"Unsupported svtype: {svtype!r}")

    svlens = (args.svlen_lower, args.svlen_upper)

    # Resolve n_truth — per-file when possible
    if args.n_truth is not None:
        n_truths = [args.n_truth] * len(csv_paths)
    elif args.sample_name is not None:
        truth_cache: dict[tuple, int] = {}
        n_truths = []
        for p in csv_paths:
            if svtypes_override is not None:
                file_svtypes = svtypes_override
            else:
                try:
                    file_svtypes = parse_benchmark_filename(p)['svtypes']
                except ValueError:
                    file_svtypes = None
            cache_key = tuple(sorted(file_svtypes)) if file_svtypes else ()
            if cache_key not in truth_cache:
                truth_cache[cache_key] = bench.get_truth_count(
                    sample_name=args.sample_name,
                    svtypes=file_svtypes or None,
                    svlens=svlens,
                    val_set_path=args.val_set_path or None,
                    val_set_name=args.val_set_name or None,
                )
                if args.verbose:
                    print(f"Resolved n_truth = {truth_cache[cache_key]} "
                          f"for svtypes={file_svtypes or 'all'}")
            n_truths.append(truth_cache[cache_key])
    else:
        n_truths = [None] * len(csv_paths)
        warnings.warn(
            "n_truth not provided and could not be resolved.  "
            "Global Recall, F1, and FN will not be computed.  "
            "Provide --n_truth or --sample_name."
        )

    if args.verbose:
        print(f"\nProcessing {len(csv_paths)} benchmark CSV(s)…")

    results = load_and_compute(
        csv_paths=csv_paths,
        labels=labels,
        thresholds=args.thresholds,
        n_truth=n_truths,
        verbose=args.verbose,
    )

    print_metrics_table(results, verbose=args.verbose)

    if args.output is not None:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        results.to_csv(out_path, index=False)
        print(f"\nMetrics written to {out_path}")
