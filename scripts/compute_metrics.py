#! /usr/bin/env python3
"""
Compute classification metrics from SVEnsemble2 benchmark CSV files.

For each benchmark CSV (produced by svensemble/benchmark.py), reports counts and
scores at both a chosen operating threshold and across the full score range:

  Counts at threshold
  -------------------
  TP   — calls at or above threshold that match a truth variant
  FP   — calls at or above threshold that do not match a truth variant
  TN   — calls below threshold that do not match a truth variant
         (SVEnsemble2: correctly low-scored calls; regular callers: correctly
          filtered calls — note this is incomplete since calls not made by the
          caller are absent from the benchmark CSV)
  FN   — calls below threshold that match a truth variant (within the callset)
  FN*  — truth variants not recalled at all (n_truth - TP); requires --n_truth
          or --sample_name so the total truth count can be resolved

  Rates at threshold
  ------------------
  Precision   = TP / (TP + FP)
  Recall      = TP / n_truth  [if n_truth available]  or  TP / (TP + FN)
  F1          = harmonic mean of Precision and Recall

  Curve metrics (threshold-free)
  -------------------------------
  AUROC  — area under the receiver-operating-characteristic curve
  AUPRC  — area under the precision-recall curve

Threshold selection per callset:
  • SVEnsemble2 predictions (score column ends with ``_pred``): use --threshold
    (default 0.5).
  • Regular callers that have a ``FILTER_PASS`` column: FILTER_PASS is used as
    the binary predictor.  Pass --ignore_filter_pass to use --threshold instead.
  • Regular callers without FILTER_PASS: use --threshold on the quality metric.

Usage:
    python scripts/compute_metrics.py \\
        --benchmark_csvs a.benchmark.csv,b.benchmark.csv \\
        --labels callerA,svensemble2_B \\
        --sample_name NA12878 \\
        --svtypes DEL,INS \\
        --svlen_lower 50 \\
        --threshold 0.5 \\
        [--output metrics.csv] \\
        [--verbose]
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
    precision_recall_curve,
    roc_auc_score,
)

if __name__ == '__main__':
    WORKDIR = "/vast/ai_projects/Papenfuss_SVEnsemble2"
    os.chdir(WORKDIR)
    if WORKDIR not in sys.path:
        sys.path.insert(0, WORKDIR)
    import svensemble._caller_params as caller_params
    import svensemble.benchmark as bench
else:
    from . import _caller_params as caller_params
    from . import benchmark as bench


# ---------------------------------------------------------------------------
# Column helpers
# ---------------------------------------------------------------------------

_META_COLS = frozenset({'INFO_EVENT', 'is_in_val', 'FILTER_PASS', 'INFO_INFERRED_SVTYPE'})


def detect_score_column(df: pd.DataFrame) -> str:
    """Return the quality/prediction score column from a benchmark CSV.

    Args:
        df: Benchmark CSV dataframe.

    Returns:
        Name of the score column (first column that is not a metadata column).

    Raises:
        ValueError: If no score column can be identified.
    """
    score_cols = [c for c in df.columns if c not in _META_COLS]
    if not score_cols:
        raise ValueError(
            f"No score column found.  Columns present: {list(df.columns)}"
        )
    return score_cols[0]


def is_svensemble2_csv(df: pd.DataFrame) -> bool:
    """Return True when the benchmark CSV comes from an SVEnsemble2 model.

    SVEnsemble2 prediction columns follow the ``{training_set}_{model}_pred``
    naming convention produced by ``validate_svensemble2_predictions``.
    """
    return detect_score_column(df).endswith('_pred')


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def compute_at_threshold(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float,
    *,
    use_filter_pass: bool = False,
    filter_pass: np.ndarray | None = None,
) -> dict:
    """Compute confusion-matrix counts and derived rates at a single threshold.

    Args:
        y_true: Binary ground-truth labels (1 = matched a truth variant).
        y_score: Quality metric or model prediction scores.
        threshold: Calls with score >= threshold are predicted positive.
        use_filter_pass: If True, use ``filter_pass`` instead of ``threshold``.
        filter_pass: Boolean/int array for FILTER_PASS (1 = PASS); required when
            ``use_filter_pass`` is True.

    Returns:
        Dict with keys: tp, fp, tn, fn, precision, recall_callset, f1_callset,
        threshold_used.
    """
    y_true = np.asarray(y_true, dtype=int)

    if use_filter_pass and filter_pass is not None:
        y_pred = np.asarray(filter_pass, dtype=int)
        threshold_label = 'FILTER_PASS'
    else:
        y_pred = (np.asarray(y_score, dtype=float) >= threshold).astype(int)
        threshold_label = f'{threshold:.3f}'

    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else float('nan')
    # Recall within the callset (TP / (TP + FN_callset))
    recall_callset = tp / (tp + fn) if (tp + fn) > 0 else float('nan')
    f1_callset = (
        2 * precision * recall_callset / (precision + recall_callset)
        if (not np.isnan(precision)) and (not np.isnan(recall_callset)) and (precision + recall_callset) > 0
        else float('nan')
    )

    return dict(
        tp=tp,
        fp=fp,
        tn=tn,
        fn=fn,
        precision=precision,
        recall_callset=recall_callset,
        f1_callset=f1_callset,
        threshold_used=threshold_label,
    )


def compute_metrics(
    df: pd.DataFrame,
    threshold: float = 0.5,
    n_truth: int | None = None,
    ignore_filter_pass: bool = False,
) -> dict:
    """Compute all metrics for a single benchmark CSV dataframe.

    Args:
        df: Benchmark CSV with at least ``is_in_val`` and a score column.
        threshold: Operating threshold for binary classification.  Ignored for
            basic-caller CSVs that have a FILTER_PASS column (unless
            ``ignore_filter_pass`` is set).
        n_truth: Total number of truth variants in the filtered validation set.
            When provided, global recall (TP / n_truth) and FN* (n_truth − TP)
            are reported in addition to the callset-level values.
        ignore_filter_pass: If True, ignore the FILTER_PASS column and apply
            ``threshold`` to the quality metric instead.

    Returns:
        Dict with keys: score_col, is_svensemble2, threshold_used, n, n_pos,
        tp, fp, tn, fn, fn_global, precision, recall, f1, auroc, auprc.
        ``fn_global`` and global recall/f1 are ``nan`` when ``n_truth`` is None.
    """
    df = df.copy()
    df = df.dropna(subset=['is_in_val'])

    score_col = detect_score_column(df)
    y_true = df['is_in_val'].astype(int).values
    y_score = df[score_col].fillna(0.0).values

    is_sve2 = score_col.endswith('_pred')
    use_filter_pass = (
        (not is_sve2)
        and (not ignore_filter_pass)
        and ('FILTER_PASS' in df.columns)
    )
    filter_pass = df['FILTER_PASS'].astype(int).values if use_filter_pass else None

    n_pos = int(y_true.sum())
    n_neg = len(y_true) - n_pos

    # AUROC — requires both positive and negative labels
    if n_pos > 0 and n_neg > 0:
        auroc = float(roc_auc_score(y_true, y_score))
    else:
        auroc = float('nan')

    # AUPRC — requires at least one positive
    auprc = float(average_precision_score(y_true, y_score)) if n_pos > 0 else float('nan')

    at_thr = compute_at_threshold(
        y_true, y_score, threshold,
        use_filter_pass=use_filter_pass,
        filter_pass=filter_pass,
    )

    tp = at_thr['tp']
    fp = at_thr['fp']
    tn = at_thr['tn']
    fn_callset = at_thr['fn']
    precision = at_thr['precision']
    recall_callset = at_thr['recall_callset']
    f1_callset = at_thr['f1_callset']

    if n_truth is not None:
        fn_global = n_truth - tp
        recall_global = tp / n_truth if n_truth > 0 else float('nan')
        f1_global = (
            2 * precision * recall_global / (precision + recall_global)
            if (not np.isnan(precision)) and (not np.isnan(recall_global)) and (precision + recall_global) > 0
            else float('nan')
        )
    else:
        fn_global = float('nan')
        recall_global = recall_callset
        f1_global = f1_callset

    return dict(
        score_col=score_col,
        is_svensemble2=is_sve2,
        threshold_used=at_thr['threshold_used'],
        n=len(df),
        n_pos=n_pos,
        tp=tp,
        fp=fp,
        tn=tn,
        fn=fn_callset,
        fn_global=fn_global,
        precision=precision,
        recall=recall_global,
        f1=f1_global,
        auroc=auroc,
        auprc=auprc,
    )


# ---------------------------------------------------------------------------
# Batch loading
# ---------------------------------------------------------------------------

def load_and_compute(
    csv_paths: List[Path],
    labels: List[str] | None = None,
    threshold: float = 0.5,
    n_truth: int | None = None,
    ignore_filter_pass: bool = False,
    verbose: bool = False,
) -> pd.DataFrame:
    """Load benchmark CSVs and compute metrics for each.

    Args:
        csv_paths: Paths to benchmark CSV files.
        labels: Optional display labels for each callset.  Defaults to the file
            stem of each path.
        threshold: Score threshold for binary classification.
        n_truth: Total number of truth variants; used for global recall and FN*.
        ignore_filter_pass: If True, ignore FILTER_PASS columns.
        verbose: Print per-file progress.

    Returns:
        DataFrame with one row per callset and columns for each metric.
    """
    if labels is None:
        labels = [p.stem for p in csv_paths]

    rows = []
    for path, label in zip(csv_paths, labels):
        if verbose:
            print(f"  Processing {path} ({label}) …")
        try:
            df = pd.read_csv(path, low_memory=False)
            metrics = compute_metrics(
                df,
                threshold=threshold,
                n_truth=n_truth,
                ignore_filter_pass=ignore_filter_pass,
            )
        except Exception as exc:
            warnings.warn(f"Failed to compute metrics for '{path}': {exc}")
            continue
        rows.append({'label': label, 'path': str(path), **metrics})

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

_FLOAT_COLS = {'precision', 'recall', 'f1', 'auroc', 'auprc'}
_INT_COLS   = {'n', 'n_pos', 'tp', 'fp', 'tn', 'fn', 'fn_global'}


def format_results(results: pd.DataFrame) -> pd.DataFrame:
    """Return a display-ready copy of the results table.

    Args:
        results: Raw metrics DataFrame from ``load_and_compute``.

    Returns:
        Formatted DataFrame suitable for printing, with float and integer
        columns rendered as strings.
    """
    display_cols = [
        'label', 'n', 'n_pos',
        'tp', 'fp', 'tn', 'fn', 'fn_global',
        'precision', 'recall', 'f1',
        'auroc', 'auprc',
        'threshold_used',
    ]
    display_cols = [c for c in display_cols if c in results.columns]

    rows = []
    for _, row in results.iterrows():
        formatted: dict = {}
        for col in display_cols:
            val = row.get(col)
            if col in _FLOAT_COLS:
                formatted[col] = f'{val:.4f}' if pd.notna(val) else 'N/A'
            elif col in _INT_COLS:
                formatted[col] = str(int(val)) if pd.notna(val) else 'N/A'
            else:
                formatted[col] = val
        rows.append(formatted)

    return pd.DataFrame(rows, columns=display_cols)


def print_metrics_table(
    results: pd.DataFrame,
    n_truth: int | None = None,
    verbose: bool = False,
) -> None:
    """Print a formatted metrics table to stdout.

    Args:
        results: Metrics DataFrame from ``load_and_compute``.
        n_truth: Total truth variants used (for header display only).
        verbose: Print per-row notes about TN interpretation.
    """
    sep = '=' * 110

    print(f'\n{sep}')
    print('METRICS SUMMARY')
    print(sep)
    if n_truth is not None:
        print(f'  Total truth variants (n_truth): {n_truth}')
    print('  TN  = calls below threshold / not FILTER_PASS that do NOT match a truth variant')
    print('        (for regular callers this count is restricted to calls present in the callset;')
    print('         calls never made by the caller are absent and cannot be counted)')
    print('  FN  = calls at or above threshold / FILTER_PASS that match a truth variant')
    print('  FN* = n_truth − TP  (requires --n_truth or --sample_name; shown in fn_global column)')
    print('  Recall, F1 use global recall = TP / n_truth when n_truth is available, else TP / (TP + FN)')
    print(sep)

    disp = format_results(results)
    print(disp.to_string(index=False))

    if verbose and 'is_svensemble2' in results.columns:
        print(f'\n{sep}')
        print('Threshold used per callset:')
        for _, row in results.iterrows():
            kind = 'SVEnsemble2' if row.get('is_svensemble2') else 'regular caller'
            print(f"  {row['label']:<50s}  [{kind}]  threshold = {row['threshold_used']}")


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
    parser.add_argument(
        '--benchmark_csvs', type=str, required=True,
        help='Comma-separated list of benchmark CSV paths.'
    )
    parser.add_argument(
        '--labels', type=str, default=None,
        help='Comma-separated display labels, one per CSV.  Defaults to file stems.'
    )

    # --- Truth count resolution -------------------------------------------
    parser.add_argument(
        '--n_truth', type=int, default=None,
        help='Total number of truth variants (for global Recall / FN*).  '
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

    # --- Threshold / behaviour --------------------------------------------
    parser.add_argument(
        '--threshold', type=float, default=0.5,
        help='Score threshold for binary classification of SVEnsemble2 predictions '
             'and regular callers without FILTER_PASS (default: %(default)s).'
    )
    parser.add_argument(
        '--ignore_filter_pass', action='store_true', default=False,
        help='Ignore FILTER_PASS columns and apply --threshold to the quality metric '
             'for all regular callers.'
    )

    # --- Output -----------------------------------------------------------
    parser.add_argument(
        '--output', type=str, default=None,
        help='Optional path to write metrics as a CSV file.'
    )
    parser.add_argument(
        '--verbose', action='store_true', default=False,
        help='Print per-file progress and threshold notes.'
    )

    args = parser.parse_args()

    # Parse CSV paths
    csv_paths = [Path(p.strip()) for p in args.benchmark_csvs.split(sep=',')]
    for p in csv_paths:
        if not p.exists():
            raise FileNotFoundError(f"Benchmark CSV not found: {p}")

    # Parse labels
    if args.labels is not None:
        labels = [l.strip() for l in args.labels.split(sep=',')]
        if len(labels) != len(csv_paths):
            raise ValueError(
                f"--labels has {len(labels)} entries but --benchmark_csvs has "
                f"{len(csv_paths)}"
            )
    else:
        labels = None

    # Parse svtypes
    svtypes = None
    if args.svtypes is not None and args.svtypes.lower() != 'all':
        svtypes = args.svtypes.split(sep=',')
        for svtype in svtypes:
            if svtype not in caller_params.default_sv_types:
                raise ValueError(f"Unsupported svtype: {svtype!r}")

    svlens = (args.svlen_lower, args.svlen_upper)

    # Resolve n_truth
    n_truth = args.n_truth
    if n_truth is None and args.sample_name is not None:
        n_truth = bench.get_truth_count(
            sample_name=args.sample_name,
            svtypes=svtypes,
            svlens=svlens,
            val_set_path=args.val_set_path or None,
            val_set_name=args.val_set_name or None,
        )
        if args.verbose:
            print(f"Resolved n_truth = {n_truth} from sample '{args.sample_name}'")

    if n_truth is None:
        warnings.warn(
            "n_truth not provided and could not be resolved.  "
            "Global Recall, F1, and FN* will not be computed.  "
            "Provide --n_truth or --sample_name."
        )

    if args.verbose:
        print(f"\nProcessing {len(csv_paths)} benchmark CSV(s)…")

    results = load_and_compute(
        csv_paths=csv_paths,
        labels=labels,
        threshold=args.threshold,
        n_truth=n_truth,
        ignore_filter_pass=args.ignore_filter_pass,
        verbose=args.verbose,
    )

    print_metrics_table(results, n_truth=n_truth, verbose=args.verbose)

    if args.output is not None:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        results.to_csv(out_path, index=False)
        print(f"\nMetrics written to {out_path}")
