#! /usr/bin/env python3

"""
Compute and assess SHAP values for SVEnsemble2 models.

Supports two data loading modes:
  - bench_pkl : from bench_preprocessed pickle (default, uses importances module)
  - workflow  : from workflow directory (model.pkl + feature_matrix.csv)

Outputs saved to:
  data/{sample}/sv_calling/svensemble2/shap_values/

Usage:
    python svensemble/sv_shap.py \\
        --sample_name NA12878 --training_set delly+manta_diploid --model_name unbiased

    python svensemble/sv_shap.py \\
        --sample_name NA12878 --training_set delly+manta_diploid --model_name unbiased \\
        --use_gpu --plot

    python svensemble/sv_shap.py \\
        --sample_name NA12878 --training_set delly+manta_diploid --model_name unbiased \\
        --source workflow --max_samples 5000 --plot
"""

import time
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import shap


from . import utils as sv_utils
from . import _caller_params as caller_params


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------
def resolve_shap_paths(
    sample_name: str,
    model_name: str,
    training_set_name: str,
    ground_truth_name: str | None = None,
) -> dict[str, Path]:
    """Resolve SVEnsemble2 paths for SHAP analysis via the unified config system.

    Delegates to ``sv_utils.create_config_dict`` so that the ``gt_`` suffix
    (e.g. ``.gt_gnomad_pass``) is applied consistently with the training
    pipeline.
    """
    config = sv_utils.create_config_dict(
        sample_name=sample_name,
        model_name=model_name,
        training_set=training_set_name,
        ground_truth_name=ground_truth_name,
    )
    workflow_dir = config[sample_name]['workflow_dir']
    return {
        'base': workflow_dir.parent,
        'workflow_dir': workflow_dir,
        'model_pkl': workflow_dir / 'model.pkl',
        'feature_matrix': workflow_dir / 'feature_matrix.csv',
        'shap_dir': workflow_dir / 'shap_values',
    }


# ---------------------------------------------------------------------------
# Feature alignment
# ---------------------------------------------------------------------------
def align_features(
    model,
    X: pd.DataFrame,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Align DataFrame columns with the model's expected feature set.

    Drops extra columns present in *X* but absent from the model, raises
    if *X* is missing features required by the model, and reorders columns
    to match ``model.feature_names_in_``.
    """
    if not hasattr(model, 'feature_names_in_'):
        return X

    model_features = set(model.feature_names_in_)
    data_features = set(X.columns)

    extra = data_features - model_features
    missing = model_features - data_features

    if extra:
        if verbose:
            print(f"Dropping {len(extra)} extra features not in model: "
                  f"{sorted(extra)[:10]}{'...' if len(extra) > 10 else ''}")
        X = X.drop(columns=list(extra))

    if missing:
        raise ValueError(
            f"Missing {len(missing)} features required by model: {sorted(missing)}"
        )

    return X[list(model.feature_names_in_)]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_from_workflow(
    sample_name: str,
    model_name: str,
    training_set_name: str,
    ground_truth_name: str | None = None,
    verbose: bool = False,
) -> tuple:
    """
    Load model and feature matrix from the workflow directory.

    Returns:
        (model, X, y) where y may be None if no ref_match column is present.
    """
    paths = resolve_shap_paths(sample_name, model_name, training_set_name,
                               ground_truth_name=ground_truth_name)
    model_path = paths['model_pkl']
    fm_path = paths['feature_matrix']

    if not model_path.exists():
        raise FileNotFoundError(f"Model pickle not found: {model_path}")
    if not fm_path.exists():
        raise FileNotFoundError(f"Feature matrix not found: {fm_path}")

    if verbose:
        print(f"Loading model from: {model_path}")
        print(f"Loading feature matrix from: {fm_path}")

    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    fm = pd.read_csv(fm_path, low_memory=False)

    y = None
    if 'ref_match' in fm.columns:
        y = fm['ref_match']
        fm = fm.drop(columns=['ref_match'])

    id_cols = [c for c in fm.columns if c.endswith('_INFO_EVENT')]
    if id_cols:
        fm = fm.set_index(id_cols)

    from . import _model_configs
    fillna = _model_configs.model_configs.get(model_name, {}).get(
        'training_params', {},
    ).get('fillna', 0)

    X = pd.get_dummies(fm)
    if fillna is not None:
        X = X.fillna(fillna)

    return model, X, y


def load_model_and_data(
    sample_name: str,
    model_name: str,
    training_set_name: str,
    ground_truth_name: str | None = None,
    verbose: bool = False,
) -> tuple:
    """
    Dispatch loader and align features with the model.

    Returns:
        (model, X, y)
    """
    model, X, y = load_from_workflow(
        sample_name, model_name, training_set_name,
        ground_truth_name=ground_truth_name, verbose=verbose,
    )

    X = align_features(model, X, verbose=verbose)

    if verbose:
        print(f"Feature matrix shape: {X.shape}")
        if y is not None:
            print(f"Label distribution: {y.value_counts().to_dict()}")

    return model, X, y


# ---------------------------------------------------------------------------
# SHAP computation
# ---------------------------------------------------------------------------
def compute_shap_values(
    model,
    X: pd.DataFrame,
    use_gpu: bool = False,
    max_samples: Optional[int] = None,
    verbose: bool = False,
) -> shap.Explanation:
    """
    Compute SHAP values with TreeExplainer.

    Args:
        model: Trained sklearn-compatible tree model.
        X: Feature matrix.
        use_gpu: Use ``shap.explainers.GPUTreeExplainer`` when True.
        max_samples: If set, subsample *X* to at most this many rows for
            the background dataset and explanation (useful for large datasets).
        verbose: Print progress information.

    Returns:
        ``shap.Explanation`` object.
    """
    if max_samples is not None and len(X) > max_samples:
        if verbose:
            print(f"Subsampling {len(X)} rows to {max_samples} for SHAP computation")
        rng = np.random.RandomState(42)
        idx = rng.choice(len(X), size=max_samples, replace=False)
        X = X.iloc[sorted(idx)]

    if verbose:
        explainer_type = "GPUTreeExplainer" if use_gpu else "TreeExplainer"
        print(f"Creating {explainer_type} (n_features={X.shape[1]}, n_samples={X.shape[0]})")

    start = time.time()

    if use_gpu:
        explainer = shap.explainers.GPUTreeExplainer(model, X)
    else:
        explainer = shap.explainers.TreeExplainer(model, X)

    shap_values = explainer(X, check_additivity=not use_gpu)
    elapsed = time.time() - start

    if verbose:
        print(f"SHAP computation completed in {elapsed:.1f}s")

    return shap_values


# ---------------------------------------------------------------------------
# SHAP assessment / summary statistics
# ---------------------------------------------------------------------------
def summarise_shap_values(
    shap_values: shap.Explanation,
    top_n: int = 20,
) -> pd.DataFrame:
    """
    Return a summary DataFrame of mean |SHAP| values per feature.

    For multi-output explanations (e.g. binary RF with two classes) the
    positive-class output is used.
    """
    vals = shap_values.values
    if vals.ndim == 3:
        vals = vals[:, :, 1]

    mean_abs = np.abs(vals).mean(axis=0)
    feature_names = shap_values.feature_names

    summary = (
        pd.DataFrame({'feature': feature_names, 'mean_abs_shap': mean_abs})
        .sort_values('mean_abs_shap', ascending=False)
        .reset_index(drop=True)
    )
    return summary.head(top_n) if top_n else summary


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------
def generate_shap_plots(
    shap_values: shap.Explanation,
    output_dir: Path,
    prefix: str,
    top_n: int = 20,
    verbose: bool = False,
) -> list[Path]:
    """
    Generate beeswarm and bar SHAP summary plots.

    Saves to *output_dir* with filenames prefixed by *prefix*.
    Returns list of saved file paths.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []

    sv = shap_values
    if sv.values.ndim == 3:
        sv = shap_values[:, :, 1]

    # Beeswarm
    try:
        fig_bee = plt.figure()
        shap.plots.beeswarm(sv, max_display=top_n, show=False)
        bee_path = output_dir / f'{prefix}.beeswarm.png'
        plt.savefig(bee_path, dpi=150, bbox_inches='tight')
        plt.close(fig_bee)
        saved.append(bee_path)
        if verbose:
            print(f"Saved beeswarm plot: {bee_path}")
    except Exception as e:
        print(f"Warning: beeswarm plot failed: {e}")

    # Bar
    try:
        fig_bar = plt.figure()
        shap.plots.bar(sv, max_display=top_n, show=False)
        bar_path = output_dir / f'{prefix}.bar.png'
        plt.savefig(bar_path, dpi=150, bbox_inches='tight')
        plt.close(fig_bar)
        saved.append(bar_path)
        if verbose:
            print(f"Saved bar plot: {bar_path}")
    except Exception as e:
        print(f"Warning: bar plot failed: {e}")

    return saved


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------
def run_shap_assessment(
    sample_name: str,
    model_name: str,
    training_set_name: str,
    ground_truth_name: str | None = None,
    use_gpu: bool = False,
    max_samples: Optional[int] = None,
    plot: bool = False,
    top_n: int = 20,
    verbose: bool = False,
) -> Path:
    """
    End-to-end SHAP assessment: load data, compute, save, and optionally plot.

    Returns:
        Path to the saved SHAP values pickle.
    """
    print(f"{'='*65}")
    print(f"SHAP assessment: {sample_name} | {model_name} | {training_set_name}")
    gt_label = ground_truth_name or '(default)'
    print(f"  ground_truth={gt_label}  use_gpu={use_gpu}  max_samples={max_samples}")
    print(f"{'='*65}")

    # --- Load ---
    model, X, y = load_model_and_data(
        sample_name, model_name, training_set_name,
        ground_truth_name=ground_truth_name, verbose=verbose,
    )

    # --- Compute ---
    shap_values = compute_shap_values(
        model, X, use_gpu=use_gpu, max_samples=max_samples, verbose=verbose,
    )

    # --- Save ---
    paths = resolve_shap_paths(sample_name, model_name, training_set_name,
                               ground_truth_name=ground_truth_name)
    shap_dir = paths['shap_dir']
    shap_dir.mkdir(parents=True, exist_ok=True)

    prefix = f'{sample_name}.{model_name}.{training_set_name}'
    shap_pkl_path = shap_dir / f'{prefix}.shap_values.pkl'
    with open(shap_pkl_path, 'wb') as f:
        pickle.dump(shap_values, f)
    print(f"Saved SHAP values: {shap_pkl_path}")

    # --- Summary ---
    summary = summarise_shap_values(shap_values, top_n=top_n)
    summary_csv_path = shap_dir / f'{prefix}.shap_summary.csv'
    summary.to_csv(summary_csv_path, index=False)
    print(f"Saved SHAP summary (top {top_n}): {summary_csv_path}")
    print(summary.to_string(index=False))

    # --- Plots ---
    if plot:
        generate_shap_plots(
            shap_values, output_dir=shap_dir, prefix=prefix,
            top_n=top_n, verbose=verbose,
        )

    print(f"\nSHAP assessment complete for {prefix}")
    return shap_pkl_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser(
        description=__doc__,
        formatter_class=__import__('argparse').RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        '--sample_name', type=str, required=True,
        help="Sample name (e.g. NA12878, HG002)",
    )
    parser.add_argument(
        '--model_name', type=str, default='unbiased',
        help="Model name (default: unbiased)",
    )
    parser.add_argument(
        '--training_set', type=str, required=True,
        help="Training set name (e.g. delly+manta_diploid)",
    )
    parser.add_argument(
        '--ground_truth_sample_name', type=str, default=None,
        help="Ground truth sample name (e.g. gnomad_pass). "
             "Appends .gt_<name> suffix to workflow directory when set "
             "and differs from the default reference set.",
    )
    parser.add_argument(
        '--use_gpu', action='store_true', default=False,
        help="Use GPUTreeExplainer (requires CUDA and max_depth <= 31)",
    )
    parser.add_argument(
        '--max_samples', type=int, default=None,
        help="Subsample to at most N rows for SHAP computation",
    )
    parser.add_argument(
        '--plot', action='store_true', default=False,
        help="Generate beeswarm and bar summary plots",
    )
    parser.add_argument(
        '--top_n', type=int, default=20,
        help="Number of top features to display/plot (default: 20)",
    )
    parser.add_argument(
        '--verbose', action='store_true', default=False,
        help="Print detailed progress information",
    )
    args = parser.parse_args()

    training_set_name = args.training_set
    if training_set_name not in caller_params.training_set_presets:
        print(f"Warning: training set '{training_set_name}' not in standard presets")

    run_shap_assessment(
        sample_name=args.sample_name,
        model_name=args.model_name,
        training_set_name=training_set_name,
        ground_truth_name=args.ground_truth_sample_name,
        use_gpu=args.use_gpu,
        max_samples=args.max_samples,
        plot=args.plot,
        top_n=args.top_n,
        verbose=args.verbose,
    )
