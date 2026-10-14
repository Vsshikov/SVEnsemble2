"""
Compute SHAP values for a trained SVEnsemble2 model.

run_shap takes a model pickle and the feature matrix written by the train
step and writes {sample}.{model_id}.{training_set}.shap_values.pkl and
.shap_summary.csv into the output directory. It backs the `svensemble shap`
subcommand.
"""

import time
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import shap


from . import _config as sv_config


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
    to match model.feature_names_in_.
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
        use_gpu: Use shap.explainers.GPUTreeExplainer when True.
        max_samples: If set, subsample *X* to at most this many rows for
            the background dataset and explanation (useful for large datasets).
        verbose: Print progress information.

    Returns:
        shap.Explanation object.
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
def run_shap(
    model_pkl: Path,
    feature_matrix: Path,
    sample_name: str,
    model_name: str,
    training_set: str,
    output_dir: Path,
    model_config: dict | None = None,
    verbose: bool = False,
) -> Path:
    """
    Pipeline-friendly SHAP entry point.  Accepts file paths directly
    rather than discovering them via sample/model configs.

    Writes {prefix}.shap_summary.csv and {prefix}.shap_values.pkl into output_dir.
    """
    # Load model
    with open(model_pkl, 'rb') as f:
        model = pickle.load(f)

    fm = pd.read_csv(feature_matrix, low_memory=False)
    fm = fm.drop(columns=['ref_match'], errors='ignore')
    id_cols = [c for c in fm.columns if c.endswith('_INFO_EVENT')]
    if id_cols:
        fm = fm.set_index(id_cols)

    resolved = (
        sv_config.resolve_model_config(config=model_config) if model_config
        else sv_config.resolve_model_config(name=model_name)
    )
    mid = sv_config.model_id(resolved)
    fillna = resolved['training_params'].get('fillna', 0)
    X = pd.get_dummies(fm)
    if fillna is not None:
        X = X.fillna(fillna)

    X = align_features(model, X, verbose=verbose)

    if verbose:
        print(f"Feature matrix shape: {X.shape}")

    shap_values = compute_shap_values(model, X, verbose=verbose)

    prefix = f'{sample_name}.{mid}.{training_set}'
    summary = summarise_shap_values(shap_values)
    summary_csv_path = output_dir / f'{prefix}.shap_summary.csv'
    summary.to_csv(summary_csv_path, index=False)
    print(f"Saved SHAP summary (top {len(summary)}): {summary_csv_path}")
    print(summary.to_string(index=False))

    shap_pkl_path = output_dir / f'{prefix}.shap_values.pkl'
    with open(shap_pkl_path, 'wb') as f:
        pickle.dump(shap_values, f)
    print(f"Saved SHAP values: {shap_pkl_path}")

    return shap_pkl_path
