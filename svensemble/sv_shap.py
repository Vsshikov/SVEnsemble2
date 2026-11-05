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
from . import io as sv_io


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
# Main orchestrator
# ---------------------------------------------------------------------------
def run_shap(
    model_pkl: Path,
    feature_matrix: Path,
    sample: str,
    model: str,
    training_set: str,
    output_dir: Path,
    model_config: dict | None = None,
    use_gpu: bool = False,
    max_samples: Optional[int] = None,
    top_n: int = 20,
    verbose: bool = False,
) -> Path:
    """
    Pipeline-friendly SHAP entry point.  Accepts file paths directly
    rather than discovering them via sample/model configs.

    Writes shap_summary.csv and shap_values.pkl (io.SHAP_SUMMARY,
    io.SHAP_VALUES) into output_dir; the sample, model and training set are
    identified by the directory, not the filenames.
    SHAP plots are made in svensemble_analyse (figures.shap_summary).

    Args:
        use_gpu: Use GPUTreeExplainer (requires CUDA).
        max_samples: Subsample the feature matrix to at most this many rows.
        top_n: Number of top features in the summary CSV.
    """
    model_name = model
    model = sv_io.read_model(model_pkl)
    fm = sv_io.read_feature_matrix(feature_matrix)
    fm = fm.drop(columns=['ref_match'], errors='ignore')

    resolved = (
        sv_config.resolve_model_config(config=model_config) if model_config
        else sv_config.resolve_model_config(name=model_name)
    )
    fillna = resolved['training_params'].get('fillna', 0)
    X = pd.get_dummies(fm)
    if fillna is not None:
        X = X.fillna(fillna)

    X = align_features(model, X, verbose=verbose)

    if verbose:
        print(f"SHAP for sample={sample} model={sv_config.model_id(resolved)} training_set={training_set}")
        print(f"Feature matrix shape: {X.shape}")

    shap_values = compute_shap_values(
        model, X, use_gpu=use_gpu, max_samples=max_samples, verbose=verbose,
    )

    output_dir = Path(output_dir)
    summary = summarise_shap_values(shap_values, top_n=top_n)
    summary_csv_path = output_dir / sv_io.SHAP_SUMMARY
    with sv_io.atomic_path(summary_csv_path) as tmp:
        summary.to_csv(tmp, index=False)
    print(f"Saved SHAP summary (top {len(summary)}): {summary_csv_path}")
    print(summary.to_string(index=False))

    shap_pkl_path = output_dir / sv_io.SHAP_VALUES
    with sv_io.atomic_path(shap_pkl_path) as tmp, open(tmp, 'wb') as f:
        pickle.dump(shap_values, f)
    print(f"Saved SHAP values: {shap_pkl_path}")

    return shap_pkl_path
