"""
PU models for SV call probability estimation.
"""
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sklearn.model_selection as skm
from sklearn.ensemble import RandomForestClassifier as RF
from xgboost import XGBClassifier as XGB

from . import matched_breakpoints as mbps


# ---------------------------------------------------------------------------
# Duplessis PU weight estimation (shared across classifiers)
# ---------------------------------------------------------------------------

# Floor for the estimated label frequency c = p(s=1|y=1) (Elkan & Noto 2008,
# Sec. 2). Guards against a degenerate weight blow-up (clpr = label_prob /
# label_freq) on training sets with very few true positives, where even the
# e1 estimator can land arbitrarily close to zero.
_LABEL_FREQ_FLOOR = 1e-3


def _estimate_duplessis_weights(
    training_set_train_test_split: dict,
    model_kwargs: dict,
) -> dict[int, float]:
    """
    Estimate PU class weights using the Duplessis method.

    Trains a small "non-traditional" Random Forest on the train split, then
    estimates the label frequency c = p(s=1|y=1) on the validation split
    using the e1 estimator of Elkan & Noto (2008, Sec. 2): the average
    predicted probability of the non-traditional classifier over the
    labeled (true-positive) validation examples. Elkan & Noto recommend e1
    over their alternative e2/e3 estimators specifically for its lower
    variance. Class weights are then derived from the label frequency
    estimate (Duplessis method).

    Args:
        training_set_train_test_split: Dict with keys
            X, y, X_train, y_train, X_val, y_val.
        model_kwargs: Keyword arguments forwarded to the non-traditional RF
            (only n_estimators and max_depth are used).

    Returns:
        Weight dict {0: w_neg, 1: w_pos}.
    """
    X = training_set_train_test_split['X']
    y = training_set_train_test_split['y']
    X_train = training_set_train_test_split['X_train']
    y_train = training_set_train_test_split['y_train']
    X_val = training_set_train_test_split['X_val']
    y_val = training_set_train_test_split['y_val']

    if y_train.sum() == 0 or y_val.sum() == 0:
        print("No positive labels in the training or validation set, skipping")
        return {
            0: y.sum() / len(y),
            1: 0,
        }

    # The non-traditional RF used for label-frequency estimation always
    # receives 0-filled data (no NaN) regardless of the final classifier.
    X_train_filled = X_train.fillna(0) if X_train.isna().any().any() else X_train
    X_val_filled = X_val.fillna(0) if X_val.isna().any().any() else X_val

    nontrad_kwargs: dict[str, Any] = {}
    if 'n_estimators' in model_kwargs:
        nontrad_kwargs['n_estimators'] = min(model_kwargs['n_estimators'], 70)
    if 'max_depth' in model_kwargs:
        nontrad_kwargs['max_depth'] = model_kwargs['max_depth']

    rf_nontrad = RF(
        max_features=int(np.sqrt(X.shape[1])),
        random_state=0,
        **nontrad_kwargs,
    )
    rf_nontrad.fit(X_train_filled, y_train)

    # e1 = (1/n) * sum_{x in P} g(x), P = labeled (true-positive) validation
    # examples, g(x) = predicted probability of the non-traditional RF.
    y_proba_nontrad = rf_nontrad.predict_proba(X_val_filled)[:, 1]
    label_freq = y_proba_nontrad[y_val == 1].mean()
    label_prob = y.sum() / len(y)

    if label_freq < _LABEL_FREQ_FLOOR:
        print(
            f"label_freq estimate ({label_freq:.2e}) is below the floor "
            f"({_LABEL_FREQ_FLOOR:.2e}): the non-traditional RF assigned "
            "near-zero probability to the true positives in the validation "
            "set. Flooring label_freq to avoid an unstable weight estimate."
        )
        label_freq = _LABEL_FREQ_FLOOR

    clpr = label_prob / label_freq

    return {
        0: 1 / (1 - label_prob),
        1: 2 * clpr / label_prob,
    }


# ---------------------------------------------------------------------------
# Per-split training: Random Forest
# ---------------------------------------------------------------------------

def train_unbiased_rf_for_split(
    training_set_train_test_split: dict,
    **model_kwargs,
) -> RF:
    """
    Train an unbiased Random Forest for a single split using the Duplessis method.
    """
    _model_kwargs: dict[str, Any] = {
        'n_estimators': 500,
        'max_depth': None,
    }
    _model_kwargs.update(model_kwargs)

    weight_dict = _estimate_duplessis_weights(
        training_set_train_test_split, _model_kwargs,
    )

    X_train = training_set_train_test_split['X_train']
    y_train = training_set_train_test_split['y_train']
    X = training_set_train_test_split['X']

    duplessis_kwargs = _model_kwargs.copy()
    if 'n_jobs' not in duplessis_kwargs:
        try:
            n_jobs = len(os.sched_getaffinity(0))
        except AttributeError:
            n_jobs = os.cpu_count() or 1
        duplessis_kwargs['n_jobs'] = n_jobs

    rf = RF(
        max_features=int(np.sqrt(X.shape[1])),
        random_state=0,
        class_weight=weight_dict,
        **duplessis_kwargs,
    )
    rf.fit(X_train, y_train)
    return rf


# ---------------------------------------------------------------------------
# Per-split training: XGBoost
# ---------------------------------------------------------------------------

def train_unbiased_xgb_for_split(
    training_set_train_test_split: dict,
    **model_kwargs,
) -> XGB:
    """
    Train an unbiased XGBoost classifier for a single split using the Duplessis method.
    """
    _model_kwargs: dict[str, Any] = {
        'n_estimators': 500,
        'max_depth': 6,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
    }
    _model_kwargs.update(model_kwargs)

    weight_dict = _estimate_duplessis_weights(
        training_set_train_test_split, _model_kwargs,
    )

    X_train = training_set_train_test_split['X_train']
    y_train = training_set_train_test_split['y_train']

    if 'n_jobs' not in _model_kwargs:
        try:
            n_jobs = len(os.sched_getaffinity(0))
        except AttributeError:
            n_jobs = os.cpu_count() or 1
        _model_kwargs['n_jobs'] = n_jobs

    xgb = XGB(
        random_state=0,
        **_model_kwargs,
    )

    sample_weights = np.where(
        y_train == 1, weight_dict[1], weight_dict[0],
    )
    xgb.fit(X_train, y_train, sample_weight=sample_weights)
    return xgb


# ---------------------------------------------------------------------------
# Workflow-level training helpers
# ---------------------------------------------------------------------------

def _get_fillna(workflow_config: dict) -> int | float | None:
    """Read the fillna strategy from workflow config, defaulting to 0."""
    return workflow_config['model']['training_params'].get('fillna', 0)


def train_unbiased_rf(
    joint_bp_df: pd.DataFrame,
    workflow_config: dict,
    ground_truth_config: dict,
    training_set_name: str,
    verbose: bool = False,
) -> tuple[pd.DataFrame, RF]:
    """
    Train an unbiased Random Forest for a single training set using the Duplessis method.
    """
    training_set_df = mbps.preprocess_matchedbps_for_pu(
        joint_bp_df,
        workflow_config,
        ground_truth_config['callsets'][list(ground_truth_config['callsets'].keys())[0]]['source'],
        training_set_name,
        verbose=verbose,
    )
    del joint_bp_df

    fillna = _get_fillna(workflow_config)
    X, y = mbps.get_X_y_from_training_df(
        training_set_df, match_col='ref_match', fillna=fillna,
    )
    X_train, X_val, y_train, y_val = skm.train_test_split(
        X, y, train_size=0.8, random_state=0, stratify=y,
    )
    training_set_splits = {
        'X': X,
        'X_train': X_train,
        'X_val': X_val,
        'y': y,
        'y_train': y_train,
        'y_val': y_val,
        'X_idx': training_set_df.index,
    }
    del training_set_df

    if verbose:
        print(f"Training PU model for training set: {training_set_name}")
    model_kwargs = workflow_config['model']['training_params']['model_kwargs']
    rf_fit = train_unbiased_rf_for_split(training_set_splits, **model_kwargs)

    return X, rf_fit


def train_unbiased_nosvlen(
    joint_bp_df: pd.DataFrame,
    workflow_config: dict,
    ground_truth_config: dict,
    training_set_name: str,
    verbose: bool = False,
) -> tuple[pd.DataFrame, RF]:
    """
    Train an unbiased XGBoost model for a single training set, with all SVLEN
    columns removed from the feature matrix prior to training.
    """
    svlen_cols = [col for col in joint_bp_df.columns if 'INFO_SVLEN' in col]
    joint_bp_df_nosvlen = joint_bp_df.drop(columns=svlen_cols)

    return train_unbiased_rf(
        joint_bp_df_nosvlen,
        workflow_config,
        ground_truth_config,
        training_set_name,
        verbose=verbose,
    )


def train_unbiased_concat_rf(
    matched_bps_sources: dict[str, pd.DataFrame | Path],
    workflow_config: dict,
    ground_truth_config: dict,
    training_set_name: str,
    verbose: bool = False,
) -> tuple[pd.DataFrame, RF]:
    """
    Train an unbiased Random Forest on a training set pooled from one or more samples.

    Args:
        matched_bps_sources: Mapping of sample_name -> matched_bps DataFrame or
            path to a matched_bps CSV. Every source is treated uniformly; there
            is no distinguished "primary" sample. Pass a single-entry dict for
            ordinary held-in training, or multiple entries to pool several
            samples (concat/cross-training). All sources must share the same
            training_set/callers as workflow_config.
    """
    _gt_name = ground_truth_config['callsets'][list(ground_truth_config['callsets'].keys())[0]]['source']
    training_set_dfs = []

    for sample_name, source in matched_bps_sources.items():
        source_df = source if isinstance(source, pd.DataFrame) else pd.read_csv(source, low_memory=False)
        sample_training_set_df = mbps.preprocess_matchedbps_for_pu(
            source_df,
            workflow_config,
            _gt_name,
            training_set_name,
            verbose=verbose,
        )
        del source_df
        training_set_dfs.append(sample_training_set_df)

    training_set_df = pd.concat(training_set_dfs, ignore_index=True)
    del training_set_dfs

    fillna = _get_fillna(workflow_config)
    X, y = mbps.get_X_y_from_training_df(
        training_set_df, match_col='ref_match', fillna=fillna,
    )
    X_train, X_val, y_train, y_val = skm.train_test_split(
        X, y, train_size=0.8, random_state=0, stratify=y,
    )
    training_set_splits = {
        'X': X,
        'X_train': X_train,
        'X_val': X_val,
        'y': y,
        'y_train': y_train,
        'y_val': y_val,
        'X_idx': training_set_df.index,
    }
    del training_set_df

    if verbose:
        print(f"Training PU model for training set: {training_set_name}")
    model_kwargs = workflow_config['model']['training_params']['model_kwargs']
    rf_fit = train_unbiased_rf_for_split(training_set_splits, **model_kwargs)

    return X, rf_fit


def train_unbiased_xgb(
    joint_bp_df: pd.DataFrame,
    workflow_config: dict,
    ground_truth_config: dict,
    training_set_name: str,
    verbose: bool = False,
) -> tuple[pd.DataFrame, XGB]:
    """
    Train an unbiased XGBoost model for a single training set using the Duplessis method.
    """
    training_set_df = mbps.preprocess_matchedbps_for_pu(
        joint_bp_df,
        workflow_config,
        ground_truth_config['callsets'][list(ground_truth_config['callsets'].keys())[0]]['source'],
        training_set_name,
        verbose=verbose,
    )
    del joint_bp_df

    fillna = _get_fillna(workflow_config)
    X, y = mbps.get_X_y_from_training_df(
        training_set_df, match_col='ref_match', fillna=fillna,
    )
    X_train, X_val, y_train, y_val = skm.train_test_split(
        X, y, train_size=0.8, random_state=0, stratify=y,
    )
    training_set_splits = {
        'X': X,
        'X_train': X_train,
        'X_val': X_val,
        'y': y,
        'y_train': y_train,
        'y_val': y_val,
        'X_idx': training_set_df.index,
    }
    del training_set_df

    if verbose:
        print(f"Training XGBoost PU model for training set: {training_set_name}")
    model_kwargs = workflow_config['model']['training_params']['model_kwargs']
    xgb_fit = train_unbiased_xgb_for_split(training_set_splits, **model_kwargs)

    return X, xgb_fit


def train_unbiased_xgb_concat(
    matched_bps_sources: dict[str, pd.DataFrame | Path],
    workflow_config: dict,
    ground_truth_config: dict,
    training_set_name: str,
    verbose: bool = False,
) -> tuple[pd.DataFrame, XGB]:
    """
    Train an unbiased XGBoost model on a training set pooled from one or more samples.

    Args:
        matched_bps_sources: Mapping of sample_name -> matched_bps DataFrame or
            path to a matched_bps CSV. Every source is treated uniformly; there
            is no distinguished "primary" sample. Pass a single-entry dict for
            ordinary held-in training, or multiple entries to pool several
            samples (concat/cross-training). All sources must share the same
            training_set/callers as workflow_config.
    """
    _gt_name = ground_truth_config['callsets'][list(ground_truth_config['callsets'].keys())[0]]['source']
    training_set_dfs = []

    for sample_name, source in matched_bps_sources.items():
        source_df = source if isinstance(source, pd.DataFrame) else pd.read_csv(source, low_memory=False)
        sample_training_set_df = mbps.preprocess_matchedbps_for_pu(
            source_df,
            workflow_config,
            _gt_name,
            training_set_name,
            verbose=verbose,
        )
        del source_df
        training_set_dfs.append(sample_training_set_df)

    training_set_df = pd.concat(training_set_dfs, ignore_index=True)
    del training_set_dfs

    fillna = _get_fillna(workflow_config)
    X, y = mbps.get_X_y_from_training_df(
        training_set_df, match_col='ref_match', fillna=fillna,
    )
    X_train, X_val, y_train, y_val = skm.train_test_split(
        X, y, train_size=0.8, random_state=0, stratify=y,
    )
    training_set_splits = {
        'X': X,
        'X_train': X_train,
        'X_val': X_val,
        'y': y,
        'y_train': y_train,
        'y_val': y_val,
        'X_idx': training_set_df.index,
    }
    del training_set_df

    if verbose:
        print(f"Training XGBoost PU model for training set: {training_set_name}")
    model_kwargs = workflow_config['model']['training_params']['model_kwargs']
    xgb_fit = train_unbiased_xgb_for_split(training_set_splits, **model_kwargs)

    return X, xgb_fit


def train_unbiased_xgb_nosvlen(
    joint_bp_df: pd.DataFrame,
    workflow_config: dict,
    ground_truth_config: dict,
    training_set_name: str,
    verbose: bool = False,
) -> tuple[pd.DataFrame, XGB]:
    """
    Train an unbiased XGBoost model for a single training set, with all SVLEN
    columns removed from the feature matrix prior to training.
    """
    svlen_cols = [col for col in joint_bp_df.columns if 'INFO_SVLEN' in col]
    joint_bp_df_nosvlen = joint_bp_df.drop(columns=svlen_cols)

    return train_unbiased_xgb(
        joint_bp_df_nosvlen,
        workflow_config,
        ground_truth_config,
        training_set_name,
        verbose=verbose,
    )

# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

model_dispatch = {
    'unbiased': train_unbiased_rf,
    'unbiased_concat': train_unbiased_concat_rf,
    'unbiased_nosvlen': train_unbiased_nosvlen,
    'unbiased_xgb': train_unbiased_xgb,
    'unbiased_xgb_concat': train_unbiased_xgb_concat,
    'unbiased_xgb_nosvlen': train_unbiased_xgb_nosvlen,
}


def train_model(
    joint_bp_df: pd.DataFrame | None,
    workflow_config: dict,
    ground_truth_config: dict,
    training_set_name: str,
    verbose: bool = False,
    matched_bps_sources: dict[str, pd.DataFrame | Path] | None = None,
) -> tuple[pd.DataFrame, RF | XGB]:
    """
    Train a model following workflow_config using joint_bp_df or matched_bps_sources.

    Args:
        joint_bp_df: DataFrame containing matched breakpoints for the training
            set. Required for ordinary (non-pooled) models; ignored for
            pooled (concat/cross) models.
        workflow_config: Dictionary containing the workflow configuration.
        ground_truth_config: Dictionary containing the ground truth configuration.
        training_set_name: Name of the training set.
        verbose: If True, print verbose output.
        matched_bps_sources: Mapping of sample_name -> matched_bps DataFrame or
            path, for pooled (concat/cross) models (see model_configs
            entries with training_params['concat'] = True). Required for
            those models; ignored otherwise.
    Returns:
        Tuple of (feature matrix, trained model).
    """
    training_params = workflow_config['model']['training_params']
    model_function = model_dispatch[training_params['function']]

    if training_params['concat']:
        if matched_bps_sources is None:
            raise ValueError(
                f"Model '{workflow_config['model']['name']}' is a concat/pooled model but no "
                f"matched_bps_sources were provided (pass --train_samples/--train_matched_bps)."
            )
        return model_function(
            matched_bps_sources, workflow_config, ground_truth_config,
            training_set_name, verbose,
        )

    if joint_bp_df is None:
        raise ValueError(f"Model '{workflow_config['model']['name']}' requires joint_bp_df.")
    return model_function(
        joint_bp_df, workflow_config, ground_truth_config,
        training_set_name, verbose,
    )
