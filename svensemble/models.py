"""
PU models for SV call probability estimation.
"""
import os
from typing import Any

import numpy as np
import pandas as pd
import sklearn.model_selection as skm
from sklearn.ensemble import RandomForestClassifier as RF
from xgboost import XGBClassifier as XGB

from . import matched_breakpoints as mbps
from . import utils as sv_utils


# ---------------------------------------------------------------------------
# Duplessis PU weight estimation (shared across classifiers)
# ---------------------------------------------------------------------------

def _estimate_duplessis_weights(
    training_set_train_test_split: dict,
    model_kwargs: dict,
) -> dict[int, float] | None:
    """
    Estimate PU class weights using the Duplessis method.

    Trains a small "non-traditional" Random Forest on the train split,
    predicts on the validation split, and derives class weights from the
    label frequency estimate.

    Args:
        training_set_train_test_split: Dict with keys
            ``X``, ``y``, ``X_train``, ``y_train``, ``X_val``, ``y_val``.
        model_kwargs: Keyword arguments forwarded to the non-traditional RF
            (only ``n_estimators`` and ``max_depth`` are used).

    Returns:
        Weight dict ``{0: w_neg, 1: w_pos}`` or ``None`` when estimation
        fails (no positives or zero label frequency).
    """
    X = training_set_train_test_split['X']
    y = training_set_train_test_split['y']
    X_train = training_set_train_test_split['X_train']
    y_train = training_set_train_test_split['y_train']
    X_val = training_set_train_test_split['X_val']
    y_val = training_set_train_test_split['y_val']

    if y_train.sum() == 0 or y_val.sum() == 0:
        print("No positive labels in the training or validation set, skipping")
        return None

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
    y_nontrad = rf_nontrad.predict(X_val_filled)

    label_freq = (y_nontrad & y_val).sum() / y_val.sum()
    label_prob = y.sum() / len(y)

    if label_freq == 0:
        print(
            "label_freq is 0: the non-traditional RF predicted no true positives "
            "in the validation set. This means no matches to ground truth were "
            "found in this dataset. Skipping this split."
        )
        return None

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
) -> RF | None:
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
    if weight_dict is None:
        return None

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
) -> XGB | None:
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
    if weight_dict is None:
        return None

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
) -> tuple[pd.DataFrame, RF | None]:
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


def train_unbiased_concat_rf(
    joint_bp_df: pd.DataFrame,
    workflow_config: dict,
    ground_truth_config: dict,
    training_set_name: str,
    verbose: bool = False,
) -> tuple[pd.DataFrame, RF | None]:
    """
    Train an unbiased Random Forest on a concatenated training set from multiple samples.
    """
    _gt_name = ground_truth_config['callsets'][list(ground_truth_config['callsets'].keys())[0]]['source']
    training_set_dfs = []
    for sample in workflow_config['model']['training_params']['function_params']['samples']:
        if sample == workflow_config['sample_name']:
            sample_workflow_config = workflow_config
            sample_joint_bp_df = joint_bp_df
        else:
            sample_workflow_config = sv_utils.create_config_dict(
                sample,
                model_name=workflow_config['model']['name'],
                training_set=training_set_name,
                ground_truth_name=_gt_name,
            )[sample]
            sample_joint_bp_df = pd.read_csv(
                sample_workflow_config['model']['matched_bps_path'],
                low_memory=False,
            )
        sample_training_set_df = mbps.preprocess_matchedbps_for_pu(
            sample_joint_bp_df,
            sample_workflow_config,
            ground_truth_config['callsets'][list(ground_truth_config['callsets'].keys())[0]]['source'],
            training_set_name,
            verbose=verbose,
        )
        del sample_joint_bp_df
        training_set_dfs.append(sample_training_set_df)
    training_set_df = pd.concat(training_set_dfs, ignore_index=True)
    del training_set_dfs

    fillna = _get_fillna(workflow_config)
    X, y = mbps.get_X_y_from_training_df(
        training_set_df, match_col='ref_match', fillna=fillna,
    )
    X_train, X_val, y_train, y_val = skm.train_test_split(
        X, y, train_size=0.8, random_state=0,
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
) -> tuple[pd.DataFrame, XGB | None]:
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
    joint_bp_df: pd.DataFrame,
    workflow_config: dict,
    ground_truth_config: dict,
    training_set_name: str,
    verbose: bool = False,
) -> tuple[pd.DataFrame, XGB | None]:
    """
    Train an unbiased XGBoost model with cross-training on one or multiple samples.
    """
    _gt_name = ground_truth_config['callsets'][list(ground_truth_config['callsets'].keys())[0]]['source']
    training_set_dfs = []
    for sample in workflow_config['model']['training_params']['function_params']['samples']:
        if sample == workflow_config['sample_name']:
            sample_workflow_config = workflow_config
            sample_joint_bp_df = joint_bp_df
        else:
            sample_workflow_config = sv_utils.create_config_dict(
                sample,
                model_name=workflow_config['model']['name'],
                training_set=training_set_name,
                ground_truth_name=_gt_name,
            )[sample]
            sample_joint_bp_df = pd.read_csv(
                sample_workflow_config['model']['matched_bps_path'],
                low_memory=False,
            )
        sample_training_set_df = mbps.preprocess_matchedbps_for_pu(
            sample_joint_bp_df,
            sample_workflow_config,
            ground_truth_config['callsets'][list(ground_truth_config['callsets'].keys())[0]]['source'],
            training_set_name,
            verbose=verbose,
        )
        del sample_joint_bp_df
        training_set_dfs.append(sample_training_set_df)
    training_set_df = pd.concat(training_set_dfs, ignore_index=True)
    del training_set_dfs

    fillna = _get_fillna(workflow_config)
    X, y = mbps.get_X_y_from_training_df(
        training_set_df, match_col='ref_match', fillna=fillna,
    )
    X_train, X_val, y_train, y_val = skm.train_test_split(
        X, y, train_size=0.8, random_state=0,
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


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

model_dispatch = {
    'unbiased': train_unbiased_rf,
    'unbiased_concat': train_unbiased_concat_rf,
    'unbiased_xgb': train_unbiased_xgb,
    'unbiased_xgb_concat': train_unbiased_xgb_concat,
}


def train_model(
    joint_bp_df: pd.DataFrame,
    workflow_config: dict,
    ground_truth_config: dict,
    training_set_name: str,
    verbose: bool = False,
) -> tuple[pd.DataFrame, RF | XGB | None]:
    """
    Train a model following workflow_config using joint_bp_df.

    Args:
        joint_bp_df: DataFrame containing matched breakpoints for the training set.
        workflow_config: Dictionary containing the workflow configuration.
        ground_truth_config: Dictionary containing the ground truth configuration.
        training_set_name: Name of the training set.
        verbose: If True, print verbose output.
    Returns:
        Tuple of (feature matrix, trained model).
    """
    model_function = model_dispatch[workflow_config['model']['training_params']['function']]
    return model_function(
        joint_bp_df, workflow_config, ground_truth_config,
        training_set_name, verbose,
    )
