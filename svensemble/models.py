"""
PU models for SV call probability estimation.
"""
import inspect
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sklearn.model_selection as skm
from sklearn.ensemble import RandomForestClassifier as RF
from xgboost import XGBClassifier as XGB

# Imports for custom classifiers
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import validate_data, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import euclidean_distances

from . import matched_breakpoints as mbps


# ---------------------------------------------------------------------------
# Duplessis PU weight estimation (shared across classifiers)
# ---------------------------------------------------------------------------

# Floor for the estimated label frequency c = p(s=1|y=1) (Elkan & Noto 2008,
# Sec. 2). Guards against a degenerate weight blow-up (clpr = label_prob /
# label_freq) on training sets with very few true positives, where even the
# e1 estimator can land arbitrarily close to zero.
_LABEL_FREQ_FLOOR = 1e-3
_VALID_MIN_MATCHES_MODES = ['majority', 'all']


def _build_nontrad_rf(model_kwargs: dict, **kwargs) -> RF:
    """Instantiate a non-traditional RF for label-frequency estimation."""
    sig_params = set(inspect.signature(RF).parameters)
    nontrad_kwargs: dict[str, Any] = {
        k: v for k, v in model_kwargs.items() if k in sig_params
    }
    if 'n_estimators' in nontrad_kwargs:
        nontrad_kwargs['n_estimators'] = min(nontrad_kwargs['n_estimators'], 70)
    if 'n_features' not in kwargs:
        raise ValueError("n_features not supplied to nontrad rf model builder.")
    n_features = kwargs.get('n_features')
    max_features = int(np.sqrt(n_features)) if n_features is not None else 'sqrt'
    return RF(
        max_features=max_features,
        random_state=0,
        **nontrad_kwargs,
    )


def _build_nontrad_xgb(model_kwargs: dict, **kwargs) -> XGB:
    """Instantiate a non-traditional XGBoost for label-frequency estimation."""
    sig_params = set(inspect.signature(XGB).parameters)
    nontrad_kwargs: dict[str, Any] = {
        k: v for k, v in model_kwargs.items() if k in sig_params
    }
    return XGB(
        random_state=0,
        **nontrad_kwargs,
    )


_nontrad_dispatch = {
    'rf': _build_nontrad_rf,
    'xgb': _build_nontrad_xgb,
}


class SVEnsembleConsensusClassifier(ClassifierMixin, BaseEstimator):
    """
    Simple consensus classifier mimicking SURVIVOR's functionality but capable of utilizing SVEnsemble2's matching machinery.
    """
    def __init__(self, min_matches_mode: int | str = "majority"):
        self.min_matches_mode = min_matches_mode

    def fit(self, X, y):
        # Check that X and y have correct shape, set n_features_in_, etc.
        X, y = validate_data(self, X, y, ensure_all_finite='allow-nan')
        # Store the classes seen during fit
        self.classes_ = unique_labels(y)

        if isinstance(self.min_matches_mode, int):
            self.min_matches_ = self.min_matches_mode
        elif self.min_matches_mode == 'majority':
            self.min_matches_ = np.ceil(X.shape[1] * 0.5)
        elif self.min_matches_mode == 'all':
            self.min_matches_ = X.shape[1]
        else:
            raise ValueError(f"Invalid min_matches mode: {self.min_matches_mode}")

        return self

    def predict_proba(self, X):
        """
        SURVIVOR-analog QUAL surrogate for ranking, packed into the sklearn
        ``(n_samples, 2)`` ``predict_proba`` contract.

        X is expected to be an ``(n_samples, m)`` matrix of per-caller quality
        scores, with NaN encoding "this caller did not call this row". Each
        column is min-max normalized across the input X (so, per caller, the
        highest QUAL seen in X maps to 1 and the lowest to 0), then the
        per-row score is the NaN-aware max across callers (mimicking
        SURVIVOR's merged-QUAL behaviour). Rows failing the ``min_matches_``
        consensus rule are forced to 0 so downstream ROC/PRC sweeps treat
        them as never called.

        Callers whose QUAL is constant across X (min == max) carry no
        ranking information and are excluded from the row-wise aggregation
        by setting their normalized column to NaN. If a row's only voting
        callers all fall in such degenerate columns, the row's aggregated
        score falls through to 1.0 (max confidence): the consensus mask
        already separates it from consensus-failing rows, and 1.0 correctly
        represents "these callers committed to the call, no QUAL
        discriminator available".

        Returns
        -------
        proba : ndarray of shape (n_samples, 2)
            ``proba[:, 1]`` is the SURVIVOR QUAL surrogate; ``proba[:, 0]``
            is ``1 - proba[:, 1]`` to satisfy the sklearn probability
            contract.
        """
        check_is_fitted(self)
        X = validate_data(self, X, reset=False, ensure_all_finite='allow-nan')

        X_maxs = np.nanmax(X, axis=0)
        X_mins = np.nanmin(X, axis=0)
        denom = X_maxs - X_mins
        degenerate_cols = denom == 0
        with np.errstate(invalid='ignore', divide='ignore'):
            X_norm = (X - X_mins) / denom
        X_norm[:, degenerate_cols] = np.nan

        votes = np.sum(~np.isnan(X), axis=1)
        p_pos = np.full(X.shape[0], 1.0, dtype=float)
        row_has_any = np.any(~np.isnan(X_norm), axis=1)
        p_pos[row_has_any] = np.nanmax(X_norm[row_has_any], axis=1)
        p_pos = np.where(votes >= self.min_matches_, p_pos, 0.0)

        return np.column_stack([1.0 - p_pos, p_pos])

    def predict(self, X):
        check_is_fitted(self)
        X = validate_data(self, X, reset=False, ensure_all_finite='allow-nan')
        pos = (np.sum(~np.isnan(X), axis=1) >= self.min_matches_).astype(int)
        return self.classes_[pos]


def _estimate_duplessis_weights(
    training_set_train_test_split: dict,
    model_kwargs: dict,
    clpr: float | None = None,
    estimator: str = 'rf',
) -> tuple[dict[str, float], dict[int, float]]:
    """
    Estimate PU class weights using the Duplessis method.

    Trains a small non-traditional classifier on the train split, then
    estimates the label frequency c = p(s=1|y=1) on the validation split
    using the e1 estimator of Elkan & Noto (2008, Sec. 2): the average
    predicted probability of the non-traditional classifier over the
    labeled (true-positive) validation examples. Elkan & Noto recommend e1
    over their alternative e2/e3 estimators specifically for its lower
    variance. Class weights are then derived from the label frequency
    estimate (Duplessis method).

    Optionally accepts a pre-computed clpr value. If provided, skips the estimation step
    and returns weight estimates based on the provided class prior value only.

    Args:
        training_set_train_test_split: Dict with keys
            X, y, X_train, y_train, X_val, y_val.
        model_kwargs: Keyword arguments forwarded to the non-traditional
            classifier (filtered by signature of the chosen estimator).
        clpr: Optional input class prior value.
        estimator: Name of the non-traditional estimator to use for
            label-frequency estimation. Must be a key in _nontrad_dispatch.

    Returns:
        Tuple (Dict with estimation parameters {param_name: param_value},
        Weight dict {0: w_neg, 1: w_pos}).
    """
    estimation_params: dict[str, float] = {}

    X = training_set_train_test_split['X']
    y = training_set_train_test_split['y']
    X_train = training_set_train_test_split['X_train']
    y_train = training_set_train_test_split['y_train']
    X_val = training_set_train_test_split['X_val']
    y_val = training_set_train_test_split['y_val']

    label_prob = y.sum() / len(y)
    estimation_params['label_prob'] = label_prob

    if y_train.sum() == 0 or y_val.sum() == 0:
        print("No positive labels in the training or validation set, skipping")
        estimation_params['clpr'] = 0
        return estimation_params, {0: label_prob, 1: 0}

    if clpr is not None:
        estimation_params['clpr'] = clpr
        return estimation_params, {
            0: 1 / (1 - label_prob),
            1: 2 * clpr / label_prob,
        }

    if estimator not in _nontrad_dispatch:
        raise ValueError(
            f"Unknown non-traditional estimator '{estimator}'. "
            f"Supported: {sorted(_nontrad_dispatch)}"
        )
    builder = _nontrad_dispatch[estimator]
    nontrad_clf = builder(model_kwargs, n_features=X.shape[1])

    # The non-traditional classifier used for label-frequency estimation
    # always receives 0-filled data (no NaN).
    X_train_filled = X_train.fillna(0) if X_train.isna().any().any() else X_train
    X_val_filled = X_val.fillna(0) if X_val.isna().any().any() else X_val

    nontrad_clf.fit(X_train_filled, y_train)

    # e1 = (1/n) * sum_{x in P} g(x), P = labeled (true-positive) validation
    # examples, g(x) = predicted probability of the non-traditional classifier.
    y_proba_nontrad = nontrad_clf.predict_proba(X_val_filled)[:, 1]
    label_freq = y_proba_nontrad[y_val == 1].mean()

    if label_freq < _LABEL_FREQ_FLOOR:
        print(
            f"label_freq estimate ({label_freq:.2e}) is below the floor "
            f"({_LABEL_FREQ_FLOOR:.2e}): the non-traditional classifier "
            "assigned near-zero probability to the true positives in the "
            "validation set. Flooring label_freq to avoid an unstable "
            "weight estimate."
        )
        label_freq = _LABEL_FREQ_FLOOR

    clpr = label_prob / label_freq
    estimation_params['clpr'] = clpr
    estimation_params['label_freq'] = label_freq

    print(f"PU estimates: clpr {clpr:.2e}, label_prob {label_prob:.2e}, label_freq {label_freq:.2e}")

    return estimation_params, {0: 1 / (1 - label_prob), 1: 2 * clpr / label_prob}


def _assign_estimation_params(model: RF | XGB, estimation_params: dict[str, float]) -> None:
    if 'clpr' in estimation_params:
        model.clpr_ = estimation_params['clpr']
    if 'label_prob' in estimation_params:
        model.label_prob_ = estimation_params['label_prob']
    if 'label_freq' in estimation_params:
        model.label_freq_ = estimation_params['label_freq']


# ---------------------------------------------------------------------------
# Per-split training: Random Forest
# ---------------------------------------------------------------------------

def train_unbiased_rf_for_split(
    training_set_train_test_split: dict,
    clpr: float | None = None,
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

    duplessis_kwargs = _model_kwargs.copy()
    if 'n_jobs' not in duplessis_kwargs:
        try:
            n_jobs = len(os.sched_getaffinity(0))
        except AttributeError:
            n_jobs = os.cpu_count() or 1
        duplessis_kwargs['n_jobs'] = n_jobs

    # The label-frequency forest gets the same n_jobs; with random_state fixed,
    # its trees do not depend on it
    estimation_params, weight_dict = _estimate_duplessis_weights(
        training_set_train_test_split,
        duplessis_kwargs,
        clpr=clpr,
        estimator='rf',
    )

    X_train = training_set_train_test_split['X_train']
    y_train = training_set_train_test_split['y_train']
    X = training_set_train_test_split['X']

    rf = RF(
        max_features=int(np.sqrt(X.shape[1])),
        random_state=0,
        class_weight=weight_dict,
        **duplessis_kwargs,
    )
    rf.fit(X_train, y_train)

    _assign_estimation_params(rf, estimation_params)

    return rf


# ---------------------------------------------------------------------------
# Per-split training: XGBoost
# ---------------------------------------------------------------------------

def train_unbiased_xgb_for_split(
    training_set_train_test_split: dict,
    clpr: float | None = None,
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

    estimation_params, weight_dict = _estimate_duplessis_weights(
        training_set_train_test_split,
        _model_kwargs,
        clpr=clpr,
        estimator='xgb',
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

    _assign_estimation_params(xgb, estimation_params)

    return xgb


# ---------------------------------------------------------------------------
# Workflow-level training helpers
# ---------------------------------------------------------------------------

def _get_fillna(workflow_config: dict) -> int | float | None:
    """Read the fillna strategy from workflow config, defaulting to 0."""
    return workflow_config['model']['training_params'].get('fillna', 0)


def _get_ground_truth_name(workflow_config: dict) -> str:
    """Extract the ground truth name from workflow_config['ground_truth']."""
    return workflow_config['ground_truth']['name']


def train_consensus_classifier(
    joint_bp_df: pd.DataFrame,
    workflow_config: dict,
    training_set_name: str,
    verbose: bool = False,
) -> tuple[pd.DataFrame, SVEnsembleConsensusClassifier]:
    """
    Adapter for consensus classifier so it can be included into the pipeline despite not requiring actual `training`
    """
    ground_truth_name = _get_ground_truth_name(workflow_config)
    training_set_df = mbps.preprocess_matchedbps_for_pu(
        joint_bp_df,
        workflow_config,
        ground_truth_name,
        training_set_name,
        verbose=verbose,
    )
    del joint_bp_df


    X, y = mbps.get_X_y_from_training_df(
        training_set_df, match_col='ref_match', fillna=None,
    )
    del training_set_df

    # Retain only the qual metric columns
    cols_to_retain = []
    for callset_name in workflow_config['callsets']:
        qual_metric = workflow_config['callsets'][callset_name]['params']['qual_metric']
        cols_to_retain.append(f'{callset_name}_{qual_metric}')
    X = X[cols_to_retain]

    model_kwargs = workflow_config['model']['training_params']['model_kwargs']
    consensus_classifier = SVEnsembleConsensusClassifier(**model_kwargs)
    consensus_classifier.fit(X, y)

    return X, consensus_classifier


def train_unbiased_rf(
    joint_bp_df: pd.DataFrame,
    workflow_config: dict,
    training_set_name: str,
    verbose: bool = False,
) -> tuple[pd.DataFrame, RF]:
    """
    Train an unbiased Random Forest for a single training set using the Duplessis method.
    """
    ground_truth_name = _get_ground_truth_name(workflow_config)
    training_set_df = mbps.preprocess_matchedbps_for_pu(
        joint_bp_df,
        workflow_config,
        ground_truth_name,
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
    training_params = workflow_config['model']['training_params']
    model_kwargs = training_params['model_kwargs']
    clpr = training_params.get('clpr')
    rf_fit = train_unbiased_rf_for_split(
        training_set_splits, clpr=clpr, **model_kwargs,
    )

    return X, rf_fit


def train_unbiased_nosvlen(
    joint_bp_df: pd.DataFrame,
    workflow_config: dict,
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
        training_set_name,
        verbose=verbose,
    )


def train_unbiased_concat_rf(
    matched_bps_sources: dict[str, pd.DataFrame | Path],
    workflow_config: dict,
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
    _gt_name = _get_ground_truth_name(workflow_config)
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
    training_params = workflow_config['model']['training_params']
    model_kwargs = training_params['model_kwargs']
    clpr = training_params.get('clpr')
    rf_fit = train_unbiased_rf_for_split(
        training_set_splits, clpr=clpr, **model_kwargs,
    )

    return X, rf_fit


def train_unbiased_xgb(
    joint_bp_df: pd.DataFrame,
    workflow_config: dict,
    training_set_name: str,
    verbose: bool = False,
) -> tuple[pd.DataFrame, XGB]:
    """
    Train an unbiased XGBoost model for a single training set using the Duplessis method.
    """
    ground_truth_name = _get_ground_truth_name(workflow_config)
    training_set_df = mbps.preprocess_matchedbps_for_pu(
        joint_bp_df,
        workflow_config,
        ground_truth_name,
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
    training_params = workflow_config['model']['training_params']
    model_kwargs = training_params['model_kwargs']
    clpr = training_params.get('clpr')
    xgb_fit = train_unbiased_xgb_for_split(
        training_set_splits, clpr=clpr, **model_kwargs,
    )

    return X, xgb_fit


def train_unbiased_xgb_concat(
    matched_bps_sources: dict[str, pd.DataFrame | Path],
    workflow_config: dict,
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
    _gt_name = _get_ground_truth_name(workflow_config)
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
    training_params = workflow_config['model']['training_params']
    model_kwargs = training_params['model_kwargs']
    clpr = training_params.get('clpr')
    xgb_fit = train_unbiased_xgb_for_split(
        training_set_splits, clpr=clpr, **model_kwargs,
    )

    return X, xgb_fit


def train_unbiased_xgb_nosvlen(
    joint_bp_df: pd.DataFrame,
    workflow_config: dict,
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
        training_set_name,
        verbose=verbose,
    )

# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

model_dispatch = {
    'consensus': train_consensus_classifier,
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
            Must include 'ground_truth' with at least a 'name' key.
        training_set_name: Name of the training set.
        verbose: If True, print verbose output.
        matched_bps_sources: Mapping of sample_name -> matched_bps DataFrame or
            path, for pooled (concat/cross) models (see MODEL_CONFIGS
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
            matched_bps_sources, workflow_config,
            training_set_name, verbose,
        )

    if joint_bp_df is None:
        raise ValueError(f"Model '{workflow_config['model']['name']}' requires joint_bp_df.")
    return model_function(
        joint_bp_df, workflow_config,
        training_set_name, verbose,
    )
