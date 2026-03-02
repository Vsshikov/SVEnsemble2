"""
PU models for SV call probability estimation.
"""
import numpy as np
from sklearn.ensemble import RandomForestClassifier as RF
import pandas as pd
import sklearn.model_selection as skm
from . import matched_breakpoints as mbps
from . import utils as sv_utils

def train_unbiased_rf_for_split(
    training_set_train_test_split: dict,
    **model_kwargs
) -> RF | None:
    """
    Train an unbiased random forest model for a single training set using the Duplessis method.

    Args:
        training_set_train_test_split (dict):
            Dictionary containing train/test splits for a training set.

    Returns:
        RF: Trained RandomForestClassifier model.
    """
    # Set default model kwargs.
    # Rewrite/update if model_kwargs are supplied
    _model_kwargs = {
        'n_estimators': 500,
        'max_depth': None
    }
    _model_kwargs.update(model_kwargs)

    X = training_set_train_test_split['X']
    y = training_set_train_test_split['y']
    X_train = training_set_train_test_split['X_train']
    y_train = training_set_train_test_split['y_train']
    X_val = training_set_train_test_split['X_val']
    y_val = training_set_train_test_split['y_val']

    # Check if any of the sets are empty
    if y_train.sum() == 0 or y_val.sum() == 0:
        print(
            "No positive labels in the training or validation set, skipping"
        )
        return None

    rf_nontrad = RF(
        max_features=int(np.sqrt(X.shape[1])),
        random_state=0,
        **_model_kwargs
    )
    rf_nontrad_fit = rf_nontrad.fit(X_train, y_train)
    y_nontrad = rf_nontrad_fit.predict(X_val)

    label_freq = (y_nontrad & y_val).sum() / y_val.sum()
    label_prob = y.sum() / len(y)

    if label_freq == 0:
        print(
            "label_freq is 0: the non-traditional RF predicted no true positives "
            "in the validation set. This means no matches to ground truth were found in this dataset. "
            "Skipping this split."
        )
        return None

    clpr = label_prob / label_freq

    weight_dict = {
        0: 1 / (1 - label_prob),         # Weight for negative class
        1: 2 * clpr / label_prob         # Weight for positive class
    }

    rf_duplessis = RF(
        max_features=int(np.sqrt(X.shape[1])),
        random_state=0,
        class_weight=weight_dict,
        **_model_kwargs
    )

    rf_duplessis_fit = rf_duplessis.fit(X_train, y_train)
    # y_duplessis = rf_duplessis_fit.predict_proba(X)

    return rf_duplessis_fit#, y_duplessis


def train_unbiased_rf(
    joint_bp_df: pd.DataFrame,
    workflow_config: dict,
    ground_truth_config: dict,
    training_set_name: str,
    verbose: bool = False
) -> tuple[pd.DataFrame, RF]:
    """
    Train an unbiased random forest model for a single training set using the Duplessis method.

    Args:
        joint_bp_df: DataFrame containing matched breakpoints for the training set.
        workflow_config: Dictionary containing the workflow configuration.
        ground_truth_config: Dictionary containing the ground truth configuration.
        training_set_name: Name of the training set.
        verbose: If True, print verbose output.
    Returns:
        tuple[pd.DataFrame, sklearn.ensemble.RandomForestClassifier]: Tuple containing the feature matrix and the trained RandomForestClassifier model.
    """
    training_set_df = mbps.preprocess_matchedbps_for_pu(
        joint_bp_df,
        workflow_config,
        ground_truth_config['callsets'][list(ground_truth_config['callsets'].keys())[0]]['source'],
        training_set_name,
        verbose=verbose
    )
    del joint_bp_df

    X, y = mbps.get_X_y_from_training_df(
        training_set_df,
        match_col='ref_match'
    )
    X_train, X_val, y_train, y_val = skm.train_test_split(
        X, y, train_size=0.8, random_state=0
    )
    training_set_splits = {
        'X': X,
        'X_train': X_train,
        'X_val': X_val,
        'y': y,
        'y_train': y_train,
        'y_val': y_val,
        'X_idx': training_set_df.index
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
    verbose: bool = False
) -> tuple[pd.DataFrame, RF]:
    """
    Train an unbiased random forest model on a concatenated training set formed from multiple
    matched breakpoints dataframes from multiple samples specified in model_config['training_params']['function_params']['samples'].
    While it accepts joint_bp_df, it will only use it if the sample is specified in model_config['training_params']['function_params']['samples']
    Loads matched breakpoints dataframes for the specified samples and concatenates them.
    If any of the samples is not found, raises an error.

    Args:
        joint_bp_df: DataFrame containing matched breakpoints for the training set.
        workflow_config: Dictionary containing the workflow configuration.
        ground_truth_config: Dictionary containing the ground truth configuration.
        training_set_name: Name of the training set.
        verbose: If True, print verbose output.
    Returns:
        tuple[pd.DataFrame, sklearn.ensemble.RandomForestClassifier]: Tuple containing the feature matrix and the trained RandomForestClassifier model.
    """
    training_set_dfs = []
    for sample in workflow_config['model']['training_params']['function_params']['samples']:
        if sample == workflow_config['sample_name']:
            sample_joint_bp_df = joint_bp_df
        else:
            sample_workflow_config = sv_utils.create_config_dict(sample, model_name=workflow_config['model']['name'], training_set=training_set_name)[sample]
            sample_joint_bp_df = pd.read_csv(sample_workflow_config['model']['matched_bps_path'], low_memory=False)
        sample_training_set_df = mbps.preprocess_matchedbps_for_pu(
            sample_joint_bp_df,
            sample_workflow_config,
            ground_truth_config['callsets'][list(ground_truth_config['callsets'].keys())[0]]['source'],
            training_set_name,
            verbose=verbose
        )
        del sample_joint_bp_df
        training_set_dfs.append(sample_training_set_df)
    training_set_df = pd.concat(training_set_dfs, ignore_index=True)
    del training_set_dfs
    X, y = mbps.get_X_y_from_training_df(
        training_set_df,
        match_col='ref_match'
    )
    X_train, X_val, y_train, y_val = skm.train_test_split(
        X, y, train_size=0.8, random_state=0
    )
    training_set_splits = {
        'X': X,
        'X_train': X_train,
        'X_val': X_val,
        'y': y,
        'y_train': y_train,
        'y_val': y_val,
        'X_idx': training_set_df.index
    }
    del training_set_df

    if verbose:
        print(f"Training PU model for training set: {training_set_name}")
    model_kwargs = workflow_config['model']['training_params']['model_kwargs']
    rf_fit = train_unbiased_rf_for_split(training_set_splits, **model_kwargs)

    return X, rf_fit

model_dispatch = {
    'unbiased': train_unbiased_rf,
    'unbiased_concat': train_unbiased_concat_rf
}

def train_model(
    joint_bp_df: pd.DataFrame,
    workflow_config: dict,
    ground_truth_config: dict,
    training_set_name: str,
    verbose: bool = False
) -> tuple[pd.DataFrame, RF]:
    """
    Train a model following workflow_config using joint_bp_df

    Args:
        joint_bp_df: DataFrame containing matched breakpoints for the training set.
        workflow_config: Dictionary containing the workflow configuration.
        ground_truth_config: Dictionary containing the ground truth configuration.
        training_set_name: Name of the training set.
        verbose: If True, print verbose output.
    Returns:
        tuple[pd.DataFrame, sklearn.ensemble.RandomForestClassifier]: Tuple containing the feature matrix and the trained RandomForestClassifier model.
    """
    model_function = model_dispatch[workflow_config['model']['training_params']['function']]
    return model_function(joint_bp_df, workflow_config, ground_truth_config, training_set_name, verbose)
