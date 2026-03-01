"""
PU models for SV call probability estimation.
"""
import numpy as np
from sklearn.ensemble import RandomForestClassifier as RF


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
