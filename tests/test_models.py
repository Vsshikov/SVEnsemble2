"""Tests for svensemble.models and related refactored components."""

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier as RF
from xgboost import XGBClassifier as XGB

from svensemble import matched_breakpoints as mbps
from svensemble import models
from svensemble import _model_configs


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_training_splits(n=200, n_features=10, pos_frac=0.3, with_nan=False):
    """Build a synthetic train/val split dict suitable for PU model training."""
    rng = np.random.RandomState(42)
    X = pd.DataFrame(
        rng.randn(n, n_features),
        columns=[f"feat_{i}" for i in range(n_features)],
    )
    y = pd.Series(
        (rng.rand(n) < pos_frac).astype(int),
        name="label",
    )
    if with_nan:
        # Simulate structural missingness: for ~30% of rows, null out
        # the last 3 features (as if a caller was absent).
        absent_mask = rng.rand(n) < 0.3
        X.loc[absent_mask, X.columns[-3:]] = np.nan

    split = int(n * 0.8)
    return {
        "X": X,
        "y": y,
        "X_train": X.iloc[:split],
        "y_train": y.iloc[:split],
        "X_val": X.iloc[split:],
        "y_val": y.iloc[split:],
    }


@pytest.fixture
def splits():
    return _make_training_splits()


@pytest.fixture
def splits_with_nan():
    return _make_training_splits(with_nan=True)


@pytest.fixture
def training_df():
    """DataFrame mimicking the output of preprocess_matchedbps_for_pu."""
    rng = np.random.RandomState(0)
    n = 50
    df = pd.DataFrame({
        "numeric_a": rng.rand(n) * 100,
        "numeric_b": rng.randint(0, 10, n).astype(float),
        "categorical": rng.choice(["DEL", "INS", "DUP"], n),
        "ref_match": (rng.rand(n) > 0.5),
    })
    # Sprinkle NaN to simulate caller-absent missingness
    df.loc[rng.rand(n) < 0.2, "numeric_a"] = np.nan
    df.loc[rng.rand(n) < 0.2, "numeric_b"] = np.nan
    return df


# ---------------------------------------------------------------------------
# get_X_y_from_training_df
# ---------------------------------------------------------------------------

class TestGetXyFromTrainingDf:

    def test_fillna_zero_no_nans_in_output(self, training_df):
        X, y = mbps.get_X_y_from_training_df(training_df, fillna=0)
        assert not X.isna().any().any()

    def test_fillna_none_preserves_nans(self, training_df):
        X, y = mbps.get_X_y_from_training_df(training_df, fillna=None)
        assert X.isna().any().any(), "NaN should be preserved when fillna=None"

    def test_fillna_none_all_float_dtypes(self, training_df):
        X, _ = mbps.get_X_y_from_training_df(training_df, fillna=None)
        for col in X.columns:
            assert X[col].dtype == np.float64, f"{col} should be float64"

    def test_fillna_zero_int_dtypes_where_possible(self, training_df):
        X, _ = mbps.get_X_y_from_training_df(training_df, fillna=0)
        for col in X.columns:
            assert X[col].dtype in (np.int64, np.float64, np.int32, np.float32, int, float)

    def test_y_is_ref_match(self, training_df):
        _, y = mbps.get_X_y_from_training_df(training_df, fillna=0)
        assert y.name == "ref_match" or (y == training_df["ref_match"]).all()

    def test_ref_match_not_in_X(self, training_df):
        X, _ = mbps.get_X_y_from_training_df(training_df, fillna=0)
        assert "ref_match" not in X.columns

    def test_categorical_one_hot_encoded(self, training_df):
        X, _ = mbps.get_X_y_from_training_df(training_df, fillna=0)
        cat_cols = [c for c in X.columns if c.startswith("categorical_")]
        assert len(cat_cols) > 0, "Categorical column should be one-hot encoded"

    def test_custom_fillna_value(self, training_df):
        X, _ = mbps.get_X_y_from_training_df(training_df, fillna=-999)
        assert not X.isna().any().any()
        assert (X == -999).any().any(), "Custom fill value should appear in output"

    def test_shapes_consistent(self, training_df):
        X, y = mbps.get_X_y_from_training_df(training_df, fillna=0)
        assert len(X) == len(y) == len(training_df)


# ---------------------------------------------------------------------------
# _estimate_duplessis_weights
# ---------------------------------------------------------------------------

@pytest.mark.slow
class TestEstimateDuplessisWeights:

    def test_returns_weight_dict(self, splits):
        result = models._estimate_duplessis_weights(splits, {"n_estimators": 50})
        assert result is not None
        assert 0 in result and 1 in result
        assert result[0] > 0 and result[1] > 0

    def test_returns_none_when_no_positives_in_train(self, splits):
        splits["y_train"] = pd.Series(np.zeros(len(splits["y_train"]), dtype=int))
        result = models._estimate_duplessis_weights(splits, {"n_estimators": 50})
        assert result is None

    def test_returns_none_when_no_positives_in_val(self, splits):
        splits["y_val"] = pd.Series(np.zeros(len(splits["y_val"]), dtype=int))
        result = models._estimate_duplessis_weights(splits, {"n_estimators": 50})
        assert result is None

    def test_handles_nan_in_features(self, splits_with_nan):
        """The non-traditional RF should fill NaN internally."""
        result = models._estimate_duplessis_weights(
            splits_with_nan, {"n_estimators": 50},
        )
        # Should succeed without error (NaN is filled internally)
        assert result is None or (0 in result and 1 in result)

    def test_caps_nontrad_estimators(self, splits):
        result = models._estimate_duplessis_weights(
            splits, {"n_estimators": 1000},
        )
        # Should not raise — n_estimators capped to 70 for non-trad RF
        assert result is None or isinstance(result, dict)


# ---------------------------------------------------------------------------
# train_unbiased_rf_for_split
# ---------------------------------------------------------------------------

@pytest.mark.slow
class TestTrainUnbiasedRfForSplit:

    def test_returns_rf_classifier(self, splits):
        model = models.train_unbiased_rf_for_split(splits, n_estimators=20)
        assert model is not None
        assert isinstance(model, RF)

    def test_has_predict_proba(self, splits):
        model = models.train_unbiased_rf_for_split(splits, n_estimators=20)
        proba = model.predict_proba(splits["X"])
        assert proba.shape == (len(splits["X"]), 2)

    def test_has_feature_names(self, splits):
        model = models.train_unbiased_rf_for_split(splits, n_estimators=20)
        assert hasattr(model, "feature_names_in_")
        assert list(model.feature_names_in_) == list(splits["X"].columns)


# ---------------------------------------------------------------------------
# train_unbiased_xgb_for_split
# ---------------------------------------------------------------------------

@pytest.mark.slow
class TestTrainUnbiasedXgbForSplit:

    def test_returns_xgb_classifier(self, splits):
        model = models.train_unbiased_xgb_for_split(splits, n_estimators=20)
        assert model is not None
        assert isinstance(model, XGB)

    def test_has_predict_proba(self, splits):
        model = models.train_unbiased_xgb_for_split(splits, n_estimators=20)
        proba = model.predict_proba(splits["X"])
        assert proba.shape == (len(splits["X"]), 2)

    def test_has_feature_names(self, splits):
        model = models.train_unbiased_xgb_for_split(splits, n_estimators=20)
        assert hasattr(model, "feature_names_in_")

    def test_handles_nan_features(self, splits_with_nan):
        """XGBoost should train without error even with NaN in features."""
        model = models.train_unbiased_xgb_for_split(
            splits_with_nan, n_estimators=20,
        )
        if model is not None:
            proba = model.predict_proba(splits_with_nan["X"])
            assert proba.shape[0] == len(splits_with_nan["X"])

    def test_predictions_differ_from_rf(self, splits):
        """XGBoost and RF should produce different probability outputs."""
        rf = models.train_unbiased_rf_for_split(splits, n_estimators=20)
        xgb = models.train_unbiased_xgb_for_split(splits, n_estimators=20)
        if rf is not None and xgb is not None:
            rf_proba = rf.predict_proba(splits["X"])[:, 1]
            xgb_proba = xgb.predict_proba(splits["X"])[:, 1]
            assert not np.allclose(rf_proba, xgb_proba), (
                "RF and XGBoost should produce different outputs"
            )


# ---------------------------------------------------------------------------
# _get_fillna
# ---------------------------------------------------------------------------

class TestGetFillna:

    def test_default_is_zero(self):
        config = {"model": {"training_params": {"model_kwargs": {}}}}
        assert models._get_fillna(config) == 0

    def test_reads_none(self):
        config = {"model": {"training_params": {"fillna": None, "model_kwargs": {}}}}
        assert models._get_fillna(config) is None

    def test_reads_custom_value(self):
        config = {"model": {"training_params": {"fillna": -999, "model_kwargs": {}}}}
        assert models._get_fillna(config) == -999


# ---------------------------------------------------------------------------
# Model dispatch
# ---------------------------------------------------------------------------

class TestModelDispatch:

    def test_unbiased_resolves(self):
        assert models.model_dispatch["unbiased"] is models.train_unbiased_rf

    def test_unbiased_concat_resolves(self):
        assert models.model_dispatch["unbiased_concat"] is models.train_unbiased_concat_rf

    def test_unbiased_xgb_resolves(self):
        assert models.model_dispatch["unbiased_xgb"] is models.train_unbiased_xgb

    def test_unbiased_concat_xgb_resolves(self):
        assert models.model_dispatch["unbiased_concat_xgb"] is models.train_unbiased_concat_xgb

    def test_all_dispatch_values_are_callable(self):
        for key, func in models.model_dispatch.items():
            assert callable(func), f"{key} is not callable"


# ---------------------------------------------------------------------------
# _model_configs: XGBoost configs structure
# ---------------------------------------------------------------------------

class TestModelConfigs:

    @pytest.mark.parametrize("config_name", [
        k for k in _model_configs.model_configs if "xgb" in k
    ])
    def test_xgb_config_has_fillna_none(self, config_name):
        cfg = _model_configs.model_configs[config_name]
        assert cfg["training_params"]["fillna"] is None

    @pytest.mark.parametrize("config_name", [
        k for k in _model_configs.model_configs if "xgb" in k
    ])
    def test_xgb_config_shares_bps_suffix(self, config_name):
        """XGBoost configs should share preprocessed data with an RF counterpart."""
        cfg = _model_configs.model_configs[config_name]
        assert cfg.get("bps_suffix") is not None
        assert "xgb" not in cfg["bps_suffix"], (
            "bps_suffix should point to RF's preprocessed data"
        )

    @pytest.mark.parametrize("config_name", [
        k for k in _model_configs.model_configs if "xgb" in k
    ])
    def test_xgb_config_has_xgb_function(self, config_name):
        cfg = _model_configs.model_configs[config_name]
        func_name = cfg["training_params"]["function"]
        assert func_name in models.model_dispatch

    @pytest.mark.parametrize("config_name", [
        k for k in _model_configs.model_configs if "xgb" not in k
    ])
    def test_rf_config_fillna_defaults_to_zero(self, config_name):
        cfg = _model_configs.model_configs[config_name]
        fillna = cfg["training_params"].get("fillna", 0)
        assert fillna == 0

    def test_all_configs_have_defaults_applied(self):
        for name, cfg in _model_configs.model_configs.items():
            assert "bps_suffix" in cfg, f"{name} missing bps_suffix"
            assert "matched_bps_suffix" in cfg, f"{name} missing matched_bps_suffix"
            assert "model_pkl_suffix" in cfg, f"{name} missing model_pkl_suffix"
            assert "feature_matrix_suffix" in cfg, f"{name} missing feature_matrix_suffix"
