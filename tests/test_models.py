"""Tests for svensemble.models and related refactored components."""

from unittest.mock import patch

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


def _make_pu_ready_df(n=40, seed=0):
    """DataFrame already shaped like preprocess_matchedbps_for_pu's output
    (i.e. what a matched_bps_sources entry looks like *after* preprocessing)."""
    rng = np.random.RandomState(seed)
    return pd.DataFrame({
        "numeric_a": rng.rand(n) * 10,
        "ref_match": (rng.rand(n) > 0.4),
    })


@pytest.fixture
def concat_workflow_config():
    return {
        "sample_name": "target_sample",
        "model": {
            "name": "unbiased_concat",
            "training_params": {
                "function": "unbiased_concat",
                "concat": True,
                "model_kwargs": {"n_estimators": 10},
            },
        },
    }


@pytest.fixture
def concat_ground_truth_config():
    return {"callsets": {"gnomad": {"source": "gnomad"}}}


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
        estimation_params, weight_dict = models._estimate_duplessis_weights(
            splits, {"n_estimators": 50},
        )
        assert 0 in weight_dict and 1 in weight_dict
        assert weight_dict[0] > 0 and weight_dict[1] > 0
        assert "label_prob" in estimation_params

    def test_zero_weight_when_no_positives_in_train(self, splits):
        splits["y_train"] = pd.Series(np.zeros(len(splits["y_train"]), dtype=int))
        estimation_params, weight_dict = models._estimate_duplessis_weights(
            splits, {"n_estimators": 50},
        )
        assert weight_dict[1] == 0

    def test_zero_weight_when_no_positives_in_val(self, splits):
        splits["y_val"] = pd.Series(np.zeros(len(splits["y_val"]), dtype=int))
        estimation_params, weight_dict = models._estimate_duplessis_weights(
            splits, {"n_estimators": 50},
        )
        assert weight_dict[1] == 0

    def test_handles_nan_in_features(self, splits_with_nan):
        """The non-traditional classifier should fill NaN internally."""
        estimation_params, weight_dict = models._estimate_duplessis_weights(
            splits_with_nan, {"n_estimators": 50},
        )
        assert 0 in weight_dict and 1 in weight_dict

    def test_caps_nontrad_estimators(self, splits):
        estimation_params, weight_dict = models._estimate_duplessis_weights(
            splits, {"n_estimators": 1000},
        )
        assert 0 in weight_dict and 1 in weight_dict

    def test_floors_label_freq_when_estimate_near_zero(self, splits):
        """When the non-traditional classifier assigns near-zero probability
        to the true positives in the validation set, label_freq must be
        floored to _LABEL_FREQ_FLOOR."""
        n_val = len(splits["y_val"])
        with patch.object(
            RF, "predict_proba",
            return_value=np.column_stack([np.full(n_val, 1 - 1e-7), np.full(n_val, 1e-7)]),
        ):
            estimation_params, weight_dict = models._estimate_duplessis_weights(
                splits, {"n_estimators": 50},
            )
        label_prob = splits["y"].sum() / len(splits["y"])
        expected_clpr = label_prob / models._LABEL_FREQ_FLOOR
        assert weight_dict[1] == pytest.approx(2 * expected_clpr / label_prob)
        assert np.isfinite(weight_dict[1])

    def test_estimator_xgb(self, splits):
        """XGBoost non-traditional estimator produces finite, positive weights."""
        estimation_params, weight_dict = models._estimate_duplessis_weights(
            splits, {"n_estimators": 50}, estimator="xgb",
        )
        assert 0 in weight_dict and 1 in weight_dict
        assert np.isfinite(weight_dict[0]) and np.isfinite(weight_dict[1])
        assert weight_dict[0] > 0 and weight_dict[1] > 0

    def test_unknown_estimator_raises(self, splits):
        """Unknown estimator string raises ValueError."""
        with pytest.raises(ValueError, match="Unknown non-traditional estimator"):
            models._estimate_duplessis_weights(
                splits, {"n_estimators": 50}, estimator="unknown",
            )

    def test_clpr_shortcircuits_estimation(self, splits):
        """Supplying clpr skips estimation and returns expected weights."""
        clpr_input = 0.4
        estimation_params, weight_dict = models._estimate_duplessis_weights(
            splits, {"n_estimators": 50}, clpr=clpr_input,
        )
        assert estimation_params["clpr"] == clpr_input
        label_prob = splits["y"].sum() / len(splits["y"])
        assert weight_dict[0] == pytest.approx(1 / (1 - label_prob))
        assert weight_dict[1] == pytest.approx(2 * clpr_input / label_prob)


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
        proba = model.predict_proba(splits_with_nan["X"])
        assert proba.shape[0] == len(splits_with_nan["X"])

    def test_predictions_differ_from_rf(self, splits):
        """XGBoost and RF should produce different probability outputs."""
        rf = models.train_unbiased_rf_for_split(splits, n_estimators=20)
        xgb = models.train_unbiased_xgb_for_split(splits, n_estimators=20)
        rf_proba = rf.predict_proba(splits["X"])[:, 1]
        xgb_proba = xgb.predict_proba(splits["X"])[:, 1]
        assert not np.allclose(rf_proba, xgb_proba), (
            "RF and XGBoost should produce different outputs"
        )


# ---------------------------------------------------------------------------
# train_unbiased_concat_rf / train_unbiased_xgb_concat
#
# preprocess_matchedbps_for_pu is patched to the identity function: these
# tests target the pooling refactor itself (uniform dict handling, DataFrame
# vs. Path sources, concatenation), not the preprocessing step.
# ---------------------------------------------------------------------------

@pytest.mark.slow
class TestTrainUnbiasedConcatRf:

    def test_single_source_dataframe(self, concat_workflow_config, concat_ground_truth_config):
        df = _make_pu_ready_df(n=40)
        with patch.object(mbps, "preprocess_matchedbps_for_pu", side_effect=lambda d, *a, **k: d):
            X, rf_fit = models.train_unbiased_concat_rf(
                {"target_sample": df}, concat_workflow_config,
                concat_ground_truth_config, "gridss+manta_diploid",
            )
        assert len(X) == len(df)
        assert isinstance(rf_fit, RF)

    def test_multi_source_mixes_dataframe_and_path(
        self, concat_workflow_config, concat_ground_truth_config, tmp_path,
    ):
        df_a = _make_pu_ready_df(n=30, seed=1)
        df_b = _make_pu_ready_df(n=20, seed=2)
        csv_b = tmp_path / "sampleB_matched_bps.csv"
        df_b.to_csv(csv_b, index=False)

        with patch.object(mbps, "preprocess_matchedbps_for_pu", side_effect=lambda d, *a, **k: d):
            X, rf_fit = models.train_unbiased_concat_rf(
                {"sampleA": df_a, "sampleB": csv_b}, concat_workflow_config,
                concat_ground_truth_config, "gridss+manta_diploid",
            )
        assert len(X) == len(df_a) + len(df_b)

    def test_every_source_preprocessed_uniformly(
        self, concat_workflow_config, concat_ground_truth_config,
    ):
        """No 'primary sample' branch: every entry in matched_bps_sources,
        regardless of key, is passed through preprocessing exactly once."""
        df_1 = _make_pu_ready_df(n=15, seed=3)
        df_2 = _make_pu_ready_df(n=25, seed=4)
        seen_lengths = []

        def _track(d, *a, **k):
            seen_lengths.append(len(d))
            return d

        with patch.object(mbps, "preprocess_matchedbps_for_pu", side_effect=_track):
            models.train_unbiased_concat_rf(
                {"other_1": df_1, "other_2": df_2}, concat_workflow_config,
                concat_ground_truth_config, "gridss+manta_diploid",
            )
        assert sorted(seen_lengths) == sorted([len(df_1), len(df_2)])


@pytest.mark.slow
class TestTrainUnbiasedXgbConcat:

    def test_single_source_dataframe(self, concat_workflow_config, concat_ground_truth_config):
        df = _make_pu_ready_df(n=40)
        with patch.object(mbps, "preprocess_matchedbps_for_pu", side_effect=lambda d, *a, **k: d):
            X, xgb_fit = models.train_unbiased_xgb_concat(
                {"target_sample": df}, concat_workflow_config,
                concat_ground_truth_config, "gridss+manta_diploid",
            )
        assert len(X) == len(df)
        assert isinstance(xgb_fit, XGB)

    def test_multi_source_pools_and_concatenates(
        self, concat_workflow_config, concat_ground_truth_config, tmp_path,
    ):
        df_a = _make_pu_ready_df(n=18, seed=5)
        df_b = _make_pu_ready_df(n=22, seed=6)
        csv_b = tmp_path / "sampleB_matched_bps.csv"
        df_b.to_csv(csv_b, index=False)

        with patch.object(mbps, "preprocess_matchedbps_for_pu", side_effect=lambda d, *a, **k: d):
            X, xgb_fit = models.train_unbiased_xgb_concat(
                {"sampleA": df_a, "sampleB": csv_b}, concat_workflow_config,
                concat_ground_truth_config, "gridss+manta_diploid",
            )
        assert len(X) == len(df_a) + len(df_b)


# ---------------------------------------------------------------------------
# train_model dispatcher
# ---------------------------------------------------------------------------

class TestTrainModelDispatcher:

    def _config(self, concat: bool) -> dict:
        return {
            "model": {
                "name": "fake_model",
                "training_params": {"function": "fake_function", "concat": concat},
            }
        }

    def test_concat_model_without_sources_raises(self, monkeypatch):
        monkeypatch.setitem(models.model_dispatch, "fake_function", lambda *a, **k: (None, None))
        with pytest.raises(ValueError, match="matched_bps_sources"):
            models.train_model(None, self._config(concat=True), {}, "ts")

    def test_non_concat_model_without_joint_bp_df_raises(self, monkeypatch):
        monkeypatch.setitem(models.model_dispatch, "fake_function", lambda *a, **k: (None, None))
        with pytest.raises(ValueError, match="joint_bp_df"):
            models.train_model(None, self._config(concat=False), {}, "ts")

    def test_concat_model_dispatches_with_matched_bps_sources(self, monkeypatch):
        captured = {}

        def fake_fn(matched_bps_sources, workflow_config, ground_truth_config, training_set_name, verbose):
            captured["matched_bps_sources"] = matched_bps_sources
            return ("X", "model")

        monkeypatch.setitem(models.model_dispatch, "fake_function", fake_fn)
        sources = {"s1": pd.DataFrame({"a": [1]})}
        result = models.train_model(
            None, self._config(concat=True), {}, "ts", matched_bps_sources=sources,
        )
        assert result == ("X", "model")
        assert captured["matched_bps_sources"] is sources

    def test_non_concat_model_dispatches_with_joint_bp_df(self, monkeypatch):
        captured = {}

        def fake_fn(joint_bp_df, workflow_config, ground_truth_config, training_set_name, verbose):
            captured["joint_bp_df"] = joint_bp_df
            return ("X", "model")

        monkeypatch.setitem(models.model_dispatch, "fake_function", fake_fn)
        df = pd.DataFrame({"a": [1]})
        result = models.train_model(df, self._config(concat=False), {}, "ts")
        assert result == ("X", "model")
        assert captured["joint_bp_df"] is df


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

    def test_unbiased_xgb_concat_resolves(self):
        assert models.model_dispatch["unbiased_xgb_concat"] is models.train_unbiased_xgb_concat

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
    def test_xgb_config_has_xgb_function(self, config_name):
        cfg = _model_configs.model_configs[config_name]
        func_name = cfg["training_params"]["function"]
        assert func_name in models.model_dispatch

    @pytest.mark.parametrize("config_name", [
        k for k in _model_configs.model_configs if "xgb" not in k
    ])
    def test_rf_config_fillna_defaults_to_zero_or_none(self, config_name):
        cfg = _model_configs.model_configs[config_name]
        fillna = cfg["training_params"].get("fillna", 0)
        # RF variants either fill NaN with 0 or (nosvlen variants) leave NaN.
        assert fillna == 0 or fillna is None

    def test_all_configs_have_base_applied(self):
        for name, cfg in _model_configs.model_configs.items():
            assert "base" in cfg, f"{name} missing base"
            assert cfg["base"] in _model_configs.BASE_MODELS, (
                f"{name} base {cfg['base']!r} is not a canonical base preset"
            )

    def test_no_legacy_suffix_keys_remain(self):
        for name, cfg in _model_configs.model_configs.items():
            for suffix_key in ("bps_suffix", "matched_bps_suffix",
                               "model_pkl_suffix", "feature_matrix_suffix"):
                assert suffix_key not in cfg, f"{name} still has {suffix_key}"

    def test_all_configs_have_concat_flag(self):
        for name, cfg in _model_configs.model_configs.items():
            assert "concat" in cfg["training_params"], f"{name} missing training_params['concat']"

    def test_only_concat_models_declare_concat_true(self):
        concat_models = {
            name for name, cfg in _model_configs.model_configs.items()
            if cfg["training_params"]["concat"]
        }
        assert concat_models == {"unbiased_concat", "unbiased_xgb_concat"}

    def test_concat_models_use_pooling_functions(self):
        for name in ("unbiased_concat", "unbiased_xgb_concat"):
            cfg = _model_configs.model_configs[name]
            assert cfg["training_params"]["function"] == name
            assert "function_params" not in cfg["training_params"], (
                f"{name} should no longer bake a fixed samples list into the config"
            )
