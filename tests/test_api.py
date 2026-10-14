"""Tests for svensemble.api steps and the svensemble.io artifact readers/writers."""

import math

import numpy as np
import pandas as pd
import pytest

from svensemble import api
from svensemble import io as sv_io


def _train_shaped_X():
    """A feature matrix as train returns it: indexed by <callset>_INFO_EVENT,
    with NaN where a callset has no record for the matched breakpoint."""
    index = pd.MultiIndex.from_tuples(
        [("d1", "m1"), ("d2", np.nan), (np.nan, "m3")],
        names=["delly_INFO_EVENT", "manta_diploid_INFO_EVENT"],
    )
    return pd.DataFrame(
        {"f_a": [1.0, 2.0, 3.0], "f_b": [10, 20, 30], "f_extra": [0.5, 0.5, 0.5]},
        index=index,
    )


class _FakeModel:
    """Records the matrix it scores; probability is f_a / 10."""
    feature_names_in_ = np.array(["f_b", "f_a", "f_missing"])

    def predict_proba(self, X):
        self.seen = X
        p = X["f_a"].to_numpy(dtype=float) / 10
        return np.column_stack([1 - p, p])


class TestFeatureMatrixIO:
    def test_index_round_trip(self, tmp_path):
        X = _train_shaped_X()
        path = sv_io.write_feature_matrix(X, tmp_path / "sub" / sv_io.FEATURE_MATRIX)
        pd.testing.assert_frame_equal(sv_io.read_feature_matrix(path), X)

    def test_missing_index_columns_rejected(self, tmp_path):
        path = tmp_path / sv_io.FEATURE_MATRIX
        pd.DataFrame({"f_a": [1.0]}).to_csv(path, index=False)
        with pytest.raises(ValueError, match="_INFO_EVENT"):
            sv_io.read_feature_matrix(path)


class TestPredict:
    @pytest.fixture
    def captured_preds(self, monkeypatch):
        captured = {}

        def fake_get_calls(matched_bps, pred_df):
            captured["pred_df"] = pred_df
            return pred_df

        monkeypatch.setattr("svensemble.matched_breakpoints.get_svensemble2_calls", fake_get_calls)
        return captured

    @pytest.mark.parametrize("model_name, fill", [("unbiased_rf", 0.0), ("unbiased_xgb", np.nan)])
    def test_accepts_train_shaped_X(self, captured_preds, model_name, fill):
        model = _FakeModel()
        api.predict(model, _train_shaped_X(), pd.DataFrame(), model_name, "manta_diploid+delly")

        assert list(model.seen.columns) == ["f_b", "f_a", "f_missing"]
        np.testing.assert_array_equal(model.seen["f_missing"], [fill] * 3)
        pred_df = captured_preds["pred_df"]
        pred_col = f"manta_diploid+delly_{api.model_id(model=model_name)}_pred"
        assert list(pred_df.columns) == ["delly_INFO_EVENT", "manta_diploid_INFO_EVENT", pred_col]
        np.testing.assert_allclose(pred_df[pred_col], [0.1, 0.2, 0.3])

    def test_in_memory_and_read_back_X_agree(self, tmp_path, captured_preds):
        X = _train_shaped_X()
        api.predict(_FakeModel(), X, pd.DataFrame(), "unbiased_xgb", "ts")
        from_memory = captured_preds["pred_df"]
        X_back = sv_io.read_feature_matrix(sv_io.write_feature_matrix(X, tmp_path / sv_io.FEATURE_MATRIX))
        api.predict(_FakeModel(), X_back, pd.DataFrame(), "unbiased_xgb", "ts")
        pd.testing.assert_frame_equal(captured_preds["pred_df"], from_memory)


class TestMatch:
    def test_does_not_modify_input(self, monkeypatch):
        captured = {}

        def fake_outer_join(bp_df_dict, ref_callset, insert_delta_threshold, max_dist, workflow_config):
            captured.update(bp_df_dict=bp_df_dict, ref_callset=ref_callset)
            return pd.DataFrame()

        monkeypatch.setattr("svensemble.chunk_match.outer_join_bps", fake_outer_join)
        caller_bps = {"delly": pd.DataFrame({"a": [1]}), "manta_diploid": pd.DataFrame({"a": [2]})}
        gt = pd.DataFrame({"a": [3]})

        api.match(caller_bps, gt, ground_truth_name="gnomad")

        assert list(caller_bps) == ["delly", "manta_diploid"]
        assert list(captured["bp_df_dict"]) == ["delly", "manta_diploid", "gnomad"]
        assert captured["bp_df_dict"]["gnomad"] is gt
        assert captured["ref_callset"] == "gnomad"


class TestTrain:
    @pytest.fixture
    def captured(self, monkeypatch):
        captured = {}

        def fake_train_model(joint_bp_df, workflow_config, training_set_name,
                             verbose=False, matched_bps_sources=None):
            captured.update(joint_bp_df=joint_bp_df, sources=matched_bps_sources,
                            workflow_config=workflow_config)
            return pd.DataFrame(), None

        monkeypatch.setattr("svensemble.models.train_model", fake_train_model)
        return captured

    def test_dataframe_is_joint_bp_df(self, captured):
        df = pd.DataFrame({"a": [1]})
        api.train(df, ["delly"], "unbiased_xgb", "delly")
        assert captured["joint_bp_df"] is df
        assert captured["sources"] is None
        assert captured["workflow_config"]["ground_truth"]["name"] == "gnomad"

    def test_mapping_is_matched_bps_sources(self, captured, tmp_path):
        sources = {"s1": tmp_path / "a.csv", "s2": pd.DataFrame({"a": [1]})}
        api.train(sources, ["delly"], "unbiased_concat", "delly")
        assert captured["joint_bp_df"] is None
        assert list(captured["sources"]) == ["s1", "s2"]
        assert all(captured["sources"][k] is v for k, v in sources.items())


class TestModelConfigs:
    def test_read_model_config_decodes_tagged_values(self, tmp_path):
        cfg = {"base": "unbiased_xgb", "training_params": {"svlens": (50, math.inf)}}
        path = sv_io.write_config(sv_io.dict_json_encode(cfg), tmp_path / "cfg.json")
        assert sv_io.read_config(path)["training_params"]["svlens"]["__type__"] == "tuple"
        assert sv_io.read_model_config(path) == cfg

    def test_read_model_config_plain_json_unchanged(self, tmp_path):
        cfg = {"base": "unbiased_xgb", "training_params": {"fillna": 0}}
        path = sv_io.write_config(cfg, tmp_path / "cfg.json")
        assert sv_io.read_model_config(path) == cfg

    @pytest.mark.parametrize("suffix", ["json", "yaml"])
    def test_generate_config_round_trip(self, tmp_path, suffix):
        out = tmp_path / f"cfg.{suffix}"
        cfg = api.generate_config("unbiased_xgb", training_params={"svlens": (50, math.inf)}, output=out)
        assert cfg == {"base": "unbiased_xgb", "training_params": {"svlens": (50, math.inf)}}
        assert api.model_id(model_config=sv_io.read_model_config(out)) == api.model_id(model_config=cfg)

    def test_generate_config_without_overrides_matches_preset(self):
        assert (api.model_id(model_config=api.generate_config("unbiased_xgb"))
                == api.model_id(model="unbiased_xgb"))

    @pytest.mark.parametrize("bad", [
        {"base": "no_such_preset"},
        {"base": "unbiased_xgb", "training_params": {"function": "no_such_function"}},
    ])
    def test_generate_config_validates(self, bad):
        with pytest.raises(ValueError):
            api.generate_config(**bad)

    def test_model_id_requires_an_argument(self):
        with pytest.raises(ValueError, match="Provide model"):
            api.model_id()


class TestValidationSet:
    def test_unknown_name_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="Unknown validation set 'nope'"):
            api.preprocess_validation_set("nope", tmp_path / "raw.vcf", tmp_path / "out.csv")

    def test_missing_raw_file_rejected(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            api.preprocess_validation_set("kosugi", tmp_path / "raw.bed", tmp_path / "out.csv")
