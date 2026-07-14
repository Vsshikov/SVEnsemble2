"""Tests for svensemble.config_io: round-trip, validation, merge."""

from pathlib import Path

import numpy as np
import pytest

from svensemble.config_io import (
    dump_config,
    load_config,
    merge_configs,
    rehydrate_config,
    validate_config,
)


class TestDumpLoad:
    def test_round_trip_basic(self, tmp_path):
        config = {
            "sample_name": "NA12878",
            "model_name": "unbiased",
            "callers": ["gridss", "manta_diploid"],
            "match_params": {"max_dist": 100, "insert_delta_threshold": 0.7},
        }
        path = tmp_path / "config.json"
        dump_config(config, path)
        loaded = load_config(path)
        assert loaded == config

    def test_path_serialization(self, tmp_path):
        config = {"sample_name": "test", "some_path": Path("/foo/bar.csv")}
        path = tmp_path / "config.json"
        dump_config(config, path)
        loaded = load_config(path)
        assert loaded["some_path"] == "/foo/bar.csv"

    def test_inf_serialization(self, tmp_path):
        config = {"sample_name": "test", "svlens": [50, np.inf]}
        path = tmp_path / "config.json"
        dump_config(config, path)
        loaded = load_config(path)
        assert loaded["svlens"] == [50, None]

    def test_nan_serialization(self, tmp_path):
        config = {"sample_name": "test", "value": np.nan}
        path = tmp_path / "config.json"
        dump_config(config, path)
        loaded = load_config(path)
        assert loaded["value"] is None

    def test_numpy_int_serialization(self, tmp_path):
        config = {"sample_name": "test", "count": np.int64(42)}
        path = tmp_path / "config.json"
        dump_config(config, path)
        loaded = load_config(path)
        assert loaded["count"] == 42
        assert isinstance(loaded["count"], int)

    def test_tuple_becomes_list(self, tmp_path):
        config = {"sample_name": "test", "svlens": (50, 1000)}
        path = tmp_path / "config.json"
        dump_config(config, path)
        loaded = load_config(path)
        assert loaded["svlens"] == [50, 1000]

    def test_set_becomes_list(self, tmp_path):
        config = {"sample_name": "test", "callers": {"gridss", "delly"}}
        path = tmp_path / "config.json"
        dump_config(config, path)
        loaded = load_config(path)
        assert set(loaded["callers"]) == {"gridss", "delly"}

    def test_creates_parent_dirs(self, tmp_path):
        path = tmp_path / "sub" / "dir" / "config.json"
        dump_config({"sample_name": "test"}, path)
        assert path.exists()


class TestRehydrate:
    def test_path_key_rehydrated(self, tmp_path):
        config = {"sample_name": "test", "ref_path": Path("/data/ref.fa")}
        path = tmp_path / "config.json"
        dump_config(config, path)
        loaded = load_config(path)
        assert loaded["ref_path"] == "/data/ref.fa"
        rehydrated = rehydrate_config(loaded, path_keys={"ref_path"})
        assert rehydrated["ref_path"] == Path("/data/ref.fa")

    def test_bare_inf_key_rehydrated(self, tmp_path):
        config = {"sample_name": "test", "max_af": np.inf}
        path = tmp_path / "config.json"
        dump_config(config, path)
        loaded = load_config(path)
        assert loaded["max_af"] is None
        rehydrated = rehydrate_config(loaded, inf_keys={"max_af"})
        assert rehydrated["max_af"] == np.inf

    def test_inf_inside_list_rehydrated(self, tmp_path):
        config = {"sample_name": "test", "svlens": [50, np.inf]}
        path = tmp_path / "config.json"
        dump_config(config, path)
        loaded = load_config(path)
        rehydrated = rehydrate_config(loaded, inf_keys={"svlens"})
        assert rehydrated["svlens"] == [50, np.inf]

    def test_nested_dict_keys_rehydrated(self, tmp_path):
        config = {"sample_name": "test", "ground_truth": {"svlens": [50, np.inf]}}
        path = tmp_path / "config.json"
        dump_config(config, path)
        loaded = load_config(path)
        rehydrated = rehydrate_config(loaded, inf_keys={"svlens"})
        assert rehydrated["ground_truth"]["svlens"] == [50, np.inf]

    def test_no_schema_is_a_noop(self, tmp_path):
        config = {"sample_name": "test", "ref_path": "already/a/string", "svlens": [50, None]}
        rehydrated = rehydrate_config(config)
        assert rehydrated == config


class TestValidation:
    def test_valid_config(self):
        validate_config({"sample_name": "NA12878"})

    def test_missing_sample_name(self):
        with pytest.raises(ValueError, match="sample_name"):
            validate_config({"model_name": "unbiased"})

    def test_empty_config_fails(self):
        with pytest.raises(ValueError):
            validate_config({})


class TestMerge:
    def test_override_wins(self):
        base = {"sample_name": "old", "model_name": "unbiased"}
        override = {"sample_name": "new"}
        merged = merge_configs(base, override)
        assert merged["sample_name"] == "new"
        assert merged["model_name"] == "unbiased"

    def test_nested_dict_merge(self):
        base = {"sample_name": "test", "match_params": {"max_dist": 100}}
        override = {"match_params": {"insert_delta_threshold": 0.7}}
        merged = merge_configs(base, override)
        assert merged["match_params"]["max_dist"] == 100
        assert merged["match_params"]["insert_delta_threshold"] == 0.7

    def test_new_keys_added(self):
        base = {"sample_name": "test"}
        override = {"callers": ["gridss"]}
        merged = merge_configs(base, override)
        assert merged["callers"] == ["gridss"]
