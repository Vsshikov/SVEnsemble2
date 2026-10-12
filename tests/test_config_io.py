"""Tests for svensemble.config_io: lossless encode/decode, pure I/O, hashing."""

import json
import math
from pathlib import Path

import numpy as np
import pytest

from svensemble.config_io import (
    canonical_json,
    dict_json_decode,
    dict_json_encode,
    dump_config,
    load_config,
    validate_config,
)


# ---------------------------------------------------------------------------
# encode → decode round-trips (lossless)
# ---------------------------------------------------------------------------

class TestEncodeDecode:
    """dict_json_encode → dict_json_decode preserves types exactly."""

    def test_plain_scalars_unchanged(self):
        obj = {"a": 1, "b": 2.5, "c": "hello", "d": True, "e": None}
        assert dict_json_decode(dict_json_encode(obj)) == obj

    def test_path_round_trips(self):
        obj = {"p": Path("/foo/bar.csv")}
        restored = dict_json_decode(dict_json_encode(obj))
        assert restored["p"] == Path("/foo/bar.csv")
        assert isinstance(restored["p"], Path)

    def test_tuple_round_trips(self):
        obj = {"t": (1, "two", 3.0)}
        restored = dict_json_decode(dict_json_encode(obj))
        assert restored["t"] == (1, "two", 3.0)
        assert isinstance(restored["t"], tuple)

    def test_set_round_trips(self):
        obj = {"s": {"b", "a", "c"}}
        restored = dict_json_decode(dict_json_encode(obj))
        assert restored["s"] == {"a", "b", "c"}
        assert isinstance(restored["s"], set)

    def test_frozenset_round_trips(self):
        obj = {"fs": frozenset([3, 1, 2])}
        restored = dict_json_decode(dict_json_encode(obj))
        assert restored["fs"] == frozenset([1, 2, 3])
        assert isinstance(restored["fs"], frozenset)

    def test_inf_round_trips(self):
        obj = {"x": float("inf")}
        restored = dict_json_decode(dict_json_encode(obj))
        assert restored["x"] == float("inf")
        assert math.isinf(restored["x"]) and restored["x"] > 0

    def test_neg_inf_round_trips(self):
        obj = {"x": float("-inf")}
        restored = dict_json_decode(dict_json_encode(obj))
        assert restored["x"] == float("-inf")
        assert math.isinf(restored["x"]) and restored["x"] < 0

    def test_nan_round_trips(self):
        obj = {"x": float("nan")}
        restored = dict_json_decode(dict_json_encode(obj))
        assert math.isnan(restored["x"])

    def test_np_inf_round_trips(self):
        obj = {"x": np.inf}
        restored = dict_json_decode(dict_json_encode(obj))
        assert restored["x"] == float("inf")

    def test_np_float64_round_trips(self):
        obj = {"x": np.float64(3.14)}
        encoded = dict_json_encode(obj)
        assert isinstance(encoded["x"], float)
        assert encoded["x"] == pytest.approx(3.14)

    def test_np_int64_round_trips(self):
        obj = {"x": np.int64(42)}
        encoded = dict_json_encode(obj)
        assert isinstance(encoded["x"], int)
        assert encoded["x"] == 42

    def test_np_ndarray_round_trips(self):
        obj = {"arr": np.array([1, 2, 3])}
        restored = dict_json_decode(dict_json_encode(obj))
        assert isinstance(restored["arr"], np.ndarray)
        np.testing.assert_array_equal(restored["arr"], [1, 2, 3])

    def test_np_ndarray_with_floats(self):
        obj = {"arr": np.array([1.5, 2.5])}
        restored = dict_json_decode(dict_json_encode(obj))
        assert isinstance(restored["arr"], np.ndarray)
        np.testing.assert_array_almost_equal(restored["arr"], [1.5, 2.5])

    def test_nested_structure(self):
        obj = {
            "sample_name": "NA12878",
            "svlens": (50, float("inf")),
            "paths": {"ref": Path("/ref.fa"), "bed": Path("/bed.gz")},
            "callers": {"gridss", "delly"},
        }
        restored = dict_json_decode(dict_json_encode(obj))
        assert restored["sample_name"] == "NA12878"
        assert restored["svlens"] == (50, float("inf"))
        assert isinstance(restored["svlens"], tuple)
        assert restored["paths"]["ref"] == Path("/ref.fa")
        assert restored["callers"] == {"gridss", "delly"}

    def test_list_of_paths(self):
        obj = {"files": [Path("/a"), Path("/b")]}
        restored = dict_json_decode(dict_json_encode(obj))
        assert restored["files"] == [Path("/a"), Path("/b")]

    def test_empty_containers(self):
        obj = {"d": {}, "l": [], "t": (), "s": set()}
        restored = dict_json_decode(dict_json_encode(obj))
        assert restored["d"] == {}
        assert restored["l"] == []
        assert restored["t"] == ()
        assert restored["s"] == set()


# ---------------------------------------------------------------------------
# Encoding detail checks
# ---------------------------------------------------------------------------

class TestEncodeDetail:
    """Verify the structure of encoded output."""

    def test_envelope_shape(self):
        encoded = dict_json_encode(Path("/foo"))
        assert encoded == {"__type__": "path", "__value__": "/foo"}

    def test_set_encoded_sorted(self):
        encoded = dict_json_encode({"c", "a", "b"})
        assert encoded["__value__"] == ["a", "b", "c"]

    def test_tuple_encoded_as_list(self):
        encoded = dict_json_encode((1, 2))
        assert encoded == {"__type__": "tuple", "__value__": [1, 2]}

    def test_finite_float_passthrough(self):
        assert dict_json_encode(3.14) == 3.14

    def test_bool_passthrough(self):
        assert dict_json_encode(True) is True

    def test_none_passthrough(self):
        assert dict_json_encode(None) is None

    def test_unknown_marker_passthrough(self):
        """Envelopes with unknown markers survive decode unchanged."""
        obj = {"__type__": "unknown_type", "__value__": 42}
        assert dict_json_decode(obj) == obj


# ---------------------------------------------------------------------------
# dump_config / load_config — pure I/O
# ---------------------------------------------------------------------------

class TestDumpLoad:
    """dump_config and load_config are pure I/O — no encoding applied."""

    def test_json_round_trip(self, tmp_path):
        config = {"sample_name": "NA12878", "callers": ["gridss", "delly"]}
        path = tmp_path / "config.json"
        dump_config(config, path)
        loaded = load_config(path)
        assert loaded == config

    def test_yaml_round_trip(self, tmp_path):
        config = {"sample_name": "NA12878", "count": 5}
        path = tmp_path / "config.yaml"
        dump_config(config, path)
        loaded = load_config(path)
        assert loaded == config

    def test_yml_extension(self, tmp_path):
        config = {"sample_name": "test"}
        path = tmp_path / "config.yml"
        dump_config(config, path)
        loaded = load_config(path)
        assert loaded == config

    def test_creates_parent_dirs(self, tmp_path):
        path = tmp_path / "sub" / "dir" / "config.json"
        dump_config({"sample_name": "test"}, path)
        assert path.exists()

    def test_encoded_then_dumped_round_trips(self, tmp_path):
        """Full workflow: encode → dump → load → decode."""
        config = {
            "sample_name": "NA12878",
            "svlens": (50, float("inf")),
            "ref": Path("/ref.fa"),
        }
        path = tmp_path / "config.json"
        dump_config(dict_json_encode(config), path)
        loaded = dict_json_decode(load_config(path))
        assert loaded == config
        assert isinstance(loaded["svlens"], tuple)
        assert loaded["svlens"][1] == float("inf")
        assert isinstance(loaded["ref"], Path)

    def test_accepts_path_string(self, tmp_path):
        config = {"sample_name": "test"}
        path = str(tmp_path / "config.json")
        dump_config(config, path)
        loaded = load_config(path)
        assert loaded == config


# ---------------------------------------------------------------------------
# canonical_json — deterministic hashing
# ---------------------------------------------------------------------------

class TestCanonicalJson:
    """canonical_json produces stable, deterministic output."""

    def test_deterministic_across_calls(self):
        obj = {"b": 2, "a": 1}
        assert canonical_json(obj) == canonical_json(obj)

    def test_key_order_irrelevant(self):
        assert canonical_json({"b": 2, "a": 1}) == canonical_json({"a": 1, "b": 2})

    def test_compact_separators(self):
        result = canonical_json({"a": 1})
        assert " " not in result

    def test_inf_does_not_raise(self):
        result = canonical_json({"x": float("inf")})
        parsed = json.loads(result)
        assert parsed["x"]["__type__"] == "inf"

    def test_nan_does_not_raise(self):
        result = canonical_json({"x": float("nan")})
        parsed = json.loads(result)
        assert parsed["x"]["__type__"] == "nan"

    def test_allow_nan_false_rejects_raw(self):
        """If a raw inf somehow bypassed encoding, json.dumps would raise."""
        with pytest.raises((ValueError, OverflowError)):
            json.dumps(float("inf"), allow_nan=False)


# ---------------------------------------------------------------------------
# validate_config
# ---------------------------------------------------------------------------

class TestValidation:
    def test_valid_config(self):
        validate_config({"sample_name": "NA12878"})

    def test_missing_sample_name(self):
        with pytest.raises(ValueError, match="sample_name"):
            validate_config({"model_name": "unbiased"})

    def test_empty_config_fails(self):
        with pytest.raises(ValueError):
            validate_config({})
