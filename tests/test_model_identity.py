"""Tests for the reconstructible model identity system:
resolve_model_config (two-branch), model_config_hash, model_id, and the
model-id CLI subcommand.
"""

import subprocess
import sys
from pathlib import Path

import pytest

from svensemble import _config as sv_config
from svensemble import _model_configs


# ---------------------------------------------------------------------------
# Hasher: determinism, reconstructibility, what it does / doesn't cover
# ---------------------------------------------------------------------------

class TestModelConfigHash:
    def test_deterministic_across_calls(self):
        cfg = sv_config.resolve_model_config(name="unbiased_xgb")
        assert sv_config.model_config_hash(cfg) == sv_config.model_config_hash(cfg)

    def test_reconstructible_across_processes(self):
        # A fresh interpreter must produce the same hash for the same config.
        code = (
            "from svensemble import _config as c;"
            "print(c.model_config_hash(c.resolve_model_config(name='unbiased_xgb')))"
        )
        out = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, check=True
        ).stdout.strip()
        assert out == sv_config.model_config_hash(
            sv_config.resolve_model_config(name="unbiased_xgb")
        )

    def test_labels_excluded_from_hash(self):
        base = sv_config.resolve_model_config(name="unbiased_xgb")
        relabelled = dict(base)
        relabelled["name"] = "totally_different_label"
        relabelled["description"] = "changed"
        assert sv_config.model_config_hash(relabelled) == sv_config.model_config_hash(base)

    def test_behavioural_change_changes_hash(self):
        base = sv_config.resolve_model_config(name="unbiased_xgb")
        changed = sv_config.resolve_model_config(
            config={"base": "unbiased_xgb", "training_params": {"model_kwargs": {"max_depth": 31}}}
        )
        assert sv_config.model_config_hash(changed) != sv_config.model_config_hash(base)

    def test_pollution_free_with_full_model_block(self):
        # A full workflow_config['model'] sub-dict (with model_id + *_path) must
        # hash identically to the bare resolved config.
        cfg = sv_config.build_workflow_config(
            sample_name="s", callers=["gridss"], model_name="unbiased_xgb",
            training_set="gridss",
        )["model"]
        bare = sv_config.resolve_model_config(name="unbiased_xgb")
        assert sv_config.model_config_hash(cfg) == sv_config.model_config_hash(bare)


# ---------------------------------------------------------------------------
# model_id format
# ---------------------------------------------------------------------------

class TestModelId:
    def test_format_base_dash_hash8(self):
        mid = sv_config.model_id(sv_config.resolve_model_config(name="unbiased_xgb"))
        base, _, h = mid.partition("-")
        assert base == "unbiased_xgb"
        assert len(h) == 8 and h.isalnum()

    def test_alias_collapses_to_base(self):
        # 'unbiased' has base 'unbiased_rf' and identical params -> same id.
        assert sv_config.model_id(sv_config.resolve_model_config(name="unbiased")) == \
            sv_config.model_id(sv_config.resolve_model_config(name="unbiased_rf"))

    def test_preset_ids_cover_all_presets(self):
        ids = sv_config.list_preset_ids()
        assert set(ids) == set(_model_configs.model_configs)
        for name, mid in ids.items():
            assert mid.startswith(_model_configs.model_configs[name]["base"] + "-")


# ---------------------------------------------------------------------------
# Two-branch resolver
# ---------------------------------------------------------------------------

class TestResolveModelConfig:
    def test_branch1_named_preset(self):
        cfg = sv_config.resolve_model_config(name="unbiased_GT")
        assert cfg["training_params"]["custom_cols"] == ["SVTYPE"]
        assert cfg["base"] == "unbiased_rf"

    def test_branch2_override_deep_merges(self):
        cfg = sv_config.resolve_model_config(config={
            "base": "unbiased_xgb",
            "name": "xgb_d31",
            "training_params": {"model_kwargs": {"max_depth": 31}},
        })
        assert cfg["base"] == "unbiased_xgb"
        assert cfg["name"] == "xgb_d31"
        assert cfg["training_params"]["model_kwargs"] == {"max_depth": 31}
        assert cfg["training_params"]["function"] == "unbiased_xgb"  # inherited

    def test_branch2_does_not_mutate_registry(self):
        before = dict(_model_configs.model_configs["unbiased_xgb"]["training_params"])
        sv_config.resolve_model_config(config={
            "base": "unbiased_xgb",
            "training_params": {"model_kwargs": {"max_depth": 99}},
        })
        assert _model_configs.model_configs["unbiased_xgb"]["training_params"] == before

    def test_unknown_name_raises(self):
        with pytest.raises(ValueError, match="Unknown model preset"):
            sv_config.resolve_model_config(name="does_not_exist")

    def test_unknown_base_raises(self):
        with pytest.raises(ValueError, match="Unknown base preset"):
            sv_config.resolve_model_config(config={"base": "does_not_exist"})

    def test_missing_base_raises(self):
        with pytest.raises(ValueError, match="must specify a 'base'"):
            sv_config.resolve_model_config(config={"training_params": {}})

    def test_bad_function_raises(self):
        with pytest.raises(ValueError, match="not a known training function"):
            sv_config.resolve_model_config(config={
                "base": "unbiased_xgb",
                "training_params": {"function": "not_a_real_function"},
            })


# ---------------------------------------------------------------------------
# Fetch-by-config helper
# ---------------------------------------------------------------------------

class TestGroundTruthConfig:
    def test_build_ground_truth_config_structure(self):
        gt = sv_config.build_ground_truth_config("gnomad")
        assert gt["name"] == "gnomad"
        assert "params" in gt
        assert gt["vcf_path"] is None
        assert gt["bps_path"] is None

    def test_build_ground_truth_config_with_paths(self):
        gt = sv_config.build_ground_truth_config(
            "gnomad", bps_path=Path("gt.bps.csv"),
        )
        assert gt["bps_path"] == Path("gt.bps.csv")

    def test_round_trip_through_workflow_config(self):
        """The pipeline chain: build_workflow_config embeds ground_truth,
        then models._get_ground_truth_name extracts it."""
        from svensemble.models import _get_ground_truth_name
        config = sv_config.build_workflow_config(
            sample_name="s",
            callers=["gridss"],
            model_name="unbiased_xgb",
            training_set="gridss",
            ground_truth_name="gnomad_pass",
        )
        assert _get_ground_truth_name(config) == "gnomad_pass"

    def test_get_ground_truth_name_missing_raises(self):
        from svensemble.models import _get_ground_truth_name
        config = sv_config.build_workflow_config(
            sample_name="s", callers=["gridss"],
        )
        with pytest.raises(KeyError):
            _get_ground_truth_name(config)

    def test_unknown_ground_truth_name_raises(self):
        with pytest.raises(ValueError, match="Unknown ground truth name"):
            sv_config.build_ground_truth_config("not_a_real_dataset")


class TestTagConstructor:
    """_tag_constructor must reproduce benchmarkFilterTag (nf-core-svensemble),
    which names the published benchmark filter directories."""

    @pytest.mark.parametrize("tag_dict, expected", [
        ({}, "unfiltered"),
        ({"svtypes": None, "svlenmin": None, "svlenmax": None}, "unfiltered"),
        ({"svlenmin": 50}, "svlenmin-50"),
        ({"svlenmin": 50.0}, "svlenmin-50"),
        ({"svlenmin": 50, "svlenmax": float("inf")}, "svlenmin-50"),
        ({"svtypes": "INS+DEL"}, "svtypes-DEL+INS"),
        ({"svtypes": ["INS", "DEL"]}, "svtypes-DEL+INS"),
        ({"svtypes": "DEL"}, "svtypes-DEL"),
        ({"svtypes": []}, "unfiltered"),
        ({"svtypes": ["DEL", "INS"], "svlenmin": 50, "svlenmax": 1000},
         "svlenmax-1000.svlenmin-50.svtypes-DEL+INS"),
    ])
    def test_matches_pipeline_filter_tag(self, tag_dict, expected):
        assert sv_config._tag_constructor(tag_dict) == expected

    def test_reproduces_published_dir_names(self):
        # Directory names present under results/*/svensemble/benchmark/<sample>/<val_set>/.
        assert sv_config._tag_constructor({"svlenmin": 50}) == "svlenmin-50"
        assert sv_config._tag_constructor(
            {"svtypes": "DEL+INS", "svlenmin": 50}
        ) == "svlenmin-50.svtypes-DEL+INS"


# ---------------------------------------------------------------------------
# model-id CLI
# ---------------------------------------------------------------------------
#
# The `svensemble gen-model-samplesheet` subcommand was retired once the
# nf-core/svensemble pipeline began resolving model_id in-DAG via the
# SVENSEMBLE_MODELID module (which shells to `svensemble model-id` per row).
# The `model-id` primitive itself is exercised below.


class TestModelIdCli:
    def _run(self, *args):
        return subprocess.run(
            [sys.executable, "-m", "svensemble", "model-id", *args],
            capture_output=True, text=True,
        )

    def test_preset_matches_python_model_id(self):
        expected = sv_config.model_id(sv_config.resolve_model_config(name="unbiased_xgb"))
        res = self._run("--model", "unbiased_xgb")
        assert res.returncode == 0, res.stderr
        assert res.stdout.strip() == expected

    def test_model_config_matches_python_model_id(self, tmp_path):
        import json
        cfg = {"base": "unbiased_xgb", "training_params": {"model_kwargs": {"max_depth": 31}}}
        cfg_path = tmp_path / "override.json"
        cfg_path.write_text(json.dumps(cfg))
        expected = sv_config.model_id(sv_config.resolve_model_config(config=cfg))
        res = self._run("--model_config", str(cfg_path))
        assert res.returncode == 0, res.stderr
        assert res.stdout.strip() == expected

    def test_missing_args_errors(self):
        res = self._run()
        assert res.returncode != 0
        assert "Provide --model" in res.stderr
