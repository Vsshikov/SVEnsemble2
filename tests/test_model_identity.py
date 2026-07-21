"""Tests for the reconstructible model identity system:
resolve_model_config (two-branch), model_config_hash, model_id, and the
model-id CLI subcommand.
"""

import subprocess
import sys
from pathlib import Path

import pytest

from svensemble import utils as sv_utils
from svensemble import _model_configs


# ---------------------------------------------------------------------------
# Hasher: determinism, reconstructibility, what it does / doesn't cover
# ---------------------------------------------------------------------------

class TestModelConfigHash:
    def test_deterministic_across_calls(self):
        cfg = sv_utils.resolve_model_config(name="unbiased_xgb")
        assert sv_utils.model_config_hash(cfg) == sv_utils.model_config_hash(cfg)

    def test_reconstructible_across_processes(self):
        # A fresh interpreter must produce the same hash for the same config.
        code = (
            "from svensemble import utils as u;"
            "print(u.model_config_hash(u.resolve_model_config(name='unbiased_xgb')))"
        )
        out = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, check=True
        ).stdout.strip()
        assert out == sv_utils.model_config_hash(
            sv_utils.resolve_model_config(name="unbiased_xgb")
        )

    def test_labels_excluded_from_hash(self):
        base = sv_utils.resolve_model_config(name="unbiased_xgb")
        relabelled = dict(base)
        relabelled["name"] = "totally_different_label"
        relabelled["description"] = "changed"
        assert sv_utils.model_config_hash(relabelled) == sv_utils.model_config_hash(base)

    def test_behavioural_change_changes_hash(self):
        base = sv_utils.resolve_model_config(name="unbiased_xgb")
        changed = sv_utils.resolve_model_config(
            config={"base": "unbiased_xgb", "training_params": {"model_kwargs": {"max_depth": 31}}}
        )
        assert sv_utils.model_config_hash(changed) != sv_utils.model_config_hash(base)

    def test_pollution_free_with_full_model_block(self):
        # A full workflow_config['model'] sub-dict (with model_id + *_path) must
        # hash identically to the bare resolved config.
        cfg = sv_utils.build_cli_workflow_config(
            sample_name="s", callers=["gridss"], model_name="unbiased_xgb",
            training_set="gridss",
        )["model"]
        bare = sv_utils.resolve_model_config(name="unbiased_xgb")
        assert sv_utils.model_config_hash(cfg) == sv_utils.model_config_hash(bare)


# ---------------------------------------------------------------------------
# model_id format
# ---------------------------------------------------------------------------

class TestModelId:
    def test_format_base_dash_hash8(self):
        mid = sv_utils.model_id(sv_utils.resolve_model_config(name="unbiased_xgb"))
        base, _, h = mid.partition("-")
        assert base == "unbiased_xgb"
        assert len(h) == 8 and h.isalnum()

    def test_alias_collapses_to_base(self):
        # 'unbiased' has base 'unbiased_rf' and identical params -> same id.
        assert sv_utils.model_id(sv_utils.resolve_model_config(name="unbiased")) == \
            sv_utils.model_id(sv_utils.resolve_model_config(name="unbiased_rf"))

    def test_preset_ids_cover_all_presets(self):
        ids = sv_utils.list_preset_ids()
        assert set(ids) == set(_model_configs.model_configs)
        for name, mid in ids.items():
            assert mid.startswith(_model_configs.model_configs[name]["base"] + "-")


# ---------------------------------------------------------------------------
# Two-branch resolver
# ---------------------------------------------------------------------------

class TestResolveModelConfig:
    def test_branch1_named_preset(self):
        cfg = sv_utils.resolve_model_config(name="unbiased_GT")
        assert cfg["training_params"]["custom_cols"] == ["SVTYPE"]
        assert cfg["base"] == "unbiased_rf"

    def test_branch2_override_deep_merges(self):
        cfg = sv_utils.resolve_model_config(config={
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
        sv_utils.resolve_model_config(config={
            "base": "unbiased_xgb",
            "training_params": {"model_kwargs": {"max_depth": 99}},
        })
        assert _model_configs.model_configs["unbiased_xgb"]["training_params"] == before

    def test_unknown_name_raises(self):
        with pytest.raises(ValueError, match="Unknown model preset"):
            sv_utils.resolve_model_config(name="does_not_exist")

    def test_unknown_base_raises(self):
        with pytest.raises(ValueError, match="Unknown base preset"):
            sv_utils.resolve_model_config(config={"base": "does_not_exist"})

    def test_missing_base_raises(self):
        with pytest.raises(ValueError, match="must specify a 'base'"):
            sv_utils.resolve_model_config(config={"training_params": {}})

    def test_bad_function_raises(self):
        with pytest.raises(ValueError, match="not a known training function"):
            sv_utils.resolve_model_config(config={
                "base": "unbiased_xgb",
                "training_params": {"function": "not_a_real_function"},
            })


# ---------------------------------------------------------------------------
# Fetch-by-config helper
# ---------------------------------------------------------------------------

class TestModelResultsDir:
    def test_matches_create_config_dict(self):
        cd = sv_utils.create_config_dict(
            "NA12878", model_name="unbiased_xgb", training_set="default"
        )["NA12878"]
        frd = sv_utils.model_results_dir(
            sv_utils.BASE_DIR, "NA12878", "unbiased_xgb", "default"
        )
        assert frd == cd["workflow_dir"]

    def test_accepts_override_dict(self):
        frd_name = sv_utils.model_results_dir(
            sv_utils.BASE_DIR, "NA12878", "unbiased_xgb", "default"
        )
        frd_cfg = sv_utils.model_results_dir(
            sv_utils.BASE_DIR, "NA12878", {"base": "unbiased_xgb"}, "default"
        )
        assert frd_name == frd_cfg


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
        expected = sv_utils.model_id(sv_utils.resolve_model_config(name="unbiased_xgb"))
        res = self._run("--model", "unbiased_xgb")
        assert res.returncode == 0, res.stderr
        assert res.stdout.strip() == expected

    def test_model_config_matches_python_model_id(self, tmp_path):
        import json
        cfg = {"base": "unbiased_xgb", "training_params": {"model_kwargs": {"max_depth": 31}}}
        cfg_path = tmp_path / "override.json"
        cfg_path.write_text(json.dumps(cfg))
        expected = sv_utils.model_id(sv_utils.resolve_model_config(config=cfg))
        res = self._run("--model_config", str(cfg_path))
        assert res.returncode == 0, res.stderr
        assert res.stdout.strip() == expected

    def test_missing_args_errors(self):
        res = self._run()
        assert res.returncode != 0
        assert "Provide --model" in res.stderr
