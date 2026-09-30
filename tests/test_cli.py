"""Tests for svensemble.cli: argument parsing, build_cli_workflow_config, --version."""

import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from svensemble.cli import main
from svensemble.utils import build_cli_workflow_config


class TestVersion:
    def test_version_flag(self, capsys):
        with pytest.raises(SystemExit) as exc:
            sys.argv = ["svensemble", "--version"]
            main()
        assert exc.value.code == 0
        captured = capsys.readouterr()
        assert "svensemble" in captured.out
        assert "0." in captured.out

    def test_version_via_module(self):
        result = subprocess.run(
            [sys.executable, "-m", "svensemble", "--version"],
            capture_output=True, text=True,
        )
        assert result.returncode == 0
        assert "svensemble" in result.stdout


class TestBuildCliWorkflowConfig:
    def test_basic_structure(self):
        config = build_cli_workflow_config(
            sample_name="NA12878",
            callers=["gridss", "manta_diploid"],
            model_name="unbiased",
            training_set="gridss+manta_diploid",
        )
        assert config["sample_name"] == "NA12878"
        assert "gridss" in config["callsets"]
        assert "manta_diploid" in config["callsets"]
        assert config["model"]["name"] == "unbiased"
        assert "gridss+manta_diploid" in config["training_sets"]
        assert config["is_reference"] is False

    def test_resolves_callers_from_training_set_preset(self):
        config = build_cli_workflow_config(
            sample_name="test",
            callers=[],
            model_name="unbiased",
            training_set="gridss+manta_diploid",
        )
        assert "gridss" in config["callsets"]
        assert "manta_diploid" in config["callsets"]

    def test_caller_params_resolved(self):
        config = build_cli_workflow_config(
            sample_name="test",
            callers=["gridss"],
            model_name="unbiased",
        )
        params = config["callsets"]["gridss"]["params"]
        assert "filter_cols" in params or "info_cols_pred" in params

    def test_custom_cols_added_for_model(self):
        config = build_cli_workflow_config(
            sample_name="test",
            callers=["gridss"],
            model_name="unbiased_GT",
            training_set="gridss",
        )
        params = config["callsets"]["gridss"]["params"]
        assert params.get("custom_cols_pred") == ["SVTYPE"]

    def test_paths_relative(self):
        config = build_cli_workflow_config(
            sample_name="test",
            callers=["gridss"],
            ref_path=Path("/some/ref.fa"),
        )
        assert config["workflow_dir"] == Path(".")
        assert config["callset_bps_dir"] == Path(".")

    def test_no_callers_empty(self):
        config = build_cli_workflow_config(sample_name="test")
        assert config["callsets"] == {}


class TestSubcommandParsing:
    def test_no_subcommand_shows_help(self, capsys):
        sys.argv = ["svensemble"]
        with pytest.raises(SystemExit) as exc:
            main()
        assert exc.value.code == 0

    def test_preprocess_raw_requires_args(self):
        sys.argv = ["svensemble", "preprocess-raw"]
        with pytest.raises(SystemExit) as exc:
            main()
        assert exc.value.code == 2


class TestTrainSubcommandPooledValidation:
    """--matched_bps and --train_samples/--train_matched_bps are mutually
    exclusive alternatives; exactly one form must be given (see
    svensemble.cli._cmd_train)."""

    def _base_argv(self, output_dir):
        return [
            "svensemble", "train",
            "--model", "unbiased_concat",
            "--training_set", "gridss+manta_diploid",
            "--callers", "gridss", "manta_diploid",
            "--sample", "target_sample",
            "--output_dir", str(output_dir),
        ]

    def test_neither_form_raises(self, tmp_path):
        sys.argv = self._base_argv(tmp_path / "out")
        with pytest.raises(ValueError, match="Either --matched_bps or --train_samples"):
            main()

    def test_matched_bps_and_pooled_together_raises(self, tmp_path):
        fake_csv = tmp_path / "matched_bps.csv"
        fake_csv.write_text("a\n1\n")
        sys.argv = self._base_argv(tmp_path / "out") + [
            "--matched_bps", str(fake_csv),
            "--train_samples", "sampleA",
            "--train_matched_bps", str(fake_csv),
        ]
        with pytest.raises(ValueError, match="cannot be combined"):
            main()

    def test_train_samples_without_train_matched_bps_raises(self, tmp_path):
        sys.argv = self._base_argv(tmp_path / "out") + ["--train_samples", "sampleA", "sampleB"]
        with pytest.raises(ValueError, match="must be provided together"):
            main()

    def test_train_matched_bps_without_train_samples_raises(self, tmp_path):
        fake_csv = tmp_path / "matched_bps.csv"
        fake_csv.write_text("a\n1\n")
        sys.argv = self._base_argv(tmp_path / "out") + ["--train_matched_bps", str(fake_csv)]
        with pytest.raises(ValueError, match="must be provided together"):
            main()

    def test_mismatched_lengths_raises(self, tmp_path):
        csv1 = tmp_path / "a.csv"
        csv1.write_text("a\n1\n")
        sys.argv = self._base_argv(tmp_path / "out") + [
            "--train_samples", "sampleA", "sampleB",
            "--train_matched_bps", str(csv1),
        ]
        with pytest.raises(ValueError, match="same length"):
            main()

    def test_valid_pooled_combo_dispatches_matched_bps_sources(self, tmp_path, monkeypatch):
        csv1 = tmp_path / "a.csv"
        csv2 = tmp_path / "b.csv"
        csv1.write_text("a\n1\n")
        csv2.write_text("a\n1\n")

        captured = {}

        def fake_train_model(
            joint_bp_df, workflow_config, training_set_name,
            verbose=False, matched_bps_sources=None,
        ):
            captured["joint_bp_df"] = joint_bp_df
            captured["matched_bps_sources"] = matched_bps_sources
            return pd.DataFrame({"f": [1, 2]}), None

        monkeypatch.setattr("svensemble.models.train_model", fake_train_model)

        sys.argv = self._base_argv(tmp_path / "out") + [
            "--train_samples", "sampleA", "sampleB",
            "--train_matched_bps", str(csv1), str(csv2),
        ]
        main()

        assert captured["joint_bp_df"] is None
        assert captured["matched_bps_sources"] == {"sampleA": csv1, "sampleB": csv2}

    def test_valid_single_matched_bps_dispatches_joint_bp_df(self, tmp_path, monkeypatch):
        csv1 = tmp_path / "matched_bps.csv"
        pd.DataFrame({"a": [1, 2]}).to_csv(csv1, index=False)

        captured = {}

        def fake_train_model(
            joint_bp_df, workflow_config, training_set_name,
            verbose=False, matched_bps_sources=None,
        ):
            captured["joint_bp_df"] = joint_bp_df
            captured["matched_bps_sources"] = matched_bps_sources
            return pd.DataFrame({"f": [1, 2]}), None

        monkeypatch.setattr("svensemble.models.train_model", fake_train_model)

        sys.argv = self._base_argv(tmp_path / "out") + ["--matched_bps", str(csv1)]
        main()

        assert captured["matched_bps_sources"] is None
        assert list(captured["joint_bp_df"]["a"]) == [1, 2]
