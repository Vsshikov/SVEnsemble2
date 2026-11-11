"""Tests for svensemble.cli: argument parsing, build_workflow_config, --version."""

import math
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from svensemble.cli import main
from svensemble._config import build_workflow_config


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
        config = build_workflow_config(
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
        config = build_workflow_config(
            sample_name="test",
            callers=[],
            model_name="unbiased",
            training_set="gridss+manta_diploid",
        )
        assert "gridss" in config["callsets"]
        assert "manta_diploid" in config["callsets"]

    def test_caller_params_resolved(self):
        config = build_workflow_config(
            sample_name="test",
            callers=["gridss"],
            model_name="unbiased",
        )
        params = config["callsets"]["gridss"]["params"]
        assert "filter_cols" in params or "info_cols_pred" in params

    def test_custom_cols_added_for_model(self):
        config = build_workflow_config(
            sample_name="test",
            callers=["gridss"],
            model_name="unbiased_GT",
            training_set="gridss",
        )
        params = config["callsets"]["gridss"]["params"]
        assert params.get("custom_cols_pred") == ["SVTYPE"]

    def test_paths_relative(self):
        config = build_workflow_config(
            sample_name="test",
            callers=["gridss"],
        )
        assert config["workflow_dir"] == Path(".")
        assert config["callset_bps_dir"] == Path(".")

    def test_no_callers_empty(self):
        config = build_workflow_config(sample_name="test")
        assert config["callsets"] == {}


class TestBuildCliWorkflowConfigGroundTruth:
    """build_workflow_config with ground_truth_name -- the path exercised
    by the Nextflow match and train steps."""

    def test_ground_truth_embedded_in_config(self):
        config = build_workflow_config(
            sample_name="NA12878",
            callers=["gridss", "manta_diploid"],
            model_name="unbiased_xgb",
            training_set="gridss+manta_diploid",
            ground_truth_name="gnomad",
        )
        gt = config["ground_truth"]
        assert gt["name"] == "gnomad"
        assert "params" in gt

    def test_ground_truth_with_bps_path(self):
        config = build_workflow_config(
            sample_name="NA12878",
            callers=["gridss"],
            ground_truth_name="gnomad",
            ground_truth_bps_path=Path("ground_truth.bps.csv"),
        )
        assert config["ground_truth"]["bps_path"] == Path("ground_truth.bps.csv")

    def test_no_ground_truth_when_omitted(self):
        config = build_workflow_config(
            sample_name="NA12878",
            callers=["gridss"],
        )
        assert "ground_truth" not in config

    def test_unknown_ground_truth_raises(self):
        with pytest.raises(ValueError, match="Unknown ground truth name"):
            build_workflow_config(
                sample_name="NA12878",
                callers=["gridss"],
                ground_truth_name="definitely_not_a_real_dataset",
            )


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

    def test_preprocess_raw_needs_no_ref(self, tmp_path):
        from .conftest import write_delly_vcf
        vcf = tmp_path / "delly.vcf"
        write_delly_vcf(vcf)
        sys.argv = ["svensemble", "preprocess-raw", "--vcf", str(vcf), "--caller", "delly",
                    "--output_dir", str(tmp_path)]
        main()
        out = pd.read_csv(tmp_path / "delly.raw_bps.csv")
        assert len(out) == 5


class TestModelIdSubcommand:
    def test_all_lists_every_preset(self, capsys):
        from svensemble import api

        sys.argv = ["svensemble", "model-id", "--all"]
        main()
        lines = capsys.readouterr().out.splitlines()
        assert dict(line.split("\t") for line in lines) == api.list_preset_ids()

    @pytest.mark.parametrize("other", [["--model", "unbiased_xgb"], ["--model_config", "cfg.json"]])
    def test_all_excludes_model_selection(self, other):
        sys.argv = ["svensemble", "model-id", "--all", *other]
        with pytest.raises(SystemExit) as exc:
            main()
        assert exc.value.code == 2

    def test_preset_prints_id_only(self, capsys):
        sys.argv = ["svensemble", "model-id", "--model", "unbiased_xgb"]
        main()
        assert capsys.readouterr().out == "unbiased_xgb-855c4c13\n"


class TestShapSubcommand:
    @pytest.fixture
    def captured(self, monkeypatch):
        pytest.importorskip("shap")
        captured = {}
        monkeypatch.setattr("svensemble.sv_shap.run_shap", lambda **kw: captured.update(kw))
        return captured

    def _argv(self, tmp_path):
        return [
            "svensemble", "shap",
            "--model_pkl", str(tmp_path / "model.pkl"),
            "--feature_matrix", str(tmp_path / "feature_matrix.csv"),
            "--sample", "s", "--model", "unbiased_xgb", "--training_set", "ts",
            "--output_dir", str(tmp_path / "shap"),
        ]

    def test_new_flags_default_off(self, tmp_path, captured):
        sys.argv = self._argv(tmp_path)
        main()
        assert (captured["use_gpu"], captured["max_samples"], captured["top_n"]) == (False, None, 20)

    def test_new_flags_passed_through(self, tmp_path, captured):
        sys.argv = self._argv(tmp_path) + ["--use_gpu", "--max_samples", "50", "--top_n", "7"]
        main()
        assert (captured["use_gpu"], captured["max_samples"], captured["top_n"]) == (True, 50, 7)


class TestValidationSetSubcommand:
    def test_dispatches_to_api(self, tmp_path, monkeypatch):
        captured = {}
        monkeypatch.setattr(
            "svensemble.api.preprocess_validation_set",
            lambda **kw: captured.update(kw) or kw["output_dir"],
        )
        sys.argv = [
            "svensemble", "validation-set", "--val_set", "hg002_val",
            "--raw", str(tmp_path / "raw.vcf.gz"), "--output_dir", str(tmp_path / "out"), "--overwrite",
        ]
        main()
        assert captured == {
            "val_set": "hg002_val", "raw": tmp_path / "raw.vcf.gz",
            "output_dir": tmp_path / "out", "overwrite": True,
        }

    def test_unknown_name_is_a_usage_error(self, tmp_path):
        sys.argv = ["svensemble", "validation-set", "--val_set", "nope", "--raw", "x", "--output_dir", "y"]
        with pytest.raises(SystemExit) as exc:
            main()
        assert exc.value.code == 2


class TestGroundTruthSubcommand:
    def test_space_separated_lists_and_renamed_flags(self, tmp_path, monkeypatch):
        captured = {}
        monkeypatch.setattr("svensemble.api.preprocess_ground_truth", lambda **kw: captured.update(kw))
        sys.argv = [
            "svensemble", "ground-truth", "--vcf", str(tmp_path / "gt.vcf.gz"),
            "--output_dir", str(tmp_path / "gt"), "--ground_truth_name", "gnomad_pass",
            "--filters", "PASS", "LowQual", "--evidence", "SR", "PE",
            "--svtypes", "DEL", "INS", "--algorithms", "manta", "depth",
            "--svlen_min", "100", "--svlen_max", "5000", "--overwrite",
        ]
        main()
        assert captured["vcf"] == tmp_path / "gt.vcf.gz"
        assert captured["output_dir"] == tmp_path / "gt"
        assert captured["filters"] == ["PASS", "LowQual"]
        assert captured["evidence"] == ["SR", "PE"]
        assert captured["svtypes"] == ["DEL", "INS"]
        assert captured["algorithms"] == ["manta", "depth"]
        assert captured["svlens"] == (100, 5000.0)
        assert captured["overwrite"] is True

    def test_defaults(self, tmp_path, monkeypatch):
        captured = {}
        monkeypatch.setattr("svensemble.api.preprocess_ground_truth", lambda **kw: captured.update(kw))
        sys.argv = ["svensemble", "ground-truth", "--vcf", "gt.vcf", "--output_dir", str(tmp_path)]
        main()
        assert captured["filters"] is None and captured["svtypes"] is None
        assert captured["svlens"] == (50, math.inf)
        assert captured["overwrite"] is False


class TestRunSubcommand:
    def _argv(self, tmp_path, *extra):
        return [
            "svensemble", "run",
            "--callers", "gridss", "manta_diploid",
            "--caller_vcfs", str(tmp_path / "g.vcf"), str(tmp_path / "m.vcf"),
            "--ref", str(tmp_path / "ref.fa"), "--model", "unbiased_xgb",
            "--training_set", "gridss+manta_diploid",
            "--ground_truth_bps", str(tmp_path / "gt.csv"),
            *extra,
        ]

    @pytest.fixture
    def captured(self, monkeypatch):
        captured = {}
        monkeypatch.setattr("svensemble.api.run", lambda **kw: captured.update(kw))
        return captured

    def test_workdir_optional(self, tmp_path, captured):
        sys.argv = self._argv(tmp_path, "--output_dir", str(tmp_path / "out"))
        main()
        assert captured["workdir"] is None
        assert captured["keep_workdir"] is False
        assert captured["save"] == set()
        assert captured["caller_vcfs"] == {"gridss": (tmp_path / "g.vcf",), "manta_diploid": (tmp_path / "m.vcf",)}
        assert captured["annotate_vcfs"] is False
        assert captured["output_dir"] == tmp_path / "out"
        assert " run " in captured["command"] and "--output_dir" in captured["command"]

    def test_repeated_caller_and_annotate_flag(self, tmp_path, captured):
        sys.argv = [
            "svensemble", "run",
            "--callers", "delly", "delly",
            "--caller_vcfs", str(tmp_path / "a.vcf"), str(tmp_path / "b.vcf"),
            "--ref", str(tmp_path / "ref.fa"), "--model", "unbiased_xgb",
            "--ground_truth_bps", str(tmp_path / "gt.csv"),
            "--output_dir", str(tmp_path / "out"), "--annotate_vcfs",
        ]
        main()
        assert captured["caller_vcfs"] == {"delly": (tmp_path / "a.vcf", tmp_path / "b.vcf")}
        assert captured["annotate_vcfs"] is True

    def test_an_output_required(self, tmp_path, captured):
        sys.argv = self._argv(tmp_path)
        with pytest.raises(SystemExit, match="--output_vcf"):
            main()
        assert captured == {}

    def test_output_vcf_alone(self, tmp_path, captured):
        sys.argv = self._argv(tmp_path, "--output_vcf", str(tmp_path / "x" / "calls.vcf.gz"))
        main()
        assert captured["output_vcf"] == tmp_path / "x" / "calls.vcf.gz"
        assert captured["output_dir"] is None

    def test_save_flags_map_to_save_set(self, tmp_path, captured):
        sys.argv = self._argv(
            tmp_path, "--output_dir", str(tmp_path / "out"), "--workdir", str(tmp_path / "wd"),
            "--save_model", "--save_feature_matrix", "--save_matched_bps", "--save_callsets",
            "--keep_workdir", "--force_preprocess_ground_truth",
        )
        main()
        assert captured["save"] == {"model", "feature_matrix", "matched_bps", "callsets"}
        assert captured["workdir"] == tmp_path / "wd"
        assert captured["keep_workdir"] is True
        assert captured["force_preprocess_ground_truth"] is True

    def test_mismatched_callers_and_vcfs(self, tmp_path, captured):
        argv = self._argv(tmp_path, "--output_dir", str(tmp_path / "out"))
        argv.remove(str(tmp_path / "m.vcf"))
        sys.argv = argv
        with pytest.raises(SystemExit, match="same length"):
            main()


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

    def test_ground_truth_propagated_to_workflow_config(self, tmp_path, monkeypatch):
        """The Nextflow train step always passes --ground_truth_name; verify
        it ends up in workflow_config['ground_truth'] so _get_ground_truth_name
        can extract it at training time."""
        csv1 = tmp_path / "matched_bps.csv"
        pd.DataFrame({"a": [1]}).to_csv(csv1, index=False)

        captured = {}

        def fake_train_model(
            joint_bp_df, workflow_config, training_set_name,
            verbose=False, matched_bps_sources=None,
        ):
            captured["workflow_config"] = workflow_config
            return pd.DataFrame({"f": [1]}), None

        monkeypatch.setattr("svensemble.models.train_model", fake_train_model)

        sys.argv = self._base_argv(tmp_path / "out") + [
            "--matched_bps", str(csv1),
            "--ground_truth_name", "gnomad",
        ]
        main()

        gt = captured["workflow_config"]["ground_truth"]
        assert gt["name"] == "gnomad"
        assert "params" in gt


class TestTrainingSetDefault:
    """--training_set is optional on train, predict and run; it defaults to
    --callers sorted and joined with '+'."""

    def test_train_without_training_set(self, tmp_path, monkeypatch):
        csv1 = tmp_path / "matched_bps.csv"
        pd.DataFrame({"a": [1]}).to_csv(csv1, index=False)
        captured = {}

        def fake_train_model(
            joint_bp_df, workflow_config, training_set_name,
            verbose=False, matched_bps_sources=None,
        ):
            captured["training_set_name"] = training_set_name
            return pd.DataFrame({"f": [1]}), None

        monkeypatch.setattr("svensemble.models.train_model", fake_train_model)
        sys.argv = [
            "svensemble", "train",
            "--matched_bps", str(csv1),
            "--model", "unbiased_xgb",
            "--callers", "manta_diploid", "delly",
            "--sample", "s",
            "--output_dir", str(tmp_path / "out"),
        ]
        main()
        assert captured["training_set_name"] == "delly+manta_diploid"

    def test_predict_passes_callers_for_default(self, tmp_path, monkeypatch):
        captured = {}

        def fake_predict(*args, **kwargs):
            captured.update(args=args, kwargs=kwargs)
            return pd.DataFrame()

        monkeypatch.setattr("svensemble.api.predict", fake_predict)
        monkeypatch.setattr("svensemble.io.read_model", lambda p: None)
        monkeypatch.setattr("svensemble.io.read_feature_matrix", lambda p: None)
        monkeypatch.setattr("svensemble.io.read_bps", lambda p: None)
        monkeypatch.setattr("svensemble.io.write_calls", lambda df, d, s, **k: (d / "c.csv", d / "c.vcf"))
        sys.argv = [
            "svensemble", "predict",
            "--model_pkl", "m.pkl", "--feature_matrix", "fm.csv", "--matched_bps", "mb.csv",
            "--sample", "s", "--callers", "manta_diploid", "delly",
            "--model", "unbiased_xgb", "--output_dir", str(tmp_path),
        ]
        main()
        assert captured["args"][4] is None
        assert captured["kwargs"]["callers"] == ["manta_diploid", "delly"]


class TestMatchAndPredictIdGraph:
    def test_match_writes_id_graph_next_to_matched_bps(self, tmp_path):
        # Feeds two callers' breakpoints in S1 and S2 plus a ground truth;
        # expects matched_bps.csv and an id_graph/ directory in --output_dir
        # whose node SV IDs are the ones in the matched_bps.csv column.
        from svensemble import io as sv_io
        from svensemble._constants import SAMPLE_ID_COL
        from svensemble.id_graph import node_table

        from .test_chunk_match import _bps, _row

        bps = _bps(
            _row('delly', 'ev1', 1000, 1500, sample='S1'),
            _row('delly', 'ev1', 1000, 1500, sample='S2'),
            _row('sniffles', 'evA', 1003, 1503, sample='S1'),
            _row('sniffles', 'evB', 1002, 1502, sample='S2'),
            _row('gnomad', 'g1', 1001, 1501, sample='S1'),
        )
        paths = {}
        for caller in ('delly', 'sniffles'):
            paths[caller] = sv_io.write_bps(bps[caller], tmp_path / f'{caller}.bps.csv')
        gt = sv_io.write_bps(bps['gnomad'].drop(columns=[SAMPLE_ID_COL]), tmp_path / 'gt.bps.csv')
        out = tmp_path / 'out'
        sys.argv = [
            "svensemble", "match", "--bps", str(paths['delly']), str(paths['sniffles']),
            "--callers", "delly", "sniffles", "--ground_truth_bps", str(gt), "--sample", "S",
            "--max_dist", "10", "--output_dir", str(out),
        ]
        main()
        assert sorted(p.name for p in out.iterdir()) == [sv_io.ID_GRAPH, sv_io.MATCHED_BPS]
        assert sorted(p.name for p in (out / sv_io.ID_GRAPH).iterdir()) == ['edges.csv', 'meta.json', 'nodes.csv']
        matched = sv_io.read_bps(out / sv_io.MATCHED_BPS)
        nodes = node_table(sv_io.read_id_graph(out / sv_io.ID_GRAPH))
        assert set(matched['svensemble_id']) == set(nodes['svensemble_id'].dropna())
        assert matched['svensemble_id'].nunique() == 1 and sorted(matched[SAMPLE_ID_COL]) == ['S1', 'S2']

    def test_predict_has_no_id_graph_option(self, tmp_path):
        # predict reads the SV IDs from the matched_bps column, so --id_graph
        # is an unknown argument (usage error, exit code 2).
        sys.argv = [
            "svensemble", "predict",
            "--model_pkl", "m.pkl", "--feature_matrix", "fm.csv", "--matched_bps", "mb.csv",
            "--sample", "s", "--callers", "delly", "--model", "unbiased_xgb",
            "--output_dir", str(tmp_path), "--id_graph", str(tmp_path / "id_graph"),
        ]
        with pytest.raises(SystemExit) as exc:
            main()
        assert exc.value.code == 2
