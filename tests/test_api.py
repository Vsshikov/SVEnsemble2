"""Tests for svensemble.api steps and the svensemble.io artifact readers/writers."""

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from svensemble import api
from svensemble import io as sv_io
from svensemble._constants import GROUND_TRUTH_SAMPLE, SAMPLE_ID_COL


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


class TestPreprocessRaw:
    def test_unknown_caller_raises(self):
        with pytest.raises(ValueError, match="Unknown caller"):
            api.preprocess_raw(pd.DataFrame(), "no_such_caller")

    def test_caller_without_recipe_raises(self):
        # gnomad has params but is ground truth, not a parse recipe source.
        with pytest.raises(ValueError, match="no parse recipe"):
            api.preprocess_raw(pd.DataFrame(), "gnomad")

    def test_delly_from_vcf(self, tmp_path):
        from .conftest import write_delly_vcf
        vcf = tmp_path / "delly.vcf"
        bnd_records = write_delly_vcf(vcf)
        raw = api.preprocess_raw(sv_io.read_vcf(vcf, "delly"), "delly")
        assert len(raw) == 1 + len(bnd_records)


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

    def test_training_set_defaults_to_sorted_callers(self, captured_preds):
        api.predict(_FakeModel(), _train_shaped_X(), pd.DataFrame(), "unbiased_rf",
                    callers=["manta_diploid", "delly"])
        pred_col = f"delly+manta_diploid_{api.model_id(model='unbiased_rf')}_pred"
        assert pred_col in captured_preds["pred_df"].columns

    def test_needs_training_set_or_callers(self, captured_preds):
        with pytest.raises(ValueError, match="training_set or a non-empty callers"):
            api.predict(_FakeModel(), _train_shaped_X(), pd.DataFrame(), "unbiased_rf")


class TestMatch:
    def test_does_not_modify_input(self, monkeypatch):
        captured = {}

        def fake_outer_join(bp_df_dict, ref_callset, insert_delta_threshold, max_dist, workflow_config):
            captured.update(bp_df_dict=bp_df_dict, ref_callset=ref_callset)
            return pd.DataFrame()

        monkeypatch.setattr("svensemble.chunk_match.outer_join_bps", fake_outer_join)
        caller_bps = {
            "delly": pd.DataFrame({"a": [1], SAMPLE_ID_COL: ["S1"]}),
            "manta_diploid": pd.DataFrame({"a": [2], SAMPLE_ID_COL: ["S1"]}),
        }
        gt = pd.DataFrame({"a": [3]})

        api.match(caller_bps, gt, ground_truth_name="gnomad")

        assert list(caller_bps) == ["delly", "manta_diploid"]
        assert list(captured["bp_df_dict"]) == ["delly", "manta_diploid", "gnomad"]
        assert list(gt.columns) == ["a"]
        assert list(captured["bp_df_dict"]["gnomad"][SAMPLE_ID_COL]) == [GROUND_TRUTH_SAMPLE]
        assert captured["ref_callset"] == "gnomad"

    def test_ground_truth_with_sample_column_rejected(self):
        caller_bps = {"delly": pd.DataFrame({"a": [1], SAMPLE_ID_COL: ["S1"]})}
        with pytest.raises(ValueError, match="sample-less"):
            api.match(caller_bps, pd.DataFrame({"a": [3], SAMPLE_ID_COL: ["S1"]}))

    def test_caller_without_sample_column_rejected(self):
        with pytest.raises(ValueError, match="without a"):
            api.match({"delly": pd.DataFrame({"a": [1]})}, pd.DataFrame({"a": [3]}))

    def test_callers_with_different_samples_rejected(self):
        caller_bps = {
            "delly": pd.DataFrame({"a": [1], SAMPLE_ID_COL: ["S1"]}),
            "manta_diploid": pd.DataFrame({"a": [2], SAMPLE_ID_COL: ["SAMPLE1"]}),
        }
        with pytest.raises(ValueError, match="share one sample set"):
            api.match(caller_bps, pd.DataFrame({"a": [3]}))


class TestTrain:
    @pytest.fixture
    def captured(self, monkeypatch):
        captured = {}

        def fake_train_model(joint_bp_df, workflow_config, training_set_name,
                             verbose=False, matched_bps_sources=None):
            captured.update(joint_bp_df=joint_bp_df, sources=matched_bps_sources,
                            workflow_config=workflow_config, training_set_name=training_set_name)
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

    def test_training_set_defaults_to_sorted_callers(self, captured):
        api.train(pd.DataFrame({"a": [1]}), ["manta_diploid", "delly"], "unbiased_xgb")
        assert captured["training_set_name"] == "delly+manta_diploid"
        assert captured["workflow_config"]["training_sets"] == {
            "delly+manta_diploid": ["delly", "manta_diploid"],
        }

    def test_default_sorts_callers_without_a_preset(self, captured):
        # delly+gridss+svaba is not a TRAINING_SET_PRESETS key.
        api.train(pd.DataFrame({"a": [1]}), ["svaba", "gridss", "delly"], "unbiased_xgb")
        assert captured["workflow_config"]["training_sets"] == {
            "delly+gridss+svaba": ["delly", "gridss", "svaba"],
        }
        assert list(captured["workflow_config"]["callsets"]) == ["delly", "gridss", "svaba"]

    def test_explicit_free_form_name_keeps_caller_order(self, captured):
        api.train(pd.DataFrame({"a": [1]}), ["svaba", "gridss"], "unbiased_xgb", "my_set")
        assert captured["workflow_config"]["training_sets"] == {"my_set": ["svaba", "gridss"]}

    def test_preset_training_set_selects_its_callers(self, captured):
        api.train(pd.DataFrame({"a": [1]}), ["gridss", "manta_diploid", "delly"], "unbiased_xgb",
                  "delly+manta_diploid")
        assert captured["workflow_config"]["training_sets"] == {
            "delly+manta_diploid": ["delly", "manta_diploid"],
        }


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
            api.preprocess_validation_set("nope", tmp_path / "raw.vcf", tmp_path / "out")

    def test_missing_raw_file_rejected(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            api.preprocess_validation_set("kosugi", tmp_path / "raw.bed", tmp_path / "out")


# ---------------------------------------------------------------------------
# api.run: scratch lifecycle, save routing, provenance
# ---------------------------------------------------------------------------

class _PicklableModel:
    """Stand-in fitted model; only needs to survive pickling."""


@pytest.fixture
def fake_pipeline(monkeypatch):
    """Replace the step functions with cheap fakes so run's file handling can be tested.

    Set ``fake_pipeline["fail_in"]`` to a step name to make that step raise.
    """
    from svensemble import vcfwrite

    state = {"fail_in": None, "gt_calls": []}

    def step(name, result):
        def f(*a, **k):
            if state["fail_in"] == name:
                raise RuntimeError(f"injected failure in {name}")
            return result(*a, **k)
        return f

    bps = pd.DataFrame({"INFO_EVENT": ["e1", "e2"], "x": [1, 2]})
    X = pd.DataFrame({"f": [0.1, 0.2]}, index=pd.Index(["e1", "e2"], name="gridss_INFO_EVENT"))

    def fake_gt(**kw):
        state["gt_calls"].append(kw)
        out = Path(kw["output_dir"]) / sv_io.CALLSET_BPS.format(caller=kw["ground_truth_name"])
        sv_io.write_bps(bps, out)
        return out

    def fake_vcf(df, path, **kw):
        Path(path).write_text("##fileformat=VCFv4.5\n")
        return Path(path)

    monkeypatch.setattr(sv_io, "read_vcf", lambda *a, **k: bps)
    monkeypatch.setattr(sv_io, "parse_samples_from_vcf", lambda path: ["S"])
    monkeypatch.setattr(api, "preprocess_raw", step("preprocess_raw", lambda *a, **k: bps))
    monkeypatch.setattr(api, "preprocess", step("preprocess", lambda *a, **k: bps))
    monkeypatch.setattr(api, "match", step("match", lambda *a, **k: bps))
    monkeypatch.setattr(api, "train", step("train", lambda *a, **k: (X, _PicklableModel())))
    monkeypatch.setattr(api, "predict", step("predict", lambda *a, **k: bps.assign(gridss_m_pred=[0.1, 0.2])))
    monkeypatch.setattr(api, "preprocess_ground_truth", fake_gt)
    monkeypatch.setattr(vcfwrite, "write_vcf", fake_vcf)
    return state


@pytest.fixture
def tmpdir_root(tmp_path, monkeypatch):
    """Point $TMPDIR at an empty directory (and drop tempfile's cached choice)."""
    import tempfile

    root = tmp_path / "TMPDIR"
    root.mkdir()
    monkeypatch.setenv("TMPDIR", str(root))
    monkeypatch.setattr(tempfile, "tempdir", None)
    return root


def _run_kwargs(tmp_path, **over):
    gt = tmp_path / "gt.csv"
    if not gt.exists():
        gt.write_text("a\n1\n")
    kw = dict(
        sample="S", caller_vcfs={"gridss": tmp_path / "g.vcf"}, ref=tmp_path / "ref.fa",
        model="unbiased_xgb", training_set="gridss", output_dir=tmp_path / "out",
        ground_truth_bps=gt,
    )
    kw.update(over)
    return kw


def _provenance(path):
    import json
    return json.loads((path / sv_io.RUN_PROVENANCE).read_text())


class TestRunScratch:
    def test_tmp_removed_on_success_and_only_calls_published(self, tmp_path, fake_pipeline, tmpdir_root):
        out = api.run(**_run_kwargs(tmp_path))
        assert list(tmpdir_root.iterdir()) == []
        assert sorted(p.name for p in (tmp_path / "out").iterdir()) == sorted(
            [sv_io.CALLS_CSV, sv_io.CALLS_VCF, sv_io.RUN_PROVENANCE]
        )
        assert set(out) == {"calls_csv", "calls_vcf"}

    def test_tmp_kept_on_failure(self, tmp_path, fake_pipeline, tmpdir_root, capsys):
        fake_pipeline["fail_in"] = "train"
        with pytest.raises(RuntimeError, match="injected failure in train"):
            api.run(**_run_kwargs(tmp_path))
        kept = list(tmpdir_root.iterdir())
        assert len(kept) == 1 and kept[0].name.startswith("svensemble_run_")
        assert (kept[0] / "match" / sv_io.MATCHED_BPS).exists()
        assert f"scratch kept at {kept[0]}" in capsys.readouterr().err
        rec = _provenance(kept[0])
        assert rec["status"] == "failed"
        assert rec["error"] == {"type": "RuntimeError", "message": "injected failure in train"}
        assert _provenance(tmp_path / "out")["status"] == "failed"

    def test_keep_workdir_keeps_tmp(self, tmp_path, fake_pipeline, tmpdir_root):
        api.run(**_run_kwargs(tmp_path), keep_workdir=True)
        kept = list(tmpdir_root.iterdir())
        assert len(kept) == 1
        assert _provenance(kept[0])["status"] == "succeeded"

    def test_explicit_workdir_never_deleted(self, tmp_path, fake_pipeline, tmpdir_root):
        wd = tmp_path / "wd"
        api.run(**_run_kwargs(tmp_path), workdir=wd)
        assert list(tmpdir_root.iterdir()) == []
        assert (wd / "preprocess_raw" / sv_io.RAW_BPS.format(caller="gridss")).exists()
        assert (wd / "preprocess" / "gridss" / sv_io.CALLSET_BPS.format(caller="gridss")).exists()
        assert (wd / "train" / sv_io.MODEL).exists()
        assert _provenance(wd)["parameters"]["workdir_is_tmp"] is False

    def test_explicit_workdir_kept_on_failure(self, tmp_path, fake_pipeline, tmpdir_root, capsys):
        fake_pipeline["fail_in"] = "predict"
        wd = tmp_path / "wd"
        with pytest.raises(RuntimeError):
            api.run(**_run_kwargs(tmp_path), workdir=wd)
        assert (wd / "train" / sv_io.MODEL).exists()
        assert "scratch kept" not in capsys.readouterr().err

    def test_save_routes_artifacts_to_output_dir(self, tmp_path, fake_pipeline, tmpdir_root):
        out = api.run(
            **_run_kwargs(tmp_path),
            save={"model", "feature_matrix", "matched_bps", "callsets"},
        )
        o = tmp_path / "out"
        assert out == {
            "calls_csv": o / sv_io.CALLS_CSV, "calls_vcf": o / sv_io.CALLS_VCF,
            "model": o / sv_io.MODEL, "feature_matrix": o / sv_io.FEATURE_MATRIX,
            "matched_bps": o / sv_io.MATCHED_BPS,
            "gridss.bps": o / "gridss.bps.csv", "gridss.vcf": o / "gridss.vcf",
        }
        assert all(p.exists() for p in out.values())

    def test_ground_truth_vcf_reused_unless_forced(self, tmp_path, fake_pipeline, tmpdir_root):
        kw = _run_kwargs(tmp_path, ground_truth_bps=None, ground_truth_vcf=tmp_path / "gt.vcf")
        out = api.run(**kw, save={"ground_truth"})
        assert out["ground_truth"] == tmp_path / "out" / "gnomad.bps.csv"
        api.run(**kw, save={"ground_truth"}, force_preprocess_ground_truth=True)
        calls = fake_pipeline["gt_calls"]
        assert [c["overwrite"] for c in calls] == [False, True]
        assert all(c["output_dir"] == tmp_path / "out" for c in calls)


class TestRunValidation:
    @pytest.mark.parametrize("over, match", [
        ({"save": {"plots"}}, "Unknown save artifacts"),
        ({"ground_truth_bps": None}, "exactly one"),
        ({"ground_truth_vcf": "gt.vcf"}, "exactly one"),
        ({"save": {"ground_truth"}}, "needs ground_truth_vcf"),
        ({"caller_vcfs": {}}, "caller_vcfs is empty"),
    ])
    def test_rejected_before_any_work(self, tmp_path, fake_pipeline, tmpdir_root, over, match):
        with pytest.raises(ValueError, match=match):
            api.run(**_run_kwargs(tmp_path, **over))
        assert list(tmpdir_root.iterdir()) == []
        assert not (tmp_path / "out").exists()


class TestRunProvenance:
    def test_success_record(self, tmp_path, fake_pipeline, tmpdir_root):
        import hashlib

        out = api.run(**_run_kwargs(tmp_path), command="svensemble run --sample S")
        rec = _provenance(tmp_path / "out")
        assert rec["status"] == "succeeded"
        assert rec["command"] == "svensemble run --sample S"
        assert rec["parameters"]["caller_vcfs"] == {"gridss": [str((tmp_path / "g.vcf").resolve())]}
        assert rec["parameters"]["samples"] == ["S"]
        assert rec["parameters"]["annotate_vcfs"] is False
        assert rec["parameters"]["workdir_is_tmp"] is True
        assert rec["model"]["model_id"] == api.model_id(model="unbiased_xgb")
        assert rec["inputs"]["ground_truth_bps"]["size"] == (tmp_path / "gt.csv").stat().st_size
        assert rec["inputs"]["caller_vcfs"]["gridss"] == [{"path": str((tmp_path / "g.vcf").resolve()), "exists": False}]
        sha = hashlib.sha256(out["calls_csv"].read_bytes()).hexdigest()
        assert rec["outputs"]["calls_csv"]["sha256"] == sha
        assert rec["end_time"] is not None and rec["duration_s"] >= 0

    def test_api_call_has_no_command(self, tmp_path, fake_pipeline, tmpdir_root):
        api.run(**_run_kwargs(tmp_path))
        rec = _provenance(tmp_path / "out")
        assert rec["command"] is None and rec["argv"] is None

    def test_record_written_at_start(self, tmp_path, fake_pipeline, tmpdir_root, monkeypatch):
        seen = {}

        def spy(*a, **k):
            seen.update(_provenance(tmp_path / "out"))
            return pd.DataFrame({"INFO_EVENT": ["e1"]})

        monkeypatch.setattr(api, "preprocess_raw", spy)
        api.run(**_run_kwargs(tmp_path))
        assert seen["status"] == "running"


class TestRunAnnotate:
    VCF = (
        '##fileformat=VCFv4.2\n'
        '#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS\n'
        'chr1\t100\tgridss1o\tN\tN[chr1:500[\t.\tPASS\tEVENT=gridss1;SVTYPE=BND\tGT\t0/1\n'
    )

    def test_annotated_vcf_written_and_returned(self, tmp_path, fake_pipeline, monkeypatch, tmpdir_root):
        from svensemble._constants import SAMPLE_ID_COL
        vcf = tmp_path / "g.vcf"
        vcf.write_text(self.VCF)
        calls = pd.DataFrame({"gridss_INFO_EVENT": ["gridss1"], SAMPLE_ID_COL: ["S"], "gridss_m_pred": [0.8]})
        monkeypatch.setattr(api, "predict", lambda *a, **k: calls)
        out = api.run(**_run_kwargs(tmp_path, annotate_vcfs=True))
        path = out["gridss.annotated_vcf"]
        assert path.parent == (tmp_path / "out").resolve()
        assert path.read_text().splitlines()[-1].endswith("GT:SVENSSC\t0/1:0.8")
        assert _provenance(tmp_path / "out")["parameters"]["annotate_vcfs"] is True

    def test_unannotatable_input_fails_before_any_work(self, tmp_path, fake_pipeline, tmpdir_root):
        vcf = tmp_path / "gt.vcf"
        vcf.write_text(self.VCF)
        with pytest.raises(KeyError, match="event-key rule|CALLSET_PARAM_DICT"):
            api.run(**_run_kwargs(tmp_path, caller_vcfs={"gnomad": vcf}, training_set="gnomad", annotate_vcfs=True))
        assert not (tmp_path / "out").exists()
        assert list(tmpdir_root.iterdir()) == []

    def test_mismatched_samples_fail_before_any_work(self, tmp_path, fake_pipeline, monkeypatch, tmpdir_root):
        monkeypatch.setattr(sv_io, "parse_samples_from_vcf", lambda path: ["A"] if "g" in Path(path).name else ["B"])
        with pytest.raises(ValueError, match="share one sample set"):
            api.run(**_run_kwargs(tmp_path, caller_vcfs={"gridss": tmp_path / "g.vcf", "manta_diploid": tmp_path / "m.vcf"}))
        assert not (tmp_path / "out").exists()
