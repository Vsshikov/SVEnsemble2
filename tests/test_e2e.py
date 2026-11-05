"""Opt-in end-to-end regression test: `api.run` outputs must match a golden workdir.

Enable with SVENSEMBLE_E2E=<spec.json> and `-m slow`. The spec holds the
`run` inputs (sample, ref, callers, caller_vcfs, ground_truth_bps,
ground_truth_name, model, training_set) and golden_dir, a workdir produced by
`svensemble run` on the same inputs. The run uses an explicit workdir with
output_dir=workdir/predict, so the golden layout predates the tmpdir change.
"""

import json
import os
from pathlib import Path

import pandas as pd
import pytest

SPEC_ENV = "SVENSEMBLE_E2E"

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        not os.environ.get(SPEC_ENV),
        reason=f"set {SPEC_ENV} to an e2e spec JSON to enable",
    ),
]


@pytest.fixture(scope="module")
def spec():
    return json.loads(Path(os.environ[SPEC_ENV]).read_text())


def _csv_outputs(callers):
    paths = []
    for c in callers:
        paths += [f"preprocess_raw/{c}.raw_bps.csv", f"preprocess/{c}/{c}.bps.csv"]
    return paths + [
        "match/matched_bps.csv",
        "train/feature_matrix.csv",
        "predict/svensemble2_calls.csv",
    ]


def _vcf_outputs(callers):
    return [f"preprocess/{c}/{c}.vcf" for c in callers] + ["predict/svensemble2_calls.vcf"]


@pytest.fixture(scope="module")
def run_workdir(spec, tmp_path_factory):
    from svensemble import api

    base = tmp_path_factory.mktemp("e2e")
    workdir = base / "workdir"
    with pytest.MonkeyPatch.context() as mp:
        mp.chdir(base)
        api.run(
            sample=spec["sample"],
            workdir=workdir,
            output_dir=workdir / "predict",
            ref=Path(spec["ref"]),
            caller_vcfs=dict(zip(spec["callers"], (Path(p) for p in spec["caller_vcfs"]))),
            ground_truth_bps=Path(spec["ground_truth_bps"]),
            ground_truth_name=spec["ground_truth_name"],
            model=spec["model"],
            training_set=spec["training_set"],
        )
    return workdir


def test_csv_outputs_match_golden(run_workdir, spec):
    golden = Path(spec["golden_dir"])
    for rel in _csv_outputs(spec["callers"]):
        got = pd.read_csv(run_workdir / rel, low_memory=False)
        expected = pd.read_csv(golden / rel, low_memory=False)
        pd.testing.assert_frame_equal(got, expected, check_like=True, check_exact=False, obj=rel)


def test_vcf_records_match_golden(run_workdir, spec):
    golden = Path(spec["golden_dir"])
    for rel in _vcf_outputs(spec["callers"]):
        got = [l for l in (run_workdir / rel).read_text().splitlines() if not l.startswith("##")]
        expected = [l for l in (golden / rel).read_text().splitlines() if not l.startswith("##")]
        assert got == expected, rel


def test_no_stray_files_outside_workdir(run_workdir):
    assert sorted(p.name for p in run_workdir.parent.iterdir()) == ["workdir"]
