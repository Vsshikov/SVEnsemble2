"""Opt-in multi-sample end-to-end test: api.run on two samples built from the e2e inputs.

Enable as test_e2e: SVENSEMBLE_E2E=<spec.json> and `-m slow`. From the spec's
Delly and Manta VCFs it builds:

  * Manta as one joint-called VCF with two identical sample columns, S1 and S2;
  * Delly as two single-sample files, S1 with every record and S2 without
    every tenth record,

then runs api.run with annotate_vcfs and checks that calls, scores and the
annotated VCFs stay per sample.
"""

import json
import os
from pathlib import Path

import pandas as pd
import pysam
import pytest

from svensemble._constants import SAMPLE_ID_COL, SCORE_FIELD

SPEC_ENV = "SVENSEMBLE_E2E"

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        not os.environ.get(SPEC_ENV),
        reason=f"set {SPEC_ENV} to an e2e spec JSON to enable",
    ),
]


def _vcf_text(path: Path) -> tuple[list[str], list[str]]:
    """(header lines, record lines) of a VCF/BCF as text, without newlines."""
    with pysam.VariantFile(str(path)) as vcf:
        header = str(vcf.header).rstrip("\n").split("\n")
        records = [str(rec).rstrip("\n") for rec in vcf]
    return header, records


def _with_samples(header: list[str], records: list[str], samples: list[str]) -> str:
    """One-sample VCF text re-emitted with *samples* columns, each a copy of the original."""
    chrom = header[-1].split("\t")[:9] + samples
    lines = header[:-1] + ["\t".join(chrom)]
    for rec in records:
        cols = rec.split("\t")
        lines.append("\t".join(cols[:9] + [cols[9]] * len(samples)))
    return "\n".join(lines) + "\n"


@pytest.fixture(scope="module")
def multi_run(tmp_path_factory):
    from svensemble import api

    spec = json.loads(Path(os.environ[SPEC_ENV]).read_text())
    vcfs = dict(zip(spec["callers"], (Path(p) for p in spec["caller_vcfs"])))
    base = tmp_path_factory.mktemp("e2e_multi")

    manta_header, manta_records = _vcf_text(vcfs["manta_diploid"])
    manta = base / "manta_diploid.S1S2.vcf"
    manta.write_text(_with_samples(manta_header, manta_records, ["S1", "S2"]))

    delly_header, delly_records = _vcf_text(vcfs["delly"])
    delly_s1 = base / "S1.delly.vcf"
    delly_s1.write_text(_with_samples(delly_header, delly_records, ["S1"]))
    dropped_ids = {rec.split("\t")[2] for rec in delly_records[::10]}
    delly_s2 = base / "S2.delly.vcf"
    delly_s2.write_text(_with_samples(
        delly_header, [rec for rec in delly_records if rec.split("\t")[2] not in dropped_ids], ["S2"],
    ))

    out = base / "out"
    outputs = api.run(
        sample="S1S2",
        caller_vcfs={"delly": (delly_s1, delly_s2), "manta_diploid": manta},
        ref=Path(spec["ref"]),
        model=spec["model"],
        output_dir=out,
        training_set=spec["training_set"],
        ground_truth_bps=Path(spec["ground_truth_bps"]),
        ground_truth_name=spec["ground_truth_name"],
        workdir=base / "workdir",
        annotate_vcfs=True,
    )
    return outputs, dropped_ids, manta


def _scores(path: Path) -> dict[str, dict[str, float | None]]:
    with pysam.VariantFile(str(path)) as vcf:
        samples = list(vcf.header.samples)
        return {rec.id: {s: rec.samples[s][SCORE_FIELD] for s in samples} for rec in vcf}


def test_calls_are_per_sample(multi_run):
    outputs, _, _ = multi_run
    calls = pd.read_csv(outputs["calls_csv"], index_col=0, dtype={SAMPLE_ID_COL: str})
    assert set(calls[SAMPLE_ID_COL]) == {"S1", "S2"}
    assert not calls.duplicated(["INFO_EVENT", SAMPLE_ID_COL]).any()


def test_calls_vcf_one_record_per_sv_with_both_samples(multi_run):
    outputs, _, _ = multi_run
    calls = pd.read_csv(outputs["calls_csv"], index_col=0)
    scores = _scores(outputs["calls_vcf"])
    with pysam.VariantFile(str(outputs["calls_vcf"])) as vcf:
        assert list(vcf.header.samples) == ["S1", "S2"]
    assert set(scores) <= set(calls["INFO_EVENT"]) | {f"{e}_{b}" for e in calls["INFO_EVENT"] for b in ("bnd1", "bnd2")}


def test_dropped_delly_records_unscored_in_s2_only(multi_run):
    outputs, dropped_ids, _ = multi_run
    calls = pd.read_csv(outputs["calls_csv"], index_col=0, dtype={SAMPLE_ID_COL: str, "delly_INFO_EVENT": str})
    delly_events = calls.dropna(subset=["delly_INFO_EVENT"])
    by_sample = {s: set(g["delly_INFO_EVENT"]) for s, g in delly_events.groupby(SAMPLE_ID_COL)}
    assert dropped_ids & by_sample["S1"]
    assert not dropped_ids & by_sample["S2"]

    s1 = _scores(outputs["delly.annotated_vcf.0"])
    s2 = _scores(outputs["delly.annotated_vcf.1"])
    assert all(rec["S1"] is not None for rec in s1.values())
    assert not dropped_ids & set(s2)


def test_joint_manta_vcf_scored_for_both_samples(multi_run):
    outputs, _, manta_input = multi_run
    path = outputs["manta_diploid.annotated_vcf"]
    scores = _scores(path)
    assert scores and all(rec["S1"] is not None and rec["S2"] is not None for rec in scores.values())
    # Only FORMAT/sample columns change in the annotated copy
    original = [line for line in manta_input.read_text().splitlines() if line]
    annotated = [
        line for line in path.read_text().splitlines()
        if line and not line.startswith(f"##FORMAT=<ID={SCORE_FIELD},")
    ]
    assert len(original) == len(annotated)
    for o, a in zip(original, annotated):
        if not o.startswith("#"):
            assert o.split("\t")[:8] == a.split("\t")[:8]
