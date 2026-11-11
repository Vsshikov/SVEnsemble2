"""
Unit tests for svensemble.vcfparse.
"""

import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest
import pysam

from svensemble.vcfparse import (
    get_callset_name,
    parse_vcf_to_df,
    _fetch_filter_fields,
    _fetch_info_fields,
    _ordered_info_col_names,
)
from svensemble import _caller_params


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_HEADER_LINES = """\
##fileformat=VCFv4.2
##contig=<ID=chr1>
##FILTER=<ID=PASS,Description="pass">
##FILTER=<ID=LowQual,Description="lq">
##FILTER=<ID=NoSupport,Description="ns">
##INFO=<ID=SVTYPE,Number=1,Type=String,Description="sv">
##INFO=<ID=SVLEN,Number=1,Type=Integer,Description="sv">
##INFO=<ID=IMPRECISE,Number=0,Type=Flag,Description="imprecise">
"""

_SAMPLE_COL = "##FORMAT=<ID=GT,Number=1,Type=String,Description=\"gt\">\n"

_CHROM_HEADER = "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n"
_CHROM_HEADER_FMT = "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS1\n"


def _write_vcf(rows: list[str], with_format: bool = False) -> str:
    """Write a temporary VCF and return its path."""
    header = _HEADER_LINES
    if with_format:
        header += _SAMPLE_COL
        header += _CHROM_HEADER_FMT
    else:
        header += _CHROM_HEADER
    content = header + "".join(rows)
    with tempfile.NamedTemporaryFile(suffix=".vcf", delete=False, mode="w") as f:
        f.write(content)
        return f.name


# ---------------------------------------------------------------------------
# _fetch_filter_fields
# ---------------------------------------------------------------------------

class TestFetchFilterFields:
    """Tests for the FILTER 1/0 accumulation using `in` membership check."""

    _FILTER_COLS = ["PASS", "LowQual", "NoSupport"]

    def _make_record(self, filter_str: str) -> pysam.VariantRecord:
        row = f"chr1\t1000\tsv1\tN\t<DEL>\t50\t{filter_str}\tSVTYPE=DEL\n"
        path = _write_vcf([row])
        try:
            with pysam.VariantFile(path) as vf:
                return next(iter(vf))
        finally:
            os.unlink(path)

    def _run(self, filter_str: str) -> dict:
        record = self._make_record(filter_str)
        acc = {f"FILTER_{n}": [] for n in self._FILTER_COLS}
        _fetch_filter_fields(record, acc, {"filter_cols": self._FILTER_COLS})
        return {k: v[0] for k, v in acc.items()}

    def test_pass_only(self):
        result = self._run("PASS")
        assert result == {"FILTER_PASS": 1, "FILTER_LowQual": 0, "FILTER_NoSupport": 0}

    def test_single_fail_filter(self):
        result = self._run("LowQual")
        assert result == {"FILTER_PASS": 0, "FILTER_LowQual": 1, "FILTER_NoSupport": 0}

    def test_multiple_fail_filters(self):
        result = self._run("LowQual;NoSupport")
        assert result == {"FILTER_PASS": 0, "FILTER_LowQual": 1, "FILTER_NoSupport": 1}

    def test_missing_filter_column(self):
        """pysam maps a '.' FILTER to PASS internally."""
        result = self._run(".")
        assert result == {"FILTER_PASS": 1, "FILTER_LowQual": 0, "FILTER_NoSupport": 0}

    def test_filter_not_in_caller_params_is_ignored(self):
        """Filters present in the VCF but not in caller_params are not collected."""
        record = self._make_record("LowQual")
        acc = {"FILTER_PASS": []}
        _fetch_filter_fields(record, acc, {"filter_cols": ["PASS"]})
        assert acc["FILTER_PASS"] == [0]
        assert "FILTER_LowQual" not in acc


# ---------------------------------------------------------------------------
# _fetch_info_fields
# ---------------------------------------------------------------------------

class TestFetchInfoFields:

    def _make_record(self, info_str: str) -> pysam.VariantRecord:
        row = f"chr1\t1000\tsv1\tN\t<DEL>\t50\tPASS\t{info_str}\n"
        path = _write_vcf([row])
        try:
            with pysam.VariantFile(path) as vf:
                return next(iter(vf))
        finally:
            os.unlink(path)

    def test_present_fields_are_collected(self):
        record = self._make_record("SVTYPE=DEL;SVLEN=500")
        names = frozenset(["SVTYPE", "SVLEN"])
        acc = {"INFO_SVTYPE": [], "INFO_SVLEN": []}
        _fetch_info_fields(record, acc, names)
        assert acc["INFO_SVTYPE"] == ["DEL"]
        assert acc["INFO_SVLEN"] == [500]

    def test_absent_field_becomes_none(self):
        record = self._make_record("SVTYPE=DEL")
        names = frozenset(["SVTYPE", "SVLEN"])
        acc = {"INFO_SVTYPE": [], "INFO_SVLEN": []}
        _fetch_info_fields(record, acc, names)
        assert acc["INFO_SVTYPE"] == ["DEL"]
        assert acc["INFO_SVLEN"] == [None]

    def test_flag_field_present_returns_true(self):
        record = self._make_record("SVTYPE=DEL;IMPRECISE")
        names = frozenset(["SVTYPE", "IMPRECISE"])
        acc = {"INFO_SVTYPE": [], "INFO_IMPRECISE": []}
        _fetch_info_fields(record, acc, names)
        assert acc["INFO_IMPRECISE"] == [True]

    def test_flag_field_absent_returns_false(self):
        """pysam returns False (not None/KeyError) for a FLAG defined in the header
        but absent from a record's INFO column. Only non-flag missing fields yield None."""
        record = self._make_record("SVTYPE=DEL")
        names = frozenset(["SVTYPE", "IMPRECISE"])
        acc = {"INFO_SVTYPE": [], "INFO_IMPRECISE": []}
        _fetch_info_fields(record, acc, names)
        assert acc["INFO_IMPRECISE"] == [False]



# ---------------------------------------------------------------------------
# Column order
# ---------------------------------------------------------------------------

class TestColumnOrder:
    """Parsed column order feeds the training feature matrix, so it must be
    fixed by caller_params rather than by per-process set iteration."""

    def test_info_col_names_dedupe_in_caller_params_order(self):
        params = {
            "info_cols_pred": ["SVLEN", "CT", "PE"],
            "info_cols_mandatory": ["SVTYPE", "SVLEN"],
        }
        assert _ordered_info_col_names(params) == ("SVLEN", "CT", "PE", "SVTYPE")

    def test_columns_independent_of_hash_seed(self):
        path = _write_vcf(["chr1\t1000\tsv1\tN\t<DEL>\t50\tPASS\tSVTYPE=DEL;SVLEN=500\n"])
        script = (
            "import sys, pysam\n"
            "from svensemble import _caller_params\n"
            "from svensemble.vcfparse import parse_vcf_to_df\n"
            "with pysam.VariantFile(sys.argv[1]) as vf:\n"
            "    df, _ = parse_vcf_to_df(vf, _caller_params.DELLY_PARAMS | {'format_cols': []})\n"
            "print(','.join(df.columns))\n"
        )
        try:
            columns = {
                subprocess.run(
                    [sys.executable, "-c", script, path],
                    env=os.environ | {"PYTHONHASHSEED": seed},
                    capture_output=True, text=True, check=True,
                ).stdout
                for seed in ("1", "2")
            }
        finally:
            os.unlink(path)
        assert len(columns) == 1


# ---------------------------------------------------------------------------
# parse_vcf_to_df on a sites-only VCF (ground truth, validation sets)
# ---------------------------------------------------------------------------

class TestParseVcfToDfSitesOnly:

    _PARAMS = _caller_params.DEFAULT_PARAMS | {
        "filter_cols": ["PASS", "LowQual"],
        "info_cols_mandatory": ["SVTYPE", "SVLEN"],
        "info_cols_pred": [],
        "format_cols": [],
    }

    def _parse(self, rows: list[str]) -> object:
        path = _write_vcf(rows)
        try:
            with pysam.VariantFile(path) as vf:
                df, _ = parse_vcf_to_df(vf, self._PARAMS)
            return df
        finally:
            os.unlink(path)

    def test_basic_columns_present(self):
        rows = ["chr1\t1000\tsv1\tN\t<DEL>\t50\tPASS\tSVTYPE=DEL;SVLEN=500\n"]
        df = self._parse(rows)
        assert list(df["CHROM"]) == ["chr1"]
        assert list(df["POS"]) == [1000]
        assert list(df["INFO_SVTYPE"]) == ["DEL"]
        assert list(df["INFO_SVLEN"]) == [500]

    def test_filter_encoding(self):
        rows = [
            "chr1\t1000\tsv1\tN\t<DEL>\t50\tPASS\tSVTYPE=DEL;SVLEN=500\n",
            "chr1\t2000\tsv2\tN\t<DEL>\t10\tLowQual\tSVTYPE=DEL;SVLEN=200\n",
        ]
        df = self._parse(rows)
        assert list(df["FILTER_PASS"]) == [1, 0]
        assert list(df["FILTER_LowQual"]) == [0, 1]

    def test_absent_info_field_is_none(self):
        rows = ["chr1\t1000\tsv1\tN\t<DEL>\t50\tPASS\tSVTYPE=DEL\n"]
        df = self._parse(rows)
        assert df["INFO_SVLEN"].isna().all()

    def test_record_count(self):
        rows = [f"chr1\t{i*1000}\tsv{i}\tN\t<DEL>\t50\tPASS\tSVTYPE=DEL;SVLEN=500\n"
                for i in range(1, 6)]
        df = self._parse(rows)
        assert len(df) == 5

    def test_no_sample_column(self):
        df = self._parse(["chr1\t1000\tsv1\tN\t<DEL>\t50\tPASS\tSVTYPE=DEL;SVLEN=500\n"])
        assert "FORMAT_SAMPLE_ID" not in df.columns


# ---------------------------------------------------------------------------
# parse_vcf_to_df with samples — integration
# ---------------------------------------------------------------------------

class TestParseVcfToDfWithSamples:

    _PARAMS = _caller_params.DEFAULT_PARAMS | {
        "filter_cols": ["PASS"],
        "info_cols_mandatory": ["SVTYPE"],
        "info_cols_pred": [],
        "format_cols": ["GT"],
    }

    def _parse(self, rows: list[str], sample_name: str | None = None):
        path = _write_vcf(rows, with_format=True)
        try:
            with pysam.VariantFile(path) as vf:
                df, _ = parse_vcf_to_df(vf, self._PARAMS, sample_name)
            return df
        finally:
            os.unlink(path)

    def test_format_field_collected(self):
        rows = ["chr1\t1000\tsv1\tN\t<DEL>\t50\tPASS\tSVTYPE=DEL\tGT\t0/1\n"]
        df = self._parse(rows)
        assert "SAMPLE_GT" in df.columns
        assert df["SAMPLE_GT"].iloc[0] == (0, 1)

    def test_raises_when_format_requested_from_sites_only_vcf(self):
        rows = ["chr1\t1000\tsv1\tN\t<DEL>\t50\tPASS\tSVTYPE=DEL\n"]
        path = _write_vcf(rows, with_format=False)
        try:
            with pysam.VariantFile(path) as vf:
                with pytest.raises(ValueError, match="FORMAT"):
                    parse_vcf_to_df(vf, self._PARAMS)
        finally:
            os.unlink(path)

    _TWO_SAMPLE_ROWS = ["chr1\t1000\tsv1\tN\t<DEL>\t50\tPASS\tSVTYPE=DEL\tGT\t0/1\t1/1\n"]

    def _parse_two_samples(self, sample_names=None, params=None):
        header = _HEADER_LINES + _SAMPLE_COL + _CHROM_HEADER_FMT.replace("\tS1\n", "\tS1\tS2\n")
        with tempfile.NamedTemporaryFile(suffix=".vcf", delete=False, mode="w") as f:
            f.write(header + "".join(self._TWO_SAMPLE_ROWS))
        try:
            with pysam.VariantFile(f.name) as vf:
                df, _ = parse_vcf_to_df(vf, params or self._PARAMS, sample_names)
            return df
        finally:
            os.unlink(f.name)

    def test_one_row_per_sample_by_default(self):
        df = self._parse_two_samples()
        assert list(df["FORMAT_SAMPLE_ID"]) == ["S1", "S2"]
        assert list(df["SAMPLE_GT"]) == [(0, 1), (1, 1)]
        assert list(df["POS"]) == [1000, 1000]

    def test_named_sample_only(self):
        df = self._parse_two_samples("S2")
        assert list(df["FORMAT_SAMPLE_ID"]) == ["S2"]
        assert list(df["SAMPLE_GT"]) == [(1, 1)]

    def test_raises_on_sample_not_in_header(self):
        with pytest.raises(ValueError, match="S3"):
            self._parse_two_samples(["S1", "S3"])

    def test_rows_per_sample_without_format_cols(self):
        df = self._parse_two_samples(params=self._PARAMS | {"format_cols": []})
        assert list(df["FORMAT_SAMPLE_ID"]) == ["S1", "S2"]
        assert not any(c.startswith("SAMPLE_") for c in df.columns)


# ---------------------------------------------------------------------------
# Annotator columns (duphold)
# ---------------------------------------------------------------------------

_DUPHOLD_FORMAT_HEADER = """\
##FORMAT=<ID=DHFC,Number=1,Type=Float,Description="duphold depth fold-change">
##FORMAT=<ID=DHBFC,Number=1,Type=Float,Description="duphold depth fold-change compared to bins with matching GC">
##FORMAT=<ID=DHFFC,Number=1,Type=Float,Description="duphold depth flank fold-change">
"""
_DUPHOLD_INFO_HEADER = (
    "##INFO=<ID=GCF,Number=1,Type=Float,Description=\"GC-content fraction\">\n"
)

# Annotated DEL, then an inter-chromosomal BND that duphold leaves unannotated.
_DUPHOLD_ROWS = [
    "chr1\t1000\tsv1\tN\t<DEL>\t50\tPASS\tSVTYPE=DEL;GCF=0.42\t"
    "GT:DHFC:DHBFC:DHFFC\t0/1:0.5:0.48:0.45\n",
    "chr1\t5000\tsv2\tN\tN[chr2:100[\t50\tPASS\tSVTYPE=BND\tGT\t0/1\n",
]


def _write_annotated_vcf(
    rows: list[str],
    format_header: str = _DUPHOLD_FORMAT_HEADER,
    info_header: str = _DUPHOLD_INFO_HEADER,
    with_sample: bool = True,
) -> str:
    """Write a temporary VCF carrying (some of) duphold's header lines."""
    header = _HEADER_LINES + "##contig=<ID=chr2>\n" + info_header + _SAMPLE_COL + format_header
    header += _CHROM_HEADER_FMT if with_sample else _CHROM_HEADER
    with tempfile.NamedTemporaryFile(suffix=".vcf", delete=False, mode="w") as f:
        f.write(header + "".join(rows))
        return f.name


class TestAnnotatorCols:

    _NO_FORMAT_PARAMS = _caller_params.DEFAULT_PARAMS | {
        "filter_cols": ["PASS"],
        "info_cols_mandatory": ["SVTYPE"],
        "info_cols_pred": [],
        "format_cols": [],
    }
    _WITH_FORMAT_PARAMS = _NO_FORMAT_PARAMS | {
        "format_cols": ["GT"],
    }
    _ANNOT_COLS = ["ANNOT_DHFC", "ANNOT_DHBFC", "ANNOT_DHFFC", "ANNOT_GCF"]

    def _parse(self, path: str, params: dict):
        try:
            with pysam.VariantFile(path) as vf:
                df, _ = parse_vcf_to_df(vf, params)
            return df
        finally:
            os.unlink(path)

    def test_parsed_without_format_cols(self):
        df = self._parse(_write_annotated_vcf(_DUPHOLD_ROWS), self._NO_FORMAT_PARAMS)
        assert [c for c in df.columns if c.startswith("ANNOT_")] == self._ANNOT_COLS
        assert not any(c.startswith("SAMPLE_") for c in df.columns)
        assert df["ANNOT_DHFC"].iloc[0] == pytest.approx(0.5)
        assert df["ANNOT_DHBFC"].iloc[0] == pytest.approx(0.48)
        assert df["ANNOT_DHFFC"].iloc[0] == pytest.approx(0.45)
        assert df["ANNOT_GCF"].iloc[0] == pytest.approx(0.42)

    def test_parsed_alongside_format_cols(self):
        df = self._parse(_write_annotated_vcf(_DUPHOLD_ROWS), self._WITH_FORMAT_PARAMS)
        assert df["SAMPLE_GT"].iloc[0] == (0, 1)
        assert [c for c in df.columns if c.startswith("SAMPLE_")] == ["SAMPLE_GT"]
        assert df["ANNOT_DHFFC"].iloc[0] == pytest.approx(0.45)

    @pytest.mark.parametrize("params_name", ["_NO_FORMAT_PARAMS", "_WITH_FORMAT_PARAMS"])
    def test_unannotated_record_is_nan(self, params_name):
        df = self._parse(_write_annotated_vcf(_DUPHOLD_ROWS), getattr(self, params_name))
        assert df[self._ANNOT_COLS].iloc[1].isna().all()

    @pytest.mark.parametrize("params_name", ["_NO_FORMAT_PARAMS", "_WITH_FORMAT_PARAMS"])
    def test_absent_without_annotator_header(self, params_name):
        rows = ["chr1\t1000\tsv1\tN\t<DEL>\t50\tPASS\tSVTYPE=DEL\tGT\t0/1\n"]
        df = self._parse(_write_vcf(rows, with_format=True), getattr(self, params_name))
        assert not any(c.startswith("ANNOT_") for c in df.columns)

    def test_partial_header_not_detected(self):
        rows = ["chr1\t1000\tsv1\tN\t<DEL>\t50\tPASS\tSVTYPE=DEL\tGT:DHFC\t0/1:0.5\n"]
        path = _write_annotated_vcf(rows, info_header="")
        df = self._parse(path, self._NO_FORMAT_PARAMS)
        assert not any(c.startswith("ANNOT_") for c in df.columns)

    def test_disabled_by_empty_annotator_cols(self):
        params = self._NO_FORMAT_PARAMS | {"annotator_cols": {}}
        df = self._parse(_write_annotated_vcf(_DUPHOLD_ROWS), params)
        assert not any(c.startswith("ANNOT_") for c in df.columns)

    def test_sites_only_vcf_skips_format_annotator(self):
        rows = ["chr1\t1000\tsv1\tN\t<DEL>\t50\tPASS\tSVTYPE=DEL;GCF=0.42\n"]
        path = _write_annotated_vcf(rows, with_sample=False)
        df = self._parse(path, self._NO_FORMAT_PARAMS)
        assert not any(c.startswith("ANNOT_") for c in df.columns)


# ---------------------------------------------------------------------------
# get_callset_name
# ---------------------------------------------------------------------------

# Header-only copies of real caller output (sample BM009-met of the
# full_run_somatic run): gridss 2.13.2, manta 1.6.0 (candidate_sv and
# diploid_sv), delly 1.7.3 (BCF, written as text), smoove 0.2.8, svaba 1.2.0,
# cuteSV 2.0.2, Sniffles2 2.7.3 and Severus 1.7. The sample is renamed to
# SAMPLE; command lines, dates, the reference and all contigs but chr1 are
# removed.
CALLER_HEADERS = Path(__file__).parent / "data" / "caller_headers"
DETECTABLE = sorted(path.stem for path in CALLER_HEADERS.glob("*.vcf"))


def _header(path) -> pysam.VariantHeader:
    with pysam.VariantFile(str(path)) as vf:
        return vf.header.copy()


class TestGetCallsetName:
    @pytest.mark.parametrize("callset", DETECTABLE)
    def test_real_header_names_its_caller(self, callset):
        assert get_callset_name(_header(CALLER_HEADERS / f"{callset}.vcf")) == callset

    def test_signatures_only_on_front_facing_callers(self):
        # _H variants copy their base params; a copied signature would make
        # every base-caller VCF match two callsets.
        signed = sorted(name for name, params in _caller_params.CALLSET_PARAM_DICT.items()
                        if params.get('header_signature'))
        assert signed == DETECTABLE

    def test_unknown_header_gives_none(self):
        path = _write_vcf([], with_format=True)
        try:
            assert get_callset_name(_header(path)) is None
        finally:
            os.unlink(path)

    def test_header_matching_two_callsets_gives_none(self, tmp_path):
        delly = (CALLER_HEADERS / "delly.vcf").read_text().splitlines(keepends=True)
        svaba_fields = [line for line in (CALLER_HEADERS / "svaba.vcf").read_text().splitlines(keepends=True)
                        if line.startswith("##INFO=")]
        path = tmp_path / "both.vcf"
        path.write_text("".join(delly[:-1] + svaba_fields + delly[-1:]))
        assert get_callset_name(_header(path)) is None
