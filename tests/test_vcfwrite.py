"""
Unit tests for svensemble.vcfwrite.

Fixture convention
------------------
Breakpoint rows are constructed directly (bypassing events.py) with explicit
POS_bnd1 / POS_bnd2 values.  Positions follow the same convention used by the
pipeline: they are stored as the raw integer that pysam writes into the VCF POS
column (no 0-based / 1-based conversion occurs in VariantRecord.pos for VCF
files — the value round-trips exactly).

INV rows are supplied as paired ``_e1`` / ``_e2`` events matching the output of
events.convert_inv_to_bnd:
  _e1: POS_bnd1=POS,   POS_bnd2=POS+SVLEN
  _e2: POS_bnd1=POS+1, POS_bnd2=POS+SVLEN+1

For all symbolic SV types, SVLEN written by vcfwrite equals
``POS_bnd2 - POS_bnd1`` (the span between the two breakends).

The INFO_INFERRED_SVTYPE column is used to pre-annotate SV types so tests do
not depend on caller-specific annotation logic.
"""

from __future__ import annotations

import pandas as pd
import pysam
import pytest

from svensemble.vcfwrite import (
    _build_header,
    _get_ins_svlen,
    _has_ci,
    _strip_inv_event_suffix,
    write_vcf,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _row(
    event_id: str,
    chrom_bnd1: str,
    pos_bnd1: int,
    alt_bnd1: str,
    chrom_bnd2: str,
    pos_bnd2: int,
    alt_bnd2: str,
    *,
    svtype: str | None = None,
    svlen: int | None = None,
    cipos: int = 0,
    ciend: int = 0,
) -> dict:
    """Build a minimal breakpoint row dict with optional CI and SVLEN."""
    d: dict = {
        'INFO_EVENT': event_id,
        'CHROM_bnd1': chrom_bnd1,
        'POS_bnd1': pos_bnd1,
        'ALT_bnd1': alt_bnd1,
        'CHROM_bnd2': chrom_bnd2,
        'POS_bnd2': pos_bnd2,
        'ALT_bnd2': alt_bnd2,
        'INFO_CIPOS_l_bnd1': -cipos,
        'INFO_CIPOS_r_bnd1': cipos,
        'INFO_CIPOS_l_bnd2': -ciend,
        'INFO_CIPOS_r_bnd2': ciend,
    }
    if svtype is not None:
        d['INFO_INFERRED_SVTYPE'] = svtype
    if svlen is not None:
        d['INFO_SVLEN'] = svlen
    return d


def _del_row(
    event_id: str,
    pos: int,
    span: int,
    chrom: str = 'chr1',
    **kw,
) -> dict:
    """DEL row where SVLEN == span == POS_bnd2 - POS_bnd1."""
    pos_bnd1 = pos
    pos_bnd2 = pos + span
    return _row(
        event_id,
        chrom, pos_bnd1, f'N[{chrom}:{pos_bnd2}[',
        chrom, pos_bnd2, f']{chrom}:{pos_bnd1}]N',
        svtype='DEL',
        **kw,
    )


def _inv_e1_row(
    event_id: str,
    pos: int,
    svlen: int,
    chrom: str = 'chr1',
    **kw,
) -> dict:
    """_e1 breakpoint row for an inversion (matches events.convert_inv_to_bnd)."""
    pos_bnd2 = pos + svlen
    return _row(
        f'{event_id}_e1',
        chrom, pos, f'N]{chrom}:{pos_bnd2}]',
        chrom, pos_bnd2, f'N]{chrom}:{pos}]',
        svtype='INV',
        **kw,
    )


def _inv_e2_row(
    event_id: str,
    pos: int,
    svlen: int,
    chrom: str = 'chr1',
    **kw,
) -> dict:
    """_e2 breakpoint row for an inversion (matches events.convert_inv_to_bnd)."""
    pos_bnd2 = pos + svlen + 1
    return _row(
        f'{event_id}_e2',
        chrom, pos + 1, f'[{chrom}:{pos_bnd2}[N',
        chrom, pos_bnd2, f'[{chrom}:{pos + 1}[N',
        svtype='INV',
        **kw,
    )


def _bnd_row(
    event_id: str,
    pos: int,
    chrom: str = 'chr1',
    chrom2: str = 'chr2',
    **kw,
) -> dict:
    """Unclassified (OTHER) BND row for a simple inter-chromosomal breakend."""
    pos_bnd2 = pos + 100
    return _row(
        event_id,
        chrom, pos, f'N[{chrom2}:{pos_bnd2}[',
        chrom2, pos_bnd2, f']{chrom}:{pos}]N',
        svtype='OTHER',
        **kw,
    )


def _df(*rows: dict) -> pd.DataFrame:
    return pd.DataFrame(list(rows)).reset_index(drop=True)


def _read_records(path) -> list:
    """Return all VariantRecord objects from a VCF file."""
    with pysam.VariantFile(str(path)) as vcf:
        return list(vcf)


# ---------------------------------------------------------------------------
# _strip_inv_event_suffix
# ---------------------------------------------------------------------------

class TestStripInvEventSuffix:
    def test_strips_e1_suffix(self):
        assert _strip_inv_event_suffix('INV_001_e1') == 'INV_001'

    def test_strips_e2_suffix(self):
        assert _strip_inv_event_suffix('INV_001_e2') == 'INV_001'

    def test_leaves_unsuffixed_unchanged(self):
        assert _strip_inv_event_suffix('INV_001') == 'INV_001'

    def test_leaves_unrelated_suffix_unchanged(self):
        assert _strip_inv_event_suffix('INV_001_e3') == 'INV_001_e3'

    def test_leaves_partial_match_unchanged(self):
        """Suffix must appear at the very end."""
        assert _strip_inv_event_suffix('INV_e1_extra') == 'INV_e1_extra'


# ---------------------------------------------------------------------------
# _has_ci
# ---------------------------------------------------------------------------

class TestHasCi:
    def test_true_when_all_four_ci_columns_present(self):
        df = pd.DataFrame([{
            'INFO_CIPOS_l_bnd1': 0, 'INFO_CIPOS_r_bnd1': 0,
            'INFO_CIPOS_l_bnd2': 0, 'INFO_CIPOS_r_bnd2': 0,
        }])
        assert _has_ci(df) is True

    def test_false_when_ci_columns_absent(self):
        df = pd.DataFrame([{'CHROM_bnd1': 'chr1'}])
        assert _has_ci(df) is False

    def test_false_when_only_bnd1_ci_columns_present(self):
        df = pd.DataFrame([{
            'INFO_CIPOS_l_bnd1': 0,
            'INFO_CIPOS_r_bnd1': 0,
        }])
        assert _has_ci(df) is False

    def test_false_when_only_bnd2_ci_columns_present(self):
        df = pd.DataFrame([{
            'INFO_CIPOS_l_bnd2': 0,
            'INFO_CIPOS_r_bnd2': 0,
        }])
        assert _has_ci(df) is False

    def test_false_when_three_of_four_ci_columns_present(self):
        df = pd.DataFrame([{
            'INFO_CIPOS_l_bnd1': 0,
            'INFO_CIPOS_r_bnd1': 0,
            'INFO_CIPOS_l_bnd2': 0,
        }])
        assert _has_ci(df) is False


# ---------------------------------------------------------------------------
# _get_ins_svlen
# ---------------------------------------------------------------------------

class TestGetInsSvlen:
    def test_uses_info_svlen_when_positive(self):
        row = pd.Series({'INFO_SVLEN': 250, 'ALT_bnd1': 'N[chr1:101['})
        assert _get_ins_svlen(row) == 250

    def test_converts_negative_info_svlen_to_positive(self):
        row = pd.Series({'INFO_SVLEN': -250, 'ALT_bnd1': 'N[chr1:101['})
        assert _get_ins_svlen(row) == 250

    def test_returns_none_when_svlen_missing(self):
        row = pd.Series({'ALT_bnd1': 'N[chr1:101['})
        assert _get_ins_svlen(row) is None

    def test_returns_none_when_svlen_is_zero(self):
        row = pd.Series({'INFO_SVLEN': 0, 'ALT_bnd1': 'N[chr1:101['})
        assert _get_ins_svlen(row) is None

    def test_returns_none_when_svlen_is_nan(self):
        import numpy as np
        row = pd.Series({'INFO_SVLEN': np.nan, 'ALT_bnd1': 'N[chr1:101['})
        assert _get_ins_svlen(row) is None


# ---------------------------------------------------------------------------
# _build_header
# ---------------------------------------------------------------------------

class TestBuildHeader:
    def test_contains_required_info_fields(self):
        df = _df(_del_row('ev1', 100, 500))
        header = _build_header(df)
        info_ids = set(header.info.keys())
        for field in ('SVTYPE', 'END', 'SVLEN', 'CIPOS', 'CIEND', 'MATEID', 'EVENT', 'IMPRECISE'):
            assert field in info_ids, f'Missing INFO field: {field}'

    def test_contains_sv_alt_definitions(self):
        df = _df(_del_row('ev1', 100, 500))
        header = _build_header(df)
        alt_ids = set(header.alts.keys())
        for alt in ('DEL', 'DUP', 'INS', 'INV'):
            assert alt in alt_ids, f'Missing ALT definition: {alt}'

    def test_contains_pass_filter(self):
        df = _df(_del_row('ev1', 100, 500))
        header = _build_header(df)
        assert 'PASS' in header.filters

    def test_contigs_derived_from_dataframe_chromosomes(self):
        df = _df(_del_row('ev1', 100, 500, chrom='chr7'))
        header = _build_header(df)
        assert 'chr7' in header.contigs

    def test_both_bnd_chroms_added_as_contigs(self):
        df = _df(_bnd_row('ev1', 100, chrom='chr3', chrom2='chr17'))
        header = _build_header(df, source='test_caller')
        assert 'chr3' in header.contigs
        assert 'chr17' in header.contigs

    def test_source_line_added_when_provided(self):
        df = _df(_del_row('ev1', 100, 500))
        header = _build_header(df, source='test_caller')
        assert '##source=test_caller' in str(header)

    def test_sample_name_added_when_provided(self):
        df = _df(_del_row('ev1', 100, 500))
        header = _build_header(df, sample_name='NA12878')
        assert 'NA12878' in header.samples

    # VCF 4.5 §3 — INFO field Number specifications
    def test_svlen_info_number_is_A_per_spec(self):
        """VCF 4.5 §3: SVLEN must be Number=A (one value per ALT allele)."""
        df = _df(_del_row('ev1', 100, 500))
        header = _build_header(df)
        assert 'Number=A' in str(header), (
            "SVLEN INFO header must declare Number=A per VCF 4.5 §3"
        )

    def test_cipos_info_number_is_dot_per_spec(self):
        """VCF 4.5 §3: CIPOS must be Number=. (variable, 2 values per ALT allele)."""
        df = _df(_del_row('ev1', 100, 500))
        header = _build_header(df)
        header_str = str(header)
        cipos_line = next(l for l in header_str.splitlines() if 'ID=CIPOS' in l)
        assert 'Number=.' in cipos_line, (
            "CIPOS INFO header must declare Number=. per VCF 4.5 §3"
        )

    def test_ciend_info_number_is_dot_per_spec(self):
        """VCF 4.5 §3: CIEND must be Number=. (variable, 2 values per ALT allele)."""
        df = _df(_del_row('ev1', 100, 500))
        header = _build_header(df)
        header_str = str(header)
        ciend_line = next(l for l in header_str.splitlines() if 'ID=CIEND' in l)
        assert 'Number=.' in ciend_line, (
            "CIEND INFO header must declare Number=. per VCF 4.5 §3"
        )


# ---------------------------------------------------------------------------
# write_vcf — symbolic records (DEL / DUP / INS)
# ---------------------------------------------------------------------------

class TestWriteSymbolicRecord:
    """Verify field values for DEL, DUP, and INS symbolic-allele records."""

    def test_del_svtype_info_field(self, tmp_path):
        df = _df(_del_row('ev1', 100, 500))
        write_vcf(df, tmp_path / 'out.vcf')
        records = _read_records(tmp_path / 'out.vcf')
        assert len(records) == 1
        assert records[0].info['SVTYPE'] == 'DEL'

    def test_del_alt_is_symbolic_allele(self, tmp_path):
        df = _df(_del_row('ev1', 100, 500))
        write_vcf(df, tmp_path / 'out.vcf')
        records = _read_records(tmp_path / 'out.vcf')
        assert records[0].alts == ('<DEL>',)

    def test_del_svlen_equals_span(self, tmp_path):
        span = 500
        df = _df(_del_row('ev1', 100, span))
        write_vcf(df, tmp_path / 'out.vcf')
        records = _read_records(tmp_path / 'out.vcf')
        assert records[0].info['SVLEN'][0] == span

    def test_del_svlen_is_positive(self, tmp_path):
        df = _df(_del_row('ev1', 100, 500))
        write_vcf(df, tmp_path / 'out.vcf')
        records = _read_records(tmp_path / 'out.vcf')
        assert records[0].info['SVLEN'][0] > 0

    def test_del_filter_is_pass(self, tmp_path):
        df = _df(_del_row('ev1', 100, 500))
        write_vcf(df, tmp_path / 'out.vcf')
        records = _read_records(tmp_path / 'out.vcf')
        assert list(records[0].filter) == ['PASS']

    def test_del_record_id_matches_event(self, tmp_path):
        df = _df(_del_row('ev_DEL_001', 100, 500))
        write_vcf(df, tmp_path / 'out.vcf')
        records = _read_records(tmp_path / 'out.vcf')
        assert records[0].id == 'ev_DEL_001'

    def test_del_pos_and_stop_match_bnd_positions(self, tmp_path):
        pos_bnd1, span = 100, 500
        df = _df(_del_row('ev1', pos_bnd1, span))
        write_vcf(df, tmp_path / 'out.vcf')
        records = _read_records(tmp_path / 'out.vcf')
        r = records[0]
        assert r.pos == pos_bnd1
        assert r.stop == pos_bnd1 + span

    def test_dup_svtype_and_alt(self, tmp_path):
        pos_bnd1, pos_bnd2 = 200, 700
        row = _row(
            'ev_DUP', 'chr1', pos_bnd1, f'N[chr1:{pos_bnd2}[',
                       'chr1', pos_bnd2, f']chr1:{pos_bnd1}]N',
            svtype='DUP',
        )
        df = _df(row)
        write_vcf(df, tmp_path / 'out.vcf')
        records = _read_records(tmp_path / 'out.vcf')
        assert len(records) == 1
        assert records[0].info['SVTYPE'] == 'DUP'
        assert records[0].alts == ('<DUP>',)
        assert records[0].info['SVLEN'][0] == pos_bnd2 - pos_bnd1

    def test_ins_svlen_taken_from_info_svlen(self, tmp_path):
        row = _row(
            'ev_INS', 'chr1', 300, 'N[chr1:301[',
                       'chr1', 301, ']chr1:300]N',
            svtype='INS', svlen=350,
        )
        df = _df(row)
        write_vcf(df, tmp_path / 'out.vcf')
        records = _read_records(tmp_path / 'out.vcf')
        assert records[0].info['SVTYPE'] == 'INS'
        assert records[0].info['SVLEN'][0] == 350

    def test_ins_svlen_absent_when_info_svlen_missing(self, tmp_path):
        """When no INFO_SVLEN is provided, SVLEN should not appear in the record."""
        row = _row(
            'ev_INS', 'chr1', 300, 'N[chr1:301[',
                       'chr1', 301, ']chr1:300]N',
            svtype='INS',
        )
        df = _df(row)
        write_vcf(df, tmp_path / 'out.vcf')
        records = _read_records(tmp_path / 'out.vcf')
        assert 'SVLEN' not in records[0].info

    def test_imprecise_flag_set_when_cipos_nonzero(self, tmp_path):
        df = _df(_del_row('ev1', 100, 500, cipos=10, ciend=10))
        write_vcf(df, tmp_path / 'out.vcf')
        records = _read_records(tmp_path / 'out.vcf')
        assert records[0].info['IMPRECISE'] is True

    def test_imprecise_flag_set_when_only_ciend_nonzero(self, tmp_path):
        """IMPRECISE must be set even when only CIEND is non-zero."""
        df = _df(_del_row('ev1', 100, 500, cipos=0, ciend=8))
        write_vcf(df, tmp_path / 'out.vcf')
        records = _read_records(tmp_path / 'out.vcf')
        assert records[0].info['IMPRECISE'] is True

    def test_imprecise_flag_absent_when_all_ci_zero(self, tmp_path):
        df = _df(_del_row('ev1', 100, 500, cipos=0, ciend=0))
        write_vcf(df, tmp_path / 'out.vcf')
        records = _read_records(tmp_path / 'out.vcf')
        assert 'IMPRECISE' not in records[0].info

    def test_cipos_and_ciend_written_when_ci_present(self, tmp_path):
        df = _df(_del_row('ev1', 100, 500, cipos=5, ciend=8))
        write_vcf(df, tmp_path / 'out.vcf')
        records = _read_records(tmp_path / 'out.vcf')
        assert records[0].info['CIPOS'] == (-5, 5)
        assert records[0].info['CIEND'] == (-8, 8)


# ---------------------------------------------------------------------------
# write_vcf — BND records (OTHER svtype)
# ---------------------------------------------------------------------------

class TestWriteBndRecords:
    """Verify BND pair produced for unclassified SVs."""

    def test_two_bnd_records_written_per_row(self, tmp_path):
        df = _df(_bnd_row('ev_BND_001', 500))
        write_vcf(df, tmp_path / 'out.vcf')
        records = _read_records(tmp_path / 'out.vcf')
        assert len(records) == 2

    def test_bnd_svtype_info_field(self, tmp_path):
        df = _df(_bnd_row('ev_BND_001', 500))
        write_vcf(df, tmp_path / 'out.vcf')
        records = _read_records(tmp_path / 'out.vcf')
        for r in records:
            assert r.info['SVTYPE'] == 'BND'

    def test_bnd_mateid_cross_references(self, tmp_path):
        df = _df(_bnd_row('ev_BND_001', 500))
        write_vcf(df, tmp_path / 'out.vcf')
        records = _read_records(tmp_path / 'out.vcf')
        id_to_mateid = {r.id: r.info['MATEID'] for r in records}
        r1_id, r2_id = [r.id for r in records]
        assert id_to_mateid[r1_id] == r2_id
        assert id_to_mateid[r2_id] == r1_id

    def test_bnd_event_field_set(self, tmp_path):
        df = _df(_bnd_row('ev_BND_001', 500))
        write_vcf(df, tmp_path / 'out.vcf')
        records = _read_records(tmp_path / 'out.vcf')
        for r in records:
            assert r.info['EVENT'] == 'ev_BND_001'

    def test_bnd_ids_use_event_bnd1_bnd2_suffix(self, tmp_path):
        df = _df(_bnd_row('ev_BND_001', 500))
        write_vcf(df, tmp_path / 'out.vcf')
        records = _read_records(tmp_path / 'out.vcf')
        ids = {r.id for r in records}
        assert ids == {'ev_BND_001_bnd1', 'ev_BND_001_bnd2'}

    def test_bnd_alt_strings_preserved_from_dataframe(self, tmp_path):
        alt1, alt2 = 'N[chr2:600[', ']chr1:500]N'
        row = _row('ev1', 'chr1', 500, alt1, 'chr2', 600, alt2, svtype='OTHER')
        df = _df(row)
        write_vcf(df, tmp_path / 'out.vcf')
        records = _read_records(tmp_path / 'out.vcf')
        alts = {r.alts[0] for r in records}
        assert alt1 in alts
        assert alt2 in alts

    def test_bnd_filter_is_pass(self, tmp_path):
        df = _df(_bnd_row('ev_BND_001', 500))
        write_vcf(df, tmp_path / 'out.vcf')
        records = _read_records(tmp_path / 'out.vcf')
        for r in records:
            assert list(r.filter) == ['PASS']


# ---------------------------------------------------------------------------
# write_vcf — INV handling (symbolic allele)
# ---------------------------------------------------------------------------

class TestWriteVcfInvSymbolic:
    """INV written as a single <INV> symbolic record (default inv_as_bnd=False)."""

    def test_two_inv_rows_produce_one_record(self, tmp_path):
        df = _df(
            _inv_e1_row('inv1', 100, 500),
            _inv_e2_row('inv1', 100, 500),
        )
        write_vcf(df, tmp_path / 'out.vcf')
        records = _read_records(tmp_path / 'out.vcf')
        assert len(records) == 1

    def test_inv_svtype_info_field(self, tmp_path):
        df = _df(
            _inv_e1_row('inv1', 100, 500),
            _inv_e2_row('inv1', 100, 500),
        )
        write_vcf(df, tmp_path / 'out.vcf')
        records = _read_records(tmp_path / 'out.vcf')
        assert records[0].info['SVTYPE'] == 'INV'

    def test_inv_alt_is_symbolic_allele(self, tmp_path):
        df = _df(
            _inv_e1_row('inv1', 100, 500),
            _inv_e2_row('inv1', 100, 500),
        )
        write_vcf(df, tmp_path / 'out.vcf')
        records = _read_records(tmp_path / 'out.vcf')
        assert records[0].alts == ('<INV>',)

    def test_inv_record_id_has_no_e1_or_e2_suffix(self, tmp_path):
        df = _df(
            _inv_e1_row('inv1', 100, 500),
            _inv_e2_row('inv1', 100, 500),
        )
        write_vcf(df, tmp_path / 'out.vcf')
        records = _read_records(tmp_path / 'out.vcf')
        assert records[0].id == 'inv1'

    def test_inv_svlen_equals_span_of_e1_row(self, tmp_path):
        svlen = 500
        df = _df(
            _inv_e1_row('inv1', 100, svlen),
            _inv_e2_row('inv1', 100, svlen),
        )
        write_vcf(df, tmp_path / 'out.vcf')
        records = _read_records(tmp_path / 'out.vcf')
        # span of _e1 row = (100 + svlen) - 100 = svlen
        assert records[0].info['SVLEN'][0] == svlen

    def test_inv_svlen_is_positive(self, tmp_path):
        df = _df(
            _inv_e1_row('inv1', 100, 500),
            _inv_e2_row('inv1', 100, 500),
        )
        write_vcf(df, tmp_path / 'out.vcf')
        records = _read_records(tmp_path / 'out.vcf')
        assert records[0].info['SVLEN'][0] > 0

    def test_two_independent_invs_produce_two_records(self, tmp_path):
        df = _df(
            _inv_e1_row('inv1', 100, 500),
            _inv_e2_row('inv1', 100, 500),
            _inv_e1_row('inv2', 1000, 200),
            _inv_e2_row('inv2', 1000, 200),
        )
        write_vcf(df, tmp_path / 'out.vcf')
        records = _read_records(tmp_path / 'out.vcf')
        assert len(records) == 2
        record_ids = {r.id for r in records}
        assert record_ids == {'inv1', 'inv2'}

    def test_e1_row_alone_still_produces_one_record(self, tmp_path):
        """Robustness: a single _e1 row without a matching _e2 is handled."""
        df = _df(_inv_e1_row('inv1', 100, 500))
        write_vcf(df, tmp_path / 'out.vcf')
        records = _read_records(tmp_path / 'out.vcf')
        assert len(records) == 1
        assert records[0].id == 'inv1'

    def test_inv_symbolic_spec_example(self, tmp_path):
        """VCF 4.5 §5.4.7: reproduce the spec INV example.

        Spec symbolic record: POS=321681, SVLEN=100000, END=421681.
        Using _e1 row: POS_bnd1=321681, POS_bnd2=421681 → SVLEN=100000.
        """
        pos, svlen = 321681, 100000
        df = _df(
            _inv_e1_row('INV0', pos, svlen, chrom='chr2'),
            _inv_e2_row('INV0', pos, svlen, chrom='chr2'),
        )
        write_vcf(df, tmp_path / 'out.vcf')
        records = _read_records(tmp_path / 'out.vcf')
        assert len(records) == 1
        r = records[0]
        assert r.info['SVTYPE'] == 'INV'
        assert r.info['SVLEN'][0] == svlen          # 100000
        assert r.pos == pos                          # 321681
        assert r.stop == pos + svlen                 # 421681 = END


# ---------------------------------------------------------------------------
# write_vcf — INV handling (BND mode)
# ---------------------------------------------------------------------------

class TestWriteVcfInvAsBnd:
    """INV written as 4 BND records when inv_as_bnd=True."""

    def test_two_inv_rows_produce_four_bnd_records(self, tmp_path):
        df = _df(
            _inv_e1_row('inv1', 100, 500),
            _inv_e2_row('inv1', 100, 500),
        )
        write_vcf(df, tmp_path / 'out.vcf', inv_as_bnd=True)
        records = _read_records(tmp_path / 'out.vcf')
        assert len(records) == 4

    def test_all_bnd_records_have_svtype_bnd(self, tmp_path):
        df = _df(
            _inv_e1_row('inv1', 100, 500),
            _inv_e2_row('inv1', 100, 500),
        )
        write_vcf(df, tmp_path / 'out.vcf', inv_as_bnd=True)
        records = _read_records(tmp_path / 'out.vcf')
        for r in records:
            assert r.info['SVTYPE'] == 'BND'

    def test_all_four_bnd_records_share_same_event_id(self, tmp_path):
        """All 4 breakends of an inversion must share the base EVENT, not _e1/_e2."""
        df = _df(
            _inv_e1_row('inv1', 100, 500),
            _inv_e2_row('inv1', 100, 500),
        )
        write_vcf(df, tmp_path / 'out.vcf', inv_as_bnd=True)
        records = _read_records(tmp_path / 'out.vcf')
        events = {r.info['EVENT'] for r in records}
        assert events == {'inv1'}, (
            f"All 4 BND records should have EVENT='inv1', got: {events}"
        )

    def test_bnd_mateid_cross_references_within_pair(self, tmp_path):
        """Each BND record's MATEID must point to its intra-event partner.

        _e1_bnd1 <-> _e1_bnd2  and  _e2_bnd1 <-> _e2_bnd2.
        Cross-event mating (_e1_bnd1 <-> _e2_bnd2) must not occur.
        """
        df = _df(
            _inv_e1_row('inv1', 100, 500),
            _inv_e2_row('inv1', 100, 500),
        )
        write_vcf(df, tmp_path / 'out.vcf', inv_as_bnd=True)
        records = _read_records(tmp_path / 'out.vcf')
        id_to_mateid = {r.id: r.info['MATEID'] for r in records}
        all_ids = set(id_to_mateid.keys())

        for own_id, mate_id in id_to_mateid.items():
            # mate must exist
            assert mate_id in all_ids, (
                f"MATEID '{mate_id}' of record '{own_id}' not found in output"
            )
            # mating is symmetric
            assert id_to_mateid[mate_id] == own_id, (
                f"MATEID relationship is not symmetric for '{own_id}' <-> '{mate_id}'"
            )
            # mating is intra-event: the _eN prefix must match between partners
            # e.g. "inv1_e1_bnd1" must mate with "inv1_e1_bnd2", not "inv1_e2_*"
            own_event_prefix = own_id.rsplit('_bnd', 1)[0]
            mate_event_prefix = mate_id.rsplit('_bnd', 1)[0]
            assert own_event_prefix == mate_event_prefix, (
                f"Cross-event mating detected: '{own_id}' <-> '{mate_id}'"
            )

    def test_bnd_alt_strings_preserved_for_inv_breakends(self, tmp_path):
        """ALT strings in the _e1 / _e2 rows must appear in the output BND ALTs."""
        pos, svlen = 100, 500
        e1 = _inv_e1_row('inv1', pos, svlen)
        e2 = _inv_e2_row('inv1', pos, svlen)
        expected_alts = {
            e1['ALT_bnd1'], e1['ALT_bnd2'],
            e2['ALT_bnd1'], e2['ALT_bnd2'],
        }
        df = _df(e1, e2)
        write_vcf(df, tmp_path / 'out.vcf', inv_as_bnd=True)
        records = _read_records(tmp_path / 'out.vcf')
        written_alts = {r.alts[0] for r in records}
        assert written_alts == expected_alts

    def test_inv_bnd_mode_two_invs_produce_eight_records(self, tmp_path):
        df = _df(
            _inv_e1_row('inv1', 100, 500),
            _inv_e2_row('inv1', 100, 500),
            _inv_e1_row('inv2', 1000, 200),
            _inv_e2_row('inv2', 1000, 200),
        )
        write_vcf(df, tmp_path / 'out.vcf', inv_as_bnd=True)
        records = _read_records(tmp_path / 'out.vcf')
        assert len(records) == 8

    def test_inv_bnd_four_positions_match_spec(self, tmp_path):
        """VCF 4.5 §5.4.7: the 4 BND records for an inversion at POS with SVLEN
        must be at POS, POS+1, POS+SVLEN, POS+SVLEN+1 — exactly the positions
        shown in the spec breakend table.
        """
        pos, svlen = 321681, 100000
        df = _df(
            _inv_e1_row('INV0', pos, svlen, chrom='chr2'),
            _inv_e2_row('INV0', pos, svlen, chrom='chr2'),
        )
        write_vcf(df, tmp_path / 'out.vcf', inv_as_bnd=True)
        records = _read_records(tmp_path / 'out.vcf')
        positions = sorted(r.pos for r in records)
        assert positions == [pos, pos + 1, pos + svlen, pos + svlen + 1], (
            f"Expected BND positions {[pos, pos+1, pos+svlen, pos+svlen+1]}, "
            f"got {positions}"
        )

    def test_inv_bnd_alt_encoding_per_spec(self, tmp_path):
        """VCF 4.5 §5.4.7: the ]..] pair (encoding 3) sits at POS and POS+SVLEN;
        the [..[ pair (encoding 4) sits at POS+1 and POS+SVLEN+1.
        """
        pos, svlen = 321681, 100000
        df = _df(
            _inv_e1_row('INV0', pos, svlen, chrom='chr2'),
            _inv_e2_row('INV0', pos, svlen, chrom='chr2'),
        )
        write_vcf(df, tmp_path / 'out.vcf', inv_as_bnd=True)
        records = _read_records(tmp_path / 'out.vcf')

        pos_to_alt = {r.pos: r.alts[0] for r in records}

        # Encoding 3 (]...] — same-strand): positions POS and POS+SVLEN
        assert pos_to_alt[pos].startswith('N]') and pos_to_alt[pos].endswith(']'), (
            f"BND at POS={pos} should be ]...] (encoding 3), got {pos_to_alt[pos]!r}"
        )
        assert pos_to_alt[pos + svlen].startswith('N]') and pos_to_alt[pos + svlen].endswith(']'), (
            f"BND at POS={pos+svlen} should be ]...] (encoding 3), "
            f"got {pos_to_alt[pos+svlen]!r}"
        )

        # Encoding 4 ([...[ — same-strand): positions POS+1 and POS+SVLEN+1
        assert pos_to_alt[pos + 1].startswith('[') and pos_to_alt[pos + 1].endswith('[N'), (
            f"BND at POS={pos+1} should be [...[N (encoding 4), "
            f"got {pos_to_alt[pos+1]!r}"
        )
        assert pos_to_alt[pos + svlen + 1].startswith('[') and pos_to_alt[pos + svlen + 1].endswith('[N'), (
            f"BND at POS={pos+svlen+1} should be [...[N (encoding 4), "
            f"got {pos_to_alt[pos+svlen+1]!r}"
        )


# ---------------------------------------------------------------------------
# write_vcf — integration / edge cases
# ---------------------------------------------------------------------------

class TestWriteVcf:
    def test_empty_dataframe_produces_valid_vcf(self, tmp_path):
        df = pd.DataFrame(columns=[
            'CHROM_bnd1', 'POS_bnd1', 'ALT_bnd1',
            'CHROM_bnd2', 'POS_bnd2', 'ALT_bnd2',
            'INFO_EVENT',
        ])
        path = write_vcf(df, tmp_path / 'empty.vcf')
        records = _read_records(path)
        assert records == []

    def test_mixed_svtypes_produce_correct_record_counts(self, tmp_path):
        df = _df(
            _del_row('ev_del', 100, 500),
            _bnd_row('ev_bnd', 200),
            _inv_e1_row('ev_inv', 300, 400),
            _inv_e2_row('ev_inv', 300, 400),
        )
        write_vcf(df, tmp_path / 'out.vcf')
        records = _read_records(tmp_path / 'out.vcf')
        # 1 DEL + 2 BND + 1 INV (deduplicated)
        assert len(records) == 4

    def test_missing_required_column_raises_value_error(self, tmp_path):
        df = pd.DataFrame([{'CHROM_bnd1': 'chr1', 'POS_bnd1': 100}])
        with pytest.raises(ValueError, match='Missing required columns'):
            write_vcf(df, tmp_path / 'out.vcf')

    def test_returns_output_path(self, tmp_path):
        df = _df(_del_row('ev1', 100, 500))
        out = tmp_path / 'out.vcf'
        result = write_vcf(df, out)
        assert result == out

    def test_records_sorted_by_pos_within_chromosome(self, tmp_path):
        df = _df(
            _del_row('ev2', 500, 100, chrom='chr1'),
            _del_row('ev1', 100, 100, chrom='chr1'),
        )
        write_vcf(df, tmp_path / 'out.vcf')
        records = _read_records(tmp_path / 'out.vcf')
        positions = [r.pos for r in records]
        assert positions == sorted(positions)

    def test_records_sorted_across_chromosomes(self, tmp_path):
        """Standard chromosome order (chr1 < chr2 < chr10 < chrX) is respected."""
        df = _df(
            _del_row('ev_x',  100, 100, chrom='chrX'),
            _del_row('ev_10', 100, 100, chrom='chr10'),
            _del_row('ev_2',  100, 100, chrom='chr2'),
            _del_row('ev_1',  100, 100, chrom='chr1'),
        )
        write_vcf(df, tmp_path / 'out.vcf')
        records = _read_records(tmp_path / 'out.vcf')
        chroms = [r.contig for r in records]
        assert chroms == ['chr1', 'chr2', 'chr10', 'chrX']

    def test_sample_name_appears_in_header(self, tmp_path):
        df = _df(_del_row('ev1', 100, 500))
        write_vcf(df, tmp_path / 'out.vcf', sample_name='NA12878')
        with pysam.VariantFile(str(tmp_path / 'out.vcf')) as vcf:
            assert 'NA12878' in vcf.header.samples

    def test_source_appears_in_header(self, tmp_path):
        df = _df(_del_row('ev1', 100, 500))
        write_vcf(df, tmp_path / 'out.vcf', source='my_caller')
        with pysam.VariantFile(str(tmp_path / 'out.vcf')) as vcf:
            assert '##source=my_caller' in str(vcf.header)



class TestWriteVcfAtomic:
    def test_failure_mid_write_leaves_no_file(self, tmp_path, monkeypatch):
        import svensemble.vcfwrite as vw

        def boom(*a, **k):
            raise RuntimeError("boom")

        monkeypatch.setattr(vw, "_write_symbolic_record", boom)
        with pytest.raises(RuntimeError, match="boom"):
            write_vcf(_df(_del_row('ev1', 100, 500)), tmp_path / 'out.vcf')
        assert list(tmp_path.iterdir()) == []

    def test_failure_keeps_previous_file(self, tmp_path, monkeypatch):
        import svensemble.vcfwrite as vw

        out = tmp_path / 'out.vcf'
        write_vcf(_df(_del_row('ev1', 100, 500)), out)
        before = out.read_bytes()
        monkeypatch.setattr(vw, "_write_symbolic_record", lambda *a, **k: (_ for _ in ()).throw(RuntimeError()))
        with pytest.raises(RuntimeError):
            write_vcf(_df(_del_row('ev2', 900, 500)), out)
        assert out.read_bytes() == before
        assert [p.name for p in tmp_path.iterdir()] == ['out.vcf']

    def test_gz_suffix_still_bgzipped(self, tmp_path):
        out = write_vcf(_df(_del_row('ev1', 100, 500)), tmp_path / 'out.vcf.gz')
        assert out == tmp_path / 'out.vcf.gz'
        assert out.read_bytes()[:2] == b'\x1f\x8b'
        assert len(_read_records(out)) == 1


# ---------------------------------------------------------------------------
# write_vcf — per-sample scores (calls VCF)
# ---------------------------------------------------------------------------

class TestWriteScores:
    PRED = 'ts_model_pred'

    def _calls(self, rows):
        from svensemble._constants import SAMPLE_ID_COL
        return pd.DataFrame([
            _del_row(ev, pos, end) | {SAMPLE_ID_COL: sample, self.PRED: score}
            for ev, pos, end, sample, score in rows
        ])

    def test_one_record_per_sv_with_per_sample_format(self, tmp_path):
        df = self._calls([
            ('sv1', 100, 500, 'S2', 0.25), ('sv1', 100, 500, 'S1', 0.9), ('sv2', 2000, 2600, 'S1', 0.5),
        ])
        write_vcf(df, tmp_path / 'out.vcf', score_col=self.PRED, group_by='INFO_EVENT')
        with pysam.VariantFile(str(tmp_path / 'out.vcf')) as vcf:
            assert list(vcf.header.samples) == ['S1', 'S2']
            recs = list(vcf)
        assert [r.id for r in recs] == ['sv1', 'sv2']
        assert recs[0].samples['S1']['SVENSSC'] == pytest.approx(0.9)
        assert recs[0].samples['S2']['SVENSSC'] == pytest.approx(0.25)
        assert recs[1].samples['S2']['SVENSSC'] is None

    def test_samples_disagreeing_on_the_site_raise(self, tmp_path):
        df = self._calls([('sv1', 100, 500, 'S1', 0.9), ('sv1', 110, 500, 'S2', 0.2)])
        with pytest.raises(ValueError, match='disagree'):
            write_vcf(df, tmp_path / 'out.vcf', score_col=self.PRED, group_by='INFO_EVENT')

    def test_several_samples_without_grouping_raise(self, tmp_path):
        df = self._calls([('sv1', 100, 500, 'S1', 0.9), ('sv1', 100, 500, 'S2', 0.2)])
        with pytest.raises(ValueError, match='no group_by'):
            write_vcf(df, tmp_path / 'out.vcf', score_col=self.PRED)

    def test_one_sample_without_grouping_one_record_per_row(self, tmp_path):
        df = self._calls([('sv1', 100, 500, 'S1', 0.9), ('sv2', 2000, 2600, 'S1', 0.2)])
        write_vcf(df, tmp_path / 'out.vcf', score_col=self.PRED)
        with pysam.VariantFile(str(tmp_path / 'out.vcf')) as vcf:
            assert list(vcf.header.samples) == ['S1']
            assert [r.id for r in vcf] == ['sv1', 'sv2']

    def test_missing_group_by_column_raises(self, tmp_path):
        df = self._calls([('sv1', 100, 500, 'S1', 0.9)])
        with pytest.raises(KeyError, match='svensemble_id'):
            write_vcf(df, tmp_path / 'out.vcf', score_col=self.PRED, group_by='svensemble_id')

    def test_without_sample_column_scores_go_to_sample_name(self, tmp_path):
        df = pd.DataFrame([_del_row('sv1', 100, 500) | {self.PRED: 0.4}])
        write_vcf(df, tmp_path / 'out.vcf', sample_name='NA12878', score_col=self.PRED)
        with pysam.VariantFile(str(tmp_path / 'out.vcf')) as vcf:
            assert list(vcf.header.samples) == ['NA12878']
            assert next(vcf).samples['NA12878']['SVENSSC'] == pytest.approx(0.4)

    def test_callset_rows_per_sample_written_once(self, tmp_path):
        from svensemble._constants import SAMPLE_ID_COL
        df = pd.DataFrame([_del_row('sv1', 100, 500) | {SAMPLE_ID_COL: s} for s in ('S1', 'S2')])
        write_vcf(df, tmp_path / 'out.vcf', sample_name='S', group_by='INFO_EVENT')
        assert len(_read_records(tmp_path / 'out.vcf')) == 1
        with pysam.VariantFile(str(tmp_path / 'out.vcf')) as vcf:
            assert list(vcf.header.samples) == ['S1', 'S2']


def test_header_lines_written(tmp_path):
    df = _df(_del_row('ev1', 100, 500))
    write_vcf(df, tmp_path / 'out.vcf', header_lines=['##svensemble_version=9.9', '##svensemble_model_id=m-1'])
    with pysam.VariantFile(str(tmp_path / 'out.vcf')) as vcf:
        header = str(vcf.header)
    assert '##svensemble_version=9.9' in header and '##svensemble_model_id=m-1' in header
