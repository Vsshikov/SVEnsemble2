"""Tests for callset-specific preprocessing functions in svensemble/callset_specific.py.

Each test class validates both transform correctness and output DataFrame schema
against the downstream contracts (event_to_bnd, bnd_to_bp).
"""

import numpy as np
import pandas as pd
import pytest

from svensemble import breakends as bnds
from svensemble import callset_specific as cs
from svensemble import io as sv_io
from svensemble import _caller_params

from .conftest import (
    assert_has_columns,
    assert_missing_columns,
    assert_column_dtypes,
    make_gridss_bnd_df,
    make_manta_evt_df,
    make_manta_bnd_df,
    make_delly_evt_df,
    make_delly_bnd_df,
    delly_bnd_alt,
    write_delly_vcf,
    make_sniffles_evt_df,
    make_sniffles_bnd_df,
    make_survivor_evt_df,
    make_svaba_bnd_df,
    EVENT_TO_BND_REQUIRED_COLS,
    EVENT_TO_BND_REQUIRED_DTYPES,
    BND_TO_BP_REQUIRED_COLS,
    CIPOS_RAW_COLS,
)


# =====================================================================
# GRIDSS
# =====================================================================

class TestGridssPreprocess:
    def test_cipos_split(self):
        df = make_gridss_bnd_df()
        result = cs.gridss_preprocess(df)
        assert result['INFO_CIPOS_l'].iloc[0] == -10
        assert result['INFO_CIPOS_r'].iloc[0] == 10

    def test_cipos_nan_uses_default(self):
        df = make_gridss_bnd_df()
        df.loc[0, 'INFO_CIPOS'] = None
        result = cs.gridss_preprocess(df)
        assert result['INFO_CIPOS_l'].iloc[0] == 0
        assert result['INFO_CIPOS_r'].iloc[0] == 0

    def test_schema_has_split_cols(self):
        result = cs.gridss_preprocess(make_gridss_bnd_df())
        assert_has_columns(result, ['INFO_CIPOS_l', 'INFO_CIPOS_r'])
        assert_column_dtypes(result, {'INFO_CIPOS_l': 'int', 'INFO_CIPOS_r': 'int'})

    def test_schema_no_raw_cipos(self):
        result = cs.gridss_preprocess(make_gridss_bnd_df())
        assert_missing_columns(result, ['INFO_CIPOS'])

    def test_input_not_mutated(self):
        df = make_gridss_bnd_df()
        original_cols = set(df.columns)
        cs.gridss_preprocess(df)
        assert set(df.columns) == original_cols
        assert 'INFO_CIPOS' in df.columns


class TestGridssPostBps:
    def _make_bp_df(self, qual1=100.0, qual2=80.0, fp1=True, fp2=False):
        return pd.DataFrame({
            'CHROM_bnd1': ['chr1'], 'POS_bnd1': [1000],
            'CHROM_bnd2': ['chr1'], 'POS_bnd2': [2000],
            'QUAL_bnd1': [qual1], 'QUAL_bnd2': [qual2],
            'FILTER_PASS_bnd1': [fp1], 'FILTER_PASS_bnd2': [fp2],
            'INFO_EVENT': ['ev1'],
        })

    def test_qual_unified_to_min(self):
        result = cs.gridss_post_bps(self._make_bp_df(100.0, 80.0))
        assert result['QUAL'].iloc[0] == 80.0

    def test_filter_pass_and_logic(self):
        result = cs.gridss_post_bps(self._make_bp_df(fp1=True, fp2=False))
        assert result['FILTER_PASS'].iloc[0] == False  # noqa: E712

    def test_filter_pass_both_true(self):
        result = cs.gridss_post_bps(self._make_bp_df(fp1=True, fp2=True))
        assert result['FILTER_PASS'].iloc[0] == True  # noqa: E712

    def test_schema_unified_columns(self):
        result = cs.gridss_post_bps(self._make_bp_df())
        assert_has_columns(result, ['QUAL', 'FILTER_PASS'])
        assert_missing_columns(result, ['FILTER_PASS_bnd1', 'FILTER_PASS_bnd2'])

    def test_idempotency_already_unified(self):
        df = pd.DataFrame({
            'CHROM_bnd1': ['chr1'], 'POS_bnd1': [1000],
            'QUAL': [50.0], 'FILTER_PASS': [True],
            'INFO_EVENT': ['ev1'],
        })
        result = cs.gridss_post_bps(df)
        assert result['QUAL'].iloc[0] == 50.0
        assert result['FILTER_PASS'].iloc[0] == True  # noqa: E712

    def test_missing_qual_raises(self):
        df = pd.DataFrame({
            'CHROM_bnd1': ['chr1'], 'INFO_EVENT': ['ev1'],
        })
        with pytest.raises(KeyError, match='No QUAL column'):
            cs.gridss_post_bps(df)


# =====================================================================
# MANTA
# =====================================================================

class TestMantaPreprocessEvtsOnly:
    def test_ciend_nan_filled_from_cipos(self):
        df = make_manta_evt_df()
        df.loc[0, 'INFO_CIEND'] = None
        result = cs.manta_preprocess_evts_only(df)
        assert 'INFO_CIEND_l' in result.columns
        assert not pd.isna(result['INFO_CIEND_l'].iloc[0])

    def test_svlen_tuple_to_scalar(self):
        df = make_manta_evt_df()
        result = cs.manta_preprocess_evts_only(df)
        for val in result['INFO_SVLEN']:
            assert not isinstance(val, tuple)

    def test_del_svlen_positive(self):
        df = make_manta_evt_df()
        result = cs.manta_preprocess_evts_only(df)
        del_rows = result[result['INFO_SVTYPE'] == 'DEL']
        assert (del_rows['INFO_SVLEN'] >= 0).all()

    def test_info_event_set_to_id(self):
        df = make_manta_evt_df()
        result = cs.manta_preprocess_evts_only(df)
        assert (result['INFO_EVENT'] == result['ID']).all()

    def test_schema_event_to_bnd_contract(self):
        result = cs.manta_preprocess_evts_only(make_manta_evt_df())
        assert_has_columns(result, EVENT_TO_BND_REQUIRED_COLS)
        ci_dtypes = {k: v for k, v in EVENT_TO_BND_REQUIRED_DTYPES.items()
                     if k != 'INFO_SVLEN'}
        assert_column_dtypes(result, ci_dtypes)
        assert pd.to_numeric(result['INFO_SVLEN'], errors='coerce').notna().all(), \
            "INFO_SVLEN values must be numeric"

    def test_schema_no_raw_cipos(self):
        result = cs.manta_preprocess_evts_only(make_manta_evt_df())
        assert_missing_columns(result, CIPOS_RAW_COLS + ['INFO_POS2'])

    def test_chr2_is_str(self):
        result = cs.manta_preprocess_evts_only(make_manta_evt_df())
        assert result['INFO_CHR2'].dtype == object


class TestMantaPreprocessBndsOnly:
    def test_cipos_split(self):
        df = make_manta_bnd_df()
        result = cs.manta_preprocess_bnds_only(df)
        assert_has_columns(result, ['INFO_CIPOS_l', 'INFO_CIPOS_r'])

    def test_schema_bnd_to_bp_contract(self):
        df = make_manta_bnd_df()
        result = cs.manta_preprocess_bnds_only(df)
        assert_has_columns(result, BND_TO_BP_REQUIRED_COLS + ['INFO_CIPOS_l', 'INFO_CIPOS_r'])
        assert_missing_columns(result, CIPOS_RAW_COLS)

    def test_info_pos2_dropped(self):
        result = cs.manta_preprocess_bnds_only(make_manta_bnd_df())
        assert_missing_columns(result, ['INFO_POS2'])

    def test_info_event_derived_from_mateid(self):
        """INFO_EVENT must be non-null and shared between paired BND records."""
        df = make_manta_bnd_df(n=4)
        result = cs.manta_preprocess_bnds_only(df)
        assert result['INFO_EVENT'].notna().all(), "INFO_EVENT must not be null after preprocessing"
        # Pairs: (0,1) and (2,3) must share the same INFO_EVENT within each pair.
        assert result.iloc[0]['INFO_EVENT'] == result.iloc[1]['INFO_EVENT']
        assert result.iloc[2]['INFO_EVENT'] == result.iloc[3]['INFO_EVENT']
        assert result.iloc[0]['INFO_EVENT'] != result.iloc[2]['INFO_EVENT']

    def test_info_event_is_min_of_id_and_mateid(self):
        """Shared key must equal min(ID, MATEID) so bnd_to_bp can form pairs."""
        df = make_manta_bnd_df(n=2)
        result = cs.manta_preprocess_bnds_only(df)
        expected = min('manta_bnd_0', 'manta_bnd_1')
        assert (result['INFO_EVENT'] == expected).all()

    def test_info_ciend_dropped(self):
        result = cs.manta_preprocess_bnds_only(make_manta_bnd_df())
        assert_missing_columns(result, ['INFO_CIEND'])


class TestMantaPreprocessCommon:
    def test_format_splitting_enabled(self):
        params = _caller_params.MANTA_DIPLOID_PARAMS
        df = make_manta_evt_df()
        result = cs.manta_preprocess_evts_only(df, caller_params=params)
        assert_has_columns(result, ['SAMPLE_SR_REF', 'SAMPLE_SR_ALT', 'SAMPLE_PR_REF', 'SAMPLE_PR_ALT'])
        assert_missing_columns(result, ['SAMPLE_SR', 'SAMPLE_PR'])

    def test_format_splitting_disabled(self):
        params = _caller_params.MANTA_CANDIDATE_PARAMS
        df = make_manta_evt_df()
        result = cs.manta_preprocess_evts_only(df, caller_params=params)
        assert_missing_columns(result, ['SAMPLE_SR_REF', 'SAMPLE_SR_ALT'])


# =====================================================================
# DELLY
# =====================================================================

class TestDellyPreprocessEvtsOnly:
    def test_cipos_ciend_split(self):
        result = cs.delly_preprocess_evts_only(make_delly_evt_df())
        assert_has_columns(result, ['INFO_CIPOS_l', 'INFO_CIPOS_r', 'INFO_CIEND_l', 'INFO_CIEND_r'])
        assert_missing_columns(result, CIPOS_RAW_COLS)

    def test_info_event_set_to_id(self):
        result = cs.delly_preprocess_evts_only(make_delly_evt_df())
        assert (result['INFO_EVENT'] == result['ID']).all()

    def test_chr2_nan_fill(self):
        df = make_delly_evt_df()
        assert pd.isna(df.loc[0, 'INFO_CHR2'])
        result = cs.delly_preprocess_evts_only(df)
        assert result['INFO_CHR2'].iloc[0] == 'chr1'

    def test_info_end_nan_raises(self):
        df = make_delly_evt_df()
        df.loc[0, 'INFO_END'] = None
        with pytest.raises(ValueError, match='INFO_END is missing'):
            cs.delly_preprocess_evts_only(df)

    def test_schema_event_to_bnd_contract(self):
        result = cs.delly_preprocess_evts_only(make_delly_evt_df())
        assert_has_columns(result, EVENT_TO_BND_REQUIRED_COLS)
        assert_column_dtypes(result, EVENT_TO_BND_REQUIRED_DTYPES)


class TestDellySplitBnds:
    # Mate ALT per CT for a chr1:pos record whose partner is the mate.
    MATE_ALT = {
        '3to5': ']chr1:{pos}]N',
        '5to3': 'N[chr1:{pos}[',
        '3to3': 'N]chr1:{pos}]',
        '5to5': '[chr1:{pos}[N',
    }
    MATE_CT = {'3to5': '5to3', '5to3': '3to5', '3to3': '3to3', '5to5': '5to5'}

    def _split(self, df):
        out = cs.delly_split_bnds(df)
        own = out[out['ID'].str.endswith('_2nd')].reset_index(drop=True)
        mate = out[out['ID'].str.endswith('_1st')].reset_index(drop=True)
        return out, own, mate

    def test_two_rows_per_record(self):
        df = make_delly_bnd_df()
        out, own, mate = self._split(df)
        assert len(out) == 2 * len(df)
        assert list(own['ID']) == [f'{i}_2nd' for i in df['ID']]
        assert list(mate['ID']) == [f'{i}_1st' for i in df['ID']]

    def test_original_alt_matches_delly(self):
        df = make_delly_bnd_df()
        _, own, _ = self._split(df)
        assert list(own['ALT']) == list(df['ALT'])
        assert list(own['CHROM']) == list(df['CHROM'])
        assert list(own['POS']) == list(df['POS'])
        assert list(own['REF']) == list(df['REF'])

    @pytest.mark.parametrize('i', range(4))
    def test_mate_fields(self, i):
        df = make_delly_bnd_df()
        _, _, mate = self._split(df)
        src, m = df.iloc[i], mate.iloc[i]
        ct = src['INFO_CT']
        assert m['CHROM'] == 'chr2' and m['POS'] == src['INFO_POS2']
        assert m['REF'] == 'N'
        assert m['ALT'] == self.MATE_ALT[ct].format(pos=src['POS'])
        assert m['INFO_CHR2'] == 'chr1' and m['INFO_POS2'] == src['POS']
        assert m['INFO_END'] == src['INFO_POS2'] + 1
        assert m['INFO_CT'] == self.MATE_CT[ct]
        assert m['INFO_CIPOS'] == src['INFO_CIEND'] and m['INFO_CIEND'] == src['INFO_CIPOS']
        assert m['QUAL'] == src['QUAL']

    def test_alts_point_at_each_other(self):
        _, own, mate = self._split(make_delly_bnd_df())
        own_alt = bnds.process_breakend_alt(own['ALT'])
        mate_alt = bnds.process_breakend_alt(mate['ALT'])
        assert list(own_alt['mate_chr']) == list(mate['CHROM'])
        assert list(own_alt['mate_pos'].astype(int)) == list(mate['POS'])
        assert list(mate_alt['mate_chr']) == list(own['CHROM'])
        assert list(mate_alt['mate_pos'].astype(int)) == list(own['POS'])
        assert own_alt['relative_pos'].notna().all() and mate_alt['relative_pos'].notna().all()

    def test_tra_uses_end_for_partner(self):
        df = make_delly_bnd_df(svtype='TRA')
        _, own, mate = self._split(df)
        assert list(mate['POS']) == list(df['INFO_END'])
        assert list(own['INFO_POS2']) == list(df['INFO_END'])
        assert list(own['ALT']) == [
            delly_bnd_alt(ct, 'A', 'chr2', p) for ct, p in zip(df['INFO_CT'], df['INFO_END'])
        ]

    @pytest.mark.parametrize('bad', ['NtoN', None])
    def test_unknown_ct_raises(self, bad):
        df = make_delly_bnd_df()
        df.loc[1, 'INFO_CT'] = bad
        with pytest.raises(ValueError, match='INFO_CT'):
            cs.delly_split_bnds(df)


class TestDellyPreprocessBndsOnly:
    def test_info_event_is_original_id(self):
        df = make_delly_bnd_df()
        result = cs.delly_preprocess_bnds_only(df)
        assert len(result) == 2 * len(df)
        assert sorted(result['INFO_EVENT']) == sorted(list(df['ID']) * 2)

    def test_cipos_split(self):
        result = cs.delly_preprocess_bnds_only(make_delly_bnd_df())
        assert_has_columns(result, ['INFO_CIPOS_l', 'INFO_CIPOS_r'])
        assert_missing_columns(result, ['INFO_CIPOS'])

    def test_info_end_from_pos2(self):
        result = cs.delly_preprocess_bnds_only(make_delly_bnd_df())
        assert (result['INFO_END'] == result['INFO_POS2'].astype(int)).all()

    def test_schema_bnd_to_bp_contract(self):
        result = cs.delly_preprocess_bnds_only(make_delly_bnd_df())
        assert_has_columns(result, BND_TO_BP_REQUIRED_COLS + ['INFO_CIPOS_l', 'INFO_CIPOS_r'])


# =====================================================================
# SNIFFLES
# =====================================================================

class TestSnifflesPreprocessEvtsOnly:
    def test_stdev_to_cipos(self):
        result = cs.sniffles_preprocess_evts_only(make_sniffles_evt_df())
        assert_has_columns(result, ['INFO_CIPOS_l', 'INFO_CIPOS_r'])
        assert result['INFO_CIPOS_l'].iloc[0] < 0
        assert result['INFO_CIPOS_r'].iloc[0] > 0

    def test_stdev_to_ciend(self):
        result = cs.sniffles_preprocess_evts_only(make_sniffles_evt_df())
        assert_has_columns(result, ['INFO_CIEND_l', 'INFO_CIEND_r'])

    def test_nan_stdev_gives_zero_ci(self):
        df = make_sniffles_evt_df()
        df.loc[0, 'INFO_STDEV_POS'] = None
        df.loc[0, 'INFO_STDEV_LEN'] = None
        result = cs.sniffles_preprocess_evts_only(df)
        assert result['INFO_CIPOS_l'].iloc[0] == 0
        assert result['INFO_CIEND_l'].iloc[0] == 0

    def test_chr2_set_to_chrom(self):
        result = cs.sniffles_preprocess_evts_only(make_sniffles_evt_df())
        assert (result['INFO_CHR2'] == result['CHROM']).all()

    def test_del_svlen_positive(self):
        result = cs.sniffles_preprocess_evts_only(make_sniffles_evt_df())
        del_rows = result[result['INFO_SVTYPE'] == 'DEL']
        assert (del_rows['INFO_SVLEN'] >= 0).all()

    def test_schema_event_to_bnd_contract(self):
        result = cs.sniffles_preprocess_evts_only(make_sniffles_evt_df())
        assert_has_columns(result, EVENT_TO_BND_REQUIRED_COLS)
        assert_column_dtypes(result, EVENT_TO_BND_REQUIRED_DTYPES)


class TestSnifflesPreprocessBndsOnly:
    def test_cipos_present(self):
        result = cs.sniffles_preprocess_bnds_only(make_sniffles_bnd_df())
        assert_has_columns(result, ['INFO_CIPOS_l', 'INFO_CIPOS_r', 'INFO_EVENT'])

    def test_schema_bnd_to_bp_contract(self):
        result = cs.sniffles_preprocess_bnds_only(make_sniffles_bnd_df())
        assert_has_columns(result, BND_TO_BP_REQUIRED_COLS + ['INFO_CIPOS_l', 'INFO_CIPOS_r'])


# =====================================================================
# SURVIVOR
# =====================================================================

class TestSurvivorPreprocessEvts:
    def test_cipos_ciend_split_and_drop(self):
        result = cs.survivor_preprocess_evts(make_survivor_evt_df())
        assert_has_columns(result, ['INFO_CIPOS_l', 'INFO_CIPOS_r', 'INFO_CIEND_l', 'INFO_CIEND_r'])
        assert_missing_columns(result, CIPOS_RAW_COLS)

    def test_info_event_set_to_id(self):
        result = cs.survivor_preprocess_evts(make_survivor_evt_df())
        assert (result['INFO_EVENT'] == result['ID']).all()

    def test_schema_event_to_bnd_contract(self):
        result = cs.survivor_preprocess_evts(make_survivor_evt_df())
        assert_has_columns(result, EVENT_TO_BND_REQUIRED_COLS)
        assert_column_dtypes(result, EVENT_TO_BND_REQUIRED_DTYPES)


# =====================================================================
# SVABA
# =====================================================================

class TestSvabaPreprocess:
    def test_info_event_derived_from_id(self):
        df = make_svaba_bnd_df()
        result = cs.svaba_preprocess(df)
        assert result['INFO_EVENT'].notna().all()
        assert result.iloc[0]['INFO_EVENT'] == result.iloc[1]['INFO_EVENT']
        assert result.iloc[2]['INFO_EVENT'] == result.iloc[3]['INFO_EVENT']
        assert result.iloc[0]['INFO_EVENT'] != result.iloc[2]['INFO_EVENT']

    def test_info_event_strips_suffix(self):
        df = make_svaba_bnd_df(n=2)
        result = cs.svaba_preprocess(df)
        assert ':' not in result['INFO_EVENT'].iloc[0]

    def test_cipos_precise(self):
        df = make_svaba_bnd_df()
        result = cs.svaba_preprocess(df)
        precise = result[~result['INFO_IMPRECISE'].astype(bool)]
        assert (precise['INFO_CIPOS_l'] == 0).all()
        assert (precise['INFO_CIPOS_r'] == 0).all()

    def test_cipos_imprecise(self):
        df = make_svaba_bnd_df()
        result = cs.svaba_preprocess(df)
        imprecise = result[result['INFO_IMPRECISE'].astype(bool)]
        assert (imprecise['INFO_CIPOS_l'] == -50).all()
        assert (imprecise['INFO_CIPOS_r'] == 50).all()

    def test_homlen_from_homseq(self):
        df = make_svaba_bnd_df()
        result = cs.svaba_preprocess(df)
        assert result['INFO_HOMLEN'].iloc[0] == 4
        assert result['INFO_HOMLEN'].iloc[2] == 0

    def test_homseq_dropped(self):
        result = cs.svaba_preprocess(make_svaba_bnd_df())
        assert_missing_columns(result, ['INFO_HOMSEQ'])

    def test_evdnc_one_hot(self):
        result = cs.svaba_preprocess(make_svaba_bnd_df())
        evdnc_cols = [f'CUSTOM_EVDNC_{v}' for v in cs._SVABA_EVDNC_VALUES]
        assert_has_columns(result, evdnc_cols)
        assert_missing_columns(result, ['INFO_EVDNC'])
        assert result['CUSTOM_EVDNC_ASSMB'].iloc[0] == 1
        assert result['CUSTOM_EVDNC_DSCRD'].iloc[0] == 0
        assert result['CUSTOM_EVDNC_DSCRD'].iloc[2] == 1

    def test_pl_tuple_split(self):
        result = cs.svaba_preprocess(make_svaba_bnd_df())
        assert_has_columns(result, ['SAMPLE_PL_0', 'SAMPLE_PL_1', 'SAMPLE_PL_2'])
        assert_missing_columns(result, ['SAMPLE_PL'])
        assert result['SAMPLE_PL_0'].iloc[0] == 60.0
        assert result['SAMPLE_PL_1'].iloc[0] == 0.0

    def test_schema_bnd_to_bp_contract(self):
        result = cs.svaba_preprocess(make_svaba_bnd_df())
        assert_has_columns(result, BND_TO_BP_REQUIRED_COLS + ['INFO_CIPOS_l', 'INFO_CIPOS_r'])

    def test_input_not_mutated(self):
        df = make_svaba_bnd_df()
        original_cols = set(df.columns)
        cs.svaba_preprocess(df)
        assert set(df.columns) == original_cols
        assert 'INFO_HOMSEQ' in df.columns


class TestSvabaPostBps:
    def _make_bp_df(self, qual1=100.0, qual2=80.0, fp1=True, fp2=False):
        return pd.DataFrame({
            'CHROM_bnd1': ['chr1'], 'POS_bnd1': [1000],
            'CHROM_bnd2': ['chr1'], 'POS_bnd2': [2000],
            'QUAL_bnd1': [qual1], 'QUAL_bnd2': [qual2],
            'FILTER_PASS_bnd1': [fp1], 'FILTER_PASS_bnd2': [fp2],
            'INFO_EVENT': ['ev1'],
        })

    def test_qual_unified_to_min(self):
        result = cs.svaba_post_bps(self._make_bp_df(100.0, 80.0))
        assert result['QUAL'].iloc[0] == 80.0

    def test_filter_pass_and_logic(self):
        result = cs.svaba_post_bps(self._make_bp_df(fp1=True, fp2=False))
        assert result['FILTER_PASS'].iloc[0] == False  # noqa: E712

    def test_filter_pass_both_true(self):
        result = cs.svaba_post_bps(self._make_bp_df(fp1=True, fp2=True))
        assert result['FILTER_PASS'].iloc[0] == True  # noqa: E712

    def test_schema_unified_columns(self):
        result = cs.svaba_post_bps(self._make_bp_df())
        assert_has_columns(result, ['QUAL', 'FILTER_PASS'])
        assert_missing_columns(result, ['FILTER_PASS_bnd1', 'FILTER_PASS_bnd2'])

    def test_idempotency_already_unified(self):
        df = pd.DataFrame({
            'CHROM_bnd1': ['chr1'], 'POS_bnd1': [1000],
            'QUAL': [50.0], 'FILTER_PASS': [True],
            'INFO_EVENT': ['ev1'],
        })
        result = cs.svaba_post_bps(df)
        assert result['QUAL'].iloc[0] == 50.0
        assert result['FILTER_PASS'].iloc[0] == True  # noqa: E712


# =====================================================================
# Cross-cutting parametrized tests
# =====================================================================

_ALL_PREPROCESS_FUNCTIONS = [
    ('gridss_preprocess', cs.gridss_preprocess, make_gridss_bnd_df),
    ('manta_preprocess_evts_only', cs.manta_preprocess_evts_only, make_manta_evt_df),
    ('manta_preprocess_bnds_only', cs.manta_preprocess_bnds_only, make_manta_bnd_df),
    ('delly_preprocess_evts_only', cs.delly_preprocess_evts_only, make_delly_evt_df),
    ('delly_preprocess_bnds_only', cs.delly_preprocess_bnds_only, make_delly_bnd_df),
    ('sniffles_preprocess_evts_only', cs.sniffles_preprocess_evts_only, make_sniffles_evt_df),
    ('sniffles_preprocess_bnds_only', cs.sniffles_preprocess_bnds_only, make_sniffles_bnd_df),
    ('survivor_preprocess_evts', cs.survivor_preprocess_evts, make_survivor_evt_df),
    ('svaba_preprocess', cs.svaba_preprocess, make_svaba_bnd_df),
]

_EVTS_ONLY_FUNCTIONS = [
    ('manta_preprocess_evts_only', cs.manta_preprocess_evts_only, make_manta_evt_df),
    ('delly_preprocess_evts_only', cs.delly_preprocess_evts_only, make_delly_evt_df),
    ('sniffles_preprocess_evts_only', cs.sniffles_preprocess_evts_only, make_sniffles_evt_df),
    ('survivor_preprocess_evts', cs.survivor_preprocess_evts, make_survivor_evt_df),
]

_BNDS_ONLY_FUNCTIONS = [
    ('manta_preprocess_bnds_only', cs.manta_preprocess_bnds_only, make_manta_bnd_df),
    ('delly_preprocess_bnds_only', cs.delly_preprocess_bnds_only, make_delly_bnd_df),
    ('sniffles_preprocess_bnds_only', cs.sniffles_preprocess_bnds_only, make_sniffles_bnd_df),
]


class TestCopySemantics:
    @pytest.mark.parametrize('name,func,builder', _ALL_PREPROCESS_FUNCTIONS, ids=[t[0] for t in _ALL_PREPROCESS_FUNCTIONS])
    def test_input_not_mutated(self, name, func, builder):
        df = builder()
        original = df.copy()
        func(df)
        pd.testing.assert_frame_equal(df, original)


class TestEmptyDataFrame:
    @pytest.mark.parametrize('name,func,builder', _ALL_PREPROCESS_FUNCTIONS, ids=[t[0] for t in _ALL_PREPROCESS_FUNCTIONS])
    def test_empty_no_crash(self, name, func, builder):
        df = builder()[:0]
        result = func(df)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0


class TestEventToBndContract:
    """Verify that all _evts_only functions produce output satisfying event_to_bnd."""

    @pytest.mark.parametrize('name,func,builder', _EVTS_ONLY_FUNCTIONS, ids=[t[0] for t in _EVTS_ONLY_FUNCTIONS])
    def test_has_required_columns(self, name, func, builder):
        result = func(builder())
        assert_has_columns(result, EVENT_TO_BND_REQUIRED_COLS)

    @pytest.mark.parametrize('name,func,builder', _EVTS_ONLY_FUNCTIONS, ids=[t[0] for t in _EVTS_ONLY_FUNCTIONS])
    def test_required_dtypes(self, name, func, builder):
        result = func(builder())
        ci_dtypes = {k: v for k, v in EVENT_TO_BND_REQUIRED_DTYPES.items()
                     if k != 'INFO_SVLEN'}
        assert_column_dtypes(result, ci_dtypes)
        assert pd.to_numeric(result['INFO_SVLEN'], errors='coerce').notna().all(), \
            f"{name}: INFO_SVLEN values must be numeric"

    @pytest.mark.parametrize('name,func,builder', _EVTS_ONLY_FUNCTIONS, ids=[t[0] for t in _EVTS_ONLY_FUNCTIONS])
    def test_raw_cipos_dropped(self, name, func, builder):
        result = func(builder())
        assert_missing_columns(result, CIPOS_RAW_COLS)


class TestBndToBpContract:
    """Verify that all _bnds_only functions produce output satisfying bnd_to_bp."""

    @pytest.mark.parametrize('name,func,builder', _BNDS_ONLY_FUNCTIONS, ids=[t[0] for t in _BNDS_ONLY_FUNCTIONS])
    def test_has_required_columns(self, name, func, builder):
        result = func(builder())
        assert_has_columns(result, BND_TO_BP_REQUIRED_COLS + ['INFO_CIPOS_l', 'INFO_CIPOS_r'])

    @pytest.mark.parametrize('name,func,builder', _BNDS_ONLY_FUNCTIONS, ids=[t[0] for t in _BNDS_ONLY_FUNCTIONS])
    def test_raw_cipos_dropped(self, name, func, builder):
        result = func(builder())
        assert_missing_columns(result, ['INFO_CIPOS'])


# =====================================================================
# Recipe dispatcher
# =====================================================================

class TestParseRecipes:
    def test_keys_are_sources(self):
        for key in cs.parse_recipes:
            assert _caller_params.CALLSET_PARAM_DICT[key]['source'] == key

    def test_svaba_h_uses_svaba_recipe(self):
        assert _caller_params.CALLSET_PARAM_DICT['svaba_H']['source'] in cs.parse_recipes

    def test_unknown_source_raises(self):
        params = dict(_caller_params.DELLY_PARAMS, source='no_such_caller')
        with pytest.raises(KeyError, match='no_such_caller'):
            cs.vcf_df_to_raw_bps(pd.DataFrame(), params)


class TestVcfDfToRawBpsDelly:
    @pytest.fixture
    def parsed(self, tmp_path):
        path = tmp_path / 'delly.vcf'
        bnd_records = write_delly_vcf(path)
        return sv_io.read_vcf(path, 'delly'), bnd_records

    def test_one_breakpoint_per_record(self, parsed):
        vcf_df, bnd_records = parsed
        bp_df = cs.vcf_df_to_raw_bps(vcf_df, _caller_params.DELLY_PARAMS)
        assert len(bp_df) == 1 + len(bnd_records)
        bnd_bps = bp_df[bp_df['INFO_EVENT'].str.startswith('BND')].set_index('INFO_EVENT')
        for i, (ct, pos, pos2) in enumerate(bnd_records):
            row = bnd_bps.loc[f'BND0000{i}']
            assert (row['CHROM_bnd1'], row['POS_bnd1']) == ('chr1', pos)
            assert (row['CHROM_bnd2'], row['POS_bnd2']) == ('chr2', pos2)
            assert row['INFO_CT_bnd1'] == ct

    def test_input_not_mutated(self, parsed):
        vcf_df, _ = parsed
        original = vcf_df.copy()
        cs.vcf_df_to_raw_bps(vcf_df, _caller_params.DELLY_PARAMS)
        pd.testing.assert_frame_equal(vcf_df, original)
