"""Tests for parse_preprocess_vcf and parse_preprocess_raw in svensemble/pipeline.py.

Mocks VariantFile and parse_vcf_to_df to avoid real VCF I/O.
End-to-end schema tests run real caller_specific functions but with synthetic DataFrames.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from svensemble import caller_specific as cs
from svensemble import _caller_params
from svensemble.pipeline import parse_preprocess_vcf, parse_preprocess_raw

from .conftest import (
    assert_has_columns,
    assert_missing_columns,
    make_gridss_bnd_df,
    make_manta_evt_df,
    make_manta_bnd_df,
    make_sniffles_evt_df,
    make_sniffles_bnd_df,
    make_survivor_evt_df,
    make_severus_evt_df,
    make_severus_bnd_df,
    make_cutesv_evt_df,
    make_cutesv_bnd_df,
    BREAKPOINT_CORE_COLS,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_vcf_to_df(df):
    """Return a (df, meta) tuple that parse_vcf_to_df would produce."""
    return df, {'samples': ['SAMPLE']}


def _make_callset(path='/fake/path.vcf', params=None, source='test'):
    """Build a minimal callset dict."""
    if params is None:
        params = _caller_params.default_params.copy()
    return {'path': path, 'params': params, 'source': source}


# =====================================================================
# TestParsePreprocessVcf -- dispatch logic
# =====================================================================

class TestParsePreprocessVcfDispatch:
    """Test the dispatch logic of parse_preprocess_vcf using a fake recipe."""

    @patch('svensemble.pipeline.vcfp.parse_vcf_to_df')
    @patch('svensemble.pipeline.VariantFile')
    def test_bnd_svtypes_none_no_event_to_bnd(self, mock_vf, mock_parse):
        """bnd_svtypes=None (gridss): entire DF goes to bnd_to_bp, no event_to_bnd."""
        df = cs.gridss_preprocess(make_gridss_bnd_df())
        mock_parse.return_value = _mock_vcf_to_df(make_gridss_bnd_df())
        callset = _make_callset(params=_caller_params.gridss_params)

        with patch('svensemble.pipeline.evts.event_to_bnd') as mock_e2b:
            result = parse_preprocess_vcf('gridss', callset)
            mock_e2b.assert_not_called()

        assert len(result) > 0

    @patch('svensemble.pipeline.vcfp.parse_vcf_to_df')
    @patch('svensemble.pipeline.VariantFile')
    def test_bnd_svtypes_empty_all_events(self, mock_vf, mock_parse):
        """bnd_svtypes=[] (survivor_short): entire DF goes through event_to_bnd."""
        mock_parse.return_value = _mock_vcf_to_df(make_survivor_evt_df())
        callset = _make_callset(params=_caller_params.survivor_params)

        with patch('svensemble.pipeline.evts.event_to_bnd') as mock_e2b:
            mock_e2b.return_value = make_gridss_bnd_df()[:2]
            with patch('svensemble.pipeline.bnds.bnd_to_bp') as mock_b2bp:
                mock_b2bp.return_value = pd.DataFrame({'CHROM_bnd1': ['chr1'], 'INFO_EVENT': ['x']})
                parse_preprocess_vcf('survivor_short', callset)
                mock_e2b.assert_called_once()

    @patch('svensemble.pipeline.vcfp.parse_vcf_to_df')
    @patch('svensemble.pipeline.VariantFile')
    def test_bnd_svtypes_list_splits_correctly(self, mock_vf, mock_parse):
        """bnd_svtypes=['BND', 'INV'] (manta): splits by INFO_SVTYPE."""
        evts = make_manta_evt_df(n=2)
        bnds = make_manta_bnd_df(n=4)
        combined = pd.concat([evts, bnds], ignore_index=True)
        mock_parse.return_value = _mock_vcf_to_df(combined)

        callset = _make_callset(params=_caller_params.manta_candidate_params)
        result = parse_preprocess_vcf('manta_candidate', callset)
        assert len(result) > 0

    @patch('svensemble.pipeline.vcfp.parse_vcf_to_df')
    @patch('svensemble.pipeline.VariantFile')
    def test_preprocess_all_called(self, mock_vf, mock_parse):
        """preprocess_all function is called before the split."""
        mock_parse.return_value = _mock_vcf_to_df(make_gridss_bnd_df())
        callset = _make_callset(params=_caller_params.gridss_params)

        with patch.dict(cs.parse_recipes['gridss'], {'preprocess_all': MagicMock(return_value=cs.gridss_preprocess(make_gridss_bnd_df()))}):
            parse_preprocess_vcf('gridss', callset)
            cs.parse_recipes['gridss']['preprocess_all'].assert_called_once()

    @patch('svensemble.pipeline.vcfp.parse_vcf_to_df')
    @patch('svensemble.pipeline.VariantFile')
    def test_post_bps_called(self, mock_vf, mock_parse):
        """post_bps function is called after bnd_to_bp."""
        mock_parse.return_value = _mock_vcf_to_df(make_gridss_bnd_df())
        callset = _make_callset(params=_caller_params.gridss_params)

        with patch.object(cs, 'gridss_post_bps', wraps=cs.gridss_post_bps) as mock_post:
            with patch.dict(cs.parse_recipes['gridss'], {'post_bps': mock_post}):
                parse_preprocess_vcf('gridss', callset)
                mock_post.assert_called_once()

    def test_unknown_callset_raises(self):
        callset = _make_callset()
        with pytest.raises(KeyError):
            parse_preprocess_vcf('nonexistent_caller', callset)


# =====================================================================
# End-to-end schema tests -- real functions, synthetic VCF DataFrames
# =====================================================================

class TestParsePreprocessVcfSchemaGridss:
    @patch('svensemble.pipeline.vcfp.parse_vcf_to_df')
    @patch('svensemble.pipeline.VariantFile')
    def test_gridss_output_schema(self, mock_vf, mock_parse):
        mock_parse.return_value = _mock_vcf_to_df(make_gridss_bnd_df())
        callset = _make_callset(params=_caller_params.gridss_params)

        result = parse_preprocess_vcf('gridss', callset)
        assert_has_columns(result, BREAKPOINT_CORE_COLS)
        assert_has_columns(result, ['QUAL', 'FILTER_PASS'])
        assert_missing_columns(result, ['FILTER_PASS_bnd1', 'FILTER_PASS_bnd2'])
        assert_missing_columns(result, ['INFO_CIPOS'])


class TestParsePreprocessVcfSchemaManta:
    @patch('svensemble.pipeline.vcfp.parse_vcf_to_df')
    @patch('svensemble.pipeline.VariantFile')
    def test_manta_candidate_output_schema(self, mock_vf, mock_parse):
        evts = make_manta_evt_df(n=2)
        bnds = make_manta_bnd_df(n=4)
        combined = pd.concat([evts, bnds], ignore_index=True)
        mock_parse.return_value = _mock_vcf_to_df(combined)
        callset = _make_callset(params=_caller_params.manta_candidate_params)

        result = parse_preprocess_vcf('manta_candidate', callset)
        assert_has_columns(result, BREAKPOINT_CORE_COLS)
        assert_has_columns(result, ['INFO_CIPOS_l_bnd1', 'INFO_CIPOS_r_bnd1'])
        assert_missing_columns(result, ['INFO_CIPOS', 'INFO_CIEND'])


class TestParsePreprocessVcfSchemaSniffles:
    @patch('svensemble.pipeline.vcfp.parse_vcf_to_df')
    @patch('svensemble.pipeline.VariantFile')
    def test_sniffles_output_schema(self, mock_vf, mock_parse):
        evts = make_sniffles_evt_df(n=2)
        bnds = make_sniffles_bnd_df(n=4)
        combined = pd.concat([evts, bnds], ignore_index=True)
        mock_parse.return_value = _mock_vcf_to_df(combined)
        callset = _make_callset(params=_caller_params.sniffles_params)

        result = parse_preprocess_vcf('sniffles', callset)
        assert_has_columns(result, BREAKPOINT_CORE_COLS)
        assert_has_columns(result, ['INFO_CIPOS_l_bnd1', 'INFO_CIPOS_r_bnd1'])


class TestParsePreprocessVcfSchemaSurvivor:
    @patch('svensemble.pipeline.vcfp.parse_vcf_to_df')
    @patch('svensemble.pipeline.VariantFile')
    def test_survivor_output_schema(self, mock_vf, mock_parse):
        mock_parse.return_value = _mock_vcf_to_df(make_survivor_evt_df())
        callset = _make_callset(params=_caller_params.survivor_params)

        result = parse_preprocess_vcf('survivor_short', callset)
        assert_has_columns(result, BREAKPOINT_CORE_COLS)


class TestParsePreprocessVcfSchemaSeverus:
    @patch('svensemble.pipeline.vcfp.parse_vcf_to_df')
    @patch('svensemble.pipeline.VariantFile')
    def test_severus_output_schema(self, mock_vf, mock_parse):
        evts = make_severus_evt_df(n=2)
        bnds = make_severus_bnd_df(n=4)
        combined = pd.concat([evts, bnds], ignore_index=True)
        mock_parse.return_value = _mock_vcf_to_df(combined)
        callset = _make_callset(params=_caller_params.severus_params)

        result = parse_preprocess_vcf('severus', callset)
        assert_has_columns(result, BREAKPOINT_CORE_COLS)
        assert_has_columns(result, ['INFO_CIPOS_l_bnd1', 'INFO_CIPOS_r_bnd1'])

    @patch('svensemble.pipeline.vcfp.parse_vcf_to_df')
    @patch('svensemble.pipeline.VariantFile')
    def test_severus_hvaf_split(self, mock_vf, mock_parse):
        """hVAF tuple is split into 3 scalar columns; original is dropped."""
        evts = make_severus_evt_df(n=2)
        mock_parse.return_value = _mock_vcf_to_df(evts)
        callset = _make_callset(params=_caller_params.severus_params)

        result = parse_preprocess_vcf('severus', callset)
        hvaf_cols = [c for c in result.columns if 'hVAF_H' in c]
        assert len(hvaf_cols) >= 3, f"Expected at least 3 hVAF scalar columns, got: {hvaf_cols}"
        raw_hvaf_cols = [c for c in result.columns if c in ('SAMPLE_hVAF', 'SAMPLE_hVAF_bnd1', 'SAMPLE_hVAF_bnd2')]
        assert len(raw_hvaf_cols) == 0, f"Original hVAF tuple column should be dropped, but found: {raw_hvaf_cols}"

    @patch('svensemble.pipeline.vcfp.parse_vcf_to_df')
    @patch('svensemble.pipeline.VariantFile')
    def test_severus_cipos_from_precision(self, mock_vf, mock_parse):
        """PRECISE variants get CI=0; IMPRECISE variants get CI=50."""
        evts = make_severus_evt_df(n=2)
        mock_parse.return_value = _mock_vcf_to_df(evts)
        callset = _make_callset(params=_caller_params.severus_params)

        result = parse_preprocess_vcf('severus', callset)
        cipos_l = result['INFO_CIPOS_l_bnd1'].values
        cipos_r = result['INFO_CIPOS_r_bnd1'].values
        for l_val, r_val in zip(cipos_l, cipos_r):
            assert l_val <= 0, f"CIPOS_l should be <= 0, got {l_val}"
            assert r_val >= 0, f"CIPOS_r should be >= 0, got {r_val}"


class TestParsePreprocessVcfSchemaCutesv:
    @patch('svensemble.pipeline.vcfp.parse_vcf_to_df')
    @patch('svensemble.pipeline.VariantFile')
    def test_cutesv_output_schema(self, mock_vf, mock_parse):
        evts = make_cutesv_evt_df(n=2)
        bnds = make_cutesv_bnd_df(n=4)
        combined = pd.concat([evts, bnds], ignore_index=True)
        mock_parse.return_value = _mock_vcf_to_df(combined)
        callset = _make_callset(params=_caller_params.cutesv_params)

        result = parse_preprocess_vcf('cutesv', callset)
        assert_has_columns(result, BREAKPOINT_CORE_COLS)
        assert_has_columns(result, ['INFO_CIPOS_l_bnd1', 'INFO_CIPOS_r_bnd1'])

    @patch('svensemble.pipeline.vcfp.parse_vcf_to_df')
    @patch('svensemble.pipeline.VariantFile')
    def test_cutesv_raw_tuple_cols_dropped(self, mock_vf, mock_parse):
        """Raw INFO_CIPOS and INFO_CILEN tuple columns are dropped."""
        evts = make_cutesv_evt_df(n=2)
        mock_parse.return_value = _mock_vcf_to_df(evts)
        callset = _make_callset(params=_caller_params.cutesv_params)

        result = parse_preprocess_vcf('cutesv', callset)
        raw_cols = [c for c in result.columns if c in (
            'INFO_CIPOS', 'INFO_CILEN',
            'INFO_CIPOS_bnd1', 'INFO_CILEN_bnd1',
            'INFO_CIPOS_bnd2', 'INFO_CILEN_bnd2',
        )]
        assert len(raw_cols) == 0, f"Raw tuple columns should be dropped: {raw_cols}"

    @patch('svensemble.pipeline.vcfp.parse_vcf_to_df')
    @patch('svensemble.pipeline.VariantFile')
    def test_cutesv_del_svlen_positive(self, mock_vf, mock_parse):
        """DEL SVLEN is converted from negative to positive."""
        evts = make_cutesv_evt_df(n=2)
        mock_parse.return_value = _mock_vcf_to_df(evts)
        callset = _make_callset(params=_caller_params.cutesv_params)

        result = parse_preprocess_vcf('cutesv', callset)
        svlen_cols = [c for c in result.columns if 'SVLEN' in c]
        for col in svlen_cols:
            vals = result[col].dropna()
            assert (vals >= 0).all(), f"SVLEN column {col} has negative values"

    @patch('svensemble.pipeline.vcfp.parse_vcf_to_df')
    @patch('svensemble.pipeline.VariantFile')
    def test_cutesv_missing_ci_sentinel_clamped(self, mock_vf, mock_parse):
        """DUP/INV rows with pysam INT64_MIN sentinels get CI=0, not overflow."""
        evts = make_cutesv_evt_df(n=4)  # DEL, INS, DUP, INV
        mock_parse.return_value = _mock_vcf_to_df(evts)
        callset = _make_callset(params=_caller_params.cutesv_params)

        result = parse_preprocess_vcf('cutesv', callset)
        ci_cols = [c for c in result.columns if 'CIPOS' in c or 'CIEND' in c]
        for col in ci_cols:
            vals = result[col].dropna()
            assert (vals.abs() < 1e6).all(), (
                f"Column {col} has sentinel-sized values: {vals[vals.abs() >= 1e6].tolist()}"
            )

    @patch('svensemble.pipeline.vcfp.parse_vcf_to_df')
    @patch('svensemble.pipeline.VariantFile')
    def test_cutesv_bnd_cipos_from_precision(self, mock_vf, mock_parse):
        """BND CIPOS is inferred from PRECISE/IMPRECISE flags."""
        bnds = make_cutesv_bnd_df(n=4)
        mock_parse.return_value = _mock_vcf_to_df(bnds)
        callset = _make_callset(params=_caller_params.cutesv_params)

        result = parse_preprocess_vcf('cutesv', callset)
        cipos_l = result['INFO_CIPOS_l_bnd1'].values
        cipos_r = result['INFO_CIPOS_r_bnd1'].values
        for l_val, r_val in zip(cipos_l, cipos_r):
            assert l_val <= 0, f"CIPOS_l should be <= 0, got {l_val}"
            assert r_val >= 0, f"CIPOS_r should be >= 0, got {r_val}"

    @patch('svensemble.pipeline.vcfp.parse_vcf_to_df')
    @patch('svensemble.pipeline.VariantFile')
    def test_cutesv_af_unpacked_to_scalar(self, mock_vf, mock_parse):
        """INFO_AF (Number=A tuple from pysam) is unpacked to a scalar float."""
        evts = make_cutesv_evt_df(n=2)
        mock_parse.return_value = _mock_vcf_to_df(evts)
        callset = _make_callset(params=_caller_params.cutesv_params)

        result = parse_preprocess_vcf('cutesv', callset)
        af_cols = [c for c in result.columns if 'AF' in c]
        for col in af_cols:
            assert result[col].dtype == float, (
                f"INFO_AF column {col} should be float, got {result[col].dtype}"
            )
            assert not result[col].astype(str).str.contains(r'[(),]').any(), (
                f"INFO_AF column {col} still contains tuple characters"
            )


# =====================================================================
# TestParsePreprocessRaw
# =====================================================================

class TestParsePreprocessRaw:
    def _make_workflow_config(self, tmp_path, callsets_dict):
        """Build a minimal workflow_config for parse_preprocess_raw."""
        bps_dir = tmp_path / 'callset_bps'
        bps_dir.mkdir(exist_ok=True)
        wf_dir = tmp_path / 'workflow'
        wf_dir.mkdir(exist_ok=True)
        return {
            'callset_bps_dir': bps_dir,
            'workflow_dir': wf_dir,
            'callsets': callsets_dict,
            'ref_path': tmp_path / 'ref.fa',
        }

    @patch('svensemble.pipeline.parse_preprocess_vcf')
    def test_recipe_caller_dispatched(self, mock_ppv, tmp_path):
        """Callers in parse_recipes are dispatched to parse_preprocess_vcf."""
        vcf_path = tmp_path / 'gridss.vcf'
        vcf_path.touch()
        mock_ppv.return_value = pd.DataFrame({'CHROM_bnd1': ['chr1'], 'INFO_EVENT': ['ev1']})
        callsets = {'gridss': _make_callset(path=str(vcf_path), params=_caller_params.gridss_params)}
        config = self._make_workflow_config(tmp_path, callsets)

        parse_preprocess_raw('sample1', config)
        mock_ppv.assert_called_once()
        assert (config['callset_bps_dir'] / 'gridss.raw_bps.csv').exists()

    @patch('svensemble.pipeline.parse_preprocess_gnomad')
    def test_custom_parser_dispatched(self, mock_gnomad, tmp_path):
        """gnomad uses the custom parser."""
        vcf_path = tmp_path / 'gnomad.vcf'
        vcf_path.touch()
        mock_gnomad.return_value = pd.DataFrame({'CHROM_bnd1': ['chr1'], 'INFO_EVENT': ['ev1']})
        callsets = {'gnomad': _make_callset(path=str(vcf_path), params=_caller_params.gnomad_params, source='gnomad')}
        config = self._make_workflow_config(tmp_path, callsets)

        parse_preprocess_raw('sample1', config)
        mock_gnomad.assert_called_once()

    def test_missing_vcf_raises(self, tmp_path):
        callsets = {'gridss': _make_callset(path='/nonexistent/path.vcf', params=_caller_params.gridss_params)}
        config = self._make_workflow_config(tmp_path, callsets)

        with pytest.raises(FileNotFoundError, match='VCF not found'):
            parse_preprocess_raw('sample1', config)

    @patch('svensemble.pipeline.parse_preprocess_vcf')
    def test_existing_output_skipped(self, mock_ppv, tmp_path):
        """Idempotency: existing raw_bps.csv is not overwritten."""
        vcf_path = tmp_path / 'gridss.vcf'
        vcf_path.touch()
        callsets = {'gridss': _make_callset(path=str(vcf_path), params=_caller_params.gridss_params)}
        config = self._make_workflow_config(tmp_path, callsets)
        (config['callset_bps_dir'] / 'gridss.raw_bps.csv').write_text('pre-existing')

        parse_preprocess_raw('sample1', config)
        mock_ppv.assert_not_called()

    def test_unregistered_callset_raises(self, tmp_path):
        vcf_path = tmp_path / 'unknown.vcf'
        vcf_path.touch()
        callsets = {'unknown_caller': _make_callset(path=str(vcf_path))}
        config = self._make_workflow_config(tmp_path, callsets)

        with pytest.raises(ValueError, match='no parser registered'):
            parse_preprocess_raw('sample1', config)

    @patch('svensemble.pipeline.parse_preprocess_vcf')
    def test_derived_callset_from_parent(self, mock_ppv, tmp_path):
        """Derived callsets (gridss_H) filter from parent raw_bps."""
        vcf_path = tmp_path / 'gridss.vcf'
        vcf_path.touch()
        parent_df = pd.DataFrame({
            'CHROM_bnd1': ['chr1', 'chr2'],
            'FILTER_PASS': [True, False],
            'INFO_EVENT': ['ev1', 'ev2'],
        })
        mock_ppv.return_value = parent_df

        callsets = {
            'gridss': _make_callset(path=str(vcf_path), params=_caller_params.gridss_params),
            'gridss_H': _make_callset(path=str(vcf_path), params=_caller_params.gridss_H_params),
        }
        config = self._make_workflow_config(tmp_path, callsets)

        parse_preprocess_raw('sample1', config)

        derived_path = config['callset_bps_dir'] / 'gridss_H.raw_bps.csv'
        assert derived_path.exists()
        derived_df = pd.read_csv(derived_path)
        assert len(derived_df) == 1
        assert derived_df['FILTER_PASS'].iloc[0] == True  # noqa: E712

    @patch('svensemble.pipeline.parse_preprocess_vcf')
    def test_output_csv_correct_names(self, mock_ppv, tmp_path):
        """Output CSVs are named {callset}.raw_bps.csv."""
        vcf_path = tmp_path / 'survivor_short.vcf'
        vcf_path.touch()
        mock_ppv.return_value = pd.DataFrame({'CHROM_bnd1': ['chr1'], 'INFO_EVENT': ['ev1']})
        callsets = {'survivor_short': _make_callset(path=str(vcf_path), params=_caller_params.survivor_params)}
        config = self._make_workflow_config(tmp_path, callsets)

        parse_preprocess_raw('sample1', config)
        assert (config['callset_bps_dir'] / 'survivor_short.raw_bps.csv').exists()
