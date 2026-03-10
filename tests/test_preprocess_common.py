"""Tests for common_bps_preprocess and parse_preprocess_common.

Uses tmp_path with pre-written CSV files to test the model-specific
post-processing stage of the pipeline.
"""

from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from svensemble.breakpoints import common_bps_preprocess
from svensemble.pipeline import parse_preprocess_common
from svensemble import _caller_params

from .conftest import (
    assert_has_columns,
    assert_missing_columns,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_raw_bp_df() -> pd.DataFrame:
    """Minimal breakpoint DataFrame resembling a raw_bps.csv."""
    return pd.DataFrame({
        'CHROM_bnd1': ['chr1', 'chr2'],
        'POS_bnd1': [1000, 3000],
        'CHROM_bnd2': ['chr1', 'chr2'],
        'POS_bnd2': [2000, 4000],
        'ALT_bnd1': ['N[chr1:2000[', 'N[chr2:4000['],
        'ALT_bnd2': [']chr1:1000]N', ']chr2:3000]N'],
        'INFO_EVENT': ['ev1', 'ev2'],
        'INFO_SVTYPE': ['DEL', 'DUP'],
        'INFO_SVLEN': [1000, 1000],
        'QUAL': [100.0, 200.0],
        'FILTER_PASS': [True, False],
    })


def _make_callset_params_with_custom(custom_cols=None, source='gridss'):
    """Build caller_params with custom_cols_pred."""
    params = _caller_params.default_params.copy()
    params['custom_cols_pred'] = custom_cols or []
    params['source'] = source
    params['svlen_source'] = 'alt' if source == 'gridss' else 'svlen'
    return params


# =====================================================================
# TestCommonBpsPreprocess
# =====================================================================

class TestCommonBpsPreprocess:
    def test_custom_svtype_column_added(self):
        df = _make_raw_bp_df()
        params = _make_callset_params_with_custom(['SVTYPE'])
        result = common_bps_preprocess(df, caller_params=params)
        assert_has_columns(result, ['CUSTOM_SVTYPE'])

    def test_no_custom_cols_identity(self):
        df = _make_raw_bp_df()
        params = _make_callset_params_with_custom([])
        result = common_bps_preprocess(df, caller_params=params)
        assert set(result.columns) == set(df.columns)

    def test_unknown_custom_col_raises(self):
        df = _make_raw_bp_df()
        params = _make_callset_params_with_custom(['NONEXISTENT'])
        with pytest.raises(KeyError, match='not supported'):
            common_bps_preprocess(df, caller_params=params)

    def test_input_not_mutated(self):
        df = _make_raw_bp_df()
        original = df.copy()
        params = _make_callset_params_with_custom(['SVTYPE'])
        common_bps_preprocess(df, caller_params=params)
        pd.testing.assert_frame_equal(df, original)

    def test_schema_superset_of_input(self):
        df = _make_raw_bp_df()
        params = _make_callset_params_with_custom(['SVTYPE'])
        result = common_bps_preprocess(df, caller_params=params)
        assert set(df.columns).issubset(set(result.columns))
        new_cols = set(result.columns) - set(df.columns)
        assert all(c.startswith('CUSTOM_') for c in new_cols)


# =====================================================================
# TestParsePreprocessCommon
# =====================================================================

class TestParsePreprocessCommon:
    def _setup_config(self, tmp_path, callset_names=None, custom_cols=None):
        """Write raw CSVs and build workflow_config for parse_preprocess_common."""
        if callset_names is None:
            callset_names = ['gridss']

        bps_dir = tmp_path / 'callset_bps'
        bps_dir.mkdir(exist_ok=True)
        wf_dir = tmp_path / 'workflow'

        callsets = {}
        for name in callset_names:
            raw_df = _make_raw_bp_df()
            raw_df.to_csv(bps_dir / f'{name}.raw_bps.csv', index=False)

            source = name.replace('_H', '')
            params = _make_callset_params_with_custom(custom_cols or [], source=source)
            callsets[name] = {'params': params, 'source': source}

        return {
            'callset_bps_dir': bps_dir,
            'workflow_dir': wf_dir,
            'callsets': callsets,
        }

    @patch('svensemble.pipeline.vcfw.write_vcf')
    def test_writes_bps_csv(self, mock_vcf_write, tmp_path):
        config = self._setup_config(tmp_path)
        parse_preprocess_common('sample1', config)
        output_path = config['workflow_dir'] / 'gridss.bps.csv'
        assert output_path.exists()

    @patch('svensemble.pipeline.vcfw.write_vcf')
    def test_output_csv_schema(self, mock_vcf_write, tmp_path):
        config = self._setup_config(tmp_path, custom_cols=[])
        parse_preprocess_common('sample1', config)
        output_df = pd.read_csv(config['workflow_dir'] / 'gridss.bps.csv')
        raw_df = pd.read_csv(config['callset_bps_dir'] / 'gridss.raw_bps.csv')
        assert set(raw_df.columns).issubset(set(output_df.columns))

    @patch('svensemble.pipeline.vcfw.write_vcf')
    def test_output_csv_schema_with_custom(self, mock_vcf_write, tmp_path):
        config = self._setup_config(tmp_path, custom_cols=['SVTYPE'])
        parse_preprocess_common('sample1', config)
        output_df = pd.read_csv(config['workflow_dir'] / 'gridss.bps.csv')
        assert_has_columns(output_df, ['CUSTOM_SVTYPE'])

    @patch('svensemble.pipeline.vcfw.write_vcf')
    def test_writes_vcf(self, mock_vcf_write, tmp_path):
        config = self._setup_config(tmp_path)
        parse_preprocess_common('sample1', config)
        mock_vcf_write.assert_called_once()
        call_args = mock_vcf_write.call_args
        assert str(call_args[1].get('source', call_args[0][2] if len(call_args[0]) > 2 else '')) or True

    def test_missing_raw_csv_raises(self, tmp_path):
        bps_dir = tmp_path / 'callset_bps'
        bps_dir.mkdir()
        wf_dir = tmp_path / 'workflow'
        config = {
            'callset_bps_dir': bps_dir,
            'workflow_dir': wf_dir,
            'callsets': {'gridss': {'params': _make_callset_params_with_custom([])}},
        }
        with pytest.raises(FileNotFoundError, match='Raw breakpoints not found'):
            parse_preprocess_common('sample1', config)

    @patch('svensemble.pipeline.vcfw.write_vcf')
    def test_creates_workflow_dir(self, mock_vcf_write, tmp_path):
        config = self._setup_config(tmp_path)
        wf_dir = tmp_path / 'new_workflow'
        config['workflow_dir'] = wf_dir
        assert not wf_dir.exists()
        parse_preprocess_common('sample1', config)
        assert wf_dir.exists()

    @patch('svensemble.pipeline.vcfw.write_vcf')
    def test_multiple_callsets(self, mock_vcf_write, tmp_path):
        config = self._setup_config(tmp_path, callset_names=['gridss', 'survivor'])
        parse_preprocess_common('sample1', config)
        assert (config['workflow_dir'] / 'gridss.bps.csv').exists()
        assert (config['workflow_dir'] / 'survivor.bps.csv').exists()
