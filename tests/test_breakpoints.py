"""
Unit tests for svensemble.breakpoints.
"""

import numpy as np
import pandas as pd
import pytest

from svensemble import _caller_params
from svensemble.breakpoints import _get_annotator_col, common_bps_preprocess


class TestGetAnnotatorCol:

    def test_unsplit_column(self):
        bp_df = pd.DataFrame({"ANNOT_DHFFC": [0.4, np.nan]})
        result = _get_annotator_col(bp_df, "DHFFC")
        pd.testing.assert_series_equal(result, bp_df["ANNOT_DHFFC"])

    def test_split_column_prefers_bnd1(self):
        bp_df = pd.DataFrame({
            "ANNOT_DHFFC_bnd1": [0.4, np.nan, np.nan],
            "ANNOT_DHFFC_bnd2": [0.9, 1.1, np.nan],
        })
        result = _get_annotator_col(bp_df, "DHFFC")
        assert result.tolist()[:2] == [0.4, 1.1]
        assert np.isnan(result.iloc[2])

    def test_missing_column_is_all_nan(self):
        bp_df = pd.DataFrame({"POS_bnd1": [1, 2]}, index=[5, 7])
        result = _get_annotator_col(bp_df, "DHFFC")
        assert result.index.tolist() == [5, 7]
        assert result.isna().all()


class TestCommonBpsPreprocessAnnotatorCols:

    def test_custom_cols_from_annotator(self):
        params = _caller_params.DEFAULT_PARAMS | {
            "custom_cols_pred": ["DHFC", "DHBFC", "DHFFC", "GCF"],
        }
        bp_df = pd.DataFrame({
            "ANNOT_DHFC": [0.5],
            "ANNOT_DHBFC": [0.48],
            "ANNOT_DHFFC_bnd1": [0.45],
            "ANNOT_DHFFC_bnd2": [0.45],
        })
        out = common_bps_preprocess(bp_df, params)
        assert out["CUSTOM_DHFC"].iloc[0] == pytest.approx(0.5)
        assert out["CUSTOM_DHBFC"].iloc[0] == pytest.approx(0.48)
        assert out["CUSTOM_DHFFC"].iloc[0] == pytest.approx(0.45)
        assert out["CUSTOM_GCF"].isna().all()
