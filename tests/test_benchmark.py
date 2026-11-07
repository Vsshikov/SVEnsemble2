"""benchmark: per-sample selection and validation sets relabelled to that sample."""

import pandas as pd
import pytest

from svensemble import benchmark
from svensemble._constants import SAMPLE_ID_COL


def _bp(event, pos, sample=None):
    row = {
        'INFO_EVENT': event, 'CHROM_bnd1': 'chr1', 'CHROM_bnd2': 'chr1',
        'POS_bnd1': pos, 'POS_bnd2': pos + 400,
        'ALT_bnd1': f'N[chr1:{pos + 400}[', 'ALT_bnd2': f']chr1:{pos}]N',
        'INFO_CIPOS_l_bnd1': 0, 'INFO_CIPOS_r_bnd1': 0, 'INFO_CIPOS_l_bnd2': 0, 'INFO_CIPOS_r_bnd2': 0,
        'INFO_SVTYPE': 'DEL', 'INFO_SVLEN': -400, 'QUAL': 10.0, 'INFO_INFERRED_SVTYPE': 'DEL',
    }
    if sample is not None:
        row[SAMPLE_ID_COL] = sample
    return row


CALLSET = pd.DataFrame([_bp('ev1', 100, 'S1'), _bp('ev2', 5000, 'S1'), _bp('ev1', 100, 'S2'), _bp('ev3', 9000, 'S2')])
VAL = pd.DataFrame([_bp('v1', 105), _bp('v3', 9002)])


def test_select_sample_rules():
    with pytest.raises(ValueError, match='choose one'):
        benchmark._select_sample(CALLSET, None)
    with pytest.raises(ValueError, match='S9'):
        benchmark._select_sample(CALLSET, 'S9')
    s2, name = benchmark._select_sample(CALLSET, 'S2')
    assert set(s2['INFO_EVENT']) == {'ev1', 'ev3'} and name == 'S2'
    one = CALLSET[CALLSET[SAMPLE_ID_COL] == 'S1']
    df, name = benchmark._select_sample(one, None)
    assert df is one and name == 'S1'
    assert benchmark._select_sample(CALLSET.iloc[:0], None)[1] is None


def test_is_in_val_computed_within_the_sample():
    s1 = benchmark.validate_basic_callset(CALLSET, VAL, 'kosugi', 'delly', val_match_margin=10, sample='S1')
    s2 = benchmark.validate_basic_callset(CALLSET, VAL, 'kosugi', 'delly', val_match_margin=10, sample='S2')
    assert dict(zip(s1['INFO_EVENT'], s1['is_in_val'])) == {'ev1': True, 'ev2': False}
    assert dict(zip(s2['INFO_EVENT'], s2['is_in_val'])) == {'ev1': True, 'ev3': True}


def test_validation_set_sample_name_is_replaced():
    # A named validation set (e.g. long-read consensus) matches whatever its name.
    val = VAL.assign(**{SAMPLE_ID_COL: 'other_name'})
    s2 = benchmark.validate_basic_callset(CALLSET, val, 'kosugi', 'delly', val_match_margin=10, sample='S2')
    assert dict(zip(s2['INFO_EVENT'], s2['is_in_val'])) == {'ev1': True, 'ev3': True}
    assert list(val[SAMPLE_ID_COL].unique()) == ['other_name']


def test_multi_sample_validation_set_rejected():
    val = pd.DataFrame([_bp('v1', 105, 'A'), _bp('v3', 9002, 'B')])
    with pytest.raises(ValueError, match='at most one'):
        benchmark.validate_basic_callset(CALLSET, val, 'kosugi', 'delly', sample='S1')


def test_empty_callset_benchmarks_to_empty():
    out = benchmark.validate_basic_callset(CALLSET.iloc[:0], VAL, 'kosugi', 'delly')
    assert out.empty
