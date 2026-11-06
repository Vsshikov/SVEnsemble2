"""benchmark: per-sample selection and sample-less validation sets."""

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
    assert set(benchmark._select_sample(CALLSET, 'S2')['INFO_EVENT']) == {'ev1', 'ev3'}
    one = CALLSET[CALLSET[SAMPLE_ID_COL] == 'S1']
    assert benchmark._select_sample(one, None) is one


def test_is_in_val_computed_within_the_sample():
    s1 = benchmark.validate_basic_callset(CALLSET, VAL, 'kosugi', 'delly', val_match_margin=10, sample='S1')
    s2 = benchmark.validate_basic_callset(CALLSET, VAL, 'kosugi', 'delly', val_match_margin=10, sample='S2')
    assert dict(zip(s1['INFO_EVENT'], s1['is_in_val'])) == {'ev1': True, 'ev2': False}
    assert dict(zip(s2['INFO_EVENT'], s2['is_in_val'])) == {'ev1': True, 'ev3': True}


def test_validation_set_with_sample_column_rejected():
    val = VAL.assign(**{SAMPLE_ID_COL: 'S1'})
    with pytest.raises(ValueError, match='sample-less'):
        benchmark.validate_basic_callset(CALLSET, val, 'kosugi', 'delly', sample='S1')
