"""Unit tests for breakends.bnd_to_bp sample-aware pairing."""

import pandas as pd

from svensemble.breakends import bnd_to_bp
from svensemble._constants import SAMPLE_ID_COL


def _bnds(samples):
    rows = []
    for sample in samples:
        rows += [
            {'CHROM': 'chr2', 'POS': 500, 'INFO_EVENT': 'ev1', 'ID': 'ev1_2', 'QUAL': 10, SAMPLE_ID_COL: sample},
            {'CHROM': 'chr1', 'POS': 100, 'INFO_EVENT': 'ev1', 'ID': 'ev1_1', 'QUAL': 10, SAMPLE_ID_COL: sample},
            {'CHROM': 'chr1', 'POS': 900, 'INFO_EVENT': 'ev2', 'ID': 'ev2_1', 'QUAL': 5, SAMPLE_ID_COL: sample},
            {'CHROM': 'chr3', 'POS': 50, 'INFO_EVENT': 'ev2', 'ID': 'ev2_2', 'QUAL': 5, SAMPLE_ID_COL: sample},
        ]
    return pd.DataFrame(rows)


def test_two_samples_give_one_breakpoint_per_sample():
    bps = bnd_to_bp(_bnds(['S1', 'S2']))
    assert len(bps) == 4
    assert SAMPLE_ID_COL in bps.columns
    assert f'{SAMPLE_ID_COL}_bnd1' not in bps.columns
    assert sorted(zip(bps['INFO_EVENT'], bps[SAMPLE_ID_COL])) == [
        ('ev1', 'S1'), ('ev1', 'S2'), ('ev2', 'S1'), ('ev2', 'S2'),
    ]
    first = bps[(bps['INFO_EVENT'] == 'ev1') & (bps[SAMPLE_ID_COL] == 'S2')].iloc[0]
    assert (first['CHROM_bnd1'], first['POS_bnd1'], first['CHROM_bnd2'], first['POS_bnd2']) == ('chr1', 100, 'chr2', 500)


def test_single_sample_same_as_without_sample_column():
    with_sample = bnd_to_bp(_bnds(['S1']))
    without = bnd_to_bp(_bnds(['S1']).drop(columns=[SAMPLE_ID_COL]))
    pd.testing.assert_frame_equal(with_sample.drop(columns=[SAMPLE_ID_COL]), without)


def test_unpaired_breakend_dropped_per_sample():
    df = _bnds(['S1', 'S2'])
    df = df[~((df['ID'] == 'ev1_2') & (df[SAMPLE_ID_COL] == 'S2'))]
    bps = bnd_to_bp(df)
    assert sorted(zip(bps['INFO_EVENT'], bps[SAMPLE_ID_COL])) == [('ev1', 'S1'), ('ev2', 'S1'), ('ev2', 'S2')]
