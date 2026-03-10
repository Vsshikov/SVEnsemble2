"""Shared fixtures, DataFrame builders, and schema helpers for preprocessing tests."""

import pandas as pd
import pytest

from svensemble import _caller_params

# ---------------------------------------------------------------------------
# Schema constants -- downstream contracts that preprocessed DataFrames must meet
# ---------------------------------------------------------------------------

EVENT_TO_BND_REQUIRED_COLS = [
    'CHROM', 'POS', 'INFO_SVTYPE', 'INFO_SVLEN',
    'INFO_CIPOS_l', 'INFO_CIPOS_r', 'INFO_CIEND_l', 'INFO_CIEND_r',
    'INFO_CHR2', 'INFO_EVENT',
]

EVENT_TO_BND_REQUIRED_DTYPES = {
    'INFO_SVLEN': 'int',
    'INFO_CIPOS_l': 'int',
    'INFO_CIPOS_r': 'int',
    'INFO_CIEND_l': 'int',
    'INFO_CIEND_r': 'int',
}

BND_TO_BP_REQUIRED_COLS = ['CHROM', 'POS', 'INFO_EVENT']

BREAKPOINT_CORE_COLS = [
    'CHROM_bnd1', 'POS_bnd1', 'CHROM_bnd2', 'POS_bnd2', 'INFO_EVENT',
]

# Columns that must NOT survive caller-specific preprocessing (they should
# have been split/dropped into _l/_r variants).
CIPOS_RAW_COLS = ['INFO_CIPOS', 'INFO_CIEND']


# ---------------------------------------------------------------------------
# Schema assertion helpers
# ---------------------------------------------------------------------------

def assert_has_columns(df: pd.DataFrame, required: list[str]) -> None:
    missing = set(required) - set(df.columns)
    assert not missing, f"Missing columns: {missing}"


def assert_missing_columns(df: pd.DataFrame, forbidden: list[str]) -> None:
    present = set(forbidden) & set(df.columns)
    assert not present, f"Columns should be absent but found: {present}"


def assert_column_dtypes(df: pd.DataFrame, dtype_map: dict[str, str]) -> None:
    for col, expected in dtype_map.items():
        if col not in df.columns:
            continue
        actual = df[col].dtype
        assert expected in str(actual), (
            f"Column '{col}': expected dtype containing '{expected}', got '{actual}'"
        )


# ---------------------------------------------------------------------------
# Minimal caller_params fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def manta_candidate_params():
    return _caller_params.manta_candidate_params


@pytest.fixture
def manta_diploid_params():
    return _caller_params.manta_diploid_params


@pytest.fixture
def gridss_params():
    return _caller_params.gridss_params


@pytest.fixture
def delly_params():
    return _caller_params.delly_params


@pytest.fixture
def sniffles_params():
    return _caller_params.sniffles_params


@pytest.fixture
def survivor_params():
    return _caller_params.survivor_params


# ---------------------------------------------------------------------------
# DataFrame builder helpers -- produce minimal synthetic DataFrames matching
# what parse_vcf_to_df would return for each caller.
# ---------------------------------------------------------------------------

def make_gridss_bnd_df(n: int = 4) -> pd.DataFrame:
    """Synthetic GRIDSS BND DataFrame (pairs of breakends)."""
    rows = []
    for i in range(n):
        event = f"gridss_event_{i // 2}"
        rows.append({
            'CHROM': 'chr1', 'POS': 1000 + i * 100, 'ID': f'gridss{i}',
            'REF': 'N', 'ALT': f'N[chr1:{2000 + i * 100}[',
            'QUAL': 50.0 + i, 'FILTER_PASS': 1,
            'INFO_SVTYPE': 'BND', 'INFO_EVENT': event,
            'INFO_MATEID': f'gridss{i ^ 1}',
            'INFO_CIPOS': (-10, 10),
        })
    return pd.DataFrame(rows)


def make_manta_evt_df(n: int = 2) -> pd.DataFrame:
    """Synthetic Manta event DataFrame (DEL + INS)."""
    rows = []
    svtypes = ['DEL', 'INS']
    for i in range(n):
        rows.append({
            'CHROM': 'chr1', 'POS': 1000 + i * 500,
            'ID': f'manta_evt_{i}', 'REF': 'N', 'ALT': '<' + svtypes[i % 2] + '>',
            'QUAL': 100.0, 'INFO_END': 1500 + i * 500,
            'FILTER_PASS': 1,
            'INFO_SVTYPE': svtypes[i % 2],
            'INFO_SVLEN': (-200,) if svtypes[i % 2] == 'DEL' else (50,),
            'INFO_CIPOS': (-5, 5), 'INFO_CIEND': (-8, 8),
            'INFO_POS2': 0, 'INFO_HOMLEN': (3,),
            'INFO_CHR2': None, 'INFO_EVENT': None,
            'SAMPLE_SR': (5, 10), 'SAMPLE_PR': (3, 7),
        })
    return pd.DataFrame(rows)


def make_manta_bnd_df(n: int = 4) -> pd.DataFrame:
    """Synthetic Manta BND DataFrame (pairs of breakends)."""
    rows = []
    for i in range(n):
        event = f'manta_bnd_event_{i // 2}'
        rows.append({
            'CHROM': 'chr1', 'POS': 2000 + i * 100,
            'ID': f'manta_bnd_{i}', 'REF': 'N',
            'ALT': f'N[chr2:{3000 + i * 100}[',
            'QUAL': 80.0, 'INFO_END': 3000 + i * 100,
            'FILTER_PASS': 1,
            'INFO_SVTYPE': 'BND', 'INFO_EVENT': event,
            'INFO_MATEID': f'manta_bnd_{i ^ 1}',
            'INFO_CIPOS': (-3, 3), 'INFO_CIEND': None,
            'INFO_POS2': 0, 'INFO_HOMLEN': (0,),
            'INFO_CHR2': None,
            'INFO_SVLEN': None,
            'SAMPLE_SR': (2, 4), 'SAMPLE_PR': (1, 3),
        })
    return pd.DataFrame(rows)


def make_delly_evt_df(n: int = 2) -> pd.DataFrame:
    """Synthetic Delly event DataFrame."""
    rows = []
    for i in range(n):
        rows.append({
            'CHROM': 'chr1', 'POS': 5000 + i * 1000,
            'ID': f'DEL0000{i}', 'REF': 'N', 'ALT': '<DEL>',
            'QUAL': 200.0, 'INFO_END': 5500 + i * 1000,
            'FILTER_PASS': 1,
            'INFO_SVTYPE': 'DEL', 'INFO_SVLEN': -500,
            'INFO_CIPOS': (-20, 20), 'INFO_CIEND': (-15, 15),
            'INFO_CHR2': None if i == 0 else 'chr1',
        })
    return pd.DataFrame(rows)


def make_delly_bnd_df(n: int = 4) -> pd.DataFrame:
    """Synthetic Delly BND DataFrame."""
    rows = []
    for i in range(n):
        rows.append({
            'CHROM': 'chr1', 'POS': 7000 + i * 100,
            'ID': f'BND00001_bp{i}', 'REF': 'N',
            'ALT': f'N[chr2:{8000 + i * 100}[',
            'QUAL': 150.0, 'INFO_END': 0,
            'FILTER_PASS': 1,
            'INFO_SVTYPE': 'BND', 'INFO_SVLEN': 0,
            'INFO_CIPOS': (-5, 5),
            'INFO_POS2': 8000 + i * 100,
            'INFO_CHR2': 'chr2',
            'INFO_EVENT': f'BND00001_bp{i}',
        })
    return pd.DataFrame(rows)


def make_sniffles_evt_df(n: int = 2) -> pd.DataFrame:
    """Synthetic Sniffles event DataFrame."""
    rows = []
    svtypes = ['DEL', 'INS']
    for i in range(n):
        rows.append({
            'CHROM': 'chr1', 'POS': 9000 + i * 1000,
            'ID': f'Sniffles2.DEL.{i}', 'REF': 'N',
            'ALT': '<' + svtypes[i % 2] + '>',
            'QUAL': 30.0, 'FILTER_PASS': 1,
            'INFO_SVTYPE': svtypes[i % 2],
            'INFO_SVLEN': -300 if svtypes[i % 2] == 'DEL' else 50,
            'INFO_STDEV_POS': 12.7,
            'INFO_STDEV_LEN': 8.3,
            'INFO_CHR2': None,
        })
    return pd.DataFrame(rows)


def make_sniffles_bnd_df(n: int = 4) -> pd.DataFrame:
    """Synthetic Sniffles BND DataFrame."""
    rows = []
    for i in range(n):
        rows.append({
            'CHROM': 'chr1', 'POS': 11000 + i * 100,
            'ID': f'Sniffles2.BND.{i}', 'REF': 'N',
            'ALT': f'N[chr3:{12000 + i * 100}[',
            'QUAL': 25.0, 'FILTER_PASS': 1,
            'INFO_SVTYPE': 'BND',
            'INFO_SVLEN': 0,
            'INFO_STDEV_POS': 5.0,
            'INFO_CHR2': 'chr3',
            'INFO_EVENT': f'Sniffles2.BND.{i}',
        })
    return pd.DataFrame(rows)


def make_survivor_evt_df(n: int = 2) -> pd.DataFrame:
    """Synthetic SURVIVOR event DataFrame."""
    rows = []
    for i in range(n):
        rows.append({
            'CHROM': 'chr1', 'POS': 13000 + i * 2000,
            'ID': f'survivor_{i}', 'REF': 'N', 'ALT': '<DEL>',
            'QUAL': 0.0, 'INFO_END': 13500 + i * 2000,
            'FILTER_PASS': 1,
            'INFO_SVTYPE': 'DEL', 'INFO_SVLEN': 500,
            'INFO_CIPOS': (-10, 10), 'INFO_CIEND': (-12, 12),
            'INFO_CHR2': 'chr1',
        })
    return pd.DataFrame(rows)
