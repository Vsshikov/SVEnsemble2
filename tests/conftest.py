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
    return _caller_params.MANTA_CANDIDATE_PARAMS


@pytest.fixture
def manta_diploid_params():
    return _caller_params.MANTA_DIPLOID_PARAMS


@pytest.fixture
def gridss_params():
    return _caller_params.GRIDSS_PARAMS


@pytest.fixture
def delly_params():
    return _caller_params.DELLY_PARAMS


@pytest.fixture
def sniffles_params():
    return _caller_params.SNIFFLES_PARAMS


@pytest.fixture
def survivor_params():
    return _caller_params.SURVIVOR_PARAMS


@pytest.fixture
def cutesv_params():
    return _caller_params.CUTESV_PARAMS


@pytest.fixture
def svaba_params():
    return _caller_params.SVABA_PARAMS


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
    """Synthetic Manta BND DataFrame (pairs of breakends).

    Reflects real Manta VCF output: BND records have no EVENT INFO field
    (INFO_EVENT is None) and carry a MATEID tuple pointing to the partner.
    manta_preprocess_bnds_only is responsible for deriving INFO_EVENT.
    """
    rows = []
    for i in range(n):
        rows.append({
            'CHROM': 'chr1', 'POS': 2000 + i * 100,
            'ID': f'manta_bnd_{i}', 'REF': 'N',
            'ALT': f'N[chr2:{3000 + i * 100}[',
            'QUAL': 80.0, 'INFO_END': 3000 + i * 100,
            'FILTER_PASS': 1,
            'INFO_SVTYPE': 'BND', 'INFO_EVENT': None,
            'INFO_MATEID': (f'manta_bnd_{i ^ 1}',),
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
            'INFO_COVERAGE': (10, 15, 20, 15, 10),
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
            'INFO_COVERAGE': (10, 15, 20, 15, 10),
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


def make_severus_evt_df(n: int = 2) -> pd.DataFrame:
    """Synthetic Severus event DataFrame."""
    rows = []
    svtypes = ['DEL', 'INS']
    for i in range(n):
        rows.append({
            'CHROM': 'chr1', 'POS': 15000 + i * 1000,
            'ID': f'severus_DEL{i}' if svtypes[i % 2] == 'DEL' else f'severus_INS{i}',
            'REF': 'N',
            'ALT': '<' + svtypes[i % 2] + '>',
            'QUAL': 60.0, 'INFO_END': 15500 + i * 1000,
            'FILTER_PASS': 1,
            'FILTER_FAIL_LOWSUPP': 0,
            'INFO_SVTYPE': svtypes[i % 2],
            'INFO_SVLEN': -500 if svtypes[i % 2] == 'DEL' else 200,
            'INFO_PRECISE': i % 2 == 0,
            'INFO_IMPRECISE': i % 2 != 0,
            'INFO_MAPQ': 60,
            'INFO_INSLEN': None,
            'SAMPLE_DR': 20,
            'SAMPLE_DV': 10,
            'SAMPLE_VAF': 0.33,
            'SAMPLE_hVAF': (0.5, 0.0, 0.0),
        })
    return pd.DataFrame(rows)


def make_severus_bnd_df(n: int = 4) -> pd.DataFrame:
    """Synthetic Severus BND DataFrame."""
    rows = []
    for i in range(n):
        rows.append({
            'CHROM': 'chr1', 'POS': 17000 + i * 100,
            'ID': f'severus_BND{i // 2}_{(i % 2) + 1}',
            'REF': 'N',
            'ALT': f'N]chr2:{18000 + i * 100}]',
            'QUAL': 60.0, 'INFO_END': 0,
            'FILTER_PASS': 1,
            'FILTER_FAIL_LOWSUPP': 0,
            'INFO_SVTYPE': 'BND',
            'INFO_SVLEN': 1000,
            'INFO_MATE_ID': f'severus_BND{i // 2}_{2 - (i % 2)}',
            'INFO_PRECISE': True,
            'INFO_IMPRECISE': False,
            'INFO_MAPQ': 60,
            'INFO_INSLEN': None,
            'SAMPLE_DR': 15,
            'SAMPLE_DV': 8,
            'SAMPLE_VAF': 0.35,
            'SAMPLE_hVAF': (0.4, 0.0, 0.0),
        })
    return pd.DataFrame(rows)


def make_cutesv_evt_df(n: int = 4) -> pd.DataFrame:
    """Synthetic CuteSV event DataFrame (DEL, INS, DUP, INV).

    DUP and INV rows simulate pysam's behaviour when CIPOS/CILEN are absent
    from the VCF: pysam returns None, which becomes NaN after expansion.
    """
    rows = []
    svtypes = ['DEL', 'INS', 'DUP', 'INV']
    for i in range(n):
        svtype = svtypes[i % 4]
        has_ci = svtype in ('DEL', 'INS')
        rows.append({
            'CHROM': 'chr1', 'POS': 19000 + i * 1000,
            'ID': f'cuteSV.{svtype}.{i}', 'REF': 'N',
            'ALT': f'<{svtype}>',
            'QUAL': 200.0, 'INFO_END': 19500 + i * 1000,
            'FILTER_PASS': 1,
            'INFO_SVTYPE': svtype,
            'INFO_SVLEN': -500 if svtype == 'DEL' else 200,
            'INFO_CIPOS': (-18, 18) if has_ci else None,
            'INFO_CILEN': (-10, 10) if has_ci else None,
            'INFO_RE': 22,
            'INFO_AF': (0.95,),
            'INFO_PRECISE': True,
            'INFO_IMPRECISE': False,
            'INFO_CHR2': None,
            'SAMPLE_GT': '1/1',
            'SAMPLE_GQ': 48,
            'SAMPLE_DR': 0,
            'SAMPLE_DV': 22,
        })
    return pd.DataFrame(rows)


def make_cutesv_bnd_df(n: int = 4) -> pd.DataFrame:
    """Synthetic CuteSV BND DataFrame."""
    rows = []
    for i in range(n):
        rows.append({
            'CHROM': 'chr1', 'POS': 21000 + i * 100,
            'ID': f'cuteSV.BND.{i}', 'REF': 'N',
            'ALT': f'N]chr2:{22000 + i * 100}]',
            'QUAL': 0.0, 'INFO_END': 0,
            'FILTER_PASS': 0,
            'FILTER_q5': 1,
            'INFO_SVTYPE': 'BND',
            'INFO_SVLEN': 0,
            'INFO_RE': 15,
            'INFO_AF': (0.2,),
            'INFO_PRECISE': i % 2 == 0,
            'INFO_IMPRECISE': i % 2 != 0,
            'INFO_CHR2': 'chr2',
            'SAMPLE_GT': '0/0',
            'SAMPLE_GQ': 30,
            'SAMPLE_DR': 60,
            'SAMPLE_DV': 15,
        })
    return pd.DataFrame(rows)


def make_svaba_bnd_df(n: int = 4) -> pd.DataFrame:
    """Synthetic SvABA BND DataFrame (pairs of breakends).

    4 records forming 2 breakpoint pairs.  IDs follow the <hash>:1 / <hash>:2
    pattern used by SvABA.  Pair 0 is precise (assembly), pair 1 is imprecise
    (discordant-only).
    """
    rows = []
    evdnc_vals = ['ASSMB', 'ASSMB', 'DSCRD', 'DSCRD']
    homseqs = ['AGCT', 'AGCT', None, None]
    for i in range(n):
        pair_id = i // 2
        mate_suffix = 2 if (i % 2 == 0) else 1
        is_imprecise = pair_id == 1
        rows.append({
            'CHROM': 'chr1', 'POS': 23000 + i * 200,
            'ID': f'{1000000 + pair_id}:{(i % 2) + 1}',
            'REF': 'N',
            'ALT': f'N[chr1:{24000 + i * 200}[',
            'QUAL': 60.0 - i * 5,
            'FILTER_PASS': 1 if pair_id == 0 else 0,
            'FILTER_LOWMAPQ': 0 if pair_id == 0 else 1,
            'INFO_SVTYPE': 'BND',
            'INFO_MATEID': f'{1000000 + pair_id}:{mate_suffix}',
            'INFO_IMPRECISE': is_imprecise,
            'INFO_EVDNC': evdnc_vals[i],
            'INFO_HOMSEQ': homseqs[i],
            'INFO_HOMLEN': None,
            'INFO_DISC_MAPQ': -1 if not is_imprecise else 45,
            'INFO_NUMPARTS': 2 if not is_imprecise else 0,
            'INFO_MAPQ': 60 if not is_imprecise else -1,
            'INFO_SPAN': 1000,
            'INFO_NM': 0 if not is_imprecise else -1,
            'INFO_MATENM': 1 if not is_imprecise else -1,
            'INFO_SUBN': None,
            'INFO_SECONDARY': False,
            'SAMPLE_GT': '0/1',
            'SAMPLE_GQ': '0',
            'SAMPLE_AD': 15,
            'SAMPLE_DP': 40,
            'SAMPLE_SR': 10 if not is_imprecise else 0,
            'SAMPLE_DR': 5 if not is_imprecise else 12,
            'SAMPLE_LO': 55.0 - i * 5,
            'SAMPLE_SL': 50.0 - i * 5,
            'SAMPLE_LR': -50.0 + i * 5,
            'SAMPLE_PL': (60.0 - i * 5, 0.0, 80.0 + i * 5),
        })
    return pd.DataFrame(rows)
