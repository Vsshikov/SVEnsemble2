"""
Utility functions for SVEnsemble
"""
import os
import secrets
from collections import abc
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import pandas as pd

from ._constants import CI_LEFT, CI_RIGHT


@contextmanager
def atomic_path(path: Path) -> Iterator[Path]:
    """Yield a temporary sibling of *path* to write to; move it onto *path* on success.

    The temporary name is ``.partial-<token>-<name>``: in the same directory, so
    the final ``os.replace`` is atomic (readers see the old file or the complete
    new one, never a partial one); with the same suffix, so suffix-driven writers
    (pysam ``wz`` for ``.gz``, pandas compression) behave as for *path*; and
    hidden, so globs and existence checks on *path* ignore it. The file itself
    is created by the writer, under the normal umask.

    On any exit other than success -- exceptions, ``SystemExit``,
    ``KeyboardInterrupt`` -- the temporary file is removed and *path* is left
    untouched. The parent directory is created if missing.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".partial-{secrets.token_hex(4)}-{path.name}")
    ok = False
    try:
        yield tmp
        ok = True
    finally:
        if ok:
            os.replace(tmp, path)
        else:
            tmp.unlink(missing_ok=True)

def get_base_dir() -> Path:
    base = os.environ.get("SVENSEMBLE_BASE_DIR")
    if base:
        return Path(base)
    # Fallback: assume we're running from the project root
    return Path.cwd()

BASE_DIR = get_base_dir()
DATA_DIR = BASE_DIR / "data"

def update(d, u):
    for k, v in u.items():
        if isinstance(v, abc.Mapping):
            d[k] = update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


human_chroms = [f'chr{i}' for i in range(1, 23)] + ['chrX', 'chrY', 'chrM']


def generate_chrom_key_dict(chrom_series: pd.Series) -> dict:
    """
    Generate a dictionary mapping chromosome names to numeric indices for sorting.

    Creates a mapping of standard chromosome names (chr1-22, X, Y, M) plus any
    non-standard chromosomes found in the input series to sequential indices.

    Args:
        chrom_series: Series containing chromosome names

    Returns:
        Dictionary mapping chromosome names to numeric indices
    """
    # Standard chromosome list in desired sort order
    chrom_list = ['chr' + str(i) for i in range(1, 23)] + ['chrX', 'chrY', 'chrM']

    # Add any non-standard chromosomes found in input
    nonstd_chrom_list = chrom_series[~chrom_series.isin(chrom_list)].unique().tolist()
    chrom_list += nonstd_chrom_list

    # Create mapping of chromosome names to indices
    return {chrom: idx for idx, chrom in enumerate(chrom_list)}


def get_chrom_keys(chrom_series: pd.Series, chrom_keys: dict | None = None) -> pd.Series:
    """
    Convert chromosome names to numeric keys for sorting.

    Maps chromosome names to numeric indices using either provided mapping
    or default standard chromosome order.

    Args:
        chrom_series: Series containing chromosome names
        chrom_keys: Optional dictionary mapping chromosomes to indices.
                   If None, uses standard chromosome order.

    Returns:
        Series with chromosome names converted to numeric indices.
        Returns original series if chromosome not found in mapping.
    """
    if chrom_keys is None:
        # Use standard chromosome order if no mapping provided
        chrom_list = ['chr' + str(i) for i in range(1, 23)] + ['chrX', 'chrY', 'chrM']
        chrom_keys = {chrom: idx for idx, chrom in enumerate(chrom_list)}

    return chrom_series.map(chrom_keys).fillna(chrom_series)


def split_cipos(cipos_col: pd.Series, default_cipos: tuple[int, int] = (0, 0)) -> pd.DataFrame:
    cipos_df = pd.DataFrame(cipos_col)
    cipos_df.loc[:, cipos_col.name.replace('INFO_CIPOS', f'INFO_CIPOS_{CI_LEFT}')] = cipos_col.str[0].fillna(default_cipos[0]).astype(int)
    cipos_df.loc[:, cipos_col.name.replace('INFO_CIPOS', f'INFO_CIPOS_{CI_RIGHT}')] = cipos_col.str[1].fillna(default_cipos[1]).astype(int)
    return cipos_df.drop(columns=[cipos_col.name])


def split_ciend(ciend_col: pd.Series, default_ciend: tuple[int, int] = (0, 0)) -> pd.DataFrame:
    ciend_df = pd.DataFrame(ciend_col)
    ciend_df.loc[:, ciend_col.name.replace('INFO_CIEND', f'INFO_CIEND_{CI_LEFT}')] = ciend_col.str[0].fillna(default_ciend[0]).astype(int)
    ciend_df.loc[:, ciend_col.name.replace('INFO_CIEND', f'INFO_CIEND_{CI_RIGHT}')] = ciend_col.str[1].fillna(default_ciend[1]).astype(int)
    return ciend_df.drop(columns=[ciend_col.name])
