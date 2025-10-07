"""
Utility functions for SVEnsemble
"""

import pandas as pd
from pathlib import Path
from ._caller_params import caller_param_dict, training_set_presets, default_reference_set

def _get_gnomad_callset_dict(sample_name: str) -> dict:
    return {
        'source': 'gnomad',
        'svensemble_dir': Path(f'data/{sample_name}/svensemble2'),
        'path': Path(f'data/{sample_name}/{sample_name}.vcf.gz'),
        'params': caller_param_dict['gnomad']
    }

def _get_gridss_callset_dict(sample_name: str) -> dict:
    return {
        'source': 'gridss',
        'svensemble_dir': Path(f'data/{sample_name}/sv_calling/gridss/svensemble2'),
        'path': Path(f'data/{sample_name}/sv_calling/gridss/out/{sample_name}.gridss.vcf'),
        'params': caller_param_dict['gridss']
    }

def _get_manta_candidate_callset_dict(sample_name: str) -> dict:
    return {
        'source': 'manta_candidate',
        'svensemble_dir': Path(f'data/{sample_name}/sv_calling/manta/svensemble2'),
        'path': Path(f'data/{sample_name}/sv_calling/manta/results/variants/candidateSV.vcf.gz'),
        'params': caller_param_dict['manta_candidate']
    }

def _get_manta_diploid_callset_dict(sample_name: str) -> dict:
    return {
        'source': 'manta_diploid',
        'svensemble_dir': Path(f'data/{sample_name}/sv_calling/manta/svensemble2'),
        'path': Path(f'data/{sample_name}/sv_calling/manta/results/variants/diploidSV.vcf.gz'),
        'params': caller_param_dict['manta_diploid']
    }

def _get_delly_callset_dict(sample_name: str) -> dict:
    return {
        'source': 'delly',
        'svensemble_dir': Path(f'data/{sample_name}/sv_calling/delly/svensemble2'),
        'path': Path(f'data/{sample_name}/sv_calling/delly/out/{sample_name}.delly.bcf'),
        'params': caller_param_dict['delly']
    }

def _get_caller_callset_dict(sample_name: str, caller_name: str) -> dict:
    caller_dispatch = {
        'gnomad': _get_gnomad_callset_dict,
        'gridss': _get_gridss_callset_dict,
        'manta_candidate': _get_manta_candidate_callset_dict,
        'manta_diploid': _get_manta_diploid_callset_dict,
        'delly': _get_delly_callset_dict
    }

    return caller_dispatch[caller_name](sample_name)

def create_config_dict(sample_name: str, ref_path: Path | None = None, is_ground_truth: bool = False, training_sets: str | dict[str, list[str]] = 'default'):
    """
    Based on _caller_params, create a config dictionary for the sample.
    """
    if isinstance(training_sets, str):
        if training_sets in training_set_presets.keys():
            _training_sets = training_set_presets[training_sets]
        else:
            raise ValueError(f'Invalid training set: {training_sets}')
    elif isinstance(training_sets, dict):
        _training_sets = training_sets
    else:
        raise ValueError(f'Invalid training set: {training_sets}')

    # Training sets are all lists of calleset names
    # Find all unique callset names in the training sets
    callset_names = set()
    for training_set in _training_sets.values():
        for callset_name in training_set:
            callset_names.add(callset_name)
    callset_names = list(callset_names)

    if is_ground_truth:
        config_dict = {
            sample_name: {
                'sample_dir': Path(f'data/{sample_name}'),
                'is_reference': True,
                'callsets': {
                    sample_name: _get_caller_callset_dict(sample_name, default_reference_set)
                }
            }
        }
        return config_dict

    config_dict = {
        sample_name: {
            'sample_dir': Path(f'data/{sample_name}'),
            'is_reference': False,
            'callsets': {
                callset_name: _get_caller_callset_dict(sample_name, caller_name=callset_name)
                for callset_name in callset_names
            },
            'training_sets': _training_sets,
            'ref_path': ref_path
        }
    }

    if not config_dict:
        raise ValueError(f"No config dictionary created for sample {sample_name}")

    return config_dict


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

    try:
        return chrom_series.apply(lambda x: chrom_keys[x])
    except KeyError:
        # Return original if chromosome not in mapping
        return chrom_series


def split_cipos(cipos_col: pd.Series, default_cipos: tuple[int, int] = (0, 0)) -> pd.DataFrame:
    cipos_df = pd.DataFrame(cipos_col)
    cipos_df.loc[:, cipos_col.name.replace('INFO_CIPOS', 'INFO_CIPOS_l')] = cipos_col.str[0].fillna(default_cipos[0]).astype(int)
    cipos_df.loc[:, cipos_col.name.replace('INFO_CIPOS', 'INFO_CIPOS_r')] = cipos_col.str[1].fillna(default_cipos[1]).astype(int)
    return cipos_df.drop(columns=[cipos_col.name])


def split_ciend(ciend_col: pd.Series, default_ciend: tuple[int, int] = (0, 0)) -> pd.DataFrame:
    ciend_df = pd.DataFrame(ciend_col)
    ciend_df.loc[:, ciend_col.name.replace('INFO_CIEND', 'INFO_CIEND_l')] = ciend_col.str[0].fillna(default_ciend[0]).astype(int)
    ciend_df.loc[:, ciend_col.name.replace('INFO_CIEND', 'INFO_CIEND_r')] = ciend_col.str[1].fillna(default_ciend[1]).astype(int)
    return ciend_df.drop(columns=[ciend_col.name])