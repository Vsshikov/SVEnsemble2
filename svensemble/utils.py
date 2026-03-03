"""
Utility functions for SVEnsemble
"""

import pandas as pd
from pathlib import Path
from collections import defaultdict, abc
from ._caller_params import caller_param_dict, training_set_presets, default_reference_set
from ._sample_params import sample_val_sets, sample_val_set_template, default_model_name
from ._model_configs import model_configs
from typing import Dict, Any

def update(d, u):
    for k, v in u.items():
        if isinstance(v, abc.Mapping):
            d[k] = update(d.get(k, {}), v)
        else:
            d[k] = v
    return d

human_chroms = [f'chr{i}' for i in range(1, 23)] + ['chrX', 'chrY', 'chrM']

def _get_gnomad_callset_dict(sample_name: str) -> dict:
    return defaultdict(str,
        {
            'source': 'gnomad',
            'svensemble_dir': Path(f'data/{sample_name}/svensemble2'),
            'path': Path(f'data/{sample_name}/{sample_name}.vcf.gz'),
            'params': caller_param_dict['gnomad']
        }
    )

def _get_gridss_callset_dict(sample_name: str) -> dict:
    return defaultdict(str,
        {
            'source': 'gridss',
            'svensemble_dir': Path(f'data/{sample_name}/sv_calling/gridss/svensemble2'),
            'path': Path(f'data/{sample_name}/sv_calling/gridss/out/{sample_name}.gridss.vcf'),
            'params': caller_param_dict['gridss']
        }
    )

def _get_manta_candidate_callset_dict(sample_name: str) -> dict:
    return defaultdict(str,
        {
            'source': 'manta_candidate',
            'svensemble_dir': Path(f'data/{sample_name}/sv_calling/manta/svensemble2'),
            'path': Path(f'data/{sample_name}/sv_calling/manta/results/variants/candidateSV.vcf.gz'),
            'params': caller_param_dict['manta_candidate']
        }
    )

def _get_manta_diploid_callset_dict(sample_name: str) -> dict:
    return defaultdict(str,
        {
            'source': 'manta_diploid',
            'svensemble_dir': Path(f'data/{sample_name}/sv_calling/manta/svensemble2'),
            'path': Path(f'data/{sample_name}/sv_calling/manta/results/variants/diploidSV.vcf.gz'),
            'params': caller_param_dict['manta_diploid']
        }
    )

def _get_delly_callset_dict(sample_name: str) -> dict:
    return defaultdict(str,
        {
            'source': 'delly',
            'svensemble_dir': Path(f'data/{sample_name}/sv_calling/delly/svensemble2'),
            'path': Path(f'data/{sample_name}/sv_calling/delly/out/{sample_name}.delly.bcf'),
            'params': caller_param_dict['delly']
        }
    )

def _get_survivor_callset_dict(sample_name: str) -> dict:
    return defaultdict(str,
        {
            'source': 'survivor',
            'svensemble_dir': Path(f'data/{sample_name}/sv_calling/survivor/svensemble2'),
            'path': Path(f'data/{sample_name}/sv_calling/survivor/{sample_name}.survivor.vcf'),
            'params': caller_param_dict['survivor']
        }
    )

def _get_gridss_H_callset_dict(sample_name: str) -> dict:
    return defaultdict(str,
        {
            'source': 'gridss',
            'svensemble_dir': Path(f'data/{sample_name}/sv_calling/gridss/svensemble2'),
            'path': Path(f'data/{sample_name}/sv_calling/gridss/out/{sample_name}.gridss.vcf'),
            'params': caller_param_dict['gridss_H']
        }
    )

def _get_sniffles_callset_dict(sample_name: str) -> dict:
    return defaultdict(str,
        {
            'source': 'sniffles',
            'svensemble_dir': Path(f'data/{sample_name}/sv_calling/sniffles/svensemble2'),
            'path': Path(f'data/{sample_name}/sv_calling/sniffles/out/{sample_name}.sniffles.vcf'),
            'params': caller_param_dict['sniffles']
        }
    )

def _get_sniffles_H_callset_dict(sample_name: str) -> dict:
    return defaultdict(str,
        {
            'source': 'sniffles',
            'svensemble_dir': Path(f'data/{sample_name}/sv_calling/sniffles/svensemble2'),
            'path': Path(f'data/{sample_name}/sv_calling/sniffles/out/{sample_name}.sniffles.vcf'),
            'params': caller_param_dict['sniffles_H']
        }
    )

def _get_caller_callset_dict(sample_name: str, caller_name: str) -> dict:
    caller_dispatch = {
        'gnomad': _get_gnomad_callset_dict,
        'gridss': _get_gridss_callset_dict,
        'manta_candidate': _get_manta_candidate_callset_dict,
        'manta_diploid': _get_manta_diploid_callset_dict,
        'delly': _get_delly_callset_dict,
        'sniffles': _get_sniffles_callset_dict,
        'sniffles_H': _get_sniffles_H_callset_dict,
        'survivor': _get_survivor_callset_dict,
        'gridss_H': _get_gridss_H_callset_dict
    }

    return caller_dispatch[caller_name](sample_name)

def create_config_dict(
    sample_name: str,
    ref_path: Path | None = None,
    is_ground_truth: bool = False,
    training_set: str | list[str] = 'default',
    val_set: None | str | dict[str, Any] = None,
    model_name: str = None,
    additional_callsets: list[str] | None = None
):
    """
    Based on _caller_params, create a config dictionary for the sample.
    """
    if isinstance(training_set, str):
        if training_set in training_set_presets.keys():
            _training_sets = {training_set: training_set_presets[training_set]}
        else:
            raise ValueError(f'Invalid training set: {training_set}')
    elif isinstance(training_set, list):
        _training_sets = {f'{'+'.join(training_set)}': training_set}
    else:
        raise ValueError(f'Invalid training set: {training_set}')

    # Training sets are all lists of calleset names
    # Find all unique callset names in the training sets
    callset_names = set()
    for training_set in _training_sets.values():
        for callset_name in training_set:
            callset_names.add(callset_name)
    callset_names = list(callset_names)
    if isinstance(additional_callsets, list):
        if any(callset_name not in caller_param_dict.keys() for callset_name in additional_callsets):
            raise ValueError(f'Invalid extra callsets: {additional_callsets}')
        callset_names.extend(additional_callsets)

    # If a validation set exists for the sample, add it to the config dict.
    if (val_set == 'use_name'):
        if sample_name not in sample_val_sets.keys():
            raise ValueError(f'Validation set not found for sample: {sample_name}')
        val_set = sample_val_sets[sample_name]
    elif val_set in sample_val_sets.keys():
        val_set = sample_val_sets[val_set]
    elif isinstance(val_set, dict):
        if all(key in val_set.keys() for key in sample_val_set_template.keys()):
            val_set = val_set
        else:
            raise ValueError(f'Invalid validation set: {val_set}')
        try:
            val_set['path'] = Path(val_set['path'])
        except Exception as e:
            raise ValueError(f'Invalid validation set path: {val_set['path']}, error: {e}')
    elif val_set is None:
        val_set = None
    else:
        raise ValueError(f'Invalid validation set: {val_set}')

    if model_name is None:
        model_name = default_model_name
    if model_name not in model_configs.keys():
        raise ValueError(f'Invalid model name: {model_name}')


    if is_ground_truth:
        config_dict = {
            sample_name: {
                'sample_dir': Path(f'data/{sample_name}'),
                'workflow_dir': Path(f'data/{sample_name}/svensemble2'),
                'is_reference': True,
                'callsets': {
                    sample_name: _get_caller_callset_dict(sample_name, default_reference_set)
                }
            }
        }
        return defaultdict(str, config_dict)

    training_set_name, training_set_callers = next(iter(_training_sets.items()))

    config_dict = {
        sample_name: {
            'sample_name': sample_name,
            'sample_dir': Path(f'data/{sample_name}'),
            'workflow_dir': Path(
                f'data/{sample_name}/sv_calling/svensemble2/'
                f'{sample_name}.{model_name}.'
                f'{training_set_name}'
            ),
            'is_reference': False,
            'callsets': {
                callset_name: _get_caller_callset_dict(
                    sample_name, caller_name=callset_name
                )
                for callset_name in callset_names
            },
            'training_sets': _training_sets,
            'ref_path': ref_path,
            'val_set': val_set,
            'model': model_configs[model_name]
        }
    }

    # If there are custom columns specified in the model config, add them to the config dict of each caller.
    if ('custom_cols' in config_dict[sample_name]['model']['training_params'].keys()) and len(config_dict[sample_name]['model']['training_params']['custom_cols']) > 0:
        print(f"Adding custom columns to the config dict for sample {sample_name}")
        print(f"Custom columns: {config_dict[sample_name]['model']['training_params']['custom_cols']}")
        for callset_name in callset_names:
            config_dict[sample_name]['callsets'][callset_name]['params']['custom_cols_pred'] = config_dict[sample_name]['model']['training_params']['custom_cols']

    # Specify the path to the matched breakpoints used for training the model
    # If the model uses matched_bps from another workflow
    if 'matched_bps_suffix' in config_dict[sample_name]['model'].keys():
        config_dict[sample_name]['model']['matched_bps_path'] = Path(
                f'data/{sample_name}/sv_calling/svensemble2/'
                f'{sample_name}.{config_dict[sample_name]['model']['matched_bps_suffix']}.'
                f'{training_set_name}'
                '/matched_bps.csv'
            )
    if 'feature_matrix_suffix' in config_dict[sample_name]['model'].keys():
        config_dict[sample_name]['model']['feature_matrix_path'] = Path(
                f'data/{sample_name}/sv_calling/svensemble2/'
                f'{sample_name}.{config_dict[sample_name]['model']['feature_matrix_suffix']}.'
                f'{training_set_name}'
                '/feature_matrix.csv'
            )

    for callset_name in callset_names:
        if config_dict[sample_name]['callsets'][callset_name]['params']['override_bps_path']:
            config_dict[sample_name]['callsets'][callset_name]['bps_path'] = config_dict[sample_name]['sample_dir'] / config_dict[sample_name]['callsets'][callset_name]['params']['bps_path']
        elif 'bps_suffix' in config_dict[sample_name]['model'].keys():
            config_dict[sample_name]['callsets'][callset_name]['bps_path'] = config_dict[sample_name]['sample_dir'] / f'sv_calling/svensemble2/{sample_name}.{config_dict[sample_name]['model']['bps_suffix']}.{training_set_name}' / config_dict[sample_name]['callsets'][callset_name]['params']['bps_path']
        else:
            config_dict[sample_name]['callsets'][callset_name]['bps_path'] = config_dict[sample_name]['workflow_dir'] / f'{callset_name}.bps.csv'
    if not config_dict:
        raise ValueError(f"No config dictionary created for sample {sample_name}")

    config_dict = defaultdict(str, config_dict)

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


def create_samples_config_table(samples: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    """
    Create a configuration table (CSV) listing all sample, training set, and model name combinations.

    Args:
        samples (Dict[str, Dict[str, Any]]):
            Dictionary mapping sample names to their configuration dictionaries.
            Each configuration dictionary is expected to contain a 'training_sets' key,
            which is iterable of training set names.

    Returns:
        pandas.DataFrame:
            DataFrame with columns ['sample_name', 'training_set_name', 'model_name'].
            Each row is a combination of sample, training set, and model name ("unbiased").
    """
    sample_names: list = []
    training_set_names: list = []
    model_names: list = []

    for sample_name, sample_config in samples.items():
        for training_set_name in sample_config['training_sets']:
            sample_names.append(sample_name)
            training_set_names.append(training_set_name)
            model_names.append('unbiased')

    return pd.DataFrame({
        'sample_name': sample_names,
        'training_set_name': training_set_names,
        'model_name': model_names
    })
