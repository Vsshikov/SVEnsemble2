"""
This module contains parameters to use for loading and processing VCF files
from specific callers supported by SVEnsemble2.
As well as parameters for PU learning with those sets.

Added callers:
- gridss
- manta

The file also contains parameters for working with VCF files from reference SV sets.
Currently, only gnomAD is supported.
"""

default_reference_set = 'gnomad'

base_cols = ['CHROM', 'POS', 'ID', 'REF', 'ALT', 'QUAL']

# Caller-specific parameters are defined as dictionaries with the following keys:
# - 'base_cols': List[str] - Standard VCF columns to parse (e.g., CHROM, POS, ID, REF, ALT, QUAL)
# - 'filter_cols': List[str] - FILTER fields to extract for pre-processing or prediction
# - 'info_cols_mandatory': List[str] - INFO fields required for correct parsing; absence may break downstream logic
# - 'info_cols_pred': List[str] - INFO fields used for call posterior probability estimation
# - 'parse_format': bool - Whether to parse FORMAT fields (default: False)
# - 'format_cols': List[str] - FORMAT fields to extract if 'parse_format' is True
# Example:
# gridss_params = {
#     'base_cols': base_cols,
#     'filter_cols': gridss_filters,
#     'info_cols_mandatory': gridss_info_mandatory_cols,
#     'info_cols_pred': gridss_info_pred_cols,
#     'parse_format': False,
#     'format_cols': []
# }

# Default parameters for all callers
default_filter_cols = []
default_info_mandatory_cols = []
default_info_pred_cols = []
default_format_cols = []
default_params = {
    'base_cols': base_cols,
    'filter_cols': default_filter_cols,
    'info_cols_mandatory': default_info_mandatory_cols,
    'info_cols_pred': default_info_pred_cols,
    'parse_format': False,
    'format_cols': default_format_cols,
    'is_reference': False,
}
training_set_presets = {
    'default': {
        'gridss': ['gridss'],
        'manta_candidate': ['manta_candidate'],
        'manta_diploid': ['manta_diploid'],
        'delly': ['delly'],
        'gridss+manta_candidate': ['gridss', 'manta_candidate'],
        'gridss+manta_diploid': ['gridss', 'manta_diploid'],
        'gridss+manta_candidate+delly': ['gridss', 'manta_candidate', 'delly'],
        'gridss+manta_diploid+delly': ['gridss', 'manta_diploid', 'delly']
    },
    'haploid': {
        'gridss': ['gridss'],
        'manta_candidate': ['manta_candidate'],
        'gridss+manta_candidate': ['gridss', 'manta_candidate']
    }
}

# Parameters for GRIDSS
# Filter entries expected from GRIDSS
gridss_filters = [
    'PASS',
    'ASSEMBLY_BIAS',
    'ASSEMBLY_ONLY',
    'ASSEMBLY_TOO_FEW_READ',
    'ASSEMBLY_TOO_SHORT',
    'INSUFFICIENT_SUPPORT',
    'LOW_MAPQ',
    'LOW_QUAL',
    'NO_ASSEMBLY',
    'NO_RP',
    'NO_SR',
    'REF',
    'SINGLE_ASSEMBLY',
    'SINGLE_SUPPORT',
    'SMALL_EVENT'
]
# INFO fields that must be present in GRIDSS VCF
gridss_info_mandatory_cols = [
    'MATEID',
    'EVENT',
    'CIPOS'
]
# INFO fields from GRIDSS VCF that will be used for call posterior probability estimation
gridss_info_pred_cols = [
    'ANRP',
    'ANRPQ',
    'ANSR',
    'ANSRQ',
    'ASQ',
    'ASRP',
    'ASSR',
    'BAQ',
    'BASRP',
    'BASSR',
    'BQ',
    'BSC',
    'BSCQ',
    'BUM',
    'BUMQ',
    'BVF',
    'CASQ',
    'IC',
    'IQ',
    'RASQ',
    'REFPAIR',
    'RP',
    'RPQ',
    'SR',
    'SRQ',
    'VF',
    'SVLEN',
    'AS',
    'BA',
    'BMQ',
    'BMQN',
    'BMQX',
    'CAS',
    'CQ',
    'IMPRECISE',
    'INSRMP',
    'MQ',
    'MQN',
    'MQX',
    'RAS',
    'RF',
    'SB',
    'SELF',
    'SVTYPE'
]

# Parameters for GRIDSS
gridss_params = default_params | {
    'filter_cols': gridss_filters,
    'info_cols_mandatory': gridss_info_mandatory_cols,
    'info_cols_pred': gridss_info_pred_cols,
    'parse_format': False,
    'svlen_source': 'alt',
    'bps_path': 'sv_calling/gridss/svensemble2/gridss_bps.csv'
}

gnomad_params = default_params | {
    'filter_cols': ['PASS'],
    'info_cols_mandatory': ['EVIDENCE', 'SVTYPE', 'CHR2', 'POS2', 'END', 'END2', 'CPX_INTERVALS', 'SVLEN'],
    'info_cols_pred': [],
    'parse_format': False,
    'svlen_source': 'svlen',
    'supported_svtypes': ['DEL', 'DUP', 'INV', 'INS'],
    'method_cipos': {
        'SR': 10,
        'PE': 100,
        'RD': 500
    },
    'is_reference': True,
    'bps_path': 'svensemble2/gnomad_bps.csv'
}

manta_candidate_params = default_params | {
    'filter_cols': ['PASS'],
    'info_cols_mandatory': ['SVTYPE', 'SVLEN', 'POS2', 'CIPOS', 'CIEND', 'MATEID', 'EVENT'],
    'info_cols_pred': ['IMPRECISE', 'HOMLEN', 'BND_DEPTH', 'SVLEN', 'PAIR_COUNT', 'UPSTREAM_PAIR_COUNT', 'DOWNSTREAM_PAIR_COUNT', 'BND_PAIR_COUNT'],
    'parse_format': False,
    'svlen_source': 'svlen',
    'bps_path': 'sv_calling/manta/svensemble2/manta_candidate_bps.csv'
}

manta_diploid_params = default_params | {
    'filter_cols': ['PASS'],
    'info_cols_mandatory': ['SVTYPE', 'SVLEN', 'POS2', 'CIPOS', 'CIEND', 'MATEID', 'EVENT'],
    'info_cols_pred': ['IMPRECISE', 'HOMLEN', 'BND_DEPTH', 'SVLEN'],
    'parse_format': True,
    'format_cols': ['FT', 'GQ', 'PR', 'SR'],
    'svlen_source': 'svlen',
    'bps_path': 'sv_calling/manta/svensemble2/manta_diploid_bps.csv'
}

delly_params = default_params | {
    'filter_cols': ['PASS', 'LowQual'],
    'info_cols_mandatory': ['SVTYPE', 'SVLEN', 'CHR2', 'POS2', 'CIPOS', 'CIEND'],
    'info_cols_pred': ['PE', 'MAPQ', 'SRMAPQ', 'SR', 'SRQ', 'CE', 'CT', 'SVLEN', 'IMPRECISE', 'SVTYPE', 'SVMETHOD', 'HOMLEN'],
    'parse_format': True,
    'format_cols': ['GT', 'GQ', 'RC', 'RCL', 'RCR', 'RDCN', 'DR', 'DV', 'RR', 'RV'],
    'svlen_source': 'svlen',
    'bps_path': 'sv_calling/delly/svensemble2/delly_bps.csv'
}

svensemble2_params = default_params | {
    'svlen_source': 'svensemble2_col'
}

kosugi_params = default_params | {
    'svlen_source': 'svlen'
}

chm13_params = default_params | {
    'svlen_source': 'svlen'
}

hg002_params = default_params | {
    'info_cols_mandatory': ['SVTYPE', 'SVLEN'],
    'svlen_source': 'svlen'
}

caller_param_dict = {
    'gnomad': gnomad_params,
    'gridss': gridss_params,
    'manta_candidate': manta_candidate_params,
    'manta_diploid': manta_diploid_params,
    'delly': delly_params,
    'svensemble2': svensemble2_params,
    'kosugi': kosugi_params,
    'chm13_val': chm13_params,
    'hg002_val': hg002_params
}

caller_alt_priority = [
    'gridss',
    'manta_candidate',
    'manta_diploid',
    'delly'
]