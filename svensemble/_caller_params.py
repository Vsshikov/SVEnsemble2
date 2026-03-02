"""
This module contains parameters to use for loading and processing VCF files
from specific callers supported by SVEnsemble2.
As well as parameters for PU learning with those sets.

Added callers:
- gridss
- manta
- delly
- sniffles

The file also contains parameters for working with VCF files from reference SV sets.
Currently, only gnomAD is supported.
"""

import numpy as np

default_reference_set = 'gnomad'

default_sv_types = ['DEL', 'INS', 'DUP', 'INV', 'TRA', 'OTHER']
default_sv_lens = (50, np.inf)

svtype_cols = ['INFO_SVTYPE', 'INFO_INFERRED_SVTYPE']

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
default_custom_cols_pred = []
default_params = {
    'base_cols': base_cols,
    'filter_cols': default_filter_cols,
    'info_cols_mandatory': default_info_mandatory_cols,
    'info_cols_pred': default_info_pred_cols,
    'custom_cols_pred': default_custom_cols_pred,
    'parse_format': False,
    'format_cols': default_format_cols,
    'is_reference': False,
    'qual_metric': 'QUAL',
    'svlen_source': 'svlen',
    'bnd2_pos': 'svlen',
    'svlens': default_sv_lens,
    'override_bps_path': False
}
# Default params for evnet notation in VCF
default_event_params = default_params | {
    'base_cols': base_cols + ['INFO_END'],
    'info_cols_mandatory': ['CIEND', 'CIPOS', 'CHR2', 'SVLEN', 'SVTYPE']
}

training_set_presets = {
    'default': ['delly', 'manta_diploid'],
    'gridss': ['gridss'],
    'manta_candidate': ['manta_candidate'],
    'manta_diploid': ['manta_diploid'],
    'delly': ['delly'],
    'gridss+manta_candidate': ['gridss', 'manta_candidate'],
    'gridss+manta_diploid': ['gridss', 'manta_diploid'],
    'gridss+delly': ['gridss', 'delly'],
    'gridss+manta_candidate+delly': ['gridss', 'manta_candidate', 'delly'],
    'gridss+manta_diploid+delly': ['gridss', 'manta_diploid', 'delly'],
    'delly+manta_candidate': ['delly', 'manta_candidate'],
    'delly+manta_diploid': ['delly', 'manta_diploid']
}

training_set_list_presets = {
    'default': [
        'gridss',
        'manta_candidate',
        'manta_diploid',
        'delly',
        'gridss+manta_candidate',
        'gridss+manta_diploid',
        'gridss+delly',
        'gridss+manta_candidate+delly',
        'gridss+manta_diploid+delly',
        'delly+manta_candidate',
        'delly+manta_diploid'
    ]
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
    'source': 'gridss',
    'filter_cols': gridss_filters,
    'info_cols_mandatory': gridss_info_mandatory_cols,
    'info_cols_pred': gridss_info_pred_cols,
    'parse_format': False,
    'svlen_source': 'alt',
    'bps_path': 'gridss.bps.csv',
    'bnd2_pos': None
}

gnomad_params = default_params | {
    'source': 'gnomad',
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
    'bps_path': 'gnomad.bps.csv'
}

manta_candidate_params = default_params | {
    'source': 'manta_candidate',
    'base_cols': base_cols + ['INFO_END'],
    'filter_cols': ['PASS'],
    'info_cols_mandatory': ['SVTYPE', 'SVLEN', 'POS2', 'CIPOS', 'CIEND', 'MATEID', 'EVENT'],
    'info_cols_pred': ['IMPRECISE', 'HOMLEN', 'BND_DEPTH', 'SVLEN', 'PAIR_COUNT', 'UPSTREAM_PAIR_COUNT', 'DOWNSTREAM_PAIR_COUNT', 'BND_PAIR_COUNT'],
    'parse_format': False,
    'svlen_source': 'svlen',
    'bps_path': 'manta_candidate.bps.csv',
    'qual_metric': 'INFO_TOTAL_PAIR_COUNT',
    'bnd2_pos': 'end'
}

manta_diploid_params = default_params | {
    'source': 'manta_diploid',
    'base_cols': base_cols + ['INFO_END'],
    'filter_cols': ['PASS'],
    'info_cols_mandatory': ['SVTYPE', 'SVLEN', 'POS2', 'CIPOS', 'CIEND', 'MATEID', 'EVENT'],
    'info_cols_pred': ['IMPRECISE', 'HOMLEN', 'BND_DEPTH', 'SVLEN'],
    'parse_format': True,
    'format_cols': ['FT', 'GQ', 'PR', 'SR'],
    'svlen_source': 'svlen',
    'bps_path': 'manta_diploid.bps.csv',
    'bnd2_pos': 'end'
}

delly_params = default_params | {
    'source': 'delly',
    'base_cols': base_cols + ['INFO_END'],
    'filter_cols': ['PASS', 'LowQual'],
    'info_cols_mandatory': ['SVTYPE', 'SVLEN', 'CHR2', 'POS2', 'CIPOS', 'CIEND'],
    'info_cols_pred': ['PE', 'MAPQ', 'SRMAPQ', 'SR', 'SRQ', 'CE', 'CT', 'SVLEN', 'IMPRECISE', 'SVTYPE', 'SVMETHOD', 'HOMLEN'],
    'parse_format': True,
    'format_cols': ['GT', 'GQ', 'RC', 'RCL', 'RCR', 'RDCN', 'DR', 'DV', 'RR', 'RV'],
    'svlen_source': 'svlen',
    'bps_path': 'delly.bps.csv',
    'bnd2_pos': 'end'
}

survivor_params = default_event_params | {
    'source': 'survivor',
    'bps_path': 'sv_calling/survivor/svensemble2/survivor.bps.csv',
    'bnd2_pos': 'end',
    'override_bps_path': True
}

svensemble2_params = default_params | {
    'svlen_source': 'svensemble2_col'
}

kosugi_params = default_params | {
    'svlen_source': 'svlen',
    'svlens': (50, np.inf)
}

parikh_params = default_params | {
    'svlen_source': 'svlen',
    'svlens': (50, np.inf)
}

chm13_params = default_params | {
    'svlen_source': 'svlen',
    'bnd2_pos': 'end'
}

hg002_params = default_params | {
    'info_cols_mandatory': ['SVTYPE', 'SVLEN'],
    'svlen_source': 'svlen'
}

hg00514_params = default_params | {
    'info_cols_mandatory': ['SVTYPE', 'SVLEN'],
    'svlen_source': 'svlen'
}

huddleston_params = default_params | {
    'base_cols': base_cols + ['INFO_END'],
    'info_cols_mandatory': ['SVTYPE', 'SVLEN'],
    'svlen_source': 'svlen',
    'svlens': (50, np.inf),
    'svtype_mapping': {
        'deletion': 'DEL',
        'insertion': 'INS',
        'inversion': 'INV'
    }
}

gridss_H_params = gridss_params | {
    'bps_path': 'sv_calling/gridss/svensemble2/gridss_H_bps.csv',
    'override_bps_path': True
}

# Parameters for Sniffles2 (long-read SV caller)
# Filter entries expected from Sniffles2
sniffles_filters = [
    'PASS',
    'GT',
    'SUPPORT_MIN',
    'STDEV_POS',
    'STDEV_LEN',
    'COV_MIN',
    'COV_MIN_GT',
    'COV_CHANGE_DEL',
    'COV_CHANGE_DUP',
    'COV_CHANGE_INS',
    'COV_CHANGE_FRAC_US',
    'COV_CHANGE_FRAC_SC',
    'COV_CHANGE_FRAC_CE',
    'COV_CHANGE_FRAC_ED',
    'COV_VAR',
    'MOSAIC_VAF',
    'NOT_MOSAIC_VAF',
    'ALN_NM',
    'STRAND_BND',
    'STRAND',
    'STRAND_MOSAIC',
    'SVLEN_MIN',
    'SVLEN_MIN_MOSAIC',
    'SVLEN_MAX_MOSAIC',
    'SINGLE_BREAK'
]
# INFO fields that must be present in Sniffles2 VCF
sniffles_info_mandatory_cols = [
    'SVTYPE',
    'SVLEN',
    'CHR2'
]
# INFO fields from Sniffles2 VCF that will be used for call posterior probability estimation
sniffles_info_pred_cols = [
    'SUPPORT',
    'VAF',
    'STDEV_POS',
    'STDEV_LEN',
    'PRECISE',
    'IMPRECISE',
    'NM',
    'SVLEN'
]

sniffles_params = default_params | {
    'source': 'sniffles',
    'filter_cols': sniffles_filters,
    'info_cols_mandatory': sniffles_info_mandatory_cols,
    'info_cols_pred': sniffles_info_pred_cols,
    'parse_format': True,
    'format_cols': ['GT', 'GQ', 'DR', 'DV'],
    'svlen_source': 'svlen',
    'bps_path': 'sniffles.bps.csv',
    'override_bps_path': False,
    'bnd2_pos': 'svlen'
}


sniffles_H_params = sniffles_params | {
    'bps_path': 'sv_calling/sniffles/svensemble2/sniffles_H.bps.csv',
    'override_bps_path': True,
}

caller_param_dict = {
    'gnomad': gnomad_params,
    'gridss': gridss_params,
    'manta_candidate': manta_candidate_params,
    'manta_diploid': manta_diploid_params,
    'delly': delly_params,
    'sniffles': sniffles_params,
    'sniffles_H': sniffles_H_params,
    'svensemble2': svensemble2_params,
    'kosugi': kosugi_params,
    'parikh': parikh_params,
    'chm13_val': chm13_params,
    'hg002_val': hg002_params,
    'huddleston': huddleston_params,
    'hg00514_val': hg00514_params,
    'survivor': survivor_params,
    'gridss_H': gridss_H_params
}

caller_alt_priority = [
    'gridss',
    'manta_candidate',
    'manta_diploid',
    'delly'
]