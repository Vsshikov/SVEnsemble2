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

base_cols = ['CHROM', 'POS', 'ID', 'REF', 'ALT', 'QUAL']

# caller-specific parameters are defined as dictionaries
# caller_params = {
#     'base_cols': list of base vcf columns in the caller. Follow from VCF documentation,
#     'filter_cols': filter columns used for pre-processing/prediction,
#     'info_cols_mandatory': list of INFO columns that must be present in the VCF many caller-specific functions will not work without them,
#     'info_cols_pred': list of INFO columns that will be used for call posterior probability estimation
# }

# Default parameters for all callers
default_filter_cols = []
default_info_mandatory_cols = []
default_info_pred_cols = []
default_params = {
    'base_cols': base_cols,
    'filter_cols': default_filter_cols,
    'info_cols_mandatory': default_info_mandatory_cols,
    'info_cols_pred': default_info_pred_cols
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
    'REF',
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
gridss_params = {
    'base_cols': base_cols,
    'filter_cols': gridss_filters,
    'info_cols_mandatory': gridss_info_mandatory_cols,
    'info_cols_pred': gridss_info_pred_cols
}

gnomad_params = {
    'base_cols': ['CHROM', 'POS', 'ID', 'REF', 'ALT', 'QUAL'],
    'filter_cols': ['PASS'],
    'info_cols_mandatory': ['EVIDENCE', 'SVTYPE', 'CHR2', 'POS2', 'END', 'END2', 'CPX_INTERVALS', 'SVLEN'],
    'info_cols_pred': []
}

gnomad_supported_svtypes = ['DEL', 'DUP', 'INV', 'INS']

gnomad_method_cipos = {
    'SR': 10,
    'PE': 100,
    'RD': 500
}

manta_params = {
    'base_cols': base_cols,
    'filter_cols': ['PASS'],
    'info_cols_mandatory': ['SVTYPE', 'SVLEN', 'POS2', 'CIPOS', 'CIEND', 'MATEID', 'EVENT'],
    'info_cols_pred': ['IMPRECISE', 'HOMLEN', 'BND_DEPTH']
}

insert_seq_source = {
    'gnomad': 'svlen',
    'gridss': 'alt',
    'manta': 'svlen'
}

caller_param_dict = {
    'gnomad': gnomad_params,
    'gridss': gridss_params,
    'manta': manta_params
}