"""
This module contains parameters to use for loading and processing VCF files
from specific callsets supported by SVEnsemble2. A callset is any named,
parseable source of SV calls: an actual caller's output (below), a reference/
ground-truth set (e.g. gnomAD), or a hand-curated validation set.
As well as parameters for PU learning with those sets.

Added callers:
- gridss
- manta
- delly
- sniffles
- cutesv
- svaba
- severus
- smoove

The file also contains parameters for working with VCF files from reference SV sets.
Currently, only gnomAD is supported.
"""

import numpy as np

default_reference_set = 'gnomad'

default_sv_types = ['DEL', 'INS', 'DUP', 'INV', 'TRA', 'OTHER']
default_sv_lens = (50, np.inf)

svtype_cols = ['INFO_SVTYPE', 'INFO_INFERRED_SVTYPE']

base_cols = ['CHROM', 'POS', 'ID', 'REF', 'ALT', 'QUAL']

# Callset parameter entries are defined as dictionaries with the following keys:
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

# Default parameters shared by all callsets
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


default_training_set = 'delly+manta_diploid'
training_set_presets = {
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
    'delly+manta_diploid': ['delly', 'manta_diploid'],
    'sniffles': ['sniffles'],
    'sniffles+gridss': ['sniffles', 'gridss'],
    'sniffles+manta_candidate': ['sniffles', 'manta_candidate'],
    'sniffles+manta_diploid': ['sniffles', 'manta_diploid'],
    'sniffles+delly': ['sniffles', 'delly'],
    'sniffles+manta_candidate+delly': ['sniffles', 'manta_candidate', 'delly'],
    'sniffles+manta_diploid+delly': ['sniffles', 'manta_diploid', 'delly'],
    'sniffles+gridss+manta_candidate': ['sniffles', 'gridss', 'manta_candidate'],
    'sniffles+gridss+manta_diploid': ['sniffles', 'gridss', 'manta_diploid'],
    'sniffles+gridss+manta_candidate+delly': ['sniffles', 'gridss', 'manta_candidate', 'delly'],
    'sniffles+gridss+manta_diploid+delly': ['sniffles', 'gridss', 'manta_diploid', 'delly'],
    'severus': ['severus'],
    'sniffles+severus': ['sniffles', 'severus'],
    'gridss+manta_diploid+delly+sniffles+severus': ['gridss', 'manta_diploid', 'delly', 'sniffles', 'severus'],
    'cutesv': ['cutesv'],
    'cutesv+sniffles': ['cutesv', 'sniffles'],
    'cutesv+severus': ['cutesv', 'severus'],
    'cutesv+sniffles+severus': ['cutesv', 'sniffles', 'severus'],
    'sniffles+severus+cutesv': ['sniffles', 'severus', 'cutesv'],
    'sniffles+severus+cutesv+gridss+manta_diploid+delly': ['sniffles', 'severus', 'cutesv', 'gridss', 'manta_diploid', 'delly'],
    'svaba': ['svaba'],
    'svaba+gridss': ['svaba', 'gridss'],
    'svaba+manta_diploid': ['svaba', 'manta_diploid'],
    'svaba+delly': ['svaba', 'delly'],
    'svaba+manta_diploid+delly': ['svaba', 'manta_diploid', 'delly'],
    'gridss+manta_diploid+delly+svaba': ['gridss', 'manta_diploid', 'delly', 'svaba'],
    'sniffles+gridss+manta_diploid+delly+svaba': ['sniffles', 'gridss', 'manta_diploid', 'delly', 'svaba'],
    'sniffles+severus+cutesv+gridss+manta_diploid+delly+svaba': ['sniffles', 'severus', 'cutesv', 'gridss', 'manta_diploid', 'delly', 'svaba'],
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
    ],
    'sniffles_preset': [
        'sniffles',
        'sniffles+gridss',
        'sniffles+manta_candidate',
        'sniffles+manta_diploid',
        'sniffles+delly',
        'sniffles+manta_candidate+delly',
        'sniffles+manta_diploid+delly',
        'sniffles+gridss+manta_candidate',
        'sniffles+gridss+manta_diploid',
        'sniffles+gridss+manta_candidate+delly',
        'sniffles+gridss+manta_diploid+delly',
    ],
    'all': [training_set for training_set in training_set_presets.keys()]
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
    'info_cols_mandatory': ['EVIDENCE', 'SVTYPE', 'CHR2', 'POS2', 'END', 'END2', 'CPX_INTERVALS', 'SVLEN', 'ALGORITHMS', 'AF', 'BOTHSIDES_SUPPORT'],
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
    'bps_path': 'gnomad.bps.csv',
}

# `filter_defaults` encodes the per-variant filter profile consumed by
# `ground_truth.preprocess_ground_truth`. Any key present here overrides the
# corresponding parameter's `None` default, and any CLI value overrides the
# `filter_defaults` value in turn. Keys must match the kwargs accepted by
# `preprocess_ground_truth` (`filters`, `evidence`, `svtypes`, `algorithms`,
# `min_af`, `max_af`, `svlens`, `bothsides_support_only`,
# `canonical_chroms_only`).
#
# `evidence=['SR', 'PE']` explicitly drops RD-only records for the strict
# gnomAD variants. RD calls have >~500 bp positional uncertainty and are
# not appropriate members of a strict PASS-quality ground-truth set. Without
# this drop, an RD-only record would flow into `gnomad_process` and be
# assigned an `INFO_CIPOS` from `method_cipos['RD']` alongside SR/PE-quality
# records - conflating quality tiers.
gnomad_filtered_params = gnomad_params | {
    'filter_defaults': {
        'filters': ['PASS'],
        'evidence': ['SR', 'PE'],
        'min_af': 0.01,
    },
    'bps_path': 'gnomad_filtered.bps.csv',
}

gnomad_pass_params = gnomad_params | {
    'filter_defaults': {
        'filters': ['PASS'],
        'evidence': ['SR', 'PE'],
    },
    'bps_path': 'gnomad_pass.bps.csv',
}

gnomad_pass_hg19_params = gnomad_pass_params

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
    'source': 'survivor_short',
    'bps_path': 'survivor_short.bps.csv',
    'bnd2_pos': 'end',
}

survivor_long_params = default_event_params | {
    'source': 'survivor_long',
    'bps_path': 'survivor_long.bps.csv',
    'bnd2_pos': 'end',
}

survivor_all_params = default_event_params | {
    'source': 'survivor_all',
    'bps_path': 'survivor_all.bps.csv',
    'bnd2_pos': 'end',
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
    'base_cols': base_cols + ['INFO_END'],
    'info_cols_mandatory': ['SVTYPE', 'SVLEN', 'CIPOS', 'CIEND', 'SAMPLE'],
    'svlen_source': 'svlen',
    'bnd2_pos': 'end'
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

hgsvc2_params = default_params | {
    'info_cols_mandatory': ['SVTYPE', 'SVLEN'],
    'parse_format': True,
    'format_cols': ['GT'],
    'svlen_source': 'svlen',
    'svlens': (50, np.inf)
}

gridss_H_params = gridss_params | {
    'bps_path': 'gridss_H.bps.csv',
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
    'CHR2',
    'COVERAGE'
]
# INFO fields from Sniffles2 VCF that will be used for call posterior probability estimation
sniffles_info_pred_cols = [
    'SUPPORT',
    'SUPPORT_INLINE',
    'SUPPORT_SA',
    'SUPPORT_LONG',
    'VAF',
    'STDEV_POS',
    'STDEV_LEN',
    'COVERAGE_1',
    'COVERAGE_2',
    'COVERAGE_3',
    'COVERAGE_4',
    'COVERAGE_5',
    'COVERAGE_VAR',
    'AC',
    'CONSENSUS_SUPPORT',
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
    'bps_path': 'sniffles_H.bps.csv',
}

# Parameters for Sniffles2 v2.0.7 (Kosugi et al. 2024 version)
# v2.0.7 has a smaller filter set and different INFO fields compared to v2.7.x
sniffles_207_filters = [
    'PASS',
    'GT',
    'SUPPORT_MIN',
    'STDEV_POS',
    'STDEV_LEN',
    'SVLEN_MIN',
    'COV_MIN',
    'COV_CHANGE',
    'STRAND',
    'NM'
]
sniffles_207_info_mandatory_cols = [
    'SVTYPE',
    'SVLEN',
    'CHR2',
    'COVERAGE'
]
sniffles_207_info_pred_cols = [
    'SUPPORT',
    'SUPPORT_LONG',
    'AF',
    'STDEV_POS',
    'STDEV_LEN',
    'COVERAGE_1',
    'COVERAGE_2',
    'COVERAGE_3',
    'COVERAGE_4',
    'COVERAGE_5',
    'PRECISE',
    'IMPRECISE',
    'NM',
    'SVLEN'
]

sniffles_207_params = default_params | {
    'source': 'sniffles_207',
    'filter_cols': sniffles_207_filters,
    'info_cols_mandatory': sniffles_207_info_mandatory_cols,
    'info_cols_pred': sniffles_207_info_pred_cols,
    'parse_format': True,
    'format_cols': ['GT', 'GQ', 'DR', 'DV'],
    'svlen_source': 'svlen',
    'bps_path': 'sniffles_207.bps.csv',
    'override_bps_path': False,
    'bnd2_pos': 'svlen'
}

sniffles_207_H_params = sniffles_207_params | {
    'bps_path': 'sniffles_207_H.bps.csv',
}

# Parameters for Severus (long-read SV caller)
severus_filters = [
    'PASS',
    'FAIL_LOWSUPP',
    'FAIL_MAP_CONS',
    'FAIL_CONN_CONS',
    'FAIL_LOWCOV_OTHER',
]
severus_info_mandatory_cols = [
    'SVTYPE',
    'SVLEN',
    'MATE_ID',
]
severus_info_pred_cols = [
    'MAPQ',
    'PRECISE',
    'IMPRECISE',
    'INSLEN',
    'SVLEN',
]

severus_params = default_params | {
    'source': 'severus',
    'base_cols': base_cols + ['INFO_END'],
    'filter_cols': severus_filters,
    'info_cols_mandatory': severus_info_mandatory_cols,
    'info_cols_pred': severus_info_pred_cols,
    'parse_format': True,
    'format_cols': ['GT', 'DR', 'DV', 'VAF', 'hVAF'],
    'svlen_source': 'svlen',
    'bps_path': 'severus.bps.csv',
    'bnd2_pos': 'end',
}

severus_H_params = severus_params | {
    'bps_path': 'severus_H.bps.csv',
}

# Parameters for CuteSV (long-read SV caller)
cutesv_filters = [
    'PASS',
    'q5',
]
cutesv_info_mandatory_cols = [
    'SVTYPE',
    'SVLEN',
    'CHR2',
    'CIPOS',
    'CILEN',
]
cutesv_info_pred_cols = [
    'RE',
    'AF',
    'PRECISE',
    'IMPRECISE',
    'SVLEN',
]

cutesv_params = default_params | {
    'source': 'cutesv',
    'base_cols': base_cols + ['INFO_END'],
    'filter_cols': cutesv_filters,
    'info_cols_mandatory': cutesv_info_mandatory_cols,
    'info_cols_pred': cutesv_info_pred_cols,
    'parse_format': True,
    'format_cols': ['GT', 'GQ', 'DR', 'DV'],
    'svlen_source': 'svlen',
    'bps_path': 'cutesv.bps.csv',
    'bnd2_pos': 'end',
    'qual_metric': 'INFO_RE',
}

cutesv_H_params = cutesv_params | {
    'bps_path': 'cutesv_H.bps.csv',
}

# Parameters for SvABA (short-read SV caller, local assembly-based)
svaba_filters = [
    'PASS',
    'LOWSPANDSCRD',
    'MULTIMATCH',
    'TOOSHORT',
    'LOWAS',
    'LOWICSUPPORT',
    'LOCALMATCH',
    'WEAKSUPPORTHIREP',
    'LOWQINVERSION',
    'HIGHHOMOLOGY',
    'NOLOCAL',
    'LOWMAPQDISC',
    'NODISC',
    'COMPETEDISC',
    'LOWMATCHLEN',
    'LOWSPAN',
    'WEAKDISC',
    'LOWSPLITSMALL',
    'DUPREADS',
    'LOWMAPQ',
    'SIMPLESEQUENCE',
    'LOWSUPPORT',
    'SINGLEBX',
]
svaba_info_mandatory_cols = [
    'MATEID',
    'SVTYPE',
    'HOMSEQ',
]
svaba_info_pred_cols = [
    'DISC_MAPQ',
    'EVDNC',
    'NUMPARTS',
    'MAPQ',
    'SPAN',
    'NM',
    'MATENM',
    'IMPRECISE',
    'HOMLEN',
    'SUBN',
    'SECONDARY',
]

svaba_params = default_params | {
    'source': 'svaba',
    'filter_cols': svaba_filters,
    'info_cols_mandatory': svaba_info_mandatory_cols,
    'info_cols_pred': svaba_info_pred_cols,
    'parse_format': True,
    'format_cols': ['GT', 'GQ', 'AD', 'DP', 'SR', 'DR', 'LO', 'SL', 'LR', 'PL'],
    'svlen_source': 'alt',
    'bps_path': 'svaba.bps.csv',
    'bnd2_pos': None,
}

svaba_H_params = svaba_params | {
    'bps_path': 'svaba_H.bps.csv',
}

# Parameters for SMOOVE/LUMPY
smoove_filters = []

smoove_info_mandatory_cols = [
    'SVTYPE',
    'SVLEN',
    'MATEID',
    'CIPOS',
    'CIEND',
    'EVENT'
]

smoove_info_pred_cols = [
    'SVLEN',
    'IMPRECISE',
    'SU',
    'PE',
    'SR',
]

smoove_params = default_params | {
    'source': 'smoove',
    'base_cols': base_cols + ['INFO_END'],
    'filter_cols': smoove_filters,
    'info_cols_mandatory': smoove_info_mandatory_cols,
    'info_cols_pred': smoove_info_pred_cols,
    'parse_format': False,
    'svlen_source': 'svlen',
    'bps_path': 'smoove.bps.csv',
    'bnd2_pos': 'end',
    # QUAL is '.' on every smoove/LUMPY record; SU (total supporting evidence)
    # is the usable per-call quality signal.
    'qual_metric': 'INFO_SU',
}

class CallsetParamDict(dict):
    """dict of callset name -> params that transparently aliases any key bearing
    a configured suffix onto an existing entry.

    Use this for callset families whose names are generated dynamically and so
    cannot be enumerated ahead of time, but which share the parsing params of a
    statically-defined callset. Every key ending in ``_ALIAS_SUFFIX`` resolves to
    the params stored under ``_ALIAS_TARGET``. Resolution is virtual: alias keys
    are never stored, so ``.keys()``/iteration (and any list derived from them,
    e.g. ``callset_alt_priority``) only ever see the real, statically-defined
    callsets.

    All access patterns used across the package are covered: subscript
    (``d[k]`` via ``__missing__``), ``.get()``, ``in`` membership
    (``__contains__``), and ``.keys()`` enumeration. The alias only fires when
    ``_ALIAS_TARGET`` actually exists; any other unknown key raises ``KeyError``
    / returns the ``.get()`` default as usual.

    Current alias: long-read consensus callsets ('{sample}_long_read_consensus')
    reuse SURVIVOR's params since their SVLEN encoding is identical.
    """

    _ALIAS_SUFFIX = '_long_read_consensus'
    _ALIAS_TARGET = 'survivor_short'

    def _alias_target(self, key):
        if (
            isinstance(key, str)
            and key.endswith(self._ALIAS_SUFFIX)
            and super().__contains__(self._ALIAS_TARGET)
        ):
            return self._ALIAS_TARGET
        return None

    def __missing__(self, key):
        target = self._alias_target(key)
        if target is not None:
            return super().__getitem__(target)
        raise KeyError(key)

    def __contains__(self, key):
        return super().__contains__(key) or self._alias_target(key) is not None

    def get(self, key, default=None):
        if super().__contains__(key):
            return super().__getitem__(key)
        target = self._alias_target(key)
        if target is not None:
            return super().__getitem__(target)
        return default


callset_param_dict = CallsetParamDict({
    'gnomad': gnomad_params,
    'gnomad_filtered': gnomad_filtered_params,
    'gnomad_pass': gnomad_pass_params,
    'gnomad_pass_hg19': gnomad_pass_hg19_params,
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
    'hgsvc2': hgsvc2_params,
    'survivor_short': survivor_params,
    'survivor_long': survivor_long_params,
    'survivor_all': survivor_all_params,
    'gridss_H': gridss_H_params,
    'sniffles_207': sniffles_207_params,
    'sniffles_207_H': sniffles_207_H_params,
    'severus': severus_params,
    'severus_H': severus_H_params,
    'cutesv': cutesv_params,
    'cutesv_H': cutesv_H_params,
    'svaba': svaba_params,
    'svaba_H': svaba_H_params,
    'smoove': smoove_params
})

# Maps each callset_name to (source, svensemble_dir_template, path_template).
# {s} is substituted with the sample name at call time.
# _H variants share source+paths with their base callset but reference separate params keys.
callset_path_configs: dict[str, tuple[str, str, str]] = {
    'gnomad': (
        'gnomad',
        'data/{s}/svensemble2',
        'data/{s}/{s}.vcf.gz',
    ),
    'gnomad_filtered': (
        'gnomad_filtered',
        'data/{s}/svensemble2',
        'data/gnomad/gnomad.vcf.gz',
    ),
    'gnomad_pass': (
        'gnomad_pass',
        'data/{s}/svensemble2',
        'data/gnomad/gnomad.vcf.gz',
    ),
    'gnomad_pass_hg19': (
        'gnomad_pass_hg19',
        'data/{s}/svensemble2',
        'data/gnomad/gnomad.vcf.gz',
    ),
    'gridss': (
        'gridss',
        'data/{s}/sv_calling/gridss/svensemble2',
        'data/{s}/sv_calling/gridss/out/{s}.gridss.vcf',
    ),
    'gridss_H': (
        'gridss',
        'data/{s}/sv_calling/gridss/svensemble2',
        'data/{s}/sv_calling/gridss/out/{s}.gridss.vcf',
    ),
    'manta_candidate': (
        'manta_candidate',
        'data/{s}/sv_calling/manta/svensemble2',
        'data/{s}/sv_calling/manta/results/variants/candidateSV.vcf.gz',
    ),
    'manta_diploid': (
        'manta_diploid',
        'data/{s}/sv_calling/manta/svensemble2',
        'data/{s}/sv_calling/manta/results/variants/diploidSV.vcf.gz',
    ),
    'delly': (
        'delly',
        'data/{s}/sv_calling/delly/svensemble2',
        'data/{s}/sv_calling/delly/out/{s}.delly.bcf',
    ),
    'survivor_short': (
        'survivor_short',
        'data/{s}/sv_calling/survivor/svensemble2',
        'data/{s}/sv_calling/survivor/{s}.survivor_short.vcf',
    ),
    'survivor_long': (
        'survivor_long',
        'data/{s}/sv_calling/survivor/svensemble2',
        'data/{s}/sv_calling/survivor/{s}.survivor_long.vcf',
    ),
    'survivor_all': (
        'survivor_all',
        'data/{s}/sv_calling/survivor/svensemble2',
        'data/{s}/sv_calling/survivor/{s}.survivor_all.vcf',
    ),
    'sniffles': (
        'sniffles',
        'data/{s}/sv_calling/sniffles/svensemble2',
        'data/{s}/sv_calling/sniffles/out/{s}.long.sniffles.vcf',
    ),
    'sniffles_H': (
        'sniffles',
        'data/{s}/sv_calling/sniffles/svensemble2',
        'data/{s}/sv_calling/sniffles/out/{s}.long.sniffles.vcf',
    ),
    'sniffles_207': (
        'sniffles_207',
        'data/{s}/sv_calling/sniffles_207/svensemble2',
        'data/{s}/sv_calling/sniffles_207/out/{s}.long.sniffles_207.vcf',
    ),
    'sniffles_207_H': (
        'sniffles_207',
        'data/{s}/sv_calling/sniffles_207/svensemble2',
        'data/{s}/sv_calling/sniffles_207/out/{s}.long.sniffles_207.vcf',
    ),
    'severus': (
        'severus',
        'data/{s}/sv_calling/severus/svensemble2',
        'data/{s}/sv_calling/severus/all_SVs/severus_all.vcf',
    ),
    'severus_H': (
        'severus',
        'data/{s}/sv_calling/severus/svensemble2',
        'data/{s}/sv_calling/severus/all_SVs/severus_all.vcf',
    ),
    'cutesv': (
        'cutesv',
        'data/{s}/sv_calling/cutesv/svensemble2',
        'data/{s}/sv_calling/cutesv/out/{s}.long.cutesv.vcf',
    ),
    'cutesv_H': (
        'cutesv',
        'data/{s}/sv_calling/cutesv/svensemble2',
        'data/{s}/sv_calling/cutesv/out/{s}.long.cutesv.vcf',
    ),
    'svaba': (
        'svaba',
        'data/{s}/sv_calling/svaba/svensemble2',
        'data/{s}/sv_calling/svaba/out/{s}.svaba.unfiltered.sv.vcf',
    ),
    'svaba_H': (
        'svaba_H',
        'data/{s}/sv_calling/svaba/svensemble2',
        'data/{s}/sv_calling/svaba/out/{s}.svaba.sv.vcf',
    ),
}

callset_alt_priority = [
    'gridss',
    'manta_candidate',
    'manta_diploid',
    'delly',
    'svaba',
    'sniffles',
    'severus',
    'cutesv',
    'survivor_short',
    'survivor_long',
    'survivor_all'
]

# Fill in the rest of the callset order with the remaining callsets
callset_alt_priority.extend([callset for callset in callset_param_dict.keys() if callset not in callset_alt_priority])