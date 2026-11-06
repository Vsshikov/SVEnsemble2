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

DEFAULT_REFERENCE_SET = 'gnomad'

DEFAULT_SV_TYPES = ['DEL', 'INS', 'DUP', 'INV', 'TRA', 'OTHER']
DEFAULT_SV_LENS = (50, np.inf)

SVTYPE_COLS = ['INFO_SVTYPE', 'INFO_INFERRED_SVTYPE']

BASE_COLS = ['CHROM', 'POS', 'ID', 'REF', 'ALT', 'QUAL']

# Callset parameter entries are defined as dictionaries with the following keys:
# - 'base_cols': List[str] - Standard VCF columns to parse (e.g., CHROM, POS, ID, REF, ALT, QUAL)
# - 'filter_cols': List[str] - FILTER fields to extract for pre-processing or prediction
# - 'info_cols_mandatory': List[str] - INFO fields required for correct parsing; absence may break downstream logic
# - 'info_cols_pred': List[str] - INFO fields used for call posterior probability estimation
# - 'format_cols': List[str] - FORMAT fields to extract for each sample (every VCF is parsed per sample)
# - 'annotator_cols': Dict[str, Dict[str, Tuple[str]]] - optional annotators (e.g. duphold) keyed by
#   name, each with 'format'/'info' field names. Parsed into ANNOT_<field> columns only when the VCF
#   header defines all of an annotator's fields. Never used as features
#   directly; expose them through custom_cols_pred.
# Example:
# GRIDSS_PARAMS = {
#     'base_cols': BASE_COLS,
#     'filter_cols': GRIDSS_FILTERS,
#     'info_cols_mandatory': GRIDSS_INFO_MANDATORY_COLS,
#     'info_cols_pred': GRIDSS_INFO_PRED_COLS,
#     'format_cols': []
# }

# Default parameters shared by all callsets
DEFAULT_FILTER_COLS = []
DEFAULT_INFO_MANDATORY_COLS = []
DEFAULT_INFO_PRED_COLS = []
DEFAULT_FORMAT_COLS = []
DEFAULT_CUSTOM_COLS_PRED = []
ANNOTATOR_COLS = {
    'duphold': {'format': ('DHFC', 'DHBFC', 'DHFFC'), 'info': ('GCF',)},
}
DEFAULT_PARAMS = {
    'base_cols': BASE_COLS,
    'filter_cols': DEFAULT_FILTER_COLS,
    'info_cols_mandatory': DEFAULT_INFO_MANDATORY_COLS,
    'info_cols_pred': DEFAULT_INFO_PRED_COLS,
    'custom_cols_pred': DEFAULT_CUSTOM_COLS_PRED,
    'format_cols': DEFAULT_FORMAT_COLS,
    'is_reference': False,
    'qual_metric': 'QUAL',
    'svlen_source': 'svlen',
    'bnd2_pos': 'svlen',
    'svlens': DEFAULT_SV_LENS,
    'annotator_cols': ANNOTATOR_COLS,
}
# Default params for evnet notation in VCF
DEFAULT_EVENT_PARAMS = DEFAULT_PARAMS | {
    'base_cols': BASE_COLS + ['INFO_END'],
    'info_cols_mandatory': ['CIEND', 'CIPOS', 'CHR2', 'SVLEN', 'SVTYPE']
}


DEFAULT_TRAINING_SET = 'delly+manta_diploid'
TRAINING_SET_PRESETS = {
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


# Parameters for GRIDSS
# Filter entries expected from GRIDSS
GRIDSS_FILTERS = [
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
GRIDSS_INFO_MANDATORY_COLS = [
    'MATEID',
    'EVENT',
    'CIPOS'
]
# INFO fields from GRIDSS VCF that will be used for call posterior probability estimation
GRIDSS_INFO_PRED_COLS = [
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
GRIDSS_PARAMS = DEFAULT_PARAMS | {
    'source': 'gridss',
    'filter_cols': GRIDSS_FILTERS,
    'info_cols_mandatory': GRIDSS_INFO_MANDATORY_COLS,
    'info_cols_pred': GRIDSS_INFO_PRED_COLS,
    'svlen_source': 'alt',
    'bnd2_pos': None
}

GNOMAD_PARAMS = DEFAULT_PARAMS | {
    'source': 'gnomad',
    'filter_cols': ['PASS'],
    'info_cols_mandatory': ['EVIDENCE', 'SVTYPE', 'CHR2', 'POS2', 'END', 'END2', 'CPX_INTERVALS', 'SVLEN', 'ALGORITHMS', 'AF', 'BOTHSIDES_SUPPORT'],
    'info_cols_pred': [],
    'svlen_source': 'svlen',
    'supported_svtypes': ['DEL', 'DUP', 'INV', 'INS'],
    'method_cipos': {
        'SR': 10,
        'PE': 100,
        'RD': 500
    },
    'is_reference': True,
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
GNOMAD_FILTERED_PARAMS = GNOMAD_PARAMS | {
    'filter_defaults': {
        'filters': ['PASS'],
        'evidence': ['SR', 'PE'],
        'min_af': 0.01,
    },
}

GNOMAD_PASS_PARAMS = GNOMAD_PARAMS | {
    'filter_defaults': {
        'filters': ['PASS'],
        'evidence': ['SR', 'PE'],
    },
}

GNOMAD_PASS_HG19_PARAMS = GNOMAD_PASS_PARAMS

MANTA_CANDIDATE_PARAMS = DEFAULT_PARAMS | {
    'source': 'manta_candidate',
    'base_cols': BASE_COLS + ['INFO_END'],
    'filter_cols': ['PASS'],
    'info_cols_mandatory': ['SVTYPE', 'SVLEN', 'POS2', 'CIPOS', 'CIEND', 'MATEID', 'EVENT'],
    'info_cols_pred': ['IMPRECISE', 'HOMLEN', 'BND_DEPTH', 'SVLEN', 'PAIR_COUNT', 'UPSTREAM_PAIR_COUNT', 'DOWNSTREAM_PAIR_COUNT', 'BND_PAIR_COUNT'],
    'svlen_source': 'svlen',
    'qual_metric': 'INFO_TOTAL_PAIR_COUNT',
    'bnd2_pos': 'end'
}

MANTA_DIPLOID_PARAMS = DEFAULT_PARAMS | {
    'source': 'manta_diploid',
    'base_cols': BASE_COLS + ['INFO_END'],
    'filter_cols': ['PASS'],
    'info_cols_mandatory': ['SVTYPE', 'SVLEN', 'POS2', 'CIPOS', 'CIEND', 'MATEID', 'EVENT'],
    'info_cols_pred': ['IMPRECISE', 'HOMLEN', 'BND_DEPTH', 'SVLEN'],
    'format_cols': ['FT', 'GQ', 'PR', 'SR'],
    'svlen_source': 'svlen',
    'bnd2_pos': 'end'
}

DELLY_PARAMS = DEFAULT_PARAMS | {
    'source': 'delly',
    'base_cols': BASE_COLS + ['INFO_END'],
    'filter_cols': ['PASS', 'LowQual'],
    'info_cols_mandatory': ['SVTYPE', 'SVLEN', 'CHR2', 'POS2', 'CIPOS', 'CIEND'],
    'info_cols_pred': ['PE', 'MAPQ', 'SRMAPQ', 'SR', 'SRQ', 'CE', 'CT', 'SVLEN', 'IMPRECISE', 'SVTYPE', 'SVMETHOD', 'HOMLEN'],
    'format_cols': ['GT', 'GQ', 'RC', 'RCL', 'RCR', 'RDCN', 'DR', 'DV', 'RR', 'RV'],
    'svlen_source': 'svlen',
    'bnd2_pos': 'end'
}

SURVIVOR_PARAMS = DEFAULT_EVENT_PARAMS | {
    'source': 'survivor_short',
    'bnd2_pos': 'end',
}

SURVIVOR_LONG_PARAMS = DEFAULT_EVENT_PARAMS | {
    'source': 'survivor_long',
    'bnd2_pos': 'end',
}

SURVIVOR_ALL_PARAMS = DEFAULT_EVENT_PARAMS | {
    'source': 'survivor_all',
    'bnd2_pos': 'end',
}

SVENSEMBLE2_PARAMS = DEFAULT_PARAMS | {
    'svlen_source': 'svensemble2_col'
}

KOSUGI_PARAMS = DEFAULT_PARAMS | {
    'svlen_source': 'svlen',
    'svlens': (50, np.inf)
}

PARIKH_PARAMS = DEFAULT_PARAMS | {
    'svlen_source': 'svlen',
    'svlens': (50, np.inf)
}

CHM13_PARAMS = DEFAULT_PARAMS | {
    'svlen_source': 'svlen',
    'bnd2_pos': 'end'
}

HG002_PARAMS = DEFAULT_PARAMS | {
    'info_cols_mandatory': ['SVTYPE', 'SVLEN'],
    'svlen_source': 'svlen'
}

HG00514_PARAMS = DEFAULT_PARAMS | {
    'base_cols': BASE_COLS + ['INFO_END'],
    'info_cols_mandatory': ['SVTYPE', 'SVLEN', 'CIPOS', 'CIEND', 'SAMPLE'],
    'svlen_source': 'svlen',
    'bnd2_pos': 'end'
}

HUDDLESTON_PARAMS = DEFAULT_PARAMS | {
    'base_cols': BASE_COLS + ['INFO_END'],
    'info_cols_mandatory': ['SVTYPE', 'SVLEN'],
    'svlen_source': 'svlen',
    'svlens': (50, np.inf),
    'svtype_mapping': {
        'deletion': 'DEL',
        'insertion': 'INS',
        'inversion': 'INV'
    }
}

HGSVC2_PARAMS = DEFAULT_PARAMS | {
    'info_cols_mandatory': ['SVTYPE', 'SVLEN'],
    'format_cols': ['GT'],
    'svlen_source': 'svlen',
    'svlens': (50, np.inf)
}

GRIDSS_H_PARAMS = GRIDSS_PARAMS.copy()

# Parameters for Sniffles2 (long-read SV caller)
# Filter entries expected from Sniffles2
SNIFFLES_FILTERS = [
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
SNIFFLES_INFO_MANDATORY_COLS = [
    'SVTYPE',
    'SVLEN',
    'CHR2',
    'COVERAGE'
]
# INFO fields from Sniffles2 VCF that will be used for call posterior probability estimation
SNIFFLES_INFO_PRED_COLS = [
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

SNIFFLES_PARAMS = DEFAULT_PARAMS | {
    'source': 'sniffles',
    'filter_cols': SNIFFLES_FILTERS,
    'info_cols_mandatory': SNIFFLES_INFO_MANDATORY_COLS,
    'info_cols_pred': SNIFFLES_INFO_PRED_COLS,
    'format_cols': ['GT', 'GQ', 'DR', 'DV'],
    'svlen_source': 'svlen',
    'bnd2_pos': 'svlen'
}


SNIFFLES_H_PARAMS = SNIFFLES_PARAMS.copy()

# Parameters for Sniffles2 v2.0.7 (Kosugi et al. 2024 version)
# v2.0.7 has a smaller filter set and different INFO fields compared to v2.7.x
SNIFFLES_207_FILTERS = [
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
SNIFFLES_207_INFO_MANDATORY_COLS = [
    'SVTYPE',
    'SVLEN',
    'CHR2',
    'COVERAGE'
]
SNIFFLES_207_INFO_PRED_COLS = [
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

SNIFFLES_207_PARAMS = DEFAULT_PARAMS | {
    'source': 'sniffles_207',
    'filter_cols': SNIFFLES_207_FILTERS,
    'info_cols_mandatory': SNIFFLES_207_INFO_MANDATORY_COLS,
    'info_cols_pred': SNIFFLES_207_INFO_PRED_COLS,
    'format_cols': ['GT', 'GQ', 'DR', 'DV'],
    'svlen_source': 'svlen',
    'bnd2_pos': 'svlen'
}

SNIFFLES_207_H_PARAMS = SNIFFLES_207_PARAMS.copy()

# Parameters for Severus (long-read SV caller)
SEVERUS_FILTERS = [
    'PASS',
    'FAIL_LOWSUPP',
    'FAIL_MAP_CONS',
    'FAIL_CONN_CONS',
    'FAIL_LOWCOV_OTHER',
]
SEVERUS_INFO_MANDATORY_COLS = [
    'SVTYPE',
    'SVLEN',
    'MATE_ID',
]
SEVERUS_INFO_PRED_COLS = [
    'MAPQ',
    'PRECISE',
    'IMPRECISE',
    'INSLEN',
    'SVLEN',
]

SEVERUS_PARAMS = DEFAULT_PARAMS | {
    'source': 'severus',
    'base_cols': BASE_COLS + ['INFO_END'],
    'filter_cols': SEVERUS_FILTERS,
    'info_cols_mandatory': SEVERUS_INFO_MANDATORY_COLS,
    'info_cols_pred': SEVERUS_INFO_PRED_COLS,
    'format_cols': ['GT', 'DR', 'DV', 'VAF', 'hVAF'],
    'svlen_source': 'svlen',
    'bnd2_pos': 'end',
}

SEVERUS_H_PARAMS = SEVERUS_PARAMS.copy()

# Parameters for CuteSV (long-read SV caller)
CUTESV_FILTERS = [
    'PASS',
    'q5',
]
CUTESV_INFO_MANDATORY_COLS = [
    'SVTYPE',
    'SVLEN',
    'CHR2',
    'CIPOS',
    'CILEN',
]
CUTESV_INFO_PRED_COLS = [
    'RE',
    'AF',
    'PRECISE',
    'IMPRECISE',
    'SVLEN',
]

CUTESV_PARAMS = DEFAULT_PARAMS | {
    'source': 'cutesv',
    'base_cols': BASE_COLS + ['INFO_END'],
    'filter_cols': CUTESV_FILTERS,
    'info_cols_mandatory': CUTESV_INFO_MANDATORY_COLS,
    'info_cols_pred': CUTESV_INFO_PRED_COLS,
    'format_cols': ['GT', 'GQ', 'DR', 'DV'],
    'svlen_source': 'svlen',
    'bnd2_pos': 'end',
    'qual_metric': 'INFO_RE',
}

CUTESV_H_PARAMS = CUTESV_PARAMS.copy()

# Parameters for SvABA (short-read SV caller, local assembly-based)
SVABA_FILTERS = [
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
SVABA_INFO_MANDATORY_COLS = [
    'MATEID',
    'SVTYPE',
    'HOMSEQ',
]
SVABA_INFO_PRED_COLS = [
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

SVABA_PARAMS = DEFAULT_PARAMS | {
    'source': 'svaba',
    'filter_cols': SVABA_FILTERS,
    'info_cols_mandatory': SVABA_INFO_MANDATORY_COLS,
    'info_cols_pred': SVABA_INFO_PRED_COLS,
    'format_cols': ['GT', 'GQ', 'AD', 'DP', 'SR', 'DR', 'LO', 'SL', 'LR', 'PL'],
    'svlen_source': 'alt',
    'bnd2_pos': None,
}

SVABA_H_PARAMS = SVABA_PARAMS.copy()

# Parameters for SMOOVE/LUMPY
SMOOVE_FILTERS = []

SMOOVE_INFO_MANDATORY_COLS = [
    'SVTYPE',
    'SVLEN',
    'MATEID',
    'CIPOS',
    'CIEND',
    'EVENT'
]

SMOOVE_INFO_PRED_COLS = [
    'SVLEN',
    'IMPRECISE',
    'SU',
    'PE',
    'SR',
]

SMOOVE_PARAMS = DEFAULT_PARAMS | {
    'source': 'smoove',
    'base_cols': BASE_COLS + ['INFO_END'],
    'filter_cols': SMOOVE_FILTERS,
    'info_cols_mandatory': SMOOVE_INFO_MANDATORY_COLS,
    'info_cols_pred': SMOOVE_INFO_PRED_COLS,
    'svlen_source': 'svlen',
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
    e.g. ``CALLSET_ALT_PRIORITY``) only ever see the real, statically-defined
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


CALLSET_PARAM_DICT = CallsetParamDict({
    'gnomad': GNOMAD_PARAMS,
    'gnomad_filtered': GNOMAD_FILTERED_PARAMS,
    'gnomad_pass': GNOMAD_PASS_PARAMS,
    'gnomad_pass_hg19': GNOMAD_PASS_HG19_PARAMS,
    'gridss': GRIDSS_PARAMS,
    'manta_candidate': MANTA_CANDIDATE_PARAMS,
    'manta_diploid': MANTA_DIPLOID_PARAMS,
    'delly': DELLY_PARAMS,
    'sniffles': SNIFFLES_PARAMS,
    'sniffles_H': SNIFFLES_H_PARAMS,
    'svensemble2': SVENSEMBLE2_PARAMS,
    'kosugi': KOSUGI_PARAMS,
    'parikh': PARIKH_PARAMS,
    'chm13_val': CHM13_PARAMS,
    'hg002_val': HG002_PARAMS,
    'huddleston': HUDDLESTON_PARAMS,
    'hg00514_val': HG00514_PARAMS,
    'hgsvc2': HGSVC2_PARAMS,
    'survivor_short': SURVIVOR_PARAMS,
    'survivor_long': SURVIVOR_LONG_PARAMS,
    'survivor_all': SURVIVOR_ALL_PARAMS,
    'gridss_H': GRIDSS_H_PARAMS,
    'sniffles_207': SNIFFLES_207_PARAMS,
    'sniffles_207_H': SNIFFLES_207_H_PARAMS,
    'severus': SEVERUS_PARAMS,
    'severus_H': SEVERUS_H_PARAMS,
    'cutesv': CUTESV_PARAMS,
    'cutesv_H': CUTESV_H_PARAMS,
    'svaba': SVABA_PARAMS,
    'svaba_H': SVABA_H_PARAMS,
    'smoove': SMOOVE_PARAMS
})

CALLSET_ALT_PRIORITY = [
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
CALLSET_ALT_PRIORITY.extend([callset for callset in CALLSET_PARAM_DICT.keys() if callset not in CALLSET_ALT_PRIORITY])