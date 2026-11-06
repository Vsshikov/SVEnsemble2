"""
Naming scheme for the DataFrame columns and IDs SVEnsemble2 derives from VCF records.

Parsed VCF fields become <PREFIX><field> columns (INFO_SVLEN, SAMPLE_GT).
Breakpoint tables hold one column per breakend, <column>_<breakend label>
(POS_bnd1), and confidence intervals are split per side, <column>_<side label>
(INFO_CIPOS_l, INFO_CIPOS_l_bnd1).

*_PREFIX and *_SUFFIX values include their underscore; labels do not.
This module imports nothing from the package, so any module can import it.
"""

# Prefixes of columns parsed from VCF fields or computed from them
FILTER_PREFIX = 'FILTER_'
INFO_PREFIX = 'INFO_'
SAMPLE_PREFIX = 'SAMPLE_'
ANNOT_PREFIX = 'ANNOT_'
CUSTOM_PREFIX = 'CUSTOM_'
TOPOLOGY_PREFIX = 'svensembletopology_'

# VCF sample each parsed row was read from
SAMPLE_ID_COL = 'FORMAT_SAMPLE_ID'

# Breakend labels
BND1 = 'bnd1'
BND2 = 'bnd2'
BNDS = (BND1, BND2)

# Confidence interval side labels
CI_LEFT = 'l'
CI_RIGHT = 'r'
CI_SIDES = (CI_LEFT, CI_RIGHT)

# Event ID suffixes of the two events an inversion is split into
INV_EVENT1_SUFFIX = '_e1'
INV_EVENT2_SUFFIX = '_e2'
INV_EVENT_SUFFIXES = (INV_EVENT1_SUFFIX, INV_EVENT2_SUFFIX)

# Suffix of model prediction columns, {training_set}_{model}_pred
PRED_SUFFIX = '_pred'
