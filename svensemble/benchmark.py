#! /usr/bin/env python3


import matplotlib as mpl
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve
import numpy as np
import pickle
from typing import Dict, Any, List, Tuple
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
import os
import sys
import pysam
from adjustText import adjust_text
from matplotlib.legend_handler import HandlerLine2D
from matplotlib.text import Text

from . import breakends as bnds
from . import events as evts
from . import utils as sv_utils
from . import _caller_params as caller_params
from . import _sample_params as sample_params
from . import vcfparse as vcfp
from . import chunk_match as chunk_match
from . import breakpoints as bps
from . import matched_breakpoints as matchedbps

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42




def load_truth_variants(ref_path):
    """Load reference variants from Kosugi study from BED file

    Args:
        ref_path: Path to BED file with reference variants

    Returns:
        DataFrame with columns:
        - Chromosome, Start, End: Variant coordinates
        - ALT: Variant type (DEL, INS, etc)
        - length: Length of variant
        - is_in_KosugiSet: Always True
        - ID: String identifier combining coordinates
    """
    ref_df = pd.read_csv(ref_path, sep='\t', names=['Chromosome', 'Start', 'End', 'ALT'])
    ref_df.loc[:, 'length'] = ref_df['End'] - ref_df['Start']
    ref_df.loc[:, 'is_in_KosugiSet'] = True
    ref_df.loc[:, 'ID'] = ref_df['Chromosome'].astype(str) + ':' + ref_df['Start'].astype(str) + '-' + ref_df['End'].astype(str)
    return ref_df

def truth_to_vcf_df(ref_df):
    """Convert truth variant set from original format to VCF-like DataFrame

    Args:
        ref_df_pr: DataFrame with reference variants in Kosugi format

    Returns:
        DataFrame with VCF-like fields including:
        - CHROM, POS: Variant position
        - INFO_END: End position
        - INFO_CHR2: Secondary chromosome
        - INFO_EVENT: Variant ID
        - INFO_SVTYPE: Variant type
        - INFO_SVLEN: Variant length
        - INFO_CIPOS_l: Left confidence interval (0)
        - INFO_CIPOS_r: Right confidence interval (0)
    """
    ref_vcf = ref_df.rename(columns={
        'Chromosome': 'CHROM',
        'Start': 'POS',
        'End': 'INFO_END'
    })

    ref_vcf.loc[:, 'INFO_CHR2'] = ref_vcf['CHROM']
    ref_vcf.loc[:, 'INFO_EVENT'] = ref_vcf['ID']
    ref_vcf.loc[:, 'INFO_SVTYPE'] = ref_vcf['ALT']
    ref_vcf.loc[:, 'INFO_SVLEN'] = ref_vcf['length']
    ref_vcf.loc[:, 'INFO_CIPOS_l'] = 0
    ref_vcf.loc[:, 'INFO_CIPOS_r'] = 0
    ref_vcf.loc[:, 'INFO_CIEND_l'] = 0
    ref_vcf.loc[:, 'INFO_CIEND_r'] = 0

    return ref_vcf

def truth_event_to_bnd(ref_vcf):
    """Convert truth event VCF to breakpoint pairs

    Args:
        ref_vcf: DataFrame with truth variants in VCF format

    Returns:
        DataFrame with breakpoint pairs including:
        - Original VCF fields split into bnd1/bnd2
        - bnd_pos_encoding: Encoding of breakpoint orientations
        - bndtype: Combined orientation encoding for the pair
    """
    ref_vcf_bnds = evts.event_to_bnd(ref_vcf)
    ref_vcf_bps = bnds.bnd_to_bp(ref_vcf_bnds, force_diff_cols=['CHROM', 'INFO_CIPOS_l', 'INFO_CIPOS_r'])
    return ref_vcf_bps



def get_plot_colors(n_callsets):
    """
    Get a list of colors for plotting based on number of callsets.
    Uses tab20 if n_callsets > 10, otherwise default color cycle (None entries).
    """
    if n_callsets > 10:
        color_map = plt.get_cmap('tab20')
        # tab20 has 20 colors; for >20 callsets colors will repeat
        colors = [color_map(i % 20) for i in range(n_callsets)]
    else:
        colors = [None] * n_callsets  # Use default color cycle
    return colors

xgfs_fancy6 = [(86, 100, 26), (192, 175, 251), (230, 161, 118), (0, 103, 138), (152, 68, 100), (94, 204, 171), (205, 205, 205)]
# Convert to RGB tuples with 0-1 floats
xgfs_fancy6 = [tuple(color / 255 for color in color_tuple) for color_tuple in xgfs_fancy6]

xgfs_fancy_ext = [(88, 55, 170), (200, 50, 50), (255, 165, 0), (0, 160, 80), (180, 40, 180), (255, 100, 50), (40, 140, 200), (200, 180, 0)]
xgfs_fancy_ext = [tuple(c / 255 for c in rgb) for rgb in xgfs_fancy_ext]

xgfs_fancy_new = [
    (140, 86, 200), (200, 80, 140), (80, 160, 120), (160, 100, 160),
    (0, 128, 160), (100, 120, 200), (180, 120, 80), (80, 140, 140),
    (120, 160, 100), (50, 130, 70), (180, 70, 70), (60, 100, 160), (140, 90, 50),
]
xgfs_fancy_new = [tuple(c / 255 for c in rgb) for rgb in xgfs_fancy_new]

_style_mappings = {}
# _style_mappings = {
#     'gridss': {
#         'line': {
#             'color': xgfs_fancy6[0],
#             'linewidth': 0.7,
#             'dashes': (2, 2),
#             'label': 'GRIDSS',
#         },
#         'text': {
#             'label': 'GR',
#         }
#     },
#     'gridss_H': {
#         'line': {
#             'color': xgfs_fancy6[0],
#             'linewidth': 1,
#             'dashes': (2, 2),
#             'label': 'GRIDSS (High Confidence Calls)',
#         },
#         'text': {
#             'label': 'GR_H',
#         }
#     },
#     'gridss_H_lazy': {
#         'line': {
#             'color': tuple([c * 1.2 for c in xgfs_fancy6[0] if c * 1.2 < 1]),
#             'linewidth': 1,
#             'dashes': (2, 2),
#             'label': 'GRIDSS (High Confidence Calls, Lazy Loading)',
#         },
#         'text': {
#             'label': 'GR_H_lazy',
#         }
#     },
#     'svensemble2_gridss': {
#         'line': {
#             'color': xgfs_fancy6[0],
#             'linewidth': 0.8,
#             'label': 'SVEnsemble2 (GRIDSS)',
#         },
#         'text': {
#             'label': 'GR',
#         }
#     },
#     'manta_diploid': {
#         'line': {
#             'color': xgfs_fancy6[3],
#             'linewidth': 0.7,
#             'dashes': (2, 2),
#             'label': 'Manta',
#         },
#         'text': {
#             'label': 'MA',
#         }
#     },
#     'svensemble2_manta_diploid': {
#         'line': {
#             'color': xgfs_fancy6[3],
#             'linewidth': 0.8,
#             'label': 'SVEnsemble2 (Manta)',
#         },
#         'text': {
#             'label': 'MA',
#         }
#     },
#     'delly': {
#         'line': {
#             'color': xgfs_fancy6[2],
#             'linewidth': 0.7,
#             'dashes': (2, 2),
#             'label': 'Delly',
#         },
#         'text': {
#             'label': 'DE',
#         }
#     },
#     'svensemble2_delly': {
#         'line': {
#             'color': xgfs_fancy6[2],
#             'linewidth': 0.8,
#             'label': 'SVEnsemble2 (Delly)',
#         },
#         'text': {
#             'label': 'DE',
#         }
#     },
#     'survivor_short': {
#         'line': {
#             'color': (0.55, 0.45, 0.95),
#             'linewidth': 1,
#             'dashes': (6, 2.5),
#             'label': 'SURVIVOR Short-Read Consensus',
#         },
#         'text': {
#             'label': 'SU-S',
#         }
#     },
#     'survivor_long': {
#         'line': {
#             'color': (0.88, 0.45, 0.72),
#             'linewidth': 1,
#             'dashes': (6, 2.5),
#             'label': 'SURVIVOR Long-Read Consensus',
#         },
#         'text': {
#             'label': 'SU-L',
#         }
#     },
#     'survivor_all': {
#         'line': {
#             'color': (0.45, 0.35, 0.65),
#             'linewidth': 1,
#             'dashes': (6, 2.5),
#             'label': 'SURVIVOR All-Caller Consensus',
#         },
#         'text': {
#             'label': 'SU-A',
#         }
#     },
#     'svensemble2_delly+manta_diploid': {
#         'line': {
#             'color': xgfs_fancy6[4],
#             'linewidth': 1,
#             'label': 'SVEnsemble2 (Delly + Manta)',
#         },
#         'text': {
#             'label': 'DE+MA',
#         }
#     },
#     'svensemble2_delly+manta_candidate': {
#         'line': {
#             'color': (0.71, 0.37, 0.49),
#             'linewidth': 1,
#             'label': 'SVEnsemble2 (Delly + Manta Candidate)',
#         },
#         'text': {
#             'label': 'DE+MC',
#         }
#     },
#     'svensemble2_gridss+manta_diploid': {
#         'line': {
#             'color': xgfs_fancy6[5],
#             'linewidth': 1,
#             'label': 'SVEnsemble2 (GRIDSS + Manta)',
#         },
#         'text': {
#             'label': 'GR+MA',
#         }
#     },
#     'svensemble2_gridss+manta_candidate': {
#         'line': {
#             'color': (0.25, 0.69, 0.57),
#             'linewidth': 1,
#             'label': 'SVEnsemble2 (GRIDSS + Manta Candidate)',
#         },
#         'text': {
#             'label': 'GR+MC',
#         }
#     },
#     'svensemble2_gridss+delly': {
#         'line': {
#             'color': (0.40, 0.60, 0.30),
#             'linewidth': 1,
#             'label': 'SVEnsemble2 (GRIDSS + Delly)',
#         },
#         'text': {
#             'label': 'GR+DE',
#         }
#     },
#     'svensemble2_gridss+manta_diploid+delly': {
#         'line': {
#             'color': xgfs_fancy6[6],
#             'linewidth': 1,
#             'label': 'SVEnsemble2 (GRIDSS + Manta + Delly)',
#         },
#         'text': {
#             'label': 'GR+MA+DE',
#         }
#     },
#     'svensemble2_gridss+manta_candidate+delly': {
#         'line': {
#             'color': (0.67, 0.67, 0.76),
#             'linewidth': 1,
#             'label': 'SVEnsemble2 (GRIDSS + Manta Candidate + Delly)',
#         },
#         'text': {
#             'label': 'GR+MC+DE',
#         }
#     },
#     'manta_candidate': {
#         'line': {
#             'color': xgfs_fancy6[3],
#             'linewidth': 0.7,
#             'dashes': (4, 2),
#             'label': 'Manta (Candidate)',
#         },
#         'text': {
#             'label': 'MC',
#         }
#     },
#     'svensemble2_manta_candidate': {
#         'line': {
#             'color': (0.16, 0.51, 0.65),
#             'linewidth': 0.8,
#             'label': 'SVEnsemble2 (Manta Candidate)',
#         },
#         'text': {
#             'label': 'MC',
#         }
#     },
#     'sniffles': {
#         'line': {
#             'color': xgfs_fancy_ext[0],
#             'linewidth': 0.7,
#             'dashes': (2, 2),
#             'label': 'Sniffles',
#         },
#         'text': {
#             'label': 'SN',
#         }
#     },
#     'sniffles_H': {
#         'line': {
#             'color': xgfs_fancy_ext[0],
#             'linewidth': 1,
#             'dashes': (2, 2),
#             'label': 'Sniffles (High Confidence Calls)',
#         },
#         'text': {
#             'label': 'SN_H',
#         }
#     },
#     'severus': {
#         'line': {
#             'color': xgfs_fancy_ext[3],
#             'linewidth': 0.7,
#             'dashes': (2, 2),
#             'label': 'Severus',
#         },
#         'text': {
#             'label': 'SV',
#         }
#     },
#     'severus_H': {
#         'line': {
#             'color': xgfs_fancy_ext[3],
#             'linewidth': 1,
#             'dashes': (2, 2),
#             'label': 'Severus (High Confidence Calls)',
#         },
#         'text': {
#             'label': 'SV_H',
#         }
#     },
#     'cutesv': {
#         'line': {
#             'color': xgfs_fancy_ext[4],
#             'linewidth': 0.7,
#             'dashes': (2, 2),
#             'label': 'CuteSV',
#         },
#         'text': {
#             'label': 'CS',
#         }
#     },
#     'cutesv_H': {
#         'line': {
#             'color': xgfs_fancy_ext[4],
#             'linewidth': 1,
#             'dashes': (2, 2),
#             'label': 'CuteSV (High Confidence Calls)',
#         },
#         'text': {
#             'label': 'CS_H',
#         }
#     },
#     'svensemble2_cutesv': {
#         'line': {
#             'color': xgfs_fancy_ext[4],
#             'linewidth': 0.8,
#             'label': 'SVEnsemble2 (CuteSV)',
#         },
#         'text': {
#             'label': 'CS',
#         }
#     },
#     'svensemble2_sniffles': {
#         'line': {
#             'color': xgfs_fancy_ext[0],
#             'linewidth': 0.8,
#             'label': 'SVEnsemble2 (Sniffles)',
#         },
#         'text': {
#             'label': 'SN',
#         }
#     },
#     'svensemble2_sniffles+gridss': {
#         'line': {
#             'color': xgfs_fancy_ext[1],
#             'linewidth': 1,
#             'label': 'SVEnsemble2 (Sniffles + GRIDSS)',
#         },
#         'text': {
#             'label': 'SN+GR',
#         }
#     },
#     'svensemble2_sniffles+manta_candidate': {
#         'line': {
#             'color': xgfs_fancy_ext[2],
#             'linewidth': 1,
#             'label': 'SVEnsemble2 (Sniffles + Manta Candidate)',
#         },
#         'text': {
#             'label': 'SN+MC',
#         }
#     },
#     'svensemble2_sniffles+manta_diploid': {
#         'line': {
#             'color': xgfs_fancy_ext[3],
#             'linewidth': 1,
#             'label': 'SVEnsemble2 (Sniffles + Manta)',
#         },
#         'text': {
#             'label': 'SN+MA',
#         }
#     },
#     'svensemble2_sniffles+delly': {
#         'line': {
#             'color': (0.58, 0.25, 0.58),
#             'linewidth': 1,
#             'label': 'SVEnsemble2 (Sniffles + Delly)',
#         },
#         'text': {
#             'label': 'SN+DE',
#         }
#     },
#     'svensemble2_sniffles+manta_candidate+delly': {
#         'line': {
#             'color': xgfs_fancy_ext[5],
#             'linewidth': 1,
#             'label': 'SVEnsemble2 (Sniffles + Manta Candidate + Delly)',
#         },
#         'text': {
#             'label': 'SN+MC+DE',
#         }
#     },
#     'svensemble2_sniffles+manta_diploid+delly': {
#         'line': {
#             'color': xgfs_fancy_ext[6],
#             'linewidth': 1,
#             'label': 'SVEnsemble2 (Sniffles + Manta + Delly)',
#         },
#         'text': {
#             'label': 'SN+MA+DE',
#         }
#     },
#     'svensemble2_sniffles+gridss+manta_diploid': {
#         'line': {
#             'color': (0.65, 0.59, 0.16),
#             'linewidth': 1,
#             'label': 'SVEnsemble2 (Sniffles + GRIDSS + Manta)',
#         },
#         'text': {
#             'label': 'SN+GR+MA',
#         }
#     },
#     'svensemble2_sniffles+gridss+manta_candidate': {
#         'line': {
#             'color': (0.55, 0.67, 0.24),
#             'linewidth': 1,
#             'label': 'SVEnsemble2 (Sniffles + GRIDSS + Manta Candidate)',
#         },
#         'text': {
#             'label': 'SN+GR+MC',
#         }
#     },
#     'svensemble2_sniffles+gridss+manta_diploid+delly': {
#         'line': {
#             'color': xgfs_fancy_ext[7],
#             'linewidth': 1,
#             'label': 'SVEnsemble2 (Sniffles + GRIDSS + Manta + Delly)',
#         },
#         'text': {
#             'label': 'SN+GR+MA+DE',
#         }
#     },
#     'svensemble2_sniffles+gridss+manta_candidate+delly': {
#         'line': {
#             'color': (0.71, 0.61, 0.31),
#             'linewidth': 1,
#             'label': 'SVEnsemble2 (Sniffles + GRIDSS + Manta Candidate + Delly)',
#         },
#         'text': {
#             'label': 'SN+GR+MC+DE',
#         }
#     },
#     'svaba': {
#         'line': {
#             'color': xgfs_fancy6[1],
#             'linewidth': 0.7,
#             'dashes': (2, 2),
#             'label': 'SvABA',
#         },
#         'text': {
#             'label': 'SB',
#         }
#     },
#     'svaba_H': {
#         'line': {
#             'color': xgfs_fancy6[1],
#             'linewidth': 1,
#             'dashes': (2, 2),
#             'label': 'SvABA (High Confidence Calls)',
#         },
#         'text': {
#             'label': 'SB_H',
#         }
#     },
#     'svensemble2_severus': {
#         'line': {
#             'color': (0.10, 0.55, 0.45),
#             'linewidth': 0.8,
#             'label': 'SVEnsemble2 (Severus)',
#         },
#         'text': {
#             'label': 'SV',
#         }
#     },
#     'svensemble2_svaba': {
#         'line': {
#             'color': xgfs_fancy6[1],
#             'linewidth': 0.8,
#             'label': 'SVEnsemble2 (SvABA)',
#         },
#         'text': {
#             'label': 'SB',
#         }
#     },
#     'svensemble2_sniffles+severus': {
#         'line': {
#             'color': xgfs_fancy_new[0],
#             'linewidth': 1,
#             'label': 'SVEnsemble2 (Sniffles + Severus)',
#         },
#         'text': {
#             'label': 'SN+SV',
#         }
#     },
#     'svensemble2_cutesv+sniffles': {
#         'line': {
#             'color': xgfs_fancy_new[1],
#             'linewidth': 1,
#             'label': 'SVEnsemble2 (CuteSV + Sniffles)',
#         },
#         'text': {
#             'label': 'CS+SN',
#         }
#     },
#     'svensemble2_cutesv+severus': {
#         'line': {
#             'color': xgfs_fancy_new[2],
#             'linewidth': 1,
#             'label': 'SVEnsemble2 (CuteSV + Severus)',
#         },
#         'text': {
#             'label': 'CS+SV',
#         }
#     },
#     'svensemble2_cutesv+sniffles+severus': {
#         'line': {
#             'color': xgfs_fancy_new[3],
#             'linewidth': 1,
#             'label': 'SVEnsemble2 (CuteSV + Sniffles + Severus)',
#         },
#         'text': {
#             'label': 'CS+SN+SV',
#         }
#     },
#     'svensemble2_sniffles+severus+cutesv': {
#         'line': {
#             'color': xgfs_fancy_new[3],
#             'linewidth': 1,
#             'label': 'SVEnsemble2 (Sniffles + Severus + CuteSV)',
#         },
#         'text': {
#             'label': 'SN+SV+CS',
#         }
#     },
#     'svensemble2_svaba+gridss': {
#         'line': {
#             'color': xgfs_fancy_new[4],
#             'linewidth': 1,
#             'label': 'SVEnsemble2 (SvABA + GRIDSS)',
#         },
#         'text': {
#             'label': 'SB+GR',
#         }
#     },
#     'svensemble2_svaba+manta_diploid': {
#         'line': {
#             'color': xgfs_fancy_new[5],
#             'linewidth': 1,
#             'label': 'SVEnsemble2 (SvABA + Manta)',
#         },
#         'text': {
#             'label': 'SB+MA',
#         }
#     },
#     'svensemble2_svaba+delly': {
#         'line': {
#             'color': xgfs_fancy_new[6],
#             'linewidth': 1,
#             'label': 'SVEnsemble2 (SvABA + Delly)',
#         },
#         'text': {
#             'label': 'SB+DE',
#         }
#     },
#     'svensemble2_svaba+manta_diploid+delly': {
#         'line': {
#             'color': xgfs_fancy_new[7],
#             'linewidth': 1,
#             'label': 'SVEnsemble2 (SvABA + Manta + Delly)',
#         },
#         'text': {
#             'label': 'SB+MA+DE',
#         }
#     },
#     'svensemble2_gridss+manta_diploid+delly+svaba': {
#         'line': {
#             'color': xgfs_fancy_new[8],
#             'linewidth': 1,
#             'label': 'SVEnsemble2 (GRIDSS + Manta + Delly + SvABA)',
#         },
#         'text': {
#             'label': 'GR+MA+DE+SB',
#         }
#     },
#     'svensemble2_gridss+manta_diploid+delly+sniffles+severus': {
#         'line': {
#             'color': xgfs_fancy_new[9],
#             'linewidth': 1,
#             'label': 'SVEnsemble2 (GRIDSS + Manta + Delly + Sniffles + Severus)',
#         },
#         'text': {
#             'label': 'GR+MA+DE+SN+SV',
#         }
#     },
#     'svensemble2_sniffles+gridss+manta_diploid+delly+svaba': {
#         'line': {
#             'color': xgfs_fancy_new[10],
#             'linewidth': 1,
#             'label': 'SVEnsemble2 (Sniffles + GRIDSS + Manta + Delly + SvABA)',
#         },
#         'text': {
#             'label': 'SN+GR+MA+DE+SB',
#         }
#     },
#     'svensemble2_sniffles+severus+cutesv+gridss+manta_diploid+delly': {
#         'line': {
#             'color': xgfs_fancy_new[11],
#             'linewidth': 1,
#             'label': 'SVEnsemble2 (6-Caller Ensemble)',
#         },
#         'text': {
#             'label': '6c',
#         }
#     },
#     'svensemble2_sniffles+severus+cutesv+gridss+manta_diploid+delly+svaba': {
#         'line': {
#             'color': xgfs_fancy_new[12],
#             'linewidth': 1,
#             'label': 'SVEnsemble2 (7-Caller Ensemble)',
#         },
#         'text': {
#             'label': '7c',
#         }
#     },
# }

_model_style_mappings = {
    'unbiased_xgb': {
        'line': {
            'color': (192 / 255, 57 / 255, 43 / 255),
            'linewidth': 1,
        }
    },
    'unbiased_d31': {
        'line': {
            'color': (39 / 255, 174 / 255, 96 / 255),
            'linewidth': 2,
            'linestyle': '-.',
        }
    },
    'unbiased_xgb_nosvlen': {
        'line': {
            'color': (231 / 255, 138 / 255, 128 / 255),
            'linewidth': 1,
            'linestyle': '--',
        }
    },
    'unbiased_d31_nosvlen': {
        'line': {
            'color': (125 / 255, 206 / 255, 160 / 255),
            'linewidth': 2,
            'linestyle': '--',
        }
    },
}

default_style_mapping = {
    'line': {
        'linewidth': 1
    },
    'text': {
        'fontfamily': 'monospace',
        'fontsize': 10,
        'color': 'white',
        'ha': 'center',
        'va': 'top',
        'bbox': {
            'boxstyle': 'round,pad=0.2'
        }
    }
}

def get_style_mapping(callset, model=None, default_style_mapping=default_style_mapping):
    """Build the merged style dict for a callset, optionally overlayed with model-specific overrides.

    Args:
        callset: Callset name used to look up the entry in ``_style_mappings``.
        model: Optional model name. If provided and present in
            ``_model_style_mappings``, that entry is recursively merged on top
            of the callset style (so it can override e.g. ``line.linewidth``
            or ``line.linestyle``). Unknown model names are silently ignored.
        default_style_mapping: Base style dict that callset/model entries are
            merged into.

    Returns:
        dict with ``line`` and ``text`` sub-dicts ready to splat into
        matplotlib calls.
    """
    _default_style_mapping = deepcopy(default_style_mapping)
    if callset in _style_mappings:
        style_mapping = sv_utils.update(_default_style_mapping, _style_mappings[callset])
    else:
        _fallback_color = plt.get_cmap('tab20')(hash(callset) % 20)
        _default_style_mapping['line']['color'] = _fallback_color
        _default_style_mapping['line']['label'] = callset
        style_mapping = _default_style_mapping
    if model is not None and model in _model_style_mappings:
        style_mapping = sv_utils.update(style_mapping, deepcopy(_model_style_mappings[model]))
        style_mapping['line']['label'] = f'{style_mapping["line"]["label"]} ({model})'
    if 'color' in style_mapping['line']:
        style_mapping['text']['bbox']['ec'] = style_mapping['line']['color']
        style_mapping['text']['bbox']['fc'] = style_mapping['line']['color']
    return style_mapping

class HandlerLineWithBadge(HandlerLine2D):
    """Legend handler that draws the standard line swatch plus a colored text badge."""

    def __init__(self, badge_label, badge_color, **kwargs):
        self.badge_label = badge_label
        self.badge_color = badge_color
        super().__init__(**kwargs)

    def create_artists(self, legend, orig_handle,
                       xdescent, ydescent, width, height, fontsize, trans):
        artists = super().create_artists(
            legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans,
        )
        badge = Text(
            -fontsize * 0.3,
            height / 2,
            self.badge_label,
            fontsize=fontsize * 0.7,
            fontfamily='monospace',
            fontweight='bold',
            color='white',
            ha='right',
            va='center',
            bbox=dict(boxstyle='round,pad=0.15', fc=self.badge_color, ec=self.badge_color),
            transform=trans,
        )
        artists.append(badge)
        return artists


def _build_badge_handler(callset, model=None):
    """Return a HandlerLineWithBadge for *callset*, or *None* if no badge text is defined.

    *model* is forwarded to ``get_style_mapping`` so model-specific overrides
    (e.g. line width / linestyle) are reflected in the legend swatch.
    """
    style = get_style_mapping(callset, model=model)
    badge_label = style['text'].get('label', '')
    if not badge_label:
        return None
    return HandlerLineWithBadge(
        badge_label=badge_label,
        badge_color=style['line'].get('color', 'gray'),
    )


# Create ordering for callsets. svensemble2 calls should always follow corresponding original callsets.
# Svensemble ensemble calls should be last, survivor calls should be just before them
# At the moment the ordering is manual
_callset_order = [
    'gridss',
    'gridss_H',
    'svensemble2_gridss',
    'manta_diploid',
    'svensemble2_manta_diploid',
    'manta_candidate',
    'svensemble2_manta_candidate',
    'delly',
    'svensemble2_delly',
    'svaba',
    'svaba_H',
    'svensemble2_svaba',
    'sniffles',
    'sniffles_H',
    'svensemble2_sniffles',
    'severus',
    'severus_H',
    'svensemble2_severus',
    'cutesv',
    'cutesv_H',
    'svensemble2_cutesv',
    'survivor_short',
    'survivor_long',
    'survivor_all',
    'svensemble2_delly+manta_diploid',
    'svensemble2_gridss+manta_diploid',
    'svensemble2_gridss+manta_diploid+delly',
    'svensemble2_sniffles+gridss',
    'svensemble2_sniffles+manta_candidate',
    'svensemble2_sniffles+manta_diploid',
    'svensemble2_sniffles+delly',
    'svensemble2_sniffles+severus',
    'svensemble2_sniffles+manta_candidate+delly',
    'svensemble2_sniffles+manta_diploid+delly',
    'svensemble2_sniffles+gridss+manta_diploid+delly',
    'svensemble2_cutesv+sniffles',
    'svensemble2_cutesv+severus',
    'svensemble2_cutesv+sniffles+severus',
    'svensemble2_sniffles+severus+cutesv',
    'svensemble2_svaba+gridss',
    'svensemble2_svaba+manta_diploid',
    'svensemble2_svaba+delly',
    'svensemble2_svaba+manta_diploid+delly',
    'svensemble2_gridss+manta_diploid+delly+svaba',
    'svensemble2_gridss+manta_diploid+delly+sniffles+severus',
    'svensemble2_sniffles+gridss+manta_diploid+delly+svaba',
    'svensemble2_sniffles+severus+cutesv+gridss+manta_diploid+delly',
    'svensemble2_sniffles+severus+cutesv+gridss+manta_diploid+delly+svaba',
]

def n_callsets(callset: str) -> int:
    """Number of underlying callers combined in a callset name."""
    return len(callset.removeprefix('svensemble2_').split('+'))

def base_name(callset: str) -> str:
    """Canonical identity: strips the svensemble2_ prefix and sorts multi-caller
    components alphabetically, so a standalone callset and its svensemble2 pair
    (and equivalent-composition training sets) collate under the same key."""
    return '+'.join(sorted(callset.removeprefix('svensemble2_').split('+')))


def callset_order_generator(callsets):
    return sorted(
        callsets,
        key=lambda c: (n_callsets(c), base_name(c), c.startswith('svensemble2_'), c),
    )


# def order_legend(plt, callsets):
#     """
#     Rearrange legend items by callset order
#     """
#     handles, labels = plt.gca().get_legend_handles_labels()
#     plt.legend([handle for handle in ])
def _ordered_callset_enumerate(callsets, callset_order):
    """
    Enumerate callsets in the order specified by callset_order
    Any callsets not in callset_order are yielded last in the order they are found in callsets
    """
    yielded_callsets = []
    for idx, callset in enumerate(callset_order):
        if callset in callsets:
            yielded_callsets.append(callset)
            yield idx, callset
    leftover_callsets = [callset for callset in callsets if callset not in yielded_callsets]
    for idx, callset in enumerate(leftover_callsets):
        yield len(callset_order) + idx, callset


def style_svensemble_lines(line, callset, model=None):
    # If svensemble2 in in the callset name, plot the curve normally
    # Otherwise, plot a dotted line instead
    style_mapping = get_style_mapping(callset, model=model)['line']
    # line.set_color(style_mapping['color'])
    # line.set_linewidth(style_mapping['linewidth'])
    # line.set_dashes(dashes)
    # line.set_dash_capstyle('round')
    line.set(**style_mapping)


def style_text(text: mpl.text.Text, callset, model=None):
    style_mapping = get_style_mapping(callset, model=model)['text']
    text.set(**style_mapping)


def plot_roc_curves(
    sample_val_dict, ax=None, use_counts=True, title=None,
    svensemble2_cutoff=None, verbose=False,
    xlim=None, ylim=None, model=None,
):
    """Plot ROC curves for multiple callsets.

    Args:
        sample_val_dict: Dict of {callset_name: DataFrame} where each
            DataFrame has a prediction score as the 2nd column and a
            boolean truth column is_in_val.
        ax: Matplotlib axis.  Created if *None*.
        use_counts: Plot as counts (True) or rates (False).
        title: Plot title.
        svensemble2_cutoff: If set, mark the operating point closest to
            this threshold on SVEnsemble2 curves.
        verbose: Print skipped callsets.
        xlim: Tuple (xmin, xmax) to set x-axis limits.
        ylim: Tuple (ymin, ymax) to set y-axis limits.
        model: Optional svensemble2 model name. When provided and present in
            ``_model_style_mappings``, the model-specific style overrides are
            applied to ``svensemble2_*`` curves only.

    Returns:
        Matplotlib axis with plotted ROC curves.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    if title is not None:
        ax.set_title(title)

    if not use_counts:
        ax.plot([0, 1], [0, 1], color='0.75', linewidth=0.8, linestyle='--', zorder=0)

    callsets = list(sample_val_dict.keys())
    callsets.sort()

    for idx, callset in _ordered_callset_enumerate(callsets, callset_order_generator(callsets)):
        callset_val_df = sample_val_dict[callset].dropna().copy()

        truth_series = callset_val_df['is_in_val']
        if sum(truth_series) == 0:
            if verbose:
                print(f'{callset} has no true positives, skipping')
            continue
        elif sum(truth_series) == len(truth_series):
            if verbose:
                print(f'{callset} has no true negatives, skipping')
            continue
        pred_series = callset_val_df.iloc[:, 1]

        fpr, tpr, thresholds = roc_curve(truth_series, pred_series, pos_label=True)

        total_tp = sum(truth_series)
        total_tn = len(truth_series) - total_tp

        if use_counts:
            tpr = tpr * total_tp
            fpr = fpr * total_tn

        style_model = model if callset.startswith('svensemble2_') else None
        ax.plot(fpr, tpr, **get_style_mapping(callset, model=style_model)['line'])

        if (svensemble2_cutoff is not None) and ('svensemble2' in callset):
            threshold_idx = np.argmin(np.abs(thresholds - svensemble2_cutoff))
            ax.plot(
                fpr[threshold_idx], tpr[threshold_idx], 'o',
                color=get_style_mapping(callset, model=style_model)['line']['color'],
            )

    if xlim is not None:
        ax.set_xlim(*xlim)
    elif not use_counts:
        ax.set_xlim(0, 1)

    if ylim is not None:
        ax.set_ylim(*ylim)
    elif not use_counts:
        ax.set_ylim(0, 1)

    ax.legend(loc='center left', bbox_transform=ax.transAxes, bbox_to_anchor=(1, 0.5), frameon=False)

    xlabel = 'False Positives' if use_counts else 'False Positive Rate'
    ylabel = 'True Positives' if use_counts else 'True Positive Rate'

    plt.xlabel(xlabel, size='large', weight=700)
    plt.ylabel(ylabel, size='large', weight=700)

    return ax


def plot_reverse_roc_curves(
    sample_val_dict, svtype=None, ax=None, use_counts=True,
    filter_pass=False, title=None, svensemble2_cutoff=None,
    verbose=False, xlim=None, ylim=None, model=None,
):
    """Plot log-scale reverse ROC curves (1/FP vs TP/FP) for multiple callsets.

    When *use_counts* is True (default) the axes show 1/FP-count and
    TP/FP-count on log scales, which makes it easier to compare callers
    with very different false-positive rates.

    Args:
        sample_val_dict: Dict of {callset_name: DataFrame} where each
            DataFrame has a prediction score as the 2nd column and a
            boolean truth column is_in_val.
        svtype: If set, restrict to rows where INFO_INFERRED_SVTYPE
            matches this value.
        ax: Matplotlib axis.  Created if *None*.
        use_counts: Plot as counts (True) or rates (False).
        filter_pass: If True, restrict to rows where FILTER_PASS is
            truthy (column must exist in the DataFrame).
        title: Plot title.
        svensemble2_cutoff: If set, mark the operating point closest to
            this threshold on SVEnsemble2 curves.
        verbose: Print skipped callsets.
        xlim: Tuple (xmin, xmax) to set x-axis limits.
        ylim: Tuple (ymin, ymax) to set y-axis limits.
        model: Optional svensemble2 model name. When provided and present in
            ``_model_style_mappings``, the model-specific style overrides are
            applied to ``svensemble2_*`` curves only.

    Returns:
        Matplotlib axis with plotted reverse ROC curves.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    if title is not None:
        ax.set_title(title)

    ax.set_xscale('log')
    ax.set_yscale('log')

    callsets = list(sample_val_dict.keys())
    callsets.sort()

    for idx, callset in _ordered_callset_enumerate(callsets, callset_order_generator(callsets)):
        callset_val_df = sample_val_dict[callset].dropna().copy()

        if filter_pass and 'FILTER_PASS' in sample_val_dict[callset].columns:
            sample_mask = callset_val_df['FILTER_PASS'].astype(bool)
        elif filter_pass:
            if verbose:
                print(f'{callset} does not have a FILTER_PASS column, ignoring filter_pass argument')
            sample_mask = np.ones(len(callset_val_df), dtype=bool)
        else:
            sample_mask = np.ones(len(callset_val_df), dtype=bool)

        if svtype is not None:
            sample_mask = sample_mask & (callset_val_df['INFO_INFERRED_SVTYPE'] == svtype)

        callset_val_df = callset_val_df.loc[sample_mask, :]
        truth_series = callset_val_df['is_in_val']
        if sum(truth_series) == 0:
            if verbose:
                print(f'{callset} has no true positives, skipping')
            continue
        elif sum(truth_series) == len(truth_series):
            if verbose:
                print(f'{callset} has no true negatives, skipping')
            continue
        pred_series = callset_val_df.iloc[:, 1]

        fpr, tpr, thresholds = roc_curve(truth_series, pred_series, pos_label=True)

        total_tp = sum(truth_series)
        total_tn = len(truth_series) - total_tp

        if use_counts:
            tp = tpr * total_tp
            fp = fpr * total_tn
            zeros_mask = (fp == 0) | (tp == 0)

            tp_no_zeros = tp[~zeros_mask]
            fp_no_zeros = fp[~zeros_mask]
            thresholds_no_zeros = thresholds[~zeros_mask]

            fp_metric = 1 / fp_no_zeros
            tp_fp_metric = tp_no_zeros / fp_no_zeros
        else:
            fp_metric = fpr
            tp_fp_metric = tpr
            thresholds_no_zeros = thresholds

        style_model = model if callset.startswith('svensemble2_') else None
        ax.plot(fp_metric, tp_fp_metric, **get_style_mapping(callset, model=style_model)['line'])

        if (svensemble2_cutoff is not None) and ('svensemble2' in callset) and len(thresholds_no_zeros):
            threshold_idx = np.argmin(np.abs(thresholds_no_zeros - svensemble2_cutoff))
            ax.plot(
                fp_metric[threshold_idx], tp_fp_metric[threshold_idx], 'o',
                color=get_style_mapping(callset, model=style_model)['line']['color'],
            )

    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)

    ax.legend(loc='center left', bbox_transform=ax.transAxes, bbox_to_anchor=(1, 0.5), frameon=False)

    xlabel = '1/FP'
    ylabel = 'TP/FP'

    plt.xlabel(xlabel, size='large', weight=700)
    plt.ylabel(ylabel, size='large', weight=700)

    return ax


def plot_precision_recall_curves(
    sample_val_dict, ax=None, use_counts=True, title=None,
    svensemble2_cutoff=None, n_truth=None, verbose=False,
    xlim=None, ylim=None, model=None,
):
    """
    Plot precision-recall curves for multiple prediction arrays against corresponding truth series

    Args:
        sample_val_dict: Dictionary containing callset names as keys and dataframes with a prediction metric as 2nd column and a truth column as "is_in_val"
        ax: matplotlib axis to plot the precision-recall curves on
        use_counts: Whether to plot curves as counts or rates
        title: Title for the plot
        n_truth: Total number of truth variants in the validation set. When provided
            and use_counts=True, a secondary x-axis showing recall (TP / n_truth) is
            drawn on top of the plot.
        xlim: Tuple (xmin, xmax) to set x-axis limits. Defaults to auto.
        ylim: Tuple (ymin, ymax) to set y-axis limits. Defaults to (0, 1).
        model: Optional svensemble2 model name. When provided and present in
            ``_model_style_mappings``, the model-specific style overrides are
            applied to ``svensemble2_*`` curves only.

    Returns:
        matplotlib axis with plotted precision-recall curves
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))

    callsets = list(sample_val_dict.keys())
    callsets.sort()

    text_objects = []
    handler_map = {}

    paired_raw_callers = set()
    for callset in sample_val_dict:
        if callset.startswith('svensemble2_'):
            raw_name = callset[len('svensemble2_'):]
            if raw_name in sample_val_dict:
                paired_raw_callers.add(raw_name)

    for idx, callset in _ordered_callset_enumerate(callsets, callset_order_generator(callsets)):
        callset_val_df = sample_val_dict[callset].dropna().copy()
        truth_series = callset_val_df['is_in_val']
        if sum(truth_series) == 0:
            if verbose:
                print(f'{callset} has no true positives, skipping')
            continue
        elif sum(truth_series) == len(truth_series):
            if verbose:
                print(f'{callset} has no true negatives, skipping')
            continue
        pred_series = callset_val_df.iloc[:, 1]

        precision, recall, thresholds = precision_recall_curve(truth_series, pred_series, pos_label=True)

        total_tp = sum(truth_series)

        if use_counts:
            tp = recall * total_tp
            zeros_mask = (tp == 0)
            tp_no_zeros = tp[~zeros_mask]
            precision_no_zeros = precision[~zeros_mask]
            tp_metric = tp_no_zeros
            precision_metric = precision_no_zeros
        else:
            tp_metric = recall
            precision_metric = precision

        style_model = model if callset.startswith('svensemble2_') else None
        line, = ax.plot(tp_metric, precision_metric, **get_style_mapping(callset, model=style_model)['line'])

        badge_handler = _build_badge_handler(callset, model=style_model)
        if badge_handler is not None:
            handler_map[line] = badge_handler

        if callset not in paired_raw_callers:
            text_label = callset if callset not in _style_mappings else _style_mappings[callset]['text']['label']
            text_style = {k: v for k, v in get_style_mapping(callset, model=style_model)['text'].items() if k != 'label'}
            t = ax.text(tp_metric[0], precision_metric[0], text_label, **text_style)
            text_objects.append(t)

        if (svensemble2_cutoff is not None) and ('svensemble2' in callset):
            threshold_idx = np.argmin(np.abs(thresholds - svensemble2_cutoff))
            ax.plot(tp_metric[threshold_idx], precision_metric[threshold_idx], 'o', color=get_style_mapping(callset, model=style_model)['line']['color'])

    if xlim is not None:
        ax.set_xlim(*xlim)
    elif not use_counts:
        ax.set_xlim(0, 1)

    if ylim is not None:
        ax.set_ylim(*ylim)
    else:
        ax.set_ylim(0, 1)

    if text_objects:
        adjust_text(
            text_objects,
            ax=ax,
            arrowprops=dict(arrowstyle='-', color='gray', lw=0.5),
            force_text=(0.5, 1.0),
            expand=(1.2, 1.4),
        )

    ax.legend(loc='center left', bbox_transform=ax.transAxes, bbox_to_anchor=(1, 0.5), frameon=False,
              handler_map=handler_map)
    ax._badge_handler_map = handler_map

    xlabel = 'Recall' if not use_counts else 'TP count'
    ylabel = 'Precision'

    plt.xlabel(xlabel, size='large', weight=700)
    plt.ylabel(ylabel, size='large', weight=700)
    if title:
        plt.title(title, size='large', weight=700)

    if use_counts and n_truth is not None:
        ax2 = ax.twiny()
        x_min, x_max = ax.get_xlim()
        ax2.set_xlim(x_min / n_truth, x_max / n_truth)
        ax2.xaxis.set_major_formatter(
            plt.FuncFormatter(lambda v, _: f'{v * 100:.0f}%')
        )
        ax2.set_xlabel('Recall', size='large', weight=700)
        ax.figure.tight_layout()

    return ax


def plot_precision_recall_gain_curves(
    sample_val_dict, ax=None, title=None,
    svensemble2_cutoff=None, verbose=False,
    text_labels=False,
    xlim=None, ylim=None, model=None,
):
    """Plot Precision-Recall-Gain curves (Flach & Kull, 2015).

    Uses per-callset prevalence as defined by the original paper:
    pi = n_pos / (n_pos + n_neg) where n_pos and n_neg are the positive
    and negative counts within each callset's prediction set.

    The gain transform is:
        precision_gain = 1 - (n_pos / n_neg) * (FP / TP)
        recall_gain    = 1 - (n_pos / n_neg) * (FN / TP)

    Points where precision or recall fall below pi (i.e. worse than the
    per-callset random baseline) map to negative gains and are clipped.

    Args:
        sample_val_dict: Dict of {callset_name: DataFrame} where each
            DataFrame has a prediction score as the 2nd column and a
            boolean truth column is_in_val.
        ax: Matplotlib axis.  Created if *None*.
        title: Plot title.
        svensemble2_cutoff: If set, mark the operating point closest to
            this threshold on SVEnsemble2 curves.
        verbose: Print skipped callsets.
        xlim: Tuple (xmin, xmax).  Defaults to (0, 1).
        ylim: Tuple (ymin, ymax).  Defaults to (0, 1).
        model: Optional svensemble2 model name. When provided and present in
            ``_model_style_mappings``, the model-specific style overrides are
            applied to ``svensemble2_*`` curves only.

    Returns:
        Matplotlib axis with plotted PRG curves.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))

    ax.plot([0, 1], [1, 0], color='0.75', linewidth=0.8, linestyle='--', zorder=0)

    callsets = list(sample_val_dict.keys())
    callsets.sort()

    text_objects = []
    handler_map = {}

    paired_raw_callers = set()
    for callset in sample_val_dict:
        if callset.startswith('svensemble2_'):
            raw_name = callset[len('svensemble2_'):]
            if raw_name in sample_val_dict:
                paired_raw_callers.add(raw_name)

    for _, callset in _ordered_callset_enumerate(callsets, callset_order_generator(callsets)):
        callset_val_df = sample_val_dict[callset].dropna().copy()
        truth_series = callset_val_df['is_in_val']
        n_pos = sum(truth_series)
        n_neg = len(truth_series) - n_pos
        if n_pos == 0:
            if verbose:
                print(f'{callset} has no true positives, skipping')
            continue
        elif n_neg == 0:
            if verbose:
                print(f'{callset} has no true negatives, skipping')
            continue
        pred_series = callset_val_df.iloc[:, 1]

        # Sort predictions descending to walk thresholds high-to-low
        order = np.argsort(-pred_series.values)
        sorted_labels = truth_series.values[order]
        sorted_scores = pred_series.values[order]

        tp = np.cumsum(sorted_labels).astype(float)
        fp = np.cumsum(~sorted_labels).astype(float)
        fn = n_pos - tp

        pos_neg_ratio = n_pos / n_neg

        with np.errstate(divide='ignore', invalid='ignore'):
            prec_gain = 1.0 - pos_neg_ratio * (fp / tp)
            rec_gain = 1.0 - pos_neg_ratio * (fn / tp)

        # Keep only points inside the unit square (both gains > 0)
        valid = (prec_gain > 0) & (rec_gain > 0) & np.isfinite(prec_gain) & np.isfinite(rec_gain)
        prec_gain = prec_gain[valid]
        rec_gain = rec_gain[valid]
        thresholds_valid = sorted_scores[valid]

        if len(prec_gain) == 0:
            if verbose:
                print(f'{callset} has no points above baseline prevalence, skipping')
            continue

        style_model = model if callset.startswith('svensemble2_') else None
        line, = ax.plot(rec_gain, prec_gain, **get_style_mapping(callset, model=style_model)['line'])

        badge_handler = _build_badge_handler(callset, model=style_model)
        if badge_handler is not None:
            handler_map[line] = badge_handler

        if (callset not in paired_raw_callers) and text_labels:
            text_label = callset if callset not in _style_mappings else _style_mappings[callset]['text']['label']
            text_style = {k: v for k, v in get_style_mapping(callset, model=style_model)['text'].items() if k != 'label'}
            t = ax.text(rec_gain[0], prec_gain[0], text_label, **text_style)
            text_objects.append(t)

        if (svensemble2_cutoff is not None) and ('svensemble2' in callset) and len(thresholds_valid):
            threshold_idx = np.argmin(np.abs(thresholds_valid - svensemble2_cutoff))
            ax.plot(
                rec_gain[threshold_idx], prec_gain[threshold_idx], 'o',
                color=get_style_mapping(callset, model=style_model)['line']['color'],
            )

    ax.set_xlim(*(xlim or (0, 1)))
    ax.set_ylim(*(ylim or (0, 1)))

    if text_objects:
        adjust_text(
            text_objects,
            ax=ax,
            arrowprops=dict(arrowstyle='-', color='gray', lw=0.5),
            force_text=(0.5, 1.0),
            expand=(1.2, 1.4),
        )

    ax.legend(loc='center left', bbox_transform=ax.transAxes, bbox_to_anchor=(1, 0.5), frameon=False,
              handler_map=handler_map)
    ax._badge_handler_map = handler_map

    plt.xlabel('Recall Gain', size='large', weight=700)
    plt.ylabel('Precision Gain', size='large', weight=700)
    if title:
        plt.title(title, size='large', weight=700)

    return ax


def get_truth_count(
    sample_name: str,
    svtypes: list[str] | None = None,
    svlens: tuple[float, float] = (50, np.inf),
    val_set_path: str | None = None,
    val_set_name: str | None = None,
) -> int:
    """Return the number of truth variants that survive the svtype/svlen filter.

    Args:
        sample_name: Sample name (key in sample_val_sets, e.g. 'NA12878').
        svtypes: List of SV types to include.  Defaults to the sample meta svtypes.
        svlens: (min, max) SVLEN range.  Defaults to (50, inf).
        val_set_path: Override path to truth BPS CSV.
        val_set_name: Override caller-param source name for annotation.

    Returns:
        Integer count of truth variants passing the filter.
    """
    from .utils import sample_val_sets
    from . import breakpoints as bps_mod

    if val_set_path is None or val_set_name is None:
        sample_cfg = sample_val_sets[sample_name]
        val_set_path = val_set_path or sample_cfg['path']
        val_set_name = val_set_name or sample_cfg['name']
        if svtypes is None:
            svtypes = sample_cfg.get('meta', {}).get('svtypes')

    truth = pd.read_csv(val_set_path)
    truth_filtered = bps_mod.svlen_svtype_filter(truth, val_set_name, svtypes=svtypes, svlens=svlens)
    return len(truth_filtered)


_val_set_preprocessors = {
    'kosugi': lambda sample, raw: preprocess_kosugi_val_set(sample, raw),
    'parikh': lambda sample, raw: preprocess_kosugi_val_set(sample, raw),
    'hg002_val': lambda sample, raw: preprocess_hg002_val_set(sample, raw),
    'huddleston': lambda sample, raw: preprocess_huddleston_val_set(sample, raw),
    'chm13_val': lambda sample, raw: preprocess_chm13_val_set(sample, raw),
    'hg00514_val': lambda sample, raw: preprocess_hg00514_val_set(sample, raw),
    'hgsvc2': lambda sample, raw: preprocess_hgsvc2_val_set(sample, raw),
}


def ensure_val_set(sample_config: dict, force_rerun: bool = False) -> Path:
    """Check that the validation breakpoint CSV exists; generate it if not.

    Args:
        sample_config: A *single-sample* config dict — the inner dict for one
            sample as returned by create_config_dict(name)[name].
            Must contain a 'val_set' key with path, name, and
            raw_path.
        force_rerun: Regenerate even when the output CSV already exists.

    Returns:
        Path to the (possibly just-created) breakpoint CSV.

    Raises:
        FileNotFoundError: If the raw input file is missing.
        KeyError: If no preprocessor is registered for the val set name.
    """
    val_cfg = sample_config['val_set']
    out_path = Path(val_cfg['path'])

    if out_path.exists() and not force_rerun:
        return out_path

    name = val_cfg['name']
    if name not in _val_set_preprocessors:
        raise KeyError(
            f"No preprocessor registered for val set '{name}'. "
            f"Available: {list(_val_set_preprocessors)}"
        )

    raw_path = Path(val_cfg.get('raw_path', ''))
    if not raw_path.name:
        raise FileNotFoundError(
            f"No raw_path configured for val set '{name}' — "
            "add it to sample_val_sets in _sample_params.py"
        )
    if not raw_path.exists():
        raise FileNotFoundError(
            f"Raw input for val set '{name}' not found at {raw_path}. "
            "Download it first."
        )

    _val_set_preprocessors[name](sample_config, raw_path)
    return out_path


def preprocess_hg002_val_set(sample, vcf_path, force_rerun=False, long_val=False):
    """
    Preprocess the HG002 validation VCF file for benchmarking by parsing, annotating,
    and converting SV events to breakpoints. Results are stored as a CSV.

    Args:
        sample (dict): Dictionary containing metadata for HG002 sample, must include correct 'val_set' key with 'path' value.
        vcf_path (str or Path): Path to the input VCF file for HG002.
        force_rerun (bool, optional): If True, forces preprocessing even if the output exists.

    Saves:
        Writes the processed breakpoints dataframe to sample['val_set']['path'] as CSV.
    """
    # Only run if the annotated file does not exist, or if force_rerun is True
    if not Path(sample['val_set']['path']).exists() or force_rerun:

        # Parse the input VCF file using the predefined caller parameters
        hg002_val_vcf = pysam.VariantFile(vcf_path)
        hg002_val_vcf_df, hg002_val_vcf_meta = vcfp.parse_vcf_to_df(
            hg002_val_vcf, caller_params=caller_params.hg002_params
        )

        # Filter out entries with missing SVTYPE
        hg002_val_vcf_df = hg002_val_vcf_df[hg002_val_vcf_df['INFO_SVTYPE'].notna()]

        # Assign unique event IDs and set them as VCF IDs
        hg002_val_vcf_df.loc[:, 'INFO_EVENT'] = [
            f'HG002_{i}' for i in range(len(hg002_val_vcf_df))
        ]
        hg002_val_vcf_df.loc[:, 'ID'] = hg002_val_vcf_df['INFO_EVENT']

        # Set dummy confidence intervals for positions and ends, and set INFO_CHR2 to chromosome name
        hg002_val_vcf_df.loc[:, 'INFO_CIPOS_l'] = -10
        hg002_val_vcf_df.loc[:, 'INFO_CIPOS_r'] = 10
        hg002_val_vcf_df.loc[:, 'INFO_CIEND_l'] = -10
        hg002_val_vcf_df.loc[:, 'INFO_CIEND_r'] = 10
        hg002_val_vcf_df.loc[:, 'INFO_CHR2'] = hg002_val_vcf_df['CHROM']

        # Transform events to breakend format, then to breakpoints
        hg002_val_vcf_df_bnds = evts.event_to_bnd(hg002_val_vcf_df)
        hg002_val_vcf_df_bps = bnds.bnd_to_bp(
            hg002_val_vcf_df_bnds,
            force_diff_cols=['INFO_CIPOS_l', 'INFO_CIPOS_r']
        )

        # If only 'long' variants are desired, filter for SVLEN > 50
        if long_val:
            hg002_val_vcf_df_bps = hg002_val_vcf_df_bps[
                hg002_val_vcf_df_bps['INFO_SVLEN'] > 50
            ]

        # Save the processed DataFrame as CSV
        hg002_val_vcf_df_bps.to_csv(sample['val_set']['path'], index=False)


def preprocess_kosugi_val_set(sample, vcf_path, force_rerun=False, long_val=False):
    """
    Preprocess the Kosugi validation set, converting to VCF-like breakpoints and saving as CSV.

    Args:
        sample (dict): Dictionary with at least key 'val_set' containing 'path'.
        force_rerun (bool, optional): If True, regenerate even if the file already exists.
        long_val (bool, optional): Only retain variants with SVLEN > 50 if True.

    Saves:
        CSV file at sample['val_set']['path'] with breakpoint-format variants.
    """
    out_path = Path(sample['val_set']['path'])

    if not out_path.exists() or force_rerun:
        val_df = load_truth_variants(
            vcf_path
        )
        val_vcf = truth_to_vcf_df(val_df)

        # Convert Kosugi variant DataFrame to breakpoint format
        val_vcf_bps = truth_event_to_bnd(val_vcf)

        if long_val:
            val_vcf_bps = val_vcf_bps[val_vcf_bps['INFO_SVLEN'] > 50]

        val_vcf_bps.to_csv(out_path, index=False)


def preprocess_chm13_val_set(sample, vcf_path, force_rerun=False):
    """
    Preprocess CHM13 validation set: filter, normalize, convert to breakpoints, save as CSV.

    Args:
        sample (dict): Dictionary with at least key 'val_set' containing 'path'.
        force_rerun (bool, optional): If True, regenerate even if the file already exists.

    Saves:
        CSV file at sample['val_set']['path'] with breakpoint-format variants.
    """
    out_path = Path(sample['val_set']['path'])

    if not out_path.exists() or force_rerun:
        # Read raw CHM13 validation annotation data
        chm_val_df = pd.read_csv(
            vcf_path
        )

        chm_val_vcf_df = chm_val_df.copy()

        # Filter for rows with valid coordinates and correct assembly
        chm_val_vcf_df = chm_val_vcf_df.dropna(subset=['Start', 'End'])
        chm_val_vcf_df = chm_val_vcf_df[
            chm_val_vcf_df['Assembly Name'] == 'GRCh38 (hg38)'
        ]

        # Standardize variant type nomenclature
        chm_rename_dict = {
            'insertion': 'INS',
            'deletion': 'DEL',
            'inversion': 'INV',
        }
        chm_val_vcf_df.loc[:, 'Variant Call type'] = chm_val_vcf_df[
            'Variant Call type'
        ].map(chm_rename_dict)

        # Ensure variant coordinates are integers
        chm_val_vcf_df.loc[:, 'Start'] = chm_val_vcf_df['Start'].astype(int)
        chm_val_vcf_df.loc[:, 'End'] = chm_val_vcf_df['End'].astype(int)

        # Prepare field names and assign standardized VCF-like values
        chm_val_cols_to_keep = [
            'CHROM', 'POS', 'INFO_CHR2', 'REF', 'ALT', 'ID',
            'INFO_EVENT', 'INFO_SVTYPE', 'INFO_SVLEN', 'INFO_END',
            'INFO_CIPOS_l', 'INFO_CIPOS_r', 'INFO_CIEND_l', 'INFO_CIEND_r'
        ]
        chm_val_vcf_df.loc[:, 'CHROM'] = 'chr' + chm_val_vcf_df['Chromosome'].astype(str)
        chm_val_vcf_df.loc[:, 'POS'] = chm_val_vcf_df['Start'] + 1
        chm_val_vcf_df.loc[:, 'INFO_CHR2'] = 'chr' + chm_val_vcf_df['Chromosome'].astype(str)
        chm_val_vcf_df.loc[:, 'INFO_END'] = chm_val_vcf_df['End']
        chm_val_vcf_df.loc[:, 'REF'] = '.'
        chm_val_vcf_df.loc[:, 'ALT'] = '.'
        chm_val_vcf_df.loc[:, 'ID'] = chm_val_vcf_df['Variant ID']
        chm_val_vcf_df.loc[:, 'INFO_EVENT'] = chm_val_vcf_df['Variant ID']
        chm_val_vcf_df.loc[:, 'INFO_SVTYPE'] = chm_val_vcf_df['Variant Call type']
        chm_val_vcf_df.loc[:, 'INFO_SVLEN'] = 0
        chm_val_vcf_df.loc[:, 'INFO_CIPOS_l'] = -10
        chm_val_vcf_df.loc[:, 'INFO_CIPOS_r'] = 10
        chm_val_vcf_df.loc[:, 'INFO_CIEND_l'] = -10
        chm_val_vcf_df.loc[:, 'INFO_CIEND_r'] = 10

        # Subset and retain only canonical chromosomes
        chm_val_vcf_df = chm_val_vcf_df[chm_val_cols_to_keep]
        chm_val_vcf_df = chm_val_vcf_df[
            chm_val_vcf_df['CHROM'].isin(sv_utils.human_chroms)
        ]

        # Convert to breakpoint format. No SVLEN in this reference set for long_val filter.
        result_bps = bnds.bnd_to_bp(
            evts.event_to_bnd(chm_val_vcf_df, bnd2_pos='end'),
            force_diff_cols=['INFO_CIPOS_l', 'INFO_CIPOS_r']
        )

        result_bps.to_csv(out_path, index=False)


def preprocess_hg00514_val_set(sample, vcf_path, force_rerun=False):
    """
    Preprocess the HG00514 validation VCF file for benchmarking by parsing, annotating,
    and converting SV events to breakpoints. Results are stored as a bps CSV.

    Args:
        sample (dict): Dictionary containing metadata for the HG00514 sample, must include
            key 'val_set' with sub-key 'path' pointing to the desired output CSV path.
        vcf_path (str or Path): Path to nstd152.GRCh38.variant_call.vcf.gz (or uncompressed).
        force_rerun (bool, optional): If True, regenerate even if the output CSV already exists.

    Saves:
        CSV file at sample['val_set']['path'] with breakpoint-format variants.
    """
    out_path = Path(sample['val_set']['path'])

    if not out_path.exists() or force_rerun:
        hg00514_vcf = pysam.VariantFile(vcf_path)
        hg00514_vcf_df, _ = vcfp.parse_vcf_to_df(
            hg00514_vcf, caller_params=caller_params.hg00514_params
        )

        # Filter to HG00514 sample and canonical SV types only
        hg00514_vcf_df = hg00514_vcf_df[
            (hg00514_vcf_df['INFO_SAMPLE'] == 'HG00514') &
            (hg00514_vcf_df['INFO_SVTYPE'].isin(['INS', 'DEL', 'DUP', 'INV']))
        ]

        # VCF uses bare chromosome numbers ("1") — prepend "chr"
        hg00514_vcf_df.loc[:, 'CHROM'] = 'chr' + hg00514_vcf_df['CHROM'].astype(str)
        hg00514_vcf_df.loc[:, 'INFO_CHR2'] = hg00514_vcf_df['CHROM']

        # SVLEN is declared Type=String in this VCF; pysam returns a tuple of strings.
        # DEL records carry a negative value — take the absolute integer.
        hg00514_vcf_df.loc[:, 'INFO_SVLEN'] = (
            hg00514_vcf_df['INFO_SVLEN']
            .apply(lambda x: abs(int(x[0])) if x is not None else 0)
        )

        # CIPOS/CIEND are Number=2 Integer tuples; split_cipos/split_ciend handle None
        # bounds (from imprecise records with "." in one position) via fillna(0).
        hg00514_vcf_df.loc[:, ['INFO_CIPOS_l', 'INFO_CIPOS_r']] = sv_utils.split_cipos(hg00514_vcf_df['INFO_CIPOS'])
        hg00514_vcf_df.loc[:, ['INFO_CIEND_l', 'INFO_CIEND_r']] = sv_utils.split_ciend(hg00514_vcf_df['INFO_CIEND'])
        hg00514_vcf_df = hg00514_vcf_df.drop(columns=['INFO_CIPOS', 'INFO_CIEND', 'INFO_SAMPLE'])

        # Use the VCF variant ID as the event identifier
        hg00514_vcf_df.loc[:, 'INFO_EVENT'] = hg00514_vcf_df['ID']

        # Convert events to BND representation then to breakpoint pairs
        hg00514_bnd_df = evts.event_to_bnd(hg00514_vcf_df, bnd2_pos='end')
        hg00514_bp_df = bnds.bnd_to_bp(
            hg00514_bnd_df,
            force_diff_cols=['INFO_CIPOS_l', 'INFO_CIPOS_r']
        )
        hg00514_bp_df.to_csv(out_path, index=False)


def preprocess_hgsvc2_val_set(sample, vcf_path, force_rerun=False, sample_name='NA12878'):
    """
    Preprocess the HGSVC2 freeze-4 integrated SV callset for benchmarking.

    The HGSVC2 VCF is a multi-sample file (35 genomes).  This function
    parses it via parse_vcf_to_df, filters for records where
    *sample_name* carries at least one ALT allele, restricts to canonical
    chromosomes and DEL/INS types, then converts events to breakpoints.

    Args:
        sample (dict): Sample config with sample['val_set']['path'] pointing
            to the desired output CSV.
        vcf_path (str or Path): Path to variants_freeze4_sv_insdel_alt.vcf.gz.
        force_rerun (bool, optional): If True, regenerate even if the output exists.
        sample_name (str, optional): Sample to extract from the multi-sample VCF.

    Saves:
        CSV file at sample['val_set']['path'] with breakpoint-format variants.
    """
    out_path = Path(sample['val_set']['path'])

    if not out_path.exists() or force_rerun:
        hgsvc2_vcf = pysam.VariantFile(str(vcf_path))
        hgsvc2_df, _ = vcfp.parse_vcf_to_df(
            hgsvc2_vcf,
            caller_params=caller_params.hgsvc2_params,
            sample_name=sample_name
        )

        # Filter to records where the sample carries at least one ALT allele
        hgsvc2_df = hgsvc2_df[
            hgsvc2_df['SAMPLE_GT'].apply(
                lambda gt: gt is not None and any(a is not None and a > 0 for a in gt)
            )
        ]

        # SVTYPE and SVLEN are per-allele tuples — unpack first element
        hgsvc2_df.loc[:, 'INFO_SVTYPE'] = hgsvc2_df['INFO_SVTYPE'].apply(
            lambda x: x[0] if isinstance(x, tuple) else x
        )
        hgsvc2_df.loc[:, 'INFO_SVLEN'] = hgsvc2_df['INFO_SVLEN'].apply(
            lambda x: abs(int(x[0])) if isinstance(x, tuple) else abs(int(x))
        )

        # Restrict to canonical chromosomes and supported SV types
        hgsvc2_df = hgsvc2_df[hgsvc2_df['CHROM'].isin(sv_utils.human_chroms)]
        hgsvc2_df = hgsvc2_df[hgsvc2_df['INFO_SVTYPE'].isin(['DEL', 'INS'])]

        hgsvc2_df.loc[:, 'INFO_CHR2'] = hgsvc2_df['CHROM']
        hgsvc2_df.loc[:, 'INFO_EVENT'] = hgsvc2_df['ID']
        hgsvc2_df.loc[:, 'INFO_CIPOS_l'] = -10
        hgsvc2_df.loc[:, 'INFO_CIPOS_r'] = 10
        hgsvc2_df.loc[:, 'INFO_CIEND_l'] = -10
        hgsvc2_df.loc[:, 'INFO_CIEND_r'] = 10

        hgsvc2_bnd_df = evts.event_to_bnd(hgsvc2_df)
        hgsvc2_bp_df = bnds.bnd_to_bp(
            hgsvc2_bnd_df,
            force_diff_cols=['INFO_CIPOS_l', 'INFO_CIPOS_r']
        )

        hgsvc2_bp_df.to_csv(out_path, index=False)


def preprocess_huddleston_val_set(sample, vcf_path, force_rerun=False):
    """
    Preprocess the Huddleston CHM13 validation set, converting to VCF-like breakpoints and saving as CSV.
    """
    out_path = Path(sample['val_set']['path'])

    if not out_path.exists() or force_rerun:
        huddleston_vcf_df, huddleston_vcf_df_meta  = vcfp.parse_vcf_to_df(pysam.VariantFile(vcf_path), caller_params=caller_params.huddleston_params, sample_name='huddleston')
        huddleston_vcf_df.loc[:, 'INFO_SVLEN'] = huddleston_vcf_df['INFO_SVLEN'].str[0]
        # Replace all strings in INFO_SVTYPE using the mapping dictionary from caller_params.huddleston_params
        mapping = caller_params.huddleston_params['svtype_mapping']
        huddleston_vcf_df.loc[:, 'INFO_SVTYPE'] = huddleston_vcf_df['INFO_SVTYPE'].map(mapping).fillna(huddleston_vcf_df['INFO_SVTYPE'])
        huddleston_vcf_df.loc[:, ['INFO_CIPOS_l', 'INFO_CIPOS_r', 'INFO_CIEND_l', 'INFO_CIEND_r']] = (-10, 10, -10, 10)
        huddleston_vcf_df.loc[:, 'ID'] = 'huddleston' + huddleston_vcf_df.index.astype(str)
        huddleston_vcf_df.loc[:, 'INFO_EVENT'] = 'huddleston' + huddleston_vcf_df.index.astype(str)
        huddleston_vcf_df.loc[:, 'INFO_CHR2'] = huddleston_vcf_df['CHROM']
        huddleston_bnd_df = evts.event_to_bnd(huddleston_vcf_df, bnd2_pos='end')
        huddleston_bp_df = bnds.bnd_to_bp(
            bnd_df=huddleston_bnd_df,
            force_diff_cols=['INFO_CIPOS_l', 'INFO_CIPOS_r']
        )

        huddleston_bp_df.to_csv(out_path, index=False)


def validate_basic_callset(callset_bps, val_set, sample_config, callset_name, val_match_margin=300, val_match_insert_delta_threshold=0.5, additional_cols: None | List[str] = None, verbose=True):
    """
    Given dataframes of a callset and a validation set, match them and return a dataframe with the following columns:
    INFO_EVENT, {qual_metric}, is_in_val, INFO_INFERRED_SVTYPE, {FILTER_PASS}

    qual_metric column is dependent on the callset and is specified in the callset params
    FILTER_PASS column is returned only if it is present in the callset
    """
    callset_bps_ = callset_bps.copy()
    val_set_ = val_set.copy()

    # Caller-specific operations go here
    if callset_name == 'manta_candidate':
        callset_bps_.loc[:, 'INFO_TOTAL_PAIR_COUNT'] = \
            callset_bps_['INFO_UPSTREAM_PAIR_COUNT'] + \
            callset_bps_['INFO_DOWNSTREAM_PAIR_COUNT'] + \
            callset_bps_['INFO_PAIR_COUNT']

    # INFO_INFERRED_SVTYPE is required downstream for svtype-stratified filtering.
    # Compute it from breakpoint geometry when the callset was not pre-annotated.
    if 'INFO_INFERRED_SVTYPE' not in callset_bps_.columns:
        callset_bps_.loc[:, 'INFO_INFERRED_SVTYPE'] = bps.annotate_svtypes(callset_bps_, callset_name)

    val_key = sample_config['val_set']['name']
    if val_key == callset_name:
        i = 0
        val_key = f'{val_key}_{i}'
        if val_key not in caller_params.callset_param_dict:
            caller_params.callset_param_dict[val_key] = caller_params.callset_param_dict[sample_config['val_set']['name']]
        else:
            while val_key in caller_params.callset_param_dict:
                i += 1
                val_key = val_key.replace(f'_{i-1}', f'_{i}')
            caller_params.callset_param_dict[val_key] = caller_params.callset_param_dict[sample_config['val_set']['name']]
            if i > 1000000:
                raise ValueError(f'Too many temporary validation set names generated for {val_key}')

    matched_ids = chunk_match.inner_join_bps(
        bp_df_dict={
            callset_name: callset_bps_,
            val_key: val_set_
        },
        max_dist=val_match_margin,
        insert_delta_threshold=val_match_insert_delta_threshold,
        ids_only=True,
        ignore_size=False
    )
    callset_bps_.loc[:, 'is_in_val'] = callset_bps_['INFO_EVENT'].astype(str).isin(matched_ids[f'{callset_name}_INFO_EVENT'])
    cols_to_keep = ['INFO_EVENT', f"{sample_config['callsets'][callset_name]['params']['qual_metric']}", 'is_in_val']
    if 'FILTER_PASS' in callset_bps_.columns:
        cols_to_keep.append('FILTER_PASS')
    if 'INFO_INFERRED_SVTYPE' in callset_bps_.columns:
        cols_to_keep.append('INFO_INFERRED_SVTYPE')
    if isinstance(additional_cols, list):
        cols_to_keep.extend([col for col in additional_cols if ((col not in cols_to_keep) and (col in callset_bps_.columns))])

    return callset_bps_.loc[callset_bps_['INFO_EVENT'].notna(), cols_to_keep]



def validate_svensemble2_predictions(svensemble2_bps, val_set, sample_config, training_set_name=None, model_name=None, pred_col=None, val_match_margin=300, val_match_insert_delta_threshold=0.5, additional_cols: None | List[str] = None, verbose=True):
    svens_calls_ = svensemble2_bps.copy()
    val_set_ = val_set.copy()

    # Resolve the single prediction column. Callers may pass it explicitly,
    # supply training_set_name + model_name to reconstruct it, or rely on
    # auto-detection of the unique '*_pred' column written by get_svensemble2_calls.
    if pred_col is None:
        if training_set_name is not None and model_name is not None:
            pred_col = f'{training_set_name}_{model_name}_pred'
        else:
            pred_candidates = [c for c in svens_calls_.columns if c.endswith('_pred')]
            if len(pred_candidates) != 1:
                raise ValueError(
                    f"Expected exactly one '*_pred' column in svensemble2 calls, found {pred_candidates}"
                )
            pred_col = pred_candidates[0]

    # INFO_INFERRED_SVTYPE is required downstream for svtype-stratified filtering.
    # Compute it from breakpoint geometry when the calls were not pre-annotated.
    if 'INFO_INFERRED_SVTYPE' not in svens_calls_.columns:
        svens_calls_.loc[:, 'INFO_INFERRED_SVTYPE'] = bps.annotate_svtypes(svens_calls_, 'svensemble2')

    val_key = sample_config['val_set']['name']
    if val_key == 'svensemble2':
        i = 0
        val_key = f'{val_key}_val_{i}'
        if val_key not in caller_params.callset_param_dict:
            caller_params.callset_param_dict[val_key] = caller_params.callset_param_dict[sample_config['val_set']['name']]
        else:
            while val_key in caller_params.callset_param_dict:
                i += 1
                val_key = val_key.replace(f'_val_{i-1}', f'_val_{i}')
            caller_params.callset_param_dict[val_key] = caller_params.callset_param_dict[sample_config['val_set']['name']]
            if i > 1000000:
                raise ValueError(f'Too many temporary validation set names generated for {val_key}')

    matched_ids = chunk_match.inner_join_bps(
        bp_df_dict={
            'svensemble2': svens_calls_,
            val_key: val_set_
        },
        max_dist=val_match_margin,
        insert_delta_threshold=val_match_insert_delta_threshold,
        ids_only=True,
        ignore_size=False
    )

    svens_calls_.loc[:, 'is_in_val'] = svens_calls_['INFO_EVENT'].astype(str).isin(matched_ids['svensemble2_INFO_EVENT'])
    cols_to_keep = ['INFO_EVENT', pred_col, 'is_in_val']
    if 'INFO_INFERRED_SVTYPE' in svens_calls_.columns:
        cols_to_keep.append('INFO_INFERRED_SVTYPE')
    if isinstance(additional_cols, list):
        cols_to_keep.extend([col for col in additional_cols if ((col not in cols_to_keep) and (col in svens_calls_.columns))])
    val_df = svens_calls_[cols_to_keep]
    val_df = val_df[val_df[pred_col].notna()]

    return val_df


# def filter_sample_by_type_len(
#     sample: Dict[str, Any],
#     svtypes: List[str] | None = None,
#     svlens: Tuple[int, int] | None = None
# ) -> Dict[str, Any]:
#     """
#     Filter a sample by SV type and length.
#     """
#     # Filters SVensemble2 and original callsets first
#     # Then uses a mask from SVEnsemble2 calls to filter matched_bps
#     # Does so for each original callset and training set
#     if svtypes is None:
#         svtypes = caller_params.default_sv_types
#     if svlens is None:
#         svlens = caller_params.default_sv_lens

#     svlen_lower, svlen_upper = svlens

#     sample_ = deepcopy(sample)

#     for training_set_name, training_set_callsets in sample_['training_sets'].items():
#         svlen_svtype_mask = bps._get_svlen_svtype_filter_mask(
#             sample_['svensemble2_calls'][training_set_name],
#             'svensemble2',
#             svlens=svlens,
#             svtypes=svtypes
#         )
#         if sample_['svensemble2_calls'] is not None:
#             sample_['svensemble2_calls'][training_set_name] = sample_['svensemble2_calls'][training_set_name][svlen_svtype_mask]
#         if sample_['matched_bps'] is not None:
#             sample_['matched_bps'][training_set_name] = sample_['matched_bps'][training_set_name][svlen_svtype_mask]

#     for callset in sample['callsets']:
#         sample_['callsets'][callset]['bps'] = bps.svlen_svtype_filter(
#             sample_['callsets'][callset]['bps'],
#             callset,
#             svlens=svlens,
#             svtypes=svtypes
#         )

#     return sample_


def run_benchmark(
    calls_csv: Path,
    caller_bps_paths: List[Path],
    callers: List[str],
    val_bps_path: Path,
    val_name: str,
    sample_name: str,
    model_name: str,
    training_set: str,
    output_dir: Path,
    verbose: bool = False,
):
    """
    DEPRECATED - use run_single_callset_benchmark instead.

    Pipeline-friendly benchmark entry point.  Accepts file paths directly
    rather than discovering them via sample/caller configs.

    Writes one CSV per caller and one for SVEnsemble2 into output_dir:
      {caller}.benchmark.csv
      svensemble2_{training_set}.benchmark.csv
    """
    val_set = pd.read_csv(val_bps_path)

    # Validate original callsets (non-ensemble predictions)
    for caller, bps_path in zip(callers, caller_bps_paths):
        callset_bps = pd.read_csv(bps_path)
        sample_config = {
            'val_set': {'name': val_name},
            'callsets': {caller: {'params': caller_params.callset_param_dict[caller]}},
        }
        val_df = validate_basic_callset(
            callset_bps=callset_bps,
            val_set=val_set,
            sample_config=sample_config,
            callset_name=caller,
            verbose=verbose
        )
        if verbose:
            match_count = val_df["is_in_val"].sum()
            total_count = len(val_df)
            print(
                f"{caller} callset for {sample_name} processed "
                f"({match_count} matches out of {total_count} predictions)"
            )
        val_df.to_csv(output_dir / f'{caller}.benchmark.csv', index=False)

    # Validate SVEnsemble2 predictions
    svensemble_preds = pd.read_csv(calls_csv)
    val_df = validate_svensemble2_predictions(
        svensemble2_bps=svensemble_preds,
        val_set=val_set,
        sample_config={'val_set': {'name': val_name}},
        training_set_name=training_set,
        model_name=model_name,
        verbose=verbose
    )
    if verbose:
        match_count = val_df["is_in_val"].sum()
        total_count = len(val_df)
        print(
            f"SVEnsemble2 predictions ({training_set}) for {sample_name} processed "
            f"({match_count} matches out of {total_count} predictions)"
        )
    val_df.to_csv(output_dir / f'svensemble2_{training_set}.benchmark.csv', index=False)


def run_single_callset_benchmark(
    callset_bps_path: Path,
    callset_name: str,
    val_bps_path: Path,
    val_name: str,
    output_dir: Path | None = None,
    svtypes: list[str] | None = None,
    svlen_min: int | None = None,
    svlen_max: int | None = None,
    verbose: bool = False,
):
    """Benchmark one callset against a validation set and write a single CSV.

    Dispatch is by callset_name: names prefixed svensemble2 are ensemble
    prediction sets (validated via validate_svensemble2_predictions); all
    other names are basic caller callsets (validated via
    validate_basic_callset, with qual_metric resolved from
    callset_param_dict[callset_name]). This unifies individual callers
    (gridss, delly, survivor_*, ...) and ensemble predictions behind one entry
    point.

    Optional SV-length and SV-type filtering is each applied independently. When
    any filter is set it is applied to BOTH the callset and the validation set
    (mirroring benchmark_model) so metrics are computed within the category.

    Args:
        callset_bps_path (Path): Path to the callset breakpoints CSV.
        callset_name (str): Callset identifier. Names prefixed svensemble2
            are treated as ensemble prediction sets; all others are basic
            caller callsets resolved against callset_param_dict.
        val_bps_path (Path): Path to the validation set breakpoints CSV.
        val_name (str): Name of the validation set.
        output_dir (Path, optional): Directory to write
            {callset_name}.benchmark.csv into. If None, the validation
            dataframe is returned instead. Default: None.
        svtypes (list[str], optional): INFO_INFERRED_SVTYPE values to keep.
            None keeps all types. Default: None.
        svlen_min (int, optional): Inclusive lower SV-length bound. None means
            0. Default: None.
        svlen_max (int, optional): Inclusive upper SV-length bound. None means
            unbounded. Default: None.
        verbose (bool, optional): Whether to print processing info.
            Default: False.

    Returns:
        pandas.DataFrame or None:
        - The validation dataframe when output_dir is None;
        - None after writing {callset_name}.benchmark.csv to output_dir when output_dir is not None.
    """
    val_set = pd.read_csv(val_bps_path)
    callset_bps = pd.read_csv(callset_bps_path, low_memory=False)
    sample_config = {'val_set': {'name': val_name}}

    source = 'svensemble2' if callset_name.startswith('svensemble2') else callset_name

    if svtypes is not None or svlen_min is not None or svlen_max is not None:
        svlens = (
            svlen_min if svlen_min is not None else 0,
            svlen_max if svlen_max is not None else np.inf,
        )
        callset_bps = bps.svlen_svtype_filter(callset_bps, source, svtypes=svtypes, svlens=svlens)
        val_set = bps.svlen_svtype_filter(val_set, val_name, svtypes=svtypes, svlens=svlens)
        if verbose:
            print(f"Filtered {callset_name} and {val_name} to svtypes={svtypes}, svlens={svlens}")

    if callset_name.startswith('svensemble2'):
        val_df = validate_svensemble2_predictions(
            svensemble2_bps=callset_bps,
            val_set=val_set,
            sample_config=sample_config,
            verbose=verbose,
        )
    else:
        sample_config['callsets'] = {
            callset_name: {'params': caller_params.callset_param_dict[callset_name]}
        }
        val_df = validate_basic_callset(
            callset_bps=callset_bps,
            val_set=val_set,
            sample_config=sample_config,
            callset_name=callset_name,
            verbose=verbose,
        )

    if verbose:
        match_count = val_df["is_in_val"].sum()
        total_count = len(val_df)
        print(
            f"{callset_name} callset processed against {val_name} "
            f"({match_count} matches out of {total_count} predictions)"
        )

    if output_dir is None:
        return val_df

    output_dir.mkdir(parents=True, exist_ok=True)
    val_df.to_csv(output_dir / f'{callset_name}.benchmark.csv', index=False)


# Benchmark a single sample's callsets and SVEnsemble2 ensemble predictions against a validation set,
# saving the results in a nested dict (callset/SVEnsemble2) structure to disk.
def benchmark_model(
    sample_name,
    model_name,
    svtypes=None,
    svlens=None,
    val_set_name=None,
    val_set_path=None,
    training_sets: List[str] | str | None = None,
    val_match_margin: int = 300,
    val_match_insert_delta_threshold: float = 0.5,
    skip_filtering: bool = False,
    additional_cols: None | List[str] = None,
    additional_callsets: None | List[str] = None,
    limit_to_callsets: None | List[str] = None,
    ground_truth_name: str | None = None,
    verbose=False
):
    """
    Benchmark callset and SVEnsemble2 predictions for a sample against the corresponding validation set.

    Args:
        sample_name (str): Name of the sample.
        model_name (str): Model identifier (e.g. 'unbiased').
        svtypes (list or tuple): Structural variant types to consider.
        svlens (tuple): SV length range (min, max).
        val_set_name (str): Name of the validation set. If not provided, the function will use the validation set name from the sample config.
            the validation set name must have a valid callset param setup in _caller_params.py
        val_set_path (str): Path to the validation set. If not provided, the function will use the validation set path from the sample config.
        training_sets (list or str): Training sets to consider.
        additional_cols (list or str): Additional columns to keep in the validation result dataframe.
        additional_callsets (list or str): Names of additional callsets to include in the benchmark.
        limit_to_callsets (list or str): If input, the function will only benchmark the callsets in the list.
            To add an svensemble2 prediction, use the format 'svensemble2_{training_set_name}'.
        verbose (bool, optional): Whether to print processing info. Default: True.

    Returns:
        None. Writes a pickled nested dict with validation results to sample dir.
    """
    # Check that there are no svtypes/svlens specified if skip_filtering is True
    # if skip_filtering and (svtypes is not None or svlens is not None):
    #     raise ValueError("Cannot specify svtypes/svlens if set to skip filtering")

    if training_sets is None:
        training_sets = caller_params.training_set_list_presets['default']
    elif isinstance(training_sets, str):
        training_sets = caller_params.training_set_list_presets[training_sets]

    if skip_filtering:
        svtypes = None
        svtypes_str = 'all'

    if svtypes is None:
        svtypes_str = 'all'
    elif isinstance(svtypes, str):
        svtypes_str = svtypes
    elif isinstance(svtypes, list):
        svtypes_str = '-'.join(svtypes)
    else:
        raise TypeError(f"Invalid type for svtypes: {type(svtypes)}")

    if skip_filtering:
        svlens = None
        svlens_str = 'all_lens'
    else:
        svlens_str = f'{svlens[0]}:{svlens[1]}'

    # Normalise sentinel empty strings (common from SLURM array jobs)
    if isinstance(val_set_name, str) and not val_set_name.strip():
        val_set_name = None
    if isinstance(val_set_path, str) and not val_set_path.strip():
        val_set_path = None

    val_set_str = ""
    if val_set_name is None and val_set_path is None:
        # Neither provided → use the validation set registered for this sample
        val_set_arg = "use_name"
    elif val_set_path is not None and val_set_name is None:
        raise ValueError(
            "--val_set_name is required when --val_set_path is provided."
        )
    elif val_set_name is not None and val_set_path is None:
        raise ValueError(
            f"--val_set_path is required when --val_set_name ('{val_set_name}') is provided."
        )
    else:
        try:
            val_set_path = Path(val_set_path)
        except Exception as e:
            raise ValueError(f"Invalid input path for validation set: {val_set_path}, error: {e}")
        val_set_arg = sample_params.sample_val_set_template.copy()
        val_set_arg['path'] = val_set_path
        val_set_arg['name'] = val_set_name
        val_set_str = val_set_name

    # Build config dict for the sample, with all paths
    config_dicts = []
    for training_set_name in training_sets:
        config_dicts.append(sv_utils.create_config_dict(
            sample_name,
            training_set=training_set_name,
            val_set=val_set_arg,
            model_name=model_name,
            additional_callsets=additional_callsets,
            ground_truth_name=ground_truth_name
        )[sample_name])

    # Get a list of callsets to load
    callsets_to_load = []
    callset_paths = []
    callset_config_dicts = []
    for config_dict in config_dicts:
        for callset_name in config_dict['callsets'].keys():
            if limit_to_callsets is not None and callset_name not in limit_to_callsets:
                continue
            if callset_name not in callsets_to_load:
                callsets_to_load.append(callset_name)
                callset_paths.append(config_dict['callsets'][callset_name]['bps_path'])
                callset_config_dicts.append(config_dict)

    # Get a list of svensemble2 predictions to load
    svensemble2_preds_to_load = []
    svensemble2_preds_paths = []
    svensemble2_preds_config_dicts = []
    for config_dict in config_dicts:
        for training_set_name in config_dict['training_sets'].keys():
            if limit_to_callsets is not None and f'svensemble2_{training_set_name}' not in limit_to_callsets:
                continue
            if training_set_name not in svensemble2_preds_to_load:
                svensemble2_preds_to_load.append(f'svensemble2_{training_set_name}')
                svensemble2_preds_paths.append(config_dict['workflow_dir'] / 'svensemble2_calls.csv')
                svensemble2_preds_config_dicts.append(config_dict)

    if additional_callsets is None:
        additional_callsets = []
    for additional_callset in additional_callsets:
        if limit_to_callsets is not None and additional_callset not in limit_to_callsets:
            continue
        if additional_callset not in callsets_to_load:
            # In a single sample with the same model all the paths to additional callsets should be the same
            additional_callset_paths = set(config_dict['callsets'][additional_callset]['bps_path'] for config_dict in config_dicts)
            if len(additional_callset_paths) > 1:
                raise ValueError(f"Multiple paths found for additional callset {additional_callset} in different config dicts, \
                                 check sample configs for {sample_name}")
            additional_callset_path = additional_callset_paths.pop()
            callsets_to_load.append(additional_callset)
            callset_paths.append(additional_callset_path)
            callset_config_dicts.append(config_dicts[0])

    val_set_names = set(config_dict['val_set']['name'] for config_dict in config_dicts)
    if len(val_set_names) > 1:
        raise ValueError(f"Multiple validation set names found in different config dicts, \
                         check sample configs for {sample_name}")
    elif len(val_set_names) == 0:
        raise FileNotFoundError(f"No validation set names found for {sample_name}, check sample configs")
    val_set_name = val_set_names.pop()

    val_set_path = set(config_dict['val_set']['path'] for config_dict in config_dicts)
    if len(val_set_path) > 1:
        raise ValueError(f"Multiple paths found for validation set in different config dicts, \
                         check sample configs for {sample_name}")
    elif len(val_set_path) == 0:
        raise FileNotFoundError(f"No validation set paths found for {sample_name}, check sample configs")
    val_set_path = val_set_path.pop()

    val_set = pd.read_csv(val_set_path)
    val_set = bps.svlen_svtype_filter(
        val_set, val_set_name, svtypes=svtypes, svlens=svlens
    )
    if val_set.shape[0] == 0:
        raise ValueError(f"No {val_set_name} calls found for {sample_name} after filtering for {svtypes_str} and {svlens_str}")

    # Validate original callsets (non-ensemble predictions)
    for callset_name, callset_path, callset_config_dict in zip(callsets_to_load, callset_paths, callset_config_dicts):
        callset_bps = pd.read_csv(callset_path)
        callset_bps = bps.svlen_svtype_filter(
            callset_bps,
            callset_name,
            svtypes=svtypes,
            svlens=svlens
        )
        val_df = validate_basic_callset(
            callset_bps=callset_bps,
            val_set=val_set,
            sample_config=callset_config_dict,
            callset_name=callset_name,
            val_match_margin=val_match_margin,
            val_match_insert_delta_threshold=val_match_insert_delta_threshold,
            additional_cols=additional_cols,
            verbose=verbose
        )
        # Store result under this callset name
        bench_dir = callset_config_dict['workflow_dir'].parent / f'{sample_name}.{model_name}.caller_benchmark' / 'benchmark'
        if not bench_dir.exists():
            bench_dir.mkdir(exist_ok=True, parents=True)

        if verbose:
            match_count = val_df["is_in_val"].sum()
            total_count = len(val_df)
            print(
                f"{callset_name} callset for {sample_name} processed "
                f"({match_count} matches out of {total_count} predictions)"
            )
        val_df.to_csv(bench_dir / f'{callset_name}.{svtypes_str}.{svlens_str}.{val_set_str}.benchmark.csv', index=False)

    # Validate SVEnsemble2 predictions for each training set
    for svensemble2_preds_to_load, svensemble2_preds_path, svensemble2_preds_config_dict in zip(svensemble2_preds_to_load, svensemble2_preds_paths, svensemble2_preds_config_dicts):
        svensemble_preds = pd.read_csv(svensemble2_preds_path)
        svensemble_preds = bps.svlen_svtype_filter(
            svensemble_preds,
            'svensemble2',
            svtypes=svtypes,
            svlens=svlens
        )
        val_df = validate_svensemble2_predictions(
            svensemble2_bps=svensemble_preds,
            val_set=val_set,
            sample_config=svensemble2_preds_config_dict,
            training_set_name=svensemble2_preds_to_load.replace('svensemble2_', ''),
            model_name=model_name,
            val_match_margin=val_match_margin,
            val_match_insert_delta_threshold=val_match_insert_delta_threshold,
            additional_cols=additional_cols,
            verbose=verbose
        )
        bench_dir = svensemble2_preds_config_dict['workflow_dir'] / 'benchmark'
        if not bench_dir.exists():
            bench_dir.mkdir(exist_ok=True, parents=True)
        if verbose:
            match_count = val_df["is_in_val"].sum()
            total_count = len(val_df)
            print(
                f"SVEnsemble2 predictions ({svensemble2_preds_to_load.replace('svensemble2_', '')}) for {sample_name} processed "
                f"({match_count} matches out of {total_count} predictions)"
            )
        val_df.to_csv(bench_dir / f'{svensemble2_preds_to_load}.{svtypes_str}.{svlens_str}.{val_set_str}.benchmark.csv', index=False)


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--sample_name', type=str, required=True)
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--svtypes', type=str, default=None)
    parser.add_argument('--svlen_lower', type=int, default=caller_params.default_sv_lens[0])
    parser.add_argument('--svlen_upper', type=int, default=caller_params.default_sv_lens[1])
    parser.add_argument('--val_set_path', type=str, default=None)
    parser.add_argument('--val_set_name', type=str, default=None)
    parser.add_argument('--training_sets', type=str, default='default')
    parser.add_argument('--val_match_margin', type=int, default=300)
    parser.add_argument('--val_match_insert_delta_threshold', type=float, default=0.5)
    parser.add_argument('--skip_filtering', action='store_true', default=False)
    parser.add_argument('--additional_cols', type=str, default=None)
    parser.add_argument('--additional_callsets', type=str, default=None)
    parser.add_argument('--limit_to_callsets', type=str, default=None)
    parser.add_argument('--ground_truth_name', type=str, default=None)
    parser.add_argument('--verbose', action='store_true', default=False)
    args = parser.parse_args()

    svtypes = args.svtypes
    if svtypes == 'all':
        svtypes = None
    if svtypes is not None:
        svtypes = svtypes.split(sep=',')
        # Check that there are only supported svtypes in the string
        for svtype in svtypes:
            if svtype not in caller_params.default_sv_types:
                raise ValueError(f"Unsupported svtype: {svtype}")

    if args.additional_callsets is not None:
        additional_callsets = args.additional_callsets.split(sep=',')
    else:
        additional_callsets = None

    if args.skip_filtering:
        svlens = None
    else:
        svlens = (args.svlen_lower, args.svlen_upper)

    if args.training_sets is not None:
        training_sets = args.training_sets.split(sep=',')
    if len(training_sets) == 1 and training_sets[0] in caller_params.training_set_list_presets.keys():
        training_sets = caller_params.training_set_list_presets[training_sets[0]]


    if args.additional_cols is not None:
        additional_cols = args.additional_cols.split(sep=',')
    else:
        additional_cols = None

    if args.limit_to_callsets is not None:
        limit_to_callsets = args.limit_to_callsets.split(sep=',')
    else:
        limit_to_callsets = None

    # If the val_set_path or val_set_name is an empty string, set it to None
    # Needed for proper argument parsing in slurm arrays
    if args.val_set_path == "":
        val_set_path = None
    else:
        val_set_path = args.val_set_path
    if args.val_set_name == "":
        val_set_name = None
    else:
        val_set_name = args.val_set_name

    benchmark_model(
        sample_name=args.sample_name,
        model_name=args.model_name,
        svtypes=svtypes,
        svlens=(args.svlen_lower, args.svlen_upper),
        training_sets=training_sets,
        val_set_path=val_set_path,
        val_set_name=val_set_name,
        val_match_margin=args.val_match_margin,
        val_match_insert_delta_threshold=args.val_match_insert_delta_threshold,
        skip_filtering=args.skip_filtering,
        additional_cols=additional_cols,
        additional_callsets=additional_callsets,
        limit_to_callsets=limit_to_callsets,
        ground_truth_name=args.ground_truth_name,
        verbose=args.verbose
    )
