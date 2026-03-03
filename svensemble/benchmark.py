#! /usr/bin/env python3

#SBATCH --job-name=svensemble2_benchmark
#SBATCH --output=logs/benchmark/svensemble2_benchmark.%j.out
#SBATCH --error=logs/benchmark/svensemble2_benchmark.%j.err
#SBATCH --time=10:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G

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

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42



import pysam

if __name__ == '__main__':
    # Set working directory and sys.path for local svensemble module
    WORKDIR = "/vast/ai_projects/Papenfuss_SVEnsemble2"
    os.chdir(WORKDIR)
    if WORKDIR not in sys.path:
        sys.path.insert(0, WORKDIR)
    import svensemble.breakends as bnds
    import svensemble.events as evts
    import svensemble.utils as sv_utils
    import svensemble._caller_params as caller_params
    import svensemble._sample_params as sample_params
    import svensemble.vcfparse as vcfp
    import svensemble.chunk_match as chunk_match
    import svensemble.breakpoints as bps
    import svensemble.matched_breakpoints as matchedbps
else:
    from . import breakends as bnds
    from . import events as evts
    from . import utils as sv_utils
    from . import _caller_params as caller_params
    from . import _sample_params as sample_params
    from . import vcfparse as vcfp
    from . import chunk_match as chunk_match
    from . import breakpoints as bps
    from . import matched_breakpoints as matchedbps

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
_style_mappings = {
    'gridss': {
        'line': {
            'color': xgfs_fancy6[0],
            'linewidth': 0.7,
            'dashes': (2, 2),
            'label': 'GRIDSS',
        },
        'text': {
            'label': 'GR',
        }
    },
    'gridss_H': {
        'line': {
            'color': xgfs_fancy6[0],
            'linewidth': 1,
            'dashes': (2, 2),
            'label': 'GRIDSS (High Confidence Calls)',
        },
        'text': {
            'label': 'GR_H',
        }
    },
    'gridss_H_lazy': {
        'line': {
            'color': tuple([c * 1.2 for c in xgfs_fancy6[0] if c * 1.2 < 1]),
            'linewidth': 1,
            'dashes': (2, 2),
            'label': 'GRIDSS (High Confidence Calls, Lazy Loading)',
        },
        'text': {
            'label': 'GR_H_lazy',
        }
    },
    'svensemble2_gridss': {
        'line': {
            'color': xgfs_fancy6[0],
            'linewidth': 1,
            'label': 'SVEnsemble2 (GRIDSS)',
        },
        'text': {
            'label': 'GR',
        }
    },
    'manta_diploid': {
        'line': {
            'color': xgfs_fancy6[3],
            'linewidth': 0.7,
            'dashes': (2, 2),
            'label': 'Manta',
        },
        'text': {
            'label': 'MA',
        }
    },
    'svensemble2_manta_diploid': {
        'line': {
            'color': xgfs_fancy6[3],
            'linewidth': 1,
            'label': 'SVEnsemble2 (Manta)',
        },
        'text': {
            'label': 'MA',
        }
    },
    'delly': {
        'line': {
            'color': xgfs_fancy6[2],
            'linewidth': 0.7,
            'dashes': (2, 2),
            'label': 'Delly',
        },
        'text': {
            'label': 'DE',
        }
    },
    'svensemble2_delly': {
        'line': {
            'color': xgfs_fancy6[2],
            'linewidth': 1,
            'label': 'SVEnsemble2 (Delly)',
        },
        'text': {
            'label': 'DE',
        }
    },
    'survivor': {
        'line': {
            'color': xgfs_fancy6[1],
            'linewidth': 1.5,
            'label': 'SURVIVOR 2 out of 3 Consensus',
        },
        'text': {
            'label': 'SU',
        }
    },
    'svensemble2_delly+manta_diploid': {
        'line': {
            'color': xgfs_fancy6[4],
            'linewidth': 1,
            'label': 'SVEnsemble2 (Delly + Manta)',
        },
        'text': {
            'label': 'DE+MA',
        }
    },
    'svensemble2_gridss+manta_diploid': {
        'line': {
            'color': xgfs_fancy6[5],
            'linewidth': 1,
            'label': 'SVEnsemble2 (GRIDSS + Manta)',
        },
        'text': {
            'label': 'GR+MA',
        }
    },
    'svensemble2_gridss+manta_diploid+delly': {
        'line': {
            'color': xgfs_fancy6[6],
            'linewidth': 1,
            'label': 'SVEnsemble2 (GRIDSS + Manta + Delly)',
        },
        'text': {
            'label': 'GR+MA+DE',
        }
    }
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

def get_style_mapping(callset, default_style_mapping=default_style_mapping):
    _default_style_mapping = deepcopy(default_style_mapping)
    if callset in _style_mappings:
        style_mapping = sv_utils.update(_default_style_mapping, _style_mappings[callset])
        style_mapping['text']['bbox']['ec'] = style_mapping['line']['color']
        style_mapping['text']['bbox']['fc'] = style_mapping['line']['color']
    else:
        style_mapping = _default_style_mapping
    return style_mapping

# Create ordering for callsets. svensemble2 calls should always follow corresponding original callsets.
# Svensemble ensemble calls should be last, survivor calls should be just before them
# At the moment the ordering is manual
_callset_order = [
    'gridss',
    'gridss_H',
    'svensemble2_gridss',
    'manta_diploid',
    'svensemble2_manta_diploid',
    'delly',
    'svensemble2_delly',
    'survivor',
    'svensemble2_delly+manta_diploid',
    'svensemble2_gridss+manta_diploid',
    'svensemble2_gridss+manta_diploid+delly',
]
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


def style_svensemble_lines(line, callset):
    # If svensemble2 in in the callset name, plot the curve normally
    # Otherwise, plot a dotted line instead
    style_mapping = get_style_mapping(callset)['line']
    # line.set_color(style_mapping['color'])
    # line.set_linewidth(style_mapping['linewidth'])
    # line.set_dashes(dashes)
    # line.set_dash_capstyle('round')
    line.set(**style_mapping)


def style_text(text: mpl.text.Text, callset):
    style_mapping = get_style_mapping(callset)['text']
    text.set(**style_mapping)


def plot_roc_curves(sample_val_dict, ax=None, use_counts=True, title=None, verbose=False):
    """
    Plot ROC curves for multiple prediction arrays against corresponding truth series


    Args:
        sample_val_dict: Dictionary containing callset names as keys and dataframes with a prediction metric as 2nd column and a truth column as "is_in_val"
        ax: matplotlib axis to plot the ROC curves on
        use_counts: Whether to plot the ROC curves as counts or rates

    Returns:
        matplotlib axis with plotted ROC curves
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8,8))
    if title is not None:
        ax.set_title(title)

    callsets = list(sample_val_dict.keys())
    callsets.sort()
    # n_callsets = len(callsets)
    # colors = get_plot_colors(n_callsets)

    for idx, callset in _ordered_callset_enumerate(callsets, _callset_order):
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

        # Drop last values in all arrays
        # fpr = fpr[:-1]
        # tpr = tpr[:-1]
        # thresholds = thresholds[:-1]

        total_tp = sum(truth_series)
        total_tn = len(truth_series) - total_tp

        if use_counts:
            # Convert rates to counts
            tpr = tpr * total_tp
            fpr = fpr * total_tn
        else:
            ax.set_ylim(0, 1)
            ax.set_xlim(0, 1)

        ax.plot(fpr, tpr, **get_style_mapping(callset)['line'])

    # Add a dotted diagonal line from (0, 0) to (total_tn, total_tp) if use_counts, else from (0, 0) to (1, 1)
    # if use_counts:
    #     ax.plot([0, total_tn], [0, total_tp], linestyle='--', color='grey', linewidth=1, label='Random')
    # else:
    #     ax.plot([0, 1], [0, 1], linestyle='--', color='grey', linewidth=1, label='Random')

    ax.legend(loc='center left', bbox_transform=ax.transAxes, bbox_to_anchor=(1, 0.5), frameon=False)

    xlabel = 'False Positives' if use_counts else 'False Positive Rate'
    ylabel = 'True Positives' if use_counts else 'True Positive Rate'

    plt.xlabel(xlabel, size='large', weight=700)
    plt.ylabel(ylabel, size='large', weight=700)

    return ax


def plot_reverse_roc_curves(sample_val_dict, svtype=None, ax=None, use_counts=True, filter_pass=False, title=None):
    """
    Plot ROC curves for multiple prediction arrays against corresponding truth series


    Args:
        sample_val_dict: Dictionary containing callset names as keys and dataframes with a prediction metric as 2nd column and a truth column as "is_in_val"
        ax: matplotlib axis to plot the ROC curves on
        use_counts: Whether to plot the ROC curves as counts or rates

    Returns:
        matplotlib axis with plotted ROC curves
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8,8))
    if title is not None:
        ax.set_title(title)

    ax.set_xscale('log')
    ax.set_yscale('log')

    callsets = list(sample_val_dict.keys())
    callsets.sort()
    # n_callsets = len(callsets)
    # colors = get_plot_colors(n_callsets)

    for idx, callset in _ordered_callset_enumerate(callsets, _callset_order):
        callset_val_df = sample_val_dict[callset].dropna().copy()

        if filter_pass and 'FILTER_PASS' in sample_val_dict[callset].columns:
            sample_mask = callset_val_df['FILTER_PASS'].astype(bool)
        elif filter_pass:
            print(f'{callset} does not have a FILTER_PASS column, ignoring filter_pass argument')
            sample_mask = np.ones(len(callset_val_df), dtype=bool)
        else:
            sample_mask = np.ones(len(callset_val_df), dtype=bool)

        if svtype is not None:
            sample_mask = sample_mask & (callset_val_df['INFO_INFERRED_SVTYPE'] == svtype)

        # Remove rows with NaNs and apply sample mask
        callset_val_df = callset_val_df.loc[sample_mask, :]
        truth_series = callset_val_df['is_in_val']
        if sum(truth_series) == 0:
            print(f'{callset} has no true positives, skipping')
            continue
        elif sum(truth_series) == len(truth_series):
            print(f'{callset} has no true negatives, skipping')
            continue
        pred_series = callset_val_df.iloc[:, 1]

        fpr, tpr, thresholds = roc_curve(truth_series, pred_series, pos_label=True)

        # Drop last values in all arrays
        # fpr = fpr[:-1]
        # tpr = tpr[:-1]
        # thresholds = thresholds[:-1]

        total_tp = sum(truth_series)
        total_tn = len(truth_series) - total_tp

        if use_counts:
            # Convert rates to counts
            tp = tpr * total_tp
            fp = fpr * total_tn
            zeros_mask = (fp == 0) | (tp == 0)

            tp_no_zeros = tp[~zeros_mask]
            fp_no_zeros = fp[~zeros_mask]

            fp_metric = 1/fp_no_zeros
            tp_fp_metric = tp_no_zeros/fp_no_zeros
        else:
            fp_metric = fpr
            tp_fp_metric = tpr

        ax.plot(fp_metric, tp_fp_metric, **get_style_mapping(callset)['line'])

    # Add a dotted diagonal line from (0, 0) to (total_tn, total_tp) if use_counts, else from (0, 0) to (1, 1)
    # if use_counts:
    #     ax.plot([0, total_tn], [0, total_tp], linestyle='--', color='grey', linewidth=1, label='Random')
    # else:
    #     ax.plot([0, 1], [0, 1], linestyle='--', color='grey', linewidth=1, label='Random')

    ax.legend(loc='center left', bbox_transform=ax.transAxes, bbox_to_anchor=(1, 0.5), frameon=False)

    xlabel = '1/FP'
    ylabel = 'TP/FP'

    plt.xlabel(xlabel, size='large', weight=700)
    plt.ylabel(ylabel, size='large', weight=700)

    return ax


def plot_precision_recall_curves(sample_val_dict, ax=None, use_counts=True, title=None, svensemble2_cutoff=None, n_truth=None, verbose=False):
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

    Returns:
        matplotlib axis with plotted precision-recall curves
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))

    callsets = list(sample_val_dict.keys())
    callsets.sort()
    # n_callsets = len(callsets)
    # colors = get_plot_colors(n_callsets)

    for idx, callset in _ordered_callset_enumerate(callsets, _callset_order):
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
            # fp = tp * (1 / precision - 1)
            zeros_mask = (tp == 0)
            tp_no_zeros = tp[~zeros_mask]
            precision_no_zeros = precision[~zeros_mask]
            # Plot: x = tp count, y = precision as count ratio (tp / (tp+fp))
            tp_metric = tp_no_zeros
            precision_metric = precision_no_zeros
            ax.set_ylim(0, 1)
        else:
            tp_metric = recall
            precision_metric = precision
            ax.set_ylim(0, 1)
            ax.set_xlim(0, 1)

        ax.plot(tp_metric, precision_metric, **get_style_mapping(callset)['line'])

        text_label = callset if callset not in _style_mappings else _style_mappings[callset]['text']['label']
        text_style = {k: v for k, v in get_style_mapping(callset)['text'].items() if k != 'label'}
        ax.text(tp_metric[0], precision_metric[0], text_label, **text_style)

        # If svensemble2_cutoff is present, add a point to the curve at the svensemble2_cutoff
        if (svensemble2_cutoff is not None) and ('svensemble2' in callset):
            # Get threshold index for svensemble2_cutoff
            threshold_idx = np.argmin(np.abs(thresholds - svensemble2_cutoff))
            # Add a point to the curve at the threshold index
            ax.plot(tp_metric[threshold_idx], precision_metric[threshold_idx], 'o', color=get_style_mapping(callset)['line']['color'])


    ax.legend(loc='center left', bbox_transform=ax.transAxes, bbox_to_anchor=(1, 0.5), frameon=False)

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
        hg002_val_vcf_df_bps = bnds.bnd_to_bp(hg002_val_vcf_df_bnds)

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
        result_bps = bnds.bnd_to_bp(evts.event_to_bnd(chm_val_vcf_df, bnd2_pos='end'))

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
        hg00514_bp_df = bnds.bnd_to_bp(hg00514_bnd_df)
        hg00514_bp_df.to_csv(out_path, index=False)


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
        huddleston_bp_df = bnds.bnd_to_bp(bnd_df=huddleston_bnd_df)

        huddleston_bp_df.to_csv(out_path, index=False)


def validate_basic_callset(callset_bps, val_set, sample_config, callset_name, val_match_margin=300, val_match_insert_size_threshold=0.7, additional_cols: None | List[str] = None, verbose=True):
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

    # GRIDSS stores per-breakend FILTER_PASS columns; unify them into a single FILTER_PASS
    if 'FILTER_PASS' not in callset_bps_.columns and \
            'FILTER_PASS_bnd1' in callset_bps_.columns and \
            'FILTER_PASS_bnd2' in callset_bps_.columns:
        callset_bps_.loc[:, 'FILTER_PASS'] = (
            callset_bps_['FILTER_PASS_bnd1'].astype(bool) &
            callset_bps_['FILTER_PASS_bnd2'].astype(bool)
        )

    matched_ids = chunk_match.inner_join_bps(
        bp_df_dict={
            callset_name: callset_bps_,
            sample_config['val_set']['name']: val_set_
        },
        max_dist=val_match_margin,
        insert_size_threshold=val_match_insert_size_threshold,
        ids_only=True,
        ignore_size=True
    )
    callset_bps_.loc[:, 'is_in_val'] = callset_bps_['INFO_EVENT'].isin(matched_ids[f'{callset_name}_INFO_EVENT'])
    cols_to_keep = ['INFO_EVENT', f"{sample_config['callsets'][callset_name]['params']['qual_metric']}", 'is_in_val']
    if 'FILTER_PASS' in callset_bps_.columns:
        cols_to_keep.append('FILTER_PASS')
    if 'INFO_INFERRED_SVTYPE' in callset_bps_.columns:
        cols_to_keep.append('INFO_INFERRED_SVTYPE')
    if isinstance(additional_cols, list):
        cols_to_keep.extend([col for col in additional_cols if ((col not in cols_to_keep) and (col in callset_bps_.columns))])

    return callset_bps_.loc[callset_bps_['INFO_EVENT'].notna(), cols_to_keep]



def validate_svensemble2_predictions(svensemble2_bps, val_set, sample_config, training_set_name, model_name, val_match_margin=300, val_match_insert_size_threshold=0.7, additional_cols: None | List[str] = None, verbose=True):
    svens_calls_ = svensemble2_bps.copy()
    val_set_ = val_set.copy()

    matched_ids = chunk_match.inner_join_bps(
        bp_df_dict={
            'svensemble2': svens_calls_,
            sample_config['val_set']['name']: val_set_
        },
        max_dist=val_match_margin,
        insert_size_threshold=val_match_insert_size_threshold,
        ids_only=True,
        ignore_size=True
    )

    svens_calls_.loc[:, 'is_in_val'] = svens_calls_['INFO_EVENT'].isin(matched_ids['svensemble2_INFO_EVENT'])
    cols_to_keep = ['INFO_EVENT', f'{training_set_name}_{model_name}_pred', 'is_in_val']
    if 'INFO_INFERRED_SVTYPE' in svens_calls_.columns:
        cols_to_keep.append('INFO_INFERRED_SVTYPE')
    if isinstance(additional_cols, list):
        cols_to_keep.extend([col for col in additional_cols if ((col not in cols_to_keep) and (col in svens_calls_.columns))])
    val_df = svens_calls_[cols_to_keep]
    val_df = val_df[val_df[f'{training_set_name}_{model_name}_pred'].notna()]

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
    val_match_insert_size_threshold: float = 0.7,
    skip_filtering: bool = False,
    additional_cols: None | List[str] = None,
    additional_callsets: None | List[str] = None,
    limit_to_callsets: None | List[str] = None,
    verbose=False
):
    """
    Benchmark callset and SVEnsemble2 predictions for a sample against the corresponding validation set.

    Args:
        sample_name (str): Name of the sample.
        model_name (str): Model identifier (e.g. 'unbiased').
        svtypes (list or tuple): Structural variant types to consider.
        svlens (tuple): SV length range (min, max).
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

    val_set_str = ""
    if val_set_path is None:
        val_set_arg = "use_name"
    else:
        try:
            val_set_path = Path(val_set_path)
        except Exception as e:
            raise ValueError(f"Invalid input path for validation set: {val_set_path}, error: {e}")
        if val_set_name is None:
            raise ValueError("Validation set name is required when providing a path for the validation set, use --val_set_name argument")
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
            additional_callsets=additional_callsets
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
            val_match_insert_size_threshold=val_match_insert_size_threshold,
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
            val_match_insert_size_threshold=val_match_insert_size_threshold,
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
    parser.add_argument('--val_match_insert_size_threshold', type=float, default=0.7)
    parser.add_argument('--skip_filtering', action='store_true', default=False)
    parser.add_argument('--additional_cols', type=str, default=None)
    parser.add_argument('--additional_callsets', type=str, default=None)
    parser.add_argument('--limit_to_callsets', type=str, default=None)
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

    benchmark_model(
        sample_name=args.sample_name,
        model_name=args.model_name,
        svtypes=svtypes,
        svlens=(args.svlen_lower, args.svlen_upper),
        training_sets=training_sets,
        val_set_path=args.val_set_path,
        val_set_name=args.val_set_name,
        val_match_margin=args.val_match_margin,
        val_match_insert_size_threshold=args.val_match_insert_size_threshold,
        skip_filtering=args.skip_filtering,
        additional_cols=additional_cols,
        additional_callsets=additional_callsets,
        limit_to_callsets=limit_to_callsets,
        verbose=args.verbose
    )
