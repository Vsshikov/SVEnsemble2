from matplotlib import pyplot as plt
import pandas as pd
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve
import numpy as np
from typing import Dict, Any

from . import breakends as bnds
from . import events as evts
from . import utils as sv_utils
from . import _caller_params as caller_params
from . import _sample_params as sample_params
from . import vcfparse as vcfp
from pathlib import Path
import pysam

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

def plot_roc_curves(sample_val_dict, ax=None, use_counts=True, filter_pass=False, title=None):
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
    n_callsets = len(callsets)
    colors = get_plot_colors(n_callsets)

    for idx, callset in enumerate(callsets):
        callset_val_df = sample_val_dict[callset].dropna().copy()

        if filter_pass and 'FILTER_PASS' in sample_val_dict[callset].columns:
            sample_mask = callset_val_df['FILTER_PASS'].astype(bool)
        elif filter_pass:
            print(f'{callset} does not have a FILTER_PASS column, ignoring filter_pass argument')
            sample_mask = np.ones(len(callset_val_df), dtype=bool)
        else:
            sample_mask = np.ones(len(callset_val_df), dtype=bool)

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
            tpr = tpr * total_tp
            fpr = fpr * total_tn

        ax.plot(fpr, tpr, label=callset, color=colors[idx])

    # Add a dotted diagonal line from (0, 0) to (total_tn, total_tp) if use_counts, else from (0, 0) to (1, 1)
    # if use_counts:
    #     ax.plot([0, total_tn], [0, total_tp], linestyle='--', color='grey', linewidth=1, label='Random')
    # else:
    #     ax.plot([0, 1], [0, 1], linestyle='--', color='grey', linewidth=1, label='Random')

    ax.legend(loc=(1.1, 0.5))

    xlabel = 'False Positives' if use_counts else 'False Positive Rate'
    ylabel = 'True Positives' if use_counts else 'True Positive Rate'

    plt.xlabel(xlabel, size='large', weight=700)
    plt.ylabel(ylabel, size='large', weight=700)

    return ax


def plot_reverse_roc_curves(sample_val_dict, ax=None, use_counts=True, filter_pass=False, title=None):
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
    n_callsets = len(callsets)
    colors = get_plot_colors(n_callsets)

    for idx, callset in enumerate(callsets):
        callset_val_df = sample_val_dict[callset].dropna().copy()

        if filter_pass and 'FILTER_PASS' in sample_val_dict[callset].columns:
            sample_mask = callset_val_df['FILTER_PASS'].astype(bool)
        elif filter_pass:
            print(f'{callset} does not have a FILTER_PASS column, ignoring filter_pass argument')
            sample_mask = np.ones(len(callset_val_df), dtype=bool)
        else:
            sample_mask = np.ones(len(callset_val_df), dtype=bool)

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

            reverse_fp = 1/fp_no_zeros
            tp_fp = tp_no_zeros/fp_no_zeros

            ax.plot(reverse_fp, tp_fp, label=callset, color=colors[idx])
        else:
            ax.plot(fpr, tpr, label=callset, color=colors[idx])

    # Add a dotted diagonal line from (0, 0) to (total_tn, total_tp) if use_counts, else from (0, 0) to (1, 1)
    # if use_counts:
    #     ax.plot([0, total_tn], [0, total_tp], linestyle='--', color='grey', linewidth=1, label='Random')
    # else:
    #     ax.plot([0, 1], [0, 1], linestyle='--', color='grey', linewidth=1, label='Random')

    ax.legend(loc=(1.1, 0.5))

    xlabel = '1/FP'
    ylabel = 'TP/FP'

    plt.xlabel(xlabel, size='large', weight=700)
    plt.ylabel(ylabel, size='large', weight=700)

    return ax


def plot_precision_recall_curves(sample_val_dict, ax=None, use_counts=True):
    """
    Plot precision-recall curves for multiple prediction arrays against corresponding truth series

    Args:
        sample_val_dict: Dictionary containing callset names as keys and dataframes with a prediction metric as 2nd column and a truth column as "is_in_val"
        ax: matplotlib axis to plot the precision-recall curves on
        use_counts: Whether to plot the precision-recall curves as counts or rates

    Returns:
        matplotlib axis with plotted precision-recall curves
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(4,4))

    for callset in sample_val_dict.keys():
        callset_val_df = sample_val_dict[callset].dropna()
        truth_series = callset_val_df['is_in_val']
        pred_series = callset_val_df.iloc[:, 1]
        precision, recall, thresholds = precision_recall_curve(truth_series, pred_series, pos_label=True)

        ax.scatter(recall, precision, label=callset, s=[3]*len(recall))

    ax.legend(loc=(1.1, 0.5))

    xlabel = 'Recall'
    ylabel = 'Precision'

    plt.xlabel(xlabel, size='large', weight=700)
    plt.ylabel(ylabel, size='large', weight=700)

    return ax


def preprocess_hg002_val_set(sample, vcf_path, force_rerun=False, long_val=True):
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


def preprocess_kosugi_val_set(sample, vcf_path, force_rerun=False, long_val=True):
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
        result_bps = bnds.bnd_to_bp(evts.event_to_bnd(chm_val_vcf_df))

        result_bps.to_csv(out_path, index=False)


