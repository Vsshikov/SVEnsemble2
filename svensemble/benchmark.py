from matplotlib import pyplot as plt
import pandas as pd
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve
import numpy as np

from . import breakends as bnds
from . import events as evts

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

    for callset in sample_val_dict.keys():
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
            ax.plot(fp, tp, label=callset)
        else:
            ax.plot(fpr, tpr, label=callset)

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

    for callset in sample_val_dict.keys():
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

            ax.plot(reverse_fp, tp_fp, label=callset)
        else:
            ax.plot(fpr, tpr, label=callset)

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
