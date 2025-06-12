from matplotlib import pyplot as plt
import pandas as pd
from sklearn.metrics import roc_curve

from . import breakends as bnds
from . import breakpoints as bps

def load_truth_variants(ref_path):
    """Load reference variants from Kosugi study from BED file
    
    Args:
        ref_path: Path to BED file with reference variants
        
    Returns:
        DataFrame with columns:
        - Chromosome, Start, End: Variant coordinates
        - ALT: Variant type (DEL, INS, etc)
        - length: Length of variant
        - is_in_ref: Always True
        - ID: String identifier combining coordinates
    """
    ref_df = pd.read_csv(ref_path, sep='\t', names=['Chromosome', 'Start', 'End', 'ALT'])
    ref_df.loc[:, 'length'] = ref_df['End'] - ref_df['Start']
    ref_df.loc[:, 'is_in_ref'] = True
    ref_df.loc[:, 'ID'] = ref_df['Chromosome'].astype(str) + ':' + ref_df['Start'].astype(str) + '-' + ref_df['End'].astype(str)
    return ref_df

def truth_to_vcf_df(ref_df, ref_match_margin=50):
    """Convert truth variant set from original format to VCF-like DataFrame
    
    Args:
        ref_df_pr: DataFrame with reference variants in Kosugi format
        ref_match_margin: Margin in bp to use for CIPOS fields (default: 50)
        
    Returns:
        DataFrame with VCF-like fields including:
        - CHROM, POS: Variant position 
        - INFO_END: End position
        - INFO_CHR2: Secondary chromosome
        - INFO_EVENT: Variant ID
        - INFO_SVTYPE: Variant type
        - INFO_SVLEN: Variant length
        - INFO_CIPOS_l: Left confidence interval (-ref_match_margin)
        - INFO_CIPOS_r: Right confidence interval (+ref_match_margin)
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
    ref_vcf.loc[:, 'INFO_CIPOS_l'] = -ref_match_margin
    ref_vcf.loc[:, 'INFO_CIPOS_r'] = ref_match_margin
    
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
    ref_vcf_bnds = bnds.event_to_bnd(ref_vcf)
    ref_vcf_bnds.loc[:, 'bnd_pos_encoding'] = bnds.encode_bnd_alts(ref_vcf_bnds['ALT'])
    ref_vcf_bps = bnds.bnd_to_bp(ref_vcf_bnds, force_diff_cols=['CHROM', 'INFO_CIPOS_l', 'INFO_CIPOS_r', 'bnd_pos_encoding'])
    ref_vcf_bps.loc[:, 'bndtype'] = ref_vcf_bps['bnd_pos_encoding_bnd1'].astype(str) + ref_vcf_bps['bnd_pos_encoding_bnd2'].astype(str)
    return ref_vcf_bps


def plot_roc_curves(truth_series, pred_series_dict, ax=None, use_counts=True):
    """Plot ROC curves for multiple prediction arrays against a truth array.
    
    Args:
        truth_series: Series/array of ground truth binary labels
        pred_series_dict: Dict mapping prediction names to their prediction arrays
        ax: Optional matplotlib axis to plot on
        use_counts: If True, plot TP/FP counts instead of rates
    
    Returns:
        matplotlib axis with plotted ROC curves
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(4,4))
        
    for pred_name, pred_series in pred_series_dict.items():
        fpr, tpr, thresholds = roc_curve(truth_series, pred_series, pos_label=True)
        
        if use_counts:
            # Convert rates to counts
            tp = tpr * sum(truth_series)
            fp = fpr * sum(truth_series)
            ax.plot(fp, tp, label=pred_name)
        else:
            ax.plot(fpr, tpr, label=pred_name)
            
    ax.legend(loc=4)
    
    xlabel = 'False Positives' if use_counts else 'False Positive Rate'
    ylabel = 'True Positives' if use_counts else 'True Positive Rate'
    
    plt.xlabel(xlabel, size='large', weight=700)
    plt.ylabel(ylabel, size='large', weight=700)
    
    return ax
