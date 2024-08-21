#!/home/users/allstaff/shikov.v/.conda/envs/SVEnsemble2/bin/python

#SBATCH --job-name=truth_set_gen
#SBATCH --mem=100GB
#SBATCH --cpus-per-task=2
#SBATCH --time=24:00:00

import os
import sys
sys.path.append('/home/users/allstaff/shikov.v/vast/SVensemble2/truth_set_gen/scripts')

import time
import re
import copy
import numpy as np
import pandas as pd
import matplotlib as mpl
import VCF_parser

def split_vcf_by_chrom(vcf_df, chrom_list=None):
    if chrom_list is not None:
        return {chrom: vcf_df.loc[vcf_df.CHROM == chrom] \
                for chrom in chrom_list}
    return {chrom: gnomad_vcf_df.loc[gnomad_vcf_df.CHROM == chrom] \
                for chrom in pd.unique(vcf_df.CHROM)}


def generate_truth_list(vcf_df, ref_df_dict, pos_interval=25, report_freq=100000):

    truth_list = []
    pos_series = vcf_df.POS
    chrom_series = vcf_df.CHROM
    
    if set(ref_df_dict.keys()) != set(chrom_series.unique()):
        raise KeyError('Chromosomes in prediction dataset do not match chromosomes in gnomad')
    
    for i, (pos, chrom) in enumerate(zip(pos_series, chrom_series)):
        # print(f'{chrom}: {pos}')
        # overlapping_ref_positions = (gnomad_df[chrom].POS > pos - 25) & (gnomad_df[chrom].POS < pos + 25)
    
        truth_list.append(
            ((ref_df_dict[chrom].POS > pos - pos_interval) & (ref_df_dict[chrom].POS < pos + pos_interval)).any()
        )
        
        if i % report_freq == 0:
            print(f'processed {i} entries')

    return truth_list


# Global vars
wd = '/stornext/Bioinf/data/bioinf-data/Papenfuss_lab_projects/SVensemble2/vcf_parsing'
gridss_path = f'{wd}/data/test_gridss_output.NA12878.vcf'
gnomad_path = '/home/users/allstaff/shikov.v/projects/SVensemble2/data/gnomad/4.1.0/downloads/gnomad.v4.1.sv.sites.vcf.gz'

gnomad_vcf_df, gnomad_meta = VCF_parser.parse_vcf(gnomad_path, output_meta=True)
gridss_vcf_df, gridss_meta = VCF_parser.parse_vcf(gridss_path, output_meta=True)

# gnomad_df_copy = gnomad_df.copy(deep=True)
# gnomad_meta_copy = copy.deepcopy(gnomad_meta)
gnomad_vcf_df, gnomad_meta = VCF_parser.get_processed_algorithm_df_and_meta(gnomad_vcf_df, gnomad_meta)
VCF_parser.delete_mult_val_cols(gnomad_vcf_df, gnomad_meta)
VCF_parser.convert_col_types(gnomad_vcf_df, gnomad_meta)

VCF_parser.delete_mult_val_cols(gridss_vcf_df, gridss_meta)
VCF_parser.convert_col_types(gridss_vcf_df, gridss_meta)

gnomad_vcf_df = split_vcf_by_chrom(gnomad_vcf_df, pd.unique(gridss_vcf_df.CHROM))

truth_list = generate_truth_list(gridss_vcf_df, gnomad_vcf_df)
gridss_vcf_df['MATCH_FOUND'] = truth_list

gnomad_vcf_df = pd.concat(gnomad_vcf_df, ignore_index=True)

gnomad_vcf_df = gnomad_vcf_df.sort_values(by=['POS'])
gridss_vcf_df = gridss_vcf_df.sort_values(by=['POS'])

truth_suffix = '_truth_set'
gnomad_vcf_df = gnomad_vcf_df.add_suffix(truth_suffix)

gridss_vcf_df = pd.concat(
    [
        gridss_vcf_df.loc[~gridss_vcf_df['MATCH_FOUND']],
        pd.merge_asof(
            gridss_vcf_df.loc[gridss_vcf_df['MATCH_FOUND']],
            gnomad_vcf_df,
            left_on='POS', right_on=f'POS{truth_suffix}',
            left_by='CHROM', right_by=f'CHROM{truth_suffix}',
            direction='nearest'
        )
    ]
)

cols_to_drop = [f'CHROM{truth_suffix}', f'POS{truth_suffix}', f'REF{truth_suffix}']
gridss_vcf_df = gridss_vcf_df[gridss_vcf_df.columns.drop(cols_to_drop)]

gridss_vcf_df = pd.concat(
    [
        gridss_vcf_df[['CHROM', 'ID', 'REF']],
        gridss_vcf_df.select_dtypes(include=np.number),
        gridss_vcf_df[['MATCH_FOUND']],
        gridss_vcf_df.filter(regex=f'INFO_ALGORITHMS{truth_suffix}')
    ],
    axis=1
)

gridss_vcf_df.to_csv('~/projects/SVensemble2/truth_set_gen/saved/gridss_vcf_df.filtered.tsv', sep='\t')
