"""
Functions for handling "event" notation dataframes
Events include both simple events used in VCF format: DEL, DUP, INV, INS,
and complex event combinations described in
"""

import numpy as np
import pandas as pd

from . import utils

drop_cols = ['INFO_CHR2', 'INFO_CIEND_l', 'INFO_CIEND_r']

def convert_del_to_bnd(del_vcf_df: pd.DataFrame) -> pd.DataFrame:
    """
    Converts deletion records to breakend notation.

    Each deletion is converted to two breakends:
    - bp1: CHROM:POS N[CHR2:POS2[ (encoding=1)
    - bp2: CHR2:POS2 ]CHROM:POS]N (encoding=2)

    Args:
        del_vcf_df: DataFrame with deletion records

    Returns:
        DataFrame containing breakend notation entries
    """
    # Create first breakend
    bp1 = del_vcf_df.copy()
    bp1.loc[:, 'ID'] = bp1['ID'] + '_bp1'
    bp1.loc[:, 'CHROM'] = bp1['CHROM']
    bp1.loc[:, 'POS'] = bp1['POS'] - 1
    bp1.loc[:, 'INFO_MATEID'] = bp1['INFO_EVENT'] + '_bp2'
    bp1.loc[:, 'ALT'] = 'N[' + bp1['INFO_CHR2'].astype(str) + ':' + (bp1['POS'].astype(int) + bp1['INFO_SVLEN'].astype(int) + 1).astype(str) + '['
    bp1.loc[:, 'INFO_CIPOS_l'] = bp1.loc[:, 'INFO_CIEND_l']
    bp1.loc[:, 'INFO_CIPOS_r'] = bp1.loc[:, 'INFO_CIEND_r']
    bp1 = bp1.drop(columns=drop_cols)
    # Create second breakend
    bp2 = del_vcf_df.copy()
    bp2.loc[:, 'ID'] = bp2['ID'] + '_bp2'
    bp2.loc[:, 'CHROM'] = bp2['INFO_CHR2']
    bp2.loc[:, 'POS'] = bp2['POS'].astype(int) + bp2['INFO_SVLEN'].astype(int) + 1
    bp2.loc[:, 'INFO_MATEID'] = bp2['INFO_EVENT'] + '_bp1'
    bp2.loc[:, 'ALT'] = ']' + bp2['CHROM'].astype(str) + ':' + (bp2['POS'].astype(int) - 1).astype(str) + ']N'
    bp2.loc[:, 'INFO_CIPOS_l'] = bp2.loc[:, 'INFO_CIEND_l']
    bp2.loc[:, 'INFO_CIPOS_r'] = bp2.loc[:, 'INFO_CIEND_r']
    bp2 = bp2.drop(columns=drop_cols)
    return pd.concat([bp1, bp2], axis=0)


def convert_inv_to_bnd(inv_vcf_df: pd.DataFrame) -> pd.DataFrame:
    """
    Converts inversion records to breakend notation.

    Each inversion is converted to four breakends:
    - bp1: CHROM:POS-1 N]CHR2:POS2] (encoding=3)
    - bp2: CHROM:POS [CHR2:POS2+1[N (encoding=4)
    - bp3: CHR2:POS2 N]CHROM:POS-1] (encoding=3)
    - bp4: CHR2:POS2+1 [CHROM:POS[N (encoding=4)

    Args:
        inv_vcf_df: DataFrame with inversion records

    Returns:
        DataFrame containing breakend notation entries
    """
    # Create first breakend
    bp1 = inv_vcf_df.copy()
    bp1.loc[:, 'ID'] = bp1['ID'] + '_bp1'
    bp1.loc[:, 'CHROM'] = bp1['CHROM']
    bp1.loc[:, 'POS'] = bp1['POS']
    bp1.loc[:, 'INFO_EVENT'] = bp1['INFO_EVENT'] + '_e1'
    bp1.loc[:, 'INFO_MATEID'] = bp1['INFO_EVENT'] + '_bp3'
    bp1.loc[:, 'ALT'] = 'N]' + bp1['INFO_CHR2'].astype(str) + ':' + (bp1['POS'].astype(int) + bp1['INFO_SVLEN'].astype(int)).astype(str) + ']'
    bp1.loc[:, 'INFO_CIPOS_l'] = bp1.loc[:, 'INFO_CIEND_l']
    bp1.loc[:, 'INFO_CIPOS_r'] = bp1.loc[:, 'INFO_CIEND_r']
    bp1 = bp1.drop(columns=drop_cols)

    # Create second breakend
    bp2 = inv_vcf_df.copy()
    bp2.loc[:, 'ID'] = bp2['ID'] + '_bp2'
    bp2.loc[:, 'CHROM'] = bp2['CHROM']
    bp2.loc[:, 'POS'] = bp2['POS'] + 1
    bp2.loc[:, 'INFO_EVENT'] = bp2['INFO_EVENT'] + '_e2'
    bp2.loc[:, 'INFO_MATEID'] = bp2['INFO_EVENT'] + '_bp4'
    bp2.loc[:, 'ALT'] = '[' + bp2['INFO_CHR2'].astype(str) + ':' + (bp2['POS'].astype(int) + bp2['INFO_SVLEN'].astype(int) + 1).astype(str) + '[N'
    bp2.loc[:, 'INFO_CIPOS_l'] = bp2.loc[:, 'INFO_CIEND_l']
    bp2.loc[:, 'INFO_CIPOS_r'] = bp2.loc[:, 'INFO_CIEND_r']
    bp2 = bp2.drop(columns=drop_cols)
    # Create third breakend
    bp3 = inv_vcf_df.copy()
    bp3.loc[:, 'ID'] = bp3['ID'] + '_bp3'
    bp3.loc[:, 'CHROM'] = bp3['INFO_CHR2']
    bp3.loc[:, 'POS'] = bp3['POS'].astype(int) + bp3['INFO_SVLEN'].astype(int)
    bp3.loc[:, 'INFO_EVENT'] = bp3['INFO_EVENT'] + '_e1'
    bp3.loc[:, 'INFO_MATEID'] = bp3['INFO_EVENT'] + '_bp1'
    bp3.loc[:, 'ALT'] = 'N]' + bp3['CHROM'].astype(str) + ':' + (bp3['POS'].astype(int)).astype(str) + ']'
    bp3.loc[:, 'INFO_CIPOS_l'] = bp3.loc[:, 'INFO_CIEND_l']
    bp3.loc[:, 'INFO_CIPOS_r'] = bp3.loc[:, 'INFO_CIEND_r']
    bp3 = bp3.drop(columns=drop_cols)
    # Create fourth breakend
    bp4 = inv_vcf_df.copy()
    bp4.loc[:, 'ID'] = bp4['ID'] + '_bp4'
    bp4.loc[:, 'CHROM'] = bp4['INFO_CHR2']
    bp4.loc[:, 'POS'] = bp4['POS'].astype(int) + bp4['INFO_SVLEN'].astype(int) + 1
    bp4.loc[:, 'INFO_EVENT'] = bp4['INFO_EVENT'] + '_e2'
    bp4.loc[:, 'INFO_MATEID'] = bp4['INFO_EVENT'] + '_bp2'
    bp4.loc[:, 'ALT'] = '[' + bp4['CHROM'].astype(str) + ':' + (bp4['POS'].astype(int) + 1).astype(str) + '[N'
    bp4.loc[:, 'INFO_CIPOS_l'] = bp4.loc[:, 'INFO_CIEND_l']
    bp4.loc[:, 'INFO_CIPOS_r'] = bp4.loc[:, 'INFO_CIEND_r']
    bp4 = bp4.drop(columns=drop_cols)

    return pd.concat([bp1, bp2, bp3, bp4], axis=0)

def convert_dup_to_bnd(dup_vcf_df: pd.DataFrame) -> pd.DataFrame:
    """
    Converts duplication records to breakend notation.

    Each duplication is converted to two breakends:
    - bp1: CHR2:POS2 N[CHROM:POS[ (encoding=1)
    - bp2: CHROM:POS ]CHR2:POS2]N (encoding=2)

    Args:
        dup_vcf_df: DataFrame with duplication records

    Returns:
        DataFrame containing breakend notation entries
    """
    # Create first breakend
    bp1 = dup_vcf_df.copy()
    bp1.loc[:, 'ID'] = bp1['ID'] + '_bp1'
    bp1.loc[:, 'CHROM'] = bp1['INFO_CHR2']
    bp1.loc[:, 'POS'] = bp1['POS'].astype(int) + bp1['INFO_SVLEN'].astype(int)
    bp1.loc[:, 'INFO_MATEID'] = bp1['INFO_EVENT'] + '_bp2'
    bp1.loc[:, 'ALT'] = 'N[' + bp1['CHROM'].astype(str) + ':' + bp1['POS'].astype(str) + '['
    bp1.loc[:, 'INFO_CIPOS_l'] = bp1.loc[:, 'INFO_CIEND_l']
    bp1.loc[:, 'INFO_CIPOS_r'] = bp1.loc[:, 'INFO_CIEND_r']
    bp1 = bp1.drop(columns=drop_cols)

    # Create second breakend
    bp2 = dup_vcf_df.copy()
    bp2.loc[:, 'ID'] = bp2['ID'] + '_bp2'
    bp2.loc[:, 'CHROM'] = bp2['CHROM']
    bp2.loc[:, 'POS'] = bp2['POS']
    bp2.loc[:, 'INFO_MATEID'] = bp2['INFO_EVENT'] + '_bp1'
    bp2.loc[:, 'ALT'] = ']' + bp2['INFO_CHR2'].astype(str) + ':' + (bp2['POS'].astype(int) + bp2['INFO_SVLEN'].astype(int)).astype(str) + ']N'
    bp2.loc[:, 'INFO_CIPOS_l'] = bp2.loc[:, 'INFO_CIEND_l']
    bp2.loc[:, 'INFO_CIPOS_r'] = bp2.loc[:, 'INFO_CIEND_r']
    bp2 = bp2.drop(columns=drop_cols)

    return pd.concat([bp1, bp2], axis=0)

def convert_ins_to_bnd(ins_vcf_df: pd.DataFrame) -> pd.DataFrame:
    """
    Converts insertion records to breakend notation.

    Each insertion is converted to two breakends:
    - bp1: CHROM:POS N[CHROM:POS[ (encoding=1)
    - bp2: CHROM:POS ]CHROM:POS]N (encoding=2)

    Args:
        ins_vcf_df: DataFrame with insertion records

    Returns:
        DataFrame containing breakend notation entries
    """
    # Create first breakend
    bp1 = ins_vcf_df.copy()
    bp1.loc[:, 'ID'] = bp1['ID'] + '_bp1'
    bp1.loc[:, 'CHROM'] = bp1['CHROM']
    bp1.loc[:, 'POS'] = bp1['POS']
    bp1.loc[:, 'INFO_MATEID'] = bp1['INFO_EVENT'] + '_bp2'
    bp1.loc[:, 'ALT'] = 'N[' + bp1['CHROM'].astype(str) + ':' + (bp1['POS'].astype(int) + 1).astype(str) + '['
    bp1.loc[:, 'INFO_CIPOS_l'] = bp1.loc[:, 'INFO_CIEND_l']
    bp1.loc[:, 'INFO_CIPOS_r'] = bp1.loc[:, 'INFO_CIEND_r']
    bp1 = bp1.drop(columns=drop_cols)

    # Create second breakend
    bp2 = ins_vcf_df.copy()
    bp2.loc[:, 'ID'] = bp2['ID'] + '_bp2'
    bp2.loc[:, 'CHROM'] = bp2['CHROM']
    bp2.loc[:, 'POS'] = bp2['POS'].astype(int) + 1
    bp2.loc[:, 'INFO_MATEID'] = bp2['INFO_EVENT'] + '_bp1'
    bp2.loc[:, 'ALT'] = ']' + bp2['CHROM'].astype(str) + ':' + (bp2['POS'].astype(int)).astype(str) + ']N'
    bp2.loc[:, 'INFO_CIPOS_l'] = bp2.loc[:, 'INFO_CIEND_l']
    bp2.loc[:, 'INFO_CIPOS_r'] = bp2.loc[:, 'INFO_CIEND_r']
    bp2 = bp2.drop(columns=drop_cols)

    return pd.concat([bp1, bp2], axis=0)

svtype_dispatch = {
    'DEL': convert_del_to_bnd,
    'DUP': convert_dup_to_bnd,
    'INV': convert_inv_to_bnd,
    'INS': convert_ins_to_bnd
}

def event_to_bnd(vcf_df: pd.DataFrame) -> pd.DataFrame:
    """
    Converts structural variant events to breakend notation.

    Handles DEL, DUP, INV and INS event types by converting each to appropriate
    breakend representation.

    Args:
        vcf_df: DataFrame with SV events

    Returns:
        DataFrame with breakend records
    """
    # Check if all required columns are present
    if not all(col in vcf_df.columns for col in ['CHROM', 'POS', 'INFO_SVTYPE', 'INFO_SVLEN', 'INFO_CIPOS_l', 'INFO_CIPOS_r', 'INFO_CIEND_l', 'INFO_CIEND_r']):
        raise ValueError("CHROM, POS, INFO_SVTYPE, INFO_SVLEN, INFO_CIPOS_l, INFO_CIPOS_r, INFO_CIEND_l, and INFO_CIEND_r columns are required")

    vcf_df_ = vcf_df.copy()
    # Check mandatory column dtypes:
    mandatory_types = {
        'CHROM': 'object',
        'POS': 'int',
        'INFO_SVTYPE': 'object',
        'INFO_SVLEN': 'int',
        'INFO_CIPOS_l': 'int',
        'INFO_CIPOS_r': 'int',
        'INFO_CIEND_l': 'int',
        'INFO_CIEND_r': 'int'
    }
    if not all(vcf_df_[col].dtype == mandatory_types[col] for col in mandatory_types):
        try:
            for col, dtype in mandatory_types.items():
                vcf_df_[col].astype(dtype)
        except ValueError as e:
            types = {col: vcf_df_[col].dtype for col in mandatory_types}
            raise ValueError(f"Column dtypes do not match the required types, types: {types}, required: {mandatory_types}") from e


    vcf_df_ = vcf_df_[~vcf_df_['CHROM'].isna()]
    vcf_df_ = vcf_df_.sort_values(['CHROM', 'POS'], key=utils.get_chrom_keys)

    # Split by SV type
    svtype_dict = {
        'DEL': vcf_df_[vcf_df_['INFO_SVTYPE'] == 'DEL'],
        'DUP': vcf_df_[vcf_df_['INFO_SVTYPE'] == 'DUP'],
        'INV': vcf_df_[vcf_df_['INFO_SVTYPE'] == 'INV'],
        'INS': vcf_df_[vcf_df_['INFO_SVTYPE'] == 'INS']
    }

    # Convert each type
    bnd_dfs = []
    for svtype, svtype_df in svtype_dict.items():
        if len(svtype_df) > 0:
            bnd_dfs.append(svtype_dispatch[svtype](svtype_df))

    if not bnd_dfs:
        print('No breakends found, returning empty DataFrame')
        return pd.DataFrame()

    return pd.concat(bnd_dfs, axis=0)

def get_insert_seq_len_from_svlen(bp_df: pd.DataFrame):
    """
    Calculates insert sequence length from SVLEN column
    Requires INFO_SVLEN and INFO_SVTYPE columns to be present in the dataframe

    Insert seq len is nonzero only for INS and DUPs
    """
    return np.where(bp_df['INFO_SVTYPE'].isin(['INS']), bp_df['INFO_SVLEN'], 0)
