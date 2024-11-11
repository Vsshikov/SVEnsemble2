import numpy as np
import pandas as pd

def find_brkend_diff_cols(vcf_df, exceptions=None):
    """
    Groups all entries by event id (INFO_EVENT) and returns an 1 x n Boolean array with n = number of columns.
    Each entry in the array is True if in all breakend pairs column is the same for both. False if not.
    """

    # Check if df is grouped. If it is not, group it by event id
    vcf_df = vcf_df.groupby('INFO_EVENT')



    def unique_cols(df):
        """
        Returns 1 x n Boolean array with n = num of cols in df. 
        Entries are True if column is the same across all rows, False if otherwise
        """
        df_array = df.to_numpy() # df.values (pandas<0.24)
        return (df_array[0] == df_array).all(0)

    # Iterate over all rows to check if there are any diffs between paired breakends in any cols.
    diff_col_list = []
    for event, group_df in vcf_df:
        
        if len(group_df) != 2:
            continue

        if group_df.reset_index()['POS'][0] == group_df.reset_index()['POS'][1]:
            continue
            
        diff_col_list.append(unique_cols(group_df))
    
    return np.logical_and.reduce(pd.DataFrame(diff_col_list))

def get_cols_for_retaining(vcf_df, exceptions='default'):
    """
    Used to decide which columns to retain before merging.
    Columns for retaining:
    - any column that is the same for both breakends in each breakpoint (defined by find_brkend_diff_cols())
    - columns from exceptions argument. If 'defauls', then includes ['ALT', 'CHROM', 'POS', 'ID', 'REF',  'FILTER_PASS']
    """
    
    if exceptions == 'default':
        exceptions = ['ALT', 'CHROM', 'POS', 'ID', 'REF', 'FILTER_PASS']
        exception_regex = '|'.join(exceptions)
    else:
        exception_regex = ''

    exc_cols = vcf_df.columns.str.fullmatch(exception_regex)

    brkend_diff_cols = find_brkend_diff_cols(vcf_df)

    return exc_cols | brkend_diff_cols

def get_group_dfs(vcf_df):
    """
    IMPORTANT - NEEDS A DF SORTED BY CHROM AND POS
    Creates 2 new dfs from the original vcf_df.
    Allocates each breakend in a pair into separate dfs
    Returns a tuple with 2 separate breakend dataframes.
    """
    group_list1 = []
    group_list2 = []
    for event, group_df in vcf_df.groupby('INFO_EVENT'):
        # print(group[1])
        # break
        if len(group_df) != 2:
            continue
        # print(group_df)
    
        # print([group_df, higher_qual_idx, lower_qual_idx])
        # print([higher_qual_idx, lower_qual_idx])
        bnd1 = group_df.iloc[0]
        bnd2 = group_df.iloc[1]

        # try:
        #     bnd1['FILTER_PASS'], bnd2['FILTER_PASS']
        # except KeyError:
        #     continue
        
        group_list1.append(bnd1)
        group_list2.append(bnd2)

    return pd.DataFrame(group_list1).reset_index(drop=True), pd.DataFrame(group_list2).reset_index(drop=True)

def merge_group_lists(group_dfs, inplace=False):
    """
    THIS FUNCTION IS ONLY OPTIMIZED FOR DELS - REVIEW LATER
    SHOULD ALSO REDO THIS TO DO CONCAT -> RENAMING/MUTATING INSTEAD.
    Merges 2 vcf_dfs with corresponding breakends and brings the resulting vcf_df into event format
    """
    if not inplace:
        df1_copy, df2_copy = (group_dfs[0].copy(), group_dfs[1].copy())

    # print(df1_copy, df2_copy)

    df1_copy = df1_copy.rename(columns={'ALT': 'LALT'})
    df2_copy = df2_copy.rename(columns={'ALT': 'RALT', 'POS': 'INFO_END', 'CHROM': 'CHR2', 'FILTER_PASS': 'FILTER_PASS_RIGHT'})

    # filter_cols2 = [col for col in df2_copy.columns if col.startswith('FILTER_') and not col.startswith('FILTER_PASS')]
    # print(np.logical_and.reduce(df2_copy[filter_cols2] == 1, axis=0))
    # df2_copy.iloc[:, 'FILTER_PASS_RIGHT'] = np.logical_and.reduce(df2_copy[filter_cols2] == 1, axis=0)
    
    df1_copy['REF'] = 'N'
    df1_copy['POS'] = df1_copy['POS'] + 1
    df1_copy['ID'] = df1_copy['INFO_EVENT']

    df2_copy.loc[:, 'INFO_END'] = df2_copy['INFO_END'] - 1

    # print(group_dfs[1])
    # cols_to_drop = (
    #     [],
    #     ['REF', 'ID', 'POS']
    # )
    combined = pd.concat([df1_copy, df2_copy[['INFO_END', 'RALT', 'CHR2', 'FILTER_PASS_RIGHT']]], axis=1)
    combined.loc[:, 'FILTER_PASS'] = combined.FILTER_PASS & combined.FILTER_PASS_RIGHT
    combined = combined.drop(columns=['FILTER_PASS_RIGHT'])
    
    return combined

def bnd_to_event(vcf_df, vocal=False):
    """
    Converts a dataframe with breakends only to dataframe with events.
    CURRENTLY ONLY WORKS WITH DELS
    """
    vcf_df_copy = vcf_df.sort_values(by=['CHROM', 'POS'])
    
    vcf_df_copy = vcf_df_copy[vcf_df_copy['ALT'].str.contains('|'.join(['\\[', '\\]']))]
    
    cols_for_retaining = get_cols_for_retaining(vcf_df_copy)

    group_dfs = get_group_dfs(vcf_df_copy.loc[:, cols_for_retaining])
    del vcf_df_copy

    group_df_total = merge_group_lists(group_dfs)

    group_df_total = group_df_total[
        (group_df_total['CHROM'] == group_df_total['CHR2']) & \
    
        (
    # 2 Ways a DEL could be recorded:
            (
                (group_df_total['RALT'].str.strip('ATGCNatgcn').str[0] == '[') & \
                (group_df_total['LALT'].str.strip('ATGCNatgcn').str[0] == ']') & \
                (group_df_total['INFO_END'] < group_df_total['POS'])
            ) | \
            (
                (group_df_total['LALT'].str.strip('ATGCNatgcn').str[0] == '[') & \
                (group_df_total['RALT'].str.strip('ATGCNatgcn').str[0] == ']') & \
                (group_df_total['POS'] < group_df_total['INFO_END'])
            )
        )
    ]

    group_df_total = group_df_total[
        (
            group_df_total['LALT'].str.contains('^[ATGCNatgcn]{1}\\[', ) & \
            group_df_total['RALT'].str.contains('\\][ATGCNatgcn]$', )
        ) | \
        (
            group_df_total['RALT'].str.contains('^[ATGCNatgcn]{1}\\[', ) & \
            group_df_total['LALT'].str.contains('\\][ATGCNatgcn]$', )
        )
    ]
    
    group_df_total.loc[:, 'REF'] = 'N'
    group_df_total.loc[:, 'ALT'] = '<DEL>'
    group_df_total.loc[:, 'INFO_SVTYPE'] = 'DEL'

    return group_df_total
