import os
import time
import re
import numpy as np
import pandas as pd
import matplotlib as mpl

# Utilities

def get_colnames_after_named_col(vcf_df, col_name='FORMAT'):
    """
    Accepts vcf_df as argument and returns all names of columns after the FORMAT field as a list
    """
    cols_after_named_col = []
    after_named_col = False
    for column in vcf_df.columns:
        if column == col_name:
            after_named_col = True
            continue
        if after_named_col:
            cols_after_named_col.append(column)
    return cols_after_named_col


def iter_joined_combs(iter1, iter2, delim='_'):
    """
    Iterates over the provided 2 iterables and yields each string containing joined iterable items using a provided delimeter
    """
    for i in iter1:
        for j in iter2:
            yield delim.join([i, j])


def get_max_set(series):
    """
    Accepts a series of objects convertible to sets and returns a set with all possible unique items in each iterable object
    """
    max_items = set()
    for itemset in series.apply(lambda x: set(x)):
        max_items.update(itemset)
    return(max_items)


def rename_duplicate_cols(df_to_rename, df_to_compare, suffix='INFO', inplace=False):
    """
    Accepts two dataframes as arguments. Compares column names in df_to_rename to those in df_to_compare.
    If any names overlap, adds a specified suffix to column names in df_to_rename and returns the modified dataframe
    """
    main_df_colnames = df_to_compare.columns
    sec_df_colnames = df_to_rename.columns
    overlapping_names = [name for name in main_df_colnames if name in sec_df_colnames]
    rename_dict = {name: name + suffix for name in overlapping_names}
    if inplace:
        df_to_rename.rename(columns=rename_dict, inplace=True)
    else:
        return df_to_rename.rename(columns=rename_dict)
     
         
# Formatting the vcf_df

def split_format_cols(vcf_df, delim=':'):
    """
    Modifies the input dataframe by splitting the FORMAT column 
    and the following genotype columns using the specified delimeter
    """
    for colname in ['FORMAT'] + get_colnames_after_named_col(vcf_df):
        vcf_df[colname] = vcf_df[colname].str.split(delim)



# Processing the info column

def process_infovalue(infovalue, type=None):
    """
    If possible, converts the value of an INFO field from string format to float, int or a list of strings/floats/ints
    """
    if re.match(r'.*,.*', infovalue):
        return [process_infovalue(infoitem) for infoitem in infovalue.split(',')]
    elif re.match(r'[0-9]*\.[0-9]+', infovalue):
        return float(infovalue)
    elif re.match(r'^[0-9]+$', infovalue):
        return int(infovalue)
    else:
        return infovalue
        

def process_info(info, delim=';', equalsign='=', dictlike=True, fieldnames=None, decimal_delim='.'):
    """
    Takes an input INFO field as string and turns it into a dictionary with INFO IDs as keys.
    """
    info = info.split(delim)

    id_regex = re.compile(r'(.+=)|(?!.*=.*)')
    val_regex = re.compile(r'=.+$')
    info_dict = {}
    for infoentry in info:

        try:
            info_id = re.match(id_regex, infoentry).group().replace('=', '')
        except AttributeError:
            raise AttributeError(f'INFO field entry without an ID')
        try:
            infovalue = process_infovalue(re.search(val_regex, infoentry).group().replace('=', ''))
        except AttributeError:
            # We assume that INFO entries with an ID but without a value are flags, and a presence of a flag would set the value to True
            infovalue = True
        # info_dict_entry = {info_id: infovalue}
        info_dict.update({info_id: infovalue})

    return info_dict


def generate_infodict(infoseries, vcf_row_num=None):
    """
    
    """
    if vcf_row_num == None:
        vcf_row_num = len(infoseries)
    first_infoentry = process_info(infoseries[0])
    infodict = {info_id: [None] * vcf_row_num for info_id in first_infoentry.keys()}
    return infodict


def update_infodict(infodict, info, infodict_index):
    """
    Replaces the placeholder values in the <index> position of infodict lists with values from info. The infodict must have lists in all fields. The lists must all have equal length.
    For each key in the info it searches for a matching key in the infodict.
    If the key is present in infodict, it appends the matching infodict list with the value from the info.
    If the key is not in infodict, it first creates a new infoentry in the infodict (filling it with NAs so it matches other lists in the infodict) and then appends it with value from info
    """
    for info_key in info.keys():
        try:
            infodict[info_key][infodict_index] = info[info_key]
        except KeyError:
            infodict_item_len = len(infodict[list(infodict.keys())[0]])
            infodict.update({info_key: [None]*infodict_item_len})
            infodict[info_key][infodict_index] = info[info_key]


def fillinfodict(infodict, infoseries):
    start_time = time.time()
    for i, infoentry in enumerate(infoseries):
        update_infodict(infodict, process_info(infoentry), i)
        if i%100000 == 0:
            current_time = time.time()
            print(f'Processed {i+1} entries out of {vcf_nrow}. Elapsed time: {current_time - start_time} seconds.')
    current_time = time.time()
    print(f'Done processing {vcf_nrow} entries. Elapsed time: {current_time - start_time} seconds.')
    


# Processing the format column

# format_ids = get_max_set(vcf_df.FORMAT)
def generate_formatdict(format_ids, cols_after_format, vcf_row_num):
    return {format_col_name: [None] * vcf_row_num for format_col_name in iter_joined_combs(format_ids, cols_after_format)}
# formatdict = {format_col_name: [None] * vcf_row_num for format_col_name in iter_joined_combs(format_ids, cols_after_format)}

def format_id_val_gen(vcf_df_row_split):
    """
    
    """
    for i, format_id in enumerate(vcf_df_row_split.FORMAT):
        for colname in cols_after_format:
            yield format_id, vcf_df_row_split[colname][i]

def update_formatdict(formatdict, vcf_row, cols_after_format, formatdict_index):
    """
    
    """
    for genotype_name in cols_after_format:
        for format_id, val in format_id_val_gen(vcf_row):
            formatdict[f"{format_id}_{genotype_name}"][formatdict_index] = val
            

# Joint processing

# Joint processing

def get_infoformatdicts(vcf_df, vocal=False, return_df=False, nrow_limit=0):
    """
    Accepts an vcf loaded as a dataframe and returns a tibble with 2 items:
    - an infodict
    - a formatdict
    !!! IMPORTANT !!! Requires either of 2:
    - no FORMAT and genotype columns in the VCF
    - if FORMAT/genotype columns are present, split_format_cols() must have been applied to the df.
    """
    vcf_row_num = len(vcf_df)
    infodict = generate_infodict(vcf_df.INFO, vcf_row_num)

    try:
        format_present = True
        format_ids = get_max_set(vcf_df.FORMAT)
        cols_after_format = get_colnames_after_named_col(vcf_df, 'FORMAT')
        formatdict = generate_formatdict(format_ids, cols_after_format, vcf_row_num)
    except AttributeError:
        format_present = False
        formatdict = None
        print('Genotype fields not detected, skipping genotype field processing steps')
        
    
    start_time = time.time()
    for i, row in vcf_df.iterrows():
        
        update_infodict(infodict, process_info(row.INFO), i)
        if format_present:
            update_formatdict(formatdict, row, cols_after_format, i)
            
        if i%100000 == 0:
            current_time = time.time()
            print(f'Processed {i+1} entries out of {vcf_row_num}. Elapsed time: {current_time - start_time} seconds.')

        if i + 1 == nrow_limit:
            break

    current_time = time.time()
    print(f'Finished processing. Processed {i+1} entries out of {vcf_row_num}. Elapsed time: {current_time - start_time} seconds.')

    
    if return_df:
        if format_present:
            return (pd.DataFrame(infodict), pd.DataFrame(formatdict))
        else:
            return (pd.DataFrame(infodict), None)
    else:
        return (infodict, formatdict)
    


def process_vcf_cols(vcf_df, nrow_limit=0):

    try:
        cols_after_format = get_colnames_after_named_col(vcf_df, 'FORMAT')
        split_format_cols(vcf_df)
        format_present = True
    except KeyError:
        format_present = False
        print('Genotype fields not detected, skipping genotype field processing steps')

    infodf, formatdf = get_infoformatdicts(vcf_df, return_df=True, nrow_limit=nrow_limit)
    
    rename_duplicate_cols(infodf, vcf_df, suffix='INFO', inplace=True)

    cols_to_drop = ['INFO']
    vcf_df = vcf_df.drop(columns = cols_to_drop).join(infodf)
    del infodf
    
    if format_present:
        formatdf = pd.DataFrame(formatdict)
        rename_duplicate_cols(formatdf, vcf_df, suffix='FORMAT', inplace=True)
        cols_to_drop += ['FORMAT'] + cols_after_format
        vcf_df = vcf_df.join(formatdf)
        del formatdf





