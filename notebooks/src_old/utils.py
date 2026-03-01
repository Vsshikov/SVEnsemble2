import re
import pandas as pd
import copy
from typing import Union
import numpy as np


def combine_meta_dict(meta_dict):
    """
    Accepts a nested meta_dict to unwrap it and return a meta dict where all the fields are mixed
    """
    return {**meta_dict['FILTER'], **meta_dict['INFO'], **meta_dict['FORMAT']}

def split_meta_dict(meta_dict_combined):
    """
    Accepts an unwrapped meta_dict to return it to nested form
    """
    meta_dict = {
        'FILTER': {},
        'INFO': {},
        'FORMAT': {}
    }
    for meta_key, meta_item in meta_dict_combined.items():
        meta_dict[re.match(r'(INFO)|(FORMAT)|(FILTER)', meta_key).group()][meta_key] = meta_item
    
    return meta_dict

def get_split_meta_entry(meta_id, meta_info, values, output_type='Flag'):
    """
    If a meta entry has multiple values, fetches them all and turns them into separate meta entries 
    with name: {meta_id}_{value}

    Returns a dictionary with new ids as keys and tuples of (1, specified output type name, description)
    """
    new_ids = [f'{meta_id}_{value}' for value in values]

    return {new_id: (1, output_type, f'SPLIT FROM {meta_id}, OLD DESC: {meta_info[2]}') for new_id in new_ids}
    

def get_processed_algorithm_df_and_meta(vcf_df: pd.DataFrame, 
                                      meta_dict: dict, 
                                      delim: str = ',', 
                                      alg_col_name: str = 'INFO_ALGORITHMS') -> tuple[pd.DataFrame, dict]:
    """
    Processes the algorithm column in a VCF dataframe by splitting it into separate binary columns for each algorithm.

    Args:
        vcf_df (pd.DataFrame): VCF dataframe containing the algorithm column
        meta_dict (dict): Dictionary containing VCF metadata
        delim (str, optional): Delimiter used to separate algorithms. Defaults to ','.
        alg_col_name (str, optional): Name of the algorithm column. Defaults to 'INFO_ALGORITHMS'.

    Returns:
        tuple: A tuple containing:
            - pd.DataFrame: Modified VCF dataframe with algorithm column split into binary columns
            - dict: Updated metadata dictionary with new algorithm entries
    """
    # algs = vcf_df[alg_col].str.split(delim).astype(set)
    alg_col = vcf_df[alg_col_name]
    alg_set = set()
    
    for uniq_alg_comb in alg_col.unique():
        alg_set = alg_set | set(uniq_alg_comb.split(delim))
        
    alg_dict = {f'{alg_col_name}_{key}': [0] * len(alg_col) for key in alg_set}
    
    for i, alg_item in enumerate(alg_col):
        for alg in alg_item.split(delim):
            alg_dict[f'{alg_col_name}_{alg}'][i] = 1

    meta_dict_copy = copy.deepcopy(meta_dict)
    old_alg_entry = meta_dict_copy['INFO'].pop(alg_col_name)
    new_alg_entry = get_split_meta_entry(alg_col_name, old_alg_entry, alg_set)
    # print(old_alg_entry, new_alg_entry)

    # print(pd.concat([vcf_df[vcf_df.columns.drop([alg_col_name])], pd.DataFrame(alg_dict)], axis=1))
    meta_dict_copy['INFO'].update(new_alg_entry)
    
    return pd.concat([vcf_df[vcf_df.columns.drop([alg_col_name])], pd.DataFrame(alg_dict)], axis=1), meta_dict_copy


def add_vcf_col(vcf_df: pd.DataFrame, meta_dict: dict, col_name: str, col_values: Union[list, pd.Series], 
                field_type: str | None = None, description: str | None = None, vocal: bool = False) -> tuple[pd.DataFrame, dict]:
    """
    Adds a new column to the VCF dataframe and updates the metadata dictionary with the new column
    
    Args:
        vcf_df: DataFrame to add column to
        meta_dict: Metadata dictionary to update
        col_name: Name of new column
        col_values: Values for new column, can be list or pandas Series
        field_type: Type of the field (Flag, Integer, Float, String). If None, inferred from col_values
        description: Description of the field. If None, uses default description
        vocal: Whether to print debug info
        
    Returns:
        tuple: A tuple containing:
            - pd.DataFrame: Modified DataFrame with new column
            - dict: Updated metadata dictionary
            
    Raises:
        ValueError: If values are not of a supported VCF type
    """
    if vocal:
        print("Starting add_vcf_col...")
        print(f"Input vcf_df shape: {vcf_df.shape}")
        print(f"Input col_name: {col_name}")
        print(f"Input col_values type: {type(col_values)}, length: {len(col_values)}")
        print(f"Input field_type: {field_type}")
    
    vcf_df_copy = vcf_df.copy()
    meta_dict_copy = copy.deepcopy(meta_dict)
    if vocal:
        print("Made copies of inputs")
    
    # Validate input
    if not isinstance(col_values, (pd.Series, list)):
        raise ValueError("col_values must be a pandas Series or list")
        
    if len(col_values) == 0:
        raise ValueError("col_values cannot be empty")
    
    if vocal:
        print("Input validation passed")
        
    # Add column to dataframe
    vcf_df_copy[col_name] = col_values
    if vocal:
        print(f"Added column {col_name} to dataframe")
    
    # Determine VCF field from column name prefix
    vcf_field_match = re.match(r'^(INFO|FORMAT|FILTER)_', col_name)
    vcf_field = vcf_field_match.group(1) if vcf_field_match else 'INFO'
    if vocal:
        print(f"Determined VCF field: {vcf_field}")
    
    # Infer field type if not provided
    if field_type is None:
        if vocal:
            print("Inferring field type...")
        sample_val = col_values[0]
        unique_vals = set(col_values)
        if vocal:
            print(f"Sample value: {sample_val}, type: {type(sample_val)}")
            print(f"Number of unique values: {len(unique_vals)}")
        
        if isinstance(sample_val, (bool, np.bool_)) or (isinstance(sample_val, (int, float, np.integer, np.floating)) and unique_vals <= {0, 1}):
            field_type = 'Flag'
        elif isinstance(sample_val, (int, np.integer)) and all(isinstance(x, (int, np.integer)) for x in col_values):
            field_type = 'Integer'  
        elif isinstance(sample_val, (float, np.floating)) and all(isinstance(x, (int, float, np.integer, np.floating)) for x in col_values):
            field_type = 'Float'
        elif all(isinstance(x, str) for x in col_values):
            field_type = 'String'
        else:
            raise ValueError("Values must all be of type bool, int, float or str. Found mixed or unsupported types.")
        if vocal:
            print(f"Inferred field type: {field_type}")
            
    if description is None:
        description = 'Added programmatically'
        if vocal:
            print("Using default description")
        
    # Update metadata
    if vcf_field not in meta_dict_copy:
        meta_dict_copy[vcf_field] = {}
        if vocal:
            print(f"Created new {vcf_field} dictionary in metadata")
    meta_dict_copy[vcf_field][col_name] = (1, field_type, description)
    if vocal:
        print(f"Updated metadata for {col_name}")
        print(f"Added column {col_name} of type {field_type}")
        
    return vcf_df_copy, meta_dict_copy


def convert_col_types(vcf_df, meta_dict, types_from='meta'):
    """
    Reads the types for INFO and FORMAT fields from the meta_dict and uses them to convert fields in vcf_df into the specified types
    """
    if types_from == 'meta':
        type_convert_dict = {
            'String': str,
            'Integer': int,
            'Float': float,
            'Flag': int
        }
        for meta_id, meta_info in (meta_dict['INFO'] | meta_dict['FORMAT']).items():
            try:
                vcf_df[meta_id] = vcf_df[meta_id].astype(type_convert_dict[meta_info[1]])
            except KeyError:
                print(f'Warning: {meta_id} not found in meta_dict, skipping type conversion')


def delete_mult_val_cols(vcf_df: pd.DataFrame, meta_dict: dict, exceptions: str | list | None = 'default', vocal: bool = False) -> None:
    """
    
    """
    if exceptions == 'default':
        exceptions = ['INFO_ALGORITHMS']
    if exceptions is None:
        exceptions = []
    
    keys_for_deletion = []
    for meta_group in meta_dict.values():
        for meta_id, meta_info in meta_group.items():
            # print(meta_id, meta_info)
            if meta_info[0] not in ['1', 1] + exceptions:
                keys_for_deletion.append(meta_id)

    if vocal:
        print(f'Dropping entries {keys_for_deletion} since they contain multiple values')

    for key_for_deletion in keys_for_deletion:
        del meta_dict[re.match(r'(INFO)|(FORMAT)|(FILTER)', key_for_deletion).group()][key_for_deletion]
    
    vcf_df.drop(keys_for_deletion, axis=1, inplace=True)


def split_evidence_col(
    vcf_df: pd.DataFrame,
    meta_dict: dict,
    evidence_col_name: str = 'INFO_EVIDENCE',
    evidence_delim: str = ',',
    evidence_types: list[str] = ['SR', 'PE', 'RD']
) -> None:
    """
    Creates binary indicator columns for each evidence type from a comma-separated evidence field.
    For each evidence type in evidence_types, adds a new column that is 1 if that evidence type is present, 0 if not.
    Updates the meta_dict to reflect the new binary columns.

    Args:
        vcf_df: DataFrame containing VCF data
        meta_dict: Dictionary containing VCF metadata
        evidence_col_name: Name of column containing evidence information
        evidence_delim: Delimiter used to separate evidence types
        evidence_types: List of evidence types to check for
        vocal: Whether to print debug information

    Returns:
        None - modifies vcf_df and meta_dict in place
    """
    evidence_col = vcf_df[evidence_col_name].str.split(evidence_delim)
    for evidence_type in evidence_types:
        vcf_df[f'{evidence_col_name}_{evidence_type}'] = evidence_col.apply(lambda x: 1 if evidence_type in x else 0)
    
    # Add best resolution column by checking evidence types in order
    # For example, if evidence_types = ['SR', 'PE', 'RD'] and we have these evidence combinations:
    # Row 1: SR=1, PE=1, RD=0 -> BEST='SR' (SR is highest resolution available)
    # Row 2: SR=0, PE=1, RD=1 -> BEST='PE' (PE is highest resolution available) 
    # Row 3: SR=0, PE=0, RD=1 -> BEST='RD' (only RD available)
    # Row 4: SR=0, PE=0, RD=0 -> BEST=None (no evidence)
    #
    # For Row 1:
    # First iteration: mask = True (no best set) & SR=1 -> sets BEST='SR'
    # Second iteration: mask = False (best already set) & PE=1 -> no change
    # Third iteration: mask = False (best already set) & RD=0 -> no change
    vcf_df[f'{evidence_col_name}_BEST'] = None
    for evidence_type in evidence_types:
        mask = vcf_df[f'{evidence_col_name}_BEST'].isna() & (vcf_df[f'{evidence_col_name}_{evidence_type}'] == 1)
        vcf_df.loc[mask, f'{evidence_col_name}_BEST'] = evidence_type
    
    meta_dict_copy = copy.deepcopy(meta_dict)
    old_evidence_entry = meta_dict_copy['INFO'].pop(evidence_col_name)
    new_evidence_entry = get_split_meta_entry(evidence_col_name, old_evidence_entry, evidence_types + ['BEST'])
    meta_dict_copy['INFO'].update(new_evidence_entry)
    meta_dict = meta_dict_copy


def encode_bnd_alts(alt_col):
    """Encode breakend ALT fields into numeric categories based on their pattern.
    
    Args:
        alt_col: pandas Series containing ALT fields from VCF
        
    Returns:
        pandas Series with numeric encodings:
        - 1: N[PARTNER[ (partner extending after breakpoint)
        - 2: ]PARTNER]N (partner extending before breakpoint) 
        - 3: N]PARTNER] (reverse complement of partner extending after breakpoint)
        - 4: [PARTNER[N (reverse complement of partner extending before breakpoint)
        - 0: No match
    """
    partner_pattern_right = r'\[\.?c?h?r?[0-9XYM]{0,2}\:?[0-9]{0,}\['
    partner_pattern_left = r'\]\.?c?h?r?[0-9XYM]{0,2}\:?[0-9]{0,}\]'
    nucl_pattern = r'[ATGCNatgcn]+'

    bnd_regex = re.compile(rf'({nucl_pattern}{partner_pattern_right})|({partner_pattern_left}{nucl_pattern})|({nucl_pattern}{partner_pattern_left})|({partner_pattern_right}{nucl_pattern})')

    encoding_df = alt_col.str.extract(bnd_regex)
    encoding_df.columns = [1, 2, 3, 4]
    encoding_df[~encoding_df.isna()] = True
    
    encodings = encoding_df.idxmax(axis=1)
    encodings[encodings.isna()] = 0
    encodings = encodings.astype(int)
    
    return encodings

def get_meta_missing_in_df(vcf_df: pd.DataFrame, meta_dict: dict) -> list:
    """Find metadata fields that are missing from the VCF DataFrame.
    
    Checks which metadata fields from meta_dict are not present as columns
    in the VCF DataFrame.
    
    Args:
        vcf_df: DataFrame containing VCF data
        meta_dict: Dictionary containing VCF metadata
        
    Returns:
        List of metadata field IDs that are missing from vcf_df
    """
    ids_for_removal = []
    for meta_id in combine_meta_dict(meta_dict).keys():
        try:
            vcf_df[meta_id]
        except KeyError:
            ids_for_removal.append(meta_id)
    return ids_for_removal


def trim_meta(vcf_df: pd.DataFrame, meta_dict: dict) -> tuple[pd.DataFrame, dict]:
    """Remove metadata fields that are missing from the VCF DataFrame.
    
    Identifies metadata fields not present in vcf_df and removes them from
    the metadata dictionary to ensure consistency.
    
    Args:
        vcf_df: DataFrame containing VCF data
        meta_dict: Dictionary containing VCF metadata
        
    Returns:
        Tuple containing:
        - Original VCF DataFrame
        - Updated metadata dictionary with missing fields removed
    """
    meta_local = combine_meta_dict(meta_dict)
    missing_meta = get_meta_missing_in_df(vcf_df, meta_dict)
    print(f'Removing items {missing_meta} from meta since they are missing in the vcf_df')
    
    for item in missing_meta:
        del meta_local[item]
        
    meta_dict = split_meta_dict(meta_local)
    return vcf_df, meta_dict

def add_missing_meta(vcf_df: pd.DataFrame, meta_dict: dict) -> tuple[pd.DataFrame, dict]:
    """Add metadata entries for DataFrame columns missing from metadata.
    
    Identifies columns in vcf_df that don't have corresponding metadata entries
    and adds metadata entries with automatically detected types and settings.
    
    Args:
        vcf_df: DataFrame containing VCF data
        meta_dict: Dictionary containing VCF metadata
        
    Returns:
        Tuple containing:
        - Original VCF DataFrame 
        - Updated metadata dictionary with new fields added
    """
    meta_local = combine_meta_dict(meta_dict)
    meta_local_fields = list(meta_local.keys())
    
    # Find columns missing from metadata
    missing_cols = [col for col in vcf_df.columns if col not in meta_local_fields]
            
    if missing_cols:
        print(f'Adding metadata entries for columns: {missing_cols}')
        
        # Add metadata entries with auto-detected types
        for col in missing_cols:
            col_dtype = vcf_df[col].dtype
            
            # Detect number of values
            if vcf_df[col].apply(lambda x: isinstance(x, (list, tuple))).any():
                number = '.'  # Array/variable number of values
            else:
                number = '1'  # Single value
                
            # Map numpy/pandas dtypes to VCF types
            if np.issubdtype(col_dtype, np.integer):
                vcf_type = 'Integer'
            elif np.issubdtype(col_dtype, np.floating):
                vcf_type = 'Float'
            elif col_dtype is bool:
                vcf_type = 'Flag'
            else:
                vcf_type = 'String'
                
            # Add the metadata entry
            meta_local[col] = {
                'Number': number,
                'Type': vcf_type,
                'Description': f'Auto-added field for column {col}. '
                             f'Detected type: {vcf_type}'
            }
    
    meta_dict = split_meta_dict(meta_local)
    return vcf_df, meta_dict


def generate_chrom_key_dict(chrom_series: pd.Series) -> dict:
    """
    Generate a dictionary mapping chromosome names to numeric indices for sorting.
    
    Creates a mapping of standard chromosome names (chr1-22, X, Y, M) plus any 
    non-standard chromosomes found in the input series to sequential indices.
    
    Args:
        chrom_series: Series containing chromosome names
        
    Returns:
        Dictionary mapping chromosome names to numeric indices
    """
    # Standard chromosome list in desired sort order
    chrom_list = ['chr' + str(i) for i in range(1, 23)] + ['chrX', 'chrY', 'chrM']
    
    # Add any non-standard chromosomes found in input
    nonstd_chrom_list = chrom_series[~chrom_series.isin(chrom_list)].unique().tolist()
    chrom_list += nonstd_chrom_list
    
    # Create mapping of chromosome names to indices
    return {chrom: idx for idx, chrom in enumerate(chrom_list)}


def get_chrom_keys(chrom_series: pd.Series, chrom_keys: dict | None = None) -> pd.Series:
    """
    Convert chromosome names to numeric keys for sorting.
    
    Maps chromosome names to numeric indices using either provided mapping
    or default standard chromosome order.
    
    Args:
        chrom_series: Series containing chromosome names
        chrom_keys: Optional dictionary mapping chromosomes to indices.
                   If None, uses standard chromosome order.
                   
    Returns:
        Series with chromosome names converted to numeric indices.
        Returns original series if chromosome not found in mapping.
    """
    if chrom_keys is None:
        # Use standard chromosome order if no mapping provided
        chrom_list = ['chr' + str(i) for i in range(1, 23)] + ['chrX', 'chrY', 'chrM']
        chrom_keys = {chrom: idx for idx, chrom in enumerate(chrom_list)}

    try:
        return chrom_series.apply(lambda x: chrom_keys[x])
    except KeyError:
        # Return original if chromosome not in mapping
        return chrom_series
