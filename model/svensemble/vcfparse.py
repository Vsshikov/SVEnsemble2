"""
Functions for parsing VCF files

Are based on pysam VariantFile object
"""

import pandas as pd
from pysam import VariantFile, VariantRecord

from . import _caller_params
# BASIC READING FUNCTIONS

def parse_vcf_to_df(
    vcf_file: VariantFile,
    field_info: dict|None=None
) -> tuple[pd.DataFrame, VariantFile.header]:
    """
    Parse a VCF file into a pandas DataFrame with variant information.
    Only creates columns specified by field_info dictionary
    
    Args:
        vcf_file: A pysam VariantFile object containing variant calls
        field_info: Dictionary containing field names to parse from VCF:
                   - base_cols: List of basic VCF fields (CHROM, POS, etc.)
                   - filter_cols: List of FILTER field names
                   - info_cols_pred: List of INFO field names
                   If None, uses default field lists: only basic entries, leaves INFO and FILTER out
        
    Returns:
        tuple[pd.DataFrame, VariantFile.header]: DataFrame containing variant information and VCF header
    """
    # If field_info is not provided, only 'CHROM', 'POS', 'ID', 'REF', 'ALT', 'QUAL' are parsed
    if field_info is None:
        field_info = _caller_params.default_params
    
    # Initialize field collections
    base_fields = {col: [] for col in ['CHROM', 'POS', 'ID', 'REF', 'ALT', 'QUAL']}
    filter_fields = {f'FILTER_{name}': [] for name in set(field_info['filter_cols'])}
    info_fields = {
        f'INFO_{name}': [] 
        for name in set(field_info['info_cols_pred'] + field_info['info_cols_mandatory'])
    }
    
    # Remove unused header fields
    _remove_unused_info_header_fields(vcf_file, field_info)
    _remove_unused_filter_header_fields(vcf_file, field_info)
    _remove_format_header_fields(vcf_file)

    # Parse records
    for record in vcf_file:
        _fetch_base_fields(record, base_fields)
        _fetch_filter_fields(record, filter_fields, field_info['filter_cols'])
        _fetch_info_fields(record, info_fields, field_info)

    return pd.DataFrame(base_fields | filter_fields | info_fields), vcf_file.header

def _remove_unused_info_header_fields(vcf_file: VariantFile, field_info: dict) -> None:
    """Remove unused INFO fields from VCF header."""
    info_dropped = []
    for info_field in vcf_file.header.info:
        if str(info_field) not in field_info['info_cols_pred'] + field_info['info_cols_mandatory']:
            info_dropped.append(info_field)
            vcf_file.header.info[info_field].remove_header
    if info_dropped:
        print(f'Dropping {info_dropped} from meta as it is not present in specified info cols')

def _remove_unused_filter_header_fields(vcf_file: VariantFile, field_info: dict) -> None:
    """Remove unused FILTER fields from VCF header."""
    filter_dropped = []
    for filter_field in vcf_file.header.filters:
        if str(filter_field) not in field_info['filter_cols']:
            filter_dropped.append(filter_field)
            vcf_file.header.filters[filter_field].remove_header
    if filter_dropped:
        print(f'Dropping {filter_dropped} from meta as it is not present in specified filter cols')

def _remove_format_header_fields(vcf_file: VariantFile) -> None:
    """Remove all FORMAT fields from VCF header."""
    print('Dropping all format fields')
    vcf_file.header.formats.remove_header

def _fetch_base_fields(record: VariantRecord, base_fields: dict) -> None:
    """Parse basic variant fields from a VCF record."""
    base_fields['CHROM'].append(record.chrom)
    base_fields['POS'].append(record.pos)
    base_fields['ID'].append(record.id)
    base_fields['REF'].append(record.ref)
    base_fields['ALT'].append(record.alts[0])
    base_fields['QUAL'].append(record.qual)

def _fetch_filter_fields(
    record: VariantRecord, 
    filter_fields: dict, 
    filter_cols: list
) -> None:
    """Parse FILTER fields from a VCF record."""
    for filter_name in filter_cols:
        try:
            record.filter[filter_name]
            filter_fields[f'FILTER_{filter_name}'].append(1)
        except KeyError:
            filter_fields[f'FILTER_{filter_name}'].append(0)

def _fetch_info_fields(
    record: VariantRecord, 
    info_fields: dict, 
    field_info: dict
) -> None:
    """Parse INFO fields from a VCF record."""
    for info_name in field_info['info_cols_pred'] + field_info['info_cols_mandatory']:
        try:
            info_value = record.info[info_name]
            info_fields[f'INFO_{info_name}'].append(info_value)
        except KeyError:
            info_fields[f'INFO_{info_name}'].append(None)
