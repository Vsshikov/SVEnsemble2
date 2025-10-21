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
    caller_params: dict,
    sample_name: str|None=None
) -> tuple[pd.DataFrame, VariantFile.header]:
    """
    Parse a VCF file into a pandas DataFrame with variant information.
    If caller_params['parse_format'] is True, parse FORMAT fields.
    Otherwise, only parse basic fields.

    Args:
        vcf_file: A pysam VariantFile object containing variant calls
        caller_params: Dictionary containing caller-specific parameters
    """
    if caller_params['parse_format']:
        return parse_vcf_to_df_with_format(vcf_file, caller_params, sample_name)
    else:
        return parse_vcf_to_df_no_format(vcf_file, caller_params)


def parse_vcf_to_df_no_format(
    vcf_file: VariantFile,
    caller_params: dict=None
) -> tuple[pd.DataFrame, VariantFile.header]:
    """
    Parse a VCF file into a pandas DataFrame with variant information.
    Only creates columns specified by field_info dictionary

    Args:
        vcf_file: A pysam VariantFile object containing variant calls
        caller_params: Dictionary containing caller-specific parameters, see _caller_params.py for details

    Returns:
        tuple[pd.DataFrame, VariantFile.header]: DataFrame containing variant information and VCF header
    """
    # If caller_params is not provided, only 'CHROM', 'POS', 'ID', 'REF', 'ALT', 'QUAL' are parsed
    if caller_params is None:
        caller_params = _caller_params.default_params

    # Initialize field collections
    base_fields = {col: [] for col in caller_params['base_cols']}
    filter_fields = {f'FILTER_{name}': [] for name in set(caller_params['filter_cols'])}
    info_fields = {
        f'INFO_{name}': []
        for name in set(caller_params['info_cols_pred'] + caller_params['info_cols_mandatory'])
    }

    # Remove unused header fields
    _remove_unused_info_header_fields(vcf_file, caller_params)
    _remove_unused_filter_header_fields(vcf_file, caller_params)
    _remove_format_header_fields(vcf_file)

    # Parse records
    for record in vcf_file:
        _fetch_base_fields(record, base_fields)
        _fetch_filter_fields(record, filter_fields, caller_params)
        _fetch_info_fields(record, info_fields, caller_params)

    return pd.DataFrame(base_fields | filter_fields | info_fields), vcf_file.header

def parse_vcf_to_df_with_format(
    vcf_file: VariantFile,
    caller_params: dict=None,
    sample_name: str|None=None
) -> tuple[pd.DataFrame, VariantFile.header]:
    """
    Parse a VCF file into a pandas DataFrame with variant information.
    Only creates columns specified by field_info dictionary

    Args:
        vcf_file: A pysam VariantFile object containing variant calls
        caller_params: Dictionary containing caller-specific parameters, see _caller_params.py for details
        sample_name: Name of the sample to parse, if None, returns first sample in VCF file

    Returns:
        tuple[pd.DataFrame, VariantFile.header]: DataFrame containing variant information and VCF header
    """
    # If caller_params is not provided, only 'CHROM', 'POS', 'ID', 'REF', 'ALT', 'QUAL' are parsed
    if caller_params is None:
        caller_params = _caller_params.default_params

    if not bool(vcf_file.header.formats):
        raise ValueError('FORMAT field presence specified but VCF file does not contain FORMAT fields')

    # Initialize field collections
    base_fields = {col: [] for col in caller_params['base_cols']}
    filter_fields = {f'FILTER_{name}': [] for name in set(caller_params['filter_cols'])}
    info_fields = {
        f'INFO_{name}': []
        for name in set(caller_params['info_cols_pred'] + caller_params['info_cols_mandatory'])
    }
    if sample_name is None:
        sample_name = vcf_file.header.samples[0]
    sample_fields = {
        f'SAMPLE_{field_name}': []
        for field_name in caller_params['format_cols']
    }

    # Remove unused header fields
    _remove_unused_info_header_fields(vcf_file, caller_params)
    _remove_unused_filter_header_fields(vcf_file, caller_params)
    _remove_unused_format_header_fields(vcf_file, caller_params)

    # Parse records
    for record in vcf_file:
        _fetch_base_fields(record, base_fields)
        _fetch_filter_fields(record, filter_fields, caller_params)
        _fetch_info_fields(record, info_fields, caller_params)
        _fetch_sample_fields(record, sample_fields, caller_params, sample_name)

    print({key: len(value) for key, value in base_fields.items()})
    print({key: len(value) for key, value in filter_fields.items()})
    print({key: len(value) for key, value in info_fields.items()})
    print({key: len(value) for key, value in sample_fields.items()})

    return pd.DataFrame(base_fields | filter_fields | info_fields | sample_fields), vcf_file.header


def _remove_unused_info_header_fields(vcf_file: VariantFile, field_info: dict) -> None:
    """Remove unused INFO fields from VCF header."""
    info_dropped = []
    for info_field in vcf_file.header.info:
        if str(info_field) not in field_info['info_cols_pred'] + field_info['info_cols_mandatory']:
            info_dropped.append(info_field)
            vcf_file.header.info[info_field].remove_header()
    if info_dropped:
        print(f'Dropping {info_dropped} from meta as it is not present in specified info cols')

def _remove_unused_filter_header_fields(vcf_file: VariantFile, field_info: dict) -> None:
    """Remove unused FILTER fields from VCF header."""
    filter_dropped = []
    for filter_field in vcf_file.header.filters:
        if str(filter_field) not in field_info['filter_cols']:
            filter_dropped.append(filter_field)
            vcf_file.header.filters[filter_field].remove_header()
    if filter_dropped:
        print(f'Dropping {filter_dropped} from meta as it is not present in specified filter cols')

def _remove_format_header_fields(vcf_file: VariantFile) -> None:
    """Remove all FORMAT fields from VCF header."""
    print('Dropping all format fields')
    for format_field in vcf_file.header.formats:
        vcf_file.header.formats[format_field].remove_header()

def _remove_unused_format_header_fields(vcf_file: VariantFile, field_info: dict) -> None:
    """Remove unused sample fields from VCF header."""
    format_dropped = []
    for format_field in vcf_file.header.formats:
        if str(format_field) not in field_info['format_cols']:
            format_dropped.append(format_field)
            vcf_file.header.formats[format_field].remove_header()
    if format_dropped:
        print(f'Dropping {format_dropped} from meta as it is not present in specified format cols')

base_field_dispatch = {
    'CHROM': lambda x: x.chrom,
    'POS': lambda x: x.pos,
    'ID': lambda x: x.id,
    'REF': lambda x: x.ref,
    'ALT': lambda x: x.alts[0],
    'QUAL': lambda x: x.qual,
    # For some callers we need to fetch INFO_END which is inaccessible through normal INFO interface
    # We need to access record.stop for this
    'INFO_END': lambda x: x.stop
}

def _fetch_base_fields(record: VariantRecord, base_fields: dict) -> None:
    """Parse basic variant fields from a VCF record."""
    for field in base_fields:
        base_fields[field].append(base_field_dispatch[field](record))

def _fetch_filter_fields(
    record: VariantRecord,
    filter_fields: dict,
    caller_params: dict
) -> None:
    """Parse FILTER fields from a VCF record and add to filter_fields dictionary."""
    for filter_name in caller_params['filter_cols']:
        try:
            record.filter[filter_name]
            filter_fields[f'FILTER_{filter_name}'].append(1)
        except KeyError:
            filter_fields[f'FILTER_{filter_name}'].append(0)

def _fetch_info_fields(
    record: VariantRecord,
    info_fields: dict,
    caller_params: dict
) -> None:
    """Parse INFO fields from a VCF record and add to info_fields dictionary."""
    for info_name in set(caller_params['info_cols_pred'] + caller_params['info_cols_mandatory']):
        try:
            info_value = record.info[info_name]
            info_fields[f'INFO_{info_name}'].append(info_value)
        except KeyError:
            info_fields[f'INFO_{info_name}'].append(None)

def _fetch_sample_fields(
    record: VariantRecord,
    format_fields: dict,
    caller_params: dict,
    sample_name: str
) -> None:
    """Parse SAMPLE and FORMAT fields from a VCF record and add to format_fields dictionary."""

    for format_name in caller_params['format_cols']:
        try:
            format_fields[f'SAMPLE_{format_name}'].append(record.samples[sample_name][format_name])
        except KeyError:
            format_fields[f'SAMPLE_{format_name}'].append(None)

