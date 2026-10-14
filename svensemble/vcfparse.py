"""
Functions for parsing VCF files.

"""

import logging
from collections.abc import Iterable

import pandas as pd
from pysam import VariantFile, VariantRecord

from . import _caller_params

logger = logging.getLogger(__name__)


# BASIC READING FUNCTIONS

def parse_vcf_to_df(
    vcf_file: VariantFile,
    caller_params: dict,
    sample_name: str | None = None
) -> tuple[pd.DataFrame, VariantFile.header]:
    """
    Parse a VCF file into a pandas DataFrame with variant information.

    Dispatches to parse_vcf_to_df_with_format when
    caller_params['parse_format'] is True, otherwise to
    parse_vcf_to_df_no_format.

    Args:
        vcf_file: A pysam VariantFile object containing variant calls.
        caller_params: Caller-specific parameters; see _caller_params.py.
        sample_name: Sample to extract FORMAT fields for (with-format path
            only). Defaults to the first sample in the header.

    Returns:
        Tuple of (DataFrame of variants, VCF header).
    """
    parse_format = caller_params['parse_format']
    logger.debug(
        "parse_vcf_to_df: parse_format=%s, sample_name=%s",
        parse_format, sample_name,
    )
    if parse_format:
        return parse_vcf_to_df_with_format(vcf_file, caller_params, sample_name)
    return parse_vcf_to_df_no_format(vcf_file, caller_params)


def parse_vcf_to_df_no_format(
    vcf_file: VariantFile,
    caller_params: dict = None
) -> tuple[pd.DataFrame, VariantFile.header]:
    """
    Parse a VCF file into a pandas DataFrame, skipping FORMAT/sample fields.

    Only columns specified by caller_params are created.

    Args:
        vcf_file: A pysam VariantFile object containing variant calls.
        caller_params: Caller-specific parameters; see _caller_params.py.
            Defaults to _caller_params.default_params when None.

    Returns:
        Tuple of (DataFrame of variants, VCF header).
    """
    if caller_params is None:
        caller_params = _caller_params.default_params

    base_fields = {col: [] for col in caller_params['base_cols']}
    filter_fields = {f'FILTER_{name}': [] for name in dict.fromkeys(caller_params['filter_cols'])}

    info_col_names = _ordered_info_col_names(caller_params)
    info_fields = {
        f'INFO_{name}': []
        for name in info_col_names
    }
    _remove_unused_info_header_fields(vcf_file, caller_params)
    _remove_unused_filter_header_fields(vcf_file, caller_params)
    _remove_format_header_fields(vcf_file)

    for record in vcf_file:
        _fetch_base_fields(record, base_fields)
        _fetch_filter_fields(record, filter_fields, caller_params)
        _fetch_info_fields(record, info_fields, info_col_names)

    df = pd.DataFrame(base_fields | filter_fields | info_fields)
    logger.debug("parse_vcf_to_df_no_format: parsed %d records", len(df))
    return df, vcf_file.header


def parse_vcf_to_df_with_format(
    vcf_file: VariantFile,
    caller_params: dict = None,
    sample_name: str | None = None
) -> tuple[pd.DataFrame, VariantFile.header]:
    """
    Parse a VCF file into a DataFrame including FORMAT/sample fields.

    Only columns specified by caller_params are created.

    Args:
        vcf_file: A pysam VariantFile object containing variant calls.
        caller_params: Caller-specific parameters; see _caller_params.py.
            Defaults to _caller_params.default_params when None.
        sample_name: Sample to extract FORMAT fields for. Defaults to the
            first sample listed in the VCF header.

    Returns:
        Tuple of (DataFrame of variants, VCF header).

    Raises:
        ValueError: If the VCF file contains no FORMAT fields.
    """
    if caller_params is None:
        caller_params = _caller_params.default_params

    if not bool(vcf_file.header.formats):
        raise ValueError(
            'FORMAT field presence specified but VCF file does not contain FORMAT fields'
        )

    base_fields = {col: [] for col in caller_params['base_cols']}
    filter_fields = {f'FILTER_{name}': [] for name in dict.fromkeys(caller_params['filter_cols'])}
    info_col_names = _ordered_info_col_names(caller_params)
    info_fields = {
        f'INFO_{name}': []
        for name in info_col_names
    }

    if sample_name is None:
        sample_name = vcf_file.header.samples[0]
        logger.debug("parse_vcf_to_df_with_format: defaulting to first sample '%s'", sample_name)

    sample_fields = {
        f'SAMPLE_{field_name}': []
        for field_name in caller_params['format_cols']
    }

    _remove_unused_info_header_fields(vcf_file, caller_params)
    _remove_unused_filter_header_fields(vcf_file, caller_params)
    _remove_unused_format_header_fields(vcf_file, caller_params)

    for record in vcf_file:
        _fetch_base_fields(record, base_fields)
        _fetch_filter_fields(record, filter_fields, caller_params)
        _fetch_info_fields(record, info_fields, info_col_names)
        _fetch_sample_fields(record, sample_fields, caller_params, sample_name)

    df = pd.DataFrame(base_fields | filter_fields | info_fields | sample_fields)
    logger.debug(
        "parse_vcf_to_df_with_format: parsed %d records (sample='%s')",
        len(df), sample_name,
    )
    return df, vcf_file.header


def _ordered_info_col_names(caller_params: dict) -> tuple[str, ...]:
    """Deduplicated INFO column names in caller_params order.

    Column order must not depend on set iteration: it reaches the training
    feature matrix, where position-based column sampling makes models
    differ between processes.
    """
    return tuple(dict.fromkeys(caller_params['info_cols_pred'] + caller_params['info_cols_mandatory']))


def _remove_unused_info_header_fields(vcf_file: VariantFile, field_info: dict) -> None:
    """Remove INFO header entries not listed in caller_params."""
    kept = set(field_info['info_cols_pred'] + field_info['info_cols_mandatory'])
    dropped = []
    for info_field in vcf_file.header.info:
        if str(info_field) not in kept:
            dropped.append(info_field)
            vcf_file.header.info[info_field].remove_header()
    if dropped:
        logger.debug("Dropping INFO header fields not in caller_params: %s", dropped)


def _remove_unused_filter_header_fields(vcf_file: VariantFile, field_info: dict) -> None:
    """Remove FILTER header entries not listed in caller_params."""
    kept = set(field_info['filter_cols'])
    dropped = []
    for filter_field in vcf_file.header.filters:
        if str(filter_field) not in kept:
            dropped.append(filter_field)
            vcf_file.header.filters[filter_field].remove_header()
    if dropped:
        logger.debug("Dropping FILTER header fields not in caller_params: %s", dropped)


def _remove_format_header_fields(vcf_file: VariantFile) -> None:
    """Remove all FORMAT header entries (used when FORMAT fields are not parsed)."""
    format_fields = list(vcf_file.header.formats)
    if not format_fields:
        return
    logger.debug("Dropping all FORMAT header fields: %s", format_fields)
    for format_field in format_fields:
        vcf_file.header.formats[format_field].remove_header()


def _remove_unused_format_header_fields(vcf_file: VariantFile, field_info: dict) -> None:
    """Remove FORMAT header entries not listed in caller_params."""
    kept = set(field_info['format_cols'])
    dropped = []
    for format_field in vcf_file.header.formats:
        if str(format_field) not in kept:
            dropped.append(format_field)
            vcf_file.header.formats[format_field].remove_header()
    if dropped:
        logger.debug("Dropping FORMAT header fields not in caller_params: %s", dropped)


base_field_dispatch = {
    'CHROM': lambda x: x.chrom,
    'POS': lambda x: x.pos,
    'ID': lambda x: x.id,
    'REF': lambda x: x.ref,
    'ALT': lambda x: x.alts[0] if x.alts else None,
    'QUAL': lambda x: x.qual,
    # record.stop is used for INFO_END because pysam exposes END via stop
    # rather than through the standard INFO dict interface for some callers.
    'INFO_END': lambda x: x.stop,
}


def _fetch_base_fields(record: VariantRecord, base_fields: dict) -> None:
    """Append base VCF fields from a record to the accumulator dict."""
    for field in base_fields:
        base_fields[field].append(base_field_dispatch[field](record))


def _fetch_filter_fields(
    record: VariantRecord,
    filter_fields: dict,
    caller_params: dict,
) -> None:
    """Append FILTER fields from a record to the accumulator dict (1/0 encoding)."""
    for filter_name in caller_params['filter_cols']:
        filter_fields[f'FILTER_{filter_name}'].append(1 if filter_name in record.filter else 0)


def _fetch_info_fields(
    record: VariantRecord,
    info_fields: dict,
    info_col_names: Iterable[str],
) -> None:
    """Append INFO fields from a record to the accumulator dict (None when absent)."""
    for info_name in info_col_names:
        try:
            info_fields[f'INFO_{info_name}'].append(record.info[info_name])
        except KeyError:
            info_fields[f'INFO_{info_name}'].append(None)


def _fetch_sample_fields(
    record: VariantRecord,
    format_fields: dict,
    caller_params: dict,
    sample_name: str,
) -> None:
    """Append FORMAT/sample fields from a record to the accumulator dict (None when absent)."""
    sample = record.samples[sample_name]
    for format_name in caller_params['format_cols']:
        try:
            format_fields[f'SAMPLE_{format_name}'].append(sample[format_name])
        except KeyError:
            format_fields[f'SAMPLE_{format_name}'].append(None)
