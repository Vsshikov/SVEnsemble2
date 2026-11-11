"""
Functions for writing VCF files from breakpoint DataFrames.

Provides write_vcf() to export breakpoint-notation DataFrames as VCF 4.5 files
using pysam. Classified SV types (DEL, DUP, INS, INV) are written as symbolic
allele records; remaining unclassified SVs are written as BND pairs.
"""

import logging
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from pysam import VariantFile, VariantHeader

from . import breakpoints as bps_mod
from . import breakends as bnds
from . import utils
from ._constants import BND1, BND2, INV_EVENT_SUFFIXES, SAMPLE_ID_COL, SCORE_FIELD

logger = logging.getLogger(__name__)



_SYMBOLIC_SVTYPES = frozenset({'DEL', 'DUP', 'INS', 'INV'})

_REQUIRED_COLS = [
    'CHROM_bnd1', 'POS_bnd1', 'ALT_bnd1',
    'CHROM_bnd2', 'POS_bnd2', 'ALT_bnd2',
    'INFO_EVENT',
]

_CI_COLS = [
    'INFO_CIPOS_l_bnd1', 'INFO_CIPOS_r_bnd1',
    'INFO_CIPOS_l_bnd2', 'INFO_CIPOS_r_bnd2',
]


def _build_header(
    bps_df: pd.DataFrame,
    sample_name: str | None = None,
    source: str | None = None,
    samples: Sequence[str] = (),
    score: bool = False,
    extra_lines: Sequence[str] = (),
) -> VariantHeader:
    """
    Build a VCF 4.5 header with SV-related INFO/ALT/FILTER definitions.

    Contigs are derived from chromosomes present in the breakpoint DataFrame
    and ordered using the standard chromosome ordering from utils.

    Args:
        bps_df: Breakpoint DataFrame used to determine the contig set.
        sample_name: Optional sample name to include in the header (used
            when *samples* is empty).
        source: Optional source program identifier for the header.
        samples: Sample columns of the header, in order.
        score: Also define the per-sample FORMAT field SCORE_FIELD.
        extra_lines: Meta lines ('##key=value') added after ##source.

    Returns:
        Configured pysam VariantHeader.
    """
    header = VariantHeader()

    if source:
        header.add_line(f'##source={source}')
    for line in extra_lines:
        header.add_line(line)

    all_chroms = set(bps_df['CHROM_bnd1'].unique()) | set(bps_df['CHROM_bnd2'].unique())
    all_chroms.discard(np.nan)
    if all_chroms:
        chrom_keys = utils.generate_chrom_key_dict(pd.Series(list(all_chroms)))
        for chrom in sorted(all_chroms, key=lambda c: chrom_keys.get(c, float('inf'))):
            header.add_line(f'##contig=<ID={chrom}>')

    header.add_line('##ALT=<ID=DEL,Description="Deletion">')
    header.add_line('##ALT=<ID=DUP,Description="Duplication">')
    header.add_line('##ALT=<ID=INS,Description="Insertion">')
    header.add_line('##ALT=<ID=INV,Description="Inversion">')

    header.add_line(
        '##INFO=<ID=SVTYPE,Number=1,Type=String,'
        'Description="Type of structural variant">'
    )
    header.add_line(
        '##INFO=<ID=END,Number=1,Type=Integer,'
        'Description="End position of the variant">'
    )
    header.add_line(
        '##INFO=<ID=SVLEN,Number=A,Type=Integer,'
        'Description="Number of inserted, duplicated, inverted, or deleted bases '
        'for INS, DUP, INV, DEL symbolic alleles">'
    )
    header.add_line(
        '##INFO=<ID=CIPOS,Number=.,Type=Integer,'
        'Description="Confidence interval around POS for symbolic structural variant representation">'
    )
    header.add_line(
        '##INFO=<ID=CIEND,Number=.,Type=Integer,'
        'Description="Confidence interval around the inferred END for symbolic structural variant representation">'
    )
    header.add_line(
        '##INFO=<ID=MATEID,Number=1,Type=String,'
        'Description="ID of mate breakend">'
    )
    header.add_line(
        '##INFO=<ID=EVENT,Number=1,Type=String,'
        'Description="ID of associated event">'
    )
    header.add_line(
        '##INFO=<ID=IMPRECISE,Number=0,Type=Flag,'
        'Description="Imprecise structural variation">'
    )

    header.add_line('##FILTER=<ID=PASS,Description="All filters passed">')

    if score:
        header.add_line(
            f'##FORMAT=<ID={SCORE_FIELD},Number=1,Type=Float,'
            'Description="SVEnsemble2 score of the SV in this sample">'
        )

    for name in samples or ([sample_name] if sample_name else []):
        header.add_sample(name)

    return header


def _get_svtypes(bps_df: pd.DataFrame, source: str | None = None) -> pd.Series:
    """
    Get or compute inferred SV types for each breakpoint row.

    Uses INFO_INFERRED_SVTYPE if present, otherwise calls
    breakpoints.annotate_svtypes which requires a source name.

    Args:
        bps_df: Breakpoint DataFrame.
        source: Caller/source name required when annotation is needed.

    Returns:
        Series of SV type strings aligned to bps_df.index.

    Raises:
        ValueError: If SV types cannot be determined.
    """
    if 'INFO_INFERRED_SVTYPE' in bps_df.columns:
        return bps_df['INFO_INFERRED_SVTYPE']
    if source is None and 'source' in bps_df.columns:
        source = bps_df['source'].iloc[0]
    if source is None:
        raise ValueError(
            "Cannot infer SV types: no INFO_INFERRED_SVTYPE column found and "
            "no source provided. Pre-annotate with breakpoints.annotate_svtypes() "
            "or pass source=."
        )
    return bps_mod.annotate_svtypes(bps_df, source)


def _has_ci(bps_df: pd.DataFrame) -> bool:
    """Check whether confidence interval columns are present."""
    return all(col in bps_df.columns for col in _CI_COLS)


def _set_ci(record, row: pd.Series, bnd_label: str, info_key: str = 'CIPOS') -> None:
    """Set CIPOS or CIEND on a pysam record and flag IMPRECISE if nonzero."""
    ci_l = int(row[f'INFO_CIPOS_l_{bnd_label}'])
    ci_r = int(row[f'INFO_CIPOS_r_{bnd_label}'])
    record.info[info_key] = (ci_l, ci_r)
    if ci_l != 0 or ci_r != 0:
        record.info['IMPRECISE'] = True


def _get_ins_svlen(row: pd.Series) -> int | None:
    """
    Determine SVLEN for an insertion from INFO_SVLEN or the ALT insert sequence.

    Returns None if no insert length can be determined.
    """
    if 'INFO_SVLEN' in row.index and pd.notna(row['INFO_SVLEN']):
        val = abs(int(row['INFO_SVLEN']))
        if val > 0:
            return val
    try:
        insert_info = bnds.process_breakend_alt(pd.Series([row['ALT_bnd1']]))
        seq = insert_info['insert_seq'].iloc[0]
        if seq:
            return len(seq)
    except (KeyError, IndexError, ValueError, TypeError):
        pass
    return None


def _strip_inv_event_suffix(event_id: str) -> str:
    """Strip _e1 / _e2 suffixes added during inversion breakpoint expansion."""
    for suffix in INV_EVENT_SUFFIXES:
        if event_id.endswith(suffix):
            return event_id.removesuffix(suffix)
    return event_id


def _set_scores(record, scores: Mapping[str, float] | None) -> None:
    """Set SCORE_FIELD for each sample that has a score (others stay missing)."""
    for sample, score in (scores or {}).items():
        if pd.notna(score):
            record.samples[sample][SCORE_FIELD] = float(score)


def _write_symbolic_record(
    vcf_out: VariantFile,
    row: pd.Series,
    svtype: str,
    has_ci: bool,
    record_id: str | None = None,
    scores: Mapping[str, float] | None = None,
) -> None:
    """
    Write a symbolic allele VCF record for a classified SV (DEL/DUP/INS/INV).

    POS is taken from POS_bnd1 and END is set to POS_bnd2.
    The stop/END field is assigned before INFO fields to avoid
    transient END < POS warnings from htslib.
    """
    rec = vcf_out.new_record()
    rec.contig = str(row['CHROM_bnd1'])
    rec.pos = int(row['POS_bnd1'])
    pos_bnd2 = int(row['POS_bnd2'])
    rec.stop = pos_bnd2

    rec.id = record_id if record_id is not None else str(row['INFO_EVENT'])
    rec.ref = 'N'
    rec.alts = (f'<{svtype}>',)
    rec.filter.add('PASS')

    rec.info['SVTYPE'] = svtype

    span = pos_bnd2 - int(row['POS_bnd1'])
    if svtype == 'DEL':
        rec.info['SVLEN'] = span
    elif svtype == 'INS':
        svlen = _get_ins_svlen(row)
        if svlen is not None:
            rec.info['SVLEN'] = svlen
    else:
        rec.info['SVLEN'] = span

    if has_ci:
        _set_ci(rec, row, BND1, 'CIPOS')
        _set_ci(rec, row, BND2, 'CIEND')

    _set_scores(rec, scores)
    vcf_out.write(rec)


def _write_bnd_records(
    vcf_out: VariantFile,
    row: pd.Series,
    has_ci: bool,
    scores: Mapping[str, float] | None = None,
) -> None:
    """
    Write a pair of BND records for an unclassified structural variant.

    Each breakpoint row produces two VCF records (one per breakend) linked
    via MATEID and EVENT INFO fields.
    """
    raw_event_id = str(row['INFO_EVENT'])
    event_id = _strip_inv_event_suffix(raw_event_id)
    bnd1_id = f'{raw_event_id}_{BND1}'
    bnd2_id = f'{raw_event_id}_{BND2}'

    for bnd_label, own_id, mate_id in [
        (BND1, bnd1_id, bnd2_id),
        (BND2, bnd2_id, bnd1_id),
    ]:
        rec = vcf_out.new_record()
        rec.contig = str(row[f'CHROM_{bnd_label}'])
        pos = int(row[f'POS_{bnd_label}'])
        rec.pos = pos
        rec.stop = pos + 1
        rec.id = own_id
        rec.ref = 'N'
        rec.alts = (str(row[f'ALT_{bnd_label}']),)
        rec.filter.add('PASS')

        rec.info['SVTYPE'] = 'BND'
        rec.info['MATEID'] = mate_id
        rec.info['EVENT'] = event_id

        if has_ci:
            _set_ci(rec, row, bnd_label, 'CIPOS')

        _set_scores(rec, scores)
        vcf_out.write(rec)


_SITE_EXTRA_COLS = ['INFO_INFERRED_SVTYPE', 'INFO_SVLEN']


def site_cols(bps_df: pd.DataFrame, include_event: bool) -> list[str]:
    """List the columns that all samples of one record share.

    Args:
        bps_df (pd.DataFrame): Breakpoint rows.
        include_event (bool): Also list INFO_EVENT.

    Returns:
        list[str]: The columns of *bps_df* among positions, ALTs, CIs,
            INFO_INFERRED_SVTYPE and INFO_SVLEN (and INFO_EVENT).
    """
    cols = _REQUIRED_COLS + _CI_COLS + _SITE_EXTRA_COLS
    return [c for c in cols if c in bps_df.columns and (include_event or c != 'INFO_EVENT')]


def _check_sites(bps_df: pd.DataFrame, key: pd.Series, cols: list[str]) -> None:
    """Raise if rows that share a record key differ in any of *cols*.

    Args:
        bps_df (pd.DataFrame): Breakpoint rows.
        key (pd.Series): Record key of each row, aligned with *bps_df*.
        cols (list[str]): Columns the rows of one record must share.

    Raises:
        ValueError: If two rows of one record differ in a column of *cols*.
    """
    per_key = bps_df[cols].groupby(key.to_numpy(), sort=False).nunique(dropna=False)
    conflicting = per_key.index[(per_key > 1).any(axis=1)].tolist()
    if conflicting:
        raise ValueError(f"Samples disagree on the site of SV(s) {conflicting[:10]}; harmonise them first")


def _pair_inversion_halves(bps_df: pd.DataFrame, key: pd.Series) -> dict[str, str]:
    """Pair the records that hold the two halves of one inversion.

    Inversions on the event path are split into <event>_e1 and <event>_e2
    (events.convert_inv_to_bnd). Looks through every <source>_INFO_EVENT
    column, and pairs the record of a row holding <event>_e1 with the record
    of the row of the same sample holding <event>_e2. Keeps only unambiguous
    pairs: a record paired with two different partners belongs to disagreeing
    inversions and keeps its own record.

    Args:
        bps_df (pd.DataFrame): Calls rows with SAMPLE_ID_COL and
            <source>_INFO_EVENT columns.
        key (pd.Series): Record key of each row, aligned with *bps_df*.

    Returns:
        dict[str, str]: For each paired record, the key of the inversion record
            both halves are written under (the _e1 half's key).
    """
    sample = bps_df[SAMPLE_ID_COL].astype(str).to_numpy()
    halves = []
    for col in [c for c in bps_df.columns if c.endswith('_INFO_EVENT')]:
        event = bps_df[col].astype('string').fillna('').to_numpy(dtype=str)
        for suffix in INV_EVENT_SUFFIXES:
            has = np.char.endswith(event, suffix)
            halves.append(pd.DataFrame({
                'col': col, 'sample': sample[has], 'base': [e[:-len(suffix)] for e in event[has]],
                'suffix': suffix, 'key': key.to_numpy()[has],
            }))
    halves = pd.concat(halves, ignore_index=True) if halves else pd.DataFrame(
        columns=['col', 'sample', 'base', 'suffix', 'key'])
    first, second = (halves[halves['suffix'] == s].drop(columns='suffix') for s in INV_EVENT_SUFFIXES)
    pairs = first.merge(second, on=['col', 'sample', 'base'], suffixes=('_e1', '_e2'))
    pairs = pairs.loc[pairs['key_e1'] != pairs['key_e2'], ['key_e1', 'key_e2']].drop_duplicates()
    counts = pd.concat([pairs['key_e1'], pairs['key_e2']]).value_counts()
    unique = pairs[(pairs['key_e1'].map(counts) == 1) & (pairs['key_e2'].map(counts) == 1)]
    return {
        **dict(zip(unique['key_e1'], unique['key_e1'])),
        **dict(zip(unique['key_e2'], unique['key_e1'])),
    }


def _group_rows(
    bps_df: pd.DataFrame,
    score_col: str | None,
    sample_name: str | None,
    group_by: str | None,
) -> tuple[pd.DataFrame, list[str], dict[str, dict[str, float]], dict[str, str]]:
    """Group per-sample rows into VCF records and collect their scores.

    Without SAMPLE_ID_COL, writes every row as its own record and gives its
    score to *sample_name*. Otherwise keys each row by its *group_by* column,
    or, with *group_by* None, by its INFO_EVENT (and raises when the rows hold
    more than one sample). Rows that share a key become one record, which takes
    the key as its INFO_EVENT and must agree on the site columns (site_cols);
    only the score stays per sample. Finds the inversion halves to collapse:
    by stripping _e1/_e2 from the key when grouping by INFO_EVENT, otherwise
    with _pair_inversion_halves.

    Args:
        bps_df (pd.DataFrame): Breakpoint or calls rows.
        score_col (str | None): Column holding the score, if any.
        sample_name (str | None): Sample to score rows without SAMPLE_ID_COL.
        group_by (str | None): Column whose shared values make rows of several
            samples one record, e.g. 'INFO_EVENT' for joint-called input or
            'svensemble_id' for calls.

    Returns:
        tuple: The rows, one per record; the header samples; each record's
            score per sample; and each record's inversion record key.

    Raises:
        KeyError: If the rows lack the *group_by* column.
        ValueError: If *group_by* is None and the rows hold several samples,
            if rows of one record disagree on the site, or if a record holds
            two rows of one sample.
    """
    if SAMPLE_ID_COL not in bps_df.columns:
        samples = [sample_name] if (score_col and sample_name) else []
        scores = (
            {str(e): {sample_name: v} for e, v in zip(bps_df['INFO_EVENT'], bps_df[score_col])}
            if samples else {}
        )
        events = bps_df['INFO_EVENT'].astype(str)
        return bps_df, samples, scores, {e: _strip_inv_event_suffix(e) for e in events}

    sample = bps_df[SAMPLE_ID_COL].astype(str)
    samples = sorted(sample.unique())
    if group_by is None and len(samples) > 1:
        raise ValueError(
            f"Rows of {len(samples)} samples {samples[:10]} and no group_by: give group_by "
            "'INFO_EVENT' (joint-called input) or the SV ID column (calls)"
        )
    if group_by is not None and group_by not in bps_df.columns:
        raise KeyError(f"group_by column {group_by!r} not in the rows")
    key = bps_df[group_by or 'INFO_EVENT'].astype(str)
    if group_by is not None:
        _check_sites(bps_df, key, site_cols(bps_df, include_event=(group_by == 'INFO_EVENT')))

    repeated = pd.DataFrame({'key': key.to_numpy(), 'sample': sample.to_numpy()}).duplicated()
    if repeated.any():
        dup = key[repeated.to_numpy()].unique().tolist()
        raise ValueError(f"Record(s) {dup[:10]} have more than one row for one sample")

    if group_by in (None, 'INFO_EVENT'):
        inversion_key = {k: _strip_inv_event_suffix(k) for k in key}
    else:
        paired = _pair_inversion_halves(bps_df, key)
        inversion_key = {k: paired.get(k, k) for k in key}

    record_df = (
        bps_df.assign(INFO_EVENT=key)
        .drop_duplicates('INFO_EVENT', keep='first')
        .drop(columns=[SAMPLE_ID_COL])
    )
    if not score_col:
        return record_df, samples, {}, inversion_key
    scores: dict[str, dict[str, float]] = {}
    for k, smp, score in zip(key, sample, bps_df[score_col]):
        scores.setdefault(k, {})[smp] = score
    return record_df, samples, scores, inversion_key


def write_vcf(
    bps_df: pd.DataFrame,
    output_path: str | Path,
    source: str | None = None,
    sample_name: str | None = None,
    inv_as_bnd: bool = False,
    score_col: str | None = None,
    header_lines: Sequence[str] = (),
    group_by: str | None = None,
) -> Path:
    """
    Write a VCF 4.5 file from a breakpoint DataFrame.

    Classified SV types (DEL, DUP, INS, INV) are written as symbolic allele
    records with SVTYPE, END, SVLEN, and optionally CIPOS/CIEND INFO fields.
    Remaining unclassified SVs (OTHER) are written as BND pairs with MATEID
    and EVENT linking the two breakend records.

    Records are sorted by chromosome and position of the first breakend
    using the standard chromosome ordering.

    Inversions in the breakpoint DataFrame are represented as two events
    (_e1 / _e2).  By default these are collapsed back into a single
    <INV> symbolic record.  When *inv_as_bnd* is True, each event is
    instead written as a BND pair (4 BND records total per inversion).

    Positions in the DataFrame are written through as-is via pysam (VCF POS
    values round-trip exactly through pysam's record.pos property).

    Args:
        bps_df: DataFrame in breakpoint notation. Required columns:
            CHROM_bnd1, POS_bnd1, ALT_bnd1, CHROM_bnd2, POS_bnd2, ALT_bnd2,
            INFO_EVENT.
            Optional columns used when present: INFO_CIPOS_l_bnd1,
            INFO_CIPOS_r_bnd1, INFO_CIPOS_l_bnd2, INFO_CIPOS_r_bnd2,
            INFO_INFERRED_SVTYPE, INFO_SVLEN, source.
        output_path: Path for the output VCF. Use .vcf.gz for bgzipped output.
        source: Caller/source name. Required for SV type inference when
            INFO_INFERRED_SVTYPE is absent and no 'source' column exists.
        sample_name: Sample name to include in the VCF header.
        inv_as_bnd: If True, write inversions as BND pairs instead of
            symbolic <INV> records.
        header_lines: Extra '##key=value' meta lines for the header.
        score_col: Column holding the SVEnsemble2 score, written as FORMAT
            SCORE_FIELD per sample; the header lists the samples in
            FORMAT_SAMPLE_ID.
        group_by: Column whose shared values make rows of several samples one
            record, e.g. 'INFO_EVENT' (joint-called input) or 'svensemble_id'
            (calls). None (the default) writes one record per row and raises
            with more than one sample. See _group_rows.

    Returns:
        Path to the written VCF file.

    Raises:
        ValueError: If required columns are missing or SV types cannot be
            inferred.
    """
    output_path = Path(output_path)

    missing = [c for c in _REQUIRED_COLS if c not in bps_df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    invalid_pos_mask = (bps_df['POS_bnd1'] <= 0) | (bps_df['POS_bnd2'] <= 0)
    if len(bps_df) and invalid_pos_mask.any():
        n_bad = invalid_pos_mask.sum()
        bad_events = bps_df.loc[invalid_pos_mask, 'INFO_EVENT'].tolist()
        logger.warning(
            "Dropping %d record(s) with non-positive POS (VCF requires POS >= 1): %s",
            n_bad, bad_events,
        )
        bps_df = bps_df[~invalid_pos_mask]

    bps_df, samples, scores, inversion_key = _group_rows(bps_df, score_col, sample_name, group_by)
    header_kwargs = dict(
        sample_name=sample_name, source=source, samples=samples, score=bool(score_col),
        extra_lines=header_lines,
    )

    # Written via a temporary sibling and moved into place, so a failure part
    # way through never leaves a truncated VCF at output_path.
    write_mode = 'wz' if str(output_path).endswith('.gz') else 'w'
    with utils.atomic_path(output_path) as tmp_path:
        if len(bps_df) == 0:
            empty_chrom_df = pd.DataFrame({
                'CHROM_bnd1': pd.Series(dtype=str),
                'CHROM_bnd2': pd.Series(dtype=str),
            })
            header = _build_header(empty_chrom_df, **header_kwargs)
            with VariantFile(str(tmp_path), write_mode, header=header):
                pass
            return output_path

        svtypes = _get_svtypes(bps_df, source)
        has_ci = _has_ci(bps_df)
        header = _build_header(bps_df, **header_kwargs)

        chrom_keys = utils.generate_chrom_key_dict(
            pd.concat([bps_df['CHROM_bnd1'], bps_df['CHROM_bnd2']])
        )
        sorted_idx = bps_df.sort_values(
            ['CHROM_bnd1', 'POS_bnd1'],
            key=lambda x: utils.get_chrom_keys(x, chrom_keys=chrom_keys)
        ).index

        symbolic_non_inv = _SYMBOLIC_SVTYPES - {'INV'}
        written_inv_events: set[str] = set()

        with VariantFile(str(tmp_path), write_mode, header=header) as vcf_out:
            for idx in sorted_idx:
                row = bps_df.loc[idx]
                svtype = svtypes.loc[idx]
                row_scores = scores.get(str(row['INFO_EVENT']))

                if svtype == 'INV':
                    if inv_as_bnd:
                        _write_bnd_records(vcf_out, row, has_ci, row_scores)
                    else:
                        base_event = inversion_key[str(row['INFO_EVENT'])]
                        if base_event in written_inv_events:
                            continue
                        written_inv_events.add(base_event)
                        _write_symbolic_record(
                            vcf_out, row, 'INV', has_ci, record_id=base_event, scores=row_scores,
                        )
                elif svtype in symbolic_non_inv:
                    _write_symbolic_record(vcf_out, row, svtype, has_ci, scores=row_scores)
                else:
                    _write_bnd_records(vcf_out, row, has_ci, row_scores)

    return output_path
