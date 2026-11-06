"""
Annotate caller VCFs with the SVEnsemble2 score as a per-sample FORMAT field.

Every function here is pure: it takes header lines, record lines and the calls
DataFrame and returns or yields annotated lines. Nothing reads or writes files;
``svensemble.io`` does that and ``api.annotate`` wires the two together.

Records are edited as text, never re-serialised, so the output differs from the
input only by the SCORE_FIELD header line, the SCORE_FIELD key in each FORMAT
column and its value in each sample column. Each record is matched to its
score through the event key the caller's preprocessing recipe derives from the
record's ID and INFO (see EVENT_KEY and callset_specific).
"""

import re
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence

import numpy as np
import pandas as pd

from ._caller_params import CALLSET_PARAM_DICT
from ._constants import INV_EVENT_SUFFIXES, PRED_SUFFIX, SAMPLE_ID_COL, SCORE_FIELD

ScoreTable = dict[tuple[str, str], float]
EventKeyFn = Callable[[str, Mapping[str, str]], str | None]

_DIGITS_FLOAT = re.compile(r'\d+\.0')
_SCORE_HEADER_PREFIX = f'##FORMAT=<ID={SCORE_FIELD},'
_LINE_ENDINGS = ('\r\n', '\n', '\r')


# -- Scores -------------------------------------------------------------------

def pred_column(calls: pd.DataFrame) -> str:
    """Name of the single ``*_pred`` column in *calls*."""
    cols = [c for c in calls.columns if str(c).endswith(PRED_SUFFIX)]
    if len(cols) != 1:
        raise ValueError(f"Expected exactly one '*{PRED_SUFFIX}' column in the calls, found {cols}")
    return cols[0]


def normalise_event_id(event: str) -> str:
    """Undo the float-ification CSV round trips apply to all-digit IDs ('21.0' -> '21')."""
    return event[:-2] if _DIGITS_FLOAT.fullmatch(event) else event


def score_table(calls: pd.DataFrame, caller: str) -> ScoreTable:
    """Map (caller event, sample) to the ensemble score.

    Rows where *caller* has no event are dropped. Event IDs are normalised
    with normalise_event_id so they compare equal to the raw VCF text.
    """
    event_col = f'{caller}_INFO_EVENT'
    missing = [c for c in (event_col, SAMPLE_ID_COL) if c not in calls.columns]
    if missing:
        raise KeyError(f"Calls lack column(s) {missing} needed to score {caller!r}")
    pred_col = pred_column(calls)
    rows = calls.loc[calls[event_col].notna(), [event_col, SAMPLE_ID_COL, pred_col]]
    events = rows[event_col].astype(str).map(normalise_event_id)
    samples = rows[SAMPLE_ID_COL].astype(str)
    return dict(zip(zip(events, samples), rows[pred_col].astype(float)))


def lookup_score(key: str | None, sample: str, table: ScoreTable) -> float | None:
    """Score of a record's event in *sample*.

    An inversion on the event path is scored as two events, <key>_e1 and
    <key>_e2; the record gets the mean of whichever halves were scored.
    """
    if key is None:
        return None
    if (key, sample) in table:
        return table[(key, sample)]
    halves = [table[(key + s, sample)] for s in INV_EVENT_SUFFIXES if (key + s, sample) in table]
    return float(np.mean(halves)) if halves else None


def format_score(score: float | None) -> str:
    """VCF text for a score; missing or NaN scores become '.'."""
    if score is None or np.isnan(score):
        return '.'
    return f'{score:.6g}'


# -- Event keys ---------------------------------------------------------------

def parse_info(info: str) -> dict[str, str]:
    """Raw INFO text as {key: value}; flags map to ''."""
    if info in ('', '.'):
        return {}
    return dict(kv.split('=', 1) if '=' in kv else (kv, '') for kv in info.split(';'))


def _id_key(id_: str, info: Mapping[str, str]) -> str:
    return id_


def _gridss_key(id_: str, info: Mapping[str, str]) -> str | None:
    return info.get('EVENT')


def _manta_key(id_: str, info: Mapping[str, str]) -> str:
    # callset_specific.manta_preprocess_bnds_only: min(ID, MATEID[0]) for the
    # BND path (SVTYPE BND or INV); pandas' skipna min falls back to ID.
    if info.get('SVTYPE') in ('BND', 'INV') and info.get('MATEID'):
        return min(id_, info['MATEID'].split(',')[0])
    return id_


def _svaba_key(id_: str, info: Mapping[str, str]) -> str:
    return id_.rsplit(':', 1)[0]


def _smoove_key(id_: str, info: Mapping[str, str]) -> str:
    return id_.rsplit('_', 1)[0] if info.get('SVTYPE') == 'BND' else id_


# Keyed by CALLSET_PARAM_DICT[caller]['source'], i.e. the parse recipe name.
# Each rule mirrors the INFO_EVENT assignment in that recipe; Delly BND/TRA
# records keep their ID because delly_split_bnds' 4-character suffixes are
# stripped again.
EVENT_KEY: dict[str, EventKeyFn] = {
    'gridss': _gridss_key,
    'manta_candidate': _manta_key,
    'manta_diploid': _manta_key,
    'delly': _id_key,
    'sniffles': _id_key,
    'sniffles_207': _id_key,
    'survivor_short': _id_key,
    'survivor_long': _id_key,
    'survivor_all': _id_key,
    'severus': _id_key,
    'cutesv': _id_key,
    'svaba': _svaba_key,
    'smoove': _smoove_key,
}


def event_key_fn(caller: str) -> EventKeyFn:
    """Event-key rule for *caller*, resolved through its parse-recipe source."""
    try:
        source = CALLSET_PARAM_DICT[caller]['source']
    except KeyError:
        raise KeyError(f"Unknown caller {caller!r}: no CALLSET_PARAM_DICT entry with a 'source'") from None
    try:
        return EVENT_KEY[source]
    except KeyError:
        raise KeyError(f"No event-key rule for {caller!r} (source {source!r})") from None


# -- Text VCF -----------------------------------------------------------------

def split_line_ending(line: str) -> tuple[str, str]:
    """(content, ending) with the ending exactly as read ('' on a final unterminated line)."""
    for ending in _LINE_ENDINGS:
        if line.endswith(ending):
            return line[:-len(ending)], ending
    return line, ''


def header_samples(header_lines: Sequence[str]) -> list[str]:
    """Sample names from the #CHROM line; raises if there is none or no sample columns."""
    chrom = [line for line in header_lines if line.startswith('#CHROM')]
    if len(chrom) != 1:
        raise ValueError(f"Expected one #CHROM header line, found {len(chrom)}")
    cols = split_line_ending(chrom[0])[0].split('\t')
    if len(cols) < 10:
        raise ValueError("VCF has no sample columns (sites-only); a FORMAT field cannot be added")
    return cols[9:]


def annotate_header(header_lines: Sequence[str], pred_col: str) -> list[str]:
    """Header with the SCORE_FIELD FORMAT line before #CHROM.

    An existing SCORE_FIELD line (from an earlier run) is replaced in place so
    its Description names the current prediction column.
    """
    header_samples(header_lines)
    chrom_idx = next(i for i, line in enumerate(header_lines) if line.startswith('#CHROM'))
    ending = split_line_ending(header_lines[chrom_idx])[1]
    new_line = (
        f'{_SCORE_HEADER_PREFIX}Number=1,Type=Float,'
        f'Description="SVEnsemble2 score ({pred_col})">{ending}'
    )
    existing = [i for i, line in enumerate(header_lines) if line.startswith(_SCORE_HEADER_PREFIX)]
    out = list(header_lines)
    if existing:
        out[existing[0]] = new_line
        return out
    return out[:chrom_idx] + [new_line] + out[chrom_idx:]


def annotate_sample_field(field: str, n_keys: int, idx: int, value: str) -> str:
    """One sample column with *value* at FORMAT position *idx*.

    VCF lets a sample drop trailing values ('0/1' under GT:GQ:DR), so the
    column is first padded with '.' to *n_keys* values; otherwise an appended
    value would land under the wrong FORMAT key.
    """
    values = field.split(':') if n_keys else []
    values += ['.'] * (n_keys - len(values))
    if idx < len(values):
        values[idx] = value
    else:
        values.append(value)
    return ':'.join(values)


def annotate_record(
    line: str,
    key_fn: EventKeyFn,
    samples: Sequence[str],
    table: ScoreTable,
) -> str:
    """One data line with SCORE_FIELD added to FORMAT and every sample column.

    Columns 1-8 and the line ending are kept exactly as read.
    """
    content, ending = split_line_ending(line)
    cols = content.split('\t')
    if len(cols) != 9 + len(samples):
        raise ValueError(
            f"Record has {len(cols)} columns, header implies {9 + len(samples)}: {content[:120]}"
        )
    key = key_fn(cols[2], parse_info(cols[7]))
    keys = [] if cols[8] == '.' else cols[8].split(':')
    if SCORE_FIELD in keys:
        idx = keys.index(SCORE_FIELD)
        fmt = cols[8]
    else:
        idx = len(keys)
        fmt = ':'.join(keys + [SCORE_FIELD])
    fields = [
        annotate_sample_field(field, len(keys), idx, format_score(lookup_score(key, sample, table)))
        for field, sample in zip(cols[9:], samples)
    ]
    return '\t'.join(cols[:8] + [fmt] + fields) + ending


def annotate_text(
    header_lines: Sequence[str],
    body_lines: Iterable[str],
    caller: str,
    calls: pd.DataFrame,
) -> Iterator[str]:
    """Annotated lines of one text VCF: the header, then each record.

    Lazy, so a large VCF streams through; blank lines pass through unchanged.
    """
    samples = header_samples(header_lines)
    table = score_table(calls, caller)
    key_fn = event_key_fn(caller)
    yield from annotate_header(header_lines, pred_column(calls))
    for line in body_lines:
        if split_line_ending(line)[0]:
            yield annotate_record(line, key_fn, samples, table)
        else:
            yield line
