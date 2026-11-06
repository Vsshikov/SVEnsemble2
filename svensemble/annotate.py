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
import struct
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from typing import BinaryIO

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


# -- BCF ----------------------------------------------------------------------
#
# BCF 2.2 (VCF spec, section "BCF specification"), on the decompressed stream:
#   magic 'BCF\2\2', uint32 l_text, header text (NUL-terminated), then records.
#   Record: uint32 l_shared, uint32 l_indiv, shared block, per-sample block.
#   Shared: int32 CHROM, int32 POS, int32 rlen, float QUAL,
#           uint32 n_allele<<16 | n_info, uint32 n_fmt<<24 | n_sample,
#           typed ID, n_allele typed alleles, typed FILTER vector,
#           n_info x (typed key, typed value).
#   Per-sample: n_fmt x (typed key, type descriptor, n_sample x values).
# Typed values start with a descriptor byte: low nibble type, high nibble
# count; count 15 means the real count follows as a typed integer.
# The shared block is copied verbatim; only ID and three INFO keys are decoded.

BCF_MAGIC = b'BCF\x02\x02'
_BCF_TYPE_SIZE = {0: 0, 1: 1, 2: 2, 3: 4, 5: 4, 7: 1}
_BCF_INT_FMT = {1: '<b', 2: '<h', 3: '<i'}
_BCF_FLOAT = 5
_BCF_FLOAT_MISSING = struct.pack('<I', 0x7F800001)
_BCF_KEY_INFO = ('EVENT', 'MATEID', 'SVTYPE')
_IDX_RE = re.compile(r'[<,]IDX=(\d+)[,>]')
_DICT_LINE_RE = re.compile(r'##(FILTER|INFO|FORMAT)=<ID=([^,>]+)')


def _read_exact(stream: BinaryIO, n: int) -> bytes:
    data = stream.read(n)
    if len(data) != n:
        raise ValueError(f"Truncated BCF: wanted {n} bytes, got {len(data)}")
    return data


def read_bcf_header(stream: BinaryIO) -> list[str]:
    """Header text lines (with endings) from a decompressed BCF stream positioned at the start."""
    magic = stream.read(len(BCF_MAGIC))
    if magic != BCF_MAGIC:
        raise ValueError(f"Not a BCF 2.2 stream (magic {magic!r})")
    (l_text,) = struct.unpack('<I', _read_exact(stream, 4))
    text = _read_exact(stream, l_text).split(b'\0', 1)[0].decode()
    return text.splitlines(keepends=True)


def iter_bcf_records(stream: BinaryIO) -> Iterator[tuple[bytes, bytes]]:
    """(shared, per-sample) blocks of each record, read after read_bcf_header."""
    while True:
        lengths = stream.read(8)
        if not lengths:
            return
        if len(lengths) != 8:
            raise ValueError("Truncated BCF record header")
        l_shared, l_indiv = struct.unpack('<II', lengths)
        yield _read_exact(stream, l_shared), _read_exact(stream, l_indiv)


def bcf_string_dict(header_lines: Sequence[str]) -> dict[str, int]:
    """FILTER/INFO/FORMAT ID -> BCF dictionary index.

    Uses IDX= attributes when the header carries them; otherwise the implicit
    order: PASS first, then each new ID in order of appearance.
    """
    out: dict[str, int] = {}
    implicit: dict[str, int] = {'PASS': 0}
    for line in header_lines:
        match = _DICT_LINE_RE.match(line)
        if not match:
            continue
        name = match.group(2)
        idx = _IDX_RE.search(line)
        if idx:
            out.setdefault(name, int(idx.group(1)))
        implicit.setdefault(name, len(implicit))
    return out if out else implicit


def _bcf_uses_idx(header_lines: Sequence[str]) -> bool:
    return any(_DICT_LINE_RE.match(line) and _IDX_RE.search(line) for line in header_lines)


def annotate_bcf_header(header_lines: Sequence[str], pred_col: str) -> tuple[bytes, int]:
    """(encoded BCF header, dictionary index of SCORE_FIELD).

    The text edit is annotate_header's. When the header carries IDX=
    attributes the SCORE_FIELD line gets one too: its existing index on a
    rerun, else one past the largest index in use.
    """
    before = bcf_string_dict(header_lines)
    lines = annotate_header(header_lines, pred_col)
    if SCORE_FIELD in before:
        idx = before[SCORE_FIELD]
    elif _bcf_uses_idx(header_lines):
        idx = max(before.values()) + 1
    else:
        idx = len(before)
    if _bcf_uses_idx(header_lines):
        lines = [
            _with_idx(line, idx) if line.startswith(_SCORE_HEADER_PREFIX) else line
            for line in lines
        ]
    after = bcf_string_dict(lines)
    if after.get(SCORE_FIELD) != idx:
        raise AssertionError(f"{SCORE_FIELD} dictionary index {after.get(SCORE_FIELD)} != {idx}")
    text = ''.join(lines).encode() + b'\0'
    return BCF_MAGIC + struct.pack('<I', len(text)) + text, idx


def _with_idx(line: str, idx: int) -> str:
    """The freshly written SCORE_FIELD header line with an IDX= attribute."""
    content, ending = split_line_ending(line)
    return f'{content[:-1]},IDX={idx}>{ending}'


def _typed_header(buf: bytes, off: int) -> tuple[int, int, int]:
    """(type, count, offset after the descriptor and any overflow count)."""
    desc = buf[off]
    typ, count = desc & 0x0F, desc >> 4
    off += 1
    if count == 15:
        ctyp, _, off = _typed_header(buf, off)
        (count,) = struct.unpack_from(_BCF_INT_FMT[ctyp], buf, off)
        off += _BCF_TYPE_SIZE[ctyp]
    return typ, count, off


def _read_typed_int(buf: bytes, off: int) -> tuple[int, int]:
    typ, count, off = _typed_header(buf, off)
    if count != 1 or typ not in _BCF_INT_FMT:
        raise ValueError(f"Expected a single typed integer (type {typ}, count {count})")
    (value,) = struct.unpack_from(_BCF_INT_FMT[typ], buf, off)
    return value, off + _BCF_TYPE_SIZE[typ]


def _skip_typed(buf: bytes, off: int) -> int:
    typ, count, off = _typed_header(buf, off)
    return off + count * _BCF_TYPE_SIZE[typ]


def _typed_text(buf: bytes, off: int) -> tuple[str, int]:
    """A typed value as VCF text (chars as a string, numbers comma-joined)."""
    typ, count, off = _typed_header(buf, off)
    end = off + count * _BCF_TYPE_SIZE[typ]
    if typ == 7:
        return buf[off:end].split(b'\0', 1)[0].decode(), end
    if typ == 0 or count == 0:
        return '', end
    fmt = '<f' if typ == _BCF_FLOAT else _BCF_INT_FMT[typ]
    values = struct.unpack_from(f'<{count}{fmt[1]}', buf, off)
    return ','.join(str(v) for v in values), end


def bcf_record_key_fields(shared: bytes, names: Mapping[int, str]) -> tuple[str, dict[str, str]]:
    """(ID, {INFO key: text}) of one record, decoding only EVENT, MATEID and SVTYPE."""
    (n_allele_info,) = struct.unpack_from('<I', shared, 16)
    n_allele, n_info = n_allele_info >> 16, n_allele_info & 0xFFFF
    id_, off = _typed_text(shared, 24)
    for _ in range(n_allele):
        off = _skip_typed(shared, off)
    off = _skip_typed(shared, off)  # FILTER
    info: dict[str, str] = {}
    for _ in range(n_info):
        key, off = _read_typed_int(shared, off)
        name = names.get(key)
        if name in _BCF_KEY_INFO:
            info[name], off = _typed_text(shared, off)
        else:
            off = _skip_typed(shared, off)
    return (id_ or '.'), info


def _typed_int_bytes(value: int) -> bytes:
    if value <= 127:
        return b'\x11' + struct.pack('<b', value)
    if value <= 32767:
        return b'\x12' + struct.pack('<h', value)
    return b'\x13' + struct.pack('<i', value)


def _float_bytes(score: float | None) -> bytes:
    if score is None or np.isnan(score):
        return _BCF_FLOAT_MISSING
    return struct.pack('<f', score)


def bcf_set_format_float(
    shared: bytes,
    indiv: bytes,
    key_idx: int,
    values: Sequence[float | None],
) -> tuple[bytes, bytes]:
    """(shared, per-sample) blocks with a one-float FORMAT field set per sample.

    An existing entry for *key_idx* (from an earlier run) is overwritten in
    place; otherwise one entry is appended and n_fmt in the shared block goes
    up by one. Nothing else in either block changes.
    """
    (n_fmt_sample,) = struct.unpack_from('<I', shared, 20)
    n_fmt, n_sample = n_fmt_sample >> 24, n_fmt_sample & 0xFFFFFF
    if len(values) != n_sample:
        raise ValueError(f"{len(values)} values for {n_sample} samples")
    payload = b''.join(_float_bytes(v) for v in values)
    off = 0
    for _ in range(n_fmt):
        key, off = _read_typed_int(indiv, off)
        typ, count, data_off = _typed_header(indiv, off)
        end = data_off + count * _BCF_TYPE_SIZE[typ] * n_sample
        if key == key_idx:
            if (typ, count) != (_BCF_FLOAT, 1):
                raise ValueError(f"Existing {SCORE_FIELD} entry is type {typ} x{count}, not one float")
            return shared, indiv[:data_off] + payload + indiv[end:]
        off = end
    if n_fmt == 255:
        raise ValueError("Record already has 255 FORMAT fields; BCF cannot hold another")
    new_shared = shared[:20] + struct.pack('<I', ((n_fmt + 1) << 24) | n_sample) + shared[24:]
    entry = _typed_int_bytes(key_idx) + bytes([0x10 | _BCF_FLOAT]) + payload
    return new_shared, indiv + entry


def annotate_bcf(
    header_lines: Sequence[str],
    records: Iterable[tuple[bytes, bytes]],
    caller: str,
    calls: pd.DataFrame,
) -> Iterator[bytes]:
    """Annotated BCF stream (uncompressed): the header, then each record."""
    samples = header_samples(header_lines)
    table = score_table(calls, caller)
    key_fn = event_key_fn(caller)
    header, key_idx = annotate_bcf_header(header_lines, pred_column(calls))
    names = {idx: name for name, idx in bcf_string_dict(header_lines).items()}
    yield header
    for shared, indiv in records:
        id_, info = bcf_record_key_fields(shared, names)
        key = key_fn(id_, info)
        scores = [lookup_score(key, sample, table) for sample in samples]
        new_shared, new_indiv = bcf_set_format_float(shared, indiv, key_idx, scores)
        yield struct.pack('<II', len(new_shared), len(new_indiv)) + new_shared + new_indiv
