"""Unit tests for svensemble.annotate (pure text and BCF annotation, no file I/O)."""

import gzip
import io
import os
import struct
from pathlib import Path

import pandas as pd
import pysam
import pytest

from svensemble import annotate as ann
from svensemble._caller_params import CALLSET_PARAM_DICT
from svensemble._constants import SAMPLE_ID_COL, SCORE_FIELD
from svensemble.callset_specific import parse_recipes

PRED = 'delly_unbiased_xgb-0000_pred'

HEADER = [
    '##fileformat=VCFv4.2\n',
    '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n',
    '##FORMAT=<ID=DV,Number=1,Type=Integer,Description="Variant reads">\n',
    '#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS1\tS2\n',
]


def _calls(rows, caller='delly'):
    return pd.DataFrame(rows, columns=[f'{caller}_INFO_EVENT', SAMPLE_ID_COL, PRED])


# -- Scores -------------------------------------------------------------------

class TestScores:
    def test_pred_column_requires_exactly_one(self):
        with pytest.raises(ValueError):
            ann.pred_column(pd.DataFrame(columns=['a_pred', 'b_pred']))
        assert ann.pred_column(pd.DataFrame(columns=['x', PRED])) == PRED

    @pytest.mark.parametrize('raw, expected', [
        ('21.0', '21'), ('21', '21'), ('DEL0001', 'DEL0001'), ('21.5', '21.5'), ('a21.0', 'a21.0'),
    ])
    def test_normalise_event_id(self, raw, expected):
        assert ann.normalise_event_id(raw) == expected

    def test_score_table_drops_missing_events_and_normalises(self):
        calls = _calls([['DEL1', 'S1', 0.9], [None, 'S1', 0.1], [21.0, 'S2', 0.4]])
        assert ann.score_table(calls, 'delly') == {('DEL1', 'S1'): 0.9, ('21', 'S2'): 0.4}

    def test_score_table_requires_sample_column(self):
        calls = pd.DataFrame({'delly_INFO_EVENT': ['DEL1'], PRED: [0.5]})
        with pytest.raises(KeyError, match=SAMPLE_ID_COL):
            ann.score_table(calls, 'delly')

    def test_lookup_direct_and_missing(self):
        table = {('DEL1', 'S1'): 0.9}
        assert ann.lookup_score('DEL1', 'S1', table) == 0.9
        assert ann.lookup_score('DEL1', 'S2', table) is None
        assert ann.lookup_score(None, 'S1', table) is None

    def test_lookup_inversion_mean_of_present_halves(self):
        both = {('INV1_e1', 'S1'): 0.2, ('INV1_e2', 'S1'): 0.6}
        assert ann.lookup_score('INV1', 'S1', both) == pytest.approx(0.4)
        one = {('INV1_e2', 'S1'): 0.6}
        assert ann.lookup_score('INV1', 'S1', one) == 0.6

    def test_format_score(self):
        assert ann.format_score(None) == '.'
        assert ann.format_score(float('nan')) == '.'
        assert ann.format_score(0.123456789) == '0.123457'


# -- Event keys ---------------------------------------------------------------

class TestEventKeys:
    def test_every_recipe_source_has_a_rule(self):
        assert set(parse_recipes) <= set(ann.EVENT_KEY)

    def test_every_caller_with_a_recipe_resolves(self):
        for caller, params in CALLSET_PARAM_DICT.items():
            if params.get('source') in parse_recipes:
                assert ann.event_key_fn(caller) is ann.EVENT_KEY[params['source']]

    def test_unknown_caller_raises(self):
        with pytest.raises(KeyError):
            ann.event_key_fn('not_a_caller')

    def test_parse_info(self):
        assert ann.parse_info('.') == {}
        assert ann.parse_info('SVTYPE=BND;IMPRECISE;MATEID=a,b') == {
            'SVTYPE': 'BND', 'IMPRECISE': '', 'MATEID': 'a,b',
        }

    @pytest.mark.parametrize('caller, id_, info, expected', [
        ('gridss', 'gridss1o', 'EVENT=gridss1;SVTYPE=BND', 'gridss1'),
        ('manta_diploid', 'MantaBND:1:0:1:0:0:0:1', 'SVTYPE=BND;MATEID=MantaBND:1:0:1:0:0:0:0',
         'MantaBND:1:0:1:0:0:0:0'),
        ('manta_diploid', 'MantaBND:1:0:1:0:0:0:0', 'SVTYPE=BND;MATEID=MantaBND:1:0:1:0:0:0:1',
         'MantaBND:1:0:1:0:0:0:0'),
        ('manta_candidate', 'MantaINV:5', 'SVTYPE=INV', 'MantaINV:5'),
        ('manta_diploid', 'MantaDEL:3', 'SVTYPE=DEL;MATEID=zzz', 'MantaDEL:3'),
        ('delly', 'BND00000012', 'SVTYPE=BND', 'BND00000012'),
        ('svaba', '3859077473:1', 'SVTYPE=BND', '3859077473'),
        ('svaba_H', '3859077473:2', 'SVTYPE=BND', '3859077473'),
        ('smoove', '21_1', 'SVTYPE=BND', '21'),
        ('smoove', '21', 'SVTYPE=DEL', '21'),
        ('sniffles', 'Sniffles2.DEL.1', 'SVTYPE=DEL', 'Sniffles2.DEL.1'),
        ('severus', 'severus_DEL1', 'SVTYPE=DEL', 'severus_DEL1'),
        ('cutesv', 'cuteSV.DEL.0', 'SVTYPE=DEL', 'cuteSV.DEL.0'),
        ('survivor_short', '7', 'SVTYPE=DEL', '7'),
    ])
    def test_rules(self, caller, id_, info, expected):
        assert ann.event_key_fn(caller)(id_, ann.parse_info(info)) == expected


# -- Text codec ---------------------------------------------------------------

class TestHeader:
    def test_inserts_before_chrom(self):
        out = ann.annotate_header(HEADER, PRED)
        assert out[:3] == HEADER[:3]
        assert out[3].startswith(f'##FORMAT=<ID={SCORE_FIELD},Number=1,Type=Float,')
        assert PRED in out[3] and out[3].endswith('\n')
        assert out[4] == HEADER[3]

    def test_replaces_existing_line(self):
        once = ann.annotate_header(HEADER, 'old_pred')
        twice = ann.annotate_header(once, PRED)
        assert len(twice) == len(once)
        assert sum(SCORE_FIELD in line for line in twice) == 1
        assert PRED in twice[3]

    def test_keeps_crlf(self):
        crlf = [line.replace('\n', '\r\n') for line in HEADER]
        assert ann.annotate_header(crlf, PRED)[3].endswith('\r\n')

    def test_sites_only_raises(self):
        sites = HEADER[:-1] + ['#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n']
        with pytest.raises(ValueError, match='sites-only'):
            ann.annotate_header(sites, PRED)


class TestSampleField:
    @pytest.mark.parametrize('field, n_keys, idx, expected', [
        ('0/1:3', 2, 2, '0/1:3:0.5'),
        ('0/1', 3, 3, '0/1:.:.:0.5'),
        ('0/1:3:0.1', 3, 2, '0/1:3:0.5'),
        ('.', 0, 0, '0.5'),
    ])
    def test_pad_then_set(self, field, n_keys, idx, expected):
        assert ann.annotate_sample_field(field, n_keys, idx, '0.5') == expected


class TestRecord:
    TABLE = {('DEL1', 'S1'): 0.9, ('DEL1', 'S2'): 0.25}

    def _line(self, fmt='GT:DV', s1='0/1:5', s2='0/0', ending='\n'):
        return '\t'.join(['chr1', '100', 'DEL1', 'N', '<DEL>', '.', 'PASS', 'SVTYPE=DEL;END=200', fmt, s1, s2]) + ending

    def test_appends_and_pads(self):
        out = ann.annotate_record(self._line(), ann._id_key, ['S1', 'S2'], self.TABLE)
        cols = out.rstrip('\n').split('\t')
        assert cols[:8] == self._line().rstrip('\n').split('\t')[:8]
        assert cols[8:] == ['GT:DV:SVENSSC', '0/1:5:0.9', '0/0:.:0.25']

    def test_unscored_sample_gets_dot(self):
        out = ann.annotate_record(self._line(), ann._id_key, ['S1', 'S3'], self.TABLE)
        assert out.rstrip('\n').split('\t')[-1] == '0/0:.:.'

    def test_rerun_replaces_in_place(self):
        once = ann.annotate_record(self._line(), ann._id_key, ['S1', 'S2'], {('DEL1', 'S1'): 0.1})
        twice = ann.annotate_record(once, ann._id_key, ['S1', 'S2'], self.TABLE)
        assert twice.rstrip('\n').split('\t')[8:] == ['GT:DV:SVENSSC', '0/1:5:0.9', '0/0:.:0.25']

    def test_dot_format(self):
        out = ann.annotate_record(self._line(fmt='.', s1='.', s2='.'), ann._id_key, ['S1', 'S2'], self.TABLE)
        assert out.rstrip('\n').split('\t')[8:] == ['SVENSSC', '0.9', '0.25']

    def test_keeps_crlf_and_unterminated(self):
        assert ann.annotate_record(self._line(ending='\r\n'), ann._id_key, ['S1', 'S2'], self.TABLE).endswith('\r\n')
        assert not ann.annotate_record(self._line(ending=''), ann._id_key, ['S1', 'S2'], self.TABLE).endswith('\n')

    def test_column_count_mismatch_raises(self):
        with pytest.raises(ValueError, match='columns'):
            ann.annotate_record(self._line(), ann._id_key, ['S1'], self.TABLE)


class TestAnnotateText:
    def test_end_to_end_only_score_changes(self):
        body = [
            '\t'.join(['chr1', '100', 'DEL1', 'N', '<DEL>', '12.5', 'PASS', 'SVTYPE=DEL;END=200', 'GT:DV', '0/1:5', '0/0:0']) + '\n',
            '\t'.join(['chr1', '500', 'INV1', 'N', '<INV>', '.', 'LowQual', 'SVTYPE=INV;END=900', 'GT', '0/1', '1/1']) + '\n',
            '\t'.join(['chr1', '900', 'DEL9', 'N', '<DEL>', '.', 'PASS', 'SVTYPE=DEL;END=950', 'GT', '0/1', '0/1']) + '\n',
        ]
        calls = _calls([
            ['DEL1', 'S1', 0.9], ['DEL1', 'S2', 0.1],
            ['INV1_e1', 'S1', 0.2], ['INV1_e2', 'S1', 0.4],
        ])
        out = list(ann.annotate_text(HEADER, body, 'delly', calls))
        assert len(out) == len(HEADER) + 1 + len(body)
        records = [line.rstrip('\n').split('\t') for line in out[len(HEADER) + 1:]]
        for rec, orig in zip(records, body):
            assert rec[:8] == orig.rstrip('\n').split('\t')[:8]
        assert records[0][8:] == ['GT:DV:SVENSSC', '0/1:5:0.9', '0/0:0:0.1']
        assert records[1][8:] == ['GT:SVENSSC', '0/1:0.3', '1/1:.']
        assert records[2][8:] == ['GT:SVENSSC', '0/1:.', '0/1:.']


# -- BCF codec ----------------------------------------------------------------


BCF_VCF_TEXT = (
    '##fileformat=VCFv4.2\n'
    '##contig=<ID=chr1,length=10000>\n'
    '##INFO=<ID=SVTYPE,Number=1,Type=String,Description="Type">\n'
    '##INFO=<ID=END,Number=1,Type=Integer,Description="End">\n'
    '##INFO=<ID=MATEID,Number=.,Type=String,Description="Mate">\n'
    '##INFO=<ID=AF,Number=A,Type=Float,Description="AF">\n'
    '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n'
    '##FORMAT=<ID=DV,Number=1,Type=Integer,Description="Variant reads">\n'
    '#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS1\tS2\n'
    'chr1\t100\tDEL1\tN\t<DEL>\t12.5\tPASS\tSVTYPE=DEL;END=200;AF=0.33333334\tGT:DV\t0/1:5\t0/0:0\n'
    'chr1\t500\tINV1\tN\t<INV>\t.\t.\tSVTYPE=INV;END=900\tGT\t0/1\t1/1\n'
    'chr1\t950\tDEL9\tN\t<DEL>\t.\tPASS\tSVTYPE=DEL;END=990\tGT:DV\t./.:.\t0/1:3\n'
)


def _make_bcf(tmp_path, text=BCF_VCF_TEXT):
    vcf = tmp_path / 'in.vcf'
    vcf.write_text(text)
    bcf = tmp_path / 'in.bcf'
    with pysam.VariantFile(str(vcf)) as vin, pysam.VariantFile(str(bcf), 'wb', header=vin.header) as vout:
        for rec in vin:
            vout.write(rec)
    return bcf


def _bcf_parts(path_or_bytes):
    data = gzip.decompress(path_or_bytes.read_bytes()) if hasattr(path_or_bytes, 'read_bytes') else path_or_bytes
    stream = io.BytesIO(data)
    return ann.read_bcf_header(stream), list(ann.iter_bcf_records(stream))


def _annotated_bcf(tmp_path, bcf, calls, caller='delly'):
    header, records = _bcf_parts(bcf)
    out = b''.join(ann.annotate_bcf(header, records, caller, calls))
    path = tmp_path / 'out.bcf'
    with pysam.BGZFile(str(path), 'wb') as fh:
        fh.write(out)
    return path


BCF_CALLS = [['DEL1', 'S1', 0.9], ['DEL1', 'S2', 0.25], ['INV1_e1', 'S2', 0.2], ['INV1_e2', 'S2', 0.4]]


class TestBcf:
    def test_shared_block_verbatim_and_indiv_extended(self, tmp_path):
        bcf = _make_bcf(tmp_path)
        out = _annotated_bcf(tmp_path, bcf, _calls(BCF_CALLS))
        _, before = _bcf_parts(bcf)
        _, after = _bcf_parts(out)
        assert len(before) == len(after)
        for (s0, i0), (s1, i1) in zip(before, after):
            assert s1[:20] == s0[:20] and s1[24:] == s0[24:]
            n0, n1 = struct.unpack_from('<I', s0, 20)[0], struct.unpack_from('<I', s1, 20)[0]
            assert n1 >> 24 == (n0 >> 24) + 1 and n1 & 0xFFFFFF == n0 & 0xFFFFFF
            assert i1.startswith(i0)

    def test_pysam_reads_same_records_plus_score(self, tmp_path):
        bcf = _make_bcf(tmp_path)
        out = _annotated_bcf(tmp_path, bcf, _calls(BCF_CALLS))
        with pysam.VariantFile(str(bcf)) as a, pysam.VariantFile(str(out)) as b:
            assert list(b.header.samples) == ['S1', 'S2']
            for ra, rb in zip(a, b):
                assert str(ra).split('\t')[:8] == str(rb).split('\t')[:8]
                for s in ('S1', 'S2'):
                    for key in ra.samples[s].keys():
                        assert ra.samples[s][key] == rb.samples[s][key]
            got = [[rec.samples[s][SCORE_FIELD] for s in ('S1', 'S2')] for rec in pysam.VariantFile(str(out))]
        assert got[0] == [pytest.approx(0.9), pytest.approx(0.25)]
        assert got[1] == [None, pytest.approx(0.3)]
        assert got[2] == [None, None]

    def test_rerun_overwrites_in_place(self, tmp_path):
        bcf = _make_bcf(tmp_path)
        first = _annotated_bcf(tmp_path, bcf, _calls([['DEL1', 'S1', 0.1]]))
        header, records = _bcf_parts(first)
        second = b''.join(ann.annotate_bcf(header, records, 'delly', _calls(BCF_CALLS)))
        h2, r2 = _bcf_parts(second)
        assert sum(line.startswith(f'##FORMAT=<ID={SCORE_FIELD},') for line in h2) == 1
        assert [len(s) for s, _ in r2] == [len(s) for s, _ in records]
        assert [len(i) for _, i in r2] == [len(i) for _, i in records]
        assert struct.unpack_from('<f', r2[0][1], len(r2[0][1]) - 8)[0] == pytest.approx(0.9)

    def test_idx_header(self, tmp_path):
        bcf = _make_bcf(tmp_path)
        header, _ = _bcf_parts(bcf)
        names = ann.bcf_string_dict(header)
        new_header, idx = ann.annotate_bcf_header(header, PRED)
        if ann._bcf_uses_idx(header):
            assert idx == max(names.values()) + 1
            assert f',IDX={idx}>' in new_header.decode(errors='ignore')
        assert ann.bcf_string_dict(new_header[9:].split(b'\0', 1)[0].decode().splitlines(True))[SCORE_FIELD] == idx

    def test_record_key_fields(self, tmp_path):
        bcf = _make_bcf(tmp_path)
        header, records = _bcf_parts(bcf)
        names = {i: n for n, i in ann.bcf_string_dict(header).items()}
        assert ann.bcf_record_key_fields(records[0][0], names) == ('DEL1', {'SVTYPE': 'DEL'})

    def test_not_bcf_raises(self):
        with pytest.raises(ValueError, match='BCF'):
            ann.read_bcf_header(io.BytesIO(b'##fileformat=VCFv4.2\n'))


E2E_DELLY_BCF = '/vast/ai_projects/Papenfuss_SVEnsemble2/svensemble2/cache/e2e/inputs/delly.chr1.bcf'


@pytest.mark.skipif(not os.path.exists(E2E_DELLY_BCF), reason='e2e Delly BCF not present')
def test_e2e_delly_bcf_round_trip(tmp_path):
    bcf = Path(E2E_DELLY_BCF)
    header, records = _bcf_parts(bcf)
    sample = ann.header_samples(header)[0]
    names = {i: n for n, i in ann.bcf_string_dict(header).items()}
    ids = [ann.bcf_record_key_fields(s, names)[0] for s, _ in records]
    calls = _calls([[ids[0], sample, 0.75], [ids[-1], sample, 0.5]])
    out = _annotated_bcf(tmp_path, bcf, calls)
    with pysam.VariantFile(str(bcf)) as a, pysam.VariantFile(str(out)) as b:
        n = 0
        for ra, rb in zip(a, b):
            assert str(ra).split('\t')[:8] == str(rb).split('\t')[:8]
            assert {k: ra.samples[sample][k] for k in ra.samples[sample].keys()} == \
                {k: rb.samples[sample][k] for k in ra.samples[sample].keys()}
            n += 1
    assert n == len(records)
    scores = [rec.samples[sample][SCORE_FIELD] for rec in pysam.VariantFile(str(out))]
    assert scores[0] == pytest.approx(0.75) and scores[-1] == pytest.approx(0.5)
    assert sum(s is not None for s in scores) == 2
