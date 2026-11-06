"""api.annotate, its io helpers and the `svensemble annotate` CLI."""

import gzip
import sys

import pandas as pd
import pysam
import pytest

from svensemble import api, cli
from svensemble import io as sv_io
from svensemble._constants import SAMPLE_ID_COL, SCORE_FIELD

PRED = 'delly_unbiased_xgb-0000_pred'

HEADER = (
    '##fileformat=VCFv4.2\n'
    '##contig=<ID=chr1,length=10000>\n'
    '##INFO=<ID=SVTYPE,Number=1,Type=String,Description="Type">\n'
    '##INFO=<ID=END,Number=1,Type=Integer,Description="End">\n'
    '##INFO=<ID=AF,Number=A,Type=Float,Description="AF">\n'
    '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n'
    '##FORMAT=<ID=DV,Number=1,Type=Integer,Description="Variant reads">\n'
)


def _vcf_text(samples=('S1',), ids=('DEL1', 'DEL2')):
    chrom = '\t'.join(['#CHROM', 'POS', 'ID', 'REF', 'ALT', 'QUAL', 'FILTER', 'INFO', 'FORMAT', *samples])
    body = ''.join(
        '\t'.join(['chr1', str(100 + 300 * i), id_, 'N', '<DEL>', '12.5', 'PASS',
                   f'SVTYPE=DEL;END={200 + 300 * i};AF=0.33333334', 'GT:DV', *(['0/1:5'] * len(samples))]) + '\n'
        for i, id_ in enumerate(ids)
    )
    return HEADER + chrom + '\n' + body


def _calls(rows):
    return pd.DataFrame(rows, columns=['delly_INFO_EVENT', SAMPLE_ID_COL, PRED])


def _write(path, text):
    path.write_text(text)
    return path


def _bgzip(src, dst):
    with pysam.BGZFile(str(dst), 'wb') as out:
        out.write(src.read_bytes())
    return dst


def _to_bcf(src, dst):
    with pysam.VariantFile(str(src)) as vin, pysam.VariantFile(str(dst), 'wb', header=vin.header) as vout:
        for rec in vin:
            vout.write(rec)
    return dst


def _assert_only_score_added(orig: str, out: str):
    o_lines, a_lines = orig.splitlines(True), out.splitlines(True)
    added = [line for line in a_lines if line.startswith(f'##FORMAT=<ID={SCORE_FIELD},')]
    assert len(added) == 1
    a_lines = [line for line in a_lines if line not in added]
    assert len(o_lines) == len(a_lines)
    for o, a in zip(o_lines, a_lines):
        if o.startswith('#'):
            assert o == a
            continue
        oc, ac = o.rstrip('\n').split('\t'), a.rstrip('\n').split('\t')
        assert oc[:8] == ac[:8]
        assert ac[8] == oc[8] + f':{SCORE_FIELD}'
        assert all(af.rsplit(':', 1)[0] == of for of, af in zip(oc[9:], ac[9:]))


class TestAnnotate:
    def test_text_vcf_byte_level(self, tmp_path):
        vcf = _write(tmp_path / 's.delly.vcf', _vcf_text())
        out = api.annotate(_calls([['DEL1', 'S1', 0.9]]), {'delly': vcf}, tmp_path / 'out')
        (path,) = out['delly']
        assert path.name == 'delly.s.delly.svensemble2.vcf'
        text = path.read_text()
        _assert_only_score_added(vcf.read_text(), text)
        assert text.splitlines()[-2].endswith('0/1:5:0.9') and text.splitlines()[-1].endswith('0/1:5:.')

    def test_gz_in_bgzf_out(self, tmp_path):
        plain = _write(tmp_path / 'p.vcf', _vcf_text())
        gz = _bgzip(plain, tmp_path / 's.vcf.gz')
        (path,) = api.annotate(_calls([['DEL2', 'S1', 0.5]]), {'delly': gz}, tmp_path / 'out')['delly']
        raw = path.read_bytes()
        assert raw[:4] == b'\x1f\x8b\x08\x04' and raw[12:14] == b'BC'
        assert raw.endswith(bytes.fromhex('1f8b08040000000000ff0600424302001b0003000000000000000000'))
        _assert_only_score_added(plain.read_text(), gzip.decompress(raw).decode())

    def test_bcf_in_bcf_out(self, tmp_path):
        bcf = _to_bcf(_write(tmp_path / 'p.vcf', _vcf_text(samples=('S1', 'S2'))), tmp_path / 's.bcf')
        calls = _calls([['DEL1', 'S2', 0.75]])
        (path,) = api.annotate(calls, {'delly': bcf}, tmp_path / 'out')['delly']
        assert path.suffix == '.bcf'
        with pysam.VariantFile(str(path)) as vf:
            recs = list(vf)
        assert recs[0].samples['S2'][SCORE_FIELD] == pytest.approx(0.75)
        assert recs[0].samples['S1'][SCORE_FIELD] is None
        assert recs[0].info['AF'] == pysam.VariantFile(str(bcf)).__next__().info['AF']

    def test_several_files_per_caller(self, tmp_path):
        a = _write(tmp_path / 'a.vcf', _vcf_text(samples=('S1',)))
        b = _write(tmp_path / 'b.vcf', _vcf_text(samples=('S2',)))
        other = _write(tmp_path / 'm.vcf', _vcf_text(samples=('S1', 'S2')))
        calls = _calls([['DEL1', 'S1', 0.1], ['DEL1', 'S2', 0.2]]).assign(manta_diploid_INFO_EVENT='DEL1')
        out = api.annotate(calls, {'delly': [a, b], 'manta_diploid': other}, tmp_path / 'out')
        assert [p.name for p in out['delly']] == ['delly.a.svensemble2.vcf', 'delly.b.svensemble2.vcf']
        assert out['delly'][1].read_text().splitlines()[-2].endswith(':0.2')

    def test_sample_in_two_files_of_one_caller_raises(self, tmp_path):
        a = _write(tmp_path / 'a.vcf', _vcf_text(samples=('S1',)))
        b = _write(tmp_path / 'b.vcf', _vcf_text(samples=('S1',)))
        with pytest.raises(ValueError, match='both'):
            api.annotate(_calls([['DEL1', 'S1', 0.1]]), {'delly': [a, b]}, tmp_path / 'out')

    def test_callers_with_different_samples_raise(self, tmp_path):
        a = _write(tmp_path / 'a.vcf', _vcf_text(samples=('S1',)))
        b = _write(tmp_path / 'b.vcf', _vcf_text(samples=('SAMPLE1',)))
        calls = _calls([['DEL1', 'S1', 0.1]]).assign(manta_diploid_INFO_EVENT='DEL1')
        with pytest.raises(ValueError, match='share one sample set'):
            api.annotate(calls, {'delly': a, 'manta_diploid': b}, tmp_path / 'out')

    def test_calls_sample_missing_from_vcfs_raises(self, tmp_path):
        a = _write(tmp_path / 'a.vcf', _vcf_text(samples=('S1',)))
        with pytest.raises(ValueError, match='not in the caller VCFs'):
            api.annotate(_calls([['DEL1', 'S9', 0.1]]), {'delly': a}, tmp_path / 'out')

    def test_legacy_calls_without_sample_column(self, tmp_path):
        a = _write(tmp_path / 'a.vcf', _vcf_text(samples=('S1',)))
        legacy = pd.DataFrame({'delly_INFO_EVENT': ['DEL1'], PRED: [0.4]})
        (path,) = api.annotate(legacy, {'delly': a}, tmp_path / 'out')['delly']
        assert path.read_text().splitlines()[-2].endswith(':0.4')
        two = _write(tmp_path / 'two.vcf', _vcf_text(samples=('S1', 'S2')))
        with pytest.raises(ValueError, match='single-sample'):
            api.annotate(legacy, {'delly': two}, tmp_path / 'out2')

    def test_sites_only_raises_before_writing(self, tmp_path):
        sites = HEADER + '#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\nchr1\t100\tDEL1\tN\t<DEL>\t.\tPASS\tSVTYPE=DEL\n'
        good = _write(tmp_path / 'good.vcf', _vcf_text())
        bad = _write(tmp_path / 'bad.vcf', sites)
        calls = _calls([['DEL1', 'S1', 0.1]]).assign(survivor_short_INFO_EVENT='DEL1')
        with pytest.raises(ValueError, match='sites-only'):
            api.annotate(calls, {'delly': good, 'survivor_short': bad}, tmp_path / 'out')
        assert not (tmp_path / 'out').exists() or not any((tmp_path / 'out').iterdir())

    def test_failure_mid_write_leaves_nothing(self, tmp_path):
        good = _write(tmp_path / 'good.vcf', _vcf_text())
        broken = _write(tmp_path / 'broken.vcf', _vcf_text() + 'chr1\t900\tDEL9\tN\t<DEL>\n')
        out_dir = tmp_path / 'out'
        calls = _calls([['DEL1', 'S1', 0.1]]).assign(manta_diploid_INFO_EVENT='DEL1')
        with pytest.raises(ValueError, match='columns'):
            api.annotate(calls, {'delly': good, 'manta_diploid': broken}, out_dir)
        assert list(out_dir.iterdir()) == []

    def test_caller_missing_from_calls_raises_before_writing(self, tmp_path):
        a = _write(tmp_path / 'a.vcf', _vcf_text())
        b = _write(tmp_path / 'b.vcf', _vcf_text())
        with pytest.raises(KeyError, match='manta_diploid_INFO_EVENT'):
            api.annotate(_calls([['DEL1', 'S1', 0.1]]), {'delly': a, 'manta_diploid': b}, tmp_path / 'out')
        assert not (tmp_path / 'out').exists()

    def test_unknown_caller_raises(self, tmp_path):
        a = _write(tmp_path / 'a.vcf', _vcf_text())
        with pytest.raises(KeyError):
            api.annotate(_calls([['DEL1', 'S1', 0.1]]), {'nope': a}, tmp_path / 'out')


class TestSampleSets:
    def test_na_like_names_rejected(self):
        with pytest.raises(ValueError, match='reserved'):
            api._compare_sample_sets({'a': {'NA'}})
        with pytest.raises(ValueError, match='reserved'):
            api._compare_sample_sets({'a': {'*'}})

    def test_sampleless_callers_skipped(self):
        assert api._compare_sample_sets({'a': {'S1'}, 'b': set()}) == ('S1',)


def test_read_calls_keeps_text(tmp_path):
    path = tmp_path / 'calls.csv'
    pd.DataFrame({'smoove_INFO_EVENT': ['21', None], SAMPLE_ID_COL: ['007', '007'], PRED: [0.1, 0.2]}).to_csv(path)
    calls = sv_io.read_calls(path)
    assert calls['smoove_INFO_EVENT'].iloc[0] == '21' and pd.isna(calls['smoove_INFO_EVENT'].iloc[1])
    assert list(calls[SAMPLE_ID_COL]) == ['007', '007']


def test_cli_annotate(tmp_path, monkeypatch):
    a = _write(tmp_path / 'a.vcf', _vcf_text(samples=('S1',)))
    b = _write(tmp_path / 'b.vcf', _vcf_text(samples=('S2',)))
    calls_csv = tmp_path / 'calls.csv'
    _calls([['DEL1', 'S1', 0.1], ['DEL2', 'S2', 0.2]]).to_csv(calls_csv)
    monkeypatch.setattr(sys, 'argv', [
        'svensemble', 'annotate', '--calls', str(calls_csv),
        '--callers', 'delly', 'delly', '--caller_vcfs', str(a), str(b),
        '--output_dir', str(tmp_path / 'out'),
    ])
    cli.main()
    assert sorted(p.name for p in (tmp_path / 'out').iterdir()) == [
        'delly.a.svensemble2.vcf', 'delly.b.svensemble2.vcf',
    ]
