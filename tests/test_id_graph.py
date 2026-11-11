"""Tests for svensemble.id_graph, its persistence, sample harmonisation and VCF records grouped by SV ID."""

import igraph as ig
import numpy as np
import pandas as pd
import pysam
import pytest

from svensemble import io as sv_io
from svensemble._constants import GROUND_TRUTH_SAMPLE, SAMPLE_ID_COL
from svensemble.chunk_match import flatten_idgraph, inner_join_bps
from svensemble.id_graph import (
    IdGraph,
    decode_pair,
    drop_edges,
    edge_table,
    encode_pair,
    node_table,
    pair_codes,
)
from svensemble.matched_breakpoints import _harmonise_samples, get_svensemble2_calls
from svensemble.vcfwrite import _pair_inversion_halves, write_vcf

from .test_chunk_match import _bps, _row

PRED = 'delly+manta_diploid_m-0_pred'


def _idg(nodes, edges=()):
    """IdGraph from (source, event, sample, svensemble_id) nodes and (i, j) edges."""
    sources = tuple(sorted({n[0] for n in nodes}))
    graph = ig.Graph(n=len(nodes), edges=list(edges), directed=False)
    graph.vs['name'] = [tuple(n[:3]) for n in nodes]
    graph.vs['svensemble_id'] = [n[3] for n in nodes]
    idx = np.array([sources.index(n[0]) for n in nodes], dtype=np.int64)
    u = np.array([e[0] for e in edges], dtype=np.int64)
    v = np.array([e[1] for e in edges], dtype=np.int64)
    graph.es['pair'] = pair_codes(idx[u], idx[v]).tolist()
    graph.es['same_sample'] = [nodes[a][2] == nodes[b][2] for a, b in edges]
    return IdGraph(graph=graph, sources=sources)


def _site(pos, span, chrom='chr1', svtype='DEL', ci=0):
    end = pos + span
    return {
        'CHROM_bnd1': chrom, 'POS_bnd1': pos, 'ALT_bnd1': f'N[{chrom}:{end}[',
        'CHROM_bnd2': chrom, 'POS_bnd2': end, 'ALT_bnd2': f']{chrom}:{pos}]N',
        'INFO_CIPOS_l_bnd1': -ci, 'INFO_CIPOS_r_bnd1': ci,
        'INFO_CIPOS_l_bnd2': -ci, 'INFO_CIPOS_r_bnd2': ci,
        'INFO_INFERRED_SVTYPE': svtype, 'INFO_SVLEN': span,
    }


def _call(event, sample, delly, manta, score, **site):
    return {
        'INFO_EVENT': event, SAMPLE_ID_COL: sample,
        'delly_INFO_EVENT': delly, 'manta_diploid_INFO_EVENT': manta,
        PRED: score, **_site(**site),
    }


# Joint-called delly D1 in S1 and S2, matched by manta M1 only in S1; G1 is ground truth
NODES = [
    ('delly', 'D1', 'S1', 'svensemble2_0'),
    ('manta_diploid', 'M1', 'S1', 'svensemble2_0'),
    ('delly', 'D1', 'S2', 'svensemble2_0'),
    ('delly', 'D9', 'S2', 'svensemble2_1'),
    ('gnomad', 'G1', GROUND_TRUTH_SAMPLE, None),
]
EDGES = [(0, 1), (0, 2), (1, 2), (0, 4)]


# ---------------------------------------------------------------------------
# Container
# ---------------------------------------------------------------------------

class TestIdGraph:
    def test_valid_graph(self):
        idg = _idg(NODES, EDGES)
        assert idg.sources == ('delly', 'gnomad', 'manta_diploid')

    def test_unsorted_sources_rejected(self):
        idg = _idg(NODES, EDGES)
        with pytest.raises(ValueError, match='sorted'):
            IdGraph(graph=idg.graph, sources=tuple(reversed(idg.sources)))

    def test_pair_code_out_of_range_rejected(self):
        graph = _idg(NODES, EDGES).graph.copy()
        graph.es['pair'] = [99] * graph.ecount()
        with pytest.raises(ValueError, match='out of range'):
            IdGraph(graph=graph, sources=('delly', 'gnomad', 'manta_diploid'))

    def test_missing_attribute_rejected(self):
        graph = _idg(NODES, EDGES).graph.copy()
        del graph.vs['name']
        with pytest.raises(ValueError, match='name'):
            IdGraph(graph=graph, sources=('delly', 'gnomad', 'manta_diploid'))

    def test_pair_codes_round_trip(self):
        idg = _idg(NODES, EDGES)
        for a in idg.sources:
            for b in idg.sources:
                assert decode_pair(idg, encode_pair(idg, a, b)) == tuple(sorted((a, b)))

    def test_drop_edges_keeps_nodes_and_original(self):
        idg = _idg(NODES, EDGES)
        pruned = drop_edges(idg, [0, 3])
        assert pruned.graph.vcount() == idg.graph.vcount()
        assert pruned.graph.ecount() == 2
        assert idg.graph.ecount() == 4
        assert pruned.sources == idg.sources


# ---------------------------------------------------------------------------
# Persistence and lookups
# ---------------------------------------------------------------------------

def test_write_read_round_trip(tmp_path):
    nodes = NODES + [('delly', '007', 'S1', 'svensemble2_2')]
    idg = _idg(nodes, EDGES)
    back = sv_io.read_id_graph(sv_io.write_id_graph(idg, tmp_path / sv_io.ID_GRAPH))
    assert back.sources == idg.sources
    pd.testing.assert_frame_equal(node_table(back), node_table(idg), check_dtype=False)
    pd.testing.assert_frame_equal(edge_table(back), edge_table(idg), check_dtype=False)
    assert node_table(back).loc[5, 'event'] == '007'
    assert node_table(back).loc[4, 'svensemble_id'] is None


# ---------------------------------------------------------------------------
# Harmonisation
# ---------------------------------------------------------------------------

def _calls():
    return pd.DataFrame([
        _call('svensemble2_a', 'S1', 'D1', 'M1', 0.9, pos=100, span=400, ci=5),
        _call('svensemble2_b', 'S2', 'D1', None, 0.3, pos=110, span=402, ci=0),
        _call('svensemble2_c', 'S2', 'D9', None, 0.5, pos=9000, span=100),
    ]).assign(svensemble_id=['svensemble2_0', 'svensemble2_0', 'svensemble2_1'])


class TestHarmoniseSamples:
    def test_single_sample_returned_as_is(self):
        calls = _calls().iloc[[0]]
        assert _harmonise_samples(calls) is calls

    def test_mean(self):
        out = _harmonise_samples(_calls(), how='mean')
        assert list(out.columns) == list(_calls().columns)
        assert len(out) == 3
        shared = out.iloc[:2]
        assert shared['POS_bnd1'].tolist() == [105, 105]
        assert shared['POS_bnd2'].tolist() == [506, 506]
        # CI envelope around the new positions: [95, 112] around 105 on bnd1
        assert shared['INFO_CIPOS_l_bnd1'].tolist() == [-10, -10]
        assert shared['INFO_CIPOS_r_bnd1'].tolist() == [5, 5]
        assert shared['ALT_bnd1'].tolist() == ['N[chr1:506[', 'N[chr1:506[']
        assert shared['ALT_bnd2'].tolist() == [']chr1:105]N', ']chr1:105]N']
        # per-sample columns untouched
        assert shared[PRED].tolist() == [0.9, 0.3]
        assert shared['INFO_EVENT'].tolist() == ['svensemble2_a', 'svensemble2_b']
        pd.testing.assert_series_equal(out.iloc[2], _calls().iloc[2])

    def test_max_score(self):
        out = _harmonise_samples(_calls(), how='max_score')
        shared = out.iloc[:2]
        assert shared['POS_bnd1'].tolist() == [100, 100]
        assert shared['INFO_CIPOS_r_bnd1'].tolist() == [5, 5]
        assert shared[PRED].tolist() == [0.9, 0.3]

    def test_unknown_mode_rejected(self):
        with pytest.raises(ValueError, match='how'):
            _harmonise_samples(_calls(), how='median')


# ---------------------------------------------------------------------------
# VCF records grouped by SV ID
# ---------------------------------------------------------------------------

def _records(path):
    with pysam.VariantFile(str(path)) as vcf:
        samples = list(vcf.header.samples)
        return [(r.id, r.pos, {s: r.samples[s]['SVENSSC'] for s in samples}) for r in vcf]


class TestGroupBySvId:
    def test_one_record_per_sv(self, tmp_path):
        calls = _harmonise_samples(_calls(), how='max_score')
        write_vcf(calls, tmp_path / 'out.vcf', score_col=PRED, group_by='svensemble_id')
        recs = _records(tmp_path / 'out.vcf')
        assert [(r[0], r[1]) for r in recs] == [('svensemble2_0', 100), ('svensemble2_1', 9000)]
        assert recs[0][2] == pytest.approx({'S1': 0.9, 'S2': 0.3})
        assert recs[1][2]['S1'] is None

    def test_unharmonised_sites_raise(self, tmp_path):
        with pytest.raises(ValueError, match='harmonise'):
            write_vcf(_calls(), tmp_path / 'out.vcf', score_col=PRED, group_by='svensemble_id')

    def _inv_calls(self, rows):
        out = []
        for event, delly, sample, sv, pos, half in rows:
            span = 1000
            site = _site(pos, span, svtype='INV')
            if half == 2:
                site['ALT_bnd1'], site['ALT_bnd2'] = f'[chr1:{pos + span}[N', f'[chr1:{pos}[N'
            else:
                site['ALT_bnd1'], site['ALT_bnd2'] = f'N]chr1:{pos + span}]', f'N]chr1:{pos}]'
            out.append({'INFO_EVENT': event, SAMPLE_ID_COL: sample, 'delly_INFO_EVENT': delly,
                        'manta_diploid_INFO_EVENT': None, 'svensemble_id': sv, PRED: 0.5, **site})
        return pd.DataFrame(out)

    def test_paired_inversion_halves_collapse(self, tmp_path):
        calls = self._inv_calls([
            ('svensemble2_x', 'I1_e1', 'S1', 'svensemble2_0', 100, 1),
            ('svensemble2_y', 'I1_e2', 'S1', 'svensemble2_1', 101, 2),
            ('svensemble2_z', 'L_e1', 'S1', 'svensemble2_2', 5000, 1),
        ])
        write_vcf(calls, tmp_path / 'out.vcf', score_col=PRED, group_by='svensemble_id')
        assert [r[0] for r in _records(tmp_path / 'out.vcf')] == ['svensemble2_0', 'svensemble2_2']

    def test_disagreeing_inversion_pairs_kept_apart(self, tmp_path):
        calls = self._inv_calls([
            ('svensemble2_a', 'I1_e1', 'S1', 'svensemble2_0', 100, 1),
            ('svensemble2_b', 'I1_e2', 'S1', 'svensemble2_1', 101, 2),
            ('svensemble2_c', 'I2_e1', 'S2', 'svensemble2_0', 100, 1),
            ('svensemble2_d', 'I2_e2', 'S2', 'svensemble2_2', 2101, 2),
        ])
        write_vcf(calls, tmp_path / 'out.vcf', score_col=PRED, group_by='svensemble_id')
        assert sorted(r[0] for r in _records(tmp_path / 'out.vcf')) == [
            'svensemble2_0', 'svensemble2_1', 'svensemble2_2',
        ]

    def test_pairing_joins_halves_of_one_sample_under_the_e1_key(self):
        # I1_e1 and I1_e2 of one sample sit in two SVs: both records map to
        # the e1 half's key. L_e1 has no partner and is left out.
        calls = self._inv_calls([
            ('svensemble2_x', 'I1_e1', 'S1', 'svensemble2_0', 100, 1),
            ('svensemble2_y', 'I1_e2', 'S1', 'svensemble2_1', 101, 2),
            ('svensemble2_z', 'L_e1', 'S1', 'svensemble2_2', 5000, 1),
        ])
        assert _pair_inversion_halves(calls, calls['svensemble_id']) == {
            'svensemble2_0': 'svensemble2_0', 'svensemble2_1': 'svensemble2_0',
        }

    @pytest.mark.parametrize('rows', [
        # the halves come from different samples
        [('a', 'I1_e1', 'S1', 'svensemble2_0', 100, 1), ('b', 'I1_e2', 'S2', 'svensemble2_1', 101, 2)],
        # the halves already share one record
        [('a', 'I1_e1', 'S1', 'svensemble2_0', 100, 1), ('b', 'I1_e2', 'S1', 'svensemble2_0', 101, 2)],
        # the halves belong to different events
        [('a', 'I1_e1', 'S1', 'svensemble2_0', 100, 1), ('b', 'I2_e2', 'S1', 'svensemble2_1', 101, 2)],
        # record 0 would pair with both record 1 (in S1) and record 2 (in S2)
        [('a', 'I1_e1', 'S1', 'svensemble2_0', 100, 1), ('b', 'I1_e2', 'S1', 'svensemble2_1', 101, 2),
         ('c', 'I2_e1', 'S2', 'svensemble2_0', 100, 1), ('d', 'I2_e2', 'S2', 'svensemble2_2', 2101, 2)],
    ])
    def test_pairing_skips_unpairable_halves(self, rows):
        # Each case holds e1/e2 halves that must keep their own records.
        calls = self._inv_calls(rows)
        assert _pair_inversion_halves(calls, calls['svensemble_id']) == {}


# ---------------------------------------------------------------------------
# flatten_idgraph
# ---------------------------------------------------------------------------

# One SV across S1 and S2 that truth G1 matches in both samples; a second SV
# in S1 only, matched by truth G2; and truth G3 standing alone.
FLAT_NODES = [
    ('delly', 'D1', 'S1', 'svensemble2_0'),
    ('manta_diploid', 'M1', 'S1', 'svensemble2_0'),
    ('delly', 'D1', 'S2', 'svensemble2_0'),
    ('gnomad', 'G1', GROUND_TRUTH_SAMPLE, 'svensemble2_0'),
    ('gnomad', 'G2', GROUND_TRUTH_SAMPLE, 'svensemble2_1'),
    ('delly', 'D5', 'S1', 'svensemble2_1'),
    ('gnomad', 'G3', GROUND_TRUTH_SAMPLE, None),
]
FLAT_EDGES = [(0, 1), (0, 2), (0, 3), (2, 3), (4, 5)]
FLAT_SOURCES = ['delly', 'manta_diploid', 'gnomad', 'sniffles']


def _flat_rows(table, cols):
    return sorted(tuple(str(v) for v in row) for row in table[cols].itertuples(index=False))


class TestFlattenIdGraph:
    def test_one_row_per_sv_and_sample_with_truth_in_each_sample_it_matches(self):
        # G1 matches D1 in both samples and joins both rows; G2 matches only
        # S1's D5; lone G3 joins no row. sniffles has no node but a column.
        table = flatten_idgraph(_idg(FLAT_NODES, FLAT_EDGES), FLAT_SOURCES, wildcard_sample=GROUND_TRUTH_SAMPLE)
        assert list(table.columns) == [SAMPLE_ID_COL] + [f'{s}_INFO_EVENT' for s in FLAT_SOURCES]
        assert (table.dtypes == 'string').all()
        assert _flat_rows(table, list(table.columns)) == [
            ('S1', 'D1', 'M1', 'G1', '<NA>'),
            ('S1', 'D5', '<NA>', 'G2', '<NA>'),
            ('S2', 'D1', '<NA>', 'G1', '<NA>'),
        ]

    def test_add_common_id_keeps_the_sv_column(self):
        # With add_common_id the SV ID stays as a column named common_id_name.
        table = flatten_idgraph(_idg(FLAT_NODES, FLAT_EDGES), FLAT_SOURCES,
                                wildcard_sample=GROUND_TRUTH_SAMPLE, add_common_id=True)
        assert _flat_rows(table, ['svensemble_id', SAMPLE_ID_COL, 'delly_INFO_EVENT']) == [
            ('svensemble2_0', 'S1', 'D1'), ('svensemble2_0', 'S2', 'D1'), ('svensemble2_1', 'S1', 'D5'),
        ]

    def test_two_nodes_of_one_source_in_one_sv_and_sample_raise(self):
        # An unpruned graph with delly D1 and D2 in one SV and sample cannot
        # pivot to one row; DataFrame.pivot raises.
        idg = _idg([('delly', 'D1', 'S1', 'svensemble2_0'), ('delly', 'D2', 'S1', 'svensemble2_0')], [(0, 1)])
        with pytest.raises(ValueError, match='duplicate'):
            flatten_idgraph(idg, ['delly'])


# ---------------------------------------------------------------------------
# get_svensemble2_calls
# ---------------------------------------------------------------------------

def _matched_and_preds(add_common_id):
    """Matched breakpoints of delly and sniffles in S1 and S2, plus a score per row."""
    matched, _ = inner_join_bps(_bps(
        _row('delly', 'ev1', 100, 500, sample='S1'),
        _row('delly', 'ev1', 100, 500, sample='S2'),
        _row('sniffles', 'evA', 103, 503, sample='S1'),
        _row('delly', 'ev2', 3000, 3500, sample='S2'),
        _row('sniffles', 'evB', 3002, 3502, sample='S2'),
    ), max_dist=10, ignore_size=True, add_common_id=add_common_id)
    preds = matched[['delly_INFO_EVENT', 'sniffles_INFO_EVENT', SAMPLE_ID_COL]].assign(
        **{'ts_m-1_pred': [0.1 * (i + 1) for i in range(len(matched))]})
    return matched, preds


class TestGetSvensemble2Calls:
    def test_keeps_the_sv_id_column(self):
        # Matched breakpoints with svensemble_id give calls that carry it
        # unchanged, row by row (keyed by delly event and sample).
        matched, preds = _matched_and_preds(add_common_id=True)
        calls = get_svensemble2_calls(matched, preds)
        assert 'svensemble_id' in calls.columns and len(calls) == len(matched) == 3
        key = ['delly_INFO_EVENT', SAMPLE_ID_COL]
        want = dict(zip(map(tuple, matched[key].astype(str).to_numpy()), matched['svensemble_id']))
        got = dict(zip(map(tuple, calls[key].astype(str).to_numpy()), calls['svensemble_id']))
        assert got == want
        assert got[('ev1', 'S1')] == got[('ev1', 'S2')] != got[('ev2', 'S2')]

    def test_without_the_sv_id_column_none_is_added(self):
        # Matched breakpoints without svensemble_id give calls without it.
        matched, preds = _matched_and_preds(add_common_id=False)
        assert 'svensemble_id' not in get_svensemble2_calls(matched, preds).columns
