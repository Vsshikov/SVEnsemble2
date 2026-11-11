"""
Unit tests for svensemble.chunk_match.get_pairwise_matches and its helpers.

Fixture convention
------------------
Breakpoint rows are constructed with precise CI (cipos=0) so that dist_matrix_ref
returns the exact integer distance between positions.  ALT strings use the
standard BND VCF format so that process_breakend_alt produces consistent
relative_pos values and chunker groups all chr1 DEL rows together.

Source names ('delly', 'sniffles') must be keys in _caller_params.CALLSET_PARAM_DICT
because get_insert_size dispatches on them.  Both callers use svlen_source='svlen',
so raw insert_size is derived from INFO_SVLEN (non-zero only for INFO_SVTYPE='INS').
"""

import numpy as np
import pandas as pd
import pytest

from svensemble._constants import GROUND_TRUTH_SAMPLE, SAMPLE_ID_COL
from svensemble.id_graph import IdGraph, encode_pair, node_table
from svensemble.chunk_match import (
    assign_svensemble_ids,
    dist_matrix_ref,
    get_pairwise_matches,
    inner_join_bps,
    join_calls,
    mark_ground_truth,
    outer_join_bps,
    same_caller_split_edges,
    same_sample_split_edges,
    sample_vertex_mask,
    smoothe_by_source,
    split_graph_by_sample,
    _get_match_matrix,
    _get_triu_mask,
    _split_by_bnd2,
)
import svensemble.chunk_match as chunk_match


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _row(source, event_id, pos_bnd1, pos_bnd2,
         chrom='chr1', svtype='DEL', svlen=0, cipos=0, sample='S1'):
    """
    Build a minimal breakpoint row dict.

    ALT strings follow the N[chr:pos[ / ]chr:pos]N convention so that
    process_breakend_alt returns relative_pos 'N[' and ']N' respectively,
    placing all rows in the same chunker group for the same chromosome.
    """
    return {
        'source': source,
        'INFO_EVENT': event_id,
        SAMPLE_ID_COL: sample,
        'CHROM_bnd1': chrom,
        'CHROM_bnd2': chrom,
        'POS_bnd1': pos_bnd1,
        'POS_bnd2': pos_bnd2,
        # ALT format: mate position embedded but relative_pos is what matters
        'ALT_bnd1': f'N[{chrom}:{pos_bnd2}[',
        'ALT_bnd2': f']chr1:{pos_bnd1}]N',
        # Precise calls: CI spans zero bases on both sides
        'INFO_CIPOS_l_bnd1': -cipos,
        'INFO_CIPOS_r_bnd1': cipos,
        'INFO_CIPOS_l_bnd2': -cipos,
        'INFO_CIPOS_r_bnd2': cipos,
        'INFO_SVTYPE': svtype,
        'INFO_SVLEN': svlen,
    }


def _df(*rows):
    """Return a DataFrame sorted by POS_bnd1 (as concat_bp_dfs would leave it)."""
    return pd.DataFrame(list(rows)).sort_values('POS_bnd1').reset_index(drop=True)


def _pair_sources(pair):
    return frozenset({pair[0][0], pair[1][0]})


def _pair_events(pair):
    return frozenset({pair[0][1], pair[1][1]})


# ---------------------------------------------------------------------------
# dist_matrix_ref
# ---------------------------------------------------------------------------

class TestDistMatrixRef:
    def test_zero_distance_when_identical_position(self):
        df = pd.DataFrame([
            {'POS_bnd1': 100, 'INFO_CIPOS_l_bnd1': 0, 'INFO_CIPOS_r_bnd1': 0},
            {'POS_bnd1': 100, 'INFO_CIPOS_l_bnd1': 0, 'INFO_CIPOS_r_bnd1': 0},
        ])
        d = dist_matrix_ref(df, df, 'bnd1')
        assert d[0, 1] == 0
        assert d[1, 0] == 0

    def test_exact_distance_when_no_overlap(self):
        df = pd.DataFrame([
            {'POS_bnd1': 100, 'INFO_CIPOS_l_bnd1': 0, 'INFO_CIPOS_r_bnd1': 0},
            {'POS_bnd1': 150, 'INFO_CIPOS_l_bnd1': 0, 'INFO_CIPOS_r_bnd1': 0},
        ])
        d = dist_matrix_ref(df, df, 'bnd1')
        assert d[0, 1] == 50
        assert d[1, 0] == 50

    def test_zero_distance_when_ci_intervals_overlap(self):
        """CI of row 0 extends right to 110; CI of row 1 extends left to 105 → overlap."""
        df = pd.DataFrame([
            {'POS_bnd1': 100, 'INFO_CIPOS_l_bnd1': 0, 'INFO_CIPOS_r_bnd1': 10},
            {'POS_bnd1': 108, 'INFO_CIPOS_l_bnd1': -5, 'INFO_CIPOS_r_bnd1': 0},
        ])
        d = dist_matrix_ref(df, df, 'bnd1')
        assert d[0, 1] == 0
        assert d[1, 0] == 0

    def test_diagonal_is_zero(self):
        df = pd.DataFrame([
            {'POS_bnd1': 100, 'INFO_CIPOS_l_bnd1': 0, 'INFO_CIPOS_r_bnd1': 0},
            {'POS_bnd1': 200, 'INFO_CIPOS_l_bnd1': 0, 'INFO_CIPOS_r_bnd1': 0},
        ])
        d = dist_matrix_ref(df, df, 'bnd1')
        assert d[0, 0] == 0
        assert d[1, 1] == 0


# ---------------------------------------------------------------------------
# _get_match_matrix and _get_triu_mask
# ---------------------------------------------------------------------------

class TestGetMatchMatrix:
    def _make_matrices(self, n):
        pos = np.ones((n, n), dtype=bool)
        ins = np.ones((n, n), dtype=bool)
        alt = np.ones((n, n), dtype=bool)
        triu = _get_triu_mask(n)
        cross = np.ones((n, n), dtype=bool)
        np.fill_diagonal(cross, False)
        return pos, ins, alt, triu, cross

    def test_upper_triangle_always_excluded(self):
        n = 3
        pos, ins, alt, triu, cross = self._make_matrices(n)
        m = _get_match_matrix(pos, ins, alt, triu, cross)
        assert not np.any(np.triu(m))

    def test_false_when_pos_not_matched(self):
        n = 2
        pos = np.array([[True, False], [False, True]], dtype=bool)
        ins = np.ones((n, n), dtype=bool)
        alt = np.ones((n, n), dtype=bool)
        triu = _get_triu_mask(n)
        cross = np.ones((n, n), dtype=bool)
        np.fill_diagonal(cross, False)
        m = _get_match_matrix(pos, ins, alt, triu, cross)
        assert not np.any(m)

    def test_false_when_same_source(self):
        """cross_source_mask=False on diagonal means same-source pairs are excluded."""
        n = 2
        pos = np.ones((n, n), dtype=bool)
        ins = np.ones((n, n), dtype=bool)
        alt = np.ones((n, n), dtype=bool)
        triu = np.zeros((n, n), dtype=bool)   # no triangle exclusion for this test
        cross = np.zeros((n, n), dtype=bool)  # all same-source
        m = _get_match_matrix(pos, ins, alt, triu, cross)
        assert not np.any(m)


# ---------------------------------------------------------------------------
# get_pairwise_matches — integration tests
# ---------------------------------------------------------------------------

class TestGetPairwiseMatches:

    def test_two_close_breakpoints_different_sources_match(self):
        """Breakpoints within max_dist from two callers should produce one pair."""
        df = _df(
            _row('delly',    'ev1', 100, 500),
            _row('sniffles', 'ev2', 105, 505),  # Δbnd1=5, Δbnd2=5 — within max_dist=10
        )
        pairs = get_pairwise_matches(df, max_dist=10, ignore_size=True, return_graph=False)
        assert len(pairs) == 1
        assert _pair_sources(pairs[0]) == {'delly', 'sniffles'}
        assert _pair_events(pairs[0]) == {'ev1', 'ev2'}

    def test_two_far_breakpoints_no_match(self):
        """Breakpoints more than max_dist apart should not be matched."""
        df = _df(
            _row('delly',    'ev1', 100, 500),
            _row('sniffles', 'ev2', 200, 600),  # Δbnd1=100 >> max_dist=10
        )
        pairs = get_pairwise_matches(df, max_dist=10, ignore_size=True, return_graph=False)
        assert len(pairs) == 0

    def test_same_source_not_matched(self):
        """Breakpoints from the same caller must never be matched to each other."""
        df = _df(
            _row('delly', 'ev1', 100, 500),
            _row('delly', 'ev2', 102, 502),  # same source, close positions
        )
        pairs = get_pairwise_matches(df, max_dist=10, ignore_size=True, return_graph=False)
        assert len(pairs) == 0

    def test_three_breakpoints_two_from_one_source(self):
        """Both delly calls should be matched to the single sniffles call, and
        to each other only when delly may match itself."""
        df = _df(
            _row('delly',    'ev1', 100, 500),
            _row('delly',    'ev2', 103, 503),
            _row('sniffles', 'ev3', 104, 504),
        )
        with_self = get_pairwise_matches(df, max_dist=10, ignore_size=True, return_graph=False)
        assert {'delly'} in [_pair_sources(pair) for pair in with_self]
        pairs = get_pairwise_matches(df, max_dist=10, ignore_size=True, return_graph=False,
                                     no_self_match=['delly'])
        sniffles_events = {ev for s, ev, _ in [p[0] for p in pairs] + [p[1] for p in pairs] if s == 'sniffles'}
        assert 'ev3' in sniffles_events
        for pair in pairs:
            assert _pair_sources(pair) == {'delly', 'sniffles'}

    def test_three_sources_all_close_produces_cross_source_pairs(self):
        """With three callers all within range each other, all cross-source pairs match."""
        df = _df(
            _row('delly',           'ev1', 100, 500),
            _row('sniffles',        'ev2', 101, 501),
            _row('manta_diploid',   'ev3', 102, 502),
        )
        pairs = get_pairwise_matches(df, max_dist=10, ignore_size=True, return_graph=False)
        pair_source_sets = [_pair_sources(p) for p in pairs]
        assert {'delly', 'sniffles'}       in pair_source_sets
        assert {'delly', 'manta_diploid'}  in pair_source_sets
        assert {'sniffles', 'manta_diploid'} in pair_source_sets

    def test_empty_input_returns_empty(self):
        df = pd.DataFrame(columns=[
            'source', 'INFO_EVENT', SAMPLE_ID_COL, 'CHROM_bnd1', 'CHROM_bnd2',
            'POS_bnd1', 'POS_bnd2', 'ALT_bnd1', 'ALT_bnd2',
            'INFO_CIPOS_l_bnd1', 'INFO_CIPOS_r_bnd1',
            'INFO_CIPOS_l_bnd2', 'INFO_CIPOS_r_bnd2',
            'INFO_SVTYPE', 'INFO_SVLEN',
        ])
        pairs = get_pairwise_matches(df, max_dist=10, ignore_size=True, return_graph=False)
        assert pairs == []

    def test_insert_delta_filter_blocks_dissimilar_ins(self):
        """
        Two INS calls with very different SVLEN should not match when
        insert_delta_threshold=0.7 is applied.
        insert_delta ratio = 101/201 ≈ 0.50 < 0.70 → no match.
        With ignore_size=True the same pair should match.
        """
        df = _df(
            _row('delly',    'ev1', 100, 100, svtype='INS', svlen=100),
            _row('sniffles', 'ev2', 102, 102, svtype='INS', svlen=200),
        )
        pairs_ignored = get_pairwise_matches(df, max_dist=10, ignore_size=True, return_graph=False)
        assert len(pairs_ignored) == 1, "Should match when size is ignored"

        pairs_filtered = get_pairwise_matches(
            df, max_dist=10, ignore_size=False, insert_delta_threshold=0.7, return_graph=False
        )
        assert len(pairs_filtered) == 0, "Should not match when sizes differ by >30%"

    def test_insert_delta_filter_passes_similar_ins(self):
        """
        INS calls with similar SVLEN (ratio ≥ 0.7) should still match.
        insert_delta ratio = 101/131 ≈ 0.77 ≥ 0.70 → match expected.
        """
        df = _df(
            _row('delly',    'ev1', 100, 100, svtype='INS', svlen=100),
            _row('sniffles', 'ev2', 102, 102, svtype='INS', svlen=130),
        )
        pairs = get_pairwise_matches(
            df, max_dist=10, ignore_size=False, insert_delta_threshold=0.7, return_graph=False
        )
        assert len(pairs) == 1

    def test_bnd1_distance_alone_prevents_match(self):
        """bnd1 distance > max_dist should block matching even if bnd2 is close."""
        df = _df(
            _row('delly',    'ev1', 100, 500),
            _row('sniffles', 'ev2', 150, 503),  # Δbnd1=50 > 10, Δbnd2=3 < 10
        )
        pairs = get_pairwise_matches(df, max_dist=10, ignore_size=True, return_graph=False)
        assert len(pairs) == 0

    def test_bnd2_distance_alone_prevents_match(self):
        """bnd2 distance > max_dist should block matching even if bnd1 is close."""
        df = _df(
            _row('delly',    'ev1', 100, 500),
            _row('sniffles', 'ev2', 103, 600),  # Δbnd1=3 < 10, Δbnd2=100 > 10
        )
        pairs = get_pairwise_matches(df, max_dist=10, ignore_size=True, return_graph=False)
        assert len(pairs) == 0

    def test_return_graph_produces_graph(self):
        """return_graph=True should return an IdGraph with correct structure."""
        df = _df(
            _row('delly',    'ev1', 100, 500),
            _row('sniffles', 'ev2', 105, 505),
        )
        idg = get_pairwise_matches(df, max_dist=10, ignore_size=True, return_graph=True)
        assert isinstance(idg, IdGraph)
        g = idg.graph
        assert g.vcount() == 2
        assert g.ecount() == 1
        names = {v['name'] for v in g.vs}
        assert ('delly', 'ev1', 'S1') in names
        assert ('sniffles', 'ev2', 'S1') in names

    def test_return_unmatched_adds_isolated_vertices(self):
        """Unmatched breakpoints should appear as isolated vertices in the graph."""
        df = _df(
            _row('delly',    'ev1', 100, 500),
            _row('sniffles', 'ev2', 105, 505),
            _row('delly',    'ev3', 9000, 9500),
        )
        idg = get_pairwise_matches(
            df, max_dist=10, ignore_size=True,
            return_graph=True, return_unmatched=True,
            skip_single_source_for=None,
        )
        assert isinstance(idg, IdGraph)
        g = idg.graph
        assert g.vcount() == 3
        assert g.ecount() == 1
        names = {v['name'] for v in g.vs}
        assert ('delly', 'ev3', 'S1') in names
        assert all(len(name) == 3 for name in names)


# ---------------------------------------------------------------------------
# _split_by_bnd2: re-cut oversized chunks by POS_bnd2
# ---------------------------------------------------------------------------

class TestSplitByBnd2:

    def test_leaves_small_chunks_alone(self):
        pos2 = np.array([500, 100, 9000, 300])
        order, bounds = _split_by_bnd2(pos2, [(0, 2), (2, 4)], max_n=2, max_dist=10)
        assert order.tolist() == [0, 1, 2, 3]
        assert bounds == [(0, 2), (2, 4)]

    def test_sorts_and_cuts_large_chunk(self):
        """Rows 1-4 form one chunk over max_n; sorted by POS_bnd2 they fall into 3 groups."""
        pos2 = np.array([7, 5000, 100, 5005, 105])
        order, bounds = _split_by_bnd2(pos2, [(0, 1), (1, 5)], max_n=3, max_dist=10)
        assert order.tolist() == [0, 2, 4, 1, 3]
        assert bounds == [(0, 1), (1, 3), (3, 5)]

    def test_cuts_at_missing_position(self):
        pos2 = np.array([100.0, np.nan, 101.0, 102.0])
        order, bounds = _split_by_bnd2(pos2, [(0, 4)], max_n=2, max_dist=10)
        assert order.tolist() == [0, 2, 3, 1]
        assert bounds == [(0, 3), (3, 4)]

    def test_hotspot_matches_same_pairs_as_one_chunk(self, monkeypatch):
        """1200 breakpoints piled within 600 bp at bnd1 and spread at bnd2 match
        the same pairs whether or not the oversized chunk is re-cut by bnd2."""
        rows = []
        for k in range(600):
            rows.append(_row('delly', f'd{k}', 1000 + k, 100_000 + 1000 * k))
            rows.append(_row('sniffles', f's{k}', 1003 + k, 100_005 + 1000 * k))
        df = _df(*rows)

        def edges():
            pairs = get_pairwise_matches(df, max_dist=10, ignore_size=True, return_graph=False)
            return {frozenset(p) for p in pairs}

        split = edges()
        monkeypatch.setattr(chunk_match, 'CHUNK_BND2_MATCH_THRESHOLD', 10**9)
        whole = edges()
        assert len(split) == 600
        assert split == whole


# ---------------------------------------------------------------------------
# Samples: node names, per-sample split, joins
# ---------------------------------------------------------------------------

def _bps(*rows):
    """Per-source breakpoint frames (without the 'source' column) from _row dicts."""
    df = pd.DataFrame(list(rows))
    return {src: g.drop(columns=['source']).reset_index(drop=True) for src, g in df.groupby('source', sort=False)}


class TestSamples:

    def test_missing_sample_column_rejected(self):
        df = _df(_row('delly', 'ev1', 100, 500)).drop(columns=[SAMPLE_ID_COL])
        with pytest.raises(ValueError, match=SAMPLE_ID_COL):
            get_pairwise_matches(df, max_dist=10, ignore_size=True)

    def test_mark_ground_truth(self):
        gt = mark_ground_truth(pd.DataFrame({'INFO_EVENT': ['g1']}))
        assert list(gt[SAMPLE_ID_COL]) == [GROUND_TRUTH_SAMPLE]
        with pytest.raises(ValueError, match='sample-less'):
            mark_ground_truth(gt)

    def test_mask_and_split(self):
        df = _df(
            _row('delly', 'ev1', 100, 500, sample='S1'),
            _row('delly', 'ev1', 100, 500, sample='S2'),
            _row('sniffles', 'evA', 102, 502, sample='S1'),
            _row('sniffles', 'evB', 102, 502, sample='S2'),
            _row('gnomad', 'g1', 101, 501, sample=GROUND_TRUTH_SAMPLE),
            _row('gnomad', 'g2', 104, 504, sample=GROUND_TRUTH_SAMPLE),
        )
        g = get_pairwise_matches(df, max_dist=10, ignore_size=True, return_graph=True).graph
        mask = sample_vertex_mask(g, ['S1'], wildcard_sample=GROUND_TRUTH_SAMPLE)
        assert {v['name'][2] for v, m in zip(g.vs, mask) if m} == {'S1', GROUND_TRUTH_SAMPLE}
        assert not any(m for v, m in zip(g.vs, sample_vertex_mask(g, ['S1']))
                       if v['name'][2] == GROUND_TRUTH_SAMPLE)
        split = split_graph_by_sample(g, GROUND_TRUTH_SAMPLE)
        assert list(split) == ['S1', 'S2']
        for sample, sub in split.items():
            assert {v['name'][2] for v in sub.vs} <= {sample, GROUND_TRUTH_SAMPLE}
            # every edge stays inside one sample (or touches the ground truth)
            assert all({sub.vs[e.source]['name'][2], sub.vs[e.target]['name'][2]} - {GROUND_TRUTH_SAMPLE}
                       <= {sample} for e in sub.es)
        # cross-sample edges (delly S1 - sniffles S2, ...) are gone
        assert sum(sub.ecount() for sub in split.values()) < g.ecount()

    def test_outer_join_one_row_per_sample_with_shared_ground_truth(self):
        bp_dict = _bps(
            _row('delly', 'ev1', 100, 500, sample='S1'),
            _row('delly', 'ev1', 100, 500, sample='S2'),
            _row('sniffles', 'evA', 102, 502, sample='S1'),
            _row('sniffles', 'evB', 102, 502, sample='S2'),
            _row('gnomad', 'g1', 101, 501, sample=GROUND_TRUTH_SAMPLE),
        )
        out, _ = outer_join_bps(bp_dict, ref_callset='gnomad', max_dist=10, ignore_size=True, wildcard_sample=GROUND_TRUTH_SAMPLE)
        rows = sorted(zip(out[SAMPLE_ID_COL], out['delly_INFO_EVENT'], out['sniffles_INFO_EVENT'], out['gnomad_INFO_EVENT']))
        assert rows == [('S1', 'ev1', 'evA', 'g1'), ('S2', 'ev1', 'evB', 'g1')]
        assert 'delly_FORMAT_SAMPLE_ID' not in out.columns

    def test_outer_join_keeps_unmatched_per_sample(self):
        bp_dict = _bps(
            _row('delly', 'ev1', 100, 500, sample='S1'),
            _row('delly', 'ev9', 7000, 7500, sample='S2'),
            _row('sniffles', 'evB', 102, 502, sample='S2'),
            _row('gnomad', 'g1', 101, 501, sample=GROUND_TRUTH_SAMPLE),
        )
        out, _ = outer_join_bps(bp_dict, ref_callset='gnomad', max_dist=10, ignore_size=True, wildcard_sample=GROUND_TRUTH_SAMPLE)
        rows = sorted(
            (smp, str(d), str(sn), str(gt))
            for smp, d, sn, gt in zip(out[SAMPLE_ID_COL], out['delly_INFO_EVENT'], out['sniffles_INFO_EVENT'], out['gnomad_INFO_EVENT'])
        )
        # delly ev1 (S1) and sniffles evB (S2) are close but in different samples: never one row
        assert rows == [('S1', 'ev1', '<NA>', 'g1'), ('S2', '<NA>', 'evB', 'g1'), ('S2', 'ev9', '<NA>', '<NA>')]

    def test_single_sample_inner_join_unchanged_by_split(self):
        bp_dict = _bps(
            _row('delly', 'ev1', 100, 500),
            _row('sniffles', 'evA', 102, 502),
            _row('sniffles', 'evC', 3000, 3500),
            _row('delly', 'ev3', 3001, 3501),
            _row('gnomad', 'g1', 101, 501, sample=GROUND_TRUTH_SAMPLE),
        )
        out, _ = inner_join_bps(bp_dict, ref_callset='gnomad', max_dist=10, ignore_size=True, ids_only=True,
                            wildcard_sample=GROUND_TRUTH_SAMPLE)
        assert list(out[SAMPLE_ID_COL]) == ['S1', 'S1']
        assert sorted(zip(out['delly_INFO_EVENT'], out['sniffles_INFO_EVENT'])) == [('ev1', 'evA'), ('ev3', 'evC')]


# ---------------------------------------------------------------------------
# Wildcard (ground-truth) breakpoints: shared by samples, expensive to cut
# ---------------------------------------------------------------------------

def _truth_rows(bp_dict):
    """Outer-join rows as sorted (sample, delly, sniffles, gnomad) event tuples."""
    out, _ = outer_join_bps(bp_dict, ref_callset='gnomad', max_dist=10, ignore_size=True,
                         ids_only=True, wildcard_sample=GROUND_TRUTH_SAMPLE)
    cols = [SAMPLE_ID_COL, 'delly_INFO_EVENT', 'sniffles_INFO_EVENT', 'gnomad_INFO_EVENT']
    return sorted(tuple(str(v) for v in row) for row in out[cols].itertuples(index=False))


def _sv_ids(bp_dict):
    """SV ID of every event after pruning, as join_calls prunes the graph."""
    g = get_pairwise_matches(
        pd.concat([df.assign(source=src) for src, df in bp_dict.items()]),
        max_dist=10, ignore_size=True, return_graph=True,
        skip_single_source_for=['gnomad'], no_self_match=['gnomad'],
    )
    dropped = same_caller_split_edges(g, wildcard_sample=GROUND_TRUTH_SAMPLE)
    dropped = same_sample_split_edges(g, dropped, wildcard_sample=GROUND_TRUTH_SAMPLE)
    out = assign_svensemble_ids(g, dropped, wildcard_sample=GROUND_TRUTH_SAMPLE).graph
    return {name[1]: sv for name, sv in zip(out.vs['name'], out.vs['svensemble_id'])}


class TestWildcard:

    def test_call_edge_cut_rather_than_truth_edge(self):
        # a (S1) and b (S2) match truth g; c (S1) matches only b. Separating S1's
        # a and c must cut b-c, not g-b: b keeps the truth and c leaves the SV.
        bp_dict = _bps(
            _row('delly', 'a', 995, 1495, sample='S1'),
            _row('gnomad', 'g', 1000, 1500, sample=GROUND_TRUTH_SAMPLE),
            _row('sniffles', 'b', 1008, 1508, sample='S2'),
            _row('delly', 'c', 1016, 1516, sample='S1'),
        )
        assert _truth_rows(bp_dict) == [
            ('S1', 'a', '<NA>', 'g'), ('S1', 'c', '<NA>', '<NA>'), ('S2', '<NA>', 'b', 'g'),
        ]
        sv = _sv_ids(bp_dict)
        assert sv['a'] == sv['b'] == sv['g'] != sv['c']

    def test_every_sample_keeps_the_truth_it_matches(self):
        # x (S1) matches truth g; g matches z1 (S2); y (S1) matches z1 and z2 (S2).
        # Separating S1's x and y must cut y's two call edges, not one truth edge.
        bp_dict = _bps(
            _row('delly', 'x', 984, 5000, sample='S1'),
            _row('gnomad', 'g', 992, 5000, sample=GROUND_TRUTH_SAMPLE),
            _row('sniffles', 'z1', 1000, 5000, sample='S2'),
            _row('delly', 'z2', 1005, 5010, sample='S2'),
            _row('delly', 'y', 1008, 5005, sample='S1'),
        )
        assert _truth_rows(bp_dict) == [
            ('S1', 'x', '<NA>', 'g'), ('S1', 'y', '<NA>', '<NA>'), ('S2', 'z2', 'z1', 'g'),
        ]

    def test_same_caller_cut_keeps_truth_with_direct_match(self):
        # d1 matches truth g; d2 reaches g only through m. Separating the two delly
        # calls cuts m-d2, not g-m.
        bp_dict = _bps(
            _row('delly', 'd1', 1000, 1500),
            _row('gnomad', 'g', 1008, 1508, sample=GROUND_TRUTH_SAMPLE),
            _row('sniffles', 'm', 1016, 1516),
            _row('delly', 'd2', 1024, 1524),
        )
        assert _truth_rows(bp_dict) == [('S1', 'd1', 'm', 'g'), ('S1', 'd2', '<NA>', '<NA>')]

    def test_two_truth_sites_in_one_component_conflict(self):
        # Truth g1 and g2 both match d: one truth site per row, the other stands alone.
        bp_dict = _bps(
            _row('gnomad', 'g1', 992, 1492, sample=GROUND_TRUTH_SAMPLE),
            _row('delly', 'd', 1000, 1500),
            _row('gnomad', 'g2', 1008, 1508, sample=GROUND_TRUTH_SAMPLE),
        )
        sv = _sv_ids(bp_dict)
        assert sv['d'] == sv['g1'] and sv['g2'] is None

    def test_without_wildcard_the_placeholder_is_an_ordinary_sample(self):
        bp_dict = _bps(
            _row('delly', 'a', 1000, 1500, sample='S1'),
            _row('gnomad', 'g', 1004, 1504, sample=GROUND_TRUTH_SAMPLE),
        )
        out, _ = outer_join_bps(bp_dict, ref_callset='gnomad', max_dist=10, ignore_size=True, ids_only=True)
        assert list(out[SAMPLE_ID_COL]) == ['S1']
        assert out['gnomad_INFO_EVENT'].isna().all()


# ---------------------------------------------------------------------------
# Self matches and edge annotations
# ---------------------------------------------------------------------------

class TestPairwiseSelfMatchAndEdges:

    def test_source_matches_itself_within_and_across_samples(self):
        # Three nearby delly calls, two in S1 and one in S2: every pair matches,
        # unless delly is listed in no_self_match.
        df = _df(
            _row('delly', 'ev1', 100, 500, sample='S1'),
            _row('delly', 'ev2', 103, 503, sample='S1'),
            _row('delly', 'ev1', 101, 501, sample='S2'),
        )
        g = get_pairwise_matches(df, max_dist=10, ignore_size=True, skip_single_source_for=None).graph
        edges = {frozenset((g.vs[e.source]['name'], g.vs[e.target]['name'])) for e in g.es}
        assert edges == {
            frozenset({('delly', 'ev1', 'S1'), ('delly', 'ev2', 'S1')}),
            frozenset({('delly', 'ev1', 'S1'), ('delly', 'ev1', 'S2')}),
            frozenset({('delly', 'ev2', 'S1'), ('delly', 'ev1', 'S2')}),
        }
        none = get_pairwise_matches(df, max_dist=10, ignore_size=True, skip_single_source_for=None,
                                    no_self_match=['delly'])
        assert none.graph.ecount() == 0

    def test_edges_carry_source_pair_and_same_sample(self):
        # delly ev1 in S1 and S2 plus sniffles evA in S1, all nearby: each edge
        # holds the code of its unordered source pair and whether both ends
        # come from one sample.
        df = _df(
            _row('delly', 'ev1', 100, 500, sample='S1'),
            _row('delly', 'ev1', 100, 500, sample='S2'),
            _row('sniffles', 'evA', 102, 502, sample='S1'),
        )
        idg = get_pairwise_matches(df, max_dist=10, ignore_size=True)
        assert idg.sources == ('delly', 'sniffles')
        seen = {}
        for e in idg.graph.es:
            a, b = idg.graph.vs[e.source]['name'], idg.graph.vs[e.target]['name']
            seen[frozenset((a, b))] = (e['pair'], e['same_sample'])
            assert e['pair'] == encode_pair(idg, a[0], b[0])
            assert e['same_sample'] == (a[2] == b[2])
        assert len(seen) == 3
        assert seen[frozenset({('delly', 'ev1', 'S1'), ('delly', 'ev1', 'S2')})] == (
            encode_pair(idg, 'delly', 'delly'), False,
        )


# ---------------------------------------------------------------------------
# Hand-checked pruning examples (max_dist=10, ignore_size, precise CIs)
# ---------------------------------------------------------------------------

def _names_to_sv(idg, common_id_name='svensemble_id'):
    """Map each node name (source, event, sample) of a pruned ID graph to its SV ID."""
    return dict(zip(idg.graph.vs['name'], idg.graph.vs[common_id_name]))


class TestHandCheckedPruning:

    def test_single_sample_chain_splits_between_the_two_delly_calls(self):
        # Chain g1-d1-m1-d2-s1, 8 bp apart, so only neighbours match. The two
        # delly calls conflict; the cut keeps d1 (first by event ID among
        # equally connected calls) with its neighbours and removes m1-d2.
        bp_dict = _bps(
            _row('gridss', 'g1', 100, 600),
            _row('delly', 'd1', 108, 608),
            _row('manta_diploid', 'm1', 116, 616),
            _row('delly', 'd2', 124, 624),
            _row('smoove', 's1', 132, 632),
        )
        out, idg = join_calls(bp_dict, max_dist=10, ignore_size=True, ids_only=True, add_common_id=True)
        cols = ['svensemble_id', 'gridss_INFO_EVENT', 'delly_INFO_EVENT',
                'manta_diploid_INFO_EVENT', 'smoove_INFO_EVENT']
        rows = sorted(tuple(str(v) for v in r) for r in out[cols].itertuples(index=False))
        assert rows == [
            ('svensemble2_0.0', 'g1', 'd1', 'm1', '<NA>'),
            ('svensemble2_0.1', '<NA>', 'd2', '<NA>', 's1'),
        ]
        assert idg.graph.ecount() == 3

    def test_cross_sample_chain_cut_once_and_truth_stays_with_its_calls(self):
        # Edges: Y(S2)-g1, g1-A(S1), Y-A, A-B(S2), B-C(S1); g1 is ground truth.
        # Both S1 (A, C) and S2 (Y, B) have two per-sample components in the
        # one SV. Cutting A-B separates both conflicts at cost 1, so it is the
        # only cut: SVs {A, Y, g1} and {B, C}.
        bp_dict = _bps(
            _row('delly', 'A', 100, 600, sample='S1'),
            _row('sniffles', 'B', 108, 608, sample='S2'),
            _row('manta_diploid', 'C', 116, 616, sample='S1'),
            _row('gnomad', 'g1', 96, 596, sample=GROUND_TRUTH_SAMPLE),
            _row('delly', 'Y', 92, 592, sample='S2'),
        )
        g = get_pairwise_matches(
            pd.concat([df.assign(source=src) for src, df in bp_dict.items()]),
            max_dist=10, ignore_size=True, skip_single_source_for=['gnomad'], no_self_match=['gnomad'],
        )
        assert g.graph.ecount() == 5
        dropped = same_caller_split_edges(g, wildcard_sample=GROUND_TRUTH_SAMPLE)
        assert len(dropped) == 0
        dropped = same_sample_split_edges(g, dropped, wildcard_sample=GROUND_TRUTH_SAMPLE)
        cut = [{g.graph.vs[g.graph.es[e].source]['name'][1], g.graph.vs[g.graph.es[e].target]['name'][1]}
               for e in dropped]
        assert cut == [{'A', 'B'}]

        out, idg = outer_join_bps(bp_dict, ref_callset='gnomad', max_dist=10, ignore_size=True, ids_only=True,
                                  wildcard_sample=GROUND_TRUTH_SAMPLE, add_common_id=True)
        sv = _names_to_sv(idg)
        assert sv[('delly', 'A', 'S1')] == sv[('delly', 'Y', 'S2')] == sv[('gnomad', 'g1', GROUND_TRUTH_SAMPLE)]
        assert sv[('sniffles', 'B', 'S2')] == sv[('manta_diploid', 'C', 'S1')] != sv[('delly', 'A', 'S1')]
        assert sorted(sv.values()) == ['svensemble2_0.0'] * 3 + ['svensemble2_0.1'] * 2
        cols = [SAMPLE_ID_COL, 'delly_INFO_EVENT', 'sniffles_INFO_EVENT', 'manta_diploid_INFO_EVENT',
                'gnomad_INFO_EVENT']
        rows = sorted(tuple(str(v) for v in r) for r in out[cols].itertuples(index=False))
        # g1 appears on A's row in S1 and on Y's row in S2
        assert rows == [
            ('S1', '<NA>', '<NA>', 'C', '<NA>'),
            ('S1', 'A', '<NA>', '<NA>', 'g1'),
            ('S2', '<NA>', 'B', '<NA>', '<NA>'),
            ('S2', 'Y', '<NA>', '<NA>', 'g1'),
        ]

    def test_joint_called_event_shares_one_sv_id_across_samples(self):
        # delly ev1 called jointly in S1 and S2 at one locus, matched by
        # sniffles in S1 only: one SV, unsplit, with one row per sample.
        bp_dict = _bps(
            _row('delly', 'ev1', 100, 500, sample='S1'),
            _row('delly', 'ev1', 100, 500, sample='S2'),
            _row('sniffles', 'evA', 103, 503, sample='S1'),
        )
        out, _ = inner_join_bps(bp_dict, max_dist=10, ignore_size=True, ids_only=True, add_common_id=True)
        assert sorted(zip(out[SAMPLE_ID_COL], out['delly_INFO_EVENT'], out['svensemble_id'])) == [
            ('S1', 'ev1', 'svensemble2_0'), ('S2', 'ev1', 'svensemble2_0'),
        ]

    def test_reused_event_ids_at_other_loci_never_share_an_sv(self):
        # S1 and S2 reuse delly IDs D1 and D2 at swapped loci: D1 sits at 100
        # in S1 but at 5000 in S2. Calls join by locus, never by ID.
        bp_dict = _bps(
            _row('delly', 'D1', 100, 600, sample='S1'),
            _row('delly', 'D2', 5000, 5600, sample='S1'),
            _row('delly', 'D1', 5002, 5602, sample='S2'),
            _row('delly', 'D2', 102, 602, sample='S2'),
            _row('sniffles', 'n1', 104, 604, sample='S1'),
        )
        out, idg = outer_join_bps(bp_dict, max_dist=10, ignore_size=True, ids_only=True, add_common_id=True)
        sv = _names_to_sv(idg)
        assert sv[('delly', 'D1', 'S1')] != sv[('delly', 'D1', 'S2')]
        assert sv[('delly', 'D2', 'S1')] != sv[('delly', 'D2', 'S2')]
        assert sv[('delly', 'D1', 'S1')] == sv[('delly', 'D2', 'S2')] == sv[('sniffles', 'n1', 'S1')]
        assert sv[('delly', 'D2', 'S1')] == sv[('delly', 'D1', 'S2')]
        per_sv = out.groupby('svensemble_id')[SAMPLE_ID_COL].apply(sorted).to_dict()
        assert sorted(per_sv.values()) == [['S1', 'S2'], ['S1', 'S2']]


# ---------------------------------------------------------------------------
# assign_svensemble_ids
# ---------------------------------------------------------------------------

class TestAssignSvensembleIds:

    def test_components_numbered_by_smallest_node_and_label_named_by_caller(self):
        # Two separate matched pairs and no drops: each component gets
        # svensemble2_<n> without a '.k' suffix, under the attribute asked for.
        df = _df(
            _row('delly', 'a', 100, 500),
            _row('sniffles', 'b', 103, 503),
            _row('delly', 'c', 9000, 9500),
            _row('sniffles', 'd', 9003, 9503),
        )
        g = get_pairwise_matches(df, max_dist=10, ignore_size=True)
        out = assign_svensemble_ids(g, [], common_id_name='sv')
        assert 'svensemble_id' not in out.graph.vs.attributes()
        nodes = node_table(out)
        first = nodes.loc[nodes.index.min(), 'sv']
        assert sorted(set(nodes['sv'])) == ['svensemble2_0', 'svensemble2_1']
        assert first == 'svensemble2_0'
        by_event = dict(zip(nodes['event'], nodes['sv']))
        assert by_event['a'] == by_event['b'] != by_event['c'] == by_event['d']

    def test_input_graph_unchanged(self):
        # Labelling with one dropped edge leaves the input graph's edges and
        # attributes alone.
        df = _df(_row('delly', 'a', 100, 500), _row('sniffles', 'b', 103, 503))
        g = get_pairwise_matches(df, max_dist=10, ignore_size=True)
        out = assign_svensemble_ids(g, [0])
        assert g.graph.ecount() == 1 and out.graph.ecount() == 0
        assert 'svensemble_id' not in g.graph.vs.attributes()
        assert sorted(out.graph.vs['svensemble_id']) == ['svensemble2_0.0', 'svensemble2_0.1']


# ---------------------------------------------------------------------------
# smoothe_by_source
# ---------------------------------------------------------------------------

class TestSmootheBySource:

    def test_returns_new_graph_and_joins_neighbours(self):
        # delly a (100) and manta m (116) both match sniffles x (108) but not
        # each other. Dropping sniffles joins a and m by a new edge without
        # attributes and leaves the input graph as it was.
        df = _df(
            _row('delly', 'a', 100, 500),
            _row('sniffles', 'x', 108, 508),
            _row('manta_diploid', 'm', 116, 516),
        )
        graph = get_pairwise_matches(df, max_dist=10, ignore_size=True).graph
        names_before, edges_before = list(graph.vs['name']), graph.get_edgelist()
        out = smoothe_by_source(graph, 'sniffles')
        assert out is not graph
        assert list(graph.vs['name']) == names_before and graph.get_edgelist() == edges_before
        assert sorted(out.vs['name']) == [('delly', 'a', 'S1'), ('manta_diploid', 'm', 'S1')]
        assert out.ecount() == 1
        assert out.es[0]['pair'] is None and out.es[0]['same_sample'] is None

    def test_keeps_existing_edge_attributes(self):
        # delly a, sniffles x and manta m all match one another. Dropping
        # sniffles would add a-m again; the existing a-m edge stays, once,
        # with its own pair code and same_sample flag.
        df = _df(
            _row('delly', 'a', 100, 500),
            _row('sniffles', 'x', 103, 503),
            _row('manta_diploid', 'm', 106, 506),
        )
        idg = get_pairwise_matches(df, max_dist=10, ignore_size=True)
        out = smoothe_by_source(idg.graph, 'sniffles')
        assert out.ecount() == 1
        assert out.es[0]['pair'] == encode_pair(idg, 'delly', 'manta_diploid')
        assert out.es[0]['same_sample'] is True
        assert idg.graph.vcount() == 3 and idg.graph.ecount() == 3


# ---------------------------------------------------------------------------
# join_calls
# ---------------------------------------------------------------------------

_TWO_SAMPLE_BPS = (
    _row('delly', 'ev1', 100, 500, sample='S1'),
    _row('delly', 'ev1', 100, 500, sample='S2'),
    _row('sniffles', 'evA', 103, 503, sample='S1'),
    _row('delly', 'ev9', 7000, 7500, sample='S2'),
    _row('gnomad', 'g1', 101, 501, sample=GROUND_TRUTH_SAMPLE),
)


class TestJoinCalls:

    def test_topology_shared_by_every_row_of_one_sv(self):
        # The SV of delly ev1 spans S1 (with sniffles evA) and S2. Topology is
        # computed on the whole SV without the reference, so both rows read
        # three nodes.
        out, _ = join_calls(_bps(*_TWO_SAMPLE_BPS), ref_callset='gnomad', max_dist=10, ignore_size=True,
                            add_topology=True, wildcard_sample=GROUND_TRUTH_SAMPLE, add_common_id=True)
        topo = [c for c in out.columns if c.startswith('svensembletopology_')]
        assert 'svensembletopology_n_nodes' in topo
        shared = out[out['delly_INFO_EVENT'] == 'ev1']
        assert sorted(shared[SAMPLE_ID_COL]) == ['S1', 'S2']
        assert shared['svensemble_id'].nunique() == 1
        assert shared['svensembletopology_n_nodes'].tolist() == [3, 3]
        for col in topo:
            assert shared[col].nunique(dropna=False) == 1, col

    def test_common_id_column_only_when_asked(self):
        # Without add_common_id the SV ID column is dropped; with it, it is
        # named after common_id_name, and the returned graph uses that name.
        bp_dict = _bps(*_TWO_SAMPLE_BPS)
        kw = dict(ref_callset='gnomad', max_dist=10, ignore_size=True, wildcard_sample=GROUND_TRUTH_SAMPLE)
        without, _ = join_calls(bp_dict, **kw)
        assert 'svensemble_id' not in without.columns
        named, idg = join_calls(bp_dict, common_id_name='sv', add_common_id=True, **kw)
        assert 'sv' in named.columns and 'svensemble_id' not in named.columns
        assert isinstance(idg, IdGraph) and 'sv' in idg.graph.vs.attributes()
        assert named['sv'].notna().all()

    def test_ids_only_returns_event_and_sample_columns(self):
        # ids_only skips the caller columns; otherwise each caller's columns
        # arrive with its name as prefix, never the reference's.
        bp_dict = _bps(*_TWO_SAMPLE_BPS)
        kw = dict(ref_callset='gnomad', max_dist=10, ignore_size=True, wildcard_sample=GROUND_TRUTH_SAMPLE)
        ids, _ = join_calls(bp_dict, ids_only=True, **kw)
        assert sorted(ids.columns) == sorted(
            [SAMPLE_ID_COL, 'delly_INFO_EVENT', 'sniffles_INFO_EVENT', 'gnomad_INFO_EVENT'])
        full, _ = join_calls(bp_dict, **kw)
        assert {'delly_POS_bnd1', 'sniffles_POS_bnd1'} <= set(full.columns)
        assert not any(c.startswith('gnomad_') and c != 'gnomad_INFO_EVENT' for c in full.columns)
        assert len(full) == len(ids)

    def test_keep_unmatched_selects_outer_or_inner_join(self):
        # delly ev9 (S2) matches nothing: the outer join keeps it as a row of
        # its own, the inner join drops it.
        bp_dict = _bps(*_TWO_SAMPLE_BPS)
        kw = dict(ref_callset='gnomad', max_dist=10, ignore_size=True, ids_only=True,
                  wildcard_sample=GROUND_TRUTH_SAMPLE)
        outer, _ = join_calls(bp_dict, keep_unmatched=True, **kw)
        inner, _ = join_calls(bp_dict, keep_unmatched=False, **kw)
        assert 'ev9' in set(outer['delly_INFO_EVENT'].dropna())
        assert 'ev9' not in set(inner['delly_INFO_EVENT'].dropna())
        pd.testing.assert_frame_equal(inner_join_bps(bp_dict, **kw)[0], inner)
        pd.testing.assert_frame_equal(outer_join_bps(bp_dict, **kw)[0], outer)


# ---------------------------------------------------------------------------
# _attach_source_columns: one lookup and concat in place of a merge per source
# ---------------------------------------------------------------------------

def _id_rows():
    """ID table as flatten_idgraph leaves it: text columns, one row per per-sample component."""
    return pd.DataFrame({
        'svensemble_id': ['sv0', 'sv1', 'sv2', 'sv3'],
        SAMPLE_ID_COL: ['S1', 'S1', 'S2', 'S2'],
        'delly_INFO_EVENT': ['e1', pd.NA, 'e1', 'e2'],
        'manta_INFO_EVENT': ['m1', 'm2', pd.NA, pd.NA],
    }, dtype='string')


def _source_frames():
    """Two sources whose columns change dtype when a row has no breakpoint (int, bool, text)."""
    return {
        'delly': pd.DataFrame({
            'INFO_EVENT': ['e1', 'e1', 'e2', 'e9'], SAMPLE_ID_COL: ['S1', 'S2', 'S2', 'S1'],
            'POS_bnd1': [10, 11, 20, 90], 'FILTER_PASS': [True, False, True, True], 'ALT_bnd1': ['A', 'B', 'C', 'D'],
        }),
        'manta': pd.DataFrame({
            'INFO_EVENT': ['m1', 'm2'], SAMPLE_ID_COL: ['S1', 'S1'], 'POS_bnd1': [12, 30], 'QUAL': [1.5, 2.5],
        }),
    }


class TestAttachSourceColumns:

    def test_matches_a_left_merge_per_source(self):
        from svensemble.chunk_match import _attach_source_columns, _merge_on_event_and_sample
        expected = _id_rows()
        for source, bp_df in _source_frames().items():
            expected = _merge_on_event_and_sample(expected, bp_df.add_prefix(f'{source}_'), source, how='left')
        pd.testing.assert_frame_equal(_attach_source_columns(_id_rows(), _source_frames()), expected)

    def test_rejects_duplicate_event_and_sample(self):
        from svensemble.chunk_match import _attach_source_columns
        frames = _source_frames()
        frames['manta'] = pd.concat([frames['manta'], frames['manta'].iloc[[0]]])
        with pytest.raises(ValueError, match='manta'):
            _attach_source_columns(_id_rows(), frames)
