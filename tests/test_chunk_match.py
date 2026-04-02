"""
Unit tests for svensemble.chunk_match.get_pairwise_matches and its helpers.

Fixture convention
------------------
Breakpoint rows are constructed with precise CI (cipos=0) so that dist_matrix_ref
returns the exact integer distance between positions.  ALT strings use the
standard BND VCF format so that process_breakend_alt produces consistent
relative_pos values and chunker groups all chr1 DEL rows together.

Source names ('delly', 'sniffles') must be keys in _caller_params.caller_param_dict
because get_insert_size dispatches on them.  Both callers use svlen_source='svlen',
so raw insert_size is derived from INFO_SVLEN (non-zero only for INFO_SVTYPE='INS').
"""

import numpy as np
import pandas as pd
import pytest

from svensemble.chunk_match import (
    dist_matrix_ref,
    get_pairwise_matches,
    _get_match_matrix,
    _get_triu_mask,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _row(source, event_id, pos_bnd1, pos_bnd2,
         chrom='chr1', svtype='DEL', svlen=0, cipos=0):
    """
    Build a minimal breakpoint row dict.

    ALT strings follow the N[chr:pos[ / ]chr:pos]N convention so that
    process_breakend_alt returns relative_pos 'N[' and ']N' respectively,
    placing all rows in the same chunker group for the same chromosome.
    """
    return {
        'source': source,
        'INFO_EVENT': event_id,
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
        pairs = get_pairwise_matches(df, max_dist=10, ignore_size=True)
        assert len(pairs) == 1
        assert _pair_sources(pairs[0]) == {'delly', 'sniffles'}
        assert _pair_events(pairs[0]) == {'ev1', 'ev2'}

    def test_two_far_breakpoints_no_match(self):
        """Breakpoints more than max_dist apart should not be matched."""
        df = _df(
            _row('delly',    'ev1', 100, 500),
            _row('sniffles', 'ev2', 200, 600),  # Δbnd1=100 >> max_dist=10
        )
        pairs = get_pairwise_matches(df, max_dist=10, ignore_size=True)
        assert len(pairs) == 0

    def test_same_source_not_matched(self):
        """Breakpoints from the same caller must never be matched to each other."""
        df = _df(
            _row('delly', 'ev1', 100, 500),
            _row('delly', 'ev2', 102, 502),  # same source, close positions
        )
        pairs = get_pairwise_matches(df, max_dist=10, ignore_size=True)
        assert len(pairs) == 0

    def test_three_breakpoints_two_from_one_source(self):
        """Both delly calls should be matched to the single sniffles call."""
        df = _df(
            _row('delly',    'ev1', 100, 500),
            _row('delly',    'ev2', 103, 503),
            _row('sniffles', 'ev3', 104, 504),
        )
        pairs = get_pairwise_matches(df, max_dist=10, ignore_size=True)
        # Each delly event can pair with the sniffles event
        sniffles_events = {ev for s, ev in [p[0] for p in pairs] + [p[1] for p in pairs] if s == 'sniffles'}
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
        pairs = get_pairwise_matches(df, max_dist=10, ignore_size=True)
        pair_source_sets = [_pair_sources(p) for p in pairs]
        assert {'delly', 'sniffles'}       in pair_source_sets
        assert {'delly', 'manta_diploid'}  in pair_source_sets
        assert {'sniffles', 'manta_diploid'} in pair_source_sets

    def test_empty_input_returns_empty(self):
        df = pd.DataFrame(columns=[
            'source', 'INFO_EVENT', 'CHROM_bnd1', 'CHROM_bnd2',
            'POS_bnd1', 'POS_bnd2', 'ALT_bnd1', 'ALT_bnd2',
            'INFO_CIPOS_l_bnd1', 'INFO_CIPOS_r_bnd1',
            'INFO_CIPOS_l_bnd2', 'INFO_CIPOS_r_bnd2',
            'INFO_SVTYPE', 'INFO_SVLEN',
        ])
        pairs = get_pairwise_matches(df, max_dist=10, ignore_size=True)
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
        pairs_ignored = get_pairwise_matches(df, max_dist=10, ignore_size=True)
        assert len(pairs_ignored) == 1, "Should match when size is ignored"

        pairs_filtered = get_pairwise_matches(
            df, max_dist=10, ignore_size=False, insert_delta_threshold=0.7
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
            df, max_dist=10, ignore_size=False, insert_delta_threshold=0.7
        )
        assert len(pairs) == 1

    def test_bnd1_distance_alone_prevents_match(self):
        """bnd1 distance > max_dist should block matching even if bnd2 is close."""
        df = _df(
            _row('delly',    'ev1', 100, 500),
            _row('sniffles', 'ev2', 150, 503),  # Δbnd1=50 > 10, Δbnd2=3 < 10
        )
        pairs = get_pairwise_matches(df, max_dist=10, ignore_size=True)
        assert len(pairs) == 0

    def test_bnd2_distance_alone_prevents_match(self):
        """bnd2 distance > max_dist should block matching even if bnd1 is close."""
        df = _df(
            _row('delly',    'ev1', 100, 500),
            _row('sniffles', 'ev2', 103, 600),  # Δbnd1=3 < 10, Δbnd2=100 > 10
        )
        pairs = get_pairwise_matches(df, max_dist=10, ignore_size=True)
        assert len(pairs) == 0
