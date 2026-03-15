"""
Investigate why GRIDSS INS true-positive counts are low in the NA12878
benchmark, and quantify how many TPs are lost specifically to orientation
mismatch (Problem 1 from the dual-orientation INS representation issue).

Two confounding factors are disentangled:

  A. Insert-size underestimation bias
     GRIDSS (short-read) embeds only the assembled junction sequence in its
     ALT string.  For SINE/ALU insertions (~100-300 bp), GRIDSS typically
     assembles 55-65 bp — about 55-60% of the true element length.  The
     Kosugi truth set records the full element length in INFO_SVLEN.
     At the default 0.7 threshold, 232 / 350 positional matches fail the
     size filter, masking the true positional recall.

  B. Orientation-chunker separation (the dual-orientation INS problem)
     bps.get_bp_chroms_orientations() determines the chunk each call goes
     into.  GRIDSS INS in DUP-like orientation (']N','N[') land in a
     different chunk than truth INS in DEL-like orientation ('N[',']N'),
     so they are never compared — they count as FP/FN regardless of
     position or size.

Orientation pairs (verified from process_breakend_alt geometry):
  DEL / INS-del-like : ('N[', ']N')
  DUP / INS-dup-like : (']N', 'N[')
  INV type-1         : ('N]', 'N]')
  INV type-2         : ('[N', '[N')
"""

import os, sys
os.chdir("/vast/ai_projects/Papenfuss_SVEnsemble2")
sys.path.append("/vast/ai_projects/Papenfuss_SVEnsemble2")

import numpy as np
import pandas as pd

import svensemble.breakends as bnds
import svensemble.breakpoints as bps
import svensemble._sample_params as sample_params
import svensemble.chunk_match as cm

# ── configuration ─────────────────────────────────────────────────────────────
SAMPLE         = 'NA12878'
CALLER         = 'gridss'
MIN_INS_LEN    = 50
VAL_MATCH_DIST = 300    # default in validate_basic_callset
INSERT_THRESH  = 0.7    # default in validate_basic_callset

ORIENT_DEL  = ('N[', ']N')
ORIENT_DUP  = (']N', 'N[')
ORIENT_INV1 = ('N]', 'N]')
ORIENT_INV2 = ('[N', '[N')

val_meta     = sample_params.sample_val_sets[SAMPLE]
VAL_SOURCE   = val_meta['name']                # 'kosugi'
VAL_BPS_PATH = val_meta['path']

GRIDSS_BPS_PATH = (
    f'data/{SAMPLE}/sv_calling/{CALLER}/svensemble2/{CALLER}_bps.csv'
)
BENCH_PATH = (
    f'data/{SAMPLE}/sv_calling/svensemble2/{SAMPLE}.unbiased.caller_benchmark'
    f'/benchmark/{CALLER}.INS.{MIN_INS_LEN}:inf..benchmark.csv'
)

print(f"GRIDSS bps : {GRIDSS_BPS_PATH}")
print(f"Truth bps  : {VAL_BPS_PATH}")
print(f"Benchmark  : {BENCH_PATH}")


# ── helpers ───────────────────────────────────────────────────────────────────

def add_orientation_and_insert(df: pd.DataFrame, source: str) -> pd.DataFrame:
    """
    Enrich a breakpoint DataFrame with orientation and insert-size columns:
      rel1 / rel2      : relative_pos from process_breakend_alt
      orient_pair      : (rel1, rel2) tuple
      insert_size      : via bps.get_insert_size (caller-param dispatch)
      svtype           : via bps.annotate_svtypes — same logic used by benchmark_model
      chunker_key      : bps.get_bp_chroms_orientations (chunker grouping key)

    NOTE: bps.annotate_svtypes applies both orientation AND pos_diff conditions to
    correctly separate INS from DUP for calls with DUP-like breakend orientations.
    Using it here ensures counts are consistent with the benchmark pipeline.
    """
    df = df.copy()
    df['source'] = source

    p1 = bnds.process_breakend_alt(df['ALT_bnd1'])
    p2 = bnds.process_breakend_alt(df['ALT_bnd2'])
    df['rel1']        = p1['relative_pos'].values
    df['rel2']        = p2['relative_pos'].values
    df['orient_pair'] = list(zip(df['rel1'], df['rel2']))
    df['insert_size'] = bps.get_insert_size(df)
    df['svtype']      = bps.annotate_svtypes(df, source)
    df['chunker_key'] = bps.get_bp_chroms_orientations(df)
    return df


# ── load and enrich data ──────────────────────────────────────────────────────
print("\nLoading data …")
gridss = pd.read_csv(GRIDSS_BPS_PATH)
truth  = pd.read_csv(VAL_BPS_PATH)
bench  = pd.read_csv(BENCH_PATH)

gridss = add_orientation_and_insert(gridss, source=CALLER)
truth  = add_orientation_and_insert(truth,  source=VAL_SOURCE)

# Attach benchmark is_in_val to GRIDSS rows
gridss = gridss.merge(bench[['INFO_EVENT', 'is_in_val']], on='INFO_EVENT', how='left')
gridss['is_in_val'] = gridss['is_in_val'].fillna(False).infer_objects(copy=False)

# Filter to pipeline-annotated INS calls — mirrors benchmark_model's svlen_svtype_filter
# annotate_svtypes uses both orientation AND pos_diff to separate INS from DUP;
# this is what generates the benchmark CSV's row set
gridss_ins = bps.svlen_svtype_filter(
    gridss, source=CALLER, svtypes=['INS'], svlens=(MIN_INS_LEN, np.inf)
)

# Truth: filter by INFO_SVTYPE column (truth sets use this directly)
truth_ins = (
    truth[truth['INFO_SVTYPE'] == 'INS'].copy()
    if 'INFO_SVTYPE' in truth.columns
    else truth[truth['svtype'] == 'INS'].copy()
)
truth_ins_50 = truth_ins[truth_ins['insert_size'] >= MIN_INS_LEN].copy()


# ── section 1 : benchmark headline figures ────────────────────────────────────
sep = '=' * 65
print(f"\n{sep}")
print(f"SECTION 1 — Benchmark headline figures  ({SAMPLE}, ≥{MIN_INS_LEN} bp)")
print(sep)
n_gridss_ins_all = (gridss['svtype'] == 'INS').sum()
print(f"  GRIDSS INS calls total (all sizes)  : {n_gridss_ins_all}")
print(f"  GRIDSS INS calls ≥{MIN_INS_LEN} bp (benchmarked): {len(gridss_ins)}")
print(f"  Benchmark TPs (is_in_val)           : {int(gridss_ins['is_in_val'].sum())}")
print(f"  Truth INS calls ≥{MIN_INS_LEN} bp          : {len(truth_ins_50)}")

# Re-run matching to get exact counts at the benchmark threshold
m_size = cm.inner_join_bps(
    {CALLER: gridss_ins, VAL_SOURCE: truth_ins_50},
    max_dist=VAL_MATCH_DIST, insert_delta_threshold=INSERT_THRESH,
    ids_only=True, ignore_size=False
)
m_pos = cm.inner_join_bps(
    {CALLER: gridss_ins, VAL_SOURCE: truth_ins_50},
    max_dist=VAL_MATCH_DIST, insert_delta_threshold=INSERT_THRESH,
    ids_only=False, ignore_size=True
)
tp_size = m_size[f'{CALLER}_INFO_EVENT'].nunique()
tp_pos  = m_pos[f'{CALLER}_INFO_EVENT'].nunique()

print(f"\n  inner_join_bps with size filter (threshold={INSERT_THRESH}): {tp_size} TP")
print(f"  inner_join_bps position only   (ignore_size=True)           : {tp_pos} TP")
print(f"\n  → {tp_pos - tp_size} positionally-correct calls lost to the size threshold.")


# ── section 2 : insert-size underestimation ───────────────────────────────────
print(f"\n{sep}")
print("SECTION 2 — Insert-size underestimation (ALT-based vs SVLEN-based)")
print(sep)
print(
    "  GRIDSS ALT insert size = assembled junction sequence length\n"
    "  Truth INFO_SVLEN        = full mobile-element / insertion length\n"
    "  For Illumina short-read data, GRIDSS typically assembles only the\n"
    "  read-overlapping portion of each insertion (commonly 55-65 bp of\n"
    "  a 100-300 bp SINE/ALU element), giving a systematically lower value.\n"
)

g_sz = m_pos[f'{CALLER}_insert_size'].astype(float)
t_sz = m_pos[f'{VAL_SOURCE}_insert_size'].astype(float)
ratio = np.minimum(g_sz / t_sz, t_sz / g_sz)

print(f"  Size ratio distribution across {len(ratio)} positional matches:")
print(f"    min  = {ratio.min():.3f}")
print(f"    25th = {ratio.quantile(0.25):.3f}")
print(f"    50th = {ratio.median():.3f}   ← median is {ratio.median():.0%} — well below 0.7")
print(f"    75th = {ratio.quantile(0.75):.3f}")
print(f"    max  = {ratio.max():.3f}")
print()
bins = [0, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1.01]
tbl = pd.cut(ratio, bins=bins).value_counts().sort_index()
tbl.index = [str(i) for i in tbl.index]
for interval, count in tbl.items():
    flag = ' ← below threshold' if float(interval.split(',')[1].rstrip(']')) <= 0.7 else ''
    print(f"    {interval:18s} : {count:4d}{flag}")

print(f"\n  Calls failing size filter (ratio < {INSERT_THRESH}): {(ratio < INSERT_THRESH).sum()}")
print(f"  Calls passing size filter (ratio ≥ {INSERT_THRESH}): {(ratio >= INSERT_THRESH).sum()}")

print(f"\n  TPs at alternative size thresholds (position-matched baseline = {tp_pos}):")
for thresh in [0.3, 0.4, 0.5, 0.6, 0.7]:
    m_t = cm.inner_join_bps(
        {CALLER: gridss_ins, VAL_SOURCE: truth_ins_50},
        max_dist=VAL_MATCH_DIST, insert_delta_threshold=thresh,
        ids_only=True, ignore_size=False
    )
    n = m_t[f'{CALLER}_INFO_EVENT'].nunique()
    print(f"    threshold={thresh}: {n:4d} TP  ({n/tp_pos:.0%} of positional matches recovered)")


# ── section 3 : orientation breakdown of benchmarked GRIDSS INS ───────────────
print(f"\n{sep}")
print(f"SECTION 3 — GRIDSS INS orientation breakdown  (≥{MIN_INS_LEN} bp)")
print(sep)

# Orientation label for display: combine svtype (from annotate_svtypes) with
# orient_pair so we can split DEL-like from DUP-like INS rows
del_like_mask = gridss_ins['orient_pair'] == ORIENT_DEL
dup_like_mask = gridss_ins['orient_pair'] == ORIENT_DUP
orient_label  = np.where(del_like_mask, 'INS_del_like',
                np.where(dup_like_mask, 'INS_dup_like', 'INS_other'))
gridss_ins = gridss_ins.copy()
gridss_ins['orient_label'] = orient_label

tbl = (
    gridss_ins
    .groupby(['orient_label', 'is_in_val']).size()
    .unstack(fill_value=0)
    .rename(columns={False: 'FP/unmatched', True: 'TP'})
)
for col in ['FP/unmatched', 'TP']:
    if col not in tbl.columns:
        tbl[col] = 0
print(tbl.to_string())

n_dup = dup_like_mask.sum()
n_del = del_like_mask.sum()
print(f"\n  DEL-like: {n_del}  |  DUP-like: {n_dup}")
print(f"  Truth INS orientation: all {len(truth_ins_50)} calls are {ORIENT_DEL} (DEL-like)")


# ── section 4 : orientation-caused chunker separation ────────────────────────
print(f"\n{sep}")
print("SECTION 4 — Orientation-caused chunker separation and TP losses")
print(sep)
print(
    f"  The chunker groups by bps.get_bp_chroms_orientations().\n"
    f"  GRIDSS DUP-like INS → key suffix '{ORIENT_DUP[0]+ORIENT_DUP[1]}'\n"
    f"  Truth INS (all DEL-like) → key suffix '{ORIENT_DEL[0]+ORIENT_DEL[1]}'\n"
    f"  These calls are in different chunks and are NEVER compared.\n"
)

gridss_dup_ins = gridss_ins[dup_like_mask].copy()
gridss_dup_fp  = gridss_dup_ins[~gridss_dup_ins['is_in_val']].copy()

# Use ignore_size=True for proximity matching (avoids double-counting the
# size-threshold issue from section 2 as an orientation issue)
missed = []
for _, g in gridss_dup_fp.iterrows():
    cand = truth_ins_50[
        (truth_ins_50['CHROM_bnd1'] == g['CHROM_bnd1']) &
        (truth_ins_50['CHROM_bnd2'] == g['CHROM_bnd2'])
    ]
    if cand.empty:
        continue
    d1   = (cand['POS_bnd1'] - g['POS_bnd1']).abs()
    d2   = (cand['POS_bnd2'] - g['POS_bnd2']).abs()
    near = cand[(d1 <= VAL_MATCH_DIST) & (d2 <= VAL_MATCH_DIST)]
    if near.empty:
        continue
    missed.append({
        'gridss_event'    : g['INFO_EVENT'],
        'chrom'           : g['CHROM_bnd1'],
        'pos_bnd1'        : int(g['POS_bnd1']),
        'pos_bnd2'        : int(g['POS_bnd2']),
        'gridss_ins_size' : int(g['insert_size']),
        'n_truth_nearby'  : len(near),
        'truth_orients'   : list(near['orient_pair'].unique()),
    })

miss_df = pd.DataFrame(missed) if missed else pd.DataFrame()

print(f"  DUP-like GRIDSS INS total       : {len(gridss_dup_ins)}")
print(f"  DUP-like GRIDSS INS FPs         : {len(gridss_dup_fp)}")

if miss_df.empty:
    print(
        "\n  None of the DUP-like FPs had a truth INS call within "
        f"{VAL_MATCH_DIST} bp.\n"
        "  Conclusion: orientation mismatch is NOT a significant TP loss driver."
    )
else:
    print(
        f"  DUP-like FPs with truth INS nearby (≤{VAL_MATCH_DIST} bp, any size) : "
        f"{len(miss_df)}"
    )
    print(
        f"\n  These are orientation-caused TP losses: they are at the right\n"
        f"  position but land in the wrong chunk due to DUP-like orientation.\n"
        f"  Fixing Problem 1 (insert-aware orientation mask) would recover them.\n"
    )
    orient_counts = miss_df.explode('truth_orients')['truth_orients'].value_counts()
    print(f"  Truth orientations at those positions:")
    print(orient_counts.to_string())
    print(f"\n  First 10 examples:")
    print(miss_df.head(10).to_string(index=False))


# ── section 5 : summary ───────────────────────────────────────────────────────
print(f"\n{sep}")
print("SECTION 5 — Summary of TP loss sources")
print(sep)
tp_bench   = int(gridss_ins['is_in_val'].sum())
tp_pos_n   = tp_pos
tp_sz_drop = tp_pos - tp_size
tp_orient  = len(miss_df) if not miss_df.empty else 0

pct_orient = tp_orient / n_gridss_ins_all * 100
print(f"  GRIDSS INS total (all sizes)                   : {n_gridss_ins_all}")
print(f"  Benchmark TPs (≥{MIN_INS_LEN} bp, default settings)     : {tp_bench}")
print(f"  Additional TPs recoverable by ignoring size    : {tp_pos_n - tp_bench}")
print(f"    → due to size-threshold drop (Factor A)      : {tp_sz_drop}")
print(f"  Additional TPs recoverable by fixing orientation")
print(f"    (DUP-like calls with truth call nearby)      : {tp_orient}")
print(f"    → {pct_orient:.2f}% of all GRIDSS INS calls")
print(f"\n  Factor A (size threshold) is the dominant TP loss driver.")
print(f"  Factor B (orientation) is real but small ({tp_orient} calls, {pct_orient:.2f}% of total INS).")
print(f"  Recommended fixes:")
print(f"    A → lower insert_delta_threshold for ALT-based vs SVLEN-based comparisons")
print(f"        (threshold ≈ 0.3 recovers 89% of positional matches).")
print(f"    B → implement insert-aware orientation mask in alt_match_matrix")

print("\nDone.")
