"""
Cross-train & concat analysis: do models generalize across samples?

Uses compute_metrics.compute_metrics() for all metric calculations to ensure
consistency with the base pipeline.

For each (test_sample, callset) pair, computes:
  - native XGBoost performance (trained & tested on same sample)
  - per-donor cross-train performance (trained on donor, tested on target)
  - concat_ALL performance (trained on all 4 concatenated, tested on target)

Deltas are per (test_sample, callset); averages are ONLY within the same
callset across test samples — never across different callsets.
"""

import sys, warnings, subprocess
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import precision_recall_curve

WORKDIR = "/vast/ai_projects/Papenfuss_SVEnsemble2"
sys.path.insert(0, WORKDIR)

from scripts.compute_metrics import (
    parse_benchmark_filename,
    compute_metrics as _compute_metrics,
)
import svensemble.benchmark as bench

warnings.filterwarnings('ignore')

ALL_CALLERS = {'gridss','delly','manta_candidate','manta_diploid','svaba',
               'cutesv','sniffles','severus'}
SAMPLES = ['NA12878', 'HG002', 'CHM13', 'HG00514']
PAT = {'NA12878': '*.hgsvc2.benchmark.csv',
       'HG002': '*..benchmark.csv', 'CHM13': '*..benchmark.csv',
       'HG00514': '*..benchmark.csv'}
METRICS = ['auroc', 'auprc', 'opt_f1', 'opt_precision', 'opt_recall']


def callers_of(callset):
    return set(callset.replace('svensemble2_', '').replace('_H', '').split('+'))


def compute_optimal_f1(csv_path, n_truth, callset):
    """Compute optimal-threshold F1 on top of the base compute_metrics output.

    Uses compute_metrics() for AUROC/AUPRC, then sweeps thresholds for best F1.
    """
    df = pd.read_csv(csv_path, low_memory=False)
    base = _compute_metrics(df, thresholds=[0.1, 0.2, 0.3],
                            n_truth=n_truth, callset=callset)

    df = df.dropna(subset=['is_in_val'])
    from scripts.compute_metrics import resolve_score_column
    score_col = resolve_score_column(df, callset=callset)
    y_true = df['is_in_val'].astype(int).values
    y_score = df[score_col].fillna(0.0).values
    n_pos = int(y_true.sum())
    is_sve2 = score_col.endswith('_pred')

    if is_sve2:
        prec_arr, rec_arr, thr_arr = precision_recall_curve(y_true, y_score)
        best_f1, best_thr, best_prec, best_rec, best_tp = -1, 0, 0, 0, 0
        for p, r_c, t in zip(prec_arr[:-1], rec_arr[:-1], thr_arr):
            tp = r_c * n_pos
            rec_g = tp / n_truth if n_truth > 0 else 0
            f1 = 2 * p * rec_g / (p + rec_g) if (p + rec_g) > 0 else 0
            if f1 > best_f1:
                best_f1, best_thr, best_prec, best_rec, best_tp = f1, t, p, rec_g, tp
    else:
        tp = n_pos
        fp = len(y_true) - n_pos
        best_prec = tp / (tp + fp) if (tp + fp) > 0 else np.nan
        best_rec = tp / n_truth if n_truth > 0 else np.nan
        best_f1 = (2 * best_prec * best_rec / (best_prec + best_rec)
                   if (best_prec + best_rec) > 0 else np.nan)
        best_thr = 'all'
        best_tp = tp

    return dict(
        callset=callset, n=base['n'], n_pos=base['n_pos'], n_truth=n_truth,
        auroc=base['auroc'], auprc=base['auprc'],
        opt_f1=best_f1, opt_threshold=best_thr,
        opt_precision=best_prec, opt_recall=best_rec, opt_tp=int(best_tp),
    )


# =========================================================================
# Collect raw results
# =========================================================================
all_rows = []

for sample in SAMPLES:
    pat = PAT[sample]
    configs = [('native', f'*unbiased_xgb.*gt_gnomad_pass*', None)]
    for donor in SAMPLES:
        if donor == sample:
            continue
        configs.append(
            (f'cross_{donor}', f'*xgb_concat_{donor}.*gt_gnomad_pass*', donor))
    configs.append(
        ('concat_ALL', '*xgb_concat_NA12878_HG002_CHM13_HG00514.*gt_gnomad_pass*', 'ALL'))

    for label, path_filter, donor in configs:
        excl = "! -path '*concat*'" if label == 'native' else ""
        cmd = (f"find data/{sample} -path '{path_filter}' "
               f"-name '{pat}' -name '*DEL-INS*' ! -name '*INV*' {excl}")
        result = subprocess.run(cmd, shell=True, capture_output=True,
                                text=True, cwd=WORKDIR)
        files = [Path(l.strip()) for l in result.stdout.strip().split('\n')
                 if l.strip()]
        seen = set()
        for f in files:
            if f.name in seen:
                continue
            seen.add(f.name)
            try:
                parsed = parse_benchmark_filename(f)
                callset = parsed['callset']
            except Exception:
                continue
            if not callers_of(callset).issubset(ALL_CALLERS):
                continue
            svtypes = parsed['svtypes']
            n_truth = bench.get_truth_count(
                sample_name=sample, svtypes=svtypes or None, svlens=(50, np.inf))
            row = compute_optimal_f1(f, n_truth, callset)
            if row is None:
                continue
            row['test_sample'] = sample
            row['training'] = label
            row['donor'] = donor
            row['n_callers'] = len(callers_of(callset))
            all_rows.append(row)

results = pd.DataFrame(all_rows)
print(f"Total rows: {len(results)}")
print(f"Training types: {sorted(results['training'].unique())}")

# =========================================================================
# Compute deltas vs native per (test_sample, callset)
# =========================================================================
native = results[results['training'] == 'native'].set_index(
    ['test_sample', 'callset'])[METRICS]
non_native = results[results['training'] != 'native']

delta_rows = []
for _, r in non_native.iterrows():
    key = (r['test_sample'], r['callset'])
    if key not in native.index:
        continue
    nat = native.loc[key]
    d = dict(test_sample=r['test_sample'], callset=r['callset'],
             training=r['training'], donor=r['donor'],
             n_callers=r['n_callers'])
    for m in METRICS:
        d[f'{m}_native'] = nat[m]
        d[f'{m}_value'] = r[m]
        d[f'{m}_delta'] = r[m] - nat[m]
    delta_rows.append(d)

deltas = pd.DataFrame(delta_rows)
cross = deltas[deltas['training'].str.startswith('cross_')]
concat = deltas[deltas['training'] == 'concat_ALL']

# =========================================================================
# HYPOTHESIS 1: Cross-train generalization
# =========================================================================
print("\n" + "=" * 130)
print("HYPOTHESIS 1: Cross-trained models vs native")
print("  (trained on donor sample, tested on different target sample)")
print("=" * 130)

# Per-callset table (averaged across test_sample x donor pairs)
cross_by_cs = cross.groupby('callset').agg(
    n_pairs=('auroc_delta', 'count'),
    n_callers=('n_callers', 'first'),
    **{f'{m}_delta_mean': (f'{m}_delta', 'mean') for m in METRICS},
    **{f'{m}_delta_std': (f'{m}_delta', 'std') for m in METRICS},
).reset_index().sort_values('auroc_delta_mean', ascending=False)

print("\n--- Per-callset cross-train delta (mean over test_sample x donor pairs) ---")
print(f"  {'callset':<60s} {'n':>4s} {'dAUROC':>9s} {'dAUPRC':>9s} {'dF1':>9s}")
print(f"  {'─'*60} {'─'*4} {'─'*9} {'─'*9} {'─'*9}")
for _, r in cross_by_cs.iterrows():
    print(f"  {r['callset']:<60s} {int(r['n_pairs']):>4d} "
          f"{r['auroc_delta_mean']:>+9.4f} {r['auprc_delta_mean']:>+9.4f} "
          f"{r['opt_f1_delta_mean']:>+9.4f}")

for m_label, m_col in [('AUROC','auroc_delta'), ('AUPRC','auprc_delta'),
                        ('opt_F1','opt_f1_delta')]:
    vals = cross[m_col].dropna()
    pct = (vals > 0).mean() * 100
    print(f"\n  {m_label}: improved in {(vals>0).sum()}/{len(vals)} "
          f"({pct:.1f}%) of pairs  |  mean delta = {vals.mean():+.4f}  "
          f"range [{vals.min():+.4f}, {vals.max():+.4f}]")

print("\n--- By DONOR (training source) ---")
for donor in sorted(cross['donor'].unique()):
    sub = cross[cross['donor'] == donor]
    print(f"  Trained on {donor:8s} (n={len(sub):3d}): "
          f"dAUROC {sub['auroc_delta'].mean():+.4f}  "
          f"dAUPRC {sub['auprc_delta'].mean():+.4f}  "
          f"dF1 {sub['opt_f1_delta'].mean():+.4f}")

print("\n--- By TEST SAMPLE ---")
for s in SAMPLES:
    sub = cross[cross['test_sample'] == s]
    if not len(sub):
        continue
    print(f"  Tested on {s:8s} (n={len(sub):3d}): "
          f"dAUROC {sub['auroc_delta'].mean():+.4f}  "
          f"dAUPRC {sub['auprc_delta'].mean():+.4f}  "
          f"dF1 {sub['opt_f1_delta'].mean():+.4f}")

for m_label, m_col in [('AUROC','auroc_delta'), ('AUPRC','auprc_delta'),
                        ('opt_F1','opt_f1_delta')]:
    base_col = m_col.replace('_delta','')
    print(f"\n--- TOP 5 cross-train IMPROVEMENTS ({m_label}) ---")
    for _, r in cross.nlargest(5, m_col).iterrows():
        print(f"  {r['test_sample']:8s} | {r['callset']:<50s} | "
              f"donor={r['donor']:8s} | "
              f"{r[f'{base_col}_native']:.4f}→{r[f'{base_col}_value']:.4f} "
              f"({r[m_col]:+.4f})")
    print(f"--- TOP 5 cross-train DEGRADATIONS ({m_label}) ---")
    for _, r in cross.nsmallest(5, m_col).iterrows():
        print(f"  {r['test_sample']:8s} | {r['callset']:<50s} | "
              f"donor={r['donor']:8s} | "
              f"{r[f'{base_col}_native']:.4f}→{r[f'{base_col}_value']:.4f} "
              f"({r[m_col]:+.4f})")

# =========================================================================
# HYPOTHESIS 2: Concat_ALL vs native
# =========================================================================
print("\n" + "=" * 130)
print("HYPOTHESIS 2: Concat_ALL (pooled 4-sample training) vs native")
print("=" * 130)

concat_by_cs = concat.groupby('callset').agg(
    n_samples=('auroc_delta', 'count'),
    n_callers=('n_callers', 'first'),
    **{f'{m}_delta_mean': (f'{m}_delta', 'mean') for m in METRICS},
    **{f'{m}_delta_std': (f'{m}_delta', 'std') for m in METRICS},
).reset_index().sort_values('auroc_delta_mean', ascending=False)

print("\n--- Per-callset concat_ALL delta (mean over 4 test samples) ---")
print(f"  {'callset':<60s} {'n':>4s} {'dAUROC':>9s} {'dAUPRC':>9s} {'dF1':>9s}")
print(f"  {'─'*60} {'─'*4} {'─'*9} {'─'*9} {'─'*9}")
for _, r in concat_by_cs.iterrows():
    print(f"  {r['callset']:<60s} {int(r['n_samples']):>4d} "
          f"{r['auroc_delta_mean']:>+9.4f} {r['auprc_delta_mean']:>+9.4f} "
          f"{r['opt_f1_delta_mean']:>+9.4f}")

for m_label, m_col in [('AUROC','auroc_delta'), ('AUPRC','auprc_delta'),
                        ('opt_F1','opt_f1_delta')]:
    vals = concat[m_col].dropna()
    pct = (vals > 0).mean() * 100
    print(f"\n  {m_label}: improved in {(vals>0).sum()}/{len(vals)} "
          f"({pct:.1f}%) of pairs  |  mean delta = {vals.mean():+.4f}  "
          f"range [{vals.min():+.4f}, {vals.max():+.4f}]")

print("\n--- By TEST SAMPLE ---")
for s in SAMPLES:
    sub = concat[concat['test_sample'] == s]
    if not len(sub):
        continue
    print(f"  {s:8s} (n={len(sub):3d}): "
          f"dAUROC {sub['auroc_delta'].mean():+.4f}  "
          f"dAUPRC {sub['auprc_delta'].mean():+.4f}  "
          f"dF1 {sub['opt_f1_delta'].mean():+.4f}")

for m_label, m_col in [('AUROC','auroc_delta'), ('AUPRC','auprc_delta'),
                        ('opt_F1','opt_f1_delta')]:
    base_col = m_col.replace('_delta','')
    print(f"\n--- TOP 5 concat_ALL IMPROVEMENTS ({m_label}) ---")
    for _, r in concat.nlargest(5, m_col).iterrows():
        print(f"  {r['test_sample']:8s} | {r['callset']:<50s} | "
              f"{r[f'{base_col}_native']:.4f}→{r[f'{base_col}_value']:.4f} "
              f"({r[m_col]:+.4f})")
    print(f"--- TOP 5 concat_ALL DEGRADATIONS ({m_label}) ---")
    for _, r in concat.nsmallest(5, m_col).iterrows():
        print(f"  {r['test_sample']:8s} | {r['callset']:<50s} | "
              f"{r[f'{base_col}_native']:.4f}→{r[f'{base_col}_value']:.4f} "
              f"({r[m_col]:+.4f})")

# =========================================================================
# Save
# =========================================================================
out = Path(WORKDIR) / 'data/metrics/cross_train_analysis.xlsx'
out.parent.mkdir(parents=True, exist_ok=True)

with pd.ExcelWriter(out, engine='openpyxl') as writer:
    deltas.to_excel(writer, sheet_name='All deltas', index=False)
    cross_by_cs.to_excel(writer, sheet_name='Cross-train by callset', index=False)
    concat_by_cs.to_excel(writer, sheet_name='Concat by callset', index=False)
    for tag, df_s in [
        ('Cross top AUROC', cross.nlargest(10,'auroc_delta')),
        ('Cross bot AUROC', cross.nsmallest(10,'auroc_delta')),
        ('Cross top F1', cross.nlargest(10,'opt_f1_delta')),
        ('Cross bot F1', cross.nsmallest(10,'opt_f1_delta')),
        ('Concat top AUROC', concat.nlargest(10,'auroc_delta')),
        ('Concat bot AUROC', concat.nsmallest(10,'auroc_delta')),
        ('Concat top F1', concat.nlargest(10,'opt_f1_delta')),
        ('Concat bot F1', concat.nsmallest(10,'opt_f1_delta')),
    ]:
        df_s.to_excel(writer, sheet_name=tag, index=False)
    for ws_name in writer.sheets:
        ws = writer.sheets[ws_name]
        for col in ws.columns:
            mx = max(len(str(c.value or '')) for c in col) + 2
            ws.column_dimensions[col[0].column_letter].width = max(mx, 12)

print(f"\nSaved to {out}")
