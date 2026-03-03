"""
Cross-training performance analysis for SVEnsemble2.

Compares the self-trained unbiased model against cross-trained models
(single external sample and all-sample combined) for a given sample.

Metrics computed (DEL+INS+DUP+INV, >=50bp):
  - AUPRC
  - Precision, Recall, F1 and Accuracy at optimal-F1 threshold
  - Precision, Recall, F1 and Accuracy at fixed threshold = 0.5

Sanity check: Self (unbiased) should closely match Cross:<sample>*
when the cross-training sample is the same as the target sample.

Usage:
    python scripts/analyse_cross_training.py --sample HG00514
    python scripts/analyse_cross_training.py --sample HG00514 --base data/HG00514/sv_calling/svensemble2
"""

import argparse
import os
import glob
import pandas as pd
import numpy as np
from sklearn.metrics import (average_precision_score, precision_recall_curve,
                             f1_score, precision_score, accuracy_score)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
def mark_sample(labels: dict, sample: str) -> str:
    labels_ = labels.copy()
    for key, value in labels_.items():
        if sample in value:
            labels_[key] = value.replace(sample, f"{sample}*")
    return labels_

MODEL_ORDER = [
    "unbiased",
    "unbiased_concat_NA12878",
    "unbiased_concat_HG002",
    "unbiased_concat_CHM13",
    "unbiased_concat_HG00514",
    "unbiased_concat_NA12878_HG002_CHM13_HG00514",
]
MODEL_LABELS = {
    "unbiased":                                    "Self (unbiased)",
    "unbiased_concat_NA12878":                     "Cross: NA12878",
    "unbiased_concat_HG002":                       "Cross: HG002",
    "unbiased_concat_CHM13":                       "Cross: CHM13",
    "unbiased_concat_HG00514":                     "Cross: HG00514",
    "unbiased_concat_NA12878_HG002_CHM13_HG00514": "All samples (4x)",
}
# Sample whose self-trained model should be compared in the sanity check.
# Extend this dict as more samples are analysed.
SELF_SAMPLE_FOR_SANITY = {
    "HG00514": "unbiased_concat_HG00514",
    "NA12878": "unbiased_concat_NA12878",
    "HG002":   "unbiased_concat_HG002",
    "CHM13":   "unbiased_concat_CHM13",
}

CROSS_SAMPLES = ["NA12878", "HG002", "CHM13", "HG00514",
                 "NA12878_HG002_CHM13_HG00514"]

SAMPLE_SVTYPES = {
    "NA12878": "DEL-INS-DUP-INV",
    "HG002": "DEL-INS",
    "CHM13": "DEL-INS-INV",
    "HG00514": "DEL-INS-DUP-INV"
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def gather_files(base: str, sample: str) -> pd.DataFrame:
    """Return a DataFrame of (model, caller, path) for all available benchmark files."""
    records = []

    # Self-trained: use DEL-INS-DUP-INV >=50bp file (matches concat 'all' filter)
    for caller_dir in sorted(glob.glob(f"{base}/{sample}.unbiased.*")):
        if "caller_benchmark" in caller_dir or "d31" in caller_dir:
            continue
        caller = caller_dir.split(".unbiased.")[-1]
        f = (f"{caller_dir}/benchmark/svensemble2_{caller}"
             f".{SAMPLE_SVTYPES[sample]}.50:inf.benchmark.csv")
        if os.path.exists(f):
            records.append(dict(model="unbiased", caller=caller, path=f))

    # Cross-trained — try both naming conventions:
    #   "all"            (older HG00514-style benchmark runs)
    #   "DEL-INS-DUP-INV" (current explicit naming used in NA12878 and later runs)
    _concat_candidates = [
        ".all.50:inf..benchmark.csv",
        f".{SAMPLE_SVTYPES[sample]}.50:inf..benchmark.csv",
    ]
    for ext in CROSS_SAMPLES:
        tag = f"unbiased_concat_{ext}"
        for caller_dir in sorted(glob.glob(f"{base}/{sample}.{tag}.*")):
            if "caller_benchmark" in caller_dir:
                continue
            caller = caller_dir.split(f".{tag}.")[-1]
            for suffix in _concat_candidates:
                f = f"{caller_dir}/benchmark/svensemble2_{caller}{suffix}"
                if os.path.exists(f):
                    records.append(dict(model=tag, caller=caller, path=f))
                    break

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def compute_metrics(y_true: np.ndarray, y_score: np.ndarray) -> dict:
    y_true  = np.asarray(y_true,  dtype=int)
    y_score = np.asarray(y_score, dtype=float)

    auprc = average_precision_score(y_true, y_score)

    prec_arr, rec_arr, thr_arr = precision_recall_curve(y_true, y_score)
    with np.errstate(invalid="ignore"):
        f1_arr = np.where(
            (prec_arr[:-1] + rec_arr[:-1]) > 0,
            2 * prec_arr[:-1] * rec_arr[:-1] / (prec_arr[:-1] + rec_arr[:-1]),
            0.0,
        )
    best_idx   = int(np.argmax(f1_arr))
    best_thr   = float(thr_arr[best_idx])
    y_pred_opt = (y_score >= best_thr).astype(int)
    y_pred_05  = (y_score >= 0.5).astype(int)

    return dict(
        n=len(y_true),
        n_pos=int(y_true.sum()),
        pos_rate=float(y_true.mean()),
        auprc=auprc,
        # optimal-F1 threshold
        opt_thr=best_thr,
        opt_prec=precision_score(y_true, y_pred_opt, zero_division=0),
        opt_rec=float(rec_arr[best_idx]),
        opt_f1=float(f1_arr[best_idx]),
        opt_acc=accuracy_score(y_true, y_pred_opt),
        # fixed threshold = 0.5
        prec_05=precision_score(y_true, y_pred_05, zero_division=0),
        rec_05=float(y_true[y_pred_05 == 1].sum() / max(y_true.sum(), 1)),
        f1_05=f1_score(y_true, y_pred_05, zero_division=0),
        acc_05=accuracy_score(y_true, y_pred_05),
    )


def load_and_compute(df_files: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in df_files.iterrows():
        df = pd.read_csv(row["path"], low_memory=False)
        score_col = [c for c in df.columns
                     if c not in ("INFO_EVENT", "is_in_val")][0]
        y_true  = df["is_in_val"].astype(int)
        y_score = df[score_col].fillna(0)
        if y_true.sum() < 5:
            continue
        rows.append({**row, **compute_metrics(y_true, y_score)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def print_sanity_check(results: pd.DataFrame, sample: str) -> None:
    sanity_model = SELF_SAMPLE_FOR_SANITY.get(sample)
    if sanity_model is None or sanity_model not in results["model"].values:
        print(f"\n[SANITY CHECK] No self-cross entry for sample {sample}, skipping.")
        return

    print("\n" + "=" * 65)
    print(f"SANITY CHECK — Self vs Cross:{sample}*")
    print("  Both trained on the same data → values should be nearly identical")
    print("=" * 65)

    self_r  = results[results["model"] == "unbiased"].set_index("caller")
    cross_r = results[results["model"] == sanity_model].set_index("caller")
    shared  = self_r.index.intersection(cross_r.index)

    chk = pd.DataFrame({
        "pos_rate Self":   self_r.loc[shared, "pos_rate"],
        f"pos_rate X*":    cross_r.loc[shared, "pos_rate"],
        "Δ pos_rate":      cross_r.loc[shared, "pos_rate"]  - self_r.loc[shared, "pos_rate"],
        "AUPRC Self":      self_r.loc[shared, "auprc"],
        f"AUPRC X*":       cross_r.loc[shared, "auprc"],
        "Δ AUPRC":         cross_r.loc[shared, "auprc"]     - self_r.loc[shared, "auprc"],
        "F1@opt Self":     self_r.loc[shared, "opt_f1"],
        f"F1@opt X*":      cross_r.loc[shared, "opt_f1"],
        "Δ F1@opt":        cross_r.loc[shared, "opt_f1"]    - self_r.loc[shared, "opt_f1"],
    }).sort_index()
    print(chk.to_string(float_format="%.4f"))

    max_pos  = chk["Δ pos_rate"].abs().max()
    max_aupr = chk["Δ AUPRC"].abs().max()
    verdict  = ("PASS ✓"
                if max_pos < 0.05 and max_aupr < 0.05
                else "WARN — larger-than-expected differences")
    print(f"\n  Max |Δ pos_rate|={max_pos:.4f}  Max |Δ AUPRC|={max_aupr:.4f}  → {verdict}")


def print_summary(results: pd.DataFrame, metric_cols: list, title: str, sample: str) -> None:
    print(f"\n{'='*65}\n{title}\n{'='*65}")
    rows_out = []
    for model in MODEL_ORDER:
        sub = results[results["model"] == model]
        if sub.empty:
            continue
        row = {"Model": mark_sample(MODEL_LABELS, sample).get(model, model), "n_callers": len(sub)}
        for c in metric_cols:
            row[c] = sub[c].mean()
        rows_out.append(row)
    print(pd.DataFrame(rows_out).set_index("Model").to_string(float_format="%.4f"))


def print_per_caller(results: pd.DataFrame, metric: str, sample: str) -> None:
    piv = results.pivot_table(index="caller", columns="model", values=metric)
    piv = piv.reindex(columns=[m for m in MODEL_ORDER if m in piv.columns])
    piv.columns = [mark_sample(MODEL_LABELS, sample).get(c, c) for c in piv.columns]
    print(piv.to_string(float_format="%.4f"))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sample", required=True,
                        help="Sample name, e.g. HG00514")
    parser.add_argument("--base", default=None,
                        help="Path to sv_calling/svensemble2 dir "
                             "(default: data/{sample}/sv_calling/svensemble2)")
    args = parser.parse_args()

    base = args.base or f"data/{args.sample}/sv_calling/svensemble2"

    df_files = gather_files(base, args.sample)
    print(f"Files found: {len(df_files)}  |  "
          f"models: {df_files['model'].nunique()}  |  "
          f"callers: {df_files['caller'].nunique()}")

    results = load_and_compute(df_files)

    print_sanity_check(results, args.sample)

    print_summary(results,
                  ["auprc", "opt_prec", "opt_rec", "opt_f1", "opt_acc", "opt_thr"],
                  "Mean metrics @ optimal-F1 threshold", args.sample)

    print_summary(results,
                  ["auprc", "prec_05", "rec_05", "f1_05", "acc_05"],
                  "Mean metrics @ fixed threshold = 0.5", args.sample)

    print(f"\n{'='*65}")
    print("Per-caller detail (optimal-F1 threshold)")
    print("=" * 65)
    for metric in ["auprc", "opt_prec", "opt_rec", "opt_f1", "opt_acc"]:
        print(f"\n--- {metric} ---")
        print_per_caller(results, metric, args.sample)


if __name__ == "__main__":
    main()
