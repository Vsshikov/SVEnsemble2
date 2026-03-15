#!/bin/bash
# =============================================================================
# runs/cross_train.sh — Cross-sample transfer learning models
#
# Runs SVEnsemble2 train+predict using concat models (pre-trained on other
# samples and a pooled combination). Preprocessing and matching are skipped
# (feature matrices already exist from a prior svensemble_full run).
#
# Usage:
#   bash scripts/data_scripts/master.sh data/<SAMPLE>/scripts/config/runs/cross_train.sh
# =============================================================================

_RUN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${_RUN_DIR}/../default_run.sh"
source "${_RUN_DIR}/../sample.sh"
unset _RUN_DIR

# --------------- Disable everything except SVEnsemble2 + benchmarking --------
SKIP_REALIGNMENT=true
RUN_BAZAM=false
RUN_BAZAM_LONG=false
RUN_MINIMAP=false
RUN_MINIMAP_LONG=false
RUN_MANTA=false
RUN_GRIDSS=false
RUN_DELLY=false
RUN_SNIFFLES=false
RUN_SNIFFLES_207=false
RUN_SEVERUS=false
RUN_SURVIVOR=false
RUN_SHAP=false

# --------------- Enable SVEnsemble2 (train + predict only) -------------------
RUN_SVENSEMBLE2=true
SVENSEMBLE2_RERUN=true

SVE_MODELS=(
    "unbiased_xgb_concat_HG00514"
    "unbiased_xgb_concat_CHM13"
    "unbiased_xgb_concat_HG002"
    "unbiased_xgb_concat_NA12878"
    "unbiased_xgb_concat_NA12878_HG002_CHM13_HG00514"
)

SVE_STEP_PREPROCESS=true
SVE_STEP_MATCH=true
SVE_STEP_TRAIN=true
SVE_STEP_PREDICT=true

# --------------- Enable benchmarking -----------------------------------------
RUN_BENCHMARK=true
BENCHMARK=true
