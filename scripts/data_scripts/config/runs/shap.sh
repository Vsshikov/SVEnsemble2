#!/bin/bash
# =============================================================================
# runs/shap.sh — Run ONLY SHAP feature-importance analysis
#
# Steps executed:
#   1. SHAP value computation  (shap_array.sh → sv_shap.py)
#
# Assumes SVEnsemble2 training/prediction AND benchmarking have already been
# completed (the default SVE_SHAP_SOURCE=bench_pkl reads bench_preprocessed.pkl).
# Alignment, SV callers, and SVEnsemble2 pipeline steps are all disabled.
#
# Usage:
#   bash scripts/data_scripts/master.sh data/<SAMPLE>/scripts/config/runs/shap.sh
# =============================================================================

_RUN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${_RUN_DIR}/../default_run.sh"
source "${_RUN_DIR}/../sample.sh"
unset _RUN_DIR

# --------------- Disable alignment and SV callers ----------------------------
SKIP_REALIGNMENT=true
RUN_BAZAM=false
RUN_MINIMAP=false
RUN_MINIMAP_LONG=false
RUN_MANTA=false
RUN_GRIDSS=false
RUN_DELLY=false
RUN_SNIFFLES=false
RUN_SNIFFLES_207=false
RUN_SEVERUS=false
RUN_SURVIVOR=false

# --------------- Disable SVEnsemble2 pipeline --------------------------------
RUN_SVENSEMBLE2=false

# --------------- Disable benchmarking ----------------------------------------
RUN_BENCHMARK=false
BENCHMARK=false

# --------------- Enable SHAP ------------------------------------------------
RUN_SHAP=true
SVE_SHAP_USE_GPU=false          # Requires gpuq partition and max_depth <= 31
SVE_SHAP_MEM=20G               # P100 nodes have ~113G total

# SVE_MODELS=(
#     "unbiased_xgb"
#     "unbiased_xgb_GT"
#     "unbiased_d31"
#     "unbiased_GT_d31"
# )