#!/bin/bash
# =============================================================================
# runs/preprocess_only.sh — Raw VCF preprocessing only
#
# Runs the model-agnostic raw VCF preprocessing step (preprocess_array.sh →
# pipeline.py --preprocess_raw), producing callset_bps/*.raw_bps.csv files
# for each callset. No model training, matching, prediction, or benchmarking.
#
# Usage:
#   bash scripts/data_scripts/master.sh data/<SAMPLE>/scripts/config/runs/preprocess_only.sh
# =============================================================================

_RUN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${_RUN_DIR}/../default_run.sh"
source "${_RUN_DIR}/../sample.sh"
unset _RUN_DIR

# --------------- Disable everything except SVEnsemble2 preprocessing ---------
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
RUN_BENCHMARK=false
BENCHMARK=false

# --------------- Enable SVEnsemble2 (triggers preprocess_array.sh) -----------
RUN_SVENSEMBLE2=true
SVENSEMBLE2_RERUN=true

# Minimize the SVEnsemble2 model×training_set array to a single no-op task.
# The preprocess job (preprocess_array.sh) runs independently before the array.
SVE_MODELS=("unbiased")
SVE_TRAINING_SETS=("delly")
SVE_STEP_PREPROCESS=false
SVE_STEP_MATCH=false
SVE_STEP_TRAIN=false
SVE_STEP_PREDICT=false
