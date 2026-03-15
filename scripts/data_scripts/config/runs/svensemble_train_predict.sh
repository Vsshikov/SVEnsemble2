#!/bin/bash
# =============================================================================
# runs/svensemble_train_predict.sh — Run ONLY training + prediction + benchmark
#
# Steps executed:
#   1. PU model training             (--train)
#   2. SV probability prediction     (--predict)
#   3. Benchmarking                  (benchmark_array.sh → benchmark.py)
#
# Skips raw VCF preprocessing, model-specific preprocessing, and breakpoint
# matching — these must have been completed in a prior run.
#
# Assumes SV callers have already been run and their outputs exist.
# Alignment is skipped (SKIP_REALIGNMENT=true).
#
# Usage:
#   bash scripts/data_scripts/master.sh data/<SAMPLE>/scripts/config/runs/svensemble_train_predict.sh
# =============================================================================

_RUN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${_RUN_DIR}/../default_run.sh"
source "${_RUN_DIR}/../sample.sh"
unset _RUN_DIR

# --------------- Disable alignment and SV callers ----------------------------
SKIP_REALIGNMENT=true
RUN_BAZAM=false
RUN_MINIMAP=
RUN_MINIMAP_LONG=
RUN_MANTA=false
RUN_GRIDSS=false
RUN_DELLY=false
RUN_SNIFFLES=false
RUN_SNIFFLES_207=false
RUN_SEVERUS=false
RUN_SURVIVOR=false

# --------------- Enable SVEnsemble2 (train + predict only) -------------------
RUN_SVENSEMBLE2=true
SVENSEMBLE2_RERUN=true

SVE_STEP_PREPROCESS=false
SVE_STEP_MATCH=false
SVE_STEP_TRAIN=true
SVE_STEP_PREDICT=true

# --------------- Enable benchmarking -----------------------------------------
RUN_BENCHMARK=true
BENCHMARK=true
