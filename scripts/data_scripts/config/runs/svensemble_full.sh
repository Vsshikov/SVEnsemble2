#!/bin/bash
# =============================================================================
# runs/svensemble_full.sh — Run the FULL SVEnsemble2 Python pipeline + benchmark
#
# Steps executed:
#   1. Raw VCF preprocessing  (preprocess_array.sh → pipeline.py --preprocess_raw)
#   2. Model-specific preprocessing  (--preprocess)
#   3. Breakpoint matching           (--match)
#   4. PU model training             (--train)
#   5. SV probability prediction     (--predict)
#   6. Benchmarking                  (benchmark_array.sh → benchmark.py)
#
# Assumes SV callers have already been run and their outputs exist.
# Alignment is skipped (SKIP_REALIGNMENT=true).
#
# Usage:
#   bash scripts/data_scripts/master.sh data/<SAMPLE>/scripts/config/runs/svensemble_full.sh
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
RUN_SEVERUS=false
RUN_SURVIVOR=false

# --------------- Enable full SVEnsemble2 pipeline ----------------------------
RUN_SVENSEMBLE2=true
SVENSEMBLE2_RERUN=true

SVE_STEP_PREPROCESS=true
SVE_STEP_MATCH=true
SVE_STEP_TRAIN=true
SVE_STEP_PREDICT=true

# --------------- Enable benchmarking -----------------------------------------
RUN_BENCHMARK=true
BENCHMARK=true
