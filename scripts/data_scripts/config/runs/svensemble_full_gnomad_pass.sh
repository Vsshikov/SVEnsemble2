#!/bin/bash
# =============================================================================
# runs/svensemble_full_gnomad_pass.sh — Full SVEnsemble2 pipeline with gnomad_pass ground truth
#
# Identical to svensemble_full.sh but uses the gnomad_pass ground truth set
# (PASS-only, no AF cutoff, SR/PE evidence, DEL/INS/DUP/INV, SVLEN>=50).
#
# Usage:
#   bash scripts/data_scripts/master.sh data/<SAMPLE>/scripts/config/runs/svensemble_full_gnomad_pass.sh
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

# --------------- Enable full SVEnsemble2 pipeline ----------------------------
RUN_SVENSEMBLE2=true
SVENSEMBLE2_RERUN=true

SVE_GROUND_TRUTH_SAMPLE_NAME=gnomad_pass

SVE_STEP_PREPROCESS=true
SVE_STEP_MATCH=true
SVE_STEP_TRAIN=true
SVE_STEP_PREDICT=true

# --------------- Enable benchmarking -----------------------------------------
RUN_BENCHMARK=true
BENCHMARK=true
