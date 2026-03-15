#!/bin/bash
# =============================================================================
# runs/bench_only.sh — Benchmarking only
#
# Re-runs benchmarking against existing SVEnsemble2 predictions without
# re-running alignment, SV callers, or the SVEnsemble2 pipeline.
#
# All benchmark parameters (SVE_BENCH_*, SVE_MODELS, SVE_TRAINING_SETS, etc.)
# are inherited from default_run.sh and sample.sh.
#
# Usage:
#   bash scripts/data_scripts/master.sh data/<SAMPLE>/scripts/config/runs/bench_only.sh
# =============================================================================

_RUN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${_RUN_DIR}/../default_run.sh"
source "${_RUN_DIR}/../sample.sh"
unset _RUN_DIR

# --------------- Disable everything except benchmarking ----------------------
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
RUN_SVENSEMBLE2=false
RUN_SHAP=false

# --------------- Enable benchmarking only ------------------------------------
RUN_BENCHMARK=true
BENCHMARK=true
