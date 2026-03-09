#!/bin/bash
# =============================================================================
# runs/TEMPLATE.sh — COPY THIS FILE TO CREATE A NEW RUN CONFIG
#
# Config cascade (Layer 3):
#   1. default_run.sh  — project-wide defaults (paths, software, param defaults)
#   2. sample.sh       — sample identity, input reads, sample-level overrides
#   3. THIS FILE       — run-specific overrides (which steps to run, params)
#
# Usage:
#   cp TEMPLATE.sh my_run.sh
#   # Edit my_run.sh — uncomment and set the options you need
#   bash ../../master.sh data/<SAMPLE>/scripts/config/runs/my_run.sh
#
# Only uncomment variables you want to OVERRIDE. Everything else inherits from
# default_run.sh and sample.sh.
# =============================================================================

_RUN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${_RUN_DIR}/../default_run.sh"
source "${_RUN_DIR}/../sample.sh"
unset _RUN_DIR

# =============================================================================
#                         RUN-SPECIFIC OVERRIDES
# =============================================================================

# --------------- Pipeline step flags -----------------------------------------
# Which steps to run. Set to "true" to enable, leave commented to skip.
# RUN_BAZAM=true                  # Extract reads from BAM/SAM
# RUN_BAZAM_LONG=true             # Extract long reads from BAM
# RUN_MINIMAP=true                # Short-read alignment with minimap2
# RUN_MINIMAP_LONG=true           # Long-read alignment with minimap2
# RUN_MANTA=true                  # Run Manta SV caller
# RUN_GRIDSS=true                 # Run GRIDSS SV caller
# RUN_DELLY=true                  # Run Delly SV caller
# RUN_SNIFFLES=true               # Run Sniffles2 SV caller (long reads)
# RUN_SEVERUS=true                # Run Severus SV caller (long reads)
# RUN_SURVIVOR=true               # Run SURVIVOR consensus caller
# RUN_SVENSEMBLE2=true            # Run SVEnsemble2 pipeline
# RUN_BENCHMARK=true              # Run benchmarking after SVEnsemble2
# RUN_SHAP=true                   # Run SHAP feature importance analysis

# --------------- Run control -------------------------------------------------
# RERUN=true                      # Force re-run of all steps (ignores existing outputs)
# SVENSEMBLE2_RERUN=true          # Re-run SVEnsemble2 only (even if RERUN=false)
# BENCHMARK=true                  # Enable benchmarking
# SKIP_REALIGNMENT=true           # Skip realignment, use existing BAM

# --------------- Input overrides ---------------------------------------------
# SAM_INPUT=/path/to/input.bam    # BAM/SAM input for read extraction
# CRAM_INPUT=/path/to/input.cram  # CRAM input for read extraction
# CRAM_FASTA=/path/to/ref.fa      # Reference FASTA for CRAM decoding
# CRAM_REF_PATH=                  # REF_PATH for CRAM
# CRAM_REF_CACHE=                 # REF_CACHE for CRAM

# --------------- SVEnsemble2 pipeline parameters -----------------------------
# SVE_GROUND_TRUTH_SAMPLE_NAME=gnomad
# SVE_VERBOSE=true
# SVE_MEM=32G                     # Memory for SVEnsemble2 SLURM jobs

# Models to run (array — one SVEnsemble2 job per model x training_set combo).
# Available: unbiased, unbiased_GT, unbiased_d31, unbiased_concat
# SVE_MODELS=(
#     "unbiased"
# )

# Training set presets (array — Cartesian product with SVE_MODELS).
# Available presets (short-read callers):
#   gridss, manta_candidate, manta_diploid, delly,
#   gridss+manta_candidate, gridss+manta_diploid, gridss+delly,
#   gridss+manta_candidate+delly, gridss+manta_diploid+delly,
#   delly+manta_candidate, delly+manta_diploid
# With long-read callers:
#   sniffles, sniffles+gridss, sniffles+manta_candidate, sniffles+manta_diploid,
#   sniffles+delly, sniffles+manta_candidate+delly, sniffles+manta_diploid+delly,
#   sniffles+gridss+manta_candidate, sniffles+gridss+manta_diploid
# SVE_TRAINING_SETS=(
#     "gridss"
#     "manta_diploid"
#     "delly"
# )

# Pipeline processing steps (all true by default)
# SVE_STEP_PREPROCESS=true        # Parse and preprocess caller VCFs
# SVE_STEP_MATCH=true             # Match breakpoints across callers
# SVE_STEP_TRAIN=true             # Train PU model
# SVE_STEP_PREDICT=true           # Predict SV probabilities

# Breakpoint matching parameters
# SVE_MATCH_MAX_DIST=200          # Max distance (bp) for breakpoint matching
# SVE_MATCH_INSERT_SIZE_THRESHOLD=0.7  # Insert size similarity threshold

# --------------- Benchmarking parameters -------------------------------------
# SV type filter combinations — one benchmark job per entry (Cartesian with SVLEN_LOWER).
# SVE_BENCH_SVTYPES=(
#     "DEL"
#     "INS"
#     "INV"
#     "DEL,INS"
#     "DEL,INS,INV"
#     "DEL,INS,DUP,INV"
# )

# SV length bounds
# SVE_BENCH_SVLEN_LOWER=(50)      # Lower bound(s) for SV length filtering
# SVE_BENCH_SVLEN_UPPER=          # Upper bound (empty = no upper limit)

# Validation matching
# SVE_BENCH_VAL_MATCH_MARGIN=200
# SVE_BENCH_VAL_MATCH_INSERT_SIZE_THRESHOLD=0.7

# Additional callsets and columns in benchmark output
# SVE_BENCH_ADDITIONAL_CALLSETS=("survivor")
# SVE_BENCH_ADDITIONAL_COLS=FILTER_PASS,FILTER_PASS_bnd1,FILTER_PASS_bnd2
# SVE_BENCH_LIMIT_TO_CALLSETS=    # Comma-separated; empty = all callsets
# SVE_BENCH_SKIP_FILTERING=false

# Custom validation set (overrides per-sample default from _sample_params.py)
# SVE_BENCH_VAL_SET_PATH=
# SVE_BENCH_VAL_SET_NAME=

# --------------- SHAP parameters ---------------------------------------------
# SVE_SHAP_SOURCE=bench_pkl       # 'bench_pkl' or 'workflow'
# SVE_SHAP_USE_GPU=false          # Requires gpuq partition and max_depth <= 31
# SVE_SHAP_MAX_SAMPLES=           # Subsample rows; empty = all
# SVE_SHAP_PLOT=true
# SVE_SHAP_TOP_N=20
