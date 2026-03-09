#!/bin/bash
# =============================================================================
# defaults.sh — PROJECT-WIDE SHARED DEFAULTS
#
# This file contains settings that are identical across every sample processed
# by this pipeline:
#   • Project root path
#   • Third-party software paths
#   • Reference genome files
#   • SVEnsemble2 algorithm and benchmarking parameter defaults
#   • Default pipeline step flags (all disabled — run configs turn them on)
#
# It is NOT the right place for:
#   • Sample identity (SAMPLE_NAME, SAMPLE_DIR)     → data/<sample>/scripts/config/sample.sh
#   • Input read paths                               → data/<sample>/scripts/config/sample.sh
#   • Which pipeline steps to run for a given run   → data/<sample>/scripts/config/runs/<name>.sh
# =============================================================================

# --------------- Project root -----------------------------------------------
PROJ_DIR="${SVENSEMBLE_BASE_DIR:?Set SVENSEMBLE_BASE_DIR before running}"
export SVENSEMBLE_BASE_DIR

# --------------- Reference genome files -------------------------------------
REF_FASTA="${PROJ_DIR}/data/hg38/downloads/hg38.no_decoys.fa"
VNTR_BED="${PROJ_DIR}/data/hg38/downloads/human_GRCh38_no_alt_analysis_set.trf.bed.gz"

# GRIDSS blacklist (exclude regions for structural variant calling with GRIDSS).
# Sourced from ENCODE project; filters centromeric, telomeric and low-complexity regions.
GRIDSS_EXCLUDE_BED="${PROJ_DIR}/data/hg38/downloads/ENCFF356LFX.bed"

# --------------- Software paths ---------------------------------------------
MINIMAP2_PATH="${PROJ_DIR}/utils/minimap/minimap2-2.30_x64-linux/minimap2"
BAZAM_JAR="${PROJ_DIR}/utils/bazam/bazam.jar"
GRIDSS_PATH="${PROJ_DIR}/callers/gridss"
MANTA_PATH="${PROJ_DIR}/callers/manta-1.6.0.centos6_x86_64"
DELLY_PATH="${PROJ_DIR}/callers/delly"
SURVIVOR_PATH="${PROJ_DIR}/callers/SURVIVOR-master/Debug/SURVIVOR"
SNIFFLES_PATH=

# --------------- Pipeline step flags (all disabled by default) --------------
# Run configs turn individual flags on; leave unset here so run configs can
# assign values without worrying about unset-variable errors.
RUN_BAZAM=
RUN_BAZAM_LONG=
RUN_MINIMAP=
RUN_MINIMAP_LONG=
RUN_MANTA=
RUN_GRIDSS=
RUN_DELLY=
RUN_SNIFFLES=
RUN_SURVIVOR=
RUN_SEVERUS=
RUN_SVENSEMBLE2=
RUN_BENCHMARK=
RUN_SHAP=

# --------------- Run control ------------------------------------------------
RERUN=false
SVENSEMBLE2_RERUN=false    # Re-run SVEnsemble2 even when RERUN=false
BENCHMARK=false            # Run benchmarking after SVEnsemble2 predictions
SKIP_REALIGNMENT=false

# --------------- BAM / SAM / CRAM input (sample-specific, empty here) ------
SAM_INPUT=
CRAM_INPUT=
CRAM_FASTA=
CRAM_REF_PATH=
CRAM_REF_CACHE=

# --------------- SVEnsemble2 pipeline parameters ----------------------------
SVE_GROUND_TRUTH_SAMPLE_NAME=gnomad
SVE_VERBOSE=true
SVE_MEM=32G

# Default model set — run configs that need additional models override this.
SVE_MODELS=(
    "unbiased"
)

# Default training-set preset list (no sniffles).
# Available presets: default, gridss, manta_candidate, manta_diploid, delly,
#   gridss+manta_candidate, gridss+manta_diploid, gridss+delly,
#   gridss+manta_candidate+delly, gridss+manta_diploid+delly,
#   delly+manta_candidate, delly+manta_diploid
#   sniffles, sniffles+gridss, sniffles+manta_candidate, sniffles+manta_diploid,
#   sniffles+delly, sniffles+manta_candidate+delly, sniffles+manta_diploid+delly,
#   sniffles+gridss+manta_candidate, sniffles+gridss+manta_diploid, etc.
SVE_TRAINING_SETS=(
    "gridss"
    "manta_candidate"
    "manta_diploid"
    "delly"
    "gridss+manta_candidate"
    "gridss+manta_diploid"
    "gridss+delly"
    "gridss+manta_candidate+delly"
    "gridss+manta_diploid+delly"
    "delly+manta_candidate"
    "delly+manta_diploid"
)

# Pipeline processing steps
SVE_STEP_PREPROCESS=true
SVE_STEP_MATCH=true
SVE_STEP_TRAIN=true
SVE_STEP_PREDICT=true

# Breakpoint matching parameters
SVE_MATCH_MAX_DIST=200
SVE_MATCH_INSERT_SIZE_THRESHOLD=0.7

# --------------- SVEnsemble2 benchmarking parameters -----------------------
# SV type filter combinations — one benchmark job per entry.
# Each entry is a comma-separated list of SV types (e.g. "DEL" or "DEL,INS,INV").
SVE_BENCH_SVTYPES=(
    "DEL"
    "INS"
    "INV"
    "DEL,INS"
    "DEL,INS,INV"
    "DEL,INS,DUP,INV"
)

# SV length lower-bound values (combined with SVE_BENCH_SVTYPES as a Cartesian product).
SVE_BENCH_SVLEN_LOWER=(
    50
)
SVE_BENCH_SVLEN_UPPER=
SVE_BENCH_VAL_MATCH_MARGIN=200
SVE_BENCH_VAL_MATCH_INSERT_SIZE_THRESHOLD=0.7

# Extra callsets and columns included in all benchmark outputs.
# Run configs override these when needed.
SVE_BENCH_ADDITIONAL_CALLSETS=("survivor")
SVE_BENCH_ADDITIONAL_COLS=FILTER_PASS,FILTER_PASS_bnd1,FILTER_PASS_bnd2
SVE_BENCH_LIMIT_TO_CALLSETS=
SVE_BENCH_SKIP_FILTERING=false

# Custom validation set override (leave empty to use per-sample default from _sample_params.py).
SVE_BENCH_VAL_SET_PATH=
SVE_BENCH_VAL_SET_NAME=

# --------------- SHAP parameters --------------------------------------------
SHAP=false
SVE_SHAP_SOURCE=bench_pkl       # 'bench_pkl' or 'workflow'
SVE_SHAP_USE_GPU=false          # Requires gpuq partition and max_depth <= 31
SVE_SHAP_MAX_SAMPLES=           # Subsample rows; empty = all
SVE_SHAP_PLOT=true
SVE_SHAP_TOP_N=20
