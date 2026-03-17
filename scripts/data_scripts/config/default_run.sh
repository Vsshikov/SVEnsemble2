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
# NOTE: When adding or removing a caller, also update SVE_CALLSETS below.
RUN_BAZAM=
RUN_BAZAM_LONG=
RUN_MINIMAP=
RUN_MINIMAP_LONG=
RUN_MANTA=true
RUN_GRIDSS=true
RUN_DELLY=true
RUN_SNIFFLES=true
RUN_SNIFFLES_207=
RUN_SURVIVOR=true
SURVIVOR_MODES=("short" "long" "all")
RUN_SEVERUS=true
RUN_SVDSS=
RUN_SVABA=true
RUN_CUTESV=true
RUN_SVENSEMBLE2=true
RUN_BENCHMARK=true
RUN_SHAP=

# --------------- Run control ------------------------------------------------
RERUN=false
SVENSEMBLE2_RERUN=true    # Re-run SVEnsemble2 even when RERUN=false
BENCHMARK=true            # Run benchmarking after SVEnsemble2 predictions
SKIP_REALIGNMENT=true

# --------------- BAM / SAM / CRAM input (sample-specific, empty here) ------
SAM_INPUT=
CRAM_INPUT=
CRAM_FASTA=
CRAM_REF_PATH=
CRAM_REF_CACHE=

# --------------- SVEnsemble2 pipeline parameters ----------------------------
SVE_GROUND_TRUTH_SAMPLE_NAME=gnomad_pass
SVE_VERBOSE=true
SVE_MEM=32G

# Default model set — run configs that need additional models override this.
SVE_MODELS=(
    # "unbiased"
    # "unbiased_GT"
    "unbiased_xgb"
    "unbiased_xgb_GT"
    "unbiased_d31"
    "unbiased_GT_d31"
)

# Callsets whose VCFs are parsed into raw breakpoints during the preprocess
# stage. Every callset referenced in SVE_TRAINING_SETS (directly or as part of
# a composite name) must appear here, plus any extra callsets needed for
# benchmarking (e.g. survivor_short).
# NOTE: When adding or removing a caller from the pipeline step flags above,
# update this list accordingly.
SVE_CALLSETS=(
    "gridss"
    "gridss_H"
    "manta_candidate"
    "manta_diploid"
    "delly"
    "sniffles"
    "sniffles_H"
    # "sniffles_207"
    # "sniffles_207_H"
    "severus"
    "severus_H"
    "cutesv"
    "cutesv_H"
    "svaba"
    "svaba_H"
    "survivor_short"
    "survivor_long"
    "survivor_all"
)

# Default training-set preset list (short-read + long-read callers).
# Available presets: default, gridss, manta_candidate, manta_diploid, delly,
#   gridss+manta_candidate, gridss+manta_diploid, gridss+delly,
#   gridss+manta_candidate+delly, gridss+manta_diploid+delly,
#   delly+manta_candidate, delly+manta_diploid
#   sniffles, sniffles+gridss, sniffles+manta_candidate, sniffles+manta_diploid,
#   sniffles+delly, sniffles+manta_candidate+delly, sniffles+manta_diploid+delly,
#   sniffles+gridss+manta_candidate, sniffles+gridss+manta_diploid, etc.
#   sniffles_207, sniffles_207+gridss, sniffles_207+manta_candidate, etc.
#   severus, sniffles+severus, gridss+manta_diploid+delly+sniffles+severus
#   cutesv, cutesv+sniffles, cutesv+severus, cutesv+sniffles+severus,
#   sniffles+severus+cutesv, sniffles+severus+cutesv+gridss+manta_diploid+delly
#   svaba, svaba+gridss, svaba+manta_diploid, svaba+delly,
#   svaba+manta_diploid+delly, gridss+manta_diploid+delly+svaba,
#   sniffles+gridss+manta_diploid+delly+svaba,
#   sniffles+severus+cutesv+gridss+manta_diploid+delly+svaba
SVE_TRAINING_SETS=(
    "gridss"
    "manta_candidate"
    "manta_diploid"
    "delly"
    "sniffles"
    # "gridss+manta_candidate"
    "gridss+manta_diploid"
    "gridss+delly"
    # "gridss+manta_candidate+delly"
    "gridss+manta_diploid+delly"
    "delly+manta_candidate"
    "delly+manta_diploid"
    "sniffles+gridss"
    # "sniffles+manta_candidate"
    "sniffles+manta_diploid"
    "sniffles+delly"
    # "sniffles+manta_candidate+delly"
    "sniffles+manta_diploid+delly"
    # "sniffles+gridss+manta_candidate"
    "sniffles+gridss+manta_diploid"
    # "sniffles+gridss+manta_candidate+delly"
    "sniffles+gridss+manta_diploid+delly"
    # "sniffles_207"
    # "sniffles_207+gridss"
    # "sniffles_207+manta_candidate"
    # "sniffles_207+manta_diploid"
    # "sniffles_207+delly"
    # "sniffles_207+manta_candidate+delly"
    # "sniffles_207+manta_diploid+delly"
    # "sniffles_207+gridss+manta_candidate"
    # "sniffles_207+gridss+manta_diploid"
    # "sniffles_207+gridss+manta_candidate+delly"
    # "sniffles_207+gridss+manta_diploid+delly"
    "cutesv"
    "severus"
    "sniffles+severus"
    "gridss+manta_diploid+delly+sniffles+severus"
    "cutesv+sniffles"
    "cutesv+severus"
    "cutesv+sniffles+severus"
    "sniffles+severus+cutesv"
    "sniffles+severus+cutesv+gridss+manta_diploid+delly"
    "svaba"
    "svaba+gridss"
    "svaba+manta_diploid+delly"
    "gridss+manta_diploid+delly+svaba"
    "sniffles+severus+cutesv+gridss+manta_diploid+delly+svaba"
)

# Pipeline processing steps
SVE_STEP_PREPROCESS=true
SVE_STEP_MATCH=true
SVE_STEP_TRAIN=true
SVE_STEP_PREDICT=true

# Breakpoint matching parameters
SVE_MATCH_MAX_DIST=200
SVE_MATCH_INSERT_DELTA_THRESHOLD=0.7

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
SVE_BENCH_VAL_MATCH_INSERT_DELTA_THRESHOLD=0.7

# Extra callsets and columns included in all benchmark outputs.
# Run configs override these when needed.
SVE_BENCH_ADDITIONAL_CALLSETS=(
    "survivor_short"
    "survivor_long"
    "survivor_all"
    "gridss_H"
    "sniffles_H"
    "severus_H"
    "cutesv_H"
    "svaba"
    "svaba_H"
)
SVE_BENCH_ADDITIONAL_COLS=(
    "FILTER_PASS"
)
SVE_BENCH_LIMIT_TO_CALLSETS=()
SVE_BENCH_SKIP_FILTERING=false

# Custom validation set override (leave empty to use per-sample default from _sample_params.py).
SVE_BENCH_VAL_SET_PATH=
SVE_BENCH_VAL_SET_NAME=

# --------------- SHAP parameters --------------------------------------------
SHAP=false
SVE_SHAP_USE_GPU=false          # Requires gpuq partition and max_depth <= 31
SVE_SHAP_GPU_PARTITION=gpuq
SVE_SHAP_GPU_GRES="gpu:P100:1"
SVE_SHAP_MEM=128G               # Override per run; P100 nodes cap at ~113G
SVE_SHAP_CUDA_PATH="/stornext/System/data/software/rhel/9/base/nvidia/CUDA/11.8"
SVE_SHAP_SOURCE=                # Source pkl; empty = default (bench_pkl)
SVE_SHAP_MAX_SAMPLES=           # Subsample rows; empty = all
SVE_SHAP_PLOT=true
SVE_SHAP_TOP_N=20
