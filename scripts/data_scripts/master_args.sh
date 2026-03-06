#!/bin/bash

# This file is used to pass arguments to the master.sh script
# It is sourced by the master.sh script and the arguments are set in the environment
# The arguments are then passed to the other scripts

# ========== Input file locations ==========

PROJ_DIR=
SAMPLE_NAME=
SAMPLE_DIR=

# REF_FASTA=$PROJ_DIR/data/hg19/downloads/hg19.fa
# REF_FASTA=$PROJ_DIR/data/hg38/downloads/hg38.fa
REF_FASTA=$PROJ_DIR/data/hg38/downloads/hg38.no_decoys.fa
VNTR_BED=$PROJ_DIR/data/hg38/downloads/human_GRCh38_no_alt_analysis_set.trf.bed.gz

# ========== PIPELINE STEPS ==========
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

RERUN=false
SVENSEMBLE2_RERUN=false # If true, will rerun SVEnsemble2 (only SVEnsemble2 and benchmarking) even if RERUN is false
BENCHMARK=true         # If true, will run benchmarking after SVEnsemble2 pipeline jobs complete
SKIP_REALIGNMENT=false

# ========== Software paths ==========
MINIMAP2_PATH="/home/users/allstaff/shikov.v/projects/Papenfuss_SVEnsemble2/utils/minimap/minimap2-2.30_x64-linux/minimap2"
BAZAM_JAR="/home/users/allstaff/shikov.v/projects/Papenfuss_SVEnsemble2/utils/bazam/bazam.jar"
GRIDSS_PATH="/home/users/allstaff/shikov.v/projects/Papenfuss_SVEnsemble2/callers/gridss"
MANTA_PATH="/home/users/allstaff/shikov.v/projects/Papenfuss_SVEnsemble2/callers/manta-1.6.0.centos6_x86_64"
DELLY_PATH="/home/users/allstaff/shikov.v/projects/Papenfuss_SVEnsemble2/callers/delly"
SURVIVOR_PATH="/home/users/allstaff/shikov.v/projects/Papenfuss_SVEnsemble2/callers/SURVIVOR-master/Debug/SURVIVOR"
SNIFFLES_PATH=

# ========== BAM/SAM processing ==========
SAM_INPUT=

# ========== CRAM processing ==========
CRAM_INPUT=
CRAM_FASTA=
CRAM_REF_PATH=
CRAM_REF_CACHE=

# ========== Input Short/Long Reads ==========
READS1_PATH=
READS2_PATH=
READS_LONG_PATHS=(

)
READS_LONG_SOURCE=

# ========== Minimap2 preset ==========
# Leave empty to auto-select based on READS_LONG_SOURCE (pacbio→map-pb, nanopore→map-ont, hifi→map-hifi).
# Presets supported by the pipeline (refer to minimap2 documentation for more details):
#   map-ont     Noisy ONT reads (~10% error rate)                [default fallback]
#   lr:hq       Accurate ONT reads with chemistry v14 (<1% error)
#   map-hifi    PacBio HiFi reads
#   map-pb      PacBio CLR reads (deprecated, prefer map-hifi)
#   map-iclr    Illumina Complete Long Reads (ICLR)
#   sr          Short-read alignment without splicing
MINIMAP_PRESET=

# ========== SVEnsemble2 Pipeline Parameters ==========
SVE_GROUND_TRUTH_SAMPLE_NAME=gnomad
SVE_VERBOSE=true

# Models and training sets are run as a cartesian product — one SLURM job per combination.
# Available models: unbiased, unbiased_GT, unbiased_d31, unbiased_GT_d31, unbiased_cross_train_NA12878
SVE_MODELS=(
    "unbiased"
)

# Available presets: default, gridss, manta_candidate, manta_diploid, delly,
#   gridss+manta_candidate, gridss+manta_diploid, gridss+delly,
#   gridss+manta_candidate+delly, gridss+manta_diploid+delly,
#   delly+manta_candidate, delly+manta_diploid
#
# gridss
# manta_candidate
# manta_diploid
# delly
# gridss+manta_candidate
# gridss+manta_diploid
# gridss+delly
# gridss+manta_candidate+delly
# gridss+manta_diploid+delly
# delly+manta_candidate
# delly+manta_diploid

SVE_TRAINING_SETS=(
    "delly+manta_diploid"
)

# Pipeline steps to execute
SVE_STEP_PREPROCESS=true
SVE_STEP_MATCH=true
SVE_STEP_TRAIN=true
SVE_STEP_PREDICT=true

# Breakpoint matching parameters
SVE_MATCH_MAX_DIST=200
SVE_MATCH_INSERT_SIZE_THRESHOLD=0.7

SVE_MEM=32G

# ========== SVEnsemble2 Benchmarking Parameters ==========
# SV type filter combinations — one benchmark job is submitted per entry.
# Each entry is a comma-separated list of SV types (e.g. "DEL" or "DEL,INS,INV").
# Leave empty to run a single job with no SV type filter (all types).
SVE_BENCH_SVTYPES=(
    # "DEL" "INS" "DUP" "INV"
    # "DEL,INS" "DEL,INS,INV" "DEL,INS,DUP,INV"
)

# SV length lower-bound values — one benchmark job is submitted per entry.
# Combined with SVE_BENCH_SVTYPES as a Cartesian product.
# Leave empty to run a single job with no lower-bound filter.
SVE_BENCH_SVLEN_LOWER=(
    50
)

SVE_BENCH_SVLEN_UPPER=
SVE_BENCH_VAL_MATCH_MARGIN=200
SVE_BENCH_VAL_MATCH_INSERT_SIZE_THRESHOLD=0.7
# Comma-separated extra callsets to include in benchmark (e.g. survivor)
SVE_BENCH_ADDITIONAL_CALLSETS=survivor
# Comma-separated extra columns to retain in benchmark output
SVE_BENCH_ADDITIONAL_COLS=FILTER_PASS,FILTER_PASS_bnd1,FILTER_PASS_bnd2
# Comma-separated callsets to restrict evaluation to (leave empty for all)
SVE_BENCH_LIMIT_TO_CALLSETS=
SVE_BENCH_SKIP_FILTERING=false
# Custom validation set — overrides the default val set for the sample defined in _sample_params.py.
# Both must be set together: VAL_SET_NAME is the key in _caller_params.caller_param_dict used to
# parse the BPS file; VAL_SET_PATH is the absolute path to a preprocessed .bps.csv file.
# Leave empty to use the default val set for the sample.
SVE_BENCH_VAL_SET_PATH=
SVE_BENCH_VAL_SET_NAME=

# ========== SVEnsemble2 SHAP Parameters ==========
# Set SHAP=true to submit SHAP assessment jobs after benchmarking completes.
SHAP=false
# Data source for SHAP: 'bench_pkl' (bench_preprocessed pickle) or 'workflow' (model.pkl + feature_matrix.csv)
SVE_SHAP_SOURCE=bench_pkl
# Set to true to use GPUTreeExplainer (requires gpuq partition and max_depth <= 31)
SVE_SHAP_USE_GPU=false
# Subsample to at most N rows for SHAP computation (leave empty for all rows)
SVE_SHAP_MAX_SAMPLES=
# Generate beeswarm and bar summary plots
SVE_SHAP_PLOT=true
# Number of top features to display in summary and plots
SVE_SHAP_TOP_N=20
