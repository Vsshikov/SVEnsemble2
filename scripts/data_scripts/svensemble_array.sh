#!/bin/bash

#SBATCH --job-name=svensemble2
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G

# SLURM array wrapper for SVEnsemble2 pipeline.py.
# Each array task reads its model and training_set from a TSV task file using
# SLURM_ARRAY_TASK_ID as a 0-based row index.
#
# Usage (called from master.sh):
#   sbatch --array=0-N svensemble_array.sh <source_file> <task_file>
#
# Task file format (one combination per line, tab-separated):
#   SVE_MODEL<TAB>SVE_TRAINING_SET

SOURCE_FILE="$1"
TASK_FILE="$2"

PROJ_DIR="/home/users/allstaff/shikov.v/projects/Papenfuss_SVEnsemble2"

source "$SOURCE_FILE"

# ─── Per-task logging helper ────────────────────────────────────────────────
# LOG_PREFIX embeds the SLURM job array ID (e.g. 12345[2]) so every line in
# every task's .out file is traceable back to its exact array index without
# needing to know which file you are reading.
# Falls back gracefully when run outside SLURM (e.g. local testing).
LOG_PREFIX="[${SLURM_ARRAY_JOB_ID:-local}.${SLURM_ARRAY_TASK_ID:-0}]"
log()  { echo "$(date -Iseconds) $LOG_PREFIX [$1] ${*:2}"; }
logI() { log INFO  "$@"; }
logE() { log ERROR "$@"; }
die()  { logE "$*"; exit 1; }
# ───────────────────────────────────────────────────────────────────────────

# Read parameters for this task (task file is 0-indexed, sed is 1-indexed)
IFS=$'\t' read -r SVE_MODEL SVE_TRAINING_SET \
    < <(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$TASK_FILE")

[[ -z "$SVE_MODEL" || -z "$SVE_TRAINING_SET" ]] && \
    die "Empty task row at index ${SLURM_ARRAY_TASK_ID} in $TASK_FILE"

logI "task=${SLURM_ARRAY_TASK_ID}  model=${SVE_MODEL}  training_set=${SVE_TRAINING_SET}"

# Build step flags from sourced config booleans
sve_step_flags=""
[[ "$SVE_STEP_PREPROCESS" == "true" ]] && sve_step_flags+=" --preprocess"
[[ "$SVE_STEP_MATCH"      == "true" ]] && sve_step_flags+=" --match"
[[ "$SVE_STEP_TRAIN"      == "true" ]] && sve_step_flags+=" --train"
[[ "$SVE_STEP_PREDICT"    == "true" ]] && sve_step_flags+=" --predict"
[[ "$SVE_VERBOSE"         == "true" ]] && sve_step_flags+=" --verbose"

module load micromamba

# shellcheck disable=SC2086
micromamba run -n SVEnsemble2 python "$PROJ_DIR/svensemble/pipeline.py" \
    --sample_name "$SAMPLE_NAME" \
    --ground_truth_sample_name "$SVE_GROUND_TRUTH_SAMPLE_NAME" \
    --ref_path "$REF_FASTA" \
    --model_name "$SVE_MODEL" \
    --training_set "$SVE_TRAINING_SET" \
    $sve_step_flags \
    --match_max_dist "$SVE_MATCH_MAX_DIST" \
    --match_insert_size_threshold "$SVE_MATCH_INSERT_SIZE_THRESHOLD"
