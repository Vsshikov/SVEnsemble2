#!/bin/bash

#SBATCH --job-name=svensemble2_benchmark
#SBATCH --time=10:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G

# SLURM array wrapper for SVEnsemble2 benchmark.py.
# Each array task reads its parameters from a TSV task file using
# SLURM_ARRAY_TASK_ID as a 0-based row index.
#
# Usage (called from master.sh):
#   sbatch --array=0-N benchmark_array.sh <source_file> <task_file>
#
# Task file format (one combination per line, tab-separated):
#   SVE_MODEL<TAB>SVE_TRAINING_SET<TAB>SVTYPES<TAB>SVLEN_LOWER
# Optional fields (SVTYPES, SVLEN_LOWER) use "." as a sentinel for "no value"
# to avoid bash collapsing consecutive tab delimiters during IFS=$'\t' read.

SOURCE_FILE="$1"
TASK_FILE="$2"

source "$SOURCE_FILE"

# ─── Per-task logging helper ────────────────────────────────────────────────
LOG_PREFIX="[${SLURM_ARRAY_JOB_ID:-local}.${SLURM_ARRAY_TASK_ID:-0}]"
log()  { echo "$(date -Iseconds) $LOG_PREFIX [$1] ${*:2}"; }
logI() { log INFO  "$@"; }
logE() { log ERROR "$@"; }
die()  { logE "$*"; exit 1; }
# ───────────────────────────────────────────────────────────────────────────

# Read parameters for this task (task file is 0-indexed, sed is 1-indexed).
# Empty optional fields are written as "." in the task file (see master.sh) so
# that IFS=$'\t' read does not collapse consecutive tabs into a single delimiter.
IFS=$'\t' read -r SVE_MODEL SVE_TRAINING_SET _svtypes _svlen_lower \
    < <(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$TASK_FILE")
[[ "$_svtypes"     == "." ]] && _svtypes=""
[[ "$_svlen_lower" == "." ]] && _svlen_lower=""

[[ -z "$SVE_MODEL" || -z "$SVE_TRAINING_SET" ]] && \
    die "Empty task row at index ${SLURM_ARRAY_TASK_ID} in $TASK_FILE"

logI "task=${SLURM_ARRAY_TASK_ID}  model=${SVE_MODEL}  training_set=${SVE_TRAINING_SET}  svtypes='${_svtypes}'  svlen_lower='${_svlen_lower}'"

# Build benchmark flags from sourced config and per-task parameters.
# Array variables are joined with commas for the Python CLI (which splits on ',').
_join_csv() { local IFS=,; echo "$*"; }

_bench_flags=""
[[ -n "$SVE_BENCH_SVLEN_UPPER" ]]                         && _bench_flags+=" --svlen_upper $SVE_BENCH_SVLEN_UPPER"
[[ -n "$SVE_BENCH_VAL_MATCH_MARGIN" ]]                    && _bench_flags+=" --val_match_margin $SVE_BENCH_VAL_MATCH_MARGIN"
[[ -n "$SVE_BENCH_VAL_MATCH_INSERT_SIZE_THRESHOLD" ]]     && _bench_flags+=" --val_match_insert_size_threshold $SVE_BENCH_VAL_MATCH_INSERT_SIZE_THRESHOLD"

_csv=$(_join_csv "${SVE_BENCH_ADDITIONAL_CALLSETS[@]}")
[[ -n "$_csv" ]]                                          && _bench_flags+=" --additional_callsets $_csv"
_csv=$(_join_csv "${SVE_BENCH_ADDITIONAL_COLS[@]}")
[[ -n "$_csv" ]]                                          && _bench_flags+=" --additional_cols $_csv"
_csv=$(_join_csv "${SVE_BENCH_LIMIT_TO_CALLSETS[@]}")
[[ -n "$_csv" ]]                                          && _bench_flags+=" --limit_to_callsets $_csv"

[[ "$SVE_BENCH_SKIP_FILTERING" == "true" ]]               && _bench_flags+=" --skip_filtering"
[[ -n "$SVE_BENCH_VAL_SET_PATH" ]]                        && _bench_flags+=" --val_set_path $SVE_BENCH_VAL_SET_PATH"
[[ -n "$SVE_BENCH_VAL_SET_NAME" ]]                        && _bench_flags+=" --val_set_name $SVE_BENCH_VAL_SET_NAME"
[[ "$SVE_VERBOSE"              == "true" ]]               && _bench_flags+=" --verbose"
[[ -n "$_svtypes" ]]                                      && _bench_flags+=" --svtypes $_svtypes"
[[ -n "$_svlen_lower" ]]                                  && _bench_flags+=" --svlen_lower $_svlen_lower"

module load micromamba

# shellcheck disable=SC2086
micromamba run -n SVEnsemble2 python -m svensemble.benchmark \
    --sample_name "$SAMPLE_NAME" \
    --model_name "$SVE_MODEL" \
    --training_sets "$SVE_TRAINING_SET" \
    $_bench_flags
