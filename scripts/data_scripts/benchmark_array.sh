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

PROJ_DIR="/home/users/allstaff/shikov.v/projects/Papenfuss_SVEnsemble2"

source "$SOURCE_FILE"

# Read parameters for this task (task file is 0-indexed, sed is 1-indexed).
# Empty optional fields are written as "." in the task file (see master.sh) so
# that IFS=$'\t' read does not collapse consecutive tabs into a single delimiter.
IFS=$'\t' read -r SVE_MODEL SVE_TRAINING_SET _svtypes _svlen_lower \
    < <(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$TASK_FILE")
[[ "$_svtypes"     == "." ]] && _svtypes=""
[[ "$_svlen_lower" == "." ]] && _svlen_lower=""

echo "$(date): Array task ${SLURM_ARRAY_TASK_ID}: model=${SVE_MODEL} training_set=${SVE_TRAINING_SET} svtypes='${_svtypes}' svlen_lower='${_svlen_lower}'"

# Build benchmark flags from sourced config and per-task parameters
_bench_flags=""
[[ -n "$SVE_BENCH_SVLEN_UPPER" ]]                         && _bench_flags+=" --svlen_upper $SVE_BENCH_SVLEN_UPPER"
[[ -n "$SVE_BENCH_VAL_MATCH_MARGIN" ]]                    && _bench_flags+=" --val_match_margin $SVE_BENCH_VAL_MATCH_MARGIN"
[[ -n "$SVE_BENCH_VAL_MATCH_INSERT_SIZE_THRESHOLD" ]]     && _bench_flags+=" --val_match_insert_size_threshold $SVE_BENCH_VAL_MATCH_INSERT_SIZE_THRESHOLD"
[[ -n "$SVE_BENCH_ADDITIONAL_CALLSETS" ]]                 && _bench_flags+=" --additional_callsets $SVE_BENCH_ADDITIONAL_CALLSETS"
[[ -n "$SVE_BENCH_ADDITIONAL_COLS" ]]                     && _bench_flags+=" --additional_cols $SVE_BENCH_ADDITIONAL_COLS"
[[ -n "$SVE_BENCH_LIMIT_TO_CALLSETS" ]]                   && _bench_flags+=" --limit_to_callsets $SVE_BENCH_LIMIT_TO_CALLSETS"
[[ "$SVE_BENCH_SKIP_FILTERING" == "true" ]]               && _bench_flags+=" --skip_filtering"
[[ "$SVE_VERBOSE"              == "true" ]]               && _bench_flags+=" --verbose"
[[ -n "$_svtypes" ]]                                      && _bench_flags+=" --svtypes $_svtypes"
[[ -n "$_svlen_lower" ]]                                  && _bench_flags+=" --svlen_lower $_svlen_lower"

module load micromamba

# shellcheck disable=SC2086
micromamba run -n SVEnsemble2 python "$PROJ_DIR/svensemble/benchmark.py" \
    --sample_name "$SAMPLE_NAME" \
    --model_name "$SVE_MODEL" \
    --training_sets "$SVE_TRAINING_SET" \
    $_bench_flags
