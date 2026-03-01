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

# Read parameters for this task (task file is 0-indexed, sed is 1-indexed)
IFS=$'\t' read -r SVE_MODEL SVE_TRAINING_SET \
    < <(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$TASK_FILE")

echo "$(date): Array task ${SLURM_ARRAY_TASK_ID}: model=${SVE_MODEL} training_set=${SVE_TRAINING_SET}"

# Build step flags from sourced config booleans
SVE_STEP_FLAGS=""
[[ "$SVE_STEP_PREPROCESS" == "true" ]] && SVE_STEP_FLAGS+=" --preprocess"
[[ "$SVE_STEP_MATCH"      == "true" ]] && SVE_STEP_FLAGS+=" --match"
[[ "$SVE_STEP_TRAIN"      == "true" ]] && SVE_STEP_FLAGS+=" --train"
[[ "$SVE_STEP_PREDICT"    == "true" ]] && SVE_STEP_FLAGS+=" --predict"
[[ "$SVE_VERBOSE"         == "true" ]] && SVE_STEP_FLAGS+=" --verbose"

module load micromamba

# shellcheck disable=SC2086
micromamba run -n SVEnsemble2 "$PROJ_DIR/svensemble/pipeline.py" \
    --sample_name "$SAMPLE_NAME" \
    --ground_truth_sample_name "$SVE_GROUND_TRUTH_SAMPLE_NAME" \
    --ref_path "$REF_FASTA" \
    --model_name "$SVE_MODEL" \
    --training_set "$SVE_TRAINING_SET" \
    $SVE_STEP_FLAGS \
    --match_max_dist "$SVE_MATCH_MAX_DIST" \
    --match_insert_size_threshold "$SVE_MATCH_INSERT_SIZE_THRESHOLD"
