#!/bin/bash

#SBATCH --job-name=svensemble2_shap
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=128G

# SLURM array wrapper for SVEnsemble2 sv_shap.py.
# Each array task reads its parameters from a TSV task file using
# SLURM_ARRAY_TASK_ID as a 0-based row index.
#
# Usage (called from master.sh):
#   sbatch --array=0-N shap_array.sh <source_file> <task_file>
#
# Task file format (one combination per line, tab-separated):
#   SVE_MODEL<TAB>SVE_TRAINING_SET<TAB>USE_GPU<TAB>SOURCE<TAB>MAX_SAMPLES
# Optional fields use "." as a sentinel for "no value".

SOURCE_FILE="$1"
TASK_FILE="$2"

source "$SOURCE_FILE"

# Read parameters for this task (task file is 0-indexed, sed is 1-indexed)
IFS=$'\t' read -r SVE_MODEL SVE_TRAINING_SET _use_gpu _source _max_samples \
    < <(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$TASK_FILE")
[[ "$_use_gpu"      == "." ]] && _use_gpu=""
[[ "$_max_samples"  == "." ]] && _max_samples=""

echo "$(date): Array task ${SLURM_ARRAY_TASK_ID}: model=${SVE_MODEL} training_set=${SVE_TRAINING_SET} gt=${SVE_GROUND_TRUTH_SAMPLE_NAME:-default} use_gpu=${_use_gpu} max_samples=${_max_samples}"

# Build SHAP flags from per-task and sourced config values
_shap_flags=""
[[ -n "$SVE_GROUND_TRUTH_SAMPLE_NAME" ]] && _shap_flags+=" --ground_truth_sample_name $SVE_GROUND_TRUTH_SAMPLE_NAME"
[[ "$_use_gpu"         == "1" ]]    && _shap_flags+=" --use_gpu"
[[ -n "$_max_samples" ]]            && _shap_flags+=" --max_samples $_max_samples"
[[ "$SVE_SHAP_PLOT"    == "true" ]] && _shap_flags+=" --plot"
[[ -n "$SVE_SHAP_TOP_N" ]]         && _shap_flags+=" --top_n $SVE_SHAP_TOP_N"
[[ "$SVE_VERBOSE"      == "true" ]] && _shap_flags+=" --verbose"

if [[ "$_use_gpu" == "1" ]]; then
    module load micromamba CUDA/11.8
    export CUDA_PATH="$SVE_SHAP_CUDA_PATH"
else
    module load micromamba
fi

# shellcheck disable=SC2086
micromamba run -n SVEnsemble2 python -m svensemble.sv_shap \
    --sample_name "$SAMPLE_NAME" \
    --model_name "$SVE_MODEL" \
    --training_set "$SVE_TRAINING_SET" \
    $_shap_flags
