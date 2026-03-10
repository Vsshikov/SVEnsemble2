#!/bin/bash

#SBATCH --job-name=sve_preprocess
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G

# Single SLURM job that parses caller VCFs into raw breakpoints in a shared
# directory (callset_bps/).  Runs once per sample before the SVEnsemble2
# model x training_set array, eliminating redundant VCF parsing.
#
# Usage (called from master.sh):
#   sbatch preprocess_array.sh <source_file>

set -euo pipefail

SOURCE_FILE="$1"
source "$SOURCE_FILE"

# --- Logging helpers --------------------------------------------------------
LOG_PREFIX="[${SLURM_JOB_ID:-local}]"
log()  { echo "$(date -Iseconds) $LOG_PREFIX [$1] ${*:2}"; }
logI() { log INFO  "$@"; }
logE() { log ERROR "$@"; }
die()  { logE "$*"; exit 1; }
# ----------------------------------------------------------------------------

logI "Starting raw VCF preprocessing for sample=${SAMPLE_NAME}"
logI "Callsets: ${SVE_CALLSETS[*]}"

module load micromamba

micromamba run -n SVEnsemble2 python -m svensemble.pipeline \
    --sample_name "$SAMPLE_NAME" \
    --ground_truth_sample_name "$SVE_GROUND_TRUTH_SAMPLE_NAME" \
    --ref_path "$REF_FASTA" \
    --preprocess_raw \
    --callsets "${SVE_CALLSETS[@]}" \
    $( [[ "$SVE_VERBOSE" == "true" ]] && echo "--verbose" )

logI "Raw VCF preprocessing complete for sample=${SAMPLE_NAME}"
