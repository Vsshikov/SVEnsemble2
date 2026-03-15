#!/bin/bash
# =============================================================================
# download_sra.sh — Download SRA accessions via prefetch + fasterq-dump
#
# Submits paired SLURM jobs (prefetch → fasterq-dump) for each accession in the
# supplied filelist.  Logs go to ${SAMPLE_DIR}/logs/download/.
#
# Usage:
#   bash data/<sample>/scripts/download_sra.sh <filelist> [<config>]
#
#   filelist — file with one SRA accession per line (required)
#   config   — run config file (default: <script_dir>/config/runs/default.sh)
# =============================================================================
set -euo pipefail

filelist="${1:?Usage: $0 <filelist> [<config>]}"
[[ -f "$filelist" ]] || {
    echo "Error: filelist not found: $filelist" >&2
    exit 1
}

_MASTER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_DEFAULT_CFG="${_MASTER_DIR}/config/runs/default.sh"
SOURCE_FILE=$(realpath -s "${2:-$_DEFAULT_CFG}")
unset _MASTER_DIR _DEFAULT_CFG

[[ -f "$SOURCE_FILE" ]] || {
    echo "Usage: $0 <filelist> [<config>]" >&2
    echo "  No config found at $SOURCE_FILE" >&2
    exit 1
}

source "$SOURCE_FILE"

JOB_NAME="${SAMPLE_NAME}_download"
MEM=16GB
CPUS_PER_TASK=8
TIME=04:00:00

LOG_DIR="${SAMPLE_DIR}/logs/download"
mkdir -p "$LOG_DIR"

module purge
module load sra-toolkit

while read -r line; do
    [[ -z "$line" || "$line" == \#* ]] && continue

    prefetch_job_id=$(sbatch \
        --job-name="${JOB_NAME}_prefetch" \
        --mem=2GB \
        --cpus-per-task=2 \
        --time=${TIME} \
        --output="${LOG_DIR}/%j_prefetch_${line}.out" \
        --error="${LOG_DIR}/%j_prefetch_${line}.err" \
        --parsable \
        --wrap="module purge; module load sra-toolkit; prefetch --max-size 50G -O ${SAMPLE_DIR}/downloads/ ${line}"
    )

    sbatch \
        --job-name="${JOB_NAME}_fasterq" \
        --mem=${MEM} \
        --cpus-per-task=${CPUS_PER_TASK} \
        --time=${TIME} \
        --output="${LOG_DIR}/%j_fasterq_${line}.out" \
        --error="${LOG_DIR}/%j_fasterq_${line}.err" \
        --dependency=afterok:${prefetch_job_id} \
        --wrap="module purge; module load sra-toolkit; fasterq-dump -O ${SAMPLE_DIR}/downloads/ ${SAMPLE_DIR}/downloads/${line}"
done < "${filelist}"
