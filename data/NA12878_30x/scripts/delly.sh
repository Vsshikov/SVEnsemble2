#!/bin/bash

#SBATCH --job-name=delly
#SBATCH --mem=31GB
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00

# Load required modules
modules="samtools"

SAMPLE_NAME=$1

# Define paths
DATA_DIR="/home/users/allstaff/shikov.v/projects/Papenfuss_SVEnsemble2/data/${SAMPLE_NAME}"
DELLY_PATH="/home/users/allstaff/shikov.v/projects/Papenfuss_SVEnsemble2/callers/delly"
DELLY_BIN="${DELLY_PATH}/delly_v1.5.0_linux_x86_64bit"

GENOME_FASTA="/home/users/allstaff/shikov.v/projects/Papenfuss_SVEnsemble2/data/hg38/downloads/hg38.no_decoys.fa"
# EXCLUDE_BED="/home/users/allstaff/shikov.v/projects/Papenfuss_SVEnsemble2/data/hg38/downloads/ENCFF356LFX.bed"

WORKDIR="${DATA_DIR}/sv_calling/delly"
INPUT_BAM="${DATA_DIR}/aligned/${SAMPLE_NAME}.minimap.bam"
OUTPUT_BCF="${WORKDIR}/out/$(basename "${INPUT_BAM}" .minimap.bam).delly.bcf"

# Prepare environment
module purge
module load ${modules}

# Create output directory if it doesn't exist
mkdir -p "${WORKDIR}/out"

# Create a sorted bam if not present.
SORTED_BAM="${WORKDIR}/temp/$(basename "${INPUT_BAM}" .minimap.bam).sorted.bam"
if [ ! -f "${SORTED_BAM}" ]; then
    mkdir -p "${WORKDIR}/temp"
    samtools sort -o "${SORTED_BAM}" "${INPUT_BAM}"
    samtools index "${SORTED_BAM}"
fi

# Run DELLY
"${DELLY_BIN}" call -g "${GENOME_FASTA}" -o "${OUTPUT_BCF}" "${SORTED_BAM}"
