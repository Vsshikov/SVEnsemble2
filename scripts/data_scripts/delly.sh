#!/bin/bash

#SBATCH --job-name=delly
#SBATCH --output=logs/standalone/%x_%j.out
#SBATCH --error=logs/standalone/%x_%j.err
#SBATCH --mem=31GB
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00

# Load required modules
MODULES="samtools"

SOURCE_FILE=$1
source $SOURCE_FILE

DELLY_BIN="${DELLY_PATH}/delly_v1.5.0_linux_x86_64bit"

# EXCLUDE_BED="/home/users/allstaff/shikov.v/projects/Papenfuss_SVEnsemble2/data/hg38/downloads/ENCFF356LFX.bed"

WORKDIR="${SAMPLE_DIR}/sv_calling/delly"
INPUT_BAM="${SAMPLE_DIR}/aligned/${SAMPLE_NAME}.minimap.bam"
OUTPUT_BCF="${WORKDIR}/out/$(basename "${INPUT_BAM}" .minimap.bam).delly.bcf"

# Prepare environment
module purge
module load ${MODULES}

# Create output directory if it doesn't exist
if [ ! -d "$WORKDIR/out" ]; then
    echo "delly dir does not exist, creating"
    mkdir -p "$WORKDIR/out"
fi

# Create a sorted bam if not present.
SORTED_BAM="${WORKDIR}/temp/$(basename "${INPUT_BAM}" .minimap.bam).sorted.bam"
SORT_TMP="$SAMPLE_DIR/temp/samtools_sort_delly"
mkdir -p "$(dirname "$SORT_TMP")"
if [ ! -f "${SORTED_BAM}" ]; then
    mkdir -p "${WORKDIR}/temp"
    samtools sort -T "$SORT_TMP" -o "${SORTED_BAM}" "${INPUT_BAM}"
    samtools index "${SORTED_BAM}"
fi

# Run DELLY
"${DELLY_BIN}" call -g "${REF_FASTA}" -o "${OUTPUT_BCF}" "${SORTED_BAM}"
