#!/bin/bash

#SBATCH --job-name=CHM13.gridss
#SBATCH --mem=31GB
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00

# Load required modules
MODULES="anaconda3 R curl samtools bwa"

# Define paths
DATA_DIR="/home/users/allstaff/shikov.v/projects/Papenfuss_SVEnsemble2/data/CHM13"
GRIDSS_PATH="/home/users/allstaff/shikov.v/projects/Papenfuss_SVEnsemble2/callers/gridss"
GRIDSS_JAR="${GRIDSS_PATH}/gridss-2.13.2-gridss-jar-with-dependencies.jar"

GENOME_FASTA="/home/users/allstaff/shikov.v/projects/Papenfuss_SVEnsemble2/data/hg38/downloads/hg38.no_decoys.fa"
EXCLUDE_BED="/home/users/allstaff/shikov.v/projects/Papenfuss_SVEnsemble2/data/hg38/downloads/ENCFF356LFX.bed"

WORKDIR="${DATA_DIR}/sv_calling/gridss"
INPUT_BAM="${DATA_DIR}/aligned/CHM13.minimap.bam"
OUTPUT_VCF="${WORKDIR}/out/$(basename "${INPUT_BAM}" .bam).gridss.vcf"

# Prepare environment
module purge
module load ${MODULES}

# Create output directory if it doesn't exist
mkdir -p "${WORKDIR}/out"

# Run GRIDSS
"${GRIDSS_PATH}/gridss" \
    -r "${GENOME_FASTA}" \
    -j "${GRIDSS_JAR}" \
    -o "${OUTPUT_VCF}" \
    -b "${EXCLUDE_BED}" \
    "${INPUT_BAM}"
