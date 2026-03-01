#!/bin/bash

#SBATCH --job-name=gridss
#SBATCH --output=logs/gridss/%j.out
#SBATCH --error=logs/gridss/%j.err
#SBATCH --mem=31GB
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00

# Load required modules
modules="anaconda3 R curl samtools bwa"

SOURCE_FILE=$1
source $SOURCE_FILE

# Define paths
GRIDSS_JAR="${GRIDSS_PATH}/gridss-2.13.2-gridss-jar-with-dependencies.jar"

EXCLUDE_BED="/home/users/allstaff/shikov.v/projects/Papenfuss_SVEnsemble2/data/hg19/downloads/ENCFF001TDO.bed"

WORKDIR="${SAMPLE_DIR}/sv_calling/gridss"
INPUT_BAM="${SAMPLE_DIR}/aligned/${SAMPLE_NAME}.minimap.bam"
OUTPUT_VCF="${WORKDIR}/out/$(basename "${INPUT_BAM}" .minimap.bam).gridss.vcf"

# Prepare environment
module purge
module load ${modules}

# Create output directory if it doesn't exist
if [ ! -d "$WORKDIR/out" ]; then
    echo "gridss dir does not exist, creating"
    mkdir -p "$WORKDIR/out"
fi

# Run GRIDSS
"${GRIDSS_PATH}/gridss" \
    -r "${REF_FASTA}" \
    -j "${GRIDSS_JAR}" \
    -o "${OUTPUT_VCF}" \
    -b "${EXCLUDE_BED}" \
    "${INPUT_BAM}"
