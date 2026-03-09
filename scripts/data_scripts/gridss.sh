#!/bin/bash

#SBATCH --job-name=gridss
#SBATCH --output=logs/standalone/%x_%j.out
#SBATCH --error=logs/standalone/%x_%j.err
#SBATCH --mem=31GB
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00

# Load required modules
MODULES="micromamba R curl samtools bwa"

SOURCE_FILE=$1
source $SOURCE_FILE

# Define paths
GRIDSS_JAR="${GRIDSS_PATH}/gridss-2.13.2-gridss-jar-with-dependencies.jar"

# GRIDSS_EXCLUDE_BED must be set in the master_args / config file for the
# correct genome build (hg38: ENCFF356LFX.bed; hg19: ENCFF001TDO.bed).
# Abort early rather than running GRIDSS without an exclude list.
if [[ -z "${GRIDSS_EXCLUDE_BED:-}" ]]; then
    echo "$(date -Iseconds) [ERROR] GRIDSS_EXCLUDE_BED is not set in $SOURCE_FILE" >&2
    exit 1
fi
if [[ ! -f "$GRIDSS_EXCLUDE_BED" ]]; then
    echo "$(date -Iseconds) [ERROR] GRIDSS_EXCLUDE_BED file not found: $GRIDSS_EXCLUDE_BED" >&2
    exit 1
fi
EXCLUDE_BED="$GRIDSS_EXCLUDE_BED"

WORKDIR="${SAMPLE_DIR}/sv_calling/gridss"
INPUT_BAM="${SAMPLE_DIR}/aligned/${SAMPLE_NAME}.minimap.bam"
OUTPUT_VCF="${WORKDIR}/out/$(basename "${INPUT_BAM}" .minimap.bam).gridss.vcf"
ASSEMBLY_BAM="${OUTPUT_VCF}.assembly.bam"
# GRIDSS expects a per-BAM working subdirectory named <bam>.gridss.working/ to exist
# inside WORKDIR. On reruns, if the assembly BAM already exists GRIDSS skips the
# assembly step and never recreates this directory, causing CollectGridssMetrics to fail.
ASSEMBLY_WORKING="${WORKDIR}/$(basename "${ASSEMBLY_BAM}").gridss.working"

# Prepare environment
module purge
module load ${MODULES}

# Create output and assembly working directories
mkdir -p "$WORKDIR/out"
mkdir -p "$ASSEMBLY_WORKING"

# Run GRIDSS
"${GRIDSS_PATH}/gridss" \
    --workingdir "${WORKDIR}" \
    -r "${REF_FASTA}" \
    -j "${GRIDSS_JAR}" \
    -o "${OUTPUT_VCF}" \
    -b "${EXCLUDE_BED}" \
    "${INPUT_BAM}"
