#!/bin/bash
#SBATCH --job-name=minimap
#SBATCH --output=logs/standalone/%x_%j.out
#SBATCH --error=logs/standalone/%x_%j.err
#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH -c 32
#SBATCH --mem=80G

set -euo pipefail

SOURCE_FILE=$1

source $SOURCE_FILE

# Check if aligned dir exists
if [ ! -d "$SAMPLE_DIR/aligned" ]; then
    echo "aligned dir does not exist, creating"
    mkdir -p "$SAMPLE_DIR/aligned"
fi

MODULES="samtools"
module purge
module load ${MODULES}

output_bam="$SAMPLE_DIR/aligned/${SAMPLE_NAME}.minimap.bam"
SORT_TMP="$SAMPLE_DIR/temp/samtools_sort_minimap"
mkdir -p "$(dirname "$SORT_TMP")"

# Validate all input files exist before opening output (prevents destroying
# an existing BAM when inputs are missing, e.g. purged from scratch).
for f in "$READS1_PATH" "$READS2_PATH"; do
    [ -f "$f" ] || { echo "ERROR: Input file not found: $f" >&2; exit 1; }
done

output_bam_tmp="${output_bam}.tmp.bam"

$MINIMAP2_PATH -t 16 -ax sr $REF_FASTA $READS1_PATH $READS2_PATH | samtools sort -@16 -T "$SORT_TMP" -o "$output_bam_tmp" -

mv "$output_bam_tmp" "$output_bam"
samtools index "$output_bam"

# Prevent job from finishing until the bam and index files can be found
while [ ! -f "$output_bam" ] || [ ! -f "$output_bam.bai" ]; do
    sleep 5s
done