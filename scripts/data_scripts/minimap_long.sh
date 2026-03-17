#!/bin/bash
#SBATCH --job-name=minimap_long
#SBATCH --output=logs/standalone/%x_%j.out
#SBATCH --error=logs/standalone/%x_%j.err
#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH -c 32
#SBATCH --mem=128G

set -euo pipefail

SOURCE_FILE=$1

source $SOURCE_FILE

# Check if aligned dir exists
if [ ! -d "$SAMPLE_DIR/aligned_long" ]; then
    echo "aligned dir for long reads does not exist, creating"
    mkdir -p "$SAMPLE_DIR/aligned_long"
fi

MODULES="samtools"
module purge
module load ${MODULES}

OUT_BAM="$SAMPLE_DIR/aligned_long/${SAMPLE_NAME}.long.minimap.bam"
SORT_TMP="$SAMPLE_DIR/temp/samtools_sort_minimap_long"
mkdir -p "$(dirname "$SORT_TMP")"

declare -A MINIMAP_SOURCE_MAPPING=(
    ["pacbio"]="map-pb"
    ["nanopore"]="map-ont"
    ["hifi"]="map-hifi"
)

# Resolve preset: explicit MINIMAP_PRESET > source-type mapping > default map-ont
if [ -n "$MINIMAP_PRESET" ]; then
    RESOLVED_PRESET="$MINIMAP_PRESET"
elif [ -n "$READS_LONG_SOURCE" ] && [ -n "${MINIMAP_SOURCE_MAPPING[$READS_LONG_SOURCE]+_}" ]; then
    RESOLVED_PRESET="${MINIMAP_SOURCE_MAPPING[$READS_LONG_SOURCE]}"
else
    RESOLVED_PRESET="map-ont"
fi

echo "Using minimap2 preset for long reads: $RESOLVED_PRESET"

# Validate all input files exist before opening output (prevents destroying
# an existing BAM when inputs are missing, e.g. purged from scratch).
for f in "${READS_LONG_PATHS[@]}"; do
    [ -f "$f" ] || { echo "ERROR: Input file not found: $f" >&2; exit 1; }
done

# For BAM inputs, verify BGZF integrity (catches truncated/corrupt files
# before committing to a 30+ minute alignment).
if [[ "${READS_LONG_PATHS[0]}" == *.bam ]]; then
    for f in "${READS_LONG_PATHS[@]}"; do
        samtools quickcheck "$f" || {
            echo "ERROR: BAM file failed integrity check (truncated/corrupt): $f" >&2
            echo "Re-download or regenerate this file before retrying." >&2
            exit 1
        }
    done
fi

OUT_BAM_TMP="${OUT_BAM}.tmp.bam"

if [[ "${READS_LONG_PATHS[0]}" == *.bam ]]; then
    echo "Input is BAM — converting to FASTQ before alignment"
    samtools cat "${READS_LONG_PATHS[@]}" \
        | samtools fastq - \
        | $MINIMAP2_PATH -t 16 -ax "$RESOLVED_PRESET" $REF_FASTA - \
        | samtools sort -m 4G -@16 -T "$SORT_TMP" -o "$OUT_BAM_TMP" -
else
    $MINIMAP2_PATH -t 16 -ax "$RESOLVED_PRESET" $REF_FASTA "${READS_LONG_PATHS[@]}" \
        | samtools sort -m 4G -@16 -T "$SORT_TMP" -o "$OUT_BAM_TMP" -
fi

mv "$OUT_BAM_TMP" "$OUT_BAM"
samtools index "$OUT_BAM"

# Prevent job from finishing until the bam and index files can be found
while [ ! -f "$OUT_BAM" ] || [ ! -f "$OUT_BAM.bai" ]; do
    sleep 5s
done