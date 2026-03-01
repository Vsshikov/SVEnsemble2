#!/bin/bash
#SBATCH --job-name=minimap
#SBATCH --output=logs/standalone/%x_%j.out
#SBATCH --error=logs/standalone/%x_%j.err
#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH -c 32
#SBATCH --mem=80G

SOURCE_FILE=$1
source $SOURCE_FILE

# Check if aligned dir exists
if [ ! -d "$SAMPLE_DIR/aligned" ]; then
    echo "aligned dir does not exist, creating"
    mkdir -p "$SAMPLE_DIR/aligned"
fi

modules="samtools"
module purge
module load ${modules}

output_bam="$SAMPLE_DIR/aligned/${SAMPLE_NAME}.minimap.bam"

$MINIMAP2_PATH -t 16 -ax sr $REF_FASTA $READS1_PATH $READS2_PATH | samtools sort -@16 -o $output_bam -

# Prevent job from finishing until the bam file can be found
while [ ! -f "$output_bam" ]; do
    sleep 5s
done
