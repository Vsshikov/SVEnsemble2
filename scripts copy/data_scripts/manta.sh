#!/bin/bash
#SBATCH --job-name=manta
#SBATCH --output=logs/manta/%j.out
#SBATCH --error=logs/manta/%j.err
#SBATCH --mem=10G
#SBATCH --cpus-per-task=10
#SBATCH --time=02:00:00

module load samtools micromamba

SOURCE_FILE=$1
source $SOURCE_FILE

ENV_NAME="manta_py2"
YAML_FILE="$SAMPLE_DIR/scripts/envs/manta_env.yaml"
HASH_MARKER="${HOME}/.cache/${ENV_NAME}.yaml.md5"
YAML_HASH=$(md5sum "$YAML_FILE" | cut -d' ' -f1)

if [[ "$(cat "$HASH_MARKER" 2>/dev/null)" != "$YAML_HASH" ]]; then
    micromamba create -y -f "$YAML_FILE" -n "$ENV_NAME"
    mkdir -p "$(dirname "$HASH_MARKER")"
    echo "$YAML_HASH" > "$HASH_MARKER"
fi

WORKDIR="$SAMPLE_DIR/sv_calling/manta"
INPUT_BAM="$SAMPLE_DIR/aligned/${SAMPLE_NAME}.minimap.bam"

# Create output directory if it doesn't exist
if [ ! -d "$WORKDIR" ]; then
    echo "manta dir does not exist, creating"
    mkdir -p "$WORKDIR"
fi

# Index BAM file if not already indexed
if [ ! -f $INPUT_BAM.bai ]; then
    samtools index $INPUT_BAM
fi

# Run configManta
micromamba run -n "$ENV_NAME" $MANTA_PATH/bin/configManta.py --bam $INPUT_BAM --referenceFasta $REF_FASTA --runDir $WORKDIR

# Run Manta
micromamba run -n "$ENV_NAME" python $WORKDIR/runWorkflow.py
