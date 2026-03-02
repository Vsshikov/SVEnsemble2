#!/bin/bash

#SBATCH --job-name=sniffles
#SBATCH --output=logs/standalone/%x_%j.out
#SBATCH --error=logs/standalone/%x_%j.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

SOURCE_FILE=$1

source $SOURCE_FILE

modules="micromamba"
module load $modules

ENV_NAME="sniffles2"
YAML_FILE="$SAMPLE_DIR/scripts/envs/sniffles.yaml"
HASH_MARKER="${HOME}/.cache/${ENV_NAME}.yaml.md5"
YAML_HASH=$(md5sum "$YAML_FILE" | cut -d' ' -f1)

if [[ "$(cat "$HASH_MARKER" 2>/dev/null)" != "$YAML_HASH" ]]; then
    micromamba create -y -f "$YAML_FILE" -n "$ENV_NAME"
    mkdir -p "$(dirname "$HASH_MARKER")"
    echo "$YAML_HASH" > "$HASH_MARKER"
fi

OUTPUT_VCF="$SAMPLE_DIR/sv_calling/sniffles/out/${SAMPLE_NAME}.long.sniffles.vcf"

mkdir -p "$(dirname "$OUTPUT_VCF")"

micromamba run -n "$ENV_NAME" sniffles \
    -i "$SAMPLE_DIR/aligned_long/${SAMPLE_NAME}.long.minimap.bam" \
    -v "$OUTPUT_VCF" \
    --reference "$REF_FASTA" \
    --threads "${SLURM_CPUS_PER_TASK:-8}"
