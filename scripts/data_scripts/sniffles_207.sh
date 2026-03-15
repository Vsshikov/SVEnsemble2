#!/bin/bash

#SBATCH --job-name=sniffles_207
#SBATCH --output=logs/standalone/%x_%j.out
#SBATCH --error=logs/standalone/%x_%j.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

SOURCE_FILE=$1

source $SOURCE_FILE

MODULES="micromamba"
module load $MODULES

ENV_NAME="sniffles2_207"
YAML_FILE="$SAMPLE_DIR/scripts/envs/sniffles_207.yaml"
HASH_MARKER="${HOME}/.cache/${ENV_NAME}.yaml.md5"
YAML_HASH=$(md5sum "$YAML_FILE" | cut -d' ' -f1)
ENV_SENTINEL="$(micromamba info --json 2>/dev/null | python3 -c "import sys,json; print(json.load(sys.stdin)['envs directories'][0])")/${ENV_NAME}/conda-meta/history"

if [[ "$(cat "$HASH_MARKER" 2>/dev/null)" != "$YAML_HASH" ]] || [[ ! -s "$ENV_SENTINEL" ]]; then
    micromamba create -y -f "$YAML_FILE" -n "$ENV_NAME"
    mkdir -p "$(dirname "$HASH_MARKER")"
    echo "$YAML_HASH" > "$HASH_MARKER"
fi

OUTPUT_VCF="$SAMPLE_DIR/sv_calling/sniffles_207/out/${SAMPLE_NAME}.long.sniffles_207.vcf"

mkdir -p "$(dirname "$OUTPUT_VCF")"

micromamba run -n "$ENV_NAME" sniffles \
    -i "$SAMPLE_DIR/aligned_long/${SAMPLE_NAME}.long.minimap.bam" \
    -v "$OUTPUT_VCF" \
    --minsvlen 50 \
    --long-ins-length 10000 \
    --long-del-length 100000 \
    -t 6 \
    --allow-overwrite
