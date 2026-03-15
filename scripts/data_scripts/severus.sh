#!/bin/bash

#SBATCH --job-name=severus
#SBATCH --output=logs/standalone/%x_%j.out
#SBATCH --error=logs/standalone/%x_%j.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G

MODULES="micromamba"
module purge
module load $MODULES

SOURCE_FILE=$1
source $SOURCE_FILE

ENV_NAME="severus_env"
YAML_FILE="$SAMPLE_DIR/scripts/envs/severus_env.yaml"
HASH_MARKER="${HOME}/.cache/${ENV_NAME}.yaml.md5"
YAML_HASH=$(md5sum "$YAML_FILE" | cut -d' ' -f1)
ENV_SENTINEL="$(micromamba info --json 2>/dev/null | python3 -c "import sys,json; print(json.load(sys.stdin)['envs directories'][0])")/${ENV_NAME}/conda-meta/history"

if [[ "$(cat "$HASH_MARKER" 2>/dev/null)" != "$YAML_HASH" ]] || [[ ! -s "$ENV_SENTINEL" ]]; then
    micromamba create -y -f "$YAML_FILE" -n "$ENV_NAME"
    mkdir -p "$(dirname "$HASH_MARKER")"
    echo "$YAML_HASH" > "$HASH_MARKER"
fi

WORKDIR="$SAMPLE_DIR/sv_calling/severus"
INPUT_BAM="$SAMPLE_DIR/aligned_long/${SAMPLE_NAME}.long.minimap.bam"

if [ -n "$VNTR_BED" ]; then
    vntr_bed_flag="--vntr-bed $VNTR_BED"
else
    vntr_bed_flag=""
fi

micromamba run -n "$ENV_NAME" severus --target-bam $INPUT_BAM \
    --out-dir $WORKDIR -t ${SLURM_CPUS_PER_TASK} \
    $vntr_bed_flag