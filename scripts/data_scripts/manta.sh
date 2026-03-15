#!/bin/bash
#SBATCH --job-name=manta
#SBATCH --output=logs/standalone/%x_%j.out
#SBATCH --error=logs/standalone/%x_%j.err
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
ENV_SENTINEL="$(micromamba info --json 2>/dev/null | python3 -c "import sys,json; print(json.load(sys.stdin)['envs directories'][0])")/${ENV_NAME}/conda-meta/history"

if [[ "$(cat "$HASH_MARKER" 2>/dev/null)" != "$YAML_HASH" ]] || [[ ! -s "$ENV_SENTINEL" ]]; then
    micromamba create -y -f "$YAML_FILE" -n "$ENV_NAME"
    mkdir -p "$(dirname "$HASH_MARKER")"
    echo "$YAML_HASH" > "$HASH_MARKER"
fi

WORKDIR="$SAMPLE_DIR/sv_calling/manta"
INPUT_BAM="$SAMPLE_DIR/aligned/${SAMPLE_NAME}.minimap.bam"

# On rerun, wipe the manta workdir so configManta gets a clean slate.
# Without this, configManta refuses to run if runWorkflow.py already exists.
if [[ "$RERUN" == "true" && -d "$WORKDIR" ]]; then
    echo "RERUN=true: removing existing manta workdir $WORKDIR"
    rm -rf "$WORKDIR"
fi

mkdir -p "$WORKDIR"

# Index BAM file if not already indexed
if [ ! -f $INPUT_BAM.bai ]; then
    samtools index $INPUT_BAM
fi

# Run configManta only if not already configured
if [ ! -f "$WORKDIR/runWorkflow.py" ]; then
    micromamba run -n "$ENV_NAME" $MANTA_PATH/bin/configManta.py --bam $INPUT_BAM --referenceFasta $REF_FASTA --runDir $WORKDIR
fi

# Run Manta
micromamba run -n "$ENV_NAME" python $WORKDIR/runWorkflow.py
