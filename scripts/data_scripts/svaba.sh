#!/bin/bash

#SBATCH --job-name=svaba
#SBATCH --output=logs/standalone/%x_%j.out
#SBATCH --error=logs/standalone/%x_%j.err
#SBATCH --mem=32G
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00

MODULES="samtools micromamba"

SOURCE_FILE=$1
source $SOURCE_FILE

module purge
module load $MODULES

ENV_NAME="svaba"
YAML_FILE="$SAMPLE_DIR/scripts/envs/svaba.yaml"
HASH_MARKER="${HOME}/.cache/${ENV_NAME}.yaml.md5"
YAML_HASH=$(md5sum "$YAML_FILE" | cut -d' ' -f1)
ENV_SENTINEL="$(micromamba info --json 2>/dev/null | python3 -c "import sys,json; print(json.load(sys.stdin)['envs directories'][0])")/${ENV_NAME}/conda-meta/history"

if [[ "$(cat "$HASH_MARKER" 2>/dev/null)" != "$YAML_HASH" ]] || [[ ! -s "$ENV_SENTINEL" ]]; then
    micromamba create -y -f "$YAML_FILE" -n "$ENV_NAME"
    mkdir -p "$(dirname "$HASH_MARKER")"
    echo "$YAML_HASH" > "$HASH_MARKER"
fi

INPUT_BAM="${SAMPLE_DIR}/aligned/${SAMPLE_NAME}.minimap.bam"
WORKDIR="${SAMPLE_DIR}/sv_calling/svaba"
OUTDIR="${WORKDIR}/out"

mkdir -p "$OUTDIR"

# SvABA needs a sorted, indexed BAM. Reuse the Delly sorted copy if available,
# otherwise create one under svaba/temp/.
DELLY_SORTED="${SAMPLE_DIR}/sv_calling/delly/temp/${SAMPLE_NAME}.sorted.bam"
if [[ -f "$DELLY_SORTED" && -f "${DELLY_SORTED}.bai" ]]; then
    SORTED_BAM="$DELLY_SORTED"
else
    SORTED_BAM="${WORKDIR}/temp/${SAMPLE_NAME}.sorted.bam"
    SORT_TMP="$SAMPLE_DIR/temp/samtools_sort_svaba"
    mkdir -p "$(dirname "$SORT_TMP")"
    if [[ ! -f "$SORTED_BAM" ]]; then
        mkdir -p "${WORKDIR}/temp"
        samtools sort -T "$SORT_TMP" -@ "${SLURM_CPUS_PER_TASK:-16}" \
            -o "$SORTED_BAM" "$INPUT_BAM"
        samtools index "$SORTED_BAM"
    fi
fi

# Run SvABA in germline mode.
# -L 6: require 6+ mate reads for cross-chromosome lookup
# -I:   skip mate-region lookup for inter-chromosomal mates
cd "$OUTDIR"
micromamba run -n "$ENV_NAME" svaba run \
    -t "$SORTED_BAM" \
    -p "${SLURM_CPUS_PER_TASK:-16}" \
    -L 6 -I \
    -a "$SAMPLE_NAME" \
    -G "$REF_FASTA"
