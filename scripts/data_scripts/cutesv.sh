#!/bin/bash

#SBATCH --job-name=cutesv
#SBATCH --output=logs/standalone/%x_%j.out
#SBATCH --error=logs/standalone/%x_%j.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G

SOURCE_FILE=$1

source $SOURCE_FILE

MODULES="micromamba"
module load $MODULES

ENV_NAME="cutesv"
YAML_FILE="$SAMPLE_DIR/scripts/envs/cutesv.yaml"
HASH_MARKER="${HOME}/.cache/${ENV_NAME}.yaml.md5"
YAML_HASH=$(md5sum "$YAML_FILE" | cut -d' ' -f1)
ENV_SENTINEL="$(micromamba info --json 2>/dev/null | python3 -c "import sys,json; print(json.load(sys.stdin)['envs directories'][0])")/${ENV_NAME}/conda-meta/history"

if [[ "$(cat "$HASH_MARKER" 2>/dev/null)" != "$YAML_HASH" ]] || [[ ! -s "$ENV_SENTINEL" ]]; then
    micromamba create -y -f "$YAML_FILE" -n "$ENV_NAME"
    mkdir -p "$(dirname "$HASH_MARKER")"
    echo "$YAML_HASH" > "$HASH_MARKER"
fi

INPUT_BAM="$SAMPLE_DIR/aligned_long/${SAMPLE_NAME}.long.minimap.bam"
OUTPUT_VCF="$SAMPLE_DIR/sv_calling/cutesv/out/${SAMPLE_NAME}.long.cutesv.vcf"
WORKDIR="$SAMPLE_DIR/sv_calling/cutesv/work"

mkdir -p "$(dirname "$OUTPUT_VCF")"
mkdir -p "$WORKDIR"

# Platform-specific clustering parameters
PLATFORM_ARGS=()
case "${READS_LONG_SOURCE:-}" in
    hifi)
        PLATFORM_ARGS=(
            --max_cluster_bias_INS 1000
            --diff_ratio_merging_INS 0.9
            --max_cluster_bias_DEL 1000
            --diff_ratio_merging_DEL 0.5
        )
        ;;
    nanopore|ont)
        PLATFORM_ARGS=(
            --max_cluster_bias_INS 100
            --diff_ratio_merging_INS 0.3
            --max_cluster_bias_DEL 100
            --diff_ratio_merging_DEL 0.3
        )
        ;;
    *)
        # PacBio CLR / unknown: use cuteSV built-in defaults
        ;;
esac

micromamba run -n "$ENV_NAME" cuteSV \
    "$INPUT_BAM" \
    "$REF_FASTA" \
    "$OUTPUT_VCF" \
    "$WORKDIR" \
    -t "${SLURM_CPUS_PER_TASK:-16}" \
    -S "$SAMPLE_NAME" \
    --genotype \
    --report_readid \
    --min_size 50 \
    "${PLATFORM_ARGS[@]}"
