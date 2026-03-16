#!/bin/bash

#SBATCH --job-name=survivor
#SBATCH --output=logs/standalone/%x_%j.out
#SBATCH --error=logs/standalone/%x_%j.err
#SBATCH --mem=10GB
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00

MODULES="bcftools"

module purge
module load ${MODULES}

SOURCE_FILE=$1
MODE=${2:?Usage: survivor.sh <source_file> <short|long|all>}
source $SOURCE_FILE

OUTPUT_DIR="${SAMPLE_DIR}/sv_calling/survivor"
mkdir -p "$OUTPUT_DIR"
TEMP_VCF_DIR="${OUTPUT_DIR}/temp"
mkdir -p "$TEMP_VCF_DIR"

# ─── Build callset list based on mode and RUN_* flags ─────────────────────
SHORT_CALLSETS=()
[[ "${RUN_DELLY:-}"        == "true" ]] && SHORT_CALLSETS+=("${SAMPLE_DIR}/sv_calling/delly/out/${SAMPLE_NAME}.delly.bcf")
[[ "${RUN_GRIDSS:-}"       == "true" ]] && SHORT_CALLSETS+=("${SAMPLE_DIR}/sv_calling/gridss/out/${SAMPLE_NAME}.gridss.vcf")
[[ "${RUN_MANTA:-}"        == "true" ]] && SHORT_CALLSETS+=("${SAMPLE_DIR}/sv_calling/manta/results/variants/candidateSV.vcf.gz")

LONG_CALLSETS=()
[[ "${RUN_SNIFFLES:-}"     == "true" ]] && LONG_CALLSETS+=("${SAMPLE_DIR}/sv_calling/sniffles/out/${SAMPLE_NAME}.long.sniffles.vcf")
[[ "${RUN_SNIFFLES_207:-}" == "true" ]] && LONG_CALLSETS+=("${SAMPLE_DIR}/sv_calling/sniffles_207/out/${SAMPLE_NAME}.long.sniffles_207.vcf")
[[ "${RUN_SEVERUS:-}"      == "true" ]] && LONG_CALLSETS+=("${SAMPLE_DIR}/sv_calling/severus/all_SVs/severus_all.vcf")
[[ "${RUN_SVDSS:-}"        == "true" ]] && LONG_CALLSETS+=("${SAMPLE_DIR}/sv_calling/svdss/out/variations-final.vcf.gz")
[[ "${RUN_CUTESV:-}"      == "true" ]] && LONG_CALLSETS+=("${SAMPLE_DIR}/sv_calling/cutesv/out/${SAMPLE_NAME}.long.cutesv.vcf")

CALLSET_PATHS=()
case "$MODE" in
    short) CALLSET_PATHS=("${SHORT_CALLSETS[@]}") ;;
    long)  CALLSET_PATHS=("${LONG_CALLSETS[@]}") ;;
    all)   CALLSET_PATHS=("${SHORT_CALLSETS[@]}" "${LONG_CALLSETS[@]}") ;;
    *)     echo "ERROR: Unknown mode '$MODE'. Must be short, long, or all." >&2; exit 1 ;;
esac

N=${#CALLSET_PATHS[@]}
echo "SURVIVOR mode=${MODE}: ${N} callsets selected"

if (( N < 2 )); then
    echo "ERROR: Need at least 2 callsets for consensus, got ${N} in mode=${MODE}. Aborting." >&2
    exit 1
fi

MIN_SUPPORT=$(( (N + 1) / 2 ))
echo "Consensus threshold: ${MIN_SUPPORT} out of ${N} (ceil(N/2))"

# ─── Normalize VCF formats (gunzip, bcf-to-vcf) ──────────────────────────
callset_paths_unzipped=()
for callset_path in "${CALLSET_PATHS[@]}"; do
    if [[ $callset_path == *.gz ]]; then
        gunzip -c "$callset_path" > "${TEMP_VCF_DIR}/$(basename "$callset_path" .gz)"
        callset_paths_unzipped+=("${TEMP_VCF_DIR}/$(basename "$callset_path" .gz)")
    else
        callset_paths_unzipped+=("$callset_path")
    fi
done

callset_paths_vcfs=()
for callset_path in "${callset_paths_unzipped[@]}"; do
    if [[ $callset_path == *.bcf ]]; then
        bcftools view -O v "$callset_path" > "${TEMP_VCF_DIR}/$(basename "$callset_path" .bcf).vcf"
        callset_paths_vcfs+=("${TEMP_VCF_DIR}/$(basename "$callset_path" .bcf).vcf")
    else
        callset_paths_vcfs+=("$callset_path")
    fi
done

echo "${#callset_paths_vcfs[@]} files ready for SURVIVOR merge"

# ─── Write manifest and run SURVIVOR ──────────────────────────────────────
MANIFEST="${OUTPUT_DIR}/${SAMPLE_NAME}.survivor_${MODE}.callsets.txt"
printf "%s\n" "${callset_paths_vcfs[@]}" > "$MANIFEST"

OUTPUT_VCF="${OUTPUT_DIR}/${SAMPLE_NAME}.survivor_${MODE}.vcf"

$SURVIVOR_PATH merge "$MANIFEST" 1000 "$MIN_SUPPORT" 1 1 0 30 "$OUTPUT_VCF"

echo "Done: ${OUTPUT_VCF}"
