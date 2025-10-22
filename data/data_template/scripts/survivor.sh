#!/bin/bash

#SBATCH --job-name=survivor
#SBATCH --mem=10GB
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00

MODULES="bcftools"

module purge
module load ${MODULES}

# Accept sample name as an argument
SAMPLE_NAME=$1

SURVIVOR_PATH="/home/users/allstaff/shikov.v/projects/Papenfuss_SVEnsemble2/callers/SURVIVOR-master/Debug/SURVIVOR"

DATA_DIR="/home/users/allstaff/shikov.v/projects/Papenfuss_SVEnsemble2/data"


OUTPUT_DIR="${DATA_DIR}/${SAMPLE_NAME}/sv_calling/survivor"
mkdir -p $OUTPUT_DIR
TEMP_VCF_DIR="${OUTPUT_DIR}/temp"
mkdir -p $TEMP_VCF_DIR

CALLSET_PATHS=(
    "${DATA_DIR}/${SAMPLE_NAME}/sv_calling/delly/out/${SAMPLE_NAME}.delly.bcf"
    "${DATA_DIR}/${SAMPLE_NAME}/sv_calling/gridss/out/${SAMPLE_NAME}.gridss.vcf"
    "${DATA_DIR}/${SAMPLE_NAME}/sv_calling/manta/results/variants/candidateSV.vcf.gz"
)

# Some of the input VCFs may be gzipped and some files may be in bcf format
# If there are any archived files, extract them
# After that, check for BCFs and convert them to VCF
# If the file is already a VCF, just copy it to the temp directory
# Also reflect those changes in the CALLSET_PATHS array creating CALLSET_TEMP_PATHS from ls command

# First unzip the gzipped files. They may still be in BCF format after unzipping
CALLSET_PATHS_UNZIPPED=()
echo "Unzipping gzipped files"
for CALLSET_PATH in "${CALLSET_PATHS[@]}"; do
    if [[ $CALLSET_PATH == *.gz ]]; then
        gunzip -c $CALLSET_PATH > ${TEMP_VCF_DIR}/$(basename $CALLSET_PATH .gz)
        CALLSET_PATHS_UNZIPPED+=("${TEMP_VCF_DIR}/$(basename $CALLSET_PATH .gz)")
    else
        CALLSET_PATHS_UNZIPPED+=($CALLSET_PATH)
    fi
done
echo "Unzipping complete, now ${#CALLSET_PATHS_UNZIPPED[@]} files in temp directory"
# Now iterate over CALLSET_PATHS_UNZIPPED and check for BCFs. If found, convert them to VCF
CALLSET_PATHS_VCFS=()
echo "Converting BCFs to VCFs"
for CALLSET_PATH in "${CALLSET_PATHS_UNZIPPED[@]}"; do
    if [[ $CALLSET_PATH == *.bcf ]]; then
        bcftools view -O v $CALLSET_PATH > ${TEMP_VCF_DIR}/$(basename $CALLSET_PATH .bcf).vcf
        CALLSET_PATHS_VCFS+=("${TEMP_VCF_DIR}/$(basename $CALLSET_PATH .bcf).vcf")
    else
        CALLSET_PATHS_VCFS+=($CALLSET_PATH)
    fi
done
echo "Conversion complete, now ${#CALLSET_PATHS_VCFS[@]} files in temp directory"


# Write the callset paths to a file in the output directory
printf "%s\n" "${CALLSET_PATHS_VCFS[@]}" > ${OUTPUT_DIR}/${SAMPLE_NAME}.survivor.callsets.txt

$SURVIVOR_PATH merge ${OUTPUT_DIR}/${SAMPLE_NAME}.survivor.callsets.txt 1000 2 1 1 0 30 ${OUTPUT_DIR}/${SAMPLE_NAME}.survivor.vcf
