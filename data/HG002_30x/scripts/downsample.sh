#!/bin/bash
#SBATCH --job-name=downsample
#SBATCH --time=3:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G

modules="gatk samtools"
module purge
module load ${modules}

INPUT_SAMPLE_NAME=$1
OUT_SAMPLE_NAME=$2
DOWNSAMPLE_FACTOR=$3

IN_BAM_PATH="/home/users/allstaff/shikov.v/projects/Papenfuss_SVEnsemble2/data/${INPUT_SAMPLE_NAME}/aligned/${INPUT_SAMPLE_NAME}.bam"
OUT_DIR="/home/users/allstaff/shikov.v/projects/Papenfuss_SVEnsemble2/data/${OUT_SAMPLE_NAME}/aligned"

OUT_BAM_PATH="${OUT_DIR}/${OUT_SAMPLE_NAME}.bam"

echo "Downsampling $IN_BAM_PATH to $OUT_BAM_PATH"

gatk DownsampleSam -I "$IN_BAM_PATH" -O "$OUT_BAM_PATH" -P $DOWNSAMPLE_FACTOR
