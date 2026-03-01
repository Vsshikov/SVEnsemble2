#!/bin/bash
#SBATCH --job-name=HG002.bam2fq
#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH -c 16
#SBATCH --mem=32G

modules="samtools"
module purge
module load ${modules}

SOURCE_FILE=$1

source $SOURCE_FILE

DATA_DIR="/home/users/allstaff/shikov.v/projects/Papenfuss_SVEnsemble2/data/$SAMPLE_NAME"

OUTPUT_DIR="${DATA_DIR}/preprocessed"
mkdir -p "$OUTPUT_DIR"
OUTPUT_1="${OUTPUT_DIR}/${SAMPLE_NAME}.R1.fq.gz"
OUTPUT_2="${OUTPUT_DIR}/${SAMPLE_NAME}.R2.fq.gz"

for output_file in $OUTPUT_1 $OUTPUT_2; do
    echo $output_file >> $OUTPUT_DIR/tmp/output_files.txt
done

java -jar /home/users/allstaff/shikov.v/projects/Papenfuss_SVEnsemble2/utils/bazam/bazam.jar \
    -bam $INPUT_BAM \
    -r1 $OUTPUT_1 \
    -r2 $OUTPUT_2

# Prevent job from finishing until the fq files can be found
while [ ! -f "$OUTPUT_1" ] || [ ! -f "$OUTPUT_2" ]; do
    sleep 5s
done