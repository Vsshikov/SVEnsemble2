#!/bin/bash
#SBATCH --job-name=HG00514.cram2fq
#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH -c 16
#SBATCH --mem=32G

modules="samtools"
module purge
module load ${modules}

SOURCE_FILE=$1

source $SOURCE_FILE


TEMP_BAM="${SAMPLE_DIR}/temp/${SAMPLE_NAME}.from_cram.bam"

mkdir -p "$(dirname $TEMP_BAM)"

java -jar $CRAMTOOLS_JAR bam  -I $CRAM_INPUT -R $CRAM_FASTA -O $TEMP_BAM

OUTPUT_FQ_1="${SAMPLE_DIR}/preprocessed/${SAMPLE_NAME}.R1.fq.gz"
OUTPUT_FQ_2="${SAMPLE_DIR}/preprocessed/${SAMPLE_NAME}.R2.fq.gz"

for output_file in $OUTPUT_FQ_1 $OUTPUT_FQ_2; do
    echo $output_file >> $SAMPLE_DIR/preprocessed/output_files.txt
done

export REF_PATH=$CRAM_REF_PATH
export REF_CACHE=$CRAM_REF_CACHE
samtools view -b  -T $CRAM_FASTA -o $TEMP_BAM $CRAM_INPUT

# If there is no index for the BAM file, index it.
if [ ! -f $TEMP_BAM.bai ]; then
    samtools index $TEMP_BAM
fi

java -Xmx12g -Dsamjdk.reference_fasta=${CRAM_FASTA} \
    -jar $BAZAM_JAR \
    -r1 $OUTPUT_FQ_1 \
    -r2 $OUTPUT_FQ_2 \
    -bam $TEMP_BAM

# Prevent the script from exiting until the output files are created
while [ ! -f $OUTPUT_FQ_1 ] || [ ! -f $OUTPUT_FQ_2 ]; do
    sleep 5s
done