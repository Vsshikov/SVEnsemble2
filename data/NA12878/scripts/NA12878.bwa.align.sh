#!/bin/bash
#SBATCH --job-name=NA12878.bwa_mem
#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH -c 32
#SBATCH --mem=80G

module load bwa samtools

fq1="/home/users/allstaff/shikov.v/projects/SVensemble2/data/NA12878/downloads/ERR194147_1.fastq"
fq2="/home/users/allstaff/shikov.v/projects/SVensemble2/data/NA12878/downloads/ERR194147_2.fastq"
ref_genome="/home/users/allstaff/shikov.v/projects/SVensemble2/data/hg38/downloads/hg38.no_decoys.fa"

output_bam="/home/users/allstaff/shikov.v/projects/SVensemble2/data/NA12878/aligned/NA12878.bwa.bam"

bwa mem -t 16 $ref_genome $fq1 $fq2 | samtools sort -@16 -o $output_bam -
