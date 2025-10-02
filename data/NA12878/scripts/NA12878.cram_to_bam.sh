#!/bin/bash
#SBATCH --job-name=NA12878.cram_to_bam
#SBATCH --time=16:00:00
#SBATCH --ntasks=1
#SBATCH -c 8
#SBATCH --mem=8G

module load samtools

input="/stornext/Bioinf/data/bioinf-data/Papenfuss_lab_projects/SVensemble2/data/NA12878/downloads/NA12878.alt_bwamem_GRCh38DH.20150706.CEU.illumina_platinum_ped.cram"
ref="/stornext/Bioinf/data/bioinf-data/Papenfuss_lab_projects/SVensemble2/data/hg38/downloads/hg38.fa"
output=$(echo $input | sed 's/.cram/.bam/g')

# -T $ref 
samtools view -T $ref --bam -o $output $input
