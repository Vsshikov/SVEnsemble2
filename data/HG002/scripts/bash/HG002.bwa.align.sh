#!/bin/bash
#SBATCH --job-name=HG002.bwa_mem
#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH -c 32
#SBATCH --mem=80G

module load bwa samtools

download_dir="/home/users/allstaff/shikov.v/projects/SVEnsemble2/data/HG002/vast/downloads"
fq1="/home/users/allstaff/shikov.v/projects/SVEnsemble2/data/HG002/vast/data/R1_merged.fastq"
fq2="/home/users/allstaff/shikov.v/projects/SVEnsemble2/data/HG002/vast/data/R2_merged.fastq"
ref_genome="/home/users/allstaff/shikov.v/projects/SVensemble2/data/hg38/downloads/hg38.no_decoys.fa"
  
if [ ! -f $fq1 ]; then
   gunzip -c $download_dir/$(basename $fq1).gz > $fq1
fi

if [ ! -f $fq2 ]; then
   gunzip -c $download_dir/$(basename $fq2).gz > $fq2
fi

output_dir="/home/users/allstaff/shikov.v/projects/SVensemble2/data/HG002/aligned"
mkdir -r $output_dir
output_bam="${output_dir}/HG002.bwa.bam"

bwa mem -t 16 $ref_genome <(gunzip $fq1) <(gunzip $fq2) | samtools sort -@16 -o $output_bam -
