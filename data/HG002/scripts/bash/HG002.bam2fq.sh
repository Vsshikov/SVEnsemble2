#!/bin/bash
#SBATCH --job-name=HG002.bam2fq
#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH -c 16
#SBATCH --mem=32G

module load samtools

data_dir="/home/users/allstaff/shikov.v/projects/Papenfuss_SVEnsemble2/data/HG002"
input_bam="${data_dir}/aligned/HG002.GRCh38.60x.1.bam"

output_dir="${data_dir}/fq"
mkdir -p "$output_dir"
base="$(basename "$input_bam" .bam)"
output_fq_1="${output_dir}/${base}.R1.fq.gz"
output_fq_2="${output_dir}/${base}.R2.fq.gz"

samtools bam2fq -@ 8  --verbosity 1 -1 $output_fq_1 -2 $output_fq_2 -0 /dev/null -s /dev/null -n $input_bam
