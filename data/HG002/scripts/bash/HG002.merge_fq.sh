#!/bin/bash
#SBATCH --job-name=HG002.merge_fq
#SBATCH --time=16:00:00
#SBATCH --ntasks=1
#SBATCH -c 1
#SBATCH --mem=4G

fq_dir="/home/users/allstaff/shikov.v/projects/Papenfuss_SVEnsemble2/data/HG002/downloads"

# Find all files with _R1_ in their name in the specified directory
r1_files=($(find "$fq_dir" -type f -name "*_R1_*"))
r2_files=($(find "$fq_dir" -type f -name "*_R2_*"))

zcat ${r1_files[@]} | gzip > $fq_dir/R1_merged.fastq.gz
zcat ${r2_files[@]} | gzip > $fq_dir/R2_merged.fastq.gz
