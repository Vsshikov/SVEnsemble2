#!/bin/bash
#SBATCH --job-name=NA12878.bwa_mem
#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH -c 32
#SBATCH --mem=80G

module load bwa

ref_genome="/home/users/allstaff/shikov.v/projects/SVensemble2/data/hg38/downloads/hg38.no_decoys.fa"


bwa index $ref_genome
