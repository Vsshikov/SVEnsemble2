#!/bin/bash
#SBATCH --job-name=downsample
#SBATCH --time=3:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1

module load gatk samtools

gatk DownsampleByDuplicateSet --fraction-to-keep 0.5 -I ../HG002/aligned/HG002.GRCh38.60x.1.bam -O aligned/HG002.GRCh38.30x.1.bam
