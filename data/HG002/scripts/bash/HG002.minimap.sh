#!/bin/bash
#SBATCH --job-name=HG002.minimap
#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH -c 32
#SBATCH --mem=80G

module load samtools

MINIMAP2="/home/users/allstaff/shikov.v/projects/Papenfuss_SVEnsemble2/utils/minimap/minimap2-2.30_x64-linux/minimap2"

REF_GENOME="/home/users/allstaff/shikov.v/projects/Papenfuss_SVEnsemble2/data/hg38/downloads/hg38.no_decoys.fa"

fq1="/home/users/allstaff/shikov.v/projects/Papenfuss_SVEnsemble2/data/HG002/fq/HG002.GRCh38.60x.1.R1.fq.gz"
fq2="/home/users/allstaff/shikov.v/projects/Papenfuss_SVEnsemble2/data/HG002/fq/HG002.GRCh38.60x.1.R2.fq.gz"

output_bam="/home/users/allstaff/shikov.v/projects/Papenfuss_SVEnsemble2/data/HG002/aligned/HG002.minimap.bam"

$MINIMAP2 -t 16 -ax sr $REF_GENOME $fq1 $fq2 | samtools sort -@16 -o $output_bam -
