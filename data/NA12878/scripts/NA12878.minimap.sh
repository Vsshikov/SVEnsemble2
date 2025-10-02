#!/bin/bash
#SBATCH --job-name=NA12878.minimap
#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH -c 32
#SBATCH --mem=80G

module load samtools

MINIMAP2="/home/users/allstaff/shikov.v/projects/Papenfuss_SVEnsemble2/utils/minimap/minimap2-2.30_x64-linux/minimap2"

REF_GENOME="/home/users/allstaff/shikov.v/projects/Papenfuss_SVEnsemble2/data/hg38/downloads/hg38.no_decoys.fa"

fq1="/home/users/allstaff/shikov.v/projects/Papenfuss_SVEnsemble2/data/NA12878/downloads/downloads/ERR194147_1.fastq"
fq2="/home/users/allstaff/shikov.v/projects/Papenfuss_SVEnsemble2/data/NA12878/downloads/downloads/ERR194147_2.fastq"

output_bam="/home/users/allstaff/shikov.v/projects/Papenfuss_SVEnsemble2/data/NA12878/aligned/NA12878.minimap.bam"

$MINIMAP2 -t 16 -ax sr $REF_GENOME $fq1 $fq2 | samtools sort -@16 -o $output_bam -
