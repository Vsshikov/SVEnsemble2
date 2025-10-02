#!/bin/bash
#SBATCH --job-name=CHM13.minimap
#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH -c 32
#SBATCH --mem=80G

module load samtools

MINIMAP2="/home/users/allstaff/shikov.v/projects/Papenfuss_SVEnsemble2/utils/minimap/minimap2-2.30_x64-linux/minimap2"

REF_GENOME="/home/users/allstaff/shikov.v/projects/Papenfuss_SVEnsemble2/data/hg38/downloads/hg38.no_decoys.fa"

fq1="/home/users/allstaff/shikov.v/projects/Papenfuss_SVEnsemble2/data/CHM13/downloads/ERR1341795_1.fastq.gz"
fq2="/home/users/allstaff/shikov.v/projects/Papenfuss_SVEnsemble2/data/CHM13/downloads/ERR1341795_2.fastq.gz"

output_bam="/home/users/allstaff/shikov.v/projects/Papenfuss_SVEnsemble2/data/CHM13/aligned/CHM13.minimap.bam"

mkdir -p $(dirname $output_bam)

$MINIMAP2 -t 16 -ax sr $REF_GENOME $fq1 $fq2 | samtools sort -@16 -o $output_bam -
