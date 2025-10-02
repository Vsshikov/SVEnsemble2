#!/bin/bash
#SBATCH --job-name=remove_decoys
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G

module load samtools

# Remove decoy regions from HG002 BAM using samtools and a BED file of non-decoy regions

NO_DECOY_BED="/home/users/allstaff/shikov.v/projects/Papenfuss_SVEnsemble2/data/hg38/downloads/no_decoy_regions.bed"
INPUT_BAM="/home/users/allstaff/shikov.v/projects/Papenfuss_SVEnsemble2/data/HG002/aligned/HG002.GRCh38.60x.1.bam"
OUTPUT_BAM="/home/users/allstaff/shikov.v/projects/Papenfuss_SVEnsemble2/data/HG002/aligned/HG002.GRCh38.60x.1.no_decoys.bam"

samtools view \
    -L "${NO_DECOY_BED}" \
    -o "${OUTPUT_BAM}" \
    "${INPUT_BAM}"