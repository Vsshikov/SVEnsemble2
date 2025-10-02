#!/bin/bash
# SBATCH --job-name=env_setup.SVEnsemble2
# SBATCH --time=1:00:00
# SBATCH --ntasks=1
# SBATCH -c 2
# SBATCH --mem=8G

# Downloads hg38 fasta file from UCSC portal
download_dir="/home/users/allstaff/shikov.v/projects/Papenfuss_SVEnsemble2/data/hg38/downloads"

# lftp -c "pget -n 20 https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz -o $download_dir/"
wget -c -P $download_dir ftp://ftp.1000genomes.ebi.ac.uk/vol1/ftp/technical/reference/GRCh38_reference_genome/GRCh38_full_analysis_set_plus_decoy_hla.fa
