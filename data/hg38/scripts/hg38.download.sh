#!/bin/bash
# SBATCH --job-name=env_setup.SVEnsemble2 
# SBATCH --time=16:00:00
# SBATCH --ntasks=1
# SBATCH -c 4
# SBATCH --mem=8G

# Downloads hg38 fasta file from UCSC portal
download_dir="/stornext/Bioinf/data/bioinf-data/Papenfuss_lab_projects/SVEnsemble2/data/hg38/downloads"

# lftp -c "pget -n 20 https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz -o $download_dir/"
wget -c -P $download_dir https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz
