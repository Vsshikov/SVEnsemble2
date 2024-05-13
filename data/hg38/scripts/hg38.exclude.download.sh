#!/bin/bash
# SBATCH --job-name=env_setup.SVensemble2 
# SBATCH --time=16:00:00
# SBATCH --ntasks=1
# SBATCH -c 4
# SBATCH --mem=8G

# Downloads hg38 fasta file from UCSC portal
download_dir="/stornext/Bioinf/data/bioinf-data/Papenfuss_lab_projects/SVensemble2/data/hg38/downloads"

wget -c -P $download_dir https://github.com/PapenfussLab/gridss/blob/master/example/ENCFF356LFX.bed
