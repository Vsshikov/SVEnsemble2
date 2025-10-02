#!/bin/bash
#SBATCH --job-name=CHM13.download
#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH -c 1
#SBATCH --mem=4G

download_dir="/home/users/allstaff/shikov.v/projects/Papenfuss_SVEnsemble2/data/CHM13/downloads"

urls=(
    "ftp://ftp.sra.ebi.ac.uk/vol1/fastq/ERR134/005/ERR1341795/ERR1341795_1.fastq.gz"
    "ftp://ftp.sra.ebi.ac.uk/vol1/fastq/ERR134/005/ERR1341795/ERR1341795_2.fastq.gz"
)

for url in "${urls[@]}"; do
    wget -P $download_dir -nc $url
done
