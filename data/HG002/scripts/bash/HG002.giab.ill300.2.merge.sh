#!/bin/bash
# SBATCH --job-name=HG002.2.merge
# SBATCH --time=1:00:00
# SBATCH --ntasks=1
# SBATCH -c 2
# SBATCH --mem=8G

# Downloads hg38 fasta file from UCSC portal
download_dir="/home/users/allstaff/shikov.v/projects/Papenfuss_SVEnsemble2/data/HG002/downloads"
store_dir="/home/users/allstaff/shikov.v/projects/Papenfuss_SVEnsemble2/data/HG002/reads"

find $download_dir | grep _R1_ | xargs gzip -c > ${download_dir}/R1_merged.fastq.gz
find $download_dir | grep _R2_ | xargs gzip -c > ${download_dir}/R2_merged.fastq.gz

# rsync --verbose --progress --compress --times --perms --recursive ${download_dir} ${store_dir}
