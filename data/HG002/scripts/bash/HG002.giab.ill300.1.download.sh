#!/bin/bash
# SBATCH --job-name=env_setup.SVEnsemble2 
# SBATCH --time=1:00:00
# SBATCH --ntasks=1
# SBATCH -c 2
# SBATCH --mem=8G

# Downloads hg38 fasta file from UCSC portal
source_list_path="/home/users/allstaff/shikov.v/projects/SVEnsemble2/data/HG002/data/filelist.tsv"
download_dir="/home/users/allstaff/shikov.v/projects/SVEnsemble2/data/HG002/vast/downloads"
store_dir="/home/users/allstaff/shikov.v/projects/SVEnsemble2/data/HG002/downloads"

# lftp -c "pget -n 20 https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz -o $download_dir/"
wget -c -P $download_dir $source_file

cut -f1 $source_list_path | xargs wget -c -P $download_dir
cut -f3 $source_list_path | xargs wget -c -P $download_dir

rsync --verbose --progress --compress --times --perms --recursive ${download_dir} ${store_dir}
