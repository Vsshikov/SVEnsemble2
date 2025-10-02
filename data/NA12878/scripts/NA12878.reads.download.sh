#!/bin/bash
#SBATCH --job-name=download.NA12878
#SBATCH --time=16:00:00
#SBATCH --ntasks=1
#SBATCH --mem=8G

# Downloads Illumina Platinum pedigree PCR-free high coverage datasets
# from IGSR: The International Genome Sample Resource (https://www.internationalgenome.org/data-portal/sample/NA12878)
download_dir="/home/users/allstaff/shikov.v/projects/SVensemble2/data/NA12878/downloads"

# ftp://ftp.sra.ebi.ac.uk/vol1/fastq/ERR194/ERR194147/ERR194147_2.fastq.gz ftp://ftp.sra.ebi.ac.uk/vol1/fastq/ERR194/ERR194147/ERR194147_1.fastq.gz

links=(ftp://ftp.sra.ebi.ac.uk/vol1/fastq/ERR194/ERR194147/ERR194147_2.fastq.gz ftp://ftp.sra.ebi.ac.uk/vol1/fastq/ERR194/ERR194147/ERR194147_1.fastq.gz)

for link in ${links}; do
    bn=$(basename $link)
    wget -c -P $download_dir $link
    gunzip $(echo ${download_dir}/${bn})
done
