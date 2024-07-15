#!/bin/bash
#SBATCH --job-name=download.NA12878
#SBATCH --time=16:00:00
#SBATCH --ntasks=1
#SBATCH --mem=8G

# Downloads Illumina Platinum pedigree PCR-free high coverage datasets
# from IGSR: The International Genome Sample Resource (https://www.internationalgenome.org/data-portal/sample/NA12878)
download_dir="/home/users/allstaff/shikov.v/projects/SVensemble2/data/NA12878/downloads"

wget -c -P $download_dir ftp://ftp.1000genomes.ebi.ac.uk/vol1/ftp/data_collections/illumina_platinum_pedigree/data/CEU/NA12878/alignment/NA12878.alt_bwamem_GRCh38DH.20150706.CEU.illumina_platinum_ped.cram
wget -c -P $download_dir ftp://ftp.1000genomes.ebi.ac.uk/vol1/ftp/data_collections/illumina_platinum_pedigree/data/CEU/NA12878/alignment/NA12878.alt_bwamem_GRCh38DH.20150706.CEU.illumina_platinum_ped.cram.crai
wget -c -P $download_dir ftp://ftp.1000genomes.ebi.ac.uk/vol1/ftp/data_collections/illumina_platinum_pedigree/data/CEU/NA12878/alignment/NA12878.alt_bwamem_GRCh38DH.20150706.CEU.illumina_platinum_ped.bam.bas
