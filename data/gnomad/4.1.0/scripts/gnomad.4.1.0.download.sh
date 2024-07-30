#!/bin/bash
# SBATCH --job-name=env_setup.SVensemble2 
# SBATCH --time=1:00:00
# SBATCH --ntasks=1
# SBATCH -c 2
# SBATCH --mem=8G

module load awscli/

# Downloads hg38 fasta file from UCSC portal
download_dir="/home/users/allstaff/shikov.v/projects/SVensemble2/data/gnomad/4.1.0/downloads/"

aws s3 cp s3://gnomad-public-us-east-1/release/4.1/genome_sv/gnomad.v4.1.sv.sites.vcf.gz $download_dir
aws s3 cp s3://gnomad-public-us-east-1/release/4.1/genome_sv/gnomad.v4.1.sv.sites.vcf.gz.tbi $download_dir
