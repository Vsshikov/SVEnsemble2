#!/bin/bash

download_dir="/home/users/allstaff/shikov.v/projects/Papenfuss_SVEnsemble2/data/HG002/downloads"

url="https://ftp.ncbi.nlm.nih.gov/ReferenceSamples/giab/data/AshkenazimTrio/HG002_NA24385_son/NIST_HiSeq_HG002_Homogeneity-10953946/NHGRI_Illumina300X_AJtrio_novoalign_bams/HG002.GRCh38.60x.1.bam"

wget -O $download_dir/HG002.GRCh38.60x.1.bam "$url"
wget -O $download_dir/HG002.GRCh38.60x.1.bam.bai "$url.bai"
