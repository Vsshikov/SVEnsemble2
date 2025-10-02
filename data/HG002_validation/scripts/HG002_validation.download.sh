#!/bin/bash

# HG002 validation links:

val_vcf="https://ftp-trace.ncbi.nlm.nih.gov/ReferenceSamples/giab/data/AshkenazimTrio/analysis/NIST_HG002_DraftBenchmark_defrabbV0.019-20241113/GRCh38_HG2-T2TQ100-V1.1_stvar.vcf.gz"

download_dir="/home/users/allstaff/shikov.v/projects/Papenfuss_SVEnsemble2/data/HG002_validation"

wget -O ${download_dir}/HG002_validation.vcf.gz ${val_vcf}
