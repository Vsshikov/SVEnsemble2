#!/bin/bash

#SBATCH --job-name=gridss_test1
#SBATCH --mem=31GB
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00

# wd=$1

modules="anaconda3/latest R/4.4.0 java/1.8.0_211 curl/7.65.0 samtools bwa"

path_default="/home/users/allstaff/shikov.v/projects/SVensemble2/gridss_testing"
wd=$path_default
# echo "Loading packages: $modules"

module purge
module load $modules

/stornext/Bioinf/data/bioinf-data/Papenfuss_lab_projects/SVensemble2/pkg/gridss/gridss \
    -r ${wd}/data/hg38.no_decoys.fa \
    -j ${wd}/../pkg/gridss/gridss-2.13.2-gridss-jar-with-dependencies.jar \
    -o ${wd}/output/test_gridss_output.NA12878.vcf \
    -b ${wd}/data/ENCFF356LFX.bed \
    ${wd}/data/NA12878.bwa.bam
