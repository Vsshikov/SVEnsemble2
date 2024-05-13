#!/bin/bash
# SBATCH --job-name=env_setup.SVensemble2 
# SBATCH --time=01:00:00
# SBATCH --ntasks=1
# SBATCH --mem=8G

projdir="/stornext/Bioinf/data/bioinf-data/Papenfuss_lab_projects/SVensemble2"
modules="anaconda3/latest R/4.4.0 java/1.8.0_211 curl/7.65.0 samtools"

# echo "Loading packages: $modules"

# module purge
module load $modules

# conda init bash

# conda activate gridss

Rscript $projdir/utils/pkg_setup.SVensemble2.R

wget -P $projdir/pkg https://github.com/PapenfussLab/gridss/releases/download/v2.13.2/gridss-2.13.2.tar.gz
gunzip $projdir/pkg/gridss-2.13.2.tar.gz
mkdir -p $projdir/pkg/gridss
tar -C $projdir/pkg/gridss -xf $projdir/pkg/gridss-2.13.2.tar
