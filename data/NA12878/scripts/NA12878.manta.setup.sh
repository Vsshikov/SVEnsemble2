#!/bin/bash

module load samtools micromamba

MANTA_DIR="/home/users/allstaff/shikov.v/projects/Papenfuss_SVEnsemble2/callers/manta-1.6.0.centos6_x86_64"

WORKDIR="/home/users/allstaff/shikov.v/projects/Papenfuss_SVEnsemble2/data/NA12878/sv_calling/manta"
HG38_FASTA="/home/users/allstaff/shikov.v/projects/Papenfuss_SVEnsemble2/data/hg38/downloads/hg38.no_decoys.fa"
INPUT_BAM="/home/users/allstaff/shikov.v/projects/Papenfuss_SVEnsemble2/data/NA12878/aligned/NA12878.minimap.bam"

# Create output directory if it doesn't exist
mkdir -p $WORKDIR

# Index BAM file if not already indexed
if [ ! -f $INPUT_BAM.bai ]; then
    samtools index $INPUT_BAM
fi

# Run configManta
micromamba run -n manta_py2 $MANTA_DIR/bin/configManta.py --bam $INPUT_BAM --referenceFasta $HG38_FASTA --runDir $WORKDIR
