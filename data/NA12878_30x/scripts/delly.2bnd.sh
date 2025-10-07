#!/bin/bash

#SBATCH --job-name=NA12878_30x.delly_2bnd
#SBATCH --mem=31GB
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00

# Load required modules
MODULES="micromamba"

# Prepare environment
module purge
module load ${MODULES}

DELLY2BND_PATH="/home/users/allstaff/shikov.v/projects/Papenfuss_SVEnsemble2/callers/delly"

DELLY_BAM="/home/users/allstaff/shikov.v/projects/Papenfuss_SVEnsemble2/data/NA12878_30x/sv_calling/delly/out/NA12878_30x.delly.bcf"
GENOME_FASTA="/home/users/allstaff/shikov.v/projects/Papenfuss_SVEnsemble2/data/hg38/downloads/hg38.no_decoys.fa"


OUTPUT_DIR="${DELLY_BAM%.bcf}.2bnd.vcf"

# Run delly2bnd.py
micromamba run -n SVEnsemble2 python "${DELLY2BND_PATH}/delly2bnd.py" -v $DELLY_BAM -r $GENOME_FASTA -o $OUTPUT_DIR
