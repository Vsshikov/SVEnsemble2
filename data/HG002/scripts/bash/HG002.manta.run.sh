#!/bin/bash
#SBATCH --job-name=HG002.manta
#SBATCH --mem=10G
#SBATCH --cpus-per-task=10
#SBATCH --time=02:00:00

module load samtools micromamba

micromamba run -n manta_py2 python /home/users/allstaff/shikov.v/projects/Papenfuss_SVEnsemble2/data/HG002/sv_calling/manta/runWorkflow.py
