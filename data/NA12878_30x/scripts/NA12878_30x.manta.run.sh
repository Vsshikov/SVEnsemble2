#!/bin/bash
#SBATCH --job-name=NA12878_30x_manta
#SBATCH --mem=10G
#SBATCH --cpus-per-task=10
#SBATCH --time=02:00:00

micromamba run -n manta_py2 python /home/users/allstaff/shikov.v/projects/Papenfuss_SVEnsemble2/data/NA12878_30x/sv_calling/manta/runWorkflow.py
