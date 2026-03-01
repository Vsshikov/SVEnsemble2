#!/bin/bash

#SBATCH --job-name=bp_match
#SBATCH --output=bp_match.out
#SBATCH --error=bp_match.err
#SBATCH --time=10:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G

module load micromamba

micromamba run -n SVEnsemble2 python /home/users/allstaff/shikov.v/projects/Papenfuss_SVEnsemble2/model/bp_match.py