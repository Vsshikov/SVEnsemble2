#!/bin/bash

#SBATCH --job-name=truth_set_gen
#SBATCH --mem=100GB
#SBATCH --cpus-per-task=2
#SBATCH --time=24:00:00

script_path=$"/home/users/allstaff/shikov.v/projects/SVensemble2/truth_set_gen/scripts/truth_set_gen.py"

conda run -n SVEnsemble2 python $script_path
