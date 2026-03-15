#!/bin/bash

#SBATCH --job-name=gridss
#SBATCH --output=logs/standalone/%x_%j.out
#SBATCH --error=logs/standalone/%x_%j.err
#SBATCH --mem=31GB
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00

sif="/vast/scratch/users/shikov.v/containers/fusorsv.sif"
if [[ -s "$sif" ]]; then
    echo "Valid"
else
    mkdir -p "$(dirname "$sif")"
    apptainer pull "$sif" docker://timothyjamesbecker/fusorsv
fi