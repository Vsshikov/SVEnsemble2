#! /bin/bash

#SBATCH --job-name=sv_shap.nogpu
#SBATCH --output=sv_shap.nogpu.out
#SBATCH --error=sv_shap.nogpu.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G

module load micromamba CUDA/11.8

export CUDA_PATH="/stornext/System/data/software/rhel/9/base/nvidia/CUDA/11.8"

TRAINING_SET_NAME=$1
SAMPLE_NAME=$2
MODEL_NAME=$3

micromamba run -n SVEnsemble2 python /home/users/allstaff/shikov.v/projects/Papenfuss_SVEnsemble2/svensemble/sv_shap.py --training_set_name $TRAINING_SET_NAME --sample_name $SAMPLE_NAME --model_name $MODEL_NAME
