#! /bin/bash

#SBATCH --job-name=sv_shap.gpu
#SBATCH --output=sv_shap.gpu.out
#SBATCH --error=sv_shap.gpu.err
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:P100:1
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=80G

module load micromamba CUDA/11.8

export CUDA_PATH="/stornext/System/data/software/rhel/9/base/nvidia/CUDA/11.8"

training_set_name=$1
sample_name=$2
model_name=$3

micromamba run -n SVEnsemble2 python /home/users/allstaff/shikov.v/projects/Papenfuss_SVEnsemble2/svensemble/sv_shap.py --training_set_name $training_set_name --sample_name $sample_name --model_name $model_name --use_gpu