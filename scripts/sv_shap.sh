#! /bin/bash

workdir="/home/users/allstaff/shikov.v/projects/Papenfuss_SVEnsemble2"

training_set_name=$1
sample_name=$2
model_name=$3
use_gpu=$4

if [ $use_gpu -eq 1 ]; then
    sbatch $workdir/scripts/sv_shap.gpu.sh $training_set_name $sample_name $model_name
else
    sbatch $workdir/scripts/sv_shap.nogpu.sh $training_set_name $sample_name $model_name
fi