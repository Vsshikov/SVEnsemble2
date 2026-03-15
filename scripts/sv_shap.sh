#! /bin/bash

WORKDIR="/home/users/allstaff/shikov.v/projects/Papenfuss_SVEnsemble2"

TRAINING_SET_NAME=$1
SAMPLE_NAME=$2
MODEL_NAME=$3
USE_GPU=$4

if [ $USE_GPU -eq 1 ]; then
    sbatch $WORKDIR/scripts/sv_shap.gpu.sh $TRAINING_SET_NAME $SAMPLE_NAME $MODEL_NAME
else
    sbatch $WORKDIR/scripts/sv_shap.nogpu.sh $TRAINING_SET_NAME $SAMPLE_NAME $MODEL_NAME
fi
