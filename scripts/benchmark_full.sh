#!/bin/bash

# Default values for arguments
SAMPLE_NAME=""
MODEL_NAME=""
SVTYPES=""
SVLEN_LOWER=""
SVLEN_UPPER=""
TRAINING_SETS=""
VAL_MATCH_MARGIN=500
SKIP_FILTERING=false
ADDITIONAL_COLS=""
PERFORM_STEPS=""
ADDITIONAL_CALLSETS=""
VERBOSE=false

# Parse named arguments
while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --sample_name)
            SAMPLE_NAME="$2"
            shift
            ;;
        --model_name)
            MODEL_NAME="$2"
            shift
            ;;
        --svtypes)
            SVTYPES="$2"
            shift
            ;;
        --svlen_lower)
            SVLEN_LOWER="$2"
            shift
            ;;
        --svlen_upper)
            SVLEN_UPPER="$2"
            shift
            ;;
        --training_sets)
            TRAINING_SETS="$2"
            shift
            ;;
        --val_match_margin)
            VAL_MATCH_MARGIN="$2"
            shift
            ;;
        --skip_filtering)
            SKIP_FILTERING=true
            ;;
        --additional_cols)
            ADDITIONAL_COLS="$2"
            shift
            ;;
        --additional_callsets)
            ADDITIONAL_CALLSETS="$2"
            shift
            ;;
        -v|--verbose)
            VERBOSE=true
            ;;
        *)
            echo "Unknown parameter passed: $1"
            exit 1
            ;;
    esac
    shift
done

module purge
module load micromamba

# submit preprocess jobs
# Construct the command
# Only add svlen arguments, svtype, training_sets, val_match_margin, skip_filtering, additional_cols, additional_callsets, verbose if they are not empty
command="micromamba run -n SVEnsemble2 sbatch --parsable /home/users/allstaff/shikov.v/projects/Papenfuss_SVEnsemble2/svensemble/benchmark.py --sample_name $SAMPLE_NAME --model_name $MODEL_NAME --perform_steps preprocess"
if [ -n "$SVTYPES" ]; then
    command+=" --svtypes $SVTYPES"
fi
if [ -n "$SVLEN_LOWER" ]; then
    command+=" --svlen_lower $SVLEN_LOWER"
fi
if [ -n "$SVLEN_UPPER" ]; then
    command+=" --svlen_upper $SVLEN_UPPER"
fi
if [ -n "$TRAINING_SETS" ]; then
    command+=" --training_sets $TRAINING_SETS"
fi
if [ -n "$VAL_MATCH_MARGIN" ]; then
    command+=" --val_match_margin $VAL_MATCH_MARGIN"
fi
if [ -n "$SKIP_FILTERING" = true ]; then
    command+=" --skip_filtering"
fi
if [ -n "$ADDITIONAL_COLS" ]; then
    command+=" --additional_cols $ADDITIONAL_COLS"
fi
if [ -n "$ADDITIONAL_CALLSETS" ]; then
    command+=" --additional_callsets $ADDITIONAL_CALLSETS"
fi
if [ -n "$VERBOSE" ]; then
    command+=" --verbose"
fi
preprocess_job_id=$($command)
echo "Submitted job for $SAMPLE_NAME with model $MODEL_NAME. Job ID: $preprocess_job_id"
echo "Command: $command"


# submit benchmark jobs
# Construct the command
# Prior to this fetch the corresponding preprocess job id
command="micromamba run -n SVEnsemble2 sbatch --parsable --dependency=afterok:$preprocess_job_id /home/users/allstaff/shikov.v/projects/Papenfuss_SVEnsemble2/svensemble/benchmark.py --sample_name $SAMPLE_NAME --model_name $MODEL_NAME --perform_steps benchmark"
if [ -n "$SVLEN_LOWER" ]; then
    command+=" --svlen_lower $SVLEN_LOWER"
fi
if [ -n "$SVLEN_UPPER" ]; then
    command+=" --svlen_upper $SVLEN_UPPER"
fi
if [ -n "$TRAINING_SETS" ]; then
    command+=" --training_sets $TRAINING_SETS"
fi
if [ -n "$VAL_MATCH_MARGIN" ]; then
    command+=" --val_match_margin $VAL_MATCH_MARGIN"
fi
if [ -n "$SKIP_FILTERING" = true ]; then
    command+=" --skip_filtering"
fi
if [ -n "$ADDITIONAL_COLS" ]; then
    command+=" --additional_cols $ADDITIONAL_COLS"
fi
if [ -n "$ADDITIONAL_CALLSETS" ]; then
    command+=" --additional_callsets $ADDITIONAL_CALLSETS"
fi
if [ -n "$VERBOSE" ]; then
    command+=" --verbose"
fi
if [ -n "$SVTYPES" ]; then
    for svtype in $SVTYPES; do
        command_final=$command" --svtypes $svtype"
        benchmark_job_id=$($command_final)
        echo "Submitted job for $SAMPLE_NAME with model $MODEL_NAME and svtypes $svtype. Job ID: $benchmark_job_id"
        echo "Command: $command_final"
    done
fi