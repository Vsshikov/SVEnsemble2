#!/bin/bash

# This script is to be used from the project StoreNext directory
# It creates new StoreNext + VAST project step directories
# and puts a link to the VAST dir inside the StoreNext dir

STEP_DIR_NAME=$1
PROJ_PATH=$(pwd)
STEP_PATH=$(echo "${PROJ_PATH}/${STEP_DIR_NAME}")

VAST_STEP_DIR_NAME=$PROJ_DIR_NAME
VAST_PATH="/vast/scratch/users/shikov.v"
VAST_PROJ_NAME=$(basename $PROJ_PATH)
VAST_PROJ_PATH=$(echo $VAST_PATH/$VAST_PROJ_NAME)
VAST_STEP_PATH=$(echo "${VAST_PROJ_PATH}/${STEP_DIR_NAME}")

mkdir -p $STEP_PATH/scripts/python $STEP_PATH/scripts/bash \
    $STEP_PATH/data $STEP_PATH/out $STEP_PATH/saved

mkdir -p $VAST_STEP_PATH/scripts/python $VAST_STEP_PATH/scripts/bash \
    $VAST_STEP_PATH/data $VAST_STEP_PATH/out $VAST_STEP_PATH/logs

ln -s $VAST_STEP_PATH $STEP_PATH/vast
ln -s $VAST_STEP_PATH/logs $STEP_PATH/logs
