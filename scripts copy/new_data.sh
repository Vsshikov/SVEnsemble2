#!/bin/bash

# Use to generate a new data directory for a new sample

SAMPLE_NAME=$1
if [[ "$SAMPLE_NAME" == "" ]]; then
    echo "Sample name is required"
    exit 1
fi

PROJ_DIR="/home/users/allstaff/shikov.v/projects/Papenfuss_SVEnsemble2"
SAMPLE_DIR="$PROJ_DIR/data/$SAMPLE_NAME"

if [[ -d "$SAMPLE_DIR" ]]; then
    echo "Sample directory already exists"
    exit 1
fi

echo "Creating new data directory for sample: $SAMPLE_NAME"

mkdir -p $SAMPLE_DIR

mkdir -p $SAMPLE_DIR/scripts
for script in $(ls $PROJ_DIR/scripts/data_scripts); do
    if [[ "$script" == "master_args.sh" ]]; then
        cp $PROJ_DIR/scripts/data_scripts/master_args.sh $SAMPLE_DIR/scripts/master_args.sh
        continue
    fi
    ln -s $PROJ_DIR/scripts/data_scripts/$script $SAMPLE_DIR/scripts/$script
done

mkdir -p $SAMPLE_DIR/envs
for yaml in $(ls $PROJ_DIR/scripts/data_scripts/envs); do
    ln -s $PROJ_DIR/scripts/data_scripts/envs/$yaml $SAMPLE_DIR/envs/$yaml
done

mkdir -p $SAMPLE_DIR/logs
mkdir -p $SAMPLE_DIR/downloads
mkdir -p $SAMPLE_DIR/preprocessed

# Set the default values in the master_args.sh file
sed -i "s/SAMPLE_NAME=/SAMPLE_NAME=$SAMPLE_NAME/" $SAMPLE_DIR/scripts/master_args.sh
sed -i "s/SAMPLE_DIR=/SAMPLE_DIR=$SAMPLE_DIR/" $SAMPLE_DIR/scripts/master_args.sh

echo "New data directory created for sample: $SAMPLE_NAME"
echo "Sample directory: $SAMPLE_DIR"
