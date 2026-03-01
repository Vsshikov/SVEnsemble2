#!/bin/bash

# SAMPLE_NAME=$1
# REF_FASTA=$2
# READS1_PATH=$3
# READS2_PATH=$4
# RERUN=$5

SOURCE_FILE=$1
PROJ_DIR="/home/users/allstaff/shikov.v/projects/Papenfuss_SVEnsemble2"

source $SOURCE_FILE

SAMPLE_DIR="$PROJ_DIR/data/$SAMPLE_NAME"

# Check if sv_calling and aligned dirs exist
if [ -d "$SAMPLE_DIR/sv_calling" ] && [ -d "$SAMPLE_DIR/aligned" ]; then
else
    echo "sv_calling and aligned dirs do not exist, creating"
    mkdir -p $SAMPLE_DIR/sv_calling
    mkdir -p $SAMPLE_DIR/aligned
fi

# ========== ========== SHORT READS ========== ==========

# ========== GENERATING READS FROM ALIGNMENT FILES ==========
SHORT_READS=false
# BAM/SAM -> FQs
if [[ -n "$SAM_INPUT" && ( "$RERUN" == "true" || ! -f "$SAMPLE_DIR/preprocessed/${SAMPLE_NAME}.R1.fq.gz" || ! -f "$SAMPLE_DIR/preprocessed/${SAMPLE_NAME}.R2.fq.gz" ) ]]; then
    preprocess_sam_id=$(sbatch --parsable --chdir=$SAMPLE_DIR $SAMPLE_DIR/scripts/bazam_sam.sh $SAMPLE_NAME $SAM_INPUT)
    # bazam_sam.sh will write the output file names to $SAMPLE_DIR/preprocessed/output_files.txt
    # Wait until the file appears and read the file names into READS1_PATH and READS2_PATH
    while [ ! -f "$SAMPLE_DIR/preprocessed/output_files.txt" ]; do
        sleep 500ms
    done
    READS1_PATH=$(cat $SAMPLE_DIR/preprocessed/output_files.txt | grep "R1.fq.gz" | awk '{print $1}')
    READS2_PATH=$(cat $SAMPLE_DIR/preprocessed/output_files.txt | grep "R2.fq.gz" | awk '{print $1}')
    SHORT_READS=true
fi

# CRAM -> FQs
if [[ -n "$CRAM_INPUT" && ( "$RERUN" == "true" || ! -f "$SAMPLE_DIR/preprocessed/${SAMPLE_NAME}.R1.fq.gz" || ! -f "$SAMPLE_DIR/preprocessed/${SAMPLE_NAME}.R2.fq.gz" ) ]]; then
    preprocess_cram_id=$(sbatch --parsable --chdir=$SAMPLE_DIR $SAMPLE_DIR/scripts/bazam_cram.sh $SAMPLE_NAME $CRAM_INPUT $CRAM_FASTA)
    # bazam_cram.sh will write the output file names to $SAMPLE_DIR/preprocessed/output_files.txt
    # Wait until the file appears and read the file names into READS1_PATH and READS2_PATH
    while [ ! -f "$SAMPLE_DIR/preprocessed/output_files.txt" ]; do
        sleep 500ms
    done
    READS1_PATH=$(cat $SAMPLE_DIR/preprocessed/output_files.txt | grep "R1.fq.gz" | awk '{print $1}')
    READS2_PATH=$(cat $SAMPLE_DIR/preprocessed/output_files.txt | grep "R2.fq.gz" | awk '{print $1}')
    SHORT_READS=true
fi

# ========== SHORT READ ALIGNMENT ==========
PREPROCESS_DEPENDENCY=""
if [[ ! -z "$preprocess_sam_id" || ! -z "$preprocess_cram_id" ]]; then
    PREPROCESS_DEPENDENCY="--dependency=afterok:$preprocess_sam_id:$preprocess_cram_id"
fi

if [[ "$READS1_PATH" != "" && "$READS2_PATH" != "" ]]; then
    SHORT_READS=true
fi

if [[ "$SHORT_READS" == "true" && ("$RERUN" == "true" || ! -f "$SAMPLE_DIR/aligned/${SAMPLE_NAME}.minimap.bam") ]]; then
    minimap_id=$(sbatch --parsable --chdir=$SAMPLE_DIR $PREPROCESS_DEPENDENCY $SAMPLE_DIR/scripts/minimap.sh $SOURCE_FILE)
    MINIMAP_DEPENDENCY="--dependency=afterok:$minimap_id"; else
    MINIMAP_DEPENDENCY="";
fi

# Run manta
if [[ "$RERUN" == "true" || ! -f "$SAMPLE_DIR/sv_calling/manta/results/variants/candidateSV.vcf.gz" ]]; then
    manta_id=$(sbatch --parsable --chdir=$SAMPLE_DIR $MINIMAP_DEPENDENCY $SAMPLE_DIR/scripts/manta.sh $SOURCE_FILE)
fi

if [[ "$RERUN" == "true" || ! -f "$SAMPLE_DIR/sv_calling/gridss/out/${SAMPLE_NAME}.gridss.vcf" ]]; then
    gridss_id=$(sbatch --parsable --chdir=$SAMPLE_DIR $MINIMAP_DEPENDENCY $SAMPLE_DIR/scripts/gridss.sh $SOURCE_FILE)
fi

if [[ "$RERUN" == "true" || ! -f "$SAMPLE_DIR/sv_calling/delly/out/${SAMPLE_NAME}.delly.bcf" ]]; then
    delly_id=$(sbatch --parsable --chdir=$SAMPLE_DIR $MINIMAP_DEPENDENCY $SAMPLE_DIR/scripts/delly.sh $SOURCE_FILE)
fi

# ========== ========== LONG READS ========== ==========

# ========== LONG READ ALIGNMENT ==========
if [[ "$RERUN" == "true" || ! -f "$SAMPLE_DIR/aligned_long/${SAMPLE_NAME}.long.minimap.bam" ]]; then
    minimap_long_id=$(sbatch --parsable --chdir=$SAMPLE_DIR $SAMPLE_DIR/scripts/minimap_long.sh $SOURCE_FILE)
    MINIMAP_LONG_DEPENDENCY="--dependency=afterok:$minimap_long_id"; else
    MINIMAP_LONG_DEPENDENCY="";
fi

# ========== LONG READ CALLERS ==========
if [[ "$RERUN" == "true" || ! -f "$SAMPLE_DIR/sv_calling/sniffles/out/${SAMPLE_NAME}.sniffles.vcf" ]]; then
    sniffles_id=$(sbatch --parsable --chdir=$SAMPLE_DIR $MINIMAP_LONG_DEPENDENCY $SAMPLE_DIR/scripts/sniffles.sh $SOURCE_FILE)
fi
# ========== ========== POST-CALLING PROCESSING ========== ==========

# ========== ENSEMBLE/CONSENSUS CALLERS ==========

if [[ ! -z "$manta_id" && ! -z "$gridss_id" && ! -z "$delly_id" ]]; then
    SURVIVOR_DEPENDENCY="--dependency=afterok:$manta_id:$gridss_id:$delly_id"
else
    SURVIVOR_DEPENDENCY="";
fi

if [[ "$RERUN" == "true" || ! -f "$SAMPLE_DIR/sv_calling/survivor/out/${SAMPLE_NAME}.survivor.vcf" ]]; then
    survivor_id=$(sbatch --parsable --chdir=$SAMPLE_DIR $SURVIVOR_DEPENDENCY $SAMPLE_DIR/scripts/survivor.sh $SOURCE_FILE)
fi

# ========== ========== SVENSEMBLE2 PIPELINE ========== ==========

module load micromamba

# Collect all caller job IDs into a single dependency string
_sve_ids=""
for _id in "$manta_id" "$gridss_id" "$delly_id" "$sniffles_id"; do
    [[ -n "$_id" ]] && _sve_ids="${_sve_ids}:${_id}"
done
[[ -n "$_sve_ids" ]] && SVE_CALLER_DEPENDENCY="--dependency=afterok${_sve_ids}" || SVE_CALLER_DEPENDENCY=""

# Build step flags from config booleans
SVE_STEP_FLAGS=""
[[ "$SVE_STEP_PREPROCESS" == "true" ]] && SVE_STEP_FLAGS+=" --preprocess"
[[ "$SVE_STEP_MATCH"      == "true" ]] && SVE_STEP_FLAGS+=" --match"
[[ "$SVE_STEP_TRAIN"      == "true" ]] && SVE_STEP_FLAGS+=" --train"
[[ "$SVE_STEP_PREDICT"    == "true" ]] && SVE_STEP_FLAGS+=" --predict"
[[ "$SVE_VERBOSE"         == "true" ]] && SVE_STEP_FLAGS+=" --verbose"

# Submit one job per model × training_set combination
if [[ "$RERUN" == "true" || "$SVENSEMBLE2_RERUN" == "true" ]]; then
    for SVE_MODEL in "${SVE_MODELS[@]}"; do
        for SVE_TRAINING_SET in "${SVE_TRAINING_SETS[@]}"; do
            micromamba run -n SVEnsemble2 sbatch --parsable --chdir=$SAMPLE_DIR $SVE_CALLER_DEPENDENCY \
                $PROJ_DIR/svensemble/pipeline.py \
                --sample_name $SAMPLE_NAME \
                --ground_truth_sample_name $SVE_GROUND_TRUTH_SAMPLE_NAME \
                --ref_path $REF_FASTA \
                --model_name $SVE_MODEL \
                --training_set $SVE_TRAINING_SET \
                $SVE_STEP_FLAGS \
                --match_max_dist $SVE_MATCH_MAX_DIST \
                --match_insert_size_threshold $SVE_MATCH_INSERT_SIZE_THRESHOLD
        done
    done
fi