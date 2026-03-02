#!/bin/bash

# SAMPLE_NAME=$1
# REF_FASTA=$2
# READS1_PATH=$3
# READS2_PATH=$4
# RERUN=$5

SOURCE_FILE=$(realpath "$1")
PROJ_DIR="/home/users/allstaff/shikov.v/projects/Papenfuss_SVEnsemble2"

source $SOURCE_FILE


SAMPLE_DIR="$PROJ_DIR/data/$SAMPLE_NAME"

# Initialize the logs directory for this run
RUN_ID=$(uuidgen)
LOG_DIR="$SAMPLE_DIR/logs/run/$RUN_ID"
mkdir -p $LOG_DIR
mkdir -p $LOG_DIR/{bazam_sam,bazam_cram,minimap,minimap_long,manta,gridss,delly,sniffles,survivor,svensemble2,benchmark}
cat $SOURCE_FILE > $LOG_DIR/master_args.sh

touch $LOG_DIR/run.log
echo "Run ID $RUN_ID"
echo "Run log: $LOG_DIR/run.log"
echo "$(date): Run ID: $RUN_ID" >> $LOG_DIR/run.log

# Check if sv_calling and aligned dirs exist
mkdir -p $SAMPLE_DIR/sv_calling
mkdir -p $SAMPLE_DIR/aligned

# When skipping realignment, SAM_INPUT (if provided) is treated as an already-aligned BAM.
# Symlink it to the pipeline's expected path so all downstream callers pick it up transparently.
# If SAM_INPUT is not provided, the aligned BAM must already exist at the expected path.
if [[ "$SKIP_REALIGNMENT" == "true" ]]; then
    echo "$(date): SKIP_REALIGNMENT=true" >> $LOG_DIR/run.log
    PIPELINE_BAM="$SAMPLE_DIR/aligned/${SAMPLE_NAME}.minimap.bam"
    echo "$(date): PIPELINE_BAM: $PIPELINE_BAM" >> $LOG_DIR/run.log
    if [[ -n "$SAM_INPUT" ]]; then
        echo "$(date): SAM_INPUT: $SAM_INPUT" >> $LOG_DIR/run.log
        if [[ ! -f "$PIPELINE_BAM" ]]; then
            echo "$(date): Linking $SAM_INPUT to $PIPELINE_BAM" >> $LOG_DIR/run.log
            ln -s "$SAM_INPUT" "$PIPELINE_BAM"
        fi
    elif [[ ! -f "$PIPELINE_BAM" && \
            ! -f "$SAMPLE_DIR/aligned_long/${SAMPLE_NAME}.long.minimap.bam" ]]; then
        echo "$(date): Error: SKIP_REALIGNMENT=true but SAM_INPUT is not set and no aligned BAM files found:" >> $LOG_DIR/run.log
        echo "$(date):   $PIPELINE_BAM" >> $LOG_DIR/run.log
        echo "$(date):   $SAMPLE_DIR/aligned_long/${SAMPLE_NAME}.long.minimap.bam" >> $LOG_DIR/run.log
        exit 1
    fi
fi

# ========== ========== SHORT READS ========== ==========

MINIMAP_DEPENDENCY=""

if [[ "$SKIP_REALIGNMENT" != "true" ]]; then

    # ========== GENERATING READS FROM ALIGNMENT FILES ==========
    SHORT_READS=false

    # BAM/SAM -> FQs
    if [[ -n "$SAM_INPUT" && ( "$RERUN" == "true" || ! -f "$SAMPLE_DIR/preprocessed/${SAMPLE_NAME}.R1.fq.gz" || ! -f "$SAMPLE_DIR/preprocessed/${SAMPLE_NAME}.R2.fq.gz" ) ]]; then
        echo "$(date): Running bazam_sam.sh" >> $LOG_DIR/run.log
        preprocess_sam_id=$(sbatch --parsable --chdir=$SAMPLE_DIR \
            --output=$LOG_DIR/bazam_sam/%j.out --error=$LOG_DIR/bazam_sam/%j.err \
            $SAMPLE_DIR/scripts/bazam_sam.sh $SAMPLE_NAME $SAM_INPUT)
        READS1_PATH=$SAMPLE_DIR/preprocessed/${SAMPLE_NAME}.R1.fq.gz
        READS2_PATH=$SAMPLE_DIR/preprocessed/${SAMPLE_NAME}.R2.fq.gz
        SHORT_READS=true
        echo "$(date): Submitted bazam_sam.sh job with ID: $preprocess_sam_id" >> $LOG_DIR/run.log
        echo "$(date): READS1_PATH: $READS1_PATH" >> $LOG_DIR/run.log
        echo "$(date): READS2_PATH: $READS2_PATH" >> $LOG_DIR/run.log
    fi

    # CRAM -> FQs
    if [[ -n "$CRAM_INPUT" && ( "$RERUN" == "true" || ! -f "$SAMPLE_DIR/preprocessed/${SAMPLE_NAME}.R1.fq.gz" || ! -f "$SAMPLE_DIR/preprocessed/${SAMPLE_NAME}.R2.fq.gz" ) ]]; then
        echo "$(date): Running bazam_cram.sh" >> $LOG_DIR/run.log
        preprocess_cram_id=$(sbatch --parsable --chdir=$SAMPLE_DIR \
            --output=$LOG_DIR/bazam_cram/%j.out --error=$LOG_DIR/bazam_cram/%j.err \
            $SAMPLE_DIR/scripts/bazam_cram.sh $SAMPLE_NAME $CRAM_INPUT $CRAM_FASTA)
        READS1_PATH=$SAMPLE_DIR/preprocessed/${SAMPLE_NAME}.R1.fq.gz
        READS2_PATH=$SAMPLE_DIR/preprocessed/${SAMPLE_NAME}.R2.fq.gz
        SHORT_READS=true
        echo "$(date): Submitted bazam_cram.sh job with ID: $preprocess_cram_id" >> $LOG_DIR/run.log
        echo "$(date): READS1_PATH: $READS1_PATH" >> $LOG_DIR/run.log
        echo "$(date): READS2_PATH: $READS2_PATH" >> $LOG_DIR/run.log
    fi

    # ========== SHORT READ ALIGNMENT ==========
    _preprocess_ids=""
    [[ -n "$preprocess_sam_id"  ]] && _preprocess_ids="${_preprocess_ids}:${preprocess_sam_id}"
    [[ -n "$preprocess_cram_id" ]] && _preprocess_ids="${_preprocess_ids}:${preprocess_cram_id}"
    [[ -n "$_preprocess_ids" ]] && PREPROCESS_DEPENDENCY="--dependency=afterok${_preprocess_ids}" \
                                 || PREPROCESS_DEPENDENCY=""

    if [[ "$READS1_PATH" != "" && "$READS2_PATH" != "" ]]; then
        SHORT_READS=true
    fi

    if [[ "$SHORT_READS" == "true" && ("$RERUN" == "true" || ! -f "$SAMPLE_DIR/aligned/${SAMPLE_NAME}.minimap.bam") ]]; then
        minimap_id=$(sbatch --parsable --chdir=$SAMPLE_DIR $PREPROCESS_DEPENDENCY \
            --output=$LOG_DIR/minimap/%j.out --error=$LOG_DIR/minimap/%j.err \
            $SAMPLE_DIR/scripts/minimap.sh $SOURCE_FILE)
        [[ -n "$minimap_id" ]] && MINIMAP_DEPENDENCY="--dependency=afterok:$minimap_id"
        echo "$(date): Submitted minimap.sh job with ID: $minimap_id" >> $LOG_DIR/run.log
    fi
fi  # SKIP_REALIGNMENT

# Run manta
if [[ "$RERUN" == "true" || ! -f "$SAMPLE_DIR/sv_calling/manta/results/variants/candidateSV.vcf.gz" ]]; then
    manta_id=$(sbatch --parsable --chdir=$SAMPLE_DIR $MINIMAP_DEPENDENCY \
        --output=$LOG_DIR/manta/%j.out --error=$LOG_DIR/manta/%j.err \
        $SAMPLE_DIR/scripts/manta.sh $SOURCE_FILE)
    echo "$(date): Submitted manta.sh job with ID: $manta_id" >> $LOG_DIR/run.log
fi

if [[ "$RERUN" == "true" || ! -f "$SAMPLE_DIR/sv_calling/gridss/out/${SAMPLE_NAME}.gridss.vcf" ]]; then
    gridss_id=$(sbatch --parsable --chdir=$SAMPLE_DIR $MINIMAP_DEPENDENCY \
        --output=$LOG_DIR/gridss/%j.out --error=$LOG_DIR/gridss/%j.err \
        $SAMPLE_DIR/scripts/gridss.sh $SOURCE_FILE)
    echo "$(date): Submitted gridss.sh job with ID: $gridss_id" >> $LOG_DIR/run.log
fi

if [[ "$RERUN" == "true" || ! -f "$SAMPLE_DIR/sv_calling/delly/out/${SAMPLE_NAME}.delly.bcf" ]]; then
    delly_id=$(sbatch --parsable --chdir=$SAMPLE_DIR $MINIMAP_DEPENDENCY \
        --output=$LOG_DIR/delly/%j.out --error=$LOG_DIR/delly/%j.err \
        $SAMPLE_DIR/scripts/delly.sh $SOURCE_FILE)
    echo "$(date): Submitted delly.sh job with ID: $delly_id" >> $LOG_DIR/run.log
fi

# ========== ========== LONG READS ========== ==========

# ========== LONG READ ALIGNMENT ==========
if [[ ! -z "$READS_LONG_PATHS" && ("$RERUN" == "true" || ! -f "$SAMPLE_DIR/aligned_long/${SAMPLE_NAME}.long.minimap.bam" )]]; then
    minimap_long_id=$(sbatch --parsable --chdir=$SAMPLE_DIR \
        --output=$LOG_DIR/minimap_long/%j.out --error=$LOG_DIR/minimap_long/%j.err \
        $SAMPLE_DIR/scripts/minimap_long.sh $SOURCE_FILE)
    echo "$(date): Submitted minimap_long.sh job with ID: $minimap_long_id" >> $LOG_DIR/run.log
    MINIMAP_LONG_DEPENDENCY="--dependency=afterok:$minimap_long_id"; else
    MINIMAP_LONG_DEPENDENCY="";
fi

# ========== LONG READ CALLERS ==========
if [[ ! -z "$READS_LONG_PATHS" && ("$RERUN" == "true" || ! -f "$SAMPLE_DIR/sv_calling/sniffles/out/${SAMPLE_NAME}.long.sniffles.vcf" )]]; then
    sniffles_id=$(sbatch --parsable --chdir=$SAMPLE_DIR $MINIMAP_LONG_DEPENDENCY \
        --output=$LOG_DIR/sniffles/%j.out --error=$LOG_DIR/sniffles/%j.err \
        $SAMPLE_DIR/scripts/sniffles.sh $SOURCE_FILE)
    echo "$(date): Submitted sniffles.sh job with ID: $sniffles_id" >> $LOG_DIR/run.log
fi
# ========== ========== POST-CALLING PROCESSING ========== ==========

# ========== ENSEMBLE/CONSENSUS CALLERS ==========

if [[ ! -z "$manta_id" && ! -z "$gridss_id" && ! -z "$delly_id" ]]; then
    SURVIVOR_DEPENDENCY="--dependency=afterok:$manta_id:$gridss_id:$delly_id"
else
    SURVIVOR_DEPENDENCY="";
fi

if [[ "$RERUN" == "true" || ! -f "$SAMPLE_DIR/sv_calling/survivor/${SAMPLE_NAME}.survivor.vcf" ]]; then
    survivor_id=$(sbatch --parsable --chdir=$SAMPLE_DIR $SURVIVOR_DEPENDENCY \
        --output=$LOG_DIR/survivor/%j.out --error=$LOG_DIR/survivor/%j.err \
        $SAMPLE_DIR/scripts/survivor.sh $SOURCE_FILE)
    echo "$(date): Submitted survivor.sh job with ID: $survivor_id" >> $LOG_DIR/run.log
fi

# ========== ========== SVENSEMBLE2 PIPELINE ========== ==========

# Collect all caller job IDs into a single dependency string
_sve_ids=""
for _id in "$manta_id" "$gridss_id" "$delly_id" "$sniffles_id"; do
    [[ -n "$_id" ]] && _sve_ids="${_sve_ids}:${_id}"
done
echo "$(date): Caller jobs submitted for $SAMPLE_NAME: $_sve_ids" >> $LOG_DIR/run.log
[[ -n "$_sve_ids" ]] && SVE_CALLER_DEPENDENCY="--dependency=afterok${_sve_ids}" || SVE_CALLER_DEPENDENCY=""

# Submit one SLURM array job covering all model × training_set combinations.
# Each task reads its parameters from a TSV task file via SLURM_ARRAY_TASK_ID.
_sve_job_ids=()
if [[ "$RERUN" == "true" || "$SVENSEMBLE2_RERUN" == "true" ]]; then
    SVE_TASK_FILE="$LOG_DIR/svensemble2/tasks.tsv"
    for SVE_MODEL in "${SVE_MODELS[@]}"; do
        for SVE_TRAINING_SET in "${SVE_TRAINING_SETS[@]}"; do
            printf '%s\t%s\n' "$SVE_MODEL" "$SVE_TRAINING_SET"
        done
    done > "$SVE_TASK_FILE"
    N_SVE_TASKS=$(wc -l < "$SVE_TASK_FILE")

    echo "$(date): SVEnsemble2 task file (${N_SVE_TASKS} tasks): $SVE_TASK_FILE" >> $LOG_DIR/run.log
    cat "$SVE_TASK_FILE" >> $LOG_DIR/run.log

    sve_array_id=$(sbatch --parsable --chdir="$SAMPLE_DIR" $SVE_CALLER_DEPENDENCY \
        --array=0-$((N_SVE_TASKS - 1)) \
        --output="$LOG_DIR/svensemble2/%A_%a.out" \
        --error="$LOG_DIR/svensemble2/%A_%a.err" \
        "$SAMPLE_DIR/scripts/svensemble_array.sh" "$SOURCE_FILE" "$SVE_TASK_FILE")
    echo "$(date): Submitted SVEnsemble2 array job with ID: $sve_array_id (${N_SVE_TASKS} tasks)" >> $LOG_DIR/run.log
    [[ -n "$sve_array_id" ]] && _sve_job_ids+=("$sve_array_id")
fi

# ========== ========== BENCHMARKING ========== ==========

if [[ "$BENCHMARK" == "true" ]]; then
    # Build dependency on all SVEnsemble2 array jobs (afterok waits for every task)
    _bench_dep_ids=""
    for _id in "${_sve_job_ids[@]}"; do
        [[ -n "$_id" ]] && _bench_dep_ids="${_bench_dep_ids}:${_id}"
    done
    [[ -n "$_bench_dep_ids" ]] && BENCH_DEPENDENCY="--dependency=afterany${_bench_dep_ids}" || BENCH_DEPENDENCY=""

    # Normalise SVE_BENCH_SVTYPES and SVE_BENCH_SVLEN_LOWER to arrays.
    # An empty array means "no filter" — one task is still written without that flag.
    _bench_svtypes_list=("${SVE_BENCH_SVTYPES[@]}")
    [[ ${#_bench_svtypes_list[@]} -eq 0 ]] && _bench_svtypes_list=("")

    _bench_svlen_lower_list=("${SVE_BENCH_SVLEN_LOWER[@]}")
    [[ ${#_bench_svlen_lower_list[@]} -eq 0 ]] && _bench_svlen_lower_list=("")

    # Submit one SLURM array job covering all model × training_set × svtypes × svlen_lower combinations.
    # Each task reads its parameters from a TSV task file via SLURM_ARRAY_TASK_ID.
    BENCH_TASK_FILE="$LOG_DIR/benchmark/tasks.tsv"
    for SVE_MODEL in "${SVE_MODELS[@]}"; do
        for SVE_TRAINING_SET in "${SVE_TRAINING_SETS[@]}"; do
            for _svtypes in "${_bench_svtypes_list[@]}"; do
                for _svlen_lower in "${_bench_svlen_lower_list[@]}"; do
                    printf '%s\t%s\t%s\t%s\n' \
                        "$SVE_MODEL" "$SVE_TRAINING_SET" "${_svtypes:-.}" "${_svlen_lower:-.}"
                done
            done
        done
    done > "$BENCH_TASK_FILE"
    N_BENCH_TASKS=$(wc -l < "$BENCH_TASK_FILE")

    echo "$(date): Benchmark task file (${N_BENCH_TASKS} tasks): $BENCH_TASK_FILE" >> $LOG_DIR/run.log
    cat "$BENCH_TASK_FILE" >> $LOG_DIR/run.log

    bench_array_id=$(sbatch --parsable --chdir="$SAMPLE_DIR" $BENCH_DEPENDENCY \
        --array=0-$((N_BENCH_TASKS - 1)) \
        --output="$LOG_DIR/benchmark/%A_%a.out" \
        --error="$LOG_DIR/benchmark/%A_%a.err" \
        "$SAMPLE_DIR/scripts/benchmark_array.sh" "$SOURCE_FILE" "$BENCH_TASK_FILE")
    echo "$(date): Submitted benchmark array job with ID: $bench_array_id (${N_BENCH_TASKS} tasks)" >> $LOG_DIR/run.log
fi

