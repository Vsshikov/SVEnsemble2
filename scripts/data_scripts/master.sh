#!/bin/bash
# Exit immediately if any command fails (-e), treat unset variables as errors
# (-u), and propagate failures through pipes (-o pipefail).
set -euo pipefail

_MASTER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_DEFAULT_CFG="${_MASTER_DIR}/config/runs/default.sh"
SOURCE_FILE=$(realpath "${1:-$_DEFAULT_CFG}")
unset _MASTER_DIR _DEFAULT_CFG

[[ -f "$SOURCE_FILE" ]] || { echo "Usage: $0 [<config/runs/*.sh>]  (no default config found at $SOURCE_FILE)" >&2; exit 1; }

source "$SOURCE_FILE"

SAMPLE_DIR="$PROJ_DIR/data/$SAMPLE_NAME"

# Initialize the logs directory for this run
RUN_ID=$(uuidgen)
LOG_DIR="$SAMPLE_DIR/logs/run/$RUN_ID"
mkdir -p "$LOG_DIR"
mkdir -p "$LOG_DIR"/{bazam_sam,bazam_cram,minimap,minimap_long,manta,gridss,delly,sniffles,survivor,svensemble2,benchmark,shap,finalizer}
cat "$SOURCE_FILE" > "$LOG_DIR/master_args.sh"

touch "$LOG_DIR/run.log"

# ─── Logging helpers ───────────────────────────────────────────────────────
# log()  writes "TIMESTAMP [LEVEL] message" to both the run log and the terminal.
# die()  logs at ERROR level then exits non-zero — use for all fatal conditions.
log()  { echo "$(date -Iseconds) [$1] ${*:2}" | tee -a "$LOG_DIR/run.log"; }
logI() { log INFO  "$@"; }
logW() { log WARN  "$@"; }
logE() { log ERROR "$@"; }
die()  { logE "$*"; exit 1; }
# ───────────────────────────────────────────────────────────────────────────

echo "Run ID:  $RUN_ID"
echo "Run log: $LOG_DIR/run.log"
logI "Run ID: $RUN_ID"

# ─── Job ID pre-initialization ─────────────────────────────────────────────
# All SLURM job ID variables must be declared empty before any conditional block.
# set -u (below) turns an unset variable reference into a hard error, so every
# variable that may or may not be assigned inside an if-block must be seeded here.
preprocess_sam_id=""
preprocess_cram_id=""
minimap_id=""
manta_id=""
gridss_id=""
delly_id=""
minimap_long_id=""
sniffles_id=""
severus_id=""
survivor_id=""
sve_array_id=""
bench_array_id=""
shap_array_id=""
short_reads=false
caller_long_dep=""
# ───────────────────────────────────────────────────────────────────────────

# ─── SLURM submission helper ───────────────────────────────────────────────
# submit_job <step_name> <dep_flag> <script> [script_args...]
#
#   step_name : short identifier used for the log subdirectory (%j.out lives
#               in $LOG_DIR/<step_name>/) and for the log message.
#   dep_flag  : a literal "--dependency=afterok:..." string, or "" for none.
#               Pass "" explicitly when there is no dependency.
#   script    : absolute path to the SLURM batch script.
#   script_args: forwarded verbatim to the batch script.
#
# Prints the assigned job ID to stdout (capture with $(...)) and writes a
# timestamped INFO line to the run log.  All log output goes to the log file
# only (not stdout) so the caller can capture the job ID cleanly.
submit_job() {
    local name="$1" dep_flag="$2"; shift 2
    local job_id
    job_id=$(sbatch --parsable \
        --chdir="$SAMPLE_DIR" \
        --output="$LOG_DIR/${name}/%j.out" \
        --error="$LOG_DIR/${name}/%j.err" \
        ${dep_flag:+"$dep_flag"} "$@")
    echo "$(date -Iseconds) [INFO] Submitted ${name}: job_id=${job_id}  script=${1}" \
        >> "$LOG_DIR/run.log"
    printf '%s' "$job_id"
}
# ───────────────────────────────────────────────────────────────────────────

# ─── Early validation ──────────────────────────────────────────────────────
# Fail before any SLURM submission if required config variables are missing
# or if expected files don't exist. Catches misconfigured master_args.sh
# immediately rather than minutes later when a queued job finally starts.
require_var()  { [[ -n "${!1}" ]] || die "Required variable '$1' is not set in $SOURCE_FILE"; }
require_file() { [[ -f "$1"    ]] || die "Required file not found: $1"; }

require_var SAMPLE_NAME
require_var PROJ_DIR
require_var REF_FASTA
require_file "$REF_FASTA"
# ───────────────────────────────────────────────────────────────────────────

# ─── Run configuration snapshot ────────────────────────────────────────────
# Printed to terminal AND appended to run.log so this run can be reproduced
# exactly by re-examining the log, even months later.
{
    printf "=== Run configuration ===\n"
    printf "SOURCE_FILE=%s\n"      "$SOURCE_FILE"
    printf "SAMPLE_NAME=%s\n"      "$SAMPLE_NAME"
    printf "REF_FASTA=%s\n"        "$REF_FASTA"
    printf "RERUN=%s  SKIP_REALIGNMENT=%s  RUN_BAZAM=%s\n" \
        "${RERUN:-}" "${SKIP_REALIGNMENT:-}" "${RUN_BAZAM:-}"
    printf "RUN_MANTA=%s  RUN_GRIDSS=%s  RUN_DELLY=%s\n" \
        "${RUN_MANTA:-}" "${RUN_GRIDSS:-}" "${RUN_DELLY:-}"
    printf "RUN_SNIFFLES=%s  RUN_SEVERUS=%s  RUN_SURVIVOR=%s\n" \
        "${RUN_SNIFFLES:-}" "${RUN_SEVERUS:-}" "${RUN_SURVIVOR:-}"
    printf "RUN_SVENSEMBLE2=%s  RUN_BENCHMARK=%s  RUN_SHAP=%s\n" \
        "${RUN_SVENSEMBLE2:-}" "${RUN_BENCHMARK:-}" "${RUN_SHAP:-}"
    printf "SVE_MODELS=(%s)\n"        "${SVE_MODELS[*]:-}"
    printf "SVE_TRAINING_SETS=(%s)\n" "${SVE_TRAINING_SETS[*]:-}"
    printf "SVE_MEM=%s\n"             "${SVE_MEM:-}"
    printf "=========================\n"
} | tee -a "$LOG_DIR/run.log"
# ───────────────────────────────────────────────────────────────────────────

# Check if sv_calling and aligned dirs exist
mkdir -p "$SAMPLE_DIR/sv_calling"
mkdir -p "$SAMPLE_DIR/aligned"

# When skipping realignment, SAM_INPUT (if provided) is treated as an already-aligned BAM.
# Symlink it to the pipeline's expected path so all downstream callers pick it up transparently.
# If SAM_INPUT is not provided, the aligned BAM must already exist at the expected path.
if [[ "$SKIP_REALIGNMENT" == "$RUN_BAZAM" ]]; then # RUN_BAZAM is the new option that will replace SKIP_REALIGNMENT in the future
    die "Conflicting options: SKIP_REALIGNMENT=true and RUN_BAZAM=true cannot both be set"
fi

if [[ "$SKIP_REALIGNMENT" == "true" || "$RUN_BAZAM" == "false" ]]; then
    logI "SKIP_REALIGNMENT=true — skipping short-read re-alignment"
    PIPELINE_BAM="$SAMPLE_DIR/aligned/${SAMPLE_NAME}.minimap.bam"
    logI "PIPELINE_BAM: $PIPELINE_BAM"
    if [[ -n "$SAM_INPUT" ]]; then
        logI "SAM_INPUT: $SAM_INPUT"
        if [[ ! -f "$PIPELINE_BAM" ]]; then
            logI "Linking $SAM_INPUT -> $PIPELINE_BAM"
            ln -s "$SAM_INPUT" "$PIPELINE_BAM"
        fi
    elif [[ ! -f "$PIPELINE_BAM" && \
            ! -f "$SAMPLE_DIR/aligned_long/${SAMPLE_NAME}.long.minimap.bam" ]]; then
        die "SKIP_REALIGNMENT=true but no aligned BAM found at:
  $PIPELINE_BAM
  $SAMPLE_DIR/aligned_long/${SAMPLE_NAME}.long.minimap.bam"
    fi
fi

# ========== ========== SHORT READS ========== ==========

minimap_dep=""
caller_dep=""

if [[ "$SKIP_REALIGNMENT" != "true" || "$RUN_BAZAM" == "true" ]]; then

    # ========== GENERATING READS FROM ALIGNMENT FILES ==========
    short_reads=false

    # BAM/SAM -> FQs
    if [[ -n "$SAM_INPUT" && ( "$RERUN" == "true" || ! -f "$SAMPLE_DIR/preprocessed/${SAMPLE_NAME}.R1.fq.gz" || ! -f "$SAMPLE_DIR/preprocessed/${SAMPLE_NAME}.R2.fq.gz" ) ]]; then
        preprocess_sam_id=$(submit_job bazam_sam "" \
            "$SAMPLE_DIR/scripts/bazam_sam.sh" "$SAMPLE_NAME" "$SAM_INPUT")
        READS1_PATH="$SAMPLE_DIR/preprocessed/${SAMPLE_NAME}.R1.fq.gz"
        READS2_PATH="$SAMPLE_DIR/preprocessed/${SAMPLE_NAME}.R2.fq.gz"
        short_reads=true
        logI "READS1_PATH: $READS1_PATH"
        logI "READS2_PATH: $READS2_PATH"
    fi

    # CRAM -> FQs
    if [[ -n "$CRAM_INPUT" && ( "$RERUN" == "true" || ! -f "$SAMPLE_DIR/preprocessed/${SAMPLE_NAME}.R1.fq.gz" || ! -f "$SAMPLE_DIR/preprocessed/${SAMPLE_NAME}.R2.fq.gz" ) ]]; then
        preprocess_cram_id=$(submit_job bazam_cram "" \
            "$SAMPLE_DIR/scripts/bazam_cram.sh" "$SAMPLE_NAME" "$CRAM_INPUT" "$CRAM_FASTA")
        READS1_PATH="$SAMPLE_DIR/preprocessed/${SAMPLE_NAME}.R1.fq.gz"
        READS2_PATH="$SAMPLE_DIR/preprocessed/${SAMPLE_NAME}.R2.fq.gz"
        short_reads=true
        logI "READS1_PATH: $READS1_PATH"
        logI "READS2_PATH: $READS2_PATH"
    fi

    # ========== SHORT READ ALIGNMENT ==========
    _minimap_dep_ids=""
    [[ -n "$preprocess_sam_id"  ]] && _minimap_dep_ids="${_minimap_dep_ids}:${preprocess_sam_id}"
    [[ -n "$preprocess_cram_id" ]] && _minimap_dep_ids="${_minimap_dep_ids}:${preprocess_cram_id}"
    [[ -n "$_minimap_dep_ids" ]] && minimap_dep="--dependency=afterok${_minimap_dep_ids}" \
                                 || minimap_dep=""
fi

if [[ "$SKIP_REALIGNMENT" != "true" || "$RUN_MINIMAP" == "true" ]]; then
    if [[ "$READS1_PATH" != "" && "$READS2_PATH" != "" ]]; then
        short_reads=true
    else
        die "READS1_PATH and READS2_PATH are not set — required for short-read alignment"
    fi
    if [[ "$short_reads" == "true" && ("$RERUN" == "true" || ! -f "$SAMPLE_DIR/aligned/${SAMPLE_NAME}.minimap.bam") ]]; then
        minimap_id=$(submit_job minimap "$minimap_dep" \
            "$SAMPLE_DIR/scripts/minimap.sh" "$SOURCE_FILE")
        [[ -n "$minimap_id" ]] && caller_dep="--dependency=afterok:$minimap_id"
    fi
fi  # SKIP_REALIGNMENT

# Run manta
if [[ "$RUN_MANTA" == "true" && ("$RERUN" == "true" || ! -f "$SAMPLE_DIR/sv_calling/manta/results/variants/candidateSV.vcf.gz" )]]; then
    manta_id=$(submit_job manta "$caller_dep" \
        "$SAMPLE_DIR/scripts/manta.sh" "$SOURCE_FILE")
fi

if [[ "$RUN_GRIDSS" == "true" && ("$RERUN" == "true" || ! -f "$SAMPLE_DIR/sv_calling/gridss/out/${SAMPLE_NAME}.gridss.vcf" )]]; then
    gridss_id=$(submit_job gridss "$caller_dep" \
        "$SAMPLE_DIR/scripts/gridss.sh" "$SOURCE_FILE")
fi

if [[ "$RUN_DELLY" == "true" && ("$RERUN" == "true" || ! -f "$SAMPLE_DIR/sv_calling/delly/out/${SAMPLE_NAME}.delly.bcf" )]]; then
    delly_id=$(submit_job delly "$caller_dep" \
        "$SAMPLE_DIR/scripts/delly.sh" "$SOURCE_FILE")
fi

# ========== ========== LONG READS ========== ==========

# ========== LONG READ ALIGNMENT ==========
if [[ "$RUN_MINIMAP_LONG" == "true" && ("$RERUN" == "true" || ! -f "$SAMPLE_DIR/aligned_long/${SAMPLE_NAME}.long.minimap.bam" )]]; then
    if [[ -z "$READS_LONG_PATHS" ]]; then
        die "READS_LONG_PATHS is not set — required for long-read alignment (RUN_MINIMAP_LONG=true)"
    fi
    minimap_long_id=$(submit_job minimap_long "" \
        "$SAMPLE_DIR/scripts/minimap_long.sh" "$SOURCE_FILE")
    caller_long_dep="--dependency=afterok:$minimap_long_id"; else
    caller_long_dep="";
fi

# ========== LONG READ CALLERS ==========
if [[ "$RUN_SNIFFLES" == "true" && ("$RERUN" == "true" || ! -f "$SAMPLE_DIR/sv_calling/sniffles/out/${SAMPLE_NAME}.long.sniffles.vcf" )]]; then
    if [[ -z "$READS_LONG_PATHS" ]]; then
        die "READS_LONG_PATHS is not set — required for sniffles (RUN_SNIFFLES=true)"
    fi
    sniffles_id=$(submit_job sniffles "$caller_long_dep" \
        "$SAMPLE_DIR/scripts/sniffles.sh" "$SOURCE_FILE")
fi

if [[ "$RUN_SEVERUS" == "true" && ("$RERUN" == "true" || ! -f "$SAMPLE_DIR/sv_calling/severus/out/${SAMPLE_NAME}.long.severus.vcf" )]]; then
    if [[ -z "$READS_LONG_PATHS" ]]; then
        die "READS_LONG_PATHS is not set — required for severus (RUN_SEVERUS=true)"
    fi
    severus_id=$(submit_job severus "$caller_long_dep" \
        "$SAMPLE_DIR/scripts/severus.sh" "$SOURCE_FILE")
fi

# ========== ========== POST-CALLING PROCESSING ========== ==========

# ========== ENSEMBLE/CONSENSUS CALLERS ==========

if [[ "$RUN_SURVIVOR" == "true" && ("$RERUN" == "true" || ! -f "$SAMPLE_DIR/sv_calling/survivor/${SAMPLE_NAME}.survivor.vcf" )]]; then
    if [[ -n "$manta_id" && -n "$gridss_id" && -n "$delly_id" ]]; then
        survivor_dep="--dependency=afterok:$manta_id:$gridss_id:$delly_id"
    else
        survivor_dep=""
    fi

    survivor_id=$(submit_job survivor "$survivor_dep" \
        "$SAMPLE_DIR/scripts/survivor.sh" "$SOURCE_FILE")
fi

# ========== ========== SVENSEMBLE2 PIPELINE ========== ==========

# Collect all caller job IDs into a single dependency string
_svensemble_dep_ids=""
for _id in "$manta_id" "$gridss_id" "$delly_id" "$sniffles_id"; do
    [[ -n "$_id" ]] && _svensemble_dep_ids="${_svensemble_dep_ids}:${_id}"
done
logI "Caller jobs submitted for $SAMPLE_NAME: ${_svensemble_dep_ids:-none}"
[[ -n "$_svensemble_dep_ids" ]] && svensemble_dep="--dependency=afterok${_svensemble_dep_ids}" || svensemble_dep=""

# Submit one SLURM array job covering all model × training_set combinations.
# Each task reads its parameters from a TSV task file via SLURM_ARRAY_TASK_ID.
_sve_job_ids=()

if [[ "$RUN_SVENSEMBLE2" == "true" ]]; then
    SVE_TASK_FILE="$LOG_DIR/svensemble2/tasks.tsv"
    for sve_model in "${SVE_MODELS[@]}"; do
        for sve_training_set in "${SVE_TRAINING_SETS[@]}"; do
            if [[ ! -f "$SAMPLE_DIR/sv_calling/svensemble2/${SAMPLE_NAME}.${sve_model}.${sve_training_set}/svensemble2_calls.csv" \
                || "$SVENSEMBLE2_RERUN" == "true" \
                || "$RERUN" == "true" ]]; then
                    printf '%s\t%s\n' "$sve_model" "$sve_training_set"
            fi
        done
    done > "$SVE_TASK_FILE"
    N_SVE_TASKS=$(wc -l < "$SVE_TASK_FILE")

    if [[ -s "$SVE_TASK_FILE" ]]; then
        logI "SVEnsemble2 task file (${N_SVE_TASKS} tasks): $SVE_TASK_FILE"
        cat "$SVE_TASK_FILE" >> "$LOG_DIR/run.log"

        if [[ -n "$SVE_MEM" ]]; then
            sve_mem_flag="--mem=$SVE_MEM"
        else
            sve_mem_flag=""
        fi

        sve_array_id=$(sbatch --parsable --chdir="$SAMPLE_DIR" $svensemble_dep $sve_mem_flag \
            --array=0-$((N_SVE_TASKS - 1)) \
            --output="$LOG_DIR/svensemble2/%A_%a.out" \
            --error="$LOG_DIR/svensemble2/%A_%a.err" \
            "$SAMPLE_DIR/scripts/svensemble_array.sh" "$SOURCE_FILE" "$SVE_TASK_FILE")
        logI "Submitted SVEnsemble2 array: job_id=$sve_array_id  tasks=$N_SVE_TASKS"
        [[ -n "$sve_array_id" ]] && _sve_job_ids+=("$sve_array_id")
    else
        logW "No SVEnsemble2 jobs to submit (task file missing or empty): $SVE_TASK_FILE"
    fi
fi

# ========== ========== BENCHMARKING ========== ==========

if [[ "$BENCHMARK" == "true" || "$RUN_BENCHMARK" == "true" ]]; then
    # Build dependency on all SVEnsemble2 array jobs (afterok waits for every task)
    _bench_dep_ids=""
    for _id in "${_sve_job_ids[@]}"; do
        [[ -n "$_id" ]] && _bench_dep_ids="${_bench_dep_ids}:${_id}"
    done
    [[ -n "$_bench_dep_ids" ]] && bench_dep="--dependency=afterany${_bench_dep_ids}" || bench_dep=""

    # Normalise SVE_BENCH_SVTYPES and SVE_BENCH_SVLEN_LOWER to arrays.
    # An empty array means "no filter" — one task is still written without that flag.
    _bench_svtypes_list=("${SVE_BENCH_SVTYPES[@]}")
    [[ ${#_bench_svtypes_list[@]} -eq 0 ]] && _bench_svtypes_list=("")

    _bench_svlen_lower_list=("${SVE_BENCH_SVLEN_LOWER[@]}")
    [[ ${#_bench_svlen_lower_list[@]} -eq 0 ]] && _bench_svlen_lower_list=("")

    # Submit one SLURM array job covering all model × training_set × svtypes × svlen_lower combinations.
    BENCH_TASK_FILE="$LOG_DIR/benchmark/tasks.tsv"
    for sve_model in "${SVE_MODELS[@]}"; do
        for sve_training_set in "${SVE_TRAINING_SETS[@]}"; do
            for _svtypes in "${_bench_svtypes_list[@]}"; do
                for _svlen_lower in "${_bench_svlen_lower_list[@]}"; do
                    printf '%s\t%s\t%s\t%s\n' \
                        "$sve_model" "$sve_training_set" "${_svtypes:-.}" "${_svlen_lower:-.}"
                done
            done
        done
    done > "$BENCH_TASK_FILE"
    N_BENCH_TASKS=$(wc -l < "$BENCH_TASK_FILE")

    logI "Benchmark task file (${N_BENCH_TASKS} tasks): $BENCH_TASK_FILE"
    cat "$BENCH_TASK_FILE" >> "$LOG_DIR/run.log"

    bench_array_id=$(sbatch --parsable --chdir="$SAMPLE_DIR" $bench_dep \
        --array=0-$((N_BENCH_TASKS - 1)) \
        --output="$LOG_DIR/benchmark/%A_%a.out" \
        --error="$LOG_DIR/benchmark/%A_%a.err" \
        "$SAMPLE_DIR/scripts/benchmark_array.sh" "$SOURCE_FILE" "$BENCH_TASK_FILE")
    logI "Submitted benchmark array: job_id=$bench_array_id  tasks=$N_BENCH_TASKS"
fi

# ========== ========== SHAP ASSESSMENT ========== ==========

if [[ "$RUN_SHAP" == "true" ]]; then
    # Build dependency: SHAP runs after benchmarking (bench_pkl source needs bench_preprocessed.pkl).
    # If benchmarking is disabled, depend on SVEnsemble2 pipeline jobs instead.
    _shap_dep_ids=""
    if [[ -n "$bench_array_id" ]]; then
        _shap_dep_ids=":${bench_array_id}"
    else
        for _id in "${_sve_job_ids[@]}"; do
            [[ -n "$_id" ]] && _shap_dep_ids="${_shap_dep_ids}:${_id}"
        done
    fi
    [[ -n "$_shap_dep_ids" ]] && shap_dep="--dependency=afterany${_shap_dep_ids}" || shap_dep=""

    # Resolve per-task GPU flag (0 or 1)
    [[ "$SVE_SHAP_USE_GPU" == "true" ]] && _gpu_flag="1" || _gpu_flag="0"

    SHAP_TASK_FILE="$LOG_DIR/shap/tasks.tsv"
    for sve_model in "${SVE_MODELS[@]}"; do
        for sve_training_set in "${SVE_TRAINING_SETS[@]}"; do
            printf '%s\t%s\t%s\t%s\t%s\n' \
                "$sve_model" "$sve_training_set" "$_gpu_flag" \
                "${SVE_SHAP_SOURCE:-.}" "${SVE_SHAP_MAX_SAMPLES:-.}"
        done
    done > "$SHAP_TASK_FILE"
    N_SHAP_TASKS=$(wc -l < "$SHAP_TASK_FILE")

    if [[ -s "$SHAP_TASK_FILE" ]]; then
        logI "SHAP task file (${N_SHAP_TASKS} tasks): $SHAP_TASK_FILE"
        cat "$SHAP_TASK_FILE" >> "$LOG_DIR/run.log"

        # Use GPU partition when requested
        _shap_sbatch_extra=""
        [[ "$SVE_SHAP_USE_GPU" == "true" ]] && _shap_sbatch_extra="--partition=gpuq --gres=gpu:P100:1"

        # shellcheck disable=SC2086
        shap_array_id=$(sbatch --parsable --chdir="$SAMPLE_DIR" $shap_dep \
            $_shap_sbatch_extra \
            --array=0-$((N_SHAP_TASKS - 1)) \
            --output="$LOG_DIR/shap/%A_%a.out" \
            --error="$LOG_DIR/shap/%A_%a.err" \
            "$SAMPLE_DIR/scripts/shap_array.sh" "$SOURCE_FILE" "$SHAP_TASK_FILE")
        logI "Submitted SHAP array: job_id=$shap_array_id  tasks=$N_SHAP_TASKS"
    else
        logW "No SHAP jobs to submit (task file missing or empty): $SHAP_TASK_FILE"
    fi
fi

# ========== ========== FINALIZER ========== ==========

_all_job_ids=()
for _id in "$preprocess_sam_id" "$preprocess_cram_id" "$minimap_id" \
           "$manta_id" "$gridss_id" "$delly_id" \
           "$minimap_long_id" "$sniffles_id" "$severus_id" \
           "$survivor_id" "$sve_array_id" "$bench_array_id" \
           "$shap_array_id"; do
    [[ -n "$_id" ]] && _all_job_ids+=("$_id")
done

if (( ${#_all_job_ids[@]} > 0 )); then
    _finalizer_dep_str=$(IFS=:; printf '%s' "${_all_job_ids[*]}")
    _finalizer_dep="--dependency=afterany:${_finalizer_dep_str}"

    # shellcheck disable=SC2086
    sbatch --parsable --chdir="$SAMPLE_DIR" $_finalizer_dep \
        --output="$LOG_DIR/finalizer/%j.out" \
        --error="$LOG_DIR/finalizer/%j.err" \
        "$PROJ_DIR/scripts/data_scripts/finalizer.sh" \
        "$LOG_DIR" "${_all_job_ids[*]}"
    logI "Submitted finalizer: depends on ${#_all_job_ids[@]} jobs (${_all_job_ids[*]})"
else
    logW "No jobs were submitted -- skipping finalizer"
fi
