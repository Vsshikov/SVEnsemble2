#!/bin/bash

#SBATCH --job-name=gnomad_ground_truth
#SBATCH --output=data/gnomad/logs/%x_%j.out
#SBATCH --error=data/gnomad/logs/%x_%j.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G

# =============================================================================
# preprocess_gnomad_ground_truth.sh
#
# SLURM batch script to preprocess a gnomAD VCF into a filtered ground truth
# breakpoint set.  Paths are resolved from callset_path_configs via
# ground_truth_name.
#
# Usage (direct):
#   sbatch scripts/preprocess_gnomad_ground_truth.sh \
#       --ground_truth_name gnomad_sr_pe \
#       --filters PASS \
#       --evidence SR,PE \
#       --svtypes DEL,DUP,INV,INS \
#       --min_af 0.01 \
#       --bothsides_support_only \
#       --verbose
# =============================================================================
set -euo pipefail

# --- Logging helpers --------------------------------------------------------
LOG_PREFIX="[${SLURM_JOB_ID:-local}]"
log()  { echo "$(date -Iseconds) $LOG_PREFIX [$1] ${*:2}"; }
logI() { log INFO  "$@"; }
logE() { log ERROR "$@"; }
die()  { logE "$*"; exit 1; }
# ----------------------------------------------------------------------------

OPTS=$(getopt \
    -o 'v' \
    -l 'ground_truth_name:,filters:,evidence:,svtypes:,algorithms:,min_af:,max_af:,svlen_lower:,svlen_upper:,bothsides_support_only,no_canonical_chroms,force_rerun,verbose' \
    -n "$0" -- "$@") || die "Invalid arguments"
eval set -- "$OPTS"

GROUND_TRUTH_NAME=""
FILTERS=""
EVIDENCE=""
SVTYPES=""
ALGORITHMS=""
MIN_AF=""
MAX_AF=""
SVLEN_LOWER=""
SVLEN_UPPER=""
BOTHSIDES_SUPPORT_ONLY=false
NO_CANONICAL_CHROMS=false
FORCE_RERUN=false
VERBOSE=false

while true; do
    case "$1" in
        --ground_truth_name)    GROUND_TRUTH_NAME="$2"; shift 2 ;;
        --filters)              FILTERS="$2";           shift 2 ;;
        --evidence)             EVIDENCE="$2";          shift 2 ;;
        --svtypes)              SVTYPES="$2";           shift 2 ;;
        --algorithms)           ALGORITHMS="$2";        shift 2 ;;
        --min_af)               MIN_AF="$2";            shift 2 ;;
        --max_af)               MAX_AF="$2";            shift 2 ;;
        --svlen_lower)          SVLEN_LOWER="$2";       shift 2 ;;
        --svlen_upper)          SVLEN_UPPER="$2";       shift 2 ;;
        --bothsides_support_only) BOTHSIDES_SUPPORT_ONLY=true; shift ;;
        --no_canonical_chroms)  NO_CANONICAL_CHROMS=true; shift ;;
        --force_rerun)          FORCE_RERUN=true;       shift ;;
        -v|--verbose)           VERBOSE=true;           shift ;;
        --)                     shift; break ;;
        *)                      die "Unexpected option: $1" ;;
    esac
done

[[ -n "$GROUND_TRUTH_NAME" ]] || die "--ground_truth_name is required"

module purge
module load micromamba

logI "Preprocessing ground truth: ${GROUND_TRUTH_NAME}"

cmd=(micromamba run -n SVEnsemble2 python -m svensemble.ground_truth
     --ground_truth_name "$GROUND_TRUTH_NAME")

[[ -n "$FILTERS" ]]    && cmd+=(--filters "$FILTERS")
[[ -n "$EVIDENCE" ]]   && cmd+=(--evidence "$EVIDENCE")
[[ -n "$SVTYPES" ]]    && cmd+=(--svtypes "$SVTYPES")
[[ -n "$ALGORITHMS" ]] && cmd+=(--algorithms "$ALGORITHMS")
[[ -n "$MIN_AF" ]]     && cmd+=(--min_af "$MIN_AF")
[[ -n "$MAX_AF" ]]     && cmd+=(--max_af "$MAX_AF")
[[ -n "$SVLEN_LOWER" ]] && cmd+=(--svlen_lower "$SVLEN_LOWER")
[[ -n "$SVLEN_UPPER" ]] && cmd+=(--svlen_upper "$SVLEN_UPPER")
[[ "$BOTHSIDES_SUPPORT_ONLY" = true ]] && cmd+=(--bothsides_support_only)
[[ "$NO_CANONICAL_CHROMS" = true ]]    && cmd+=(--no_canonical_chroms)
[[ "$FORCE_RERUN" = true ]]            && cmd+=(--force_rerun)
[[ "$VERBOSE" = true ]]                && cmd+=(--verbose)

logI "Command: ${cmd[*]}"
"${cmd[@]}"
logI "Ground truth preprocessing complete for ${GROUND_TRUTH_NAME}"
