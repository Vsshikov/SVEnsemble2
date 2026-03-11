#!/bin/bash

#SBATCH --job-name=gnomad_ground_truth
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

while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --ground_truth_name)
            GROUND_TRUTH_NAME="$2"
            shift
            ;;
        --filters)
            FILTERS="$2"
            shift
            ;;
        --evidence)
            EVIDENCE="$2"
            shift
            ;;
        --svtypes)
            SVTYPES="$2"
            shift
            ;;
        --algorithms)
            ALGORITHMS="$2"
            shift
            ;;
        --min_af)
            MIN_AF="$2"
            shift
            ;;
        --max_af)
            MAX_AF="$2"
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
        --bothsides_support_only)
            BOTHSIDES_SUPPORT_ONLY=true
            ;;
        --no_canonical_chroms)
            NO_CANONICAL_CHROMS=true
            ;;
        --force_rerun)
            FORCE_RERUN=true
            ;;
        -v|--verbose)
            VERBOSE=true
            ;;
        *)
            die "Unknown parameter passed: $1"
            ;;
    esac
    shift
done

if [ -z "$GROUND_TRUTH_NAME" ]; then
    die "--ground_truth_name is required"
fi

module purge
module load micromamba

logI "Preprocessing ground truth: ${GROUND_TRUTH_NAME}"

command="micromamba run -n SVEnsemble2 python -m svensemble.ground_truth --ground_truth_name $GROUND_TRUTH_NAME"

if [ -n "$FILTERS" ]; then
    command+=" --filters $FILTERS"
fi
if [ -n "$EVIDENCE" ]; then
    command+=" --evidence $EVIDENCE"
fi
if [ -n "$SVTYPES" ]; then
    command+=" --svtypes $SVTYPES"
fi
if [ -n "$ALGORITHMS" ]; then
    command+=" --algorithms $ALGORITHMS"
fi
if [ -n "$MIN_AF" ]; then
    command+=" --min_af $MIN_AF"
fi
if [ -n "$MAX_AF" ]; then
    command+=" --max_af $MAX_AF"
fi
if [ -n "$SVLEN_LOWER" ]; then
    command+=" --svlen_lower $SVLEN_LOWER"
fi
if [ -n "$SVLEN_UPPER" ]; then
    command+=" --svlen_upper $SVLEN_UPPER"
fi
if [ "$BOTHSIDES_SUPPORT_ONLY" = true ]; then
    command+=" --bothsides_support_only"
fi
if [ "$NO_CANONICAL_CHROMS" = true ]; then
    command+=" --no_canonical_chroms"
fi
if [ "$FORCE_RERUN" = true ]; then
    command+=" --force_rerun"
fi
if [ "$VERBOSE" = true ]; then
    command+=" --verbose"
fi

logI "Command: $command"
$command
logI "Ground truth preprocessing complete for ${GROUND_TRUTH_NAME}"
