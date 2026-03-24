#!/bin/bash
# =============================================================================
# new_data.sh — Scaffold a new sample data directory
#
# Creates the full directory tree, symlinks shared scripts and configs, and
# generates a skeleton sample.sh and default run config for the new sample.
#
# Usage:
#   bash scripts/new_data.sh <SAMPLE_NAME>
#
# The script must be run from the project root, or with SVENSEMBLE_BASE_DIR set.
# =============================================================================
set -euo pipefail

# --------------- Resolve project root ----------------------------------------
PROJ_DIR="${SVENSEMBLE_BASE_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
SCRIPTS_DIR="$PROJ_DIR/scripts/data_scripts"
SCRATCH_DATA_DIR="/vast/scratch/users/shikov.v/Papenfuss_SVEnsemble2/data"

# --------------- Validate arguments ------------------------------------------
if [[ $# -lt 1 || -z "$1" ]]; then
    echo "Usage: $0 <SAMPLE_NAME>" >&2
    exit 1
fi

SAMPLE_NAME="$1"
SAMPLE_DIR="$PROJ_DIR/data/$SAMPLE_NAME"

if [[ -d "$SAMPLE_DIR" ]]; then
    echo "Error: sample directory already exists: $SAMPLE_DIR" >&2
    exit 1
fi

echo "Creating sample directory: $SAMPLE_DIR"

# --------------- Create directory tree ---------------------------------------
mkdir -p "$SAMPLE_DIR"/{scripts,logs,preprocessed,aligned,sv_calling}
mkdir -p "$SAMPLE_DIR/scripts/config/runs"

# --------------- Create scratch dirs and symlink ----------------------------
# downloads/, temp/, and svensemble2 results live on scratch to avoid filling
# the VAST project quota.
SCRATCH_SAMPLE_DIR="$SCRATCH_DATA_DIR/$SAMPLE_NAME"
mkdir -p "$SCRATCH_SAMPLE_DIR"/{downloads,temp,sv_calling/svensemble2}
ln -sn "$SCRATCH_SAMPLE_DIR/downloads"              "$SAMPLE_DIR/downloads"
ln -sn "$SCRATCH_SAMPLE_DIR/temp"                   "$SAMPLE_DIR/temp"
ln -sn "$SCRATCH_SAMPLE_DIR/sv_calling/svensemble2" "$SAMPLE_DIR/sv_calling/svensemble2"

# --------------- Symlink shared batch scripts --------------------------------
# Only symlink regular files (*.sh) from data_scripts/; skip directories and
# files that should be sample-specific.
for script in "$SCRIPTS_DIR"/*.sh; do
    [[ -f "$script" ]] || continue
    name="$(basename "$script")"
    ln -sr "$script" "$SAMPLE_DIR/scripts/$name"
done

# --------------- Symlink shared config files ---------------------------------
# default_run.sh is shared across all samples via symlink.
ln -sr "$SCRIPTS_DIR/config/default_run.sh" "$SAMPLE_DIR/scripts/config/default_run.sh"

# Shared run configs (TEMPLATE, svensemble_full, svensemble_train_predict, etc.)
for run_cfg in "$SCRIPTS_DIR"/config/runs/*.sh; do
    [[ -f "$run_cfg" ]] || continue
    ln -sr "$run_cfg" "$SAMPLE_DIR/scripts/config/runs/$(basename "$run_cfg")"
done

# --------------- Generate sample.sh skeleton ---------------------------------
cat > "$SAMPLE_DIR/scripts/config/sample.sh" <<SAMPLE_EOF
#!/bin/bash
# =============================================================================
# sample.sh — ${SAMPLE_NAME} SAMPLE-LEVEL CONFIGURATION
#
# Layer 2 in the config cascade (sourced after default_run.sh).
# Contains:
#   - Sample identity (name, directory)
#   - Input data paths
#   - Sample-level overrides that apply to ALL runs
# =============================================================================

SAMPLE_NAME=${SAMPLE_NAME}
SAMPLE_DIR="\${PROJ_DIR}/data/\${SAMPLE_NAME}"

# --------------- Input reads ------------------------------------------------
# Short reads (Illumina)
READS1_PATH=
READS2_PATH=

# Long reads (PacBio HiFi / ONT)
READS_LONG_PATHS=()
READS_LONG_SOURCE=             # "hifi" or "ont"
MINIMAP_PRESET=                # "map-hifi" or "map-ont"

# Pre-aligned input (set ONE of these, leave others empty)
SAM_INPUT=
CRAM_INPUT=
CRAM_FASTA=
CRAM_REF_PATH=
CRAM_REF_CACHE=

# --------------- Sample-level overrides -------------------------------------
# Uncomment and modify as needed. Everything else inherits from default_run.sh.
# SVE_MODELS=("unbiased")
# SVE_TRAINING_SETS=("gridss" "delly" "manta_diploid")
# SVE_BENCH_ADDITIONAL_CALLSETS=("survivor_short")
SAMPLE_EOF

# --------------- Generate default run config ---------------------------------
cat > "$SAMPLE_DIR/scripts/config/runs/default.sh" <<RUN_EOF
#!/bin/bash
# =============================================================================
# runs/default.sh — ${SAMPLE_NAME}: default run configuration
#
# Edit this file to enable the pipeline steps you need.
# See TEMPLATE.sh for all available parameters.
# =============================================================================

_RUN_DIR="\$(cd "\$(dirname "\${BASH_SOURCE[0]}")" && pwd)"
source "\${_RUN_DIR}/../default_run.sh"
source "\${_RUN_DIR}/../sample.sh"
unset _RUN_DIR

# --------------- Run-specific overrides -------------------------------------
# RUN_MANTA=true
# RUN_GRIDSS=true
# RUN_DELLY=true
# RUN_SVENSEMBLE2=true
# RUN_BENCHMARK=true
# BENCHMARK=true
RUN_EOF

echo ""
echo "Sample scaffolded successfully:"
echo "  Directory:    $SAMPLE_DIR"
echo "  Sample config: $SAMPLE_DIR/scripts/config/sample.sh"
echo "  Default run:   $SAMPLE_DIR/scripts/config/runs/default.sh"
echo "  Run template:  $SAMPLE_DIR/scripts/config/runs/TEMPLATE.sh (symlink)"
echo ""
echo "Next steps:"
echo "  1. Edit scripts/config/sample.sh — set input read paths"
echo "  2. Edit scripts/config/runs/default.sh — enable pipeline steps"
echo "  3. Run: bash scripts/data_scripts/master.sh $SAMPLE_DIR/scripts/config/runs/default.sh"
