#!/bin/bash
#
# Propagate shared scripts, env YAMLs, and run configs to all sample directories.
# Safe to re-run: creates/updates symlinks, never overwrites real (non-symlink) files.
#
# Usage: scripts/sync_sample_scripts.sh [-n|--dry-run] [-c|--clean] [-h|--help] [SAMPLE ...]
# Run with -h for full help.

set -euo pipefail

PROJ_DIR="$(cd "$(dirname "$0")/.." && pwd)"
DATA_DIR="$PROJ_DIR/data"
CANONICAL_SCRIPTS="$PROJ_DIR/scripts/data_scripts"
CANONICAL_DOWNLOAD="$PROJ_DIR/scripts/download_scripts"

usage() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS] [SAMPLE ...]

Propagate shared scripts, env YAMLs, and run configs to all (or specified)
sample directories under data/.

Options:
  -n, --dry-run   Preview changes without applying them
  -c, --clean     Remove dangling symlinks in sample dirs
  -h, --help      Show this help message

Examples:
  $(basename "$0")                  # sync all samples
  $(basename "$0") HG002 NA12878   # sync two samples
  $(basename "$0") -nc              # dry-run + clean (combined short opts)
EOF
    exit 0
}

OPTS=$(getopt -o nch -l dry-run,clean,help -n "$(basename "$0")" -- "$@") || exit 1
eval set -- "$OPTS"

DRY_RUN=false
CLEAN=false

while true; do
    case "$1" in
        -n|--dry-run) DRY_RUN=true; shift ;;
        -c|--clean)   CLEAN=true;   shift ;;
        -h|--help)    usage ;;
        --)           shift; break ;;
    esac
done

FILTER_SAMPLES=("$@")

# ---------------------------------------------------------------------------
# Discover sample directories (those containing scripts/config/sample.sh)
# ---------------------------------------------------------------------------
sample_scripts_dirs=()
for candidate in "$DATA_DIR"/*/scripts/config/sample.sh; do
    [[ -f "$candidate" ]] || continue
    scripts_dir="$(dirname "$(dirname "$candidate")")"   # data/<sample>/scripts
    sample_name="$(basename "$(dirname "$scripts_dir")")" # <sample>

    if [[ ${#FILTER_SAMPLES[@]} -gt 0 ]]; then
        match=false
        for f in "${FILTER_SAMPLES[@]}"; do
            [[ "$sample_name" == "$f" ]] && match=true
        done
        $match || continue
    fi

    sample_scripts_dirs+=("$scripts_dir")
done

if [[ ${#sample_scripts_dirs[@]} -eq 0 ]]; then
    echo "No matching sample directories found." >&2
    exit 1
fi

echo "Samples (${#sample_scripts_dirs[@]}):"
for d in "${sample_scripts_dirs[@]}"; do
    printf '  %s\n' "$(basename "$(dirname "$d")")"
done
echo

# ---------------------------------------------------------------------------
# Sync helpers
# ---------------------------------------------------------------------------
total_created=0
total_updated=0
total_skipped=0
total_dangling=0

sync_files() {
    local src_dir="$1"
    local dest_subdir="$2"
    local glob="$3"
    shift 3
    local -a excludes=("$@")

    [[ -d "$src_dir" ]] || return 0

    for src_file in "$src_dir"/$glob; do
        [[ -e "$src_file" ]] || continue
        local fname
        fname="$(basename "$src_file")"

        # Check exclusion list
        local excluded=false
        for pattern in "${excludes[@]+"${excludes[@]}"}"; do
            # shellcheck disable=SC2254
            case "$fname" in $pattern) excluded=true ;; esac
        done
        $excluded && continue

        for scripts_dir in "${sample_scripts_dirs[@]}"; do
            local dest_dir="$scripts_dir/$dest_subdir"
            local dest="$dest_dir/$fname"
            local sample
            sample="$(basename "$(dirname "$scripts_dir")")"

            # Real file — never touch
            if [[ -e "$dest" && ! -L "$dest" ]]; then
                total_skipped=$((total_skipped + 1))
                continue
            fi

            # Already a correct relative symlink — nothing to do
            if [[ -L "$dest" ]]; then
                local current_target
                current_target="$(readlink "$dest")"
                if [[ "$current_target" != /* ]] && [[ "$(realpath "$dest")" == "$(realpath "$src_file")" ]]; then
                    continue
                fi
            fi

            if [[ -L "$dest" ]]; then
                local action="update"
                total_updated=$((total_updated + 1))
            else
                local action="create"
                total_created=$((total_created + 1))
            fi

            local display_subdir="${dest_subdir#.}"
            display_subdir="${display_subdir:+${display_subdir}/}"

            if $DRY_RUN; then
                if [[ "$action" == "update" ]]; then
                    echo "  [$action] $sample/${display_subdir}$fname  ($(readlink "$dest") -> $(realpath --relative-to="$dest_dir" "$src_file"))"
                else
                    echo "  [$action] $sample/${display_subdir}$fname"
                fi
            else
                mkdir -p "$dest_dir"
                ln -srf "$src_file" "$dest"
            fi
        done
    done
}

clean_dangling() {
    local dest_subdir="$1"
    local glob="$2"

    for scripts_dir in "${sample_scripts_dirs[@]}"; do
        local dest_dir="$scripts_dir/$dest_subdir"
        [[ -d "$dest_dir" ]] || continue
        local sample
        sample="$(basename "$(dirname "$scripts_dir")")"

        for link in "$dest_dir"/$glob; do
            [[ -L "$link" ]] || continue
            if [[ ! -e "$link" ]]; then
                total_dangling=$((total_dangling + 1))
                local fname
                fname="$(basename "$link")"
                if $DRY_RUN; then
                    echo "  [dangling] $sample/$dest_subdir/$fname -> $(readlink "$link")"
                else
                    echo "  Removing dangling symlink: $sample/$dest_subdir/$fname"
                    rm "$link"
                fi
            fi
        done
    done
}

# ---------------------------------------------------------------------------
# Sync
# ---------------------------------------------------------------------------
$DRY_RUN && echo "=== DRY RUN ==="
echo "Syncing shared scripts..."

sync_files "$CANONICAL_SCRIPTS"              "."            "*.sh"    "master_args*.sh"
sync_files "$CANONICAL_DOWNLOAD"             "."            "*.sh"
sync_files "$CANONICAL_SCRIPTS/envs"         "envs"         "*.yaml"
sync_files "$CANONICAL_SCRIPTS/config"       "config"       "default_run.sh"
sync_files "$CANONICAL_SCRIPTS/config/runs"  "config/runs"  "*.sh"

if $CLEAN; then
    echo "Cleaning dangling symlinks..."
    clean_dangling "."            "*.sh"
    clean_dangling "envs"         "*.yaml"
    clean_dangling "config"       "*.sh"
    clean_dangling "config/runs"  "*.sh"
fi

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo
echo "Summary:"
echo "  Created:  $total_created"
echo "  Updated:  $total_updated"
echo "  Skipped:  $total_skipped (real files)"
[[ "$CLEAN" == true ]] && echo "  Dangling: $total_dangling removed"
$DRY_RUN && echo "  (dry run — no changes applied)"
echo "Done."
