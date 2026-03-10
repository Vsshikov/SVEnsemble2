#!/bin/bash

#SBATCH --job-name=pipeline_finalizer
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=128M

# Lightweight finalizer job submitted with --dependency=afterany:<all_job_ids>.
# By the time SLURM schedules this, every pipeline job has either completed,
# failed, or been cancelled.  We query sacct for their final statuses and
# append a human-readable summary to the run log.
#
# Usage (called from master.sh):
#   sbatch --dependency=afterany:id1:id2:... finalizer.sh <log_dir> <space-separated job IDs>

set -euo pipefail

LOG_DIR="$1"
shift
read -ra JOB_IDS <<< "$*"

RUN_LOG="$LOG_DIR/run.log"

SACCT_FMT="JobID%-20,JobName%-24,State%-16,ExitCode,Elapsed,MaxRSS"

{
    printf '\n'
    printf '=%.0s' {1..72}
    printf '\n'
    printf '  PIPELINE COMPLETION SUMMARY  (%s)\n' "$(date -Iseconds)"
    printf '=%.0s' {1..72}
    printf '\n\n'

    sacct -j "$(IFS=,; printf '%s' "${JOB_IDS[*]}")" \
        --format="$SACCT_FMT" --parsable2 --noheader \
    | grep -v '\.batch' \
    | grep -v '\.extern' \
    | while IFS='|' read -r job_id job_name state exit_code elapsed max_rss; do
        printf '  %-20s %-24s %-16s exit=%-6s %s  %s\n' \
            "$job_id" "$job_name" "$state" "$exit_code" "$elapsed" "$max_rss"
    done

    printf '\n'

    sacct_data=$(sacct -j "$(IFS=,; printf '%s' "${JOB_IDS[*]}")" \
        --format=JobID%-20,JobName%-24,State --noheader --parsable2 \
        | grep -v '\.batch' \
        | grep -v '\.extern' \
        | grep -v '\+$' \
        | cut -d'|' -f2-)

    n_total=$(printf '%s\n' "$sacct_data" | grep -c . || true)
    n_failed=$(printf '%s\n' "$sacct_data" | grep -cviE 'COMPLETED$' || true)

    # Per-step breakdown
    printf '  %-24s %8s %8s\n' "STEP" "OK" "FAILED"
    printf '  %-24s %8s %8s\n' "------------------------" "--------" "--------"
    printf '%s\n' "$sacct_data" | awk -F'|' '{
        name=$1; gsub(/^[ \t]+|[ \t]+$/, "", name);
        state=$2; gsub(/^[ \t]+|[ \t]+$/, "", state);
        total[name]++;
        if (state == "COMPLETED") ok[name]++;
        else fail[name]++;
    } END {
        for (name in total)
            printf "  %-24s %8d %8d\n", name, ok[name]+0, fail[name]+0
    }' | sort

    printf '\n'

    if (( n_failed == 0 )); then
        printf '  Result: ALL %d jobs COMPLETED successfully.\n' "$n_total"
    else
        printf '  Result: %d of %d jobs did NOT complete successfully.\n' \
            "$n_failed" "$n_total"
    fi

    printf '\n'
    printf '=%.0s' {1..72}
    printf '\n'
} >> "$RUN_LOG"
