#!/usr/bin/env bash
set -euo pipefail

SAMPLES=(NA12878 HG002 CHM13 HG00514)

for sample_name in "${SAMPLES[@]}"; do
    mkdir -p "data/${sample_name}/sv_calling/metrics/"
    find "data/${sample_name}" -name '*.benchmark.csv' \
        | python scripts/compute_metrics.py \
            --benchmark_csvs_file - \
            --sample_name "$sample_name" \
            --output "data/${sample_name}/sv_calling/metrics/metrics_all.csv"
done