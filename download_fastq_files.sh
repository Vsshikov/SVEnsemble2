#!/usr/bin/env bash
# download_fastq_files.sh
#
# Usage:
#   download_fastq_files.sh <filelist.tsv> <output_directory>
#
# The TSV file is expected to have a header line followed by rows with these tab-separated columns:
#   1) URL to first FASTQ
#   2) MD5 checksum of first FASTQ
#   3) URL to second FASTQ (mate-pair)
#   4) MD5 checksum of second FASTQ
#   5) (Optional) Sample identifier (ignored by this script)
#
# The script will:
#   • Create the output directory if it does not exist.
#   • Download each file (skipping ones that are already present).
#   • Verify each file's MD5 hash and abort on mismatch.
#
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $(basename "$0") <filelist.tsv> <output_directory>" >&2
  exit 1
fi

filelist="$1"
outdir="$2"

mkdir -p "$outdir"

echo "Downloading files listed in $filelist to $outdir"

# Skip the first line (header) using tail
# shellcheck disable=SC2002
cat "$filelist" | tail -n +2 | while IFS=$'\t' read -r url1 md5_1 url2 md5_2 _sample_rest; do
  # Skip empty lines or lines beginning with '#'
  [[ -z "${url1}" || "${url1}" == \#* ]] && continue

  for idx in 1 2; do
    url_var="url${idx}"
    md5_var="md5_${idx}"
    url="${!url_var}"
    checksum="${!md5_var}"

    [[ -z "${url}" ]] && continue  # Handle missing columns

    filename="$(basename "${url}")"
    dest="${outdir}/${filename}"

    if [[ -f "${dest}" ]]; then
      echo "[SKIP] ${filename} already exists."
    else
      echo "[DL]   ${filename}"
      wget -q --show-progress -c -O "${dest}" "${url}"
    fi

    echo "[CHK]  ${filename}"
    if echo "${checksum}  ${dest}" | md5sum -c -; then
      echo "[OK]   ${filename} verified."
    else
      echo "[ERR]  Checksum failed for ${filename}. Removing corrupt file." >&2
      rm -f "${dest}"
      exit 1
    fi
  done

done

echo "All downloads complete and verified."