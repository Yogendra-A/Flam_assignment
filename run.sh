#!/usr/bin/env bash
# run.sh — run the full Hindi NLP pipeline
# Usage: bash run.sh [raw_dir] [out_dir]
set -euo pipefail

RAW_DIR="${1:-raw_data}"
OUT_DIR="${2:-out_data}"

echo "Running pipeline: raw=$RAW_DIR  out=$OUT_DIR"
python3 src/pipeline.py --raw-dir "$RAW_DIR" --out-dir "$OUT_DIR"
