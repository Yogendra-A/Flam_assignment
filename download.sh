#!/usr/bin/env bash
# download.sh — fetch raw corpora for the Hindi NLP pipeline
# Usage: bash download.sh [raw_dir]
set -euo pipefail

RAW_DIR="${1:-raw_data}"
mkdir -p "$RAW_DIR"

echo "[1/3] Downloading Hindi Wikipedia (first 12 000 records)…"
python3 - <<'PYEOF'
import os, json, sys
from datasets import load_dataset

raw_dir = os.environ.get("RAW_DIR", "raw_data")
out_path = os.path.join(raw_dir, "wiki_hi.jsonl")

ds = load_dataset(
    "wikimedia/wikipedia",
    "20231101.hi",
    split="train",
    streaming=True,
    trust_remote_code=False,
)

count = 0
with open(out_path, "w", encoding="utf-8") as fh:
    for record in ds:
        if count >= 12000:
            break
        row = {
            "id":     record.get("id", ""),
            "text":   record.get("text", ""),
            "source": "wiki_hi",
        }
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        count += 1
        if count % 1000 == 0:
            print(f"  {count} records written…", flush=True)

print(f"Saved {count} records to {out_path}")
PYEOF

echo "[2/3] Downloading WillHeld/hinglish_top…"
python3 - <<'PYEOF'
import os, json
from datasets import load_dataset

raw_dir = os.environ.get("RAW_DIR", "raw_data")
out_path = os.path.join(raw_dir, "hinglish.jsonl")

ds = load_dataset("WillHeld/hinglish_top", split="train", trust_remote_code=False)

# Inspect column names dynamically
cols = ds.column_names
text_col = "sentence" if "sentence" in cols else "text"
print(f"  Using column '{text_col}' (available: {cols})")

with open(out_path, "w", encoding="utf-8") as fh:
    for record in ds:
        row = {
            "id":     str(record.get("id", "")),
            "text":   record.get(text_col, ""),
            "source": "hinglish_top",
        }
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")

print(f"Saved {len(ds)} records to {out_path}")
PYEOF

echo "[3/3] Writing sources.json…"
python3 - <<'PYEOF'
import os, json
from datetime import date

raw_dir = os.environ.get("RAW_DIR", "raw_data")
today = date.today().isoformat()

sources = [
    {
        "name":          "Hindi Wikipedia",
        "hf_dataset":    "wikimedia/wikipedia",
        "hf_config":     "20231101.hi",
        "url":           "https://huggingface.co/datasets/wikimedia/wikipedia",
        "license":       "CC BY-SA 4.0",
        "records_used":  12000,
        "download_date": today,
    },
    {
        "name":          "Hinglish TOP",
        "hf_dataset":    "WillHeld/hinglish_top",
        "url":           "https://huggingface.co/datasets/WillHeld/hinglish_top",
        "license":       "MIT",
        "download_date": today,
    },
]

out_path = os.path.join(raw_dir, "sources.json")
with open(out_path, "w", encoding="utf-8") as fh:
    json.dump(sources, fh, ensure_ascii=False, indent=2)
print(f"Provenance written to {out_path}")
PYEOF

# Export RAW_DIR so the heredoc python3 subprocesses can read it
export RAW_DIR
echo "Done. Raw data written to: $RAW_DIR/"
