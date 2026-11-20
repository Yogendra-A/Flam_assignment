#!/usr/bin/env bash
# download.sh - fetch raw corpora for the Hindi NLP pipeline
# Usage: ./download.sh [raw_dir]
set -euo pipefail

RAW_DIR="${1:-raw_data}"
mkdir -p "$RAW_DIR"
export RAW_DIR

echo "[1/3] Downloading Hindi Wikipedia (first 12 000 records)..."
python3 - <<'PYEOF'
import json
import os
from datasets import load_dataset

raw_dir = os.environ['RAW_DIR']
out_path = os.path.join(raw_dir, 'wiki_hi.jsonl')
ds = load_dataset('wikimedia/wikipedia', '20231101.hi', split='train', streaming=True, trust_remote_code=False)
count = 0
with open(out_path, 'w', encoding='utf-8') as fh:
    for record in ds:
        if count >= 12000:
            break
        fh.write(json.dumps({'id': record.get('id', ''), 'text': record.get('text', ''), 'source': 'wiki_hi'}, ensure_ascii=False) + '\n')
        count += 1
        if count % 1000 == 0:
            print(f'  {count} records written...', flush=True)
print(f'Saved {count} records to {out_path}')
PYEOF

echo "[2/3] Downloading WillHeld/hinglish_top (first 2 500 records)..."
python3 - <<'PYEOF'
import json
import os
from datasets import load_dataset

raw_dir = os.environ['RAW_DIR']
out_path = os.path.join(raw_dir, 'hinglish.jsonl')
ds = load_dataset('WillHeld/hinglish_top', split='train', streaming=True, trust_remote_code=False)
cols = ds.column_names
text_col = 'sentence' if 'sentence' in cols else ('cs_query' if 'cs_query' in cols else 'text')
count = 0
with open(out_path, 'w', encoding='utf-8') as fh:
    for record in ds:
        if count >= 2500:
            break
        fh.write(json.dumps({'id': str(record.get('id', count)), 'text': record.get(text_col, ''), 'source': 'hinglish_top'}, ensure_ascii=False) + '\n')
        count += 1
print(f'Saved {count} records to {out_path}')
PYEOF

echo "[3/3] Writing sources.json..."
python3 - <<'PYEOF'
import json
import os
from datetime import date

raw_dir = os.environ['RAW_DIR']
sources = [
    {
        'name': 'Hindi Wikipedia',
        'hf_dataset': 'wikimedia/wikipedia',
        'hf_config': '20231101.hi',
        'url': 'https://huggingface.co/datasets/wikimedia/wikipedia',
        'license': 'CC BY-SA 4.0',
        'records_used': 12000,
        'file_size_mb': round(os.path.getsize(os.path.join(raw_dir, 'wiki_hi.jsonl')) / 1_000_000, 3),
        'download_date': date.today().isoformat(),
    },
    {
        'name': 'Hinglish TOP',
        'hf_dataset': 'WillHeld/hinglish_top',
        'hf_config': 'default',
        'url': 'https://huggingface.co/datasets/WillHeld/hinglish_top',
        'license': 'MIT',
        'records_used': 2500,
        'file_size_mb': round(os.path.getsize(os.path.join(raw_dir, 'hinglish.jsonl')) / 1_000_000, 3),
        'download_date': date.today().isoformat(),
    },
]
with open(os.path.join(raw_dir, 'sources.json'), 'w', encoding='utf-8') as fh:
    json.dump(sources, fh, ensure_ascii=False, indent=2)
print(f'Provenance written to {raw_dir}/sources.json')
PYEOF

echo "Done. Raw data written to: $RAW_DIR/"
