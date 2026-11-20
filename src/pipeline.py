#!/usr/bin/env python3
"""
pipeline.py — Streaming orchestrator for the Hindi NLP corpus pipeline.

Usage:
    python src/pipeline.py --raw-dir raw_data --out-dir out_data

Outputs:
    <out_dir>/clean/data.jsonl.gz   — cleaned corpus (id, text, lang, source)
    <out_dir>/ledger.jsonl          — one row per input doc
    <out_dir>/eval.jsonl            — held-out eval set (~200 docs)
    <out_dir>/contamination.jsonl   — contamination check results
    <out_dir>/manifest.json         — provenance, hashes, counts
    <out_dir>/sample_1000.jsonl     — 1000-row random sample (seed=42)
    <out_dir>/debug_dropped_samples.jsonl  — sample of dropped docs
    <out_dir>/debug_neardup_pairs.jsonl    — borderline near-dup pairs
"""
import argparse
import csv
import gzip
import hashlib
import json
import os
import random
import shutil
import sys
import time
from pathlib import Path
from typing import Iterator, Dict, Any, List, Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from text_normalizer import normalize
from language_identifier import classify as langid_classify
from quality_filters import apply_filters
from deduplicator import DedupState
from pii_redactor import redact as pii_redact
from contamination_checker import ContaminationChecker, should_holdout


def sha256_file(path: Path) -> str:
    """Compute SHA-256 of a file."""
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(65536), b''):
            h.update(chunk)
    return h.hexdigest()


def iter_jsonl(path: Path) -> Iterator[dict]:
    """Stream records from a .jsonl or .jsonl.gz file."""
    open_fn = gzip.open if path.suffix == '.gz' else open
    mode = 'rt' if path.suffix == '.gz' else 'r'
    with open_fn(path, mode, encoding='utf-8') as f:
        for line_num, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                yield {'_error': str(e), '_line': line[:200], '_line_num': line_num}


def iter_txt(path: Path) -> Iterator[dict]:
    """Stream records from a .txt file (one doc per paragraph)."""
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()
    paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
    for i, para in enumerate(paragraphs):
        yield {'id': f"{path.stem}_{i}", 'text': para, 'source': str(path.name)}


def iter_csv(path: Path) -> Iterator[dict]:
    """Stream records from a .csv file."""
    with open(path, 'r', encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            # Find text column
            text_col = None
            for col in ('text', 'sentence', 'content', 'body'):
                if col in row:
                    text_col = col
                    break
            if text_col is None and row:
                text_col = list(row.keys())[0]
            yield {
                'id': row.get('id', f"{path.stem}_{i}"),
                'text': row.get(text_col, ''),
                'source': str(path.name)
            }


def stream_raw_dir(raw_dir: Path) -> Iterator[tuple]:
    """
    Stream all documents from raw_dir.
    Yields (file_path, raw_record) tuples.
    Files are sorted deterministically.
    """
    patterns = ['*.jsonl', '*.jsonl.gz', '*.txt', '*.csv']
    files = []
    for pat in patterns:
        files.extend(raw_dir.glob(pat))
    
    # Sort deterministically
    files = sorted(files, key=lambda p: p.name)
    
    for file_path in files:
        suffix = file_path.suffix.lower()
        full_suffix = ''.join(file_path.suffixes).lower()
        
        if full_suffix in ('.jsonl.gz',) or suffix == '.gz':
            yield from ((file_path, rec) for rec in iter_jsonl(file_path))
        elif suffix == '.jsonl':
            yield from ((file_path, rec) for rec in iter_jsonl(file_path))
        elif suffix == '.txt':
            yield from ((file_path, rec) for rec in iter_txt(file_path))
        elif suffix == '.csv':
            yield from ((file_path, rec) for rec in iter_csv(file_path))


def run_pipeline(raw_dir: Path, out_dir: Path, threshold: float = 0.80, sample_size: int = 1000):
    """Main pipeline runner."""
    # Keep the staging directory beside out_dir so replacing out_dir does not
    # move the staging tree along with it on Windows.
    tmp_dir = out_dir.parent / (out_dir.name + '.tmp_run')
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True)
    (tmp_dir / 'clean').mkdir()
    
    print(f"[pipeline] raw_dir={raw_dir} out_dir={out_dir} threshold={threshold}")
    start_time = time.time()
    
    # Count source file hashes for manifest
    raw_files = sorted(raw_dir.glob('*'), key=lambda p: p.name)
    file_hashes = {}
    for f in raw_files:
        if f.is_file() and f.suffix in ('.jsonl', '.gz', '.txt', '.csv'):
            file_hashes[f.name] = sha256_file(f)
    
    # State
    dedup_state = DedupState(threshold=threshold)
    contamination_checker = ContaminationChecker(near_dup_threshold=0.70, cross_script_threshold=0.60)
    
    # Counts per stage per lang
    counts = {
        'total_input': 0,
        'malformed': 0,
        'filtered_quality': 0,
        'filtered_exact_dup': 0,
        'filtered_near_dup': 0,
        'eval_holdout': 0,
        'contamination_removed': 0,
        'kept': 0,
        'by_lang': {'hi': 0, 'hi-Latn': 0, 'en': 0, 'other': 0},
        'eval_by_lang': {'hi': 0, 'hi-Latn': 0},
    }
    
    # Open output files
    ledger_path = tmp_dir / 'ledger.jsonl'
    eval_path = tmp_dir / 'eval.jsonl'
    contamination_path = tmp_dir / 'contamination.jsonl'
    debug_dropped_path = tmp_dir / 'debug_dropped_samples.jsonl'
    debug_neardup_path = tmp_dir / 'debug_neardup_pairs.jsonl'
    clean_path = tmp_dir / 'clean' / 'data.jsonl.gz'
    
    dropped_samples = []
    eval_docs = []
    train_docs_for_contamination = []
    
    # Pass 1: Normalize, filter, dedup, PII — collect eval and training sets
    print("[pipeline] Pass 1: normalize -> langid -> filter -> dedup -> pii...")
    
    ledger_records = []
    with open(ledger_path, 'w', encoding='utf-8') as ledger_f:
        def write_ledger(doc_id, status, stage, reason='', dup_of=None, lang=None):
            rec = {'id': doc_id, 'status': status, 'stage': stage, 'reason': reason}
            if dup_of:
                rec['dup_of'] = dup_of
            if lang:
                rec['lang'] = lang
            ledger_records.append(rec)
        
        for file_path, raw_rec in stream_raw_dir(raw_dir):
            counts['total_input'] += 1
            
            # Handle malformed records
            if '_error' in raw_rec:
                doc_id = f"malformed_{counts['total_input']}"
                counts['malformed'] += 1
                write_ledger(doc_id, 'dropped', 'ingest', 'malformed_record')
                continue
            
            doc_id = str(raw_rec.get('id', f"doc_{counts['total_input']}"))
            raw_text = raw_rec.get('text', '')
            source = raw_rec.get('source', str(file_path.name))
            
            if not raw_text or not str(raw_text).strip():
                counts['malformed'] += 1
                write_ledger(doc_id, 'dropped', 'ingest', 'empty_text')
                continue
            
            raw_text = str(raw_text)
            
            # Stage 1: Normalize
            norm_text = normalize(raw_text)
            
            # Stage 2: Language ID
            lang = langid_classify(norm_text)
            
            # Only keep hi and hi-Latn
            if lang not in ('hi', 'hi-Latn'):
                write_ledger(doc_id, 'dropped', 'langid', f'lang_{lang}', lang=lang)
                continue
            
            # Stage 3: Quality filter
            keep, filter_reason = apply_filters(norm_text)
            if not keep:
                counts['filtered_quality'] += 1
                write_ledger(doc_id, 'dropped', 'filter', filter_reason, lang=lang)
                # Sample dropped docs for debug
                if len(dropped_samples) < 200:
                    dropped_samples.append({
                        'id': doc_id, 'text': norm_text[:500],
                        'lang': lang, 'reason': filter_reason
                    })
                continue
            
            # Stage 4: Dedup
            is_dup, dup_of = dedup_state.check_and_add(doc_id, norm_text)
            if is_dup:
                if dup_of and dedup_state.exact_hashes.get(
                    hashlib.sha256(norm_text.encode()).hexdigest()
                ) != doc_id:
                    counts['filtered_near_dup'] += 1
                    write_ledger(doc_id, 'dropped', 'dedup', 'near_duplicate', dup_of, lang)
                else:
                    counts['filtered_exact_dup'] += 1
                    write_ledger(doc_id, 'dropped', 'dedup', 'exact_duplicate', dup_of, lang)
                continue
            
            # Stage 5: PII redaction
            clean_text, pii_counts = pii_redact(norm_text)
            
            # Stage 6: Eval holdout (deterministic by id hash)
            if should_holdout(doc_id, lang):
                if lang == 'hi' and counts['eval_by_lang']['hi'] < 150:
                    eval_docs.append({
                        'id': doc_id, 'text': clean_text, 'lang': lang,
                        'source': source, 'norm_text': norm_text
                    })
                    counts['eval_by_lang']['hi'] += 1
                    counts['eval_holdout'] += 1
                    write_ledger(doc_id, 'kept', 'eval_holdout', 'eval', lang=lang)
                elif lang == 'hi-Latn' and counts['eval_by_lang']['hi-Latn'] < 50:
                    eval_docs.append({
                        'id': doc_id, 'text': clean_text, 'lang': lang,
                        'source': source, 'norm_text': norm_text
                    })
                    counts['eval_by_lang']['hi-Latn'] += 1
                    counts['eval_holdout'] += 1
                    write_ledger(doc_id, 'kept', 'eval_holdout', 'eval', lang=lang)
                else:
                    # Over quota for this lang — treat as training
                    train_docs_for_contamination.append({
                        'id': doc_id, 'text': clean_text, 'lang': lang,
                        'source': source, 'norm_text': norm_text
                    })
                    write_ledger(doc_id, 'kept', 'output', 'training', lang=lang)
                    counts['kept'] += 1
                    counts['by_lang'][lang] += 1
            else:
                train_docs_for_contamination.append({
                    'id': doc_id, 'text': clean_text, 'lang': lang,
                    'source': source, 'norm_text': norm_text
                })
                write_ledger(doc_id, 'kept', 'output', 'training', lang=lang)
                counts['kept'] += 1
                counts['by_lang'][lang] += 1
            
            if counts['total_input'] % 1000 == 0:
                print(f"  {counts['total_input']} docs processed, {counts['kept']} kept...")
    
    print(f"[pipeline] Pass 1 done. {counts['total_input']} input, {counts['kept']} training, {counts['eval_holdout']} eval")
    
    # Write eval.jsonl
    print(f"[pipeline] Writing eval.jsonl ({len(eval_docs)} docs)...")
    with open(eval_path, 'w', encoding='utf-8') as f:
        for doc in eval_docs:
            out = {'id': doc['id'], 'text': doc['text'], 'lang': doc['lang'], 'source': doc['source']}
            f.write(json.dumps(out, ensure_ascii=False) + '\n')
    
    # Build contamination index from eval docs
    print("[pipeline] Building contamination index...")
    for doc in eval_docs:
        contamination_checker.add_eval(doc)
    
    # Pass 2: Contamination check on training docs
    print("[pipeline] Pass 2: contamination check on training docs...")
    contamination_records = []
    contaminated_ids = set()
    
    for doc in train_docs_for_contamination:
        result = contamination_checker.check_train(doc)
        if result:
            contaminated_ids.add(doc['id'])
            contamination_records.append(result)
            counts['contamination_removed'] += 1
            counts['kept'] -= 1
            counts['by_lang'][doc['lang']] = max(0, counts['by_lang'][doc['lang']] - 1)
    
    # Write contamination.jsonl
    with open(contamination_path, 'w', encoding='utf-8') as f:
        for rec in contamination_records:
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')
    
    # Write debug_contamination_rejected.jsonl
    debug_contamination_rejected_path = tmp_dir / 'debug_contamination_rejected.jsonl'
    with open(debug_contamination_rejected_path, 'w', encoding='utf-8') as f:
        for rec in contamination_checker.rejected_borderline:
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')
    
    # Write clean corpus (excluding contaminated docs)
    print(f"[pipeline] Writing clean corpus (excluding {len(contaminated_ids)} contaminated docs)...")
    clean_docs = []
    with gzip.open(clean_path, 'wt', encoding='utf-8') as f:
        for doc in train_docs_for_contamination:
            if doc['id'] not in contaminated_ids:
                out = {'id': doc['id'], 'text': doc['text'], 'lang': doc['lang'], 'source': doc['source']}
                f.write(json.dumps(out, ensure_ascii=False) + '\n')
                clean_docs.append(out)
    
    # Update the existing ledger rows in place: exactly one row per input.
    contamination_by_id = {
        train_id: rec
        for rec in contamination_records
        for train_id in rec['train_ids']
    }
    for entry in ledger_records:
        rec = contamination_by_id.get(entry['id'])
        if rec:
            entry.update({
                'status': 'dropped', 'stage': 'contamination',
                'reason': f"contamination_{rec['method']}",
                'dup_of': rec['eval_id']
            })
    with open(ledger_path, 'w', encoding='utf-8') as ledger_f:
        for entry in ledger_records:
            ledger_f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    
    # Write sample_1000.jsonl
    print(f"[pipeline] Writing sample of {sample_size} docs...")
    rng = random.Random(42)
    sample = rng.sample(clean_docs, min(sample_size, len(clean_docs)))
    sample_path = tmp_dir / 'sample_1000.jsonl'
    with open(sample_path, 'w', encoding='utf-8') as f:
        for doc in sample:
            f.write(json.dumps(doc, ensure_ascii=False) + '\n')
    
    # Write debug_dropped_samples.jsonl
    with open(debug_dropped_path, 'w', encoding='utf-8') as f:
        for doc in dropped_samples:
            f.write(json.dumps(doc, ensure_ascii=False) + '\n')
    
    # Write debug_neardup_pairs.jsonl
    with open(debug_neardup_path, 'w', encoding='utf-8') as f:
        for pair in dedup_state.get_borderline_pairs():
            f.write(json.dumps(pair, ensure_ascii=False) + '\n')
    
    # Compute output hashes
    output_hashes = {
        'clean_data': sha256_file(clean_path),
        'ledger': sha256_file(ledger_path),
        'eval': sha256_file(eval_path),
        'contamination': sha256_file(contamination_path),
        'sample_1000': sha256_file(sample_path),
    }
    
    # Write manifest.json
    # Keep the manifest byte-for-byte reproducible across reruns. Runtime is
    # useful interactively but would make the requested deterministic outputs
    # differ for no data-related reason.
    source_rows = _load_sources(raw_dir)
    source_dates = [row.get('download_date') for row in source_rows if row.get('download_date')]
    reproducible_timestamp = f"{max(source_dates)}T00:00:00Z" if source_dates else None
    manifest = {
        'run_timestamp': reproducible_timestamp,
        'elapsed_seconds': None,
        'raw_dir': str(raw_dir),
        'out_dir': str(out_dir),
        'thresholds': {
            'near_dup': threshold,
            'contamination_near_dup': 0.70,
            'contamination_cross_script': 0.60,
        },
        'input_files': file_hashes,
        'counts': {
            **counts,
            'contamination_removed': counts['contamination_removed'],
            'final_training_docs': len(clean_docs),
            'eval_docs': len(eval_docs),
        },
        'output_hashes': output_hashes,
        'sources': source_rows,
    }
    
    manifest_path = tmp_dir / 'manifest.json'
    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    
    print(f"[pipeline] manifest.json written. Counts: {json.dumps(counts, indent=2)}")
    
    # Atomic swap: tmp_dir → out_dir
    print(f"[pipeline] Atomic swap: {tmp_dir} -> {out_dir}")
    if out_dir.exists():
        old_dir = out_dir.parent / (out_dir.name + '.old')
        if old_dir.exists():
            shutil.rmtree(old_dir)
        out_dir.rename(old_dir)
    tmp_dir.rename(out_dir)
    
    # Clean up old dir (keep .tmp_run removed)
    old_dir = out_dir.parent / (out_dir.name + '.old')
    if old_dir.exists():
        shutil.rmtree(old_dir)
    
    print(f"[pipeline] Done! Outputs in {out_dir}")
    print(f"  Total input: {counts['total_input']}")
    print(f"  Malformed: {counts['malformed']}")
    print(f"  Quality filtered: {counts['filtered_quality']}")
    print(f"  Dedup exact: {counts['filtered_exact_dup']}")
    print(f"  Dedup near: {counts['filtered_near_dup']}")
    print(f"  Eval holdout: {counts['eval_holdout']}")
    print(f"  Contamination removed: {counts['contamination_removed']}")
    print(f"  Final training docs: {len(clean_docs)}")
    print(f"  By lang: {counts['by_lang']}")


def _load_sources(raw_dir: Path) -> list:
    sources_path = raw_dir / 'sources.json'
    if sources_path.exists():
        with open(sources_path) as f:
            return json.load(f)
    return []


def main():
    parser = argparse.ArgumentParser(description='Hindi NLP corpus pipeline')
    parser.add_argument('--raw-dir', default='raw_data', help='Raw data directory')
    parser.add_argument('--out-dir', default='out_data', help='Output directory')
    parser.add_argument('--threshold', type=float, default=0.80, help='Near-dup Jaccard threshold')
    parser.add_argument('--sample-size', type=int, default=1000, help='Sample size for sample_1000.jsonl')
    args = parser.parse_args()
    
    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    
    if not raw_dir.exists():
        print(f"ERROR: raw_dir {raw_dir} does not exist", file=sys.stderr)
        sys.exit(1)
    
    run_pipeline(raw_dir, out_dir, threshold=args.threshold, sample_size=args.sample_size)


if __name__ == '__main__':
    main()
