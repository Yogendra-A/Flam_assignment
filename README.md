# Hindi + Hinglish Data Curation Pipeline

## Overview

This repository contains a deterministic, offline-first curation pipeline for Hindi and Hinglish LLM data. It normalizes Unicode, identifies Devanagari Hindi and Latin-script Hinglish, removes low-quality text, redacts Indian PII, performs exact and MinHash near-deduplication, creates a deterministic held-out evaluation set, and checks train/eval contamination including cross-script matches.

## Repository structure

| Path | Responsibility |
|---|---|
| `download.sh` | Streams the pinned Hindi Wikipedia and Hinglish sources and writes `sources.json`. |
| `run.sh` | Runs the pipeline offline from a raw-data directory to an output directory. |
| `src/text_normalizer.py` | Devanagari-aware NFC and whitespace normalization. |
| `src/language_identifier.py` | Classifies text as `hi`, `hi-Latn`, `en`, or `other`. |
| `src/quality_filters.py` | Removes malformed, spammy, markup-heavy, and repetitive text. |
| `src/deduplicator.py` | Exact SHA-256 and MinHash/LSH duplicate detection. |
| `src/pii_redactor.py` | Redacts Indian phone, email, PAN, and validated Aadhaar numbers. |
| `src/contamination_checker.py` | Builds eval holdouts and checks four contamination tiers. |
| `src/pipeline.py` | Deterministic streaming orchestrator and output writer. |
| `DECISIONS.md` | Run-specific design decisions, counts, examples, and limitations. |
| `deliverables/` | Committed eval, contamination, manifest, and sample artifacts. |

## Quickstart

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Download the pinned sources:

```bash
./download.sh raw_data
```

Run offline on an existing raw directory:

```bash
./run.sh <raw_dir> <out_dir>
# Example:
./run.sh raw_data out_data
```

The pipeline does not access the network during `run.sh`.

## Outputs

The output directory contains:

- `clean/data.jsonl.gz`: cleaned training records with IDs, text, language, and source.
- `ledger.jsonl`: exactly one disposition row per raw input record.
- `eval.jsonl`: deterministic held-out Hindi/Hinglish evaluation records.
- `contamination.jsonl`: training records removed because they matched eval content.
- `manifest.json`: source metadata, thresholds, counts, and output hashes.
- `sample_1000.jsonl`: deterministic sample of the clean training output.

The committed copies of the requested release artifacts are in [`deliverables/`](deliverables/). Raw data and generated clean output are excluded from Git.

## Design decisions

See [`DECISIONS.md`](DECISIONS.md) for the run-specific funnel, source licenses and sizes, near-duplicate threshold rationale, filter examples, contamination policy, reproducibility recipe, 2 TB scaling plan, and wrap-up.
