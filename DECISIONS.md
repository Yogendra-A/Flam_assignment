# Decisions

## 1. Sources

| Source | License | Docs | JSONL size |
|---|---|---:|---:|
| `wikimedia/wikipedia`, config `20231101.hi` | CC BY-SA 4.0 | 12,000 | 75.422 MB |
| `WillHeld/hinglish_top`, config `default` (`cs_query`) | MIT | 2,496 | 0.279 MB |

The run used 14,496 raw records dated 2026-11-20. Raw files and the clean corpus are intentionally not committed.

## 2. Funnel

Counts are documents in -> out at each stage, by classifier label.

| Stage | hi | hi-Latn | en | other |
|---|---:|---:|---:|---:|
| Language ID | 11,984 -> 11,984 | 2,375 -> 2,375 | 137 -> 0 | 0 -> 0 |
| Quality filter | 11,984 -> 11,801 | 2,375 -> 2,375 | 0 -> 0 | 0 -> 0 |
| Dedup (unique, including eval candidates) | 11,801 -> 10,214 | 2,375 -> 2,327 | 0 -> 0 | 0 -> 0 |
| Final clean training | 10,214 -> 9,436 | 2,327 -> 2,274 | 0 -> 0 | 0 -> 0 |

The remaining dedup survivors are the 191-item eval set: 141 `hi`, 50 `hi-Latn`. Final clean output has 11,710 rows before the eval split is accounted for in the ledger; `clean/data.jsonl.gz` contains 11,710 training rows.

## 3. Near-duplicates

The method is SHA-256 exact matching followed by 128-permutation MinHash over word 3-grams (or character 5-grams for short text) with LSH threshold 0.80. The threshold was chosen to remove high-overlap copies while retaining paraphrases; borderline audit candidates use the 0.65-0.80 band. Three real borderline pairs, all kept, were: `hinglish_top_204` / `hinglish_top_13` (0.7500; “7 pm ... alarm cancel” vs “6:10 ... alarm cancel”, distinct time), `hinglish_top_205` / `hinglish_top_85` (0.7109; “Please timer ... restart” vs “timer ... restart”, harmless wording variation), and `hinglish_top_216` / `hinglish_top_34` (0.7109; “umbrella chahiye?” vs “umbrella laana chahiye?”, distinct wording/intent). The run removed 1,635 near-duplicates and 0 exact duplicates.

## 4. Filters

The filter removed 183 documents. Three likely false positives are real encyclopedic Hindi articles: id `14` (daily पूजा विधि), id `1570` (महाद्वीप), and id `2462` (मंदिर), all flagged `repetitive_ngrams`; the repetition heuristic is firing on long, formulaic Wikipedia prose. Id `1570` was flagged `html_boilerplate` in the debug sample as well. These are candidates for a future article-aware exception, not silent inclusion in this release.

## 5. Contamination & eval set

The eval set measures held-out Hindi/Hinglish generalization without train/eval text overlap. Holdout is deterministic by SHA-256(id) targeting 150 Devanagari and 50 Latin-script items; this run yielded 141 and 50 after filtering/dedup. Contamination checks use exact hash, normalized hash, MinHash near-duplicate at 0.70, and cross-script Devanagari-to-Latin skeleton matching at 0.60. Cross-script indexing was enabled; 640 items were affected, all reported as `near_dup` in this run. Candidate `hinglish_top_9` (“please 50 minutes ... timer”) vs eval `hinglish_top_1072` (“2 hours ... timer”) scored 0.5078, below 0.70, so it was rejected as contamination (different duration and a borderline template match).

## 6. Reproducibility

Six months from now: use Python 3.11+ with `requirements.txt`, run `./download.sh raw_data` against the same dataset names/configs and record the source snapshot date, then run `./run.sh raw_data out_data`. The pipeline sorts filenames, uses deterministic normalization, SHA-256 holdout selection, fixed thresholds, and sample seed 42. Verify the downloaded file hashes against `deliverables/manifest.json` and compare output hashes before publishing.

## 7. Scale (2 TB)

At 2 TB, the in-memory exact-hash/MinHash-LSH state is the first thing that breaks: RAM pressure and Python object overhead will OOM before the streaming reader does. Replace it with sharded on-disk hashes, partitioned MinHash/LSH indexes (or an ANN service), and a two-pass external-sort/merge workflow; keep only bounded candidate windows in memory.

## 8. Wrap-up

Not completed, in priority order: (1) improve article-aware quality filtering to recover valid long Wikipedia prose, (2) add a larger manually reviewed cross-script benchmark, (3) replace the in-memory indexes with a disk-backed 2 TB implementation, and (4) add CI tests for download-schema drift. Actual time spent: approximately 3 hours.
