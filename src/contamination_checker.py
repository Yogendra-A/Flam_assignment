"""
contamination_checker.py — Eval holdout and 4-tier contamination checker.

Eval holdout:
  - ~200 items: 150 hi + 50 hi-Latn, determined by hash(id) % N

Contamination tiers:
  1. exact       — raw SHA-256 match
  2. normalized  — post-normalization SHA-256
  3. near_dup    — MinHash Jaccard >= 0.70
  4. cross_script — Devanagari → phonetic Latin skeleton, char 4-gram Jaccard >= 0.60
"""
import hashlib
import re
from typing import List, Dict, Optional, Tuple, Iterator

try:
    from datasketch import MinHash, MinHashLSH
    HAS_DATASKETCH = True
except ImportError:
    HAS_DATASKETCH = False


# ─── Devanagari → Latin phonetic mapping ─────────────────────────────────────
# Coarse consonant skeleton: map Devanagari consonants to approximate Latin
DEVA_TO_LATIN = {
    'क': 'k', 'ख': 'kh', 'ग': 'g', 'घ': 'gh', 'ङ': 'ng',
    'च': 'ch', 'छ': 'chh', 'ज': 'j', 'झ': 'jh', 'ञ': 'ny',
    'ट': 't', 'ठ': 'th', 'ड': 'd', 'ढ': 'dh', 'ण': 'n',
    'त': 't', 'थ': 'th', 'द': 'd', 'ध': 'dh', 'न': 'n',
    'प': 'p', 'फ': 'ph', 'ब': 'b', 'भ': 'bh', 'म': 'm',
    'य': 'y', 'र': 'r', 'ल': 'l', 'व': 'v', 'श': 'sh',
    'ष': 'sh', 'स': 's', 'ह': 'h', 'ळ': 'l', 'क्ष': 'ksh',
    'त्र': 'tr', 'ज्ञ': 'gny',
    # Vowels (coarse)
    'अ': 'a', 'आ': 'aa', 'इ': 'i', 'ई': 'ii', 'उ': 'u',
    'ऊ': 'uu', 'ए': 'e', 'ऐ': 'ai', 'ओ': 'o', 'औ': 'au',
    'ऋ': 'ri',
}

def _deva_to_skeleton(text: str) -> str:
    """Convert Devanagari text to a coarse phonetic Latin consonant skeleton."""
    result = []
    i = 0
    while i < len(text):
        # Try 2-char combinations first
        if i + 1 < len(text):
            two = text[i:i+2]
            if two in DEVA_TO_LATIN:
                result.append(DEVA_TO_LATIN[two])
                i += 2
                continue
        ch = text[i]
        if ch in DEVA_TO_LATIN:
            result.append(DEVA_TO_LATIN[ch])
        elif '\u0900' <= ch <= '\u097F':
            # Unknown Devanagari char — keep as placeholder
            pass
        else:
            # Latin/ASCII — keep
            if ch.isalpha():
                result.append(ch.lower())
        i += 1
    return ''.join(result)


def _char_ngrams(text: str, n: int = 4):
    for i in range(len(text) - n + 1):
        yield text[i:i+n]


def _make_minhash(text: str, num_perm: int = 128) -> Optional[object]:
    if not HAS_DATASKETCH:
        return None
    m = MinHash(num_perm=num_perm)
    # Bound work for long Wikipedia articles while keeping the prefix
    # deterministic; the LSH is a candidate generator, not the final score.
    for i, ng in enumerate(_char_ngrams(text, 4)):
        if i >= 256:
            break
        m.update(ng.encode('utf-8'))
    return m


def _jaccard_minhash(m1, m2) -> float:
    if m1 is None or m2 is None:
        return 0.0
    return m1.jaccard(m2)


def _char4_jaccard(a: str, b: str) -> float:
    """Exact char-4gram Jaccard (for small strings)."""
    if not a or not b:
        return 0.0
    set_a = set(_char_ngrams(a, 4))
    set_b = set(_char_ngrams(b, 4))
    if not set_a or not set_b:
        return 0.0
    inter = len(set_a & set_b)
    union = len(set_a | set_b)
    return inter / union if union > 0 else 0.0


def should_holdout(doc_id: str, lang: str, hi_budget: int = 150, hilatn_budget: int = 50) -> bool:
    """
    Deterministically decide whether a doc goes to eval holdout.
    Uses hash(id) modulo to get ~target number of docs.
    """
    h = int(hashlib.sha256(doc_id.encode()).hexdigest(), 16)
    if lang == 'hi' and (h % 80) == 0:  # ~1/80 of hi docs → ~150 from 12000
        return True
    if lang == 'hi-Latn' and (h % 40) == 0:  # ~1/40 of hi-Latn → ~50
        return True
    return False


class ContaminationChecker:
    """
    4-tier contamination checker.

    Usage:
        checker = ContaminationChecker()
        # First, build eval index from eval.jsonl docs
        for doc in eval_docs:
            checker.add_eval(doc)
        # Then check training docs
        for doc in train_docs:
            result = checker.check_train(doc)
    """

    def __init__(self, near_dup_threshold: float = 0.70, cross_script_threshold: float = 0.60):
        self.near_dup_threshold = near_dup_threshold
        self.cross_script_threshold = cross_script_threshold

        # Eval index structures
        self.eval_exact_hashes: Dict[str, str] = {}       # raw_hash → eval_id
        self.eval_norm_hashes: Dict[str, str] = {}        # norm_hash → eval_id
        self.eval_minhashes: Dict[str, object] = {}       # eval_id → minhash
        self.eval_skeletons: Dict[str, Tuple[str, object]] = {}  # eval_id → (skeleton, minhash)
        self.eval_docs: Dict[str, dict] = {}               # eval_id → doc
        if HAS_DATASKETCH:
            self.near_lsh = MinHashLSH(threshold=near_dup_threshold, num_perm=128)
            self.near_border_lsh = MinHashLSH(threshold=0.50, num_perm=128)
            self.cross_lsh = MinHashLSH(threshold=cross_script_threshold, num_perm=128)
            self.cross_border_lsh = MinHashLSH(threshold=0.40, num_perm=128)
        else:
            self.near_lsh = self.near_border_lsh = None
            self.cross_lsh = self.cross_border_lsh = None

        # Results
        self.contamination_records = []  # {eval_id, train_ids, method}
        self.rejected_borderline = []    # {eval_id, train_id, method, score}

    def add_eval(self, doc: dict):
        """Add an eval document to the index."""
        doc_id = doc['id']
        text = doc['text']
        norm_text = doc.get('norm_text', text)
        lang = doc.get('lang', 'hi')

        self.eval_docs[doc_id] = doc

        # Tier 1: raw hash
        raw_h = hashlib.sha256(text.encode()).hexdigest()
        self.eval_exact_hashes[raw_h] = doc_id

        # Tier 2: normalized hash
        norm_h = hashlib.sha256(norm_text.encode()).hexdigest()
        self.eval_norm_hashes[norm_h] = doc_id

        # Tier 3: MinHash
        m = _make_minhash(text)
        if m:
            self.eval_minhashes[doc_id] = m
            self.near_lsh.insert(doc_id, m)
            self.near_border_lsh.insert(doc_id, m)

        # Tier 4: Cross-script skeleton (only for hi docs)
        if lang == 'hi':
            skeleton = _deva_to_skeleton(text)
            sm = _make_minhash(skeleton) if skeleton else None
            self.eval_skeletons[doc_id] = (skeleton, sm)
            if sm:
                self.cross_lsh.insert(doc_id, sm)
                self.cross_border_lsh.insert(doc_id, sm)

    def check_train(self, doc: dict) -> Optional[dict]:
        """
        Check a training doc for contamination against all eval docs.

        Returns contamination record if found, else None.
        """
        doc_id = doc['id']
        text = doc['text']
        norm_text = doc.get('norm_text', text)
        lang = doc.get('lang', 'other')

        # Tier 1: Exact (raw SHA-256)
        raw_h = hashlib.sha256(text.encode()).hexdigest()
        if raw_h in self.eval_exact_hashes:
            eval_id = self.eval_exact_hashes[raw_h]
            return {'eval_id': eval_id, 'train_ids': [doc_id], 'method': 'exact'}

        # Tier 2: Normalized SHA-256
        norm_h = hashlib.sha256(norm_text.encode()).hexdigest()
        if norm_h in self.eval_norm_hashes:
            eval_id = self.eval_norm_hashes[norm_h]
            return {'eval_id': eval_id, 'train_ids': [doc_id], 'method': 'normalized'}

        # Tier 3: Near-dup MinHash
        train_m = _make_minhash(text)
        if train_m and self.eval_minhashes:
            best_j = 0.0
            best_eval_id = None
            candidates = set(self.near_lsh.query(train_m)) if self.near_lsh else set()
            borderline_candidates = set(self.near_border_lsh.query(train_m)) if self.near_border_lsh else set()
            for eval_id in candidates:
                eval_m = self.eval_minhashes[eval_id]
                j = _jaccard_minhash(train_m, eval_m)
                if j >= self.near_dup_threshold:
                    if j > best_j:
                        best_j = j
                        best_eval_id = eval_id
            for eval_id in borderline_candidates - candidates:
                eval_m = self.eval_minhashes[eval_id]
                j = _jaccard_minhash(train_m, eval_m)
                if 0.50 <= j < self.near_dup_threshold:
                    self.rejected_borderline.append({
                        'eval_id': eval_id, 'train_id': doc_id,
                        'method': 'near_dup', 'score': round(j, 4)
                    })
            if best_eval_id:
                return {'eval_id': best_eval_id, 'train_ids': [doc_id], 'method': 'near_dup'}

        # Tier 4: Cross-script (hi-Latn train vs hi eval skeleton)
        if lang == 'hi-Latn' and self.eval_skeletons:
            # Map train text to lowercase
            train_latin = text.lower()
            train_m_skel = _make_minhash(train_latin)

            best_j = 0.0
            best_eval_id = None
            candidates = set(self.cross_lsh.query(train_m_skel)) if self.cross_lsh and train_m_skel else set()
            borderline_candidates = set(self.cross_border_lsh.query(train_m_skel)) if self.cross_border_lsh and train_m_skel else set()
            for eval_id in candidates:
                skel, eval_skel_m = self.eval_skeletons[eval_id]
                if not skel:
                    continue
                if train_m_skel and eval_skel_m:
                    j = _jaccard_minhash(train_m_skel, eval_skel_m)
                else:
                    j = _char4_jaccard(train_latin, skel)

                if j >= self.cross_script_threshold:
                    if j > best_j:
                        best_j = j
                        best_eval_id = eval_id
            for eval_id in borderline_candidates - candidates:
                skel, eval_skel_m = self.eval_skeletons[eval_id]
                if not skel:
                    continue
                j = _jaccard_minhash(train_m_skel, eval_skel_m) if train_m_skel and eval_skel_m else _char4_jaccard(train_latin, skel)
                if 0.40 <= j < self.cross_script_threshold:
                    self.rejected_borderline.append({
                        'eval_id': eval_id, 'train_id': doc_id,
                        'method': 'cross_script', 'score': round(j, 4)
                    })
            if best_eval_id:
                return {'eval_id': best_eval_id, 'train_ids': [doc_id], 'method': 'cross_script'}

        return None


if __name__ == '__main__':
    # Quick test
    checker = ContaminationChecker()

    eval_doc = {
        'id': 'eval_1',
        'text': 'यह एक परीक्षण वाक्य है।',
        'lang': 'hi'
    }
    checker.add_eval(eval_doc)

    # Exact match
    r = checker.check_train({'id': 'train_1', 'text': 'यह एक परीक्षण वाक्य है।', 'lang': 'hi'})
    print(f"Exact match: {r}")

    # No match
    r = checker.check_train({'id': 'train_2', 'text': 'बिल्कुल अलग content है।', 'lang': 'hi'})
    print(f"No match: {r}")

    # holdout test
    print(f"holdout hi: {should_holdout('wiki_hi_100', 'hi')}")
    print(f"holdout hi-Latn: {should_holdout('hinglish_100', 'hi-Latn')}")
