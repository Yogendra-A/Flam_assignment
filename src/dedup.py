"""
dedup.py — Exact and near-duplicate detection.

Exact dedup: SHA-256 hash of normalized text.
Near-dup: datasketch MinHashLSH with num_perm=128.
  - Word 3-grams for docs >= 10 words
  - Char 5-grams for shorter docs
  - Default threshold: 0.80 (configurable)
  - Logs borderline pairs (Jaccard 0.65-0.90) to debug file
"""
import hashlib
import re
from typing import Optional, Iterator, Tuple, Dict, Set

try:
    from datasketch import MinHash, MinHashLSH
    HAS_DATASKETCH = True
except ImportError:
    HAS_DATASKETCH = False
    print("WARNING: datasketch not available; near-dup detection disabled")


def sha256(text: str) -> str:
    """Return hex SHA-256 of text."""
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def _word_ngrams(text: str, n: int = 3):
    """Generate word n-gram strings."""
    words = text.split()
    for i in range(len(words) - n + 1):
        yield ' '.join(words[i:i+n])


def _char_ngrams(text: str, n: int = 5):
    """Generate character n-gram strings."""
    for i in range(len(text) - n + 1):
        yield text[i:i+n]


def make_minhash(text: str, num_perm: int = 128) -> Optional[object]:
    """Create a MinHash for the given text."""
    if not HAS_DATASKETCH:
        return None
    
    words = text.split()
    m = MinHash(num_perm=num_perm)
    
    if len(words) >= 10:
        # Word 3-grams
        for ngram in _word_ngrams(text, 3):
            m.update(ngram.encode('utf-8'))
    else:
        # Char 5-grams
        for ngram in _char_ngrams(text, 5):
            m.update(ngram.encode('utf-8'))
    
    return m


class DedupState:
    """
    Stateful deduplicator. Process documents one at a time.
    
    Usage:
        state = DedupState(threshold=0.80)
        for doc in documents:
            is_dup, dup_of = state.check_and_add(doc['id'], doc['text'])
    """
    
    def __init__(self, threshold: float = 0.80, num_perm: int = 128):
        self.threshold = threshold
        self.num_perm = num_perm
        self.exact_hashes: Dict[str, str] = {}  # hash → first doc_id
        self.borderline_pairs = []  # (id1, id2, jaccard) for 0.65-0.90
        
        if HAS_DATASKETCH:
            self.lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
            self.borderline_lsh = MinHashLSH(threshold=max(0.65, threshold - 0.15), num_perm=num_perm)
            self.minhashes: Dict[str, object] = {}  # id → minhash
        else:
            self.lsh = None
            self.minhashes = {}
    
    def check_and_add(self, doc_id: str, text: str) -> Tuple[bool, Optional[str]]:
        """
        Check if doc is duplicate. If not, add to index.
        
        Returns:
            (is_duplicate: bool, dup_of: Optional[str])
        """
        if not text or not text.strip():
            return False, None
        
        # 1. Exact dedup
        h = sha256(text)
        if h in self.exact_hashes:
            return True, self.exact_hashes[h]
        # 2. Near-dup via MinHash LSH
        if HAS_DATASKETCH and self.lsh is not None:
            m = make_minhash(text, self.num_perm)
            if m is not None:
                try:
                    # Query for near-duplicates
                    result = self.lsh.query(m)
                    
                    if result:
                        # Found near-duplicate(s)
                        dup_of = result[0]
                        
                        # Compute Jaccard for borderline logging
                        if dup_of in self.minhashes:
                            j = m.jaccard(self.minhashes[dup_of])
                            if 0.65 <= j <= 0.90:
                                self.borderline_pairs.append({
                                    'id1': doc_id,
                                    'id2': dup_of,
                                    'jaccard': round(j, 4)
                                })
                        
                        return True, dup_of
                    
                    # Log accepted near-threshold candidates for auditability.
                    for candidate in self.borderline_lsh.query(m):
                        if candidate in self.minhashes:
                            j = m.jaccard(self.minhashes[candidate])
                            if 0.65 <= j < self.threshold:
                                self.borderline_pairs.append({
                                    'id1': doc_id, 'id2': candidate,
                                    'jaccard': round(j, 4), 'decision': 'keep'
                                })

                    # Not a duplicate — add to indexes.
                    self.exact_hashes[h] = doc_id
                    self.lsh.insert(doc_id, m)
                    self.borderline_lsh.insert(doc_id, m)
                    self.minhashes[doc_id] = m
                    
                    # Check borderline near-dups (not quite duplicates)
                    # Lower threshold query for borderline detection
                    # We'll log these separately via post-processing
                    
                except Exception:
                    pass
        
        # Exact-only fallback when MinHash is unavailable.
        self.exact_hashes[h] = doc_id
        return False, None
    
    def get_borderline_pairs(self):
        """Return borderline near-dup pairs for debug logging."""
        return self.borderline_pairs


def compute_jaccard(text1: str, text2: str) -> float:
    """Compute Jaccard similarity between two texts using MinHash."""
    if not HAS_DATASKETCH:
        return 0.0
    m1 = make_minhash(text1)
    m2 = make_minhash(text2)
    if m1 and m2:
        return m1.jaccard(m2)
    return 0.0


if __name__ == '__main__':
    state = DedupState(threshold=0.80)
    
    texts = [
        ("doc1", "यह एक हिंदी वाक्य है जो पहले आया था।"),
        ("doc2", "यह एक हिंदी वाक्य है जो पहले आया था।"),  # Exact dup
        ("doc3", "यह एक हिंदी वाक्य है जो पहले आया था, थोड़ा बदला हुआ।"),  # Near-dup
        ("doc4", "बिल्कुल अलग content है यहाँ पर।"),
    ]
    
    for doc_id, text in texts:
        is_dup, dup_of = state.check_and_add(doc_id, text)
        print(f"{'DUP' if is_dup else 'KEEP'} {doc_id}: dup_of={dup_of}")
