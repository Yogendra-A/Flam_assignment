"""
filters.py — Quality filters for Hindi NLP pipeline.

Drops:
  - Garbled encoding (replacement chars, excessive control chars)
  - Heavy HTML/markup boilerplate
  - URL/spam text
  - Repetitive n-gram loops

Preserves deliberately:
  - Short proverbs/poems (>= 15 chars, high alpha ratio, no markup)
  - Structured numeric tables (tabs/pipes/newlines with numbers)
"""
import re
import unicodedata
from typing import Tuple

# Regexes
HTML_TAG_RE = re.compile(r'<[^>]{1,100}>')
URL_RE = re.compile(r'https?://\S+|www\.\S+')
CONTROL_RE = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]')
REPLACEMENT_RE = re.compile(r'\ufffd')

def _alpha_ratio(text: str) -> float:
    if not text:
        return 0.0
    alpha = sum(1 for c in text if c.isalpha())
    return alpha / len(text)

def _digit_ratio(text: str) -> float:
    if not text:
        return 0.0
    digits = sum(1 for c in text if c.isdigit())
    return digits / len(text)

def _is_structured_table(text: str) -> bool:
    """Return True if text looks like a numeric table (tabs, pipes, aligned cols)."""
    lines = text.split('\n')
    if len(lines) < 3:
        return False
    pipe_lines = sum(1 for l in lines if '|' in l or '\t' in l)
    digit_lines = sum(1 for l in lines if re.search(r'\d', l))
    return pipe_lines >= len(lines) * 0.4 or (digit_lines >= len(lines) * 0.6 and len(lines) >= 5)

def _repetition_score(text: str, n: int = 4) -> float:
    """Return ratio of text covered by repeated n-grams. High = spammy."""
    words = text.split()
    if len(words) < n * 2:
        return 0.0
    ngrams = [tuple(words[i:i+n]) for i in range(len(words)-n+1)]
    if not ngrams:
        return 0.0
    from collections import Counter
    counts = Counter(ngrams)
    repeated_tokens = sum((v-1)*n for v in counts.values() if v > 1)
    return repeated_tokens / max(len(words), 1)

def _html_ratio(text: str) -> float:
    tags = HTML_TAG_RE.findall(text)
    tag_chars = sum(len(t) for t in tags)
    return tag_chars / max(len(text), 1)

def _url_density(text: str) -> float:
    urls = URL_RE.findall(text)
    url_chars = sum(len(u) for u in urls)
    return url_chars / max(len(text), 1)

def apply_filters(text: str, min_chars: int = 15) -> Tuple[bool, str]:
    """
    Apply quality filters.
    
    Returns:
        (keep: bool, reason: str)
        reason is 'ok' if kept, else a short reason code
    """
    if not text or not text.strip():
        return False, 'empty'
    
    text_stripped = text.strip()
    length = len(text_stripped)
    
    # 1. Garbled encoding — replacement characters
    replacement_count = len(REPLACEMENT_RE.findall(text_stripped))
    if replacement_count > 3 or (length > 0 and replacement_count / length > 0.01):
        return False, 'garbled_encoding'
    
    # 2. Excessive control characters
    control_count = len(CONTROL_RE.findall(text_stripped))
    if control_count > 5 or (length > 10 and control_count / length > 0.05):
        return False, 'control_chars'
    
    # 3. Too short — but allow short proverbs/poems
    if length < min_chars:
        # Allow if high alpha ratio and no markup
        alpha_r = _alpha_ratio(text_stripped)
        has_markup = HTML_TAG_RE.search(text_stripped) or URL_RE.search(text_stripped)
        if alpha_r >= 0.6 and not has_markup and length >= 5:
            pass  # Allow short poetic content
        else:
            return False, 'too_short'
    
    # 4. HTML boilerplate
    html_r = _html_ratio(text_stripped)
    if html_r > 0.15:
        return False, 'html_boilerplate'
    
    # 5. URL spam
    url_d = _url_density(text_stripped)
    if url_d > 0.3:
        return False, 'url_spam'
    
    # 6. High digit ratio — but exempt structured tables
    digit_r = _digit_ratio(text_stripped)
    if digit_r > 0.4:
        if _is_structured_table(text_stripped):
            pass  # Preserve numeric tables
        else:
            return False, 'high_digit_ratio'
    
    # 7. Repetitive n-gram loops (spam)
    rep_score = _repetition_score(text_stripped)
    if rep_score > 0.3:
        return False, 'repetitive_ngrams'
    
    # 8. Mostly non-alphabetic (symbols, numbers, punctuation)
    alpha_r = _alpha_ratio(text_stripped)
    if alpha_r < 0.3 and length > 50:
        # Allow if structured table
        if _is_structured_table(text_stripped):
            pass
        else:
            return False, 'low_alpha_ratio'
    
    return True, 'ok'


if __name__ == '__main__':
    tests = [
        ('यह एक सामान्य हिंदी वाक्य है।', True),
        ('\ufffd\ufffd\ufffd garbled text \ufffd\ufffd', False),
        ('<html><body><p>Heavy HTML markup</p></body></html>' * 5, False),
        ('सत्यमेव जयते', True),  # Short proverb — should keep
        ('1\t2\t3\n4\t5\t6\n7\t8\t9\n10\t11\t12\n13\t14\t15', True),  # Table
        ('spam spam spam spam spam spam spam spam spam', False),  # Repetitive
    ]
    for text, expected_keep in tests:
        keep, reason = apply_filters(text)
        status = 'PASS' if keep == expected_keep else f'FAIL expected_keep={expected_keep}'
        print(f'{status} ({reason}) {ascii(text[:50])}')
