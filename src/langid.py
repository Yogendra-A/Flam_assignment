"""
langid.py — Offline language classifier.

Classes:
  hi      — Devanagari-dominant text (including Devanagari with English words)
  hi-Latn — Latin-script Hindi / Hinglish (code-mixed)
  en      — English
  other   — Everything else

Method:
  1. Count script ratios (Devanagari, Latin, other)
  2. If Devanagari ratio > 0.3 → hi
  3. If Latin ratio > 0.6:
     - Check against ~120-word Hinglish function-word lexicon
     - If Hinglish score > threshold → hi-Latn
     - Else if Latin text looks English → en
  4. Otherwise → other
"""
import re
from functools import lru_cache

# Devanagari Unicode block
DEVANAGARI_RE = re.compile(r'[\u0900-\u097F]')
LATIN_RE = re.compile(r'[A-Za-z]')

# ~120-word Hinglish function-word lexicon (common Hindi words in Latin script)
HINGLISH_LEXICON = frozenset([
    # Hindi pronouns / particles in Roman
    'mai', 'mujhe', 'mera', 'meri', 'mere', 'hum', 'hamara', 'hamari',
    'tu', 'tum', 'tumhara', 'aap', 'aapka', 'aapki', 'woh', 'wo', 'unka',
    'unki', 'yeh', 'ye', 'iska', 'iski', 'iske', 'uska', 'uski', 'uske',
    # Verbs
    'hai', 'hain', 'tha', 'thi', 'ho', 'hoga', 'hogi', 'kar',
    'karo', 'karta', 'karti', 'karte', 'kiya', 'kiye', 'karenge', 'karunga',
    'ja', 'jao', 'jana', 'gaya', 'gayi', 'gaye', 'aa', 'aao', 'aana',
    'aaya', 'aayi', 'aaye', 'le', 'lo', 'lena', 'liya', 'liye', 'de',
    'dena', 'diya', 'diye', 'raha', 'rahi', 'rahe', 'baad', 'pehle',
    # Common Hindi words
    'nahi', 'nahin', 'na', 'haan', 'aur', 'par', 'lekin', 'magar', 'kyunki',
    'toh', 'to', 'bhi', 'hi', 'se', 'ko', 'ke', 'ka', 'ki', 'mein', 'pe',
    'kya', 'kyon', 'kab', 'kahan', 'kaisa', 'kaisi', 'kaise', 'kitna',
    'ek', 'do', 'teen', 'char', 'paanch', 'sab', 'kuch', 'bahut', 'thoda',
    'accha', 'achha', 'acha', 'bura', 'zyada', 'kam', 'naya', 'purana',
    'bada', 'chota', 'yaar', 'dost', 'bhai', 'behen', 'maa', 'baap',
    'ghar', 'kaam', 'din', 'raat', 'subah', 'shaam', 'kal', 'aaj', 'abhi',
    'phir', 'dobara', 'wala', 'wali', 'wale', 'waala', 'waali', 'waale',
    'matlab', 'samajh', 'pata', 'lagta', 'lagti', 'lagte',
    # Code-mix markers
    'bolta', 'bolti', 'bolte', 'bol', 'sun', 'suno', 'dekh', 'dekho',
    'chal', 'chalo', 'ruk', 'rukk', 'bas', 'theek', 'thik', 'bilkul',
    'zaroor', 'pakka', 'sach', 'jhooth', 'dhyan', 'yaad', 'khushi',
    'dukh', 'pyaar', 'mohabbat', 'dil', 'zindagi', 'duniya',
])

# Common English function words (to distinguish en from hi-Latn)
ENGLISH_FUNCTION_WORDS = frozenset([
    'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
    'should', 'may', 'might', 'shall', 'can', 'this', 'that', 'these',
    'those', 'it', 'its', 'they', 'them', 'their', 'we', 'our', 'you',
    'your', 'he', 'she', 'his', 'her', 'i', 'my', 'me', 'in', 'on',
    'at', 'by', 'for', 'with', 'about', 'from', 'to', 'of', 'and',
    'or', 'but', 'not', 'if', 'as', 'so', 'than', 'then', 'when',
    'where', 'who', 'which', 'how', 'what', 'there', 'here',
])


def _script_ratios(text: str):
    """Return (devanagari_ratio, latin_ratio, total_alpha_chars)."""
    total = 0
    deva = 0
    latin = 0
    for ch in text:
        if ch.isalpha():
            total += 1
            if '\u0900' <= ch <= '\u097F':
                deva += 1
            elif ch.isascii():
                latin += 1
    if total == 0:
        return 0.0, 0.0, 0
    return deva / total, latin / total, total


def _hinglish_score(words):
    """Fraction of words in the Hinglish lexicon."""
    if not words:
        return 0.0
    hits = sum(1 for w in words if w.lower() in HINGLISH_LEXICON)
    return hits / len(words)


def _english_score(words):
    """Fraction of words in English function-word list."""
    if not words:
        return 0.0
    hits = sum(1 for w in words if w.lower() in ENGLISH_FUNCTION_WORDS)
    return hits / len(words)


def classify(text: str) -> str:
    """Classify text into 'hi', 'hi-Latn', 'en', or 'other'.

    Args:
        text: Input string to classify.

    Returns:
        One of 'hi', 'hi-Latn', 'en', or 'other'.
    """
    if not text or not text.strip():
        return 'other'

    deva_ratio, latin_ratio, total_alpha = _script_ratios(text)

    # Devanagari-dominant → hi (even with English words mixed in)
    if deva_ratio >= 0.25:
        return 'hi'

    # Also hi if meaningful Devanagari alongside Latin (code-mixed Devanagari)
    if deva_ratio >= 0.08 and latin_ratio < 0.85:
        return 'hi'

    words = re.findall(r'[A-Za-z]+', text)

    if latin_ratio >= 0.5 or (latin_ratio > 0 and total_alpha > 0):
        hinglish = _hinglish_score(words)
        english = _english_score(words)

        # Strong Hinglish signal
        if hinglish >= 0.08:
            return 'hi-Latn'

        # Mixed signal — lean hi-Latn if more Hinglish than English
        if hinglish > english * 0.5 and hinglish >= 0.04:
            return 'hi-Latn'

        # English dominant
        if english >= 0.08 or latin_ratio >= 0.85:
            return 'en'

        # Weak latin with some Hinglish
        if hinglish >= 0.02:
            return 'hi-Latn'

    return 'other'


if __name__ == '__main__':
    tests = [
        ('यह हिंदी में एक वाक्य है।', 'hi'),
        ('यह text mixed है English के साथ।', 'hi'),
        ('Yeh bahut accha hai yaar, kya bolta hai tu?', 'hi-Latn'),
        ('The quick brown fox jumps over the lazy dog.', 'en'),
        ('中文文字测试', 'other'),
    ]
    all_pass = True
    for text, expected in tests:
        got = classify(text)
        status = 'PASS' if got == expected else f'FAIL expected={expected}'
        if got != expected:
            all_pass = False
        print(f'{status} classify({ascii(text[:40])}) -> {got}')
    import sys
    sys.exit(0 if all_pass else 1)
