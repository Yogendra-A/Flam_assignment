"""
normalize.py — Devanagari-aware Unicode normalization.

Rules:
- NFC normalization
- Strip BOM (\ufeff), zero-width space (\u200b), soft hyphen (\u00ad), control chars (except \t \n \r)
- PRESERVE ZWJ (\u200d) and ZWNJ (\u200c) when adjacent to Devanagari (\u0900-\u097F)
- Strip ZWJ/ZWNJ elsewhere
- Normalize curly quotes → straight quotes, em/en dashes → hyphen
- Collapse excess whitespace (multiple spaces → single, multiple newlines → max 2)
"""
import re, unicodedata

DEVANAGARI = re.compile(r'[\u0900-\u097F]')

def _has_devanagari_neighbor(text, i):
    """Check if position i has a Devanagari char within 1 position."""
    for j in (i-1, i+1):
        if 0 <= j < len(text) and DEVANAGARI.match(text[j]):
            return True
    return False

def normalize(text: str) -> str:
    if not text:
        return text
    
    # NFC normalization
    text = unicodedata.normalize('NFC', text)
    
    # Strip BOM
    text = text.lstrip('\ufeff')
    
    # Handle ZWJ (\u200d) and ZWNJ (\u200c) — preserve near Devanagari
    result = []
    for i, ch in enumerate(text):
        if ch in ('\u200d', '\u200c'):
            if _has_devanagari_neighbor(text, i):
                result.append(ch)
            # else skip
        elif ch == '\u200b':  # zero-width space — always strip
            pass
        elif ch == '\u00ad':  # soft hyphen — always strip
            pass
        elif ch == '\ufeff':  # BOM — always strip
            pass
        elif unicodedata.category(ch).startswith('C') and ch not in ('\t', '\n', '\r'):
            # Control characters — strip
            pass
        else:
            result.append(ch)
    text = ''.join(result)
    
    # Normalize quotes and dashes
    text = text.replace('\u2018', "'").replace('\u2019', "'")  # curly single quotes
    text = text.replace('\u201c', '"').replace('\u201d', '"')  # curly double quotes
    text = text.replace('\u2014', '-').replace('\u2013', '-')  # em dash, en dash
    text = text.replace('\u2026', '...')  # ellipsis
    
    # Collapse excess whitespace
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = text.strip()
    
    return text


if __name__ == '__main__':
    # Quick self-test
    tests = [
        ('\ufeffहिंदी\u200btext', 'हिंदी text'),
        ('hello\u200dworld', 'helloworld'),  # ZWJ not near Devanagari → strip
        ('क\u200dख', 'क\u200dख'),  # ZWJ between Devanagari → keep
        ('"hello"', '"hello"'),
    ]
    for inp, expected in tests:
        got = normalize(inp)
        status = 'PASS' if got == expected else f'FAIL got={repr(got)}'
        print(f'{status} normalize({ascii(inp)[:40]})')
