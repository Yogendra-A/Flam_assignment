"""
pii_redactor.py — Indian PII redaction.

Redacts:
  - Indian phone numbers → <PHONE>
  - Email addresses → <EMAIL>
  - PAN card numbers → <PAN>
  - Aadhaar numbers (Verhoeff-validated) → <AADHAAR>
"""
import re
from typing import Tuple


# ─── Verhoeff checksum tables ───────────────────────────────────────────────
# Multiplication table
_D = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    [1, 2, 3, 4, 0, 6, 7, 8, 9, 5],
    [2, 3, 4, 0, 1, 7, 8, 9, 5, 6],
    [3, 4, 0, 1, 2, 8, 9, 5, 6, 7],
    [4, 0, 1, 2, 3, 9, 5, 6, 7, 8],
    [5, 9, 8, 7, 6, 0, 4, 3, 2, 1],
    [6, 5, 9, 8, 7, 1, 0, 4, 3, 2],
    [7, 6, 5, 9, 8, 2, 1, 0, 4, 3],
    [8, 7, 6, 5, 9, 3, 2, 1, 0, 4],
    [9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
]
# Permutation table
_P = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    [1, 5, 7, 6, 2, 8, 3, 0, 9, 4],
    [5, 8, 0, 3, 7, 9, 6, 1, 4, 2],
    [8, 9, 1, 6, 0, 4, 3, 5, 2, 7],
    [9, 4, 5, 3, 1, 2, 6, 8, 7, 0],
    [4, 2, 8, 6, 5, 7, 3, 9, 0, 1],
    [2, 7, 9, 3, 8, 0, 6, 4, 1, 5],
    [7, 0, 4, 6, 9, 1, 3, 2, 5, 8],
]
# Inverse table
_INV = [0, 4, 3, 2, 1, 5, 6, 7, 8, 9]


def _verhoeff_check(number: str) -> bool:
    """Return True if the number string passes Verhoeff checksum."""
    try:
        digits = [int(d) for d in reversed(number)]
        c = 0
        for i, d in enumerate(digits):
            c = _D[c][_P[i % 8][d]]
        return c == 0
    except (ValueError, IndexError):
        return False


# ─── Regex patterns ──────────────────────────────────────────────────────────

# Indian phone: +91-XXXXXXXXXX, 0XXXXXXXXXX, +91 XXXXX XXXXX, 10 digits
# Must be at word boundary
PHONE_RE = re.compile(
    r'(?<!\d)'
    r'(?:'
    r'(?:\+91[-\s]?)?'          # optional country code
    r'(?:0)?'                    # optional leading 0
    r'[6-9]\d{9}'               # Indian mobile: starts 6-9, total 10 digits
    r'|'
    r'\+91[-\s]?\d{5}[-\s]?\d{5}'  # +91 XXXXX XXXXX
    r')'
    r'(?!\d)',
    re.ASCII
)

# Email
EMAIL_RE = re.compile(
    r'\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b',
    re.ASCII
)

# PAN: 5 uppercase letters + 4 digits + 1 uppercase letter
PAN_RE = re.compile(
    r'\b[A-Z]{5}[0-9]{4}[A-Z]\b',
    re.ASCII
)

# Aadhaar: 12 digits (possibly space/dash-separated: XXXX XXXX XXXX)
# We'll capture candidates and validate with Verhoeff
AADHAAR_RE = re.compile(
    r'(?<!\d)'
    r'(?:'
    r'\d{4}[-\s]\d{4}[-\s]\d{4}'  # XXXX XXXX XXXX
    r'|\d{12}'                      # XXXXXXXXXXXX
    r')'
    r'(?!\d)',
    re.ASCII
)


def _redact_aadhaar(text: str) -> str:
    """Redact Aadhaar numbers validated by Verhoeff checksum."""
    def replace(m):
        raw = m.group(0)
        digits = re.sub(r'[\s\-]', '', raw)
        if len(digits) == 12 and _verhoeff_check(digits):
            return '<AADHAAR>'
        return raw  # Not valid Aadhaar — keep as-is

    return AADHAAR_RE.sub(replace, text)


def redact(text: str) -> Tuple[str, dict]:
    """
    Redact PII from text.

    Returns:
        (redacted_text, counts_dict)
    """
    counts = {'phone': 0, 'email': 0, 'pan': 0, 'aadhaar': 0}

    if not text:
        return text, counts

    # Email first (before phone, to avoid partial matches)
    def count_email(m):
        counts['email'] += 1
        return '<EMAIL>'
    text = EMAIL_RE.sub(count_email, text)

    # Phone
    def count_phone(m):
        counts['phone'] += 1
        return '<PHONE>'
    text = PHONE_RE.sub(count_phone, text)

    # PAN
    def count_pan(m):
        counts['pan'] += 1
        return '<PAN>'
    text = PAN_RE.sub(count_pan, text)

    # Aadhaar (with Verhoeff validation)
    text = _redact_aadhaar(text)
    # Count replacements
    counts['aadhaar'] = text.count('<AADHAAR>')

    return text, counts


if __name__ == '__main__':
    test_cases = [
        # (input, expected_to_contain)
        ('Call me at 9876543210 for details', '<PHONE>'),
        ('Email: user@example.com for info', '<EMAIL>'),
        ('PAN: ABCDE1234F is required', '<PAN>'),
        # Aadhaar with valid Verhoeff checksum — this is a known valid test number
        # 1234 5678 9012 — may or may not pass Verhoeff
        ('My phone is 1234567890 (not Indian mobile, starts with 1)', '1234567890'),  # Should NOT be redacted
        ('Random 12-digit: 123456789012', '123456789012'),  # Should NOT be redacted (Verhoeff fails)
        ('The year 2023 and amount 500', '2023'),  # Short numbers — not PII
    ]

    for text, expected_contains in test_cases:
        result, counts = redact(text)
        contains = expected_contains in result
        print(f"{'PASS' if contains else 'FAIL'} {repr(text[:60])} -> {repr(result[:60])} counts={counts}")
