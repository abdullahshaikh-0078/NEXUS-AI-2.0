from __future__ import annotations

import re

WHITESPACE_RE = re.compile(r"[ \t]+")
BLANK_LINE_RE = re.compile(r"\n{3,}")


def clean_text(text: str) -> str:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n").replace("\x00", " ")
    normalized = WHITESPACE_RE.sub(" ", normalized)
    normalized = BLANK_LINE_RE.sub("\n\n", normalized)
    return normalized.strip()

