from __future__ import annotations

import re
import unicodedata

SMART_CHARACTER_MAP = str.maketrans(
    {
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2013": "-",
        "\u2014": "-",
        "\u00a0": " ",
    }
)
TOKEN_PATTERN = re.compile(r"[a-z0-9_]+")
NON_ALNUM_PATTERN = re.compile(r"[^a-z0-9_\s]")
WHITESPACE_PATTERN = re.compile(r"\s+")
CONTROL_PATTERN = re.compile(r"[\x00-\x1f\x7f]")


def clean_query(query: str) -> str:
    cleaned = unicodedata.normalize("NFKC", query).translate(SMART_CHARACTER_MAP)
    cleaned = CONTROL_PATTERN.sub(" ", cleaned)
    cleaned = WHITESPACE_PATTERN.sub(" ", cleaned).strip()
    if not cleaned:
        raise ValueError("Query must not be empty after cleaning")
    return cleaned


def normalize_query(query: str, preserve_original_case: bool = False) -> str:
    normalized = query if preserve_original_case else query.lower()
    normalized = NON_ALNUM_PATTERN.sub(" ", normalized)
    normalized = WHITESPACE_PATTERN.sub(" ", normalized).strip()
    return normalized


def tokenize_query(normalized_query: str) -> list[str]:
    return TOKEN_PATTERN.findall(normalized_query)
