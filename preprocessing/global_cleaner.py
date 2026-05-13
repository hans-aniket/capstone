"""
preprocessing/global_cleaner.py
────────────────────────────────
Shared cleaning utilities applied to ALL model tiers before the
model-specific preprocessing paths diverge.

Rules:
  - Only structural noise is removed (HTML, whitespace, encoding)
  - No lowercasing, stemming, stop-word removal, or punctuation stripping
  - Safe for classical ML, deep learning, AND transformers
"""

import re
import html
import unicodedata

import pandas as pd

from preprocessing.config import MIN_TEXT_LENGTH

# ── Compiled patterns (module-level for performance) ──────────────────────────
_HTML_TAG_RE    = re.compile(r"<[^>]+>")
_HTML_ENTITY_RE = re.compile(r"&[a-zA-Z]{2,6};|&#\d+;")
_WHITESPACE_RE  = re.compile(r"\s+")


# ── Individual cleaning steps ─────────────────────────────────────────────────

def strip_html(text: str) -> str:
    """Remove HTML tags and decode HTML entities."""
    text = _HTML_TAG_RE.sub(" ", text)
    text = html.unescape(text)
    text = _HTML_ENTITY_RE.sub(" ", text)
    return text


def normalize_whitespace(text: str) -> str:
    """Collapse all whitespace sequences (tabs, newlines, spaces) to a single space."""
    return _WHITESPACE_RE.sub(" ", text).strip()


def normalize_unicode(text: str) -> str:
    """NFC normalization: unify composed vs. decomposed character forms."""
    return unicodedata.normalize("NFC", text)


def is_valid(text: str, min_len: int = MIN_TEXT_LENGTH) -> bool:
    """Return True if the text has enough content to be informative."""
    return isinstance(text, str) and len(text.strip()) >= min_len


# ── Composed pipeline ─────────────────────────────────────────────────────────

def clean(text: str, min_len: int = MIN_TEXT_LENGTH) -> str | None:
    """
    Apply the global cleaning pipeline to a single string.

    Returns:
        Cleaned string if valid, or None if the text is too short after cleaning.

    Safe for all model tiers — does NOT alter semantics.
    """
    if not isinstance(text, str):
        return None
    text = strip_html(text)
    text = normalize_unicode(text)
    text = normalize_whitespace(text)
    return text if is_valid(text, min_len) else None


def clean_series(series: pd.Series, min_len: int = MIN_TEXT_LENGTH) -> pd.Series:
    """
    Vectorized global cleaning for a pandas Series.
    Rows that fail the minimum length check are set to None.
    """
    return series.map(lambda t: clean(t, min_len))


def clean_dataframe(
    df: pd.DataFrame,
    text_col: str = "text",
    min_len: int  = MIN_TEXT_LENGTH,
    drop_invalid: bool = True,
) -> pd.DataFrame:
    """
    Apply global cleaning to a DataFrame's text column.

    Args:
        df:           Input DataFrame with at least a text column and a label column.
        text_col:     Name of the text column.
        min_len:      Minimum character length after cleaning.
        drop_invalid: If True, drop rows where cleaning yields None.

    Returns:
        DataFrame with cleaned text column (and invalid rows removed if requested).
    """
    df = df.copy()
    df[text_col] = clean_series(df[text_col], min_len)
    if drop_invalid:
        before = len(df)
        df = df.dropna(subset=[text_col]).reset_index(drop=True)
        dropped = before - len(df)
        if dropped:
            print(f"[GlobalCleaner] Dropped {dropped} near-empty rows.")
    return df
