"""Conservative text cleaning: fixes extraction artifacts without discarding content."""
from __future__ import annotations

import re
import unicodedata

# Control chars (except \n and \t), zero-width chars, BOM and soft hyphens.
_JUNK = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f\u200b\u200c\u200d\u2060\ufeff\u00ad]")
_HYPHEN_BREAK = re.compile(r"([a-z])-\n([a-z])")  # "connec-\ntion" -> "connection"
_PAGE_NUMBER_LINE = re.compile(r"^(page\s+)?\d{1,4}(\s*(of|/)\s*\d{1,4})?$", re.IGNORECASE)
_SPACES = re.compile(r"[ \t]+")


def clean_text(text: str) -> str:
    """Normalise a page/section of raw text.

    * NFKC normalisation (turns ligatures like "ﬁ" into "fi", non-breaking spaces into spaces)
    * unified line breaks, control/zero-width characters removed
    * words split by end-of-line hyphenation are re-joined
    * lines that are only a page number are dropped
    * single line breaks inside a paragraph become spaces; blank lines stay paragraph breaks
    """
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _JUNK.sub("", text)
    text = _HYPHEN_BREAK.sub(r"\1\2", text)

    paragraphs: list[str] = []
    current: list[str] = []
    for line in text.split("\n"):
        line = _SPACES.sub(" ", line).strip()
        if line and _PAGE_NUMBER_LINE.match(line):
            continue
        if line:
            current.append(line)
        elif current:
            paragraphs.append(" ".join(current))
            current = []
    if current:
        paragraphs.append(" ".join(current))
    return "\n\n".join(paragraphs)
