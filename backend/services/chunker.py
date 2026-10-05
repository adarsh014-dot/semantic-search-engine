"""Splits cleaned text into overlapping chunks that keep source/page metadata."""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable

from backend.services.document_processor import ExtractedPage
from backend.services.text_cleaner import clean_text
from backend.utils.helpers import utc_now_iso

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])")


@dataclass
class Chunk:
    chunk_id: str
    document_id: str
    source: str
    page: int | None
    chunk_index: int
    text: str
    created_at: str
    # Room for author / course / subject / document_type / tags without schema changes.
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class Chunker:
    """Greedy sentence-aware chunker.

    Sizes are given in *tokens* but measured in words (tokens ~= words * tokens_per_word),
    which avoids a tokenizer dependency and is accurate enough for English prose.
    Chunks never span pages, so every chunk has exactly one page number.
    Overlap is made of whole trailing sentences, so chunks start at natural boundaries.
    """

    def __init__(self, chunk_size_tokens: int = 500, overlap_tokens: int = 50,
                 tokens_per_word: float = 1.3):
        self.max_words = max(1, int(chunk_size_tokens / tokens_per_word))
        self.overlap_words = max(0, min(int(overlap_tokens / tokens_per_word), self.max_words - 1))

    # ---- public API ----
    def chunk_pages(self, pages: Iterable[ExtractedPage], document_id: str,
                    extra: dict[str, Any] | None = None) -> list[Chunk]:
        chunks: list[Chunk] = []
        now = utc_now_iso()
        for page in pages:
            for piece in self.split_text(clean_text(page.text)):
                index = len(chunks)
                chunks.append(Chunk(
                    chunk_id=f"{document_id}-{index:05d}",  # unique + keeps ordering
                    document_id=document_id, source=page.source, page=page.page,
                    chunk_index=index, text=piece, created_at=now, extra=dict(extra or {}),
                ))
        return chunks

    def split_text(self, text: str) -> list[str]:
        units = self._units(text)
        chunks: list[str] = []
        cur: list[tuple[str, int]] = []  # (sentence, word_count)
        cur_words = 0
        for unit, n in units:
            if cur and cur_words + n > self.max_words:
                chunks.append(" ".join(u for u, _ in cur))
                cur = self._overlap_tail(cur)
                cur_words = sum(w for _, w in cur)
                if cur_words + n > self.max_words:  # overlap would overflow: drop it
                    cur, cur_words = [], 0
            cur.append((unit, n))
            cur_words += n
        if cur:
            chunks.append(" ".join(u for u, _ in cur))
        return chunks

    # ---- internals ----
    def _units(self, text: str) -> list[tuple[str, int]]:
        """Sentences (never longer than max_words) with their word counts."""
        units: list[tuple[str, int]] = []
        for paragraph in text.split("\n\n"):
            for sentence in _SENTENCE_SPLIT.split(paragraph):
                words = sentence.split()
                for i in range(0, len(words), self.max_words):  # hard-split giant "sentences"
                    part = words[i:i + self.max_words]
                    if part:
                        units.append((" ".join(part), len(part)))
        return units

    def _overlap_tail(self, cur: list[tuple[str, int]]) -> list[tuple[str, int]]:
        tail: list[tuple[str, int]] = []
        total = 0
        for unit, n in reversed(cur):
            if total + n > self.overlap_words:
                break
            tail.insert(0, (unit, n))
            total += n
        return tail
