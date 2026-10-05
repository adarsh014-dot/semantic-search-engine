"""Turns an uploaded file into a list of pages of raw text, keeping source + page info."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from backend.utils.errors import DocumentProcessingError

logger = logging.getLogger(__name__)


@dataclass
class ExtractedPage:
    text: str
    source: str  # display filename
    page: int | None  # 1-based page number; None when the format has no pages


class DocumentProcessor:
    """Detects the file type from the extension and dispatches to the right extractor."""

    def extract(self, path: Path, source: str) -> list[ExtractedPage]:
        ext = path.suffix.lower()
        extractor = {".pdf": self._pdf, ".docx": self._docx, ".txt": self._txt}.get(ext)
        if extractor is None:
            raise DocumentProcessingError("Unsupported file type. Please upload PDF, TXT, or DOCX.")
        pages = extractor(path, source)
        if not any(p.text.strip() for p in pages):
            raise DocumentProcessingError(
                "No text could be extracted. The document is empty or may be a scanned image "
                "(OCR is not supported in this version)."
            )
        return pages

    # ---- PDF (PyMuPDF) ----
    def _pdf(self, path: Path, source: str) -> list[ExtractedPage]:
        import pymupdf  # imported lazily so the app still starts if it is missing

        try:
            with pymupdf.open(path) as doc:
                if doc.needs_pass:
                    raise DocumentProcessingError("The PDF is password protected.")
                return [
                    ExtractedPage(text=page.get_text("text"), source=source, page=i + 1)
                    for i, page in enumerate(doc)
                ]
        except DocumentProcessingError:
            raise
        except Exception as exc:
            logger.exception("PDF extraction failed for %s", source)
            raise DocumentProcessingError("The PDF is corrupt or could not be read.") from exc

    # ---- DOCX (python-docx) ----
    def _docx(self, path: Path, source: str) -> list[ExtractedPage]:
        from docx import Document
        from docx.table import Table
        from docx.text.paragraph import Paragraph

        try:
            doc = Document(str(path))
            blocks: list[str] = []
            # Walk the body in order so paragraphs and tables stay where the author put them.
            for child in doc.element.body.iterchildren():
                if child.tag.endswith("}p"):
                    text = Paragraph(child, doc).text.strip()
                    if text:
                        blocks.append(text)
                elif child.tag.endswith("}tbl"):
                    for row in Table(child, doc).rows:
                        cells = [c.text.strip() for c in row.cells if c.text.strip()]
                        if cells:
                            blocks.append(" | ".join(cells))
        except Exception as exc:
            logger.exception("DOCX extraction failed for %s", source)
            raise DocumentProcessingError("The DOCX file is corrupt or could not be read.") from exc
        # DOCX has no stable page numbers (pagination is decided at render time).
        return [ExtractedPage(text="\n\n".join(blocks), source=source, page=None)]

    # ---- TXT ----
    def _txt(self, path: Path, source: str) -> list[ExtractedPage]:
        raw = path.read_bytes()
        for encoding in ("utf-8-sig", "utf-16", "cp1252"):
            try:
                if encoding == "utf-16" and not raw.startswith((b"\xff\xfe", b"\xfe\xff")):
                    continue  # only trust utf-16 when a BOM says so
                text = raw.decode(encoding)
                break
            except UnicodeDecodeError:
                continue
        else:
            text = raw.decode("latin-1")  # never fails
        return [ExtractedPage(text=text, source=source, page=None)]
