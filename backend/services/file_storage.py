"""Safe handling of uploaded files: validation, hashing, storage under generated names."""
from __future__ import annotations

import hashlib
import logging
import os
import zipfile
from pathlib import Path

from fastapi import UploadFile

from backend.config import ALLOWED_EXTENSIONS, Settings
from backend.models.schemas import DocumentRecord
from backend.services.document_registry import DocumentRegistry
from backend.utils.errors import AppError
from backend.utils.helpers import new_id, sanitize_filename, utc_now_iso

logger = logging.getLogger(__name__)
_UNSUPPORTED = "Unsupported file type. Please upload PDF, TXT, or DOCX."


class DuplicateDocument(Exception):
    def __init__(self, existing: DocumentRecord):
        self.existing = existing


def _validate_content(path: Path, ext: str) -> None:
    """Check the bytes really look like the claimed type (the extension alone is not trusted)."""
    head = path.read_bytes()[:8192] if ext == ".txt" else path.open("rb").read(8)
    if ext == ".pdf":
        ok = head.startswith(b"%PDF-")
    elif ext == ".docx":
        try:
            with zipfile.ZipFile(path) as z:
                info = z.getinfo("word/document.xml")
                ok = info.file_size < 200 * 1024 * 1024  # crude zip-bomb guard
        except (zipfile.BadZipFile, KeyError):
            ok = False
    else:  # .txt: reject binary data (NUL bytes) unless it is BOM-marked UTF-16
        ok = head.startswith((b"\xff\xfe", b"\xfe\xff")) or b"\x00" not in head
    if not ok:
        raise AppError(400, "invalid_content", "The file content does not match its extension.")


async def save_upload(upload: UploadFile, settings: Settings,
                      registry: DocumentRegistry) -> DocumentRecord:
    display_name = sanitize_filename(upload.filename or "document")
    ext = Path(display_name).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise AppError(400, "unsupported_type", _UNSUPPORTED)

    settings.documents_dir.mkdir(parents=True, exist_ok=True)
    tmp = settings.documents_dir / f".upload-{new_id()}.tmp"
    sha, size = hashlib.sha256(), 0
    try:
        with tmp.open("wb") as fh:
            while chunk := await upload.read(1024 * 1024):  # stream: never hold the file in RAM
                size += len(chunk)
                if size > settings.max_upload_bytes:
                    raise AppError(413, "file_too_large",
                                   f"File is larger than the {settings.max_upload_mb} MB limit.")
                sha.update(chunk)
                fh.write(chunk)
        if size == 0:
            raise AppError(400, "empty_file", "The file is empty.")
        _validate_content(tmp, ext)

        digest = sha.hexdigest()
        existing = registry.find_by_hash(digest)
        if existing:
            raise DuplicateDocument(existing)

        doc_id = new_id()
        stored_name = f"{doc_id}{ext}"  # generated name: user input never touches the path
        os.replace(tmp, settings.documents_dir / stored_name)
        record = DocumentRecord(
            id=doc_id, filename=display_name, file_type=ext.lstrip("."), size_bytes=size,
            file_hash=digest, status="uploaded", created_at=utc_now_iso(), stored_name=stored_name)
        registry.add(record)
        return record
    finally:
        tmp.unlink(missing_ok=True)  # no-op if it was renamed
