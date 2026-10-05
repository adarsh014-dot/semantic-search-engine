"""Small shared helpers: IDs, timestamps, filename sanitising, atomic file writes."""
from __future__ import annotations

import json
import os
import re
import unicodedata
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def new_id() -> str:
    return uuid.uuid4().hex[:16]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


_SAFE_CHARS = re.compile(r"[^A-Za-z0-9._ \-()]+")


def sanitize_filename(name: str, max_len: int = 120) -> str:
    """Return a display-safe filename: no directories, no odd characters.

    The sanitised name is only used for DISPLAY. Files are stored on disk under a
    generated id, so the user can never influence the real path (no traversal).
    """
    name = name.replace("\\", "/").split("/")[-1]  # drop any directory part
    name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    name = _SAFE_CHARS.sub("_", name).strip(" .")
    stem, dot, ext = name.rpartition(".")
    if not dot:
        stem, ext = name, ""
    stem = (stem or "document")[: max_len - len(ext) - 1]
    return f"{stem}.{ext}" if ext else stem


def atomic_write_text(path: Path, text: str) -> None:
    """Write via temp file + rename so a crash never leaves a half-written file."""
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def atomic_write_json(path: Path, data: Any) -> None:
    atomic_write_text(path, json.dumps(data, ensure_ascii=False, indent=2))


def read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))
