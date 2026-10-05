"""Document management endpoints: upload, index, list, inspect, delete, re-index."""
from __future__ import annotations

import logging

from fastapi import APIRouter, BackgroundTasks, Depends, File, UploadFile

from backend.api.deps import get_services
from backend.container import Services
from backend.models.schemas import (ChunkContext, ChunkOut, DocumentInfo, IndexRequest, JobResponse,
                                    UploadItemResult, UploadResponse)
from backend.services.file_storage import DuplicateDocument, save_upload
from backend.utils.errors import AppError
from backend.utils.helpers import read_json

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/documents", tags=["documents"])


def _info(rec) -> DocumentInfo:
    return DocumentInfo.model_validate(rec.model_dump())  # drops internal fields


def _get_or_404(svc: Services, doc_id: str):
    rec = svc.registry.get(doc_id)
    if rec is None:
        raise AppError(404, "document_not_found", "Document not found.")
    return rec


@router.post("/upload", response_model=UploadResponse)
async def upload_documents(files: list[UploadFile] = File(...), svc: Services = Depends(get_services)):
    """Store one or more files. Each file succeeds or fails independently."""
    results: list[UploadItemResult] = []
    for f in files:
        name = f.filename or "document"
        try:
            rec = await save_upload(f, svc.settings, svc.registry)
            results.append(UploadItemResult(filename=rec.filename, status="uploaded",
                                            message="Uploaded.", document=_info(rec)))
        except DuplicateDocument as dup:
            results.append(UploadItemResult(filename=name, status="duplicate",
                                            message="Document already exists.",
                                            document=_info(dup.existing)))
        except AppError as err:
            results.append(UploadItemResult(filename=name, status="rejected", message=err.message))
        except Exception:
            logger.exception("Upload failed for %s", name)
            results.append(UploadItemResult(filename=name, status="rejected",
                                            message="The file could not be saved."))
    return UploadResponse(results=results)


def _queue(svc: Services, bg: BackgroundTasks, ids: list[str], rebuild_all: bool = False) -> JobResponse:
    if not rebuild_all:
        svc.indexer.ensure_model_consistent()
    for doc_id in ids:
        svc.registry.update(doc_id, status="processing", error=None)  # visible to the UI at once
    bg.add_task(svc.indexer.run, ids, rebuild_all)  # sync fn -> runs in a worker thread
    return JobResponse(accepted=ids, message=f"Processing {len(ids)} document(s).")


@router.post("/index", response_model=JobResponse, status_code=202)
async def index_documents(body: IndexRequest, bg: BackgroundTasks, svc: Services = Depends(get_services)):
    """Process uploaded documents in the background (poll GET /api/documents for status)."""
    if body.document_ids is None:
        ids = [d.id for d in svc.registry.list() if d.status in ("uploaded", "failed")]
    else:
        ids = [_get_or_404(svc, i).id for i in dict.fromkeys(body.document_ids)]
    if any(svc.registry.get(i).status == "processing" for i in ids):
        raise AppError(409, "already_processing", "A document is already being processed.")
    if not ids:
        return JobResponse(accepted=[], message="Nothing to process.")
    return _queue(svc, bg, ids)


@router.post("/reindex", response_model=JobResponse, status_code=202)
async def reindex_documents(body: IndexRequest, bg: BackgroundTasks, svc: Services = Depends(get_services)):
    """Re-extract and re-embed documents. With no ids, rebuilds the WHOLE index from scratch
    (use this after changing EMBEDDING_MODEL or chunking settings)."""
    everything = body.document_ids is None
    ids = [d.id for d in svc.registry.list()] if everything else \
        [_get_or_404(svc, i).id for i in dict.fromkeys(body.document_ids)]
    if any(svc.registry.get(i).status == "processing" for i in ids):
        raise AppError(409, "already_processing", "A document is already being processed.")
    if not ids:
        return JobResponse(accepted=[], message="Nothing to re-index.")
    return _queue(svc, bg, ids, rebuild_all=everything)


@router.get("", response_model=list[DocumentInfo])
async def list_documents(svc: Services = Depends(get_services)):
    return [_info(d) for d in svc.registry.list()]


@router.get("/{doc_id}", response_model=DocumentInfo)
async def get_document(doc_id: str, svc: Services = Depends(get_services)):
    return _info(_get_or_404(svc, doc_id))


@router.delete("/{doc_id}", status_code=200)
async def delete_document(doc_id: str, svc: Services = Depends(get_services)):
    """Remove the file, its vectors (FAISS remove_ids), metadata and processed chunks."""
    rec = _get_or_404(svc, doc_id)
    if rec.status == "processing":
        raise AppError(409, "already_processing", "Wait for processing to finish before deleting.")
    svc.indexer.remove(doc_id)
    (svc.settings.documents_dir / rec.stored_name).unlink(missing_ok=True)
    svc.registry.delete(doc_id)
    return {"deleted": doc_id}


@router.get("/{doc_id}/chunks/{chunk_index}", response_model=ChunkContext)
async def chunk_context(doc_id: str, chunk_index: int, svc: Services = Depends(get_services)):
    """A chunk plus its neighbours, for the 'View source' panel."""
    rec = _get_or_404(svc, doc_id)
    data = read_json(svc.indexer.processed_path(doc_id), None)
    if not data:
        raise AppError(404, "chunks_not_found", "This document has not been processed yet.")
    by_index = {c["chunk_index"]: c for c in data["chunks"]}
    if chunk_index not in by_index:
        raise AppError(404, "chunk_not_found", "Chunk not found.")

    def out(i: int) -> ChunkOut | None:
        c = by_index.get(i)
        return ChunkOut(chunk_id=c["chunk_id"], chunk_index=i, page=c["page"], text=c["text"]) if c else None

    return ChunkContext(document=_info(rec), chunk=out(chunk_index),
                        previous=out(chunk_index - 1), next=out(chunk_index + 1))
