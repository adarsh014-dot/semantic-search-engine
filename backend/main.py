"""FastAPI application factory.  Run with:  uvicorn backend.main:app --reload"""
from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException

from backend.api import documents, search
from backend.api.deps import get_services
from backend.config import ALLOWED_EXTENSIONS, BASE_DIR, Settings, get_settings
from backend.container import build_services
from backend.models.schemas import HealthResponse
from backend.services.embedding_service import BaseEmbedder
from backend.services.vector_store import FaissVectorStore
from backend.utils.errors import AppError, EmbeddingError, VectorStoreError

logger = logging.getLogger("semantic_search")


def _error(status: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(status_code=status, content={"error": {"code": code, "message": message}})


def create_app(settings: Settings | None = None, embedder: BaseEmbedder | None = None) -> FastAPI:
    settings = settings or get_settings()
    logging.basicConfig(level=settings.log_level,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    services = build_services(settings, embedder)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if settings.preload_model:  # warm the model in the background; server is usable meanwhile
            async def warm():
                try:
                    await asyncio.to_thread(services.embedder.warm_up)
                except Exception:
                    logger.warning("Model preload failed; it will be retried on first use.")
            asyncio.create_task(warm())
        yield

    app = FastAPI(title="Semantic Embedding Search Engine", version="1.0.0", lifespan=lifespan)
    app.state.services = services

    if settings.cors_origins:
        app.add_middleware(CORSMiddleware, allow_origins=[o.strip() for o in settings.cors_origins.split(",")],
                           allow_methods=["*"], allow_headers=["*"])

    # ---- error handling: friendly JSON, details only in the logs ----
    @app.exception_handler(AppError)
    async def _app_error(_: Request, exc: AppError):
        return _error(exc.status_code, exc.code, exc.message)

    @app.exception_handler(EmbeddingError)
    async def _embedding_error(_: Request, exc: EmbeddingError):
        return _error(503, "embedding_unavailable", str(exc))

    @app.exception_handler(VectorStoreError)
    async def _store_error(_: Request, exc: VectorStoreError):
        logger.error("Vector store error: %s", exc)
        return _error(500, "index_error", "The search index reported an error. Try re-indexing.")

    @app.exception_handler(RequestValidationError)
    async def _validation(_: Request, exc: RequestValidationError):
        first = exc.errors()[0] if exc.errors() else {}
        msg = str(first.get("msg", "Invalid request.")).removeprefix("Value error, ")
        field = ".".join(str(p) for p in first.get("loc", []) if p not in ("body",))
        return _error(422, "invalid_request", f"{field}: {msg}" if field else msg)

    @app.exception_handler(StarletteHTTPException)
    async def _http(_: Request, exc: StarletteHTTPException):
        return _error(exc.status_code, "http_error", str(exc.detail))

    @app.exception_handler(Exception)
    async def _unhandled(_: Request, exc: Exception):
        logger.exception("Unhandled error on %s", _.url.path)
        return _error(500, "server_error", "Something went wrong on the server. Please try again.")

    app.include_router(documents.router)
    app.include_router(search.router)

    @app.get("/api/health", response_model=HealthResponse, tags=["system"])
    def health():
        s = app.state.services
        return HealthResponse(
            status="ok", embedding_model=s.embedder.model_name, model_loaded=s.embedder.is_loaded,
            dimension=s.store.dimension, indexed_chunks=s.store.count,
            documents=len(s.registry.list()), metric="cosine_similarity",
            max_upload_mb=s.settings.max_upload_mb, allowed_extensions=list(ALLOWED_EXTENSIONS),
            default_top_k=s.settings.default_top_k)

    frontend = Path(BASE_DIR) / "frontend"
    if frontend.exists():  # mounted last so /api/* routes win
        app.mount("/", StaticFiles(directory=frontend, html=True), name="frontend")
    return app


app = create_app()
