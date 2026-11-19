# Semantic Embedding Search Engine

![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)
![FAISS](https://img.shields.io/badge/FAISS-vector%20search-0467DF)
![tests](https://img.shields.io/badge/tests-37%20passing-brightgreen)
![license](https://img.shields.io/badge/license-MIT-blue)

Upload PDF, TXT and DOCX files, and search them **by meaning** instead of by exact keywords.
This is a *retrieval* system: it returns the most relevant document chunks with real
similarity scores and sources. There is deliberately **no LLM and no generated answer**, so you can
see exactly how semantic retrieval behaves. It is built as the base for later RAG, Code RAG,
evaluation and multimodal projects.

```
Document -> Extraction -> Cleaning -> Chunking -> Embedding -> FAISS
Query    -> Embedding  -> Similarity search -> Top-K chunks + scores + sources
```

## What is semantic search?

**Keyword search** matches words. The query "How does TCP start communication?" would miss the sentence
"TCP establishes communication using a three-way handshake" if the words "start" and "handshake" had to match.

**Semantic search** converts text into vectors (lists of numbers) with a neural *embedding model*. Texts with
similar meaning end up close together in that vector space, so the query and the sentence match even though
they share few words.

## Architecture

```
            UPLOAD                                   INDEXING (background task)
 browser ──> POST /api/documents/upload ──> data/documents/<id>.<ext>
 browser ──> POST /api/documents/index  ──> DocumentProcessor  (PyMuPDF / python-docx / txt)
                                            -> text_cleaner -> Chunker (500 tok, 50 overlap, per page)
                                            -> EmbeddingService (batched, normalised)
                                            -> FaissVectorStore.add  + chunks.json metadata
                                            -> data/index/{faiss.index, chunks.json}, data/processed/<id>.json

            SEARCH
 browser ──> POST /api/search ──> embed_query -> FAISS inner-product search -> metadata lookup
                              ──> [{chunk_id, source, page, text, score}, ...]
```

| Folder | Responsibility |
|---|---|
| `backend/main.py` | App factory, error handlers, serves the frontend |
| `backend/config.py` | All settings (env vars / `.env`) |
| `backend/api/` | Thin HTTP routers: `documents.py`, `search.py` |
| `backend/services/` | The real logic: `document_processor`, `text_cleaner`, `chunker`, `embedding_service`, `vector_store`, plus `indexer`, `search_service`, `file_storage`, `document_registry` |
| `backend/models/schemas.py` | Pydantic request/response models |
| `backend/evaluation/` | Retrieval metrics, evaluator, CLI, demo dataset |
| `frontend/` | `index.html`, `style.css`, `app.js` (no framework) |
| `data/documents/` | Uploaded files (stored under generated ids, never user-supplied paths) |
| `data/processed/` | Document registry (`documents.json`) + extracted chunks per document |
| `data/index/` | `faiss.index` + `chunks.json` (FAISS id -> chunk metadata) |
| `sample_data/` | Three small original documents for trying the app |
| `tests/` | pytest suite |

Extension points: the app depends on `BaseEmbedder` and `VectorStore` (abstract classes). Add a new embedding
backend or vector database by implementing the interface; the API, chunker and UI do not change.

## Technologies

| Tech | Why |
|---|---|
| FastAPI + Uvicorn + Pydantic | Async API, automatic validation and docs at `/docs` |
| PyMuPDF | Fast PDF text extraction with page numbers |
| python-docx | Reads DOCX paragraphs and tables in document order |
| Sentence Transformers | Pretrained embedding models; swap by changing `EMBEDDING_MODEL` |
| FAISS | Fast vector similarity search library |
| Plain HTML/CSS/JS | No build step; easy to read |

## Installation

Requires Python 3.10+.

```bash
python -m venv .venv
# macOS / Linux
source .venv/bin/activate
# Windows (PowerShell / cmd)
.venv\Scripts\activate

pip install -r requirements.txt
cp .env.example .env        # optional; Windows: copy .env.example .env
```

The **first run downloads the embedding model** (about 130 MB for the default) from Hugging Face, so it needs
internet once. After that it is cached.

## Running

```bash
uvicorn backend.main:app --reload
```

Open **http://127.0.0.1:8000** (the backend serves the frontend). Interactive API docs: http://127.0.0.1:8000/docs

### Try it with the sample data

1. Drag the three files from `sample_data/` onto the upload area and click **Upload and index**.
2. Wait for each document to show *Indexed*.
3. Search `How does TCP establish a connection?`, `What translates a website name into an IP address?`,
   or `How do plants turn sunlight into chemical energy?`.
4. Stop the server (Ctrl+C), start it again and search again. The index is loaded from disk, no re-upload needed.

Regenerate the sample files with `python scripts/make_sample_data.py`.

## Configuration

See `.env.example`. The most useful settings:

| Variable | Default | Meaning |
|---|---|---|
| `EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | Any Sentence Transformers model |
| `EMBEDDING_QUERY_PREFIX` | bge instruction | Prefix added to **queries** only. Set empty for models that don't use one |
| `CHUNK_SIZE_TOKENS` / `CHUNK_OVERLAP_TOKENS` | 500 / 50 | Chunking (approximate tokens) |
| `SEARCH_MIN_SCORE` | 0.2 | Drop results below this cosine similarity (model dependent, tune it) |
| `MAX_UPLOAD_MB` | 25 | Per-file upload limit |

**Changing the model:** edit `.env`, restart, then click **Re-index all** (or `POST /api/documents/reindex`).
Vectors from different models are not comparable, so search returns a clear 409 until you re-index.
Note: `all-MiniLM-L6-v2` truncates input at 256 word-pieces, so with 500-token chunks it would ignore the end of each chunk.
Lower `CHUNK_SIZE_TOKENS` to about 200 for it. The default bge model accepts 512.

## API

All errors look like `{"error": {"code": "...", "message": "..."}}`. No stack traces are ever returned.

| Method & path | Purpose |
|---|---|
| `POST /api/documents/upload` | Multipart field `files` (one or many). Returns a per-file result: `uploaded`, `duplicate` ("Document already exists.") or `rejected` with the reason |
| `POST /api/documents/index` | Body `{"document_ids": [...]}` (omit for all pending/failed). Returns `202`; processing runs in the background |
| `POST /api/documents/reindex` | Same body. With ids: re-embed those. Without: rebuild the **whole** index |
| `GET /api/documents` | List documents with `status` (`uploaded`/`processing`/`indexed`/`failed`), `pages`, `chunks`, `error` |
| `GET /api/documents/{id}` | One document |
| `DELETE /api/documents/{id}` | Removes file, vectors, metadata |
| `GET /api/documents/{id}/chunks/{chunk_index}` | Chunk with previous/next chunk (used by "View source") |
| `POST /api/search` | `{"query": "...", "top_k": 5}` -> `{"results": [{chunk_id, document_id, source, page, chunk_index, text, score}], "took_ms", ...}` |
| `GET /api/health` | Model, dimension, number of indexed chunks |

## How embeddings work

An embedding model reads a passage and outputs a vector, for the default model 384 numbers. You can think of
each number as a coordinate; passages about similar things get similar coordinates. We never look at individual
numbers. We only compare vectors: the **angle** between two vectors tells us how similar their meaning is.

## Similarity vs distance (cosine, explained)

Cosine similarity is `cos(angle between a and b) = (a . b) / (|a||b|)`; 1 means same direction, 0 unrelated,
negative opposite. We L2-normalise every vector to length 1, so `|a| = |b| = 1` and the formula collapses to
`a . b`, the plain **inner product**. FAISS's `IndexFlatIP` computes inner products, so its output **is** the cosine
similarity (higher = more similar). That is a *similarity*. A *distance* (like the L2 index returns) is the
opposite: lower = more similar. This app only uses similarity and shows it as a percentage in the UI; the
number is the raw cosine similarity from FAISS, never rescaled or invented.

## How FAISS works (here)

FAISS stores vectors and finds the nearest ones to a query very efficiently. We use `IndexFlatIP`, an **exact**
brute-force scan, which is fast and perfectly accurate up to roughly hundreds of thousands of chunks. For millions
you would switch to an approximate index (IVF or HNSW), which only affects `vector_store.py`.
FAISS holds vectors only, so each vector gets an integer id (`IndexIDMap2`) and `chunks.json` maps
id -> chunk metadata. Both files are written atomically; on startup they are cross-checked, and if they disagree
the index is set aside as `*.corrupt` and documents are flagged for re-indexing instead of returning wrong results.

## Evaluation

```bash
python -m backend.evaluation.run --k 5          # evaluates the index currently on disk
```

Computes Precision@K, Recall@K, MRR, Hit Rate and latency (mean / p95) for every query in
`backend/evaluation/sample_eval_dataset.json`. That file is a **tiny demonstration dataset** for the files in
`sample_data/` and cannot say anything meaningful about quality; replace it with your own
`query / expected_document / expected_chunk` entries. No numbers are shipped in this repo; run it to get yours.
To compare models: change `EMBEDDING_MODEL`, re-index, run again, and compare dimension, scores, latency and
`data/index` size.

## Tests

```bash
pytest -q
```

**37 tests passing** (verified on Windows 11 with Python 3.14.0: `pytest -q` → `37 passed in ~42s`; the real-model
tests need a one-time model download and are skipped automatically when offline).

* `test_chunker.py`: chunking, overlap, metadata, cleaning
* `test_search.py`: FAISS insert/search/metadata mapping/delete/persistence/corruption recovery
* `test_api.py`: upload, index, search, errors, duplicates, traversal, restart, delete, re-index
* `test_embeddings.py`: embedding contract; the **real-model** tests are skipped automatically if the model can't be downloaded
* `test_evaluation.py`: metric math

API and vector tests use a small deterministic *test-only* embedder (`tests/conftest.py`) so they run offline in under a
second. It only captures word overlap and is never used by the app. Real semantic behaviour is covered by the real-model tests in `test_embeddings.py`.

## Security notes

Extensions **and** file contents are validated (PDF magic bytes, DOCX zip structure, no binary TXT); sizes are
limited while streaming; filenames are sanitised and files are stored under generated ids (no path traversal,
no internal paths exposed); uploaded files are only parsed, never executed; all UI text is inserted via `textContent`.
There is no authentication, so this is meant for local or portfolio use.

## Limitations

* **Scanned PDFs** (images of text) have no extractable text and are rejected; there is no OCR.
* DOCX and TXT have no reliable page numbers, so results show no page for them.
* Chunk sizes are estimated from word counts (about 1.3 tokens per word), not a real tokenizer.
* Cleaning joins single line breaks, so bullet lists or code in TXT files lose their line structure.
* Tables are flattened to text. Images and charts are ignored.
* Retrieval is dense-only. Exact identifiers (error codes, names) can be missed; hybrid keyword + vector search would help.
* Flat FAISS index: exact but linear in index size. One embedding model per index.
* Similarity scores are only comparable within one model; `SEARCH_MIN_SCORE` needs tuning per model.
* Background indexing uses an in-process worker: a server crash mid-job leaves that document marked *failed* (re-index it). One job runs at a time.
* Upload hashing detects *identical* files only, not near-duplicates.

## Licence

Released under the **MIT Licence** — see [LICENSE](LICENSE).
