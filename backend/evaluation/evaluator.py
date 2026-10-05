"""Runs a labelled query set against the live search service and reports metrics."""
from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any

from backend.evaluation import metrics
from backend.models.schemas import SearchResult
from backend.services.search_service import SearchService

DEFAULT_DATASET = Path(__file__).with_name("sample_eval_dataset.json")


def _norm(s: str) -> str:
    return " ".join(s.lower().split())


def is_relevant(result: SearchResult, expected_document: str, expected_chunk: str | None) -> bool:
    """Relevant = from the expected document and, if given, containing the expected passage."""
    if result.source != expected_document:
        return False
    return expected_chunk is None or _norm(expected_chunk) in _norm(result.text)


def load_dataset(path: Path = DEFAULT_DATASET) -> list[dict[str, Any]]:
    return json.loads(Path(path).read_text(encoding="utf-8"))["queries"]


def evaluate(search: SearchService, dataset: list[dict[str, Any]], k: int = 5) -> dict[str, Any]:
    per_query, flags_all, latencies = [], [], []
    for item in dataset:
        resp = search.search(item["query"], top_k=k)
        flags = [is_relevant(r, item["expected_document"], item.get("expected_chunk"))
                 for r in resp.results]
        flags_all.append(flags)
        latencies.append(resp.took_ms)
        per_query.append({"query": item["query"], "relevant_flags": flags,
                          f"precision@{k}": metrics.precision_at_k(flags, k),
                          f"recall@{k}": metrics.recall_at_k(flags, k),
                          "reciprocal_rank": metrics.reciprocal_rank(flags),
                          "latency_ms": resp.took_ms})
    n = len(dataset)
    return {
        "model": search.embedder.model_name, "k": k, "num_queries": n,
        f"precision@{k}": statistics.fmean(q[f"precision@{k}"] for q in per_query) if n else 0.0,
        f"recall@{k}": statistics.fmean(q[f"recall@{k}"] for q in per_query) if n else 0.0,
        "mrr": statistics.fmean(q["reciprocal_rank"] for q in per_query) if n else 0.0,
        f"hit_rate@{k}": metrics.hit_rate(flags_all, k),
        "latency_ms_mean": statistics.fmean(latencies) if latencies else 0.0,
        "latency_ms_p95": metrics.percentile(latencies, 95),
        "per_query": per_query,
    }
