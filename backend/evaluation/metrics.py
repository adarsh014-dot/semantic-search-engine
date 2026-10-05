"""Retrieval metrics. Pure functions: no I/O, easy to unit test.

Each query has ONE expected target (a document, optionally narrowed to a chunk), so:
  * a ranked result is relevant (True) or not (False)
  * Recall@K = 1 if any relevant result is in the top K, else 0  (single target)
  * Precision@K = relevant results in top K / K
  * MRR = 1 / rank of the first relevant result (0 if none)
  * Hit rate = fraction of queries with Recall@K = 1
"""
from __future__ import annotations

import statistics


def precision_at_k(flags: list[bool], k: int) -> float:
    return sum(flags[:k]) / k if k > 0 else 0.0


def recall_at_k(flags: list[bool], k: int) -> float:
    return 1.0 if any(flags[:k]) else 0.0


def reciprocal_rank(flags: list[bool]) -> float:
    for rank, ok in enumerate(flags, start=1):
        if ok:
            return 1.0 / rank
    return 0.0


def hit_rate(per_query_flags: list[list[bool]], k: int) -> float:
    return statistics.fmean(recall_at_k(f, k) for f in per_query_flags) if per_query_flags else 0.0


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, round(pct / 100 * (len(ordered) - 1))))
    return ordered[idx]
