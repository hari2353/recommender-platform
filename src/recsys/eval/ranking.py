"""Ranking-quality metrics for retrieval eval — the RecBole/implicit
baseline set a recommender platform reports alongside CTR."""

from __future__ import annotations

import math


def precision_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    if k <= 0 or not relevant:
        return 0.0
    hits = sum(1 for cid in ranked[:k] if cid in relevant)
    return hits / k


def recall_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    hits = sum(1 for cid in ranked[:k] if cid in relevant)
    return hits / len(relevant)


def hit_rate_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    return 1.0 if any(cid in relevant for cid in ranked[:k]) else 0.0


def mrr(ranked: list[str], relevant: set[str]) -> float:
    for i, cid in enumerate(ranked, start=1):
        if cid in relevant:
            return 1.0 / i
    return 0.0


def _dcg(gains: list[float]) -> float:
    return sum(g / math.log2(pos + 1) for pos, g in enumerate(gains, start=1))


def ndcg_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    if k <= 0 or not relevant:
        return 0.0
    gains = [1.0 if cid in relevant else 0.0 for cid in ranked[:k]]
    ideal = [1.0] * min(k, len(relevant))
    ideal_dcg = _dcg(ideal)
    if ideal_dcg == 0.0:
        return 0.0
    return _dcg(gains) / ideal_dcg
