"""Recommendation evaluation sliced by user cohort (DataPerf discipline)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class CohortMetric:
    name: str
    users: int
    precision_at_k: float
    hit_rate: float
    mrr: float
    low_support: bool


def evaluate_cohorts(
    cases: list[tuple[str, list[str], set[str]]],
    *,
    k: int = 10,
    min_support: int = 3,
) -> dict:
    """Score recommendations per cohort.

    ``cases`` is [(cohort, ranked_items, relevant_set)]. Reports
    precision@k, hit rate and MRR per cohort, a macro over cohorts with
    enough support, and which cohorts are too small to trust."""
    if not cases:
        raise ValueError("cases must be non-empty")
    if k < 1:
        raise ValueError("k must be >= 1")
    if min_support < 1:
        raise ValueError("min_support must be >= 1")

    grouped: dict[str, list[tuple[list[str], set[str]]]] = {}
    for cohort, ranked, relevant in cases:
        grouped.setdefault(cohort, []).append((ranked, relevant))

    slices: dict[str, CohortMetric] = {}
    for name, items in sorted(grouped.items()):
        precisions = []
        hits = 0
        reciprocals = []
        for ranked, relevant in items:
            top = ranked[:k]
            hit_items = [item for item in top if item in relevant]
            precisions.append(len(hit_items) / k)
            if hit_items:
                hits += 1
                reciprocals.append(1.0 / (top.index(hit_items[0]) + 1))
            else:
                reciprocals.append(0.0)
        n = len(items)
        slices[name] = CohortMetric(
            name=name,
            users=n,
            precision_at_k=round(sum(precisions) / n, 4),
            hit_rate=round(hits / n, 4),
            mrr=round(sum(reciprocals) / n, 4),
            low_support=n < min_support,
        )

    reliable = [s for s in slices.values() if not s.low_support]
    micro_precision = sum(
        len(set(ranked[:k]) & relevant) / k
        for _cohort, ranked, relevant in cases
    ) / len(cases)
    micro_hits = sum(1 for _cohort, ranked, relevant in cases if set(ranked[:k]) & relevant)
    return {
        "k": k,
        "micro_precision_at_k": round(micro_precision, 4),
        "micro_hit_rate": round(micro_hits / len(cases), 4),
        "macro_precision_at_k": round(
            sum(s.precision_at_k for s in reliable) / len(reliable), 4
        ) if reliable else 0.0,
        "macro_mrr": round(
            sum(s.mrr for s in reliable) / len(reliable), 4
        ) if reliable else 0.0,
        "slices": slices,
        "low_support_slices": sorted(s.name for s in slices.values() if s.low_support),
    }
