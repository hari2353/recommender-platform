from __future__ import annotations

from datetime import datetime, timezone

import numpy as np

from ..content.index import ContentIndex, hashing_embed


class CandidateReranker:
    def __init__(self, index: ContentIndex, relevance_weight: float = 0.6, freshness_weight: float = 0.2, popularity_weight: float = 0.2) -> None:
        self.index = index
        self.relevance_weight = relevance_weight
        self.freshness_weight = freshness_weight
        self.popularity_weight = popularity_weight
        self._query_cache: dict[str, np.ndarray] = {}

    def _query_vec(self, query_text: str) -> np.ndarray:
        if query_text not in self._query_cache:
            self._query_cache[query_text] = hashing_embed(query_text, self.index.dim)
        return self._query_cache[query_text]

    def freshness(self, content_id: str) -> float:
        item = self.index.items.get(content_id)
        if not item or not item.get("published_at"):
            return 0.5
        try:
            published = datetime.fromisoformat(item["published_at"]).replace(tzinfo=timezone.utc)
        except ValueError:
            return 0.5
        age_days = max((datetime.now(timezone.utc) - published).days, 0)
        return float(0.5 ** (age_days / 365))

    def rerank(self, candidate_ids: list[str], query_text: str, popularity: dict[str, float] | None = None, top_n: int | None = None) -> list[tuple[str, float]]:
        popularity = popularity or {}
        qv = self._query_vec(query_text)
        scored = []
        for cid in candidate_ids:
            item = self.index.items.get(cid)
            if item is None:
                continue
            rel = float(np.dot(self.index.matrix[self.index.ids.index(cid)], qv))
            fresh = self.freshness(cid)
            pop = popularity.get(cid, 0.5)
            total = round(
                self.relevance_weight * rel + self.freshness_weight * fresh + self.popularity_weight * pop,
                4,
            )
            scored.append((cid, total))
        scored.sort(key=lambda pair: -pair[1])
        return scored[:top_n] if top_n else scored
