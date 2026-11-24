from __future__ import annotations

import math
import re
from collections import Counter
from datetime import datetime, timezone

import numpy as np

from ..content.index import ContentIndex, hashing_embed

_WORD = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> list[str]:
    return _WORD.findall(text.lower())


def _bm25(
    docs: dict[str, list[str]], query: str, k1: float = 1.5, b: float = 0.75
) -> dict[str, float]:
    """Okapi BM25 over the candidate documents only; no index-wide statistics."""
    terms = _tokens(query)
    if not docs or not terms:
        return {}
    avg_len = sum(len(d) for d in docs.values()) / len(docs) or 1.0
    scores: dict[str, float] = {}
    for cid, tokens in docs.items():
        counts = Counter(tokens)
        score = 0.0
        for term in set(terms):
            tf = counts.get(term, 0)
            if not tf:
                continue
            df = sum(1 for other in docs.values() if term in other)
            idf = math.log(1 + (len(docs) - df + 0.5) / (df + 0.5))
            score += (
                idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * len(tokens) / avg_len))
            )
        scores[cid] = score
    return scores


class CandidateReranker:
    def __init__(
        self,
        index: ContentIndex,
        relevance_weight: float = 0.6,
        freshness_weight: float = 0.2,
        popularity_weight: float = 0.2,
    ) -> None:
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
            published = datetime.fromisoformat(item["published_at"]).replace(
                tzinfo=timezone.utc
            )
        except ValueError:
            return 0.5
        age_days = max((datetime.now(timezone.utc) - published).days, 0)
        return float(0.5 ** (age_days / 365))

    def rerank(
        self,
        candidate_ids: list[str],
        query_text: str,
        popularity: dict[str, float] | None = None,
        top_n: int | None = None,
    ) -> list[tuple[str, float]]:
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
                self.relevance_weight * rel
                + self.freshness_weight * fresh
                + self.popularity_weight * pop,
                4,
            )
            scored.append((cid, total))
        scored.sort(key=lambda pair: -pair[1])
        return scored[:top_n] if top_n else scored

    def rerank_hybrid(
        self,
        candidate_ids: list[str],
        query_text: str,
        popularity: dict[str, float] | None = None,
        top_n: int | None = None,
        lexical_weight: float = 0.3,
        rrf_k: int = 60,
    ) -> list[tuple[str, float]]:
        """Fuse the dense reranker with candidate-local BM25 using weighted RRF."""
        if not candidate_ids or not query_text.strip():
            return []
        dense = self.rerank(candidate_ids, query_text, popularity)
        if not dense:
            return []
        docs = {
            cid: _tokens(
                " ".join(
                    [item.get("title", ""), item.get("text", "")]
                    + [str(s) for s in item.get("skills", [])]
                    + [str(o) for o in item.get("objectives", [])]
                )
            )
            for cid in (pair[0] for pair in dense)
            if (item := self.index.items.get(cid))
        }
        lexical_scores = _bm25(docs, query_text)
        dense_scores = dict(dense)
        dense_rank = {
            cid: 1 + sum(1 for other in dense_scores.values() if other > score)
            for cid, score in dense_scores.items()
        }
        lexical_rank = {
            cid: 1
            + sum(
                1
                for other in lexical_scores.values()
                if other > lexical_scores.get(cid, 0.0)
            )
            for cid in dense_scores
        }
        fused = {
            cid: lexical_weight / (rrf_k + lexical_rank[cid])
            + (1 - lexical_weight) / (rrf_k + dense_rank[cid])
            for cid in dense_scores
        }
        ranked = sorted(fused.items(), key=lambda pair: (-pair[1], pair[0]))
        return ranked[:top_n] if top_n else ranked
