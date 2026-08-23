from __future__ import annotations

import numpy as np

from ..content.index import hashing_embed


class ChunkStore:
    def __init__(self, dim: int = 256) -> None:
        self.dim = dim
        self.chunks: dict[str, dict] = {}
        self._ids: list[str] = []
        self._matrix: np.ndarray | None = None

    def add_chunk(self, content_id: str, chunk: dict) -> str:
        chunk_key = f"{content_id}:{chunk.get('chunk_id', '0')}"
        self.chunks[chunk_key] = {
            "content_id": content_id,
            "text": chunk["text"],
            "section": chunk.get("section", ""),
            "index": chunk.get("index", 0),
            "embedding": hashing_embed(chunk["text"], self.dim),
        }
        self._matrix = None
        return chunk_key

    def ingest_document(self, content_id: str, text: str, target_words: int = 120, overlap_words: int = 25) -> int:
        from .chunker import contextual_chunks

        for chunk in contextual_chunks(text, target_words, overlap_words):
            self.add_chunk(content_id, chunk)
        return len([k for k in self.chunks if k.startswith(f"{content_id}:")])

    def remove_content(self, content_id: str) -> int:
        keys = [k for k in self.chunks if k.startswith(f"{content_id}:")]
        for key in keys:
            del self.chunks[key]
        self._matrix = None
        return len(keys)

    def _ensure_matrix(self) -> None:
        if self._matrix is None:
            self._ids = sorted(self.chunks)
            if self._ids:
                self._matrix = np.stack([self.chunks[k]["embedding"] for k in self._ids])
            else:
                self._matrix = np.zeros((0, self.dim), dtype=np.float32)

    def search(self, question: str, k: int = 4, content_filter: list[str] | None = None) -> list[dict]:
        self._ensure_matrix()
        if not self._ids:
            return []
        qv = hashing_embed(question, self.dim)
        sims = self._matrix @ qv
        order = np.argsort(-sims)
        results = []
        allowed = set(content_filter) if content_filter else None
        for i in order:
            key = self._ids[i]
            meta = self.chunks[key]
            if allowed and meta["content_id"] not in allowed:
                continue
            results.append(
                {
                    "chunk_key": key,
                    "content_id": meta["content_id"],
                    "section": meta["section"],
                    "score": round(float(sims[i]), 4),
                    "text": meta["text"],
                }
            )
            if len(results) >= k:
                break
        return results
