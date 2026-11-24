from __future__ import annotations

import os

from ..content.index import hashing_embed
from .chunker import agentic_chunks, semantic_chunks, structural_chunks
from .embeddings import get_provider


class ChunkStore:
    def __init__(
        self, provider_name: str = "hashing", dim: int = 256, region: str | None = None
    ) -> None:
        self.provider = get_provider(provider_name, dim=dim, region=region)
        self.chunks: dict[str, dict] = {}
        self._ids: list[str] = []
        self._matrix = None

    def _embed_passage(self, text: str):
        if hasattr(self.provider, "embed_passage"):
            return self.provider.embed_passage(text)
        return self.provider.embed(text)

    def add_chunk(self, content_id: str, chunk: dict) -> str:
        key = f"{content_id}:{chunk.get('chunk_id', '0')}"
        self.chunks[key] = {
            "content_id": content_id,
            "text": chunk["text"],
            "section": chunk.get("section", ""),
            "index": chunk.get("index", 0),
            "strategy": chunk.get("strategy", "structural"),
            "embedding": self._embed_passage(chunk["text"]),
        }
        self._matrix = None
        return key

    def ingest_document(
        self, content_id: str, text: str, strategy: str = "structural"
    ) -> int:
        if strategy == "agentic" and os.getenv("SKILLS_LLM_BASE_URL"):
            chunks = agentic_chunks(
                text,
                os.getenv("SKILLS_LLM_BASE_URL", ""),
                os.getenv("SKILLS_LLM_MODEL", "gpt-4o-mini"),
                os.getenv("SKILLS_LLM_API_KEY", ""),
            )
        elif strategy == "semantic":
            chunks = semantic_chunks(text, self.provider)
        else:
            chunks = structural_chunks(text)
        for chunk in chunks:
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
                self._matrix = np_stack(
                    [self.chunks[k]["embedding"] for k in self._ids]
                )
            else:
                self._matrix = np_zeros(0, self._dim())

    def _dim(self) -> int:
        return (
            len(next(iter(self.chunks.values()))["embedding"])
            if self.chunks
            else getattr(self.provider, "dim", 256)
        )

    def search(
        self, question: str, k: int = 4, content_filter: list[str] | None = None
    ) -> list[dict]:
        import numpy as np

        self._ensure_matrix()
        if not self._ids:
            return []
        if hasattr(self.provider, "embed_query"):
            qv = self.provider.embed_query(question)
        else:
            qv = hashing_embed(question, self._dim())
        sims = self._matrix @ qv
        n = len(self._ids)
        k_eff = min(k * 4, n)
        top = np.argpartition(-sims, k_eff - 1)[:k_eff] if k_eff < n else np.arange(n)
        order = top[np.argsort(-sims[top])]
        allowed = set(content_filter) if content_filter else None
        results = []
        for i in order:
            meta = self.chunks[self._ids[i]]
            if allowed and meta["content_id"] not in allowed:
                continue
            results.append(
                {
                    "chunk_key": self._ids[i],
                    "content_id": meta["content_id"],
                    "section": meta["section"],
                    "strategy": meta["strategy"],
                    "score": round(float(sims[i]), 4),
                    "text": meta["text"],
                }
            )
            if len(results) >= k:
                break
        return results


def np_stack(vectors: list):
    import numpy as np

    return np.stack(vectors)


def np_zeros(rows: int, dim: int):
    import numpy as np

    return np.zeros((rows, dim), dtype=np.float32)


class RagService:
    def __init__(self, store: ChunkStore) -> None:
        self.store = store

    def ask(
        self, question: str, k: int = 4, content_filter: list[str] | None = None
    ) -> dict:
        hits = self.store.search(question, k=k, content_filter=content_filter)
        answer = self._synthesize(question, hits)
        return {
            "question": question,
            "answer": answer,
            "citations": [
                {
                    "n": i + 1,
                    "content_id": h["content_id"],
                    "section": h["section"],
                    "score": h["score"],
                    "snippet": h["text"][:220],
                }
                for i, h in enumerate(hits)
            ],
            "generated_by": "llm"
            if (os.getenv("SKILLS_LLM_BASE_URL") and hits)
            else "extractive",
        }

    def _synthesize(self, question: str, hits: list[dict]) -> str:
        base_url = os.getenv("SKILLS_LLM_BASE_URL", "")
        if base_url and hits:
            try:
                import httpx

                context = "\n\n".join(
                    f"[{i + 1}] {h['text']}" for i, h in enumerate(hits)
                )
                prompt = (
                    "Answer the question using ONLY the provided context. Cite sources as [1], [2]. "
                    f"If the context is insufficient, say so.\n\nQuestion: {question}\n\nContext:\n{context}"
                )
                resp = httpx.post(
                    f"{base_url.rstrip('/')}/chat/completions",
                    json={
                        "model": os.getenv("SKILLS_LLM_MODEL", "gpt-4o-mini"),
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": 0,
                    },
                    headers={
                        "Authorization": f"Bearer {os.getenv('SKILLS_LLM_API_KEY', '')}"
                    }
                    if os.getenv("SKILLS_LLM_API_KEY")
                    else {},
                    timeout=45.0,
                )
                resp.raise_for_status()
                return resp.json()["choices"][0]["message"]["content"].strip()
            except Exception:
                pass

        import re

        q_tokens = set(re.findall(r"[a-z0-9]+", question.lower()))
        scored_sentences: list[tuple[float, str, int]] = []
        for i, hit in enumerate(hits):
            for sentence in re.split(r"(?<=[.!?])\s+", hit["text"]):
                tokens = set(re.findall(r"[a-z0-9]+", sentence.lower()))
                overlap = len(q_tokens & tokens)
                if overlap:
                    scored_sentences.append(
                        (overlap / max(len(q_tokens), 1), sentence.strip(), i + 1)
                    )
        scored_sentences.sort(key=lambda t: (-t[0], t[2]))
        top = scored_sentences[:3]
        if not top:
            return "No relevant passage found in the ingested documents."
        return " ".join(f"{s} [{n}]" for _, s, n in top)
