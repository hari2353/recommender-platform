from __future__ import annotations

import re

import numpy as np

_HEADING_RE = re.compile(r"^(#{1,3}\s+.+|[A-Z][A-Z\s\-]{6,})$")
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def _sentences(text: str) -> list[str]:
    out: list[str] = []
    for line in text.splitlines():
        if line.strip():
            out.extend(s for s in _SENT_SPLIT_RE.split(line.strip()) if s)
    return out


def structural_chunks(text: str, target_words: int = 120, overlap_words: int = 25) -> list[dict]:
    sections: list[tuple[str, list[str]]] = []
    current_heading = ""
    current: list[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if _HEADING_RE.match(line):
            if current:
                sections.append((current_heading, current))
                current = []
            current_heading = line.lstrip("# ").strip()
            continue
        current.extend(_SENT_SPLIT_RE.split(line))
    if current:
        sections.append((current_heading, current))

    chunks: list[dict] = []
    for section, sentences in sections:
        buffer: list[str] = []
        count = 0
        idx = 0
        for sentence in sentences:
            buffer.append(sentence)
            count += len(sentence.split())
            if count >= target_words:
                chunks.append({"text": " ".join(buffer).strip(), "section": section, "index": idx})
                idx += 1
                tail: list[str] = []
                tail_count = 0
                for s in reversed(buffer):
                    tail.insert(0, s)
                    tail_count += len(s.split())
                    if tail_count >= overlap_words:
                        break
                buffer, count = tail, tail_count
        if buffer and " ".join(buffer).strip():
            chunks.append({"text": " ".join(buffer).strip(), "section": section, "index": idx})

    for i, chunk in enumerate(chunks):
        chunk["chunk_id"] = f"ch{i:04d}"
        chunk["strategy"] = "structural"
    return chunks


def semantic_chunks(text: str, provider, threshold: float = 0.35, min_words: int = 40, max_words: int = 220) -> list[dict]:
    sentences = [s for s in _sentences(text) if len(s.split()) >= 3]
    if not sentences:
        return []
    embeddings = provider.embed_batch(sentences)

    chunks: list[dict] = []
    current: list[str] = []
    current_vecs: list[np.ndarray] = []
    current_words = 0

    def flush() -> None:
        nonlocal current, current_vecs, current_words
        if current:
            chunks.append({"text": " ".join(current).strip(), "section": "", "index": len(chunks), "strategy": "semantic"})
            current, current_vecs, current_words = [], [], 0

    for sentence, vec in zip(sentences, embeddings):
        n_words = len(sentence.split())
        if current and current_words + n_words > max_words:
            flush()
        if current_vecs:
            centroid = np.mean(np.stack(current_vecs), axis=0)
            denom = (np.linalg.norm(centroid) * np.linalg.norm(vec)) or 1.0
            similarity = float(np.dot(centroid, vec) / denom)
            if similarity < threshold and current_words >= min_words:
                flush()
        current.append(sentence)
        current_vecs.append(vec)
        current_words += n_words
    flush()

    merged: list[dict] = []
    for chunk in chunks:
        if merged and len(chunk["text"].split()) < min_words // 2:
            merged[-1]["text"] = (merged[-1]["text"] + " " + chunk["text"]).strip()
        else:
            merged.append(chunk)
    for i, chunk in enumerate(merged):
        chunk["chunk_id"] = f"ch{i:04d}"
    return chunks


AGENTIC_PROMPT = (
    "You are a document chunking agent. Split the text into self-contained, semantically coherent chunks. "
    "Each chunk must be understandable without the others. Keep every original sentence.\n"
    'Reply JSON only: {"chunks": ["chunk 1 full text", "chunk 2 full text", ...]}\n\n'
    "Text:\n{text}"
)


def agentic_chunks(text: str, llm_base_url: str, llm_model: str = "gpt-4o-mini", api_key: str = "") -> list[dict]:
    import json as _json

    import httpx

    payload = {
        "model": llm_model,
        "messages": [{"role": "user", "content": AGENTIC_PROMPT.format(text=text[:6000])}],
        "temperature": 0,
    }
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    resp = httpx.post(f"{llm_base_url.rstrip('/')}/chat/completions", json=payload, headers=headers, timeout=60.0)
    resp.raise_for_status()
    content = resp.json()["choices"][0]["message"]["content"].strip()
    content = re.sub(r"^```(?:json)?\s*|\s*```$", "", content)
    data = _json.loads(content)
    items = data.get("chunks", []) if isinstance(data, dict) else data
    chunks = [
        {"text": str(c).strip(), "section": "", "index": i, "strategy": "agentic"}
        for i, c in enumerate(items)
        if str(c).strip()
    ]
    for i, chunk in enumerate(chunks):
        chunk["chunk_id"] = f"ch{i:04d}"
    return chunks
