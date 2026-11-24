from __future__ import annotations

import csv
import re
from pathlib import Path

import numpy as np

from ..rag.embeddings import get_provider


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


class NormalizerLite:
    """Self-contained stand-in for skills-engine resolution: exact > alias > blended vector+lexical."""

    def __init__(
        self, taxonomy_csv: Path, dim: int = 256, threshold: float = 0.6
    ) -> None:
        self.threshold = threshold
        self.exact: dict[str, str] = {}
        self.alias: dict[str, str] = {}
        entries: list[tuple[str, str]] = []
        with open(taxonomy_csv, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                sid, name = row["skill_id"], row["name"]
                self.exact.setdefault(_key(name), sid)
                for alias in (row.get("aliases") or "").split("|"):
                    if alias.strip():
                        self.alias.setdefault(_key(alias), sid)
                        entries.append((sid, alias.strip()))
                entries.append((sid, name))
        provider = get_provider("hashing", dim=dim)
        self.provider = provider
        self.entries = entries
        self.matrix = (
            provider.embed_batch([e[1] for e in entries])
            if entries
            else np.zeros((0, dim), dtype=np.float32)
        )

    def __call__(self, label: str) -> dict:
        key = _key(label)
        for table, method, confidence in (
            (self.exact, "exact", 1.0),
            (self.alias, "alias", 0.95),
        ):
            hit = table.get(key)
            if hit:
                return {
                    "skill_id": hit,
                    "canonical_name": _name_for(self.entries, hit),
                    "tier": "master",
                    "confidence": confidence,
                    "method": method,
                }
        qv = self.provider.embed(label)
        sims = self.matrix @ qv
        best = int(np.argmax(sims)) if len(sims) else -1
        score = float(sims[best]) if best >= 0 else 0.0
        lexical = (
            _f1(_tokens(label), _tokens(self.entries[best][1])) if best >= 0 else 0.0
        )
        blended = 0.5 * score + 0.5 * lexical
        if blended >= self.threshold and best >= 0:
            return {
                "skill_id": self.entries[best][0],
                "canonical_name": self.entries[best][1],
                "tier": "master",
                "confidence": round(blended, 4),
                "method": "vector_reranked",
            }
        return {"skill_id": None, "confidence": 0.0}


def _key(text: str) -> str:
    import unicodedata

    text = unicodedata.normalize("NFKC", text).casefold()
    text = re.sub(r"[^\w\s]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _f1(q: set[str], c: set[str]) -> float:
    if not q or not c:
        return 0.0
    overlap = len(q & c)
    p, r = overlap / len(c), overlap / len(q)
    return 2 * p * r / (p + r) if p + r else 0.0


def _name_for(entries, skill_id: str) -> str:
    for sid, name in entries:
        if sid == skill_id:
            return name
    return skill_id
