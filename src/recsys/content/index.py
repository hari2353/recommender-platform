from __future__ import annotations

import csv
import hashlib
import re
from datetime import datetime
from pathlib import Path

import numpy as np

TOKEN_RE = re.compile(r"[a-z0-9]+")


def hashing_embed(text: str, dim: int = 256) -> np.ndarray:
    vec = np.zeros(dim, dtype=np.float32)
    tokens = TOKEN_RE.findall(text.lower())
    grams = tokens + [f"{a} {b}" for a, b in zip(tokens, tokens[1:])]
    for gram in grams:
        digest = hashlib.md5(gram.encode("utf-8")).digest()
        head = int.from_bytes(digest[:8], "little")
        sign = 1.0 if digest[8] & 1 else -1.0
        vec[head % dim] += sign
    norm = float(np.linalg.norm(vec))
    return vec / norm if norm else vec


class ContentItem(dict):
    @property
    def content_id(self) -> str:
        return self["content_id"]

    @property
    def primary_skill(self) -> str:
        skills = self.get("skills") or []
        return skills[0] if skills else "none"


class ContentIndex:
    def __init__(self, dim: int = 256) -> None:
        self.dim = dim
        self.items: dict[str, ContentItem] = {}
        self.matrix: np.ndarray | None = None
        self.ids: list[str] = []

    def load_csv(self, path: Path) -> int:
        with path.open(newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                self.upsert(
                    content_id=row["content_id"],
                    title=row["title"],
                    duration_class=row.get("duration_class", "short"),
                    locale=row.get("locale", "en"),
                    premium=row.get("premium", "").strip().lower() == "true",
                    published_at=row.get("published_at", ""),
                    skills=[s.strip() for s in row.get("skills", "").split("|") if s.strip()],
                    content_version=int(row.get("content_version") or 1),
                )
        return len(self.items)

    def upsert(
        self,
        content_id: str,
        title: str,
        duration_class: str = "short",
        locale: str = "en",
        premium: bool = False,
        published_at: str = "",
        skills: list[str] | None = None,
        objectives: list[str] | None = None,
        modality: str | None = None,
        duration_minutes: int | None = None,
        content_version: int = 1,
        text_override: str | None = None,
    ) -> ContentItem:
        skills = skills or []
        item = ContentItem(
            content_id=content_id,
            title=title,
            duration_class=duration_class,
            locale=locale,
            premium=premium,
            published_at=published_at,
            skills=skills,
            objectives=objectives or [],
            modality=modality or ("reading" if duration_class == "long" else "multimedia"),
            duration_minutes=duration_minutes or 0,
            content_version=content_version,
            text=text_override or f"{title} {' '.join(skills)}",
        )
        self.items[content_id] = item
        self.rebuild()
        return item

    def remove(self, content_id: str) -> bool:
        if content_id not in self.items:
            return False
        del self.items[content_id]
        self.rebuild()
        return True

    def stale_version(self, content_id: str, incoming_version: int) -> bool:
        existing = self.items.get(content_id)
        return existing is not None and incoming_version <= int(existing.get("content_version", 0))

    def rebuild(self) -> None:
        self.ids = sorted(self.items)
        self.matrix = np.stack([hashing_embed(self.items[cid]["text"], self.dim) for cid in self.ids]) if self.ids else np.zeros((0, self.dim), dtype=np.float32)

    def search_by_skills(self, skill_weights: dict[str, float], top_n: int = 20) -> list[tuple[str, float]]:
        scored = []
        for cid in self.ids:
            item = self.items[cid]
            overlap = sum(w * 1.0 for sid, w in skill_weights.items() if sid in item["skills"])
            if overlap > 0:
                scored.append((cid, round(float(overlap), 4)))
        scored.sort(key=lambda pair: -pair[1])
        return scored[:top_n]

    def similar(self, content_id: str, k: int = 5) -> list[tuple[str, float]]:
        if content_id not in self.items or self.matrix is None or len(self.ids) == 0:
            return []
        idx = self.ids.index(content_id)
        sims = self.matrix @ self.matrix[idx]
        order = np.argsort(-sims)[: k + 1]
        return [(self.ids[i], float(sims[i])) for i in order if self.ids[i] != content_id][:k]

    def filter_items(self, cids: list[str], locale: str | None = None, allow_premium: bool = True, duration_class: str | None = None) -> list[str]:
        out = []
        now_iso = datetime.now().date().isoformat()
        for cid in cids:
            item = self.items.get(cid)
            if item is None:
                continue
            if item["premium"] and not allow_premium:
                continue
            if locale and item["locale"] != locale:
                continue
            if duration_class and item["duration_class"] != duration_class:
                continue
            if item["published_at"] and item["published_at"] > now_iso:
                continue
            out.append(cid)
        return out

    def by_skills_lookup(self) -> dict[str, list[dict]]:
        lookup: dict[str, list[dict]] = {}
        for cid in self.ids:
            item = self.items[cid]
            for sid in item["skills"]:
                lookup.setdefault(sid, []).append(item)
        return lookup

    def popularity(self) -> dict[str, float]:
        return {cid: 0.5 for cid in self.ids}
