from __future__ import annotations

from collections import deque

from pydantic import BaseModel

from ..content import sources
from ..content.tagger import SkillTagger


class ContentEvent(BaseModel):
    event_id: str
    type: str
    content_id: str
    tenant_id: str = "default"
    content_version: int = 1
    source_type: str = "transcript"
    title: str = ""
    text: str = ""
    published_at: str = ""
    duration_class: str | None = None
    locale: str = "en"
    premium: bool = False
    chunk_strategy: str = "structural"


class ContentEventProcessor:
    def __init__(
        self, index, chunk_store, tagger: SkillTagger, dedup_capacity: int = 50000
    ) -> None:
        self.index = index
        self.chunk_store = chunk_store
        self.tagger = tagger
        self._seen: deque[str] = deque(maxlen=dedup_capacity)
        self._seen_set: set[str] = set()
        self.stats = {
            "created": 0,
            "updated": 0,
            "deleted": 0,
            "duplicates": 0,
            "stale_version": 0,
            "invalid": 0,
        }

    def process_dict(self, raw: dict) -> bool:
        try:
            event = ContentEvent.model_validate(raw)
        except Exception:
            self.stats["invalid"] += 1
            return False
        if event.event_id in self._seen_set:
            self.stats["duplicates"] += 1
            return False
        self._seen.append(event.event_id)
        self._seen_set.add(event.event_id)
        if len(self._seen) == self._seen.maxlen:
            self._seen_set.discard(self._seen[0])

        if event.type == "course.deleted":
            self.index.remove(event.content_id)
            removed_chunks = self.chunk_store.remove_content(event.content_id)
            self.stats["deleted"] += 1
            self.last_detail = {"removed": True, "chunks_removed": removed_chunks}
            return True

        if event.type not in ("course.created", "course.updated"):
            self.stats["invalid"] += 1
            return False

        if self.index.stale_version(event.content_id, event.content_version):
            self.stats["stale_version"] += 1
            return False

        extracted_text = event.text
        objectives: list[str] = []
        title = event.title
        if (
            event.source_type.lower() in sources.MODALITY_BY_SOURCE
            and event.text
            and event.source_type.lower() != "transcript"
        ):
            parsed = sources.extract(event.source_type, event.text)
            extracted_text = parsed["text"] or event.text
            title = title or parsed["title"]
            objectives = parsed["objectives"]

        tagging = self.tagger.tag(
            title or event.content_id, extracted_text, source_type=event.source_type
        )
        skill_ids = [t["skill_id"] for t in tagging["skills"]]
        meta = tagging["metadata"]

        duration_class = event.duration_class or (
            "long" if meta["duration_minutes"] >= 45 else "short"
        )
        self.index.upsert(
            content_id=event.content_id,
            title=title or f"Content {event.content_id}",
            duration_class=duration_class,
            locale=event.locale,
            premium=event.premium,
            published_at=event.published_at,
            skills=skill_ids,
            objectives=objectives
            or tagging["skills"]
            and [t["name"] for t in tagging["skills"][:3]]
            or [],
            modality=meta["modality"],
            duration_minutes=meta["duration_minutes"],
            content_version=event.content_version,
            text_override=f"{title} {' '.join(skill_ids)}",
        )

        chunks_built = 0
        if len(extracted_text.split()) > 60:
            chunks_built = self.chunk_store.ingest_document(
                event.content_id, extracted_text, strategy=event.chunk_strategy
            )
        key_stat = "created" if event.type == "course.created" else "updated"
        self.stats[key_stat] += 1
        self.last_detail = {
            "skills": [t["name"] for t in tagging["skills"]],
            "tagger": meta["tagger"],
            "duration_minutes": meta["duration_minutes"],
            "modality": meta["modality"],
            "chunks": chunks_built,
            "version": event.content_version,
        }
        return True

    def process_many(self, raws) -> dict:
        details = []
        for raw in raws:
            ok = self.process_dict(raw)
            if ok and hasattr(self, "last_detail"):
                details.append({"event_id": raw.get("event_id"), **self.last_detail})
        return {"stats": dict(self.stats), "details": details}
