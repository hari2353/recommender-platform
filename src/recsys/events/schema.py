from __future__ import annotations

from datetime import date, datetime, timezone
from enum import Enum

from pydantic import BaseModel, Field


class EventType(str, Enum):
    VIEW = "view"
    LIKE = "like"
    DISLIKE = "dislike"
    COMMENT = "comment"
    COMPLETION = "completion"
    IMPRESSION_NO_CLICK = "impression_no_click"


class Event(BaseModel):
    event_id: str
    user_id: str
    tenant_id: str = "default"
    type: EventType
    content_id: str
    recommendation_id: str | None = None
    dwell_ms: int = 0
    ts: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    locale: str = "en"

    @property
    def dt(self) -> datetime:
        return datetime.fromisoformat(self.ts)


class ImpressionRecord(BaseModel):
    recommendation_id: str
    user_id: str
    items: list[str]
    arms: list[str]
    propensities: list[float]
    model_version: str
    ts: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    resolved: bool = False


EVENT_WEIGHTS = {
    EventType.VIEW: 1.0,
    EventType.LIKE: 3.0,
    EventType.COMMENT: 4.0,
    EventType.COMPLETION: 2.5,
}


def today_iso() -> str:
    return date.today().isoformat()
