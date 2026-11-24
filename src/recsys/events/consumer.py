from __future__ import annotations

from collections import deque
from datetime import datetime, timezone
from typing import Callable

from .schema import Event, EVENT_WEIGHTS


class EventProcessor:
    def __init__(
        self,
        on_valid_event: Callable | None = None,
        dedup_capacity: int = 100000,
        late_event_days: int = 7,
    ) -> None:
        self.on_valid_event = on_valid_event
        self.late_event_days = late_event_days
        self._seen: deque[str] = deque(maxlen=dedup_capacity)
        self.dedup_capacity = dedup_capacity
        self._seen_set: set[str] = set()
        self.stats = {"accepted": 0, "duplicates": 0, "invalid": 0, "late": 0}

    def process_dict(self, raw: dict) -> bool:
        try:
            event = Event.model_validate(raw)
        except Exception:
            self.stats["invalid"] += 1
            return False

        if event.event_id in self._seen_set:
            self.stats["duplicates"] += 1
            return False
        self._seen.append(event.event_id)
        self._seen_set.add(event.event_id)
        if len(self._seen) == self._seen.maxlen:
            evicted = self._seen[0]
            self._seen_set.discard(evicted)

        age_days = (datetime.now(timezone.utc) - event.dt).total_seconds() / 86400
        if age_days > self.late_event_days:
            self.stats["late"] += 1
            return False

        if event.type not in EVENT_WEIGHTS and event.type.value != "dislike":
            self.stats["invalid"] += 1
            return False

        self.stats["accepted"] += 1
        if self.on_valid_event is not None:
            self.on_valid_event(event)
        return True

    def process_many(self, raws) -> dict:
        for raw in raws:
            self.process_dict(raw)
        return dict(self.stats)
