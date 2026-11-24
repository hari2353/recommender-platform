from recsys.events.consumer import EventProcessor
from recsys.events.schema import EventType


def base_event(**overrides):
    payload = {
        "event_id": "e1",
        "user_id": "u1",
        "type": "view",
        "content_id": "c001",
        "dwell_ms": 30000,
    }
    payload.update(overrides)
    return payload


def test_valid_event_accepted_and_dispatched():
    seen = []
    p = EventProcessor(on_valid_event=lambda e: seen.append(e))
    assert p.process_dict(base_event()) is True
    assert seen[0].type == EventType.VIEW


def test_duplicate_events_deduped():
    p = EventProcessor()
    assert p.process_dict(base_event()) is True
    assert p.process_dict(base_event()) is False
    assert p.stats["duplicates"] == 1
    assert p.stats["accepted"] == 1


def test_invalid_payload_counted_not_raised():
    p = EventProcessor()
    assert p.process_dict({"event_id": "x"}) is False
    assert p.stats["invalid"] == 1


def test_late_events_rejected_by_watermark():
    p = EventProcessor(late_event_days=7)
    assert (
        p.process_dict(base_event(event_id="old", ts="2020-01-01T00:00:00+00:00"))
        is False
    )
    assert p.stats["late"] == 1


def test_dislike_is_valid_without_weight():
    p = EventProcessor()
    assert p.process_dict(base_event(event_id="d1", type="dislike")) is True


def test_process_many_returns_stats():
    p = EventProcessor()
    stats = p.process_many(
        [base_event(), base_event(event_id="e2", content_id="c002"), {"bad": 1}]
    )
    assert stats["accepted"] == 2 and stats["invalid"] == 1
