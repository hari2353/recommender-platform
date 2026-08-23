from datetime import datetime, timedelta, timezone

from recsys.scoring.profile_builder import ProfileStore, UserProfile


def test_decay_halves_at_half_life():
    from recsys.scoring.weights import decay

    assert decay(0, 30) == 1.0
    assert decay(30, 30) == 0.5
    assert decay(-5, 30) == 1.0


def test_profile_short_and_long_split():
    profile = UserProfile("u1")
    now = datetime.now(timezone.utc)
    profile.apply_event({"esco__004": 3.0}, now, short_half_life=30, long_half_life=180)
    norm = profile.normalized(now, 30, 180)
    assert norm["esco__004"]["short"] == 1.0
    assert norm["esco__004"]["long"] == 1.0


def test_old_affinities_decay_relative_to_recent():
    profile = UserProfile("u1")
    old = datetime.now(timezone.utc) - timedelta(days=90)
    recent = datetime.now(timezone.utc)
    profile.apply_event({"skill_a": 1.0}, old, 30, 180)
    aff = profile.skills["skill_a"]
    aff.updated_at = old
    s_old, _ = aff.effective(datetime.now(timezone.utc), 30, 180)
    profile.apply_event({"skill_b": 1.0}, recent, 30, 180)
    s_new, _ = profile.skills["skill_b"].effective(recent, 30, 180)
    assert s_new > s_old * 2


def test_segment_progression():
    p = ProfileStore().get_or_create("u9")
    assert p.segment == "new"
    for _ in range(12):
        p.apply_event({"s": 1.0}, datetime.now(timezone.utc), 30, 180)
    assert p.segment == "engaged"


def test_hard_filters_and_gdpr_delete():
    store = ProfileStore()
    p = store.get_or_create("u2")
    p.hard_filters.add("c001")
    assert "c001" in store.get("u2").hard_filters
    assert store.delete("u2") is True
    assert store.get("u2") is None
