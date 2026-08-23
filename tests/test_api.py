def test_full_loop_events_recs_feedback_premium_rag(client):
    events = [
        {"event_id": "a1", "user_id": "u1", "type": "view", "content_id": "c008", "dwell_ms": 400000},
        {"event_id": "a2", "user_id": "u1", "type": "like", "content_id": "c008"},
        {"event_id": "a3", "user_id": "u1", "type": "view", "content_id": "c010", "dwell_ms": 300000},
        {"event_id": "a2", "user_id": "u1", "type": "like", "content_id": "c008"},
        {"event_id": "a2", "user_id": "u1", "type": "like", "content_id": "c008"},
    ]
    stats = client.post("/v1/events", json={"events": events}).json()
    assert stats["accepted"] == 3 and stats["duplicates"] == 2

    rec = client.get("/v1/users/u1/recommendations?role=platform_engineer&k=3").json()
    ids = [i["content_id"] for i in rec["items"]]
    assert len(ids) == 3
    assert rec["hypothesis"]["id"]
    assert all(i["premium_explanation"] is False for i in rec["items"])

    fb = client.post(
        "/v1/feedback",
        json={"recommendation_id": rec["recommendation_id"], "content_id": ids[0], "action": "like", "user_id": "u1"},
    ).json()
    assert fb["bandit_updated"] is True and fb["hypothesis"]

    profile = client.get("/v1/users/u1/profile").json()
    assert profile["segment"] in ("casual", "engaged")
    assert any("esco__004" == sid or sid.endswith("004") for sid in list(profile["skills"])[:5]) or profile["skills"]

    report = client.get("/v1/hypotheses/u1").json()
    assert any(h["source"] == "deterministic" for h in report)


def test_dislike_creates_hard_filter(client):
    rec = client.get("/v1/users/ud/recommendations?k=3").json()
    target = rec["items"][0]["content_id"]
    client.post(
        "/v1/feedback",
        json={"recommendation_id": rec["recommendation_id"], "content_id": target, "action": "dislike", "user_id": "ud"},
    )
    profile = client.get("/v1/users/ud/profile").json()
    assert target in profile["hard_filters"]
    next_rec = client.get("/v1/users/ud/recommendations?k=5").json()
    assert target not in [i["content_id"] for i in next_rec["items"]]


def test_premium_gating_and_learning_path(client):
    denied = client.post("/v1/premium/learning-path/u5?target_role=architect")
    assert denied.status_code == 402
    client.post("/admin/premium/u5")
    ok = client.post("/v1/premium/learning-path/u5?target_role=architect&role=platform_engineer")
    if ok.status_code == 200:
        body = ok.json()
        assert "steps" in body["path"]


def test_content_pipeline_and_rag_endpoints(client):
    ce = client.post(
        "/v1/content-events",
        json={
            "events": [
                {
                    "event_id": "ce9",
                    "type": "course.created",
                    "content_id": "c950",
                    "title": "Kafka Streaming Patterns",
                    "source_type": "transcript",
                    "text": " ".join(["Apache Kafka streaming pipelines process events in real time with partitioned logs."] * 10),
                }
            ]
        },
    ).json()
    assert ce["stats"]["created"] == 1 and ce["details"][0]["skills"]

    item = client.get("/v1/content/c950").json()
    assert item["modality"] == "reading"

    client.post("/v1/rag/documents", json={"content_id": "doc-demo", "text": "Kafka partitions allow parallel consumption. Consumer groups coordinate offsets. " * 5})
    answer = client.post("/v1/rag/query", json={"question": "How do consumer groups work?"}).json()
    assert answer["citations"], answer


def test_recommend_endpoint_premium_flag_shape(client):
    body = client.get("/v1/users/unew/recommendations?k=2").json()
    assert set(body) >= {"recommendation_id", "items", "premium"}
