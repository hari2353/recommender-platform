from recsys.content.sources import (
    estimate_duration_minutes,
    extract_from_aicc,
    extract_from_html,
    extract_from_scorm,
    parse_captions,
)
from recsys.events.content_consumer import ContentEventProcessor


def test_html_extraction_strips_scripts():
    html = "<html><style>x{}</style><body><h1>Kubernetes Basics</h1><p>Learn K8s deployment.</p></body></html>"
    text = extract_from_html(html)
    assert "Kubernetes" in text and "x{}" not in text and "<p>" not in text


def test_scorm_manifest_parsing():
    xml = """<manifest xmlns="http://www.imsglobal.org/xsd/imscp_v1p1">
      <organizations><organization><title>Spark at Scale</title>
      <item><objectives><objective><description>Master Spark tuning</description></objective></objectives></item>
      </organization></organizations></manifest>"""
    out = extract_from_scorm(xml)
    assert out["title"] == "Spark at Scale"
    assert "Spark tuning" in out["objectives"][0]


def test_aicc_parsing_and_captions():
    crs = "[Course]\nCourse_Title=Docker Deep Dive\n[Course_Blocks]\nBlock_1=Containers"
    out = extract_from_aicc(crs)
    assert "Docker" in out["title"]
    vtt = "WEBVTT\n\n1\n00:00:01.000 --> 00:00:04.000\nWelcome to the course\n"
    assert parse_captions(vtt) == "Welcome to the course"


def test_duration_estimation():
    assert estimate_duration_minutes("word " * 2600) == 20
    assert estimate_duration_minutes("tiny") == 1


class FakeNormalizer:
    def __call__(self, label):
        mapping = {"Kubernetes": "esco__004", "AWS": "esco__016", "Python": "esco__007"}
        for name, sid in mapping.items():
            if name.lower() in label.lower():
                return {
                    "skill_id": sid,
                    "canonical_name": name,
                    "tier": "master",
                    "confidence": 1.0,
                }
        return {"skill_id": None, "confidence": 0.0}


def make_processor(index, chunk_store):
    tagger = __import__("recsys.content.tagger", fromlist=["SkillTagger"]).SkillTagger(
        normalize_fn=FakeNormalizer(),
        candidate_skills=[{"id": "esco__004", "name": "Kubernetes"}],
    )
    return ContentEventProcessor(index, chunk_store, tagger)


def test_content_created_tagged_indexed_chunked(stack):
    _, index, _, _, _, _, service = stack
    from recsys.rag.rag import ChunkStore

    chunks = ChunkStore()
    proc = make_processor(index, chunks)
    ok = proc.process_dict(
        {
            "event_id": "ce1",
            "type": "course.created",
            "content_id": "c900",
            "title": "Advanced Kubernetes and AWS Operations",
            "source_type": "transcript",
            "text": " ".join(
                [
                    "This course covers Kubernetes operations on AWS clusters with deep security practices."
                ]
                * 12
            ),
            "content_version": 3,
        }
    )
    assert ok is True and proc.stats["created"] == 1
    item = index.items["c900"]
    assert set(item["skills"]) >= {"esco__004"}
    assert item["content_version"] == 3
    detail = proc.last_detail
    assert detail["chunks"] > 0 and detail["tagger"] == "rules"


def test_stale_version_and_duplicate_rejected(stack):
    _, index, *_, _, _, _, service = stack[:7]
    from recsys.rag.rag import ChunkStore

    proc = make_processor(index, ChunkStore())
    base = {
        "event_id": "v1",
        "type": "course.created",
        "content_id": "c901",
        "title": "T",
        "text": "x" * 10,
        "content_version": 5,
    }
    assert proc.process_dict(base) is True
    stale = dict(base, event_id="v2", content_version=4)
    assert proc.process_dict(stale) is False
    dup = dict(base, event_id="v1")
    assert proc.process_dict(dup) is False


def test_deleted_removes_content_and_chunks(stack):
    _, index, *_ = stack[:5]
    from recsys.rag.rag import ChunkStore

    chunks_store = ChunkStore()
    proc = make_processor(index, chunks_store)
    proc.process_dict(
        {
            "event_id": "d0",
            "type": "course.created",
            "content_id": "c902",
            "title": "RAG Systems Guide",
            "source_type": "transcript",
            "text": " ".join(
                [
                    "Retrieval augmented generation explained for engineers building RAG systems today."
                ]
                * 14
            ),
        }
    )
    before = len(chunks_store.chunks)
    assert before > 0
    assert (
        proc.process_dict(
            {"event_id": "d1", "type": "course.deleted", "content_id": "c902"}
        )
        is True
    )
    assert "c902" not in index.items and len(chunks_store.chunks) < before
