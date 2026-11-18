from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from fastapi import Depends, FastAPI, Header, HTTPException, Query
from pydantic import BaseModel, Field

from ..config import Settings
from ..content.index import ContentIndex
from ..content.normalizer_lite import NormalizerLite
from ..content.tagger import SkillTagger
from ..engine.cmab import ThompsonBandit
from ..engine.recommender import RecommenderService
from ..events.bus import get_bus
from ..events.content_consumer import ContentEventProcessor
from ..events.consumer import EventProcessor
from ..hypotheses.manager import HypothesisManager, RoleLadder
from ..premium.entitlement import EntitlementService
from ..premium.explainer import LlmExplainer
from ..rag.rag import ChunkStore, RagService
from ..scoring.profile_builder import ProfileStore


class EventsIn(BaseModel):
    events: list[dict] = Field(max_length=500)


class FeedbackIn(BaseModel):
    recommendation_id: str = Field(max_length=64)
    content_id: str = Field(max_length=128)
    action: str = Field(max_length=16)
    user_id: str = Field(max_length=128)


class RagQueryIn(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    k: int = Field(default=4, ge=1, le=20)
    content_filter: list[str] | None = Field(default=None, max_length=50)


class RagIngestIn(BaseModel):
    content_id: str = Field(max_length=128)
    text: str = Field(min_length=1, max_length=400000)
    strategy: Literal["structural", "semantic", "agentic"] = "structural"


def require_admin(x_admin_token: str | None = Header(default=None)) -> None:
    expected = os.getenv("RECSYS_ADMIN_TOKEN")
    if expected and x_admin_token != expected:
        raise HTTPException(status_code=401, detail="admin token required")


def wire(data_dir: Path | None = None):
    settings = Settings(data_dir=data_dir)
    index = ContentIndex()
    index.load_csv(settings.data_dir / "content_catalog.csv")
    ladder = RoleLadder.load_csv(settings.data_dir / "roles.csv")
    profiles = ProfileStore()
    hypotheses = HypothesisManager(ladder)
    entitlements = EntitlementService()
    explainer = LlmExplainer()
    bandit = ThompsonBandit()

    def on_event(event):
        profile = profiles.get_or_create(event.user_id, event.tenant_id)
        if event.type.value == "dislike":
            item = index.items.get(event.content_id)
            if item:
                profile.hard_filters.add(event.content_id)
            return
        if event.type.value == "completion":
            # finished content is never re-served (serving hygiene)
            item = index.items.get(event.content_id)
            if item:
                profile.hard_filters.add(event.content_id)
            return
        if event.type.value == "impression_no_click":
            return
        from ..events.schema import EVENT_WEIGHTS

        weight = EVENT_WEIGHTS[event.type] * (1.0 + min(event.dwell_ms / 60000, 2.0) * 0.25)
        profile = profiles.get_or_create(event.user_id, event.tenant_id)
        item = index.items.get(event.content_id)
        skill_weights = {}
        for sid in (item["skills"] if item else []):
            skill_weights[sid] = weight
        if skill_weights:
            from datetime import datetime, timezone

            profile.apply_event(skill_weights, datetime.now(timezone.utc), settings.short_half_life_days, settings.long_half_life_days)
            hypotheses.ensure_base_hypotheses(profile)

    processor = EventProcessor(on_valid_event=on_event, dedup_capacity=settings.event_dedup_capacity, late_event_days=settings.late_event_days)
    bus = get_bus(settings.bus_backend, url=settings.pulsar_url, topic=settings.pulsar_topic)
    bus.subscribe(processor.process_dict)

    service = RecommenderService(index, profiles, hypotheses, bandit=bandit, diversity_lambda=settings.diversity_lambda)
    popularity = {cid: 0.4 for cid in index.ids}
    service.set_popularity(popularity)

    taxonomy_csv = settings.data_dir / "taxonomy_skills.csv"
    lite = NormalizerLite(taxonomy_csv)
    candidate_skills = [{"id": sid, "name": name} for sid, name in lite.entries]
    tagger = SkillTagger(normalize_fn=lite, candidate_skills=candidate_skills)
    chunk_store = ChunkStore(provider_name=__import__("os").getenv("RECSYS_EMBEDDING_PROVIDER", "hashing"))
    rag_service = RagService(chunk_store)
    content_processor = ContentEventProcessor(index, chunk_store, tagger)

    return (
        settings,
        processor,
        bus,
        service,
        profiles,
        hypotheses,
        entitlements,
        explainer,
        index,
        content_processor,
        chunk_store,
        rag_service,
    )


def create_app(data_dir: Path | None = None) -> FastAPI:
    app = FastAPI(title="Recommender Platform", version="0.1.0")
    (
        settings,
        processor,
        bus,
        service,
        profiles,
        hypotheses,
        entitlements,
        explainer,
        index,
        content_processor,
        chunk_store,
        rag_service,
    ) = wire(data_dir)

    @app.middleware("http")
    async def security_headers(request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        return response

    @app.get("/healthz")
    def healthz() -> dict:
        return {"status": "ok", "bus": settings.bus_backend}

    @app.post("/v1/events")
    def ingest(body: EventsIn) -> dict:
        stats = processor.process_many(body.events)
        return stats

    @app.get("/v1/users/{user_id}/recommendations")
    def recommend(user_id: str, k: int = Query(5, ge=1, le=50), role: str | None = Query(None, max_length=64), tenant_id: str = Query("default", max_length=64)) -> dict:
        premium = entitlements.is_premium(user_id, tenant_id)
        result = service.serve(user_id, tenant_id, k=k, allow_premium=True, ladder_role_id=role)
        hyp = result.get("hypothesis") or {}
        enriched = []
        for item in result["items"]:
            entry = dict(item)
            if premium:
                ctx = {
                    "top_skills": list(profiles.get_or_create(user_id, tenant_id).blend())[:3],
                    "hypothesis_type": hyp.get("type"),
                    "hypothesis_target": hyp.get("id", "").split(":")[-1],
                }
                entry["explanation"] = explainer.explain_recommendation(ctx, entry)
            else:
                entry["premium_explanation"] = False
            enriched.append(entry)
        result["items"] = enriched
        result["premium"] = premium
        return result

    @app.post("/v1/feedback")
    def feedback(body: FeedbackIn) -> dict:
        if body.action not in ("like", "dislike"):
            raise HTTPException(status_code=422, detail="action must be like|dislike")
        success = body.action == "like"
        outcome = service.apply_feedback(body.user_id, body.recommendation_id, body.content_id, success, weight=1.0)
        if not success:
            profile = profiles.get_or_create(body.user_id)
            profile.hard_filters.add(body.content_id)
        outcome["status"] = "ok"
        return outcome

    @app.get("/v1/users/{user_id}/profile")
    def get_profile(user_id: str, tenant_id: str = "default") -> dict:
        profile = profiles.get(user_id, tenant_id)
        if profile is None:
            raise HTTPException(status_code=404, detail="profile not found")
        return {
            "user_id": user_id,
            "role": profile.role,
            "segment": profile.segment,
            "skills": profile.normalized(datetime.now(timezone.utc), settings.short_half_life_days, settings.long_half_life_days),
            "hard_filters": sorted(profile.hard_filters),
        }

    @app.delete("/v1/users/{user_id}")
    def delete_user(user_id: str, tenant_id: str = "default") -> dict:
        deleted = profiles.delete(user_id, tenant_id)
        return {"deleted": deleted}

    @app.get("/v1/hypotheses/{user_id}")
    def hypothesis_report(user_id: str) -> list[dict]:
        return hypotheses.report(user_id)

    @app.post("/v1/premium/learning-path/{user_id}")
    def learning_path(user_id: str, target_role: str, tenant_id: str = "default") -> dict:
        if not entitlements.is_premium(user_id, tenant_id):
            raise HTTPException(status_code=402, detail="premium feature")
        path = service.learning_path(user_id, target_role)
        narrative = explainer.generate_learning_path_narrative(path)
        return {"path": path, "narrative": narrative}

    @app.post("/admin/premium/{user_id}", dependencies=[Depends(require_admin)])
    def grant_premium(user_id: str, tenant_id: str = Query("default", max_length=64)) -> dict:
        if len(user_id) > 128:
            raise HTTPException(status_code=422, detail="user_id too long")
        entitlements.grant(user_id, tenant_id)
        return {"user_id": user_id, "premium": True}

    @app.post("/admin/rebuild-index", dependencies=[Depends(require_admin)])
    def rebuild_index() -> dict:
        index.rebuild()
        return {"content_items": len(index.ids)}

    @app.post("/v1/content-events")
    def content_events(body: EventsIn) -> dict:
        return content_processor.process_many(body.events)

    @app.get("/v1/content/{content_id}")
    def get_content(content_id: str) -> dict:
        item = index.items.get(content_id)
        if item is None:
            raise HTTPException(status_code=404, detail="unknown content")
        return {k: v for k, v in item.items() if k != "text"}

    @app.post("/v1/rag/query")
    def rag_query(body: RagQueryIn) -> dict:
        if not body.question.strip():
            raise HTTPException(status_code=422, detail="question required")
        return rag_service.ask(body.question, k=body.k, content_filter=body.content_filter)

    @app.post("/v1/rag/documents")
    def rag_ingest(body: RagIngestIn) -> dict:
        chunks = chunk_store.ingest_document(body.content_id, body.text, strategy=body.strategy)
        return {"content_id": body.content_id, "chunks": chunks}

    return app


app = create_app()


