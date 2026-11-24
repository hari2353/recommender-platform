from __future__ import annotations

import random
import uuid
from datetime import datetime, timezone

from ..content.index import ContentIndex
from ..events.schema import ImpressionRecord
from ..hypotheses.learning_path import build_learning_path
from ..hypotheses.manager import HypothesisManager
from ..scoring.profile_builder import ProfileStore
from .cmab import ThompsonBandit
from .reranker import CandidateReranker

MODEL_VERSION = "recsys-0.1.0"


def apply_repeat_cooldown(
    served_counts: dict[str, int], ranked: list, k: int, limit: int
) -> list:
    """Drop items already served `limit` times for this user; if that would
    leave fewer than k picks, top up with the suppressed items in ranked
    order so the slate is never short."""
    fresh = [c for c in ranked if served_counts.get(c[0], 0) < limit]
    if len(fresh) >= k:
        return fresh[:k]
    suppressed = [c for c in ranked if served_counts.get(c[0], 0) >= limit]
    return (fresh + suppressed)[:k]


class RecommenderService:
    def __init__(
        self,
        index: ContentIndex,
        profiles: ProfileStore,
        hypotheses: HypothesisManager,
        bandit: ThompsonBandit | None = None,
        diversity_lambda: float = 0.35,
        repeat_serve_limit: int = 3,
    ) -> None:
        self.index = index
        self.profiles = profiles
        self.hypotheses = hypotheses
        self.bandit = bandit or ThompsonBandit()
        self.reranker = CandidateReranker(index)
        self.diversity_lambda = diversity_lambda
        self.repeat_serve_limit = repeat_serve_limit
        self._rng = random.Random(13)
        self.impressions: dict[str, ImpressionRecord] = {}
        self.impression_hypothesis: dict[str, str] = {}
        self.popularity: dict[str, float] = {}

    def _query_text_for_hypothesis(self, hypothesis, profile) -> str:
        if hypothesis.type.value == "skill_interest":
            return (
                hypothesis.target.replace("__", " ")
                .replace("esco ", "")
                .replace("_", " ")
            )
        if hypothesis.type.value == "next_role_prep":
            role = self.hypotheses.ladder.get(hypothesis.target)
            if role:
                gap = self.hypotheses.ladder.gap_skills(
                    profile.role or "", hypothesis.target
                )
                skills = list(gap or role.skills)
                return (
                    " ".join(s.replace("__", " ").replace("_", " ") for s in skills)
                    + f" {role.title}"
                )
        role = self.hypotheses.ladder.get(profile.role)
        if role:
            return " ".join(role.skills.keys()).replace("__", " ").replace("_", " ")
        return (
            " ".join(list(profile.blend().keys())[:5])
            .replace("__", " ")
            .replace("_", " ")
        )

    def _candidates_from_hypothesis(
        self, hypothesis, profile, allow_premium: bool, k: int = 5
    ) -> list[str]:
        query_text = self._query_text_for_hypothesis(hypothesis, profile)
        blend = profile.blend()
        role = self.hypotheses.ladder.get(profile.role)

        if hypothesis.type.value == "skill_interest":
            skill_query = {
                hypothesis.target: max(blend.get(hypothesis.target, 0.5), 0.5)
            }
            for sid, score in list(blend.items())[:4]:
                if sid != hypothesis.target:
                    skill_query[sid] = max(skill_query.get(sid, 0.0), score * 0.7)
            if role:
                for sid, w in role.skills.items():
                    skill_query[sid] = max(skill_query.get(sid, 0.0), w * 0.4)
            base = self.index.search_by_skills(skill_query, top_n=15)
        elif hypothesis.type.value == "next_role_prep":
            gaps = self.hypotheses.ladder.gap_skills(
                profile.role or "", hypothesis.target
            )
            weights = gaps or (
                self.hypotheses.ladder.get(hypothesis.target).skills
                if self.hypotheses.ladder.get(hypothesis.target)
                else {}
            )
            base = self.index.search_by_skills(weights or blend, top_n=15)
        elif hypothesis.type.value == "depth_in_role" and role:
            weights = dict(role.skills)
            for sid, score in list(blend.items())[:3]:
                weights[sid] = max(weights.get(sid, 0.0), score * 0.9)
            base = self.index.search_by_skills(weights, top_n=15)
        else:
            weights = dict(blend)
            if role:
                for sid, w in role.skills.items():
                    weights[sid] = max(weights.get(sid, 0.0), w * 0.8)
            base = self.index.search_by_skills(weights, top_n=15)

        ids = [cid for cid, _ in base]
        ids = [cid for cid in ids if cid not in profile.hard_filters]
        allow_premium_final = (
            allow_premium and hypothesis.type.value != "next_role_prep"
        )
        ids = self.index.filter_items(
            ids, locale="en", allow_premium=allow_premium_final
        )

        reranked = self.reranker.rerank(ids, query_text, popularity=self.popularity)
        arm_bonus = []
        for cid, score in reranked:
            arm_key = f"{profile.segment}:{self.index.items[cid].primary_skill}"
            drawn = self.bandit.arm(arm_key).sample(self.bandit.rng)
            arm_bonus.append(
                (cid, round(score * (0.7 + 0.3 * drawn), 4), arm_key, drawn)
            )
        arm_bonus.sort(key=lambda t: -t[1])
        if len(arm_bonus) < k:
            seen_ids = {c[0] for c in arm_bonus}
            weights = dict(blend)
            if role:
                for sid, w in role.skills.items():
                    weights[sid] = max(weights.get(sid, 0.0), w * 0.6)
            generic = self.index.search_by_skills(
                weights
                or {sid: 1.0 for sid in list(self.index.by_skills_lookup())[:5]},
                top_n=20,
            )
            generic_ids = [
                cid
                for cid, _ in generic
                if cid not in seen_ids and cid not in profile.hard_filters
            ]
            generic_ids = self.index.filter_items(
                generic_ids, locale="en", allow_premium=allow_premium
            )
            extra = self.reranker.rerank(
                generic_ids, query_text, popularity=self.popularity
            )
            for cid, score in extra:
                arm_key = f"{profile.segment}:{self.index.items[cid].primary_skill}"
                drawn = self.bandit.arm(arm_key).sample(self.bandit.rng)
                arm_bonus.append(
                    (cid, round(score * (0.7 + 0.3 * drawn), 4), arm_key, drawn)
                )
        return arm_bonus

    def _mmr_select(self, scored: list[tuple], k: int) -> list[tuple]:
        selected: list[tuple] = []
        pool = list(scored)
        while pool and len(selected) < k:
            best = None
            best_score = -1e9
            for cand in pool:
                cid = cand[0]
                item = self.index.items[cid]
                diversity_penalty = 0.0
                for chosen in selected:
                    other = self.index.items[chosen[0]]
                    shared = len(set(item["skills"]) & set(other["skills"]))
                    union = len(set(item["skills"]) | set(other["skills"])) or 1
                    diversity_penalty = max(diversity_penalty, shared / union)
                mmr = (1 - self.diversity_lambda) * cand[
                    1
                ] - self.diversity_lambda * diversity_penalty
                if mmr > best_score:
                    best_score = mmr
                    best = cand
            if best is None:
                break
            selected.append(best)
            pool.remove(best)
        return selected

    def serve(
        self,
        user_id: str,
        tenant_id: str = "default",
        k: int = 5,
        allow_premium: bool = True,
        ladder_role_id: str | None = None,
    ) -> dict:
        profile = self.profiles.get_or_create(user_id, tenant_id)
        if ladder_role_id and not profile.role:
            profile.role = ladder_role_id
        self.hypotheses.ensure_base_hypotheses(profile, ladder_role_id=ladder_role_id)
        epsilon = (
            0.4
            if profile.segment == "new"
            else (0.25 if profile.segment == "casual" else None)
        )
        hypothesis = self.hypotheses.sample(user_id, epsilon=epsilon)
        rec_id = uuid.uuid4().hex[:12]

        if hypothesis is None:
            popular = [
                (cid, score)
                for cid, score in sorted(self.popularity.items(), key=lambda kv: -kv[1])
                if cid not in profile.hard_filters
            ]
            popular = apply_repeat_cooldown(
                profile.served_counts, popular, k, self.repeat_serve_limit
            )
            for cid, _ in popular:
                profile.served_counts[cid] = profile.served_counts.get(cid, 0) + 1
            return {
                "recommendation_id": rec_id,
                "user_id": user_id,
                "hypothesis": None,
                "items": [
                    {
                        "content_id": cid,
                        "title": self.index.items[cid]["title"]
                        if cid in self.index.items
                        else "",
                        "score": score,
                    }
                    for cid, score in popular
                ],
            }

        candidates = self._candidates_from_hypothesis(
            hypothesis, profile, allow_premium, k=k
        )
        candidates = apply_repeat_cooldown(
            profile.served_counts, candidates, k, self.repeat_serve_limit
        )
        selected = self._mmr_select(candidates, k)
        for cid, *_ in selected:
            profile.served_counts[cid] = profile.served_counts.get(cid, 0) + 1

        if selected and self._rng.random() < 0.12:
            shown = {c[0] for c in selected}
            pool = [
                cid
                for cid in self.index.ids
                if cid not in shown and cid not in profile.hard_filters
            ]
            if pool:
                serendipity = self._rng.choice(pool)
                selected[-1] = (
                    serendipity,
                    round(selected[-1][1] * 0.5, 4),
                    f"explore:{self.index.items[serendipity].primary_skill}",
                    0.3,
                )

        record = ImpressionRecord(
            recommendation_id=rec_id,
            user_id=user_id,
            items=[c[0] for c in selected],
            arms=[c[2] for c in selected],
            propensities=[round(min(c[3], 0.99), 4) for c in selected],
            model_version=MODEL_VERSION,
        )
        self.impressions[rec_id] = record
        self.impression_hypothesis[rec_id] = hypothesis.id

        return {
            "recommendation_id": rec_id,
            "user_id": user_id,
            "hypothesis": {
                "id": hypothesis.id,
                "type": hypothesis.type.value,
                "statement": hypothesis.statement,
            },
            "items": [
                {
                    "content_id": c[0],
                    "title": self.index.items[c[0]]["title"],
                    "score": c[1],
                    "duration_class": self.index.items[c[0]]["duration_class"],
                    "premium": self.index.items[c[0]]["premium"],
                }
                for c in selected
            ],
        }

    def resolve_pending_impressions(
        self, user_id: str, feedback_given: set[str]
    ) -> int:
        now = datetime.now(timezone.utc)
        resolved = 0
        for record in self.impressions.values():
            if record.user_id != user_id or record.resolved:
                continue
            age_hours = (now - datetime.fromisoformat(record.ts)).total_seconds() / 3600
            if age_hours < 24:
                continue
            unclicked = [i for i in record.items if i not in feedback_given]
            for cid in unclicked:
                idx = record.items.index(cid)
                arm_key = record.arms[idx]
                self.bandit.update(arm_key, success=False, weight=0.5)
        record.resolved = True
        resolved += 1
        if len(self.impressions) > 100000:
            for key in [k for k, v in self.impressions.items() if v.resolved][:20000]:
                del self.impressions[key]
                self.impression_hypothesis.pop(key, None)
        return resolved

    def apply_feedback(
        self,
        user_id: str,
        recommendation_id: str,
        content_id: str,
        success: bool,
        weight: float = 1.0,
    ) -> dict:
        record = self.impressions.get(recommendation_id)
        outcome: dict = {"bandit_updated": False}
        if record and content_id in record.items:
            idx = record.items.index(content_id)
            arm_key = record.arms[idx]
            self.bandit.update(arm_key, success, weight)
            outcome["bandit_updated"] = True
            outcome["arm"] = arm_key
        hyp_id = self.impression_hypothesis.get(recommendation_id)
        if hyp_id:
            self.hypotheses.apply_feedback(user_id, hyp_id, success, weight)
            outcome["hypothesis"] = hyp_id
        if record:
            remaining = [i for i in record.items if i != content_id]
            for cid in remaining:
                idx = record.items.index(cid)
                self.bandit.update(record.arms[idx], success=False, weight=0.25)
            record.resolved = True
        return outcome

    def apply_no_click(self, recommendation_id: str, weight: float = 0.25) -> bool:
        record = self.impressions.get(recommendation_id)
        if not record or record.resolved:
            return False
        for idx in range(len(record.items)):
            self.bandit.update(record.arms[idx], success=False, weight=weight)
        hyp_id = self.impression_hypothesis.get(recommendation_id)
        if hyp_id:
            self.hypotheses.apply_feedback(
                record.user_id, hyp_id, success=False, weight=weight
            )
        record.resolved = True
        return True

    def learning_path(self, user_id: str, target_role_id: str) -> dict:
        profile = self.profiles.get(user_id)
        lookup = self.index.by_skills_lookup()
        return build_learning_path(
            self.hypotheses.ladder,
            profile.role if profile else None,
            target_role_id,
            lookup,
        )

    def set_popularity(self, popularity: dict[str, float]) -> None:
        self.popularity = popularity
