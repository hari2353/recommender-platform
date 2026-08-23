from __future__ import annotations

import json
import random
import re
from pathlib import Path

from ..scoring.profile_builder import UserProfile
from .models import Hypothesis, HypothesisStatus, HypothesisType, RoleDefinition


class RoleLadder:
    def __init__(self, roles: dict[str, RoleDefinition]) -> None:
        self.roles = roles

    @classmethod
    def load_csv(cls, path: Path) -> "RoleLadder":
        import csv

        roles: dict[str, RoleDefinition] = {}
        with path.open(newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                skills = {}
                for pair in row["skills"].split("|"):
                    if ":" in pair:
                        sid, w = pair.split(":", 1)
                        skills[sid.strip()] = float(w)
                    elif pair.strip():
                        skills[pair.strip()] = 1.0
                next_roles = [n.strip() for n in row.get("next_roles", "").split("|") if n.strip()]
                roles[row["role_id"]] = RoleDefinition(role_id=row["role_id"], title=row["title"], skills=skills, next_roles=next_roles)
        return cls(roles)

    def get(self, role_id: str | None) -> RoleDefinition | None:
        return self.roles.get(role_id or "")

    def gap_skills(self, current_role_id: str, next_role_id: str) -> dict[str, float]:
        current = self.get(current_role_id)
        nxt = self.get(next_role_id)
        if not current or not nxt:
            return {}
        gaps = {}
        for skill_id, required_weight in nxt.skills.items():
            have = current.skills.get(skill_id, 0.0)
            if required_weight > have:
                gaps[skill_id] = round(required_weight - have, 3)
        return dict(sorted(gaps.items(), key=lambda kv: -kv[1]))


class HypothesisManager:
    def __init__(self, ladder: RoleLadder, max_active: int = 8, seed: int = 42, epsilon: float = 0.15) -> None:
        self.ladder = ladder
        self.max_active = max_active
        self.epsilon = epsilon
        self._rng = random.Random(seed)
        self._by_user: dict[str, list[Hypothesis]] = {}

    def hypotheses(self, user_id: str) -> list[Hypothesis]:
        return [h for h in self._by_user.get(user_id, []) if h.status != HypothesisStatus.RETIRED]

    def all_hypotheses(self, user_id: str) -> list[Hypothesis]:
        return list(self._by_user.get(user_id, []))

    def ensure_base_hypotheses(self, profile: UserProfile, ladder_role_id: str | None = None) -> list[Hypothesis]:
        existing = self._by_user.setdefault(profile.user_id, [])
        seen_ids = {h.id for h in existing}
        created = []

        role = self.ladder.get(ladder_role_id or profile.role)

        if role is not None:
            hid = f"depth:{role.role_id}"
            if hid not in seen_ids:
                h = Hypothesis(
                    id=hid,
                    type=HypothesisType.DEPTH_IN_ROLE,
                    statement=f"{profile.user_id} wants to deepen expertise in their current role '{role.title}'",
                    target=role.role_id,
                    rationale=f"current role with {len(role.skills)} expected skills",
                )
                existing.append(h)
                created.append(h)

            for next_id in role.next_roles[:2]:
                nid = f"next:{role.role_id}->{next_id}"
                if nid not in seen_ids:
                    gap = self.ladder.gap_skills(role.role_id, next_id)
                    h = Hypothesis(
                        id=nid,
                        type=HypothesisType.NEXT_ROLE_PREP,
                        statement=f"{profile.user_id} is preparing for the next role '{next_id}'",
                        target=next_id,
                        rationale=f"skill gap: {', '.join(list(gap)[:5]) or 'none'}",
                    )
                    existing.append(h)
                    created.append(h)

        touched = sorted(
            ((sid, aff) for sid, aff in profile.skills.items() if aff.events >= 1),
            key=lambda pair: -pair[1].events,
        )
        for skill_id, _aff in touched[:4]:
            sid = f"interest:{skill_id}"
            if sid not in seen_ids:
                h = Hypothesis(
                    id=sid,
                    type=HypothesisType.SKILL_INTEREST,
                    statement=f"{profile.user_id} shows interest in skill {skill_id}",
                    target=skill_id,
                    rationale=f"{_aff.events} interaction(s)",
                )
                existing.append(h)
                created.append(h)

        active = [h for h in existing if h.status == HypothesisStatus.ACTIVE]
        if len(active) > self.max_active:
            by_mean = sorted(active, key=lambda h: h.mean)
            for weak in by_mean[: len(active) - self.max_active]:
                weak.status = HypothesisStatus.WEAK
        return created

    def generate_with_llm(self, profile: UserProfile, llm_chat, recent_summary: str) -> list[Hypothesis]:
        prompt = (
            "You are a talent-development strategist. Given the learner profile and recent signals, "
            "propose 1-3 hypotheses about what content will help them.\n"
            'Reply JSON only: {"hypotheses": [{"type": "depth_in_role|next_role_prep|skill_interest", '
            '"target": "<role_id or skill_id>", "statement": "...", "rationale": "..."}]}\n\n'
            f"Profile: role={profile.role}, top_skills={json.dumps(profile.blend())}\n"
            f"Recent signals: {recent_summary}"
        )
        payload = {
            "model": _model_from_env(),
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
        }
        import os

        import httpx

        base_url = os.getenv("SKILLS_LLM_BASE_URL", "").rstrip("/")
        headers = {"Authorization": f"Bearer {os.getenv('SKILLS_LLM_API_KEY', '')}"} if os.getenv("SKILLS_LLM_API_KEY") else {}
        resp = httpx.post(f"{base_url}/chat/completions", json=payload, headers=headers, timeout=30.0)
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"]
        content = re.sub(r"^```(?:json)?\s*|\s*```$", "", content.strip())
        data = json.loads(content)

        existing = self._by_user.setdefault(profile.user_id, [])
        created = []
        for item in data.get("hypotheses", []):
            hid = f"llm:{item['type']}:{item['target']}"
            if any(h.id == hid for h in existing):
                continue
            h = Hypothesis(
                id=hid,
                type=HypothesisType(item.get("type", "llm_generated")),
                statement=item.get("statement", ""),
                target=item["target"],
                rationale=item.get("rationale", ""),
                source="llm",
            )
            existing.append(h)
            created.append(h)
        return created

    def sample(self, user_id: str) -> Hypothesis | None:
        candidates = self.hypotheses(user_id)
        if not candidates:
            return None
        if self._rng.random() < self.epsilon:
            return self._rng.choice(candidates)
        draws = [(self._rng.betavariate(h.alpha, h.beta), h) for h in candidates]
        draws.sort(key=lambda pair: -pair[0])
        return draws[0][1]

    def apply_feedback(self, user_id: str, hypothesis_id: str, success: bool, weight: float = 1.0) -> None:
        for h in self.all_hypotheses(user_id):
            if h.id == hypothesis_id:
                h.update(success, weight)
                return

    def report(self, user_id: str) -> list[dict]:
        return [
            {
                "id": h.id,
                "type": h.type.value,
                "statement": h.statement,
                "mean": round(h.mean, 4),
                "pulls": h.pulls,
                "status": h.status.value,
                "source": h.source,
            }
            for h in self.all_hypotheses(user_id)
        ]


def _model_from_env() -> str:
    import os

    return os.getenv("SKILLS_LLM_MODEL", "gpt-4o-mini")
