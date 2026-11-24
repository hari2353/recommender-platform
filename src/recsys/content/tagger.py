from __future__ import annotations

import json
import os
import re

from .sources import MODALITY_BY_SOURCE, estimate_duration_minutes


class SkillTagger:
    def __init__(
        self,
        normalize_fn,
        candidate_skills: list[dict],
        llm_base_url: str | None = None,
    ) -> None:
        """normalize_fn: raw label -> {skill_id, canonical_name, tier, confidence} (skills-engine resolver).
        candidate_skills: [{"id": ..., "name": ...}] taxonomy shortlist for the LLM prompt."""
        self.normalize_fn = normalize_fn
        self.candidate_skills = candidate_skills
        self.base_url = (llm_base_url or os.getenv("SKILLS_LLM_BASE_URL", "")).rstrip(
            "/"
        )
        self.model = os.getenv("SKILLS_LLM_MODEL", "gpt-4o-mini")

    @property
    def llm_enabled(self) -> bool:
        return bool(self.base_url)

    def tag(self, title: str, text: str, source_type: str = "transcript") -> dict:
        raw_labels: list[str] = []
        if self.llm_enabled:
            try:
                raw_labels = self._llm_labels(title, text)
            except Exception:
                raw_labels = []
        if not raw_labels:
            raw_labels = self._rule_candidates(title, text)

        tagged = []
        seen: set[str] = set()
        for label in raw_labels[:10]:
            result = self.normalize_fn(label)
            if (
                result.get("skill_id")
                and result["skill_id"] not in seen
                and float(result.get("confidence", 0)) >= 0.5
            ):
                seen.add(result["skill_id"])
                tagged.append(
                    {
                        "skill_id": result["skill_id"],
                        "name": result.get("canonical_name") or label,
                        "tier": result.get("tier"),
                        "confidence": result.get("confidence"),
                    }
                )
        if title:
            tagged.sort(key=lambda t: (-(t["confidence"] or 0),))
        return {
            "skills": tagged,
            "metadata": {
                "modality": MODALITY_BY_SOURCE.get(source_type.lower(), "reading"),
                "duration_minutes": estimate_duration_minutes(f"{title} {text}"),
                "objectives_source": "provided",
                "tagger": "llm" if (self.llm_enabled and raw_labels) else "rules",
            },
        }

    def _llm_labels(self, title: str, text: str) -> list[str]:
        import httpx

        shortlist = ", ".join(s["name"] for s in self.candidate_skills[:120])
        prompt = (
            "Tag this course with skills from the taxonomy. Prefer exact taxonomy names.\n"
            f"Taxonomy options: {shortlist}\n"
            f"Course title: {title}\n"
            'Reply JSON only: {"skills": ["...", "..."]}\n\n'
            f"Course content:\n{text[:4000]}"
        )
        headers = (
            {"Authorization": f"Bearer {os.getenv('SKILLS_LLM_API_KEY', '')}"}
            if os.getenv("SKILLS_LLM_API_KEY")
            else {}
        )
        resp = httpx.post(
            f"{self.base_url}/chat/completions",
            json={
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0,
            },
            headers=headers,
            timeout=30.0,
        )
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"].strip()
        content = re.sub(r"^```(?:json)?\s*|\s*```$", "", content)
        data = json.loads(content)
        items = data.get("skills", []) if isinstance(data, dict) else data
        return [str(s).strip() for s in items if str(s).strip()]

    def _rule_candidates(self, title: str, text: str) -> list[str]:
        from .extract_rules import gazetteer_extract

        return gazetteer_extract(
            title, text, [s["name"] for s in self.candidate_skills]
        )
