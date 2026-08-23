from __future__ import annotations

import json
import os
import re


class LlmExplainer:
    def __init__(self, base_url: str | None = None, api_key: str = "", model: str = "") -> None:
        self.base_url = (base_url or os.getenv("SKILLS_LLM_BASE_URL", "")).rstrip("/")
        self.api_key = api_key or os.getenv("SKILLS_LLM_API_KEY", "")
        self.model = model or os.getenv("SKILLS_LLM_MODEL", "gpt-4o-mini")

    @property
    def configured(self) -> bool:
        return bool(self.base_url)

    def explain_recommendation(self, user_context: dict, item: dict) -> dict:
        if not self.configured:
            return self._template_explanation(user_context, item)
        try:
            return self._llm_explanation(user_context, item)
        except Exception:
            return self._template_explanation(user_context, item)

    def _template_explanation(self, user_context: dict, item: dict) -> dict:
        skills = ", ".join(user_context.get("top_skills", [])[:3]) or "your recent activity"
        duration = "quick" if item.get("duration_class") == "short" else "in-depth"
        reason = f"Matches your interest in {skills} — a {duration} {item.get('duration_class', 'short')} course."
        if user_context.get("hypothesis_type") == "next_role_prep":
            reason += f" It closes a gap toward your next role ({user_context.get('hypothesis_target', '')})."
        elif user_context.get("hypothesis_type") == "depth_in_role":
            reason += " It deepens expertise for your current role."
        return {"explanation": reason, "generated_by": "template"}

    def _llm_explanation(self, user_context: dict, item: dict) -> dict:
        import httpx

        prompt = (
            "Explain in 1-2 friendly sentences why this course fits this learner. "
            f"Learner: {json.dumps(user_context)}. Course: {json.dumps(item)}."
        )
        payload = {"model": self.model, "messages": [{"role": "user", "content": prompt}], "temperature": 0}
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        resp = httpx.post(f"{self.base_url}/chat/completions", json=payload, headers=headers, timeout=30.0)
        resp.raise_for_status()
        text = resp.json()["choices"][0]["message"]["content"].strip()
        return {"explanation": re.sub(r"\s+", " ", text), "generated_by": "llm"}

    def generate_learning_path_narrative(self, path: dict) -> dict:
        steps = [f"{i+1}. {s['title']} ({s['phase'].replace('_', ' ')})" for i, s in enumerate(path["steps"])]
        summary = (
            f"Path to '{path['to_role']}': close {path['total_gap_skills']} skill gap(s) with short courses first, "
            "then deepen existing strengths with long-form content.\n" + "\n".join(steps)
        )
        return {"summary": summary, "steps": steps, "generated_by": "template"}
