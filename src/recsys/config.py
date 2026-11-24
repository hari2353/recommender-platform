from __future__ import annotations

import os
from pathlib import Path


class Settings:
    def __init__(self, data_dir: Path | None = None) -> None:
        self.data_dir = data_dir or Path(__file__).resolve().parents[2] / "data"
        self.bus_backend = os.getenv("RECSYS_BUS", "memory")
        self.pulsar_url = os.getenv("RECSYS_PULSAR_URL", "pulsar://localhost:6650")
        self.pulsar_topic = os.getenv(
            "RECSYS_PULSAR_TOPIC", "persistent://public/default/user-events"
        )
        self.short_half_life_days = float(os.getenv("RECSYS_SHORT_HALFLIFE_DAYS", "30"))
        self.long_half_life_days = float(os.getenv("RECSYS_LONG_HALFLIFE_DAYS", "180"))
        self.event_dedup_capacity = int(os.getenv("RECSYS_DEDUP_CAPACITY", "100000"))
        self.late_event_days = int(os.getenv("RECSYS_LATE_EVENT_DAYS", "7"))
        self.default_k = int(os.getenv("RECSYS_DEFAULT_K", "5"))
        self.diversity_lambda = float(os.getenv("RECSYS_DIVERSITY_LAMBDA", "0.35"))
        self.llm_base_url = os.getenv("SKILLS_LLM_BASE_URL")
        self.llm_api_key = os.getenv("SKILLS_LLM_API_KEY", "")
        self.llm_model = os.getenv("SKILLS_LLM_MODEL", "gpt-4o-mini")
