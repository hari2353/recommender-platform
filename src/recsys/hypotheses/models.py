from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, Field


class HypothesisType(str, Enum):
    DEPTH_IN_ROLE = "depth_in_role"
    NEXT_ROLE_PREP = "next_role_prep"
    SKILL_INTEREST = "skill_interest"
    LLM_GENERATED = "llm_generated"


class HypothesisStatus(str, Enum):
    ACTIVE = "active"
    WEAK = "weak"
    RETIRED = "retired"


class Hypothesis(BaseModel):
    id: str
    type: HypothesisType
    statement: str
    target: str
    rationale: str = ""
    source: str = "deterministic"
    alpha: float = 1.0
    beta: float = 1.0
    pulls: int = 0
    status: HypothesisStatus = HypothesisStatus.ACTIVE
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    @property
    def mean(self) -> float:
        return self.alpha / (self.alpha + self.beta)

    def update(self, success: bool, weight: float = 1.0) -> None:
        if success:
            self.alpha += weight
        else:
            self.beta += weight
        self.pulls += 1
        self._refresh_status()

    def _refresh_status(self) -> None:
        if self.pulls >= 10 and self.mean < 0.30:
            self.status = HypothesisStatus.RETIRED
        elif self.pulls >= 5 and self.mean < 0.40:
            self.status = HypothesisStatus.WEAK
        elif self.status != HypothesisStatus.ACTIVE and self.mean >= 0.45:
            self.status = HypothesisStatus.ACTIVE


class RoleDefinition(BaseModel):
    role_id: str
    title: str
    skills: dict[str, float]
    next_roles: list[str] = Field(default_factory=list)
