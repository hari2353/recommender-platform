from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone


def decay(age_days: float, half_life_days: float) -> float:
    return 0.5 ** (max(age_days, 0.0) / half_life_days)


@dataclass
class SkillAffinity:
    short: float = 0.0
    long: float = 0.0
    events: int = 0
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def apply(
        self,
        weight: float,
        now: datetime,
        short_half_life: float,
        long_half_life: float,
    ) -> None:
        age = (now - self.updated_at).total_seconds() / 86400
        self.short = self.short * decay(age, short_half_life) + weight
        self.long = self.long * decay(age, long_half_life) + weight * 0.5
        self.events += 1
        self.updated_at = now

    def effective(
        self, now: datetime, short_half_life: float, long_half_life: float
    ) -> tuple[float, float]:
        age = (now - self.updated_at).total_seconds() / 86400
        return (
            self.short * decay(age, short_half_life),
            self.long * decay(age, long_half_life),
        )


class UserProfile:
    def __init__(
        self, user_id: str, tenant_id: str = "default", role: str | None = None
    ) -> None:
        self.user_id = user_id
        self.tenant_id = tenant_id
        self.role = role
        self.skills: dict[str, SkillAffinity] = {}
        self.hard_filters: set[str] = set()
        self.served_counts: dict[str, int] = {}
        self.total_events = 0

    def apply_event(
        self,
        skill_weights: dict[str, float],
        now: datetime,
        short_half_life: float,
        long_half_life: float,
    ) -> None:
        for skill_id, w in skill_weights.items():
            affinity = self.skills.setdefault(skill_id, SkillAffinity())
            affinity.apply(w, now, short_half_life, long_half_life)
        self.total_events += 1

    def normalized(
        self, now: datetime, short_half_life: float, long_half_life: float
    ) -> dict[str, dict[str, float]]:
        raw = {
            sid: aff.effective(now, short_half_life, long_half_life)
            for sid, aff in self.skills.items()
        }
        max_short = max((v[0] for v in raw.values()), default=0.0)
        max_long = max((v[1] for v in raw.values()), default=0.0)
        out = {}
        for sid, (s, l) in raw.items():
            out[sid] = {
                "short": round(s / max_short, 4) if max_short > 0 else 0.0,
                "long": round(l / max_long, 4) if max_long > 0 else 0.0,
                "events": self.skills[sid].events,
            }
        return out

    @property
    def segment(self) -> str:
        if self.total_events == 0:
            return "new"
        if self.total_events < 10:
            return "casual"
        return "engaged"

    def blend(self, alpha_short: float = 0.6) -> dict[str, float]:
        norm = self.normalized(
            datetime.now(timezone.utc), _DEFAULT_SHORT, _DEFAULT_LONG
        )
        blended = {}
        for sid, values in norm.items():
            blended[sid] = round(
                alpha_short * values["short"] + (1 - alpha_short) * values["long"], 4
            )
        return dict(sorted(blended.items(), key=lambda kv: -kv[1]))


_DEFAULT_SHORT = 30.0
_DEFAULT_LONG = 180.0


class ProfileStore:
    def __init__(self) -> None:
        self._profiles: dict[str, UserProfile] = {}

    def get_or_create(self, user_id: str, tenant_id: str = "default") -> UserProfile:
        key = f"{tenant_id}:{user_id}"
        if key not in self._profiles:
            self._profiles[key] = UserProfile(user_id, tenant_id)
        return self._profiles[key]

    def get(self, user_id: str, tenant_id: str = "default") -> UserProfile | None:
        return self._profiles.get(f"{tenant_id}:{user_id}")

    def delete(self, user_id: str, tenant_id: str = "default") -> bool:
        return self._profiles.pop(f"{tenant_id}:{user_id}", None) is not None

    def all_profiles(self) -> list[UserProfile]:
        return list(self._profiles.values())
