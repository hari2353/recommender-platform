from __future__ import annotations

from ..events.schema import EVENT_WEIGHTS

__all__ = ["decay", "EVENT_WEIGHTS"]


def decay(age_days: float, half_life_days: float) -> float:
    return 0.5 ** (max(age_days, 0.0) / half_life_days)
