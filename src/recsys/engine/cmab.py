from __future__ import annotations

import random


class BetaArm:
    __slots__ = ("alpha", "beta", "pulls")

    def __init__(self, alpha: float = 1.0, beta: float = 1.0) -> None:
        self.alpha = alpha
        self.beta = beta
        self.pulls = 0

    @property
    def mean(self) -> float:
        return self.alpha / (self.alpha + self.beta)

    def update(self, success: bool, weight: float = 1.0) -> None:
        if success:
            self.alpha += weight
        else:
            self.beta += weight
        self.pulls += 1

    def sample(self, rng: random.Random) -> float:
        return rng.betavariate(self.alpha, self.beta)


class ThompsonBandit:
    def __init__(self, seed: int = 42) -> None:
        self.rng = random.Random(seed)
        self.arms: dict[str, BetaArm] = {}

    def arm(self, key: str) -> BetaArm:
        if key not in self.arms:
            self.arms[key] = BetaArm()
        return self.arms[key]

    def sample_all(self, keys: list[str]) -> list[tuple[str, float]]:
        draws = [(key, self.arm(key).sample(self.rng)) for key in keys]
        draws.sort(key=lambda pair: -pair[1])
        return draws

    def update(self, key: str, success: bool, weight: float = 1.0) -> None:
        self.arm(key).update(success, weight)

    def report(self) -> list[dict]:
        return [
            {"arm": key, "mean": round(arm.mean, 4), "alpha": round(arm.alpha, 3), "beta": round(arm.beta, 3), "pulls": arm.pulls}
            for key, arm in sorted(self.arms.items(), key=lambda kv: -kv[1].mean)
        ]
