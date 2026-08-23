from __future__ import annotations

import os


class EntitlementService:
    def __init__(self, premium_users: set[str] | None = None) -> None:
        self._premium = premium_users or set()

    def is_premium(self, user_id: str, tenant_id: str = "default") -> bool:
        return f"{tenant_id}:{user_id}" in self._premium or user_id in self._premium

    def grant(self, user_id: str, tenant_id: str = "default") -> None:
        self._premium.add(f"{tenant_id}:{user_id}")

    def revoke(self, user_id: str, tenant_id: str = "default") -> bool:
        return self._premium.discard(f"{tenant_id}:{user_id}") is None
