from __future__ import annotations

from .manager import RoleLadder
from .models import RoleDefinition


def build_learning_path(
    ladder: RoleLadder,
    profile_role_id: str | None,
    target_role_id: str,
    content_lookup: dict[str, list[dict]],
    max_steps: int = 6,
) -> dict:
    current = ladder.get(profile_role_id)
    gaps = ladder.gap_skills(profile_role_id, target_role_id)

    steps: list[dict] = []
    for skill_id, gap_weight in gaps.items():
        options = content_lookup.get(skill_id, [])
        short = [c for c in options if c.get("duration_class") == "short"]
        long = [c for c in options if c.get("duration_class") == "long"]
        pick_short = short[0] if short else (options[0] if options else None)
        if pick_short:
            steps.append(
                {
                    "skill_id": skill_id,
                    "gap": gap_weight,
                    "phase": "close_gap",
                    "content_id": pick_short["content_id"],
                    "title": pick_short.get("title", ""),
                }
            )

    depth_pool: dict[str, list[dict]] = {}
    for skill_id in current.skills if current else {}:
        options = content_lookup.get(skill_id, [])
        long = [c for c in options if c.get("duration_class") == "long"]
        if long:
            depth_pool[skill_id] = long

    for skill_id, options in sorted(depth_pool.items())[
        : max(0, max_steps - len(steps))
    ]:
        steps.append(
            {
                "skill_id": skill_id,
                "gap": 0.0,
                "phase": "deepen_existing",
                "content_id": options[0]["content_id"],
                "title": options[0].get("title", ""),
            }
        )

    return {
        "from_role": profile_role_id,
        "to_role": target_role_id,
        "strategy": "close_gaps_first_then_deepen",
        "steps": steps[:max_steps],
        "total_gap_skills": len(gaps),
    }
