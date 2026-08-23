import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from recsys.content.index import ContentIndex
from recsys.engine.cmab import ThompsonBandit
from recsys.engine.recommender import RecommenderService
from recsys.events.schema import EVENT_WEIGHTS, EventType
from recsys.hypotheses.manager import HypothesisManager, RoleLadder
from recsys.scoring.profile_builder import ProfileStore

DATA = Path(__file__).resolve().parents[1] / "data"


def main() -> None:
    rng = random.Random(7)
    index = ContentIndex()
    index.load_csv(DATA / "content_catalog.csv")
    ladder = RoleLadder.load_csv(DATA / "roles.csv")
    profiles = ProfileStore()
    hypotheses = HypothesisManager(ladder)
    bandit = ThompsonBandit(seed=23)
    service = RecommenderService(index, profiles, hypotheses, bandit=bandit)
    service.set_popularity({cid: 0.4 for cid in index.ids})

    personas = [
        {"user": "sim-platform", "role": "swe", "loved": ["esco__004", "esco__016"], "rounds": 140},
        {"user": "sim-data", "role": "data_engineer", "loved": ["esco__012", "esco__005"], "rounds": 140},
        {"user": "sim-ml", "role": "ml_engineer", "loved": ["esco__011", "esco__004"], "rounds": 140},
    ]

    for persona in personas:
        user = persona["user"]
        profile = profiles.get_or_create(user)
        profile.role = persona["role"]
        hypotheses.ensure_base_hypotheses(profile)

        early_clicks = late_clicks = early_served = late_served = 0

        def simulate(round_idx: int) -> tuple[int, int]:
            served = clicked = 0
            for _ in range(3):
                result = service.serve(user, k=3)
                hyp = result.get("hypothesis") or {}
                for item in result["items"]:
                    served += 1
                    loved = any(s in (item["content_id"], str(item.get("score", ""))) for s in [])
                    content_skills = index.items[item["content_id"]]["skills"]
                    is_relevant = any(s in persona["loved"] or s in [x.split(":")[-1] for x in persona["loved"]] for s in content_skills)
                    click_prob = 0.8 if is_relevant else 0.1
                    if rng.random() < click_prob:
                        clicked += 1
                        service.apply_feedback(user, result["recommendation_id"], item["content_id"], True, weight=1.0)
                        skill_w = {sid: EVENT_WEIGHTS[EventType.LIKE] for sid in content_skills}
                        from datetime import datetime, timezone

                        profile.apply_event(skill_w, datetime.now(timezone.utc), 30, 180)
            return served, clicked

        for round_idx in range(persona["rounds"]):
            s, c = simulate(round_idx)
            if round_idx < 35:
                early_served += s
                early_clicks += c
            elif round_idx >= persona["rounds"] - 35:
                late_served += s
                late_clicks += c

        early_ctr = early_clicks / max(early_served, 1)
        late_ctr = late_clicks / max(late_served, 1)
        uplift = ((late_ctr - early_ctr) / max(early_ctr, 1e-9)) * 100
        print(f"{user}: CTR {early_ctr:.3f} -> {late_ctr:.3f} ({uplift:+.1f}% uplift)")
        top_hyp = sorted(hypotheses.report(user), key=lambda h: -h["mean"])[:2]
        for h in top_hyp:
            print(f"   kept hypothesis: [{h['status']}] {h['id']} mean={h['mean']}")

    print("\nbandit arms:")
    for row in bandit.report()[:5]:
        print(f"   {row['arm']}: mean={row['mean']} pulls={row['pulls']}")


if __name__ == "__main__":
    main()


