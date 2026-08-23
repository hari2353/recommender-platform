from __future__ import annotations


def ips_estimate(impressions: list[dict], feedback_by_rec: dict[str, set[str]], new_policy_propensity) -> float:
    """
    Inverse propensity scoring over logged impressions.

    impressions: [{recommendation_id, items, propensities}]
    feedback_by_rec: rec_id -> set of clicked content_ids
    new_policy_propensity: fn(item, impression_row) -> propensity under candidate policy

    Returns estimated value V(pi_new) in [0, 1].
    """
    total_weight = 0.0
    weighted_reward = 0.0
    for row in impressions:
        clicked = feedback_by_rec.get(row["recommendation_id"], set())
        for item, logged_p in zip(row["items"], row["propensities"]):
            reward = 1.0 if item in clicked else 0.0
            new_p = min(max(new_policy_propensity(item, row), 1e-4), 1.0)
            logged_p = max(logged_p, 1e-4)
            weighted_reward += reward * (new_p / logged_p)
            total_weight += new_p / logged_p
    if total_weight == 0:
        return 0.0
    return round(weighted_reward / len(impressions), 4)


def ctr_metrics(impressions: list[dict], feedback_by_rec: dict[str, set[str]]) -> dict:
    served = sum(len(r["items"]) for r in impressions)
    clicked = sum(len(feedback_by_rec.get(r["recommendation_id"], set())) for r in impressions)
    per_rec_ctr = []
    for r in impressions:
        hit = 1 if feedback_by_rec.get(r["recommendation_id"]) else 0
        per_rec_ctr.append(hit)
    return {
        "items_served": served,
        "items_clicked": clicked,
        "item_ctr": round(clicked / served, 4) if served else 0.0,
        "request_ctr": round(sum(per_rec_ctr) / len(per_rec_ctr), 4) if per_rec_ctr else 0.0,
    }
