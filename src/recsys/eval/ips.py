from __future__ import annotations


def _propensity_ratios(
    impressions: list[dict],
    feedback_by_rec: dict[str, set[str]],
    new_policy_propensity,
) -> list[float]:
    """new_p / logged_p for every (row, item) cell, clamped to [1e-4, 1]."""
    ratios = []
    for row in impressions:
        clicked = feedback_by_rec.get(row["recommendation_id"], set())
        for item, logged_p in zip(row["items"], row["propensities"]):
            new_p = min(max(new_policy_propensity(item, row), 1e-4), 1.0)
            logged_p = max(logged_p, 1e-4)
            ratios.append(new_p / logged_p)
    return ratios


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


def snips_estimate(impressions: list[dict], feedback_by_rec: dict[str, set[str]], new_policy_propensity) -> float:
    """Self-normalized IPS (SNIPS): sum(w*r) / sum(w). Robust to small
    logs where the plain IPS mean drifts with the number of rows."""
    weighted_reward = 0.0
    for row in impressions:
        clicked = feedback_by_rec.get(row["recommendation_id"], set())
        for item, logged_p in zip(row["items"], row["propensities"]):
            reward = 1.0 if item in clicked else 0.0
            new_p = min(max(new_policy_propensity(item, row), 1e-4), 1.0)
            weighted_reward += reward * (new_p / max(logged_p, 1e-4))
    total_weight = sum(_propensity_ratios(impressions, feedback_by_rec, new_policy_propensity))
    if total_weight == 0:
        return 0.0
    return round(weighted_reward / total_weight, 4)


def effective_sample_size(impressions: list[dict], feedback_by_rec: dict[str, set[str]], new_policy_propensity) -> float:
    """ESS = (sum w)^2 / sum(w^2). An ESS far below the impression count
    means the off-policy estimate rests on a handful of dominant weights."""
    ratios = _propensity_ratios(impressions, feedback_by_rec, new_policy_propensity)
    if not ratios:
        return 0.0
    total = sum(ratios)
    if total == 0:
        return 0.0
    return round(total * total / sum(r * r for r in ratios), 4)


def compare_policies(
    impressions: list[dict],
    feedback_by_rec: dict[str, set[str]],
    policies: dict[str, object],
) -> dict[str, dict[str, float]]:
    """Score every candidate policy on the same logged impressions:
    {name: {ips, snips, ess}} — the minimal honest off-policy report."""
    report: dict[str, dict[str, float]] = {}
    for name, propensity in policies.items():
        report[name] = {
            "ips": ips_estimate(impressions, feedback_by_rec, propensity),
            "snips": snips_estimate(impressions, feedback_by_rec, propensity),
            "ess": effective_sample_size(impressions, feedback_by_rec, propensity),
        }
    return report


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
