from datetime import datetime, timezone

from recsys.eval.ips import ctr_metrics, ips_estimate


def test_ips_prefers_higher_value_policy():
    logged = [
        {"recommendation_id": "r1", "items": ["a", "b"], "propensities": [0.5, 0.5]},
        {"recommendation_id": "r2", "items": ["a", "b"], "propensities": [0.5, 0.5]},
    ]
    feedback = {"r1": {"a"}, "r2": {"b"}}
    good_policy = lambda item, row: (
        0.9
        if (item == "a" and row["recommendation_id"] == "r1")
        or (item == "b" and row["recommendation_id"] == "r2")
        else 0.1
    )
    bad_policy = lambda item, row: (
        0.1
        if (item == "a" and row["recommendation_id"] == "r1")
        or (item == "b" and row["recommendation_id"] == "r2")
        else 0.9
    )
    assert ips_estimate(logged, feedback, good_policy) > ips_estimate(
        logged, feedback, bad_policy
    )


def test_ctr_metrics_math():
    rows = [
        {"recommendation_id": "r1", "items": ["x", "y"]},
        {"recommendation_id": "r2", "items": ["z"]},
    ]
    metrics = ctr_metrics(rows, {"r1": {"x"}, "r2": set()})
    assert metrics["items_served"] == 3 and metrics["items_clicked"] == 1
    assert metrics["item_ctr"] == round(1 / 3, 4)
    assert metrics["request_ctr"] == 0.5


def test_hypothesis_weeding_lifecycle():
    from recsys.hypotheses.manager import RoleLadder, HypothesisManager
    from pathlib import Path
    import sys

    data = Path(__file__).resolve().parents[1] / "data"
    mgr = HypothesisManager(RoleLadder.load_csv(data / "roles.csv"))
    profile = __import__(
        "recsys.scoring.profile_builder", fromlist=["UserProfile"]
    ).UserProfile("hx", role="swe")
    mgr.ensure_base_hypotheses(profile)
    hyp = mgr.hypotheses("hx")[0]
    for _ in range(12):
        hyp.update(success=False, weight=1.0)
    assert hyp.status.value == "retired"
    assert hyp not in mgr.hypotheses("hx")


def test_thompson_samples_active_hypotheses(stack):
    _, index, ladder, profiles, hypotheses, bandit, service = stack
    p = profiles.get_or_create("huser")
    p.role = "swe"
    hypotheses.ensure_base_hypotheses(p)
    picks = {mgr.id for mgr in [hypotheses.sample("huser") for _ in range(30)]}
    assert len(picks) >= 1
    report = hypotheses.report("huser")
    assert any(h["type"] == "depth_in_role" for h in report)
