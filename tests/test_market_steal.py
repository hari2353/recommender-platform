"""Market-steal additions (2026-11-18): serving fatigue controls, ranking
metrics, off-policy eval hardening.

Compared products treat these as baseline: YouTube/LinkedIn-style serving
suppresses completed items and cools down repeats; RecBole-era evals report
precision@k/recall@k/MRR/NDCG; off-policy evaluation ships SNIPS plus
effective sample size alongside plain IPS.
"""

from __future__ import annotations

import pytest

from recsys.engine.recommender import apply_repeat_cooldown
from recsys.eval.ips import (
    compare_policies,
    ctr_metrics,
    effective_sample_size,
    ips_estimate,
    snips_estimate,
)
from recsys.eval.ranking import (
    hit_rate_at_k,
    mrr,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)


# ── repeat-serve cooldown ────────────────────────────────────────────────────
class TestRepeatCooldownHelper:
    def test_fresh_slate_unchanged(self) -> None:
        ranked = [("a", 0.9, "h", 0.5), ("b", 0.8, "h", 0.5), ("c", 0.7, "h", 0.5)]
        out = apply_repeat_cooldown({}, ranked, k=2, limit=3)
        assert [c[0] for c in out] == ["a", "b"]

    def test_saturated_item_replaced_by_next_ranked(self) -> None:
        ranked = [("a", 0.9, "h", 0.5), ("b", 0.8, "h", 0.5), ("c", 0.7, "h", 0.5)]
        counts = {"a": 3}  # a is saturated at limit 3
        out = apply_repeat_cooldown(counts, ranked, k=2, limit=3)
        assert [c[0] for c in out] == ["b", "c"]

    def test_never_empties_the_slate(self) -> None:
        ranked = [("a", 0.9, "h", 0.5), ("b", 0.8, "h", 0.5)]
        counts = {"a": 3, "b": 3}  # everything saturated
        out = apply_repeat_cooldown(counts, ranked, k=2, limit=3)
        assert [c[0] for c in out] == ["a", "b"]  # top-up fallback in ranked order

    def test_partial_fresh_top_up(self) -> None:
        ranked = [("a", 0.9, "h", 0.5), ("b", 0.8, "h", 0.5), ("c", 0.7, "h", 0.5)]
        counts = {"a": 3, "b": 3}
        out = apply_repeat_cooldown(counts, ranked, k=3, limit=3)
        assert [c[0] for c in out] == ["c", "a", "b"]


class TestServingFatigueIntegration:
    def test_counts_tracked_and_slate_rotates(self, stack) -> None:
        *_, service = stack
        service.repeat_serve_limit = 1
        first = service.serve("fat-user", k=3)  # fresh user -> popular fallback
        assert first["hypothesis"] is None
        profile = service.profiles.get("fat-user")
        first_ids = [i["content_id"] for i in first["items"]]
        assert all(profile.served_counts[c] == 1 for c in first_ids)

        second = service.serve("fat-user", k=3)
        second_ids = [i["content_id"] for i in second["items"]]
        assert not set(first_ids) & set(second_ids)  # saturated items rotated out

        third = service.serve("fat-user", k=3)
        third_ids = [i["content_id"] for i in third["items"]]
        assert not set(third_ids) & (set(first_ids) | set(second_ids))

        # keep serving until the whole catalog is saturated: the cooldown
        # must never hand back an empty slate, and eventually it falls back
        # to re-serving saturated items (top-up rule)
        catalog = len(service.index.ids)
        for _ in range(catalog // 3 + 2):
            slate = service.serve("fat-user", k=3)
            assert len(slate["items"]) == 3

    def test_catalog_saturated_items_return_via_fallback(self, stack) -> None:
        *_, service = stack
        service.repeat_serve_limit = 2
        ids = {i["content_id"] for i in service.serve("u", k=3)["items"]}
        service.serve("u", k=3)
        third = service.serve("u", k=3)
        third_ids = [i["content_id"] for i in third["items"]]
        assert third_ids  # never an empty slate


# ── completion suppression (API path) ───────────────────────────────────────
class TestCompletionSuppression:
    def test_completed_item_never_served_again(self, client) -> None:
        rec = client.get("/v1/users/ucomp/recommendations?k=3").json()
        target = rec["items"][0]["content_id"]
        event = {
            "event_id": "c1",
            "user_id": "ucomp",
            "type": "completion",
            "content_id": target,
        }
        assert (
            client.post("/v1/events", json={"events": [event]}).json()["accepted"] == 1
        )
        # a repeat completion (new event id) is accepted as a new event,
        # but the filter is a set: it lands in hard_filters exactly once
        assert (
            client.post(
                "/v1/events", json={"events": [dict(event, event_id="c2")]}
            ).json()["accepted"]
            == 2
        )

        profile = client.get("/v1/users/ucomp/profile").json()
        assert profile["hard_filters"].count(target) == 1
        again = client.get("/v1/users/ucomp/recommendations?k=5").json()
        assert target not in [i["content_id"] for i in again["items"]]


# ── ranking metrics ──────────────────────────────────────────────────────────
class TestRankingMetrics:
    RANKED = ["a", "b", "c", "d"]

    def test_precision_and_recall(self) -> None:
        assert precision_at_k(self.RANKED, {"a", "c"}, k=3) == pytest.approx(2 / 3)
        assert recall_at_k(self.RANKED, {"a", "c"}, k=3) == pytest.approx(1.0)
        assert recall_at_k(self.RANKED, {"a", "b", "d"}, k=3) == pytest.approx(2 / 3)

    def test_mrr_stops_at_first_hit(self) -> None:
        assert mrr(self.RANKED, {"a"}) == pytest.approx(1.0)
        assert mrr(self.RANKED, {"c"}) == pytest.approx(1 / 3)
        assert mrr(self.RANKED, {"z"}) == 0.0

    def test_ndcg_hand_computed(self) -> None:
        # DCG = 1/log2(2) + 0 + 1/log2(4) = 1.5; IDCG = 1 + 1/log2(3)
        expected = 1.5 / (1 + 1 / __import__("math").log2(3))
        assert ndcg_at_k(self.RANKED, {"a", "c"}, k=3) == pytest.approx(
            expected, rel=1e-4
        )
        assert ndcg_at_k(["a"], {"a"}, k=1) == pytest.approx(1.0)

    def test_hit_rate(self) -> None:
        assert hit_rate_at_k(self.RANKED, {"b"}, k=2) == 1.0
        assert hit_rate_at_k(self.RANKED, {"z"}, k=3) == 0.0

    def test_empty_relevant_set_is_zero(self) -> None:
        assert precision_at_k(self.RANKED, set(), k=3) == 0.0
        assert ndcg_at_k(self.RANKED, set(), k=3) == 0.0


# ── off-policy eval hardening ────────────────────────────────────────────────
IMPRESSIONS = [
    {"recommendation_id": "r1", "items": ["i1", "i2"], "propensities": [0.5, 0.5]},
    {"recommendation_id": "r2", "items": ["i3", "i4"], "propensities": [0.25, 0.25]},
]
FEEDBACK = {"r1": {"i1"}}


class TestOffPolicyEval:
    def test_ips_reference_value(self) -> None:
        # weights 1,1,2,2; reward only on i1 -> weighted 1.0 over 2 rows
        assert ips_estimate(
            IMPRESSIONS, FEEDBACK, lambda item, row: 0.5
        ) == pytest.approx(0.5)

    def test_snips_self_normalized(self) -> None:
        # SNIPS = sum(w*r)/sum(w) = 1.0/6.0
        assert snips_estimate(
            IMPRESSIONS, FEEDBACK, lambda item, row: 0.5
        ) == pytest.approx(1 / 6, rel=1e-3)

    def test_effective_sample_size(self) -> None:
        # ESS = (sum w)^2 / sum(w^2) = 36/10
        assert effective_sample_size(
            IMPRESSIONS, FEEDBACK, lambda item, row: 0.5
        ) == pytest.approx(3.6, rel=1e-3)

    def test_uniform_propensities_agree(self) -> None:
        fn = lambda item, row: 0.5  # noqa: E731
        w = 0.5 / 0.5
        n = 4  # items
        rows = [
            {"recommendation_id": "r", "items": ["a", "b"], "propensities": [0.5, 0.5]}
        ]
        fb = {"r": {"a"}}
        # IPS mean over rows = (w*1 + w*0)/1 row = 1.0; SNIPS = 1/2 = 0.5
        assert ips_estimate(rows, fb, fn) == 1.0
        assert snips_estimate(rows, fb, fn) == pytest.approx(0.5)

    def test_compare_policies_report(self) -> None:
        report = compare_policies(
            IMPRESSIONS,
            FEEDBACK,
            {"half": lambda item, row: 0.5, "low": lambda item, row: 0.2},
        )
        assert set(report) == {"half", "low"}
        for entry in report.values():
            assert set(entry) == {"ips", "snips", "ess"}
        assert report["half"]["ips"] == pytest.approx(0.5)

    def test_ctr_metrics_untouched(self) -> None:
        metrics = ctr_metrics(IMPRESSIONS, FEEDBACK)
        assert metrics["items_served"] == 4 and metrics["items_clicked"] == 1
