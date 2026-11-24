import pytest

from recsys.content.index import ContentIndex
from recsys.engine.cmab import ThompsonBandit
from recsys.engine.reranker import CandidateReranker


def test_index_metadata_filters(stack):
    _, index, *_ = stack
    all_ids = index.ids
    free_only = index.filter_items(all_ids, allow_premium=False)
    premium_ids = {cid for cid in all_ids if index.items[cid]["premium"]}
    assert premium_ids and not (set(free_only) & premium_ids)
    shorts = index.filter_items(all_ids, duration_class="short")
    assert all(index.items[c]["duration_class"] == "short" for c in shorts)


def test_skill_search_ranks_matching_content_first(stack):
    _, index, *_ = stack
    hits = index.search_by_skills({"esco__004": 1.0}, top_n=5)
    assert hits and all("esco__004" in index.items[cid]["skills"] for cid, _ in hits)


def test_reranker_blends_relevance_freshness_popularity(stack):
    _, index, *_ = stack
    reranker = CandidateReranker(index)
    out = reranker.rerank(
        ["c008", "c009", "c010"], "kubernetes operations security", top_n=3
    )
    assert len(out) == 3
    scores = [s for _, s in out]
    assert scores == sorted(scores, reverse=True)


def test_thompson_bandit_prefers_rewarded_arm():
    bandit = ThompsonBandit(seed=1)
    for _ in range(200):
        chosen = max(bandit.sample_all(["good", "bad"]), key=lambda t: t[1])[0]
        bandit.update(chosen, success=chosen == "good")
    means = {row["arm"]: row["mean"] for row in bandit.report()}
    assert means["good"] > means["bad"]
    assert means["bad"] < 0.45


def test_stale_version_guard(stack):
    _, index, *_ = stack
    index.upsert(content_id="cx", title="v2", content_version=2)
    assert index.stale_version("cx", 2) is True
    assert index.stale_version("cx", 3) is False
