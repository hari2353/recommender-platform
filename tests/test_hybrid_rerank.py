from recsys.content.index import ContentIndex
from recsys.engine.reranker import CandidateReranker


def build_index() -> ContentIndex:
    index = ContentIndex(dim=64)
    index.upsert(content_id="a", title="Python testing", skills=["python"], objectives=["unit tests"])
    index.upsert(content_id="b", title="Python testing", skills=["python"], objectives=["zebra workflows"])
    index.upsert(content_id="c", title="Kubernetes operations", skills=["kubernetes"])
    return index


def test_rare_term_lifts_document_above_dense_tie():
    reranker = CandidateReranker(build_index())
    dense = reranker.rerank(["b", "a"], "python zebra")
    assert dense[0][1] == dense[1][1]

    hybrid = reranker.rerank_hybrid(["b", "a"], "python zebra")

    assert [cid for cid, _ in hybrid] == ["b", "a"]


def test_zero_lexical_weight_matches_dense_order():
    reranker = CandidateReranker(build_index())
    candidates = ["a", "b", "c"]
    dense_order = [cid for cid, _ in reranker.rerank(candidates, "kubernetes security")]

    hybrid_order = [cid for cid, _ in reranker.rerank_hybrid(candidates, "kubernetes security", lexical_weight=0.0)]

    assert hybrid_order == dense_order


def test_empty_inputs_return_empty_list():
    reranker = CandidateReranker(build_index())

    assert reranker.rerank_hybrid([], "python") == []
    assert reranker.rerank_hybrid(["a", "b"], "") == []
    assert reranker.rerank_hybrid(["a", "b"], "   ") == []
    assert reranker.rerank_hybrid(["missing"], "python") == []


def test_top_n_respected():
    reranker = CandidateReranker(build_index())
    candidates = ["a", "b", "c"]

    full = reranker.rerank_hybrid(candidates, "python kubernetes")
    top_two = reranker.rerank_hybrid(candidates, "python kubernetes", top_n=2)

    assert len(top_two) == 2
    assert top_two == full[:2]


def test_ties_break_by_candidate_id():
    reranker = CandidateReranker(build_index())

    assert [cid for cid, _ in reranker.rerank_hybrid(["b", "a"], "python")] == ["a", "b"]
