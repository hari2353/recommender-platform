from recsys.rag.chunker import semantic_chunks, structural_chunks
from recsys.rag.embeddings import get_provider
from recsys.rag.rag import ChunkStore, RagService

LONG_TEXT = "\n".join([f"Paragraph {i}: " + "Kubernetes networking policies control traffic flow between pods. " * 4 for i in range(8)])


def test_structural_chunk_sizes_and_overlap():
    chunks = structural_chunks(LONG_TEXT, target_words=60, overlap_words=10)
    assert len(chunks) >= 3
    for chunk in chunks:
        assert len(chunk["text"].split()) <= 80


def test_structural_legacy_kwargs_shape():
    legacy = structural_chunks(LONG_TEXT, target_words=50, overlap_words=5)
    assert all({"text", "section", "index"} <= set(c) for c in legacy)


def test_semantic_chunking_splits_on_topic_shift():
    provider = get_provider("hashing")
    text = (
        "Cooking pasta requires boiling salted water first. Fresh pasta cooks faster than dried. "
        "Salt enhances the flavor of the dish. "
        "Kubernetes clusters schedule containers across nodes. Pod affinity rules affect placement. "
        "Cluster autoscaling adds nodes under load."
    )
    chunks = semantic_chunks(text, provider, min_words=5, max_words=100)
    assert len(chunks) >= 2
    joined = " ".join(c["text"] for c in chunks).lower()
    assert "pasta" in joined and "kubernetes" in joined


def test_chunk_store_search_and_prefix_delete():
    store = ChunkStore(provider_name="hashing")
    n = store.ingest_document("doc1", LONG_TEXT, strategy="structural")
    assert n > 0
    hits = store.search("traffic between pods", k=2)
    assert hits and hits[0]["content_id"] == "doc1"
    assert store.remove_content("doc1") == n
    assert store.search("anything") == []


def test_rag_extractive_answer_with_citations():
    store = ChunkStore(provider_name="hashing")
    store.ingest_document("c008", "Pod security policies restrict container permissions. Network policies limit pod traffic. " * 6, strategy="structural")
    rag = RagService(store)
    out = rag.ask("What limits pod traffic?")
    assert "[1]" in out["answer"]
    assert out["citations"][0]["content_id"] == "c008"
