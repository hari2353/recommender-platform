from __future__ import annotations

import numpy as np

from ..content.index import hashing_embed


class HashingProvider:
    name = "hashing"

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim

    def embed(self, text: str) -> np.ndarray:
        return hashing_embed(text, self.dim)

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        return np.stack([self.embed(t) for t in texts]) if texts else np.zeros((0, self.dim), dtype=np.float32)


class E5MultilingualProvider:
    """intfloat/multilingual-e5-large — open-weights multilingual embeddings (1024-dim).
    E5 expects 'query: ' / 'passage: ' prefixes for asymmetric retrieval."""

    name = "e5"

    def __init__(self, model_name: str = "intfloat/multilingual-e5-large") -> None:
        from sentence_transformers import SentenceTransformer

        self.dim = 1024
        self._model = SentenceTransformer(model_name)

    def _encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.asarray(self._model.encode(texts, normalize_embeddings=True), dtype=np.float32)

    def embed(self, text: str) -> np.ndarray:
        return self._encode(["passage: " + text])[0]

    def embed_query(self, text: str) -> np.ndarray:
        return self._encode(["query: " + text])[0]

    def embed_passage(self, text: str) -> np.ndarray:
        return self._encode(["passage: " + text])[0]

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        return self._encode(["passage: " + t for t in texts])


class MultilingualSTProvider:
    name = "multilingual_st"

    def __init__(self, model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2") -> None:
        from sentence_transformers import SentenceTransformer

        self.dim = 384
        self._model = SentenceTransformer(model_name)

    def embed(self, text: str) -> np.ndarray:
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.asarray(self._model.encode(texts, normalize_embeddings=True), dtype=np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed(text)

    def embed_passage(self, text: str) -> np.ndarray:
        return self.embed(text)


class TitanBedrockProvider:
    """Amazon Titan Embeddings v2 (512-dim) via Bedrock. Requires AWS credentials."""

    name = "titan"

    def __init__(self, region: str | None = None, model_id: str = "amazon.titan-embed-text-v2:0") -> None:
        import boto3

        self.dim = 512
        self.model_id = model_id
        self._client = boto3.client("bedrock-runtime", region_name=region)

    def embed(self, text: str) -> np.ndarray:
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        out = []
        for text in texts:
            body = json.dumps({"inputText": text[:8000], "dimensions": self.dim, "normalize": True})
            resp = self._client.invoke_model(modelId=self.model_id, body=body)
            out.append(np.asarray(json.loads(resp["body"].read())["embedding"], dtype=np.float32))
        return np.stack(out) if out else np.zeros((0, self.dim), dtype=np.float32)


import json


def get_provider(name: str = "hashing", **kwargs):
    if name == "titan":
        return TitanBedrockProvider(region=kwargs.get("region"))
    if name in ("e5", "multilingual_st"):
        if name == "e5":
            return E5MultilingualProvider()
        return MultilingualSTProvider()
    if name == "hashing":
        return HashingProvider(kwargs.get("dim", 256))
    raise ValueError(f"unknown embedding provider '{name}'")
