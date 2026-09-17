from __future__ import annotations

import hashlib
import logging
import math
import os
import re
from dataclasses import dataclass
from typing import Iterable, List, Protocol

LOGGER = logging.getLogger(__name__)


class EmbeddingModel(Protocol):
    def embed(self, texts: Iterable[str]) -> List[List[float]]:
        ...


@dataclass
class HashingEmbeddingModel:
    dimensions: int = 384

    def embed(self, texts: Iterable[str]) -> List[List[float]]:
        return [self._embed_one(text) for text in texts]

    def _embed_one(self, text: str) -> List[float]:
        vector = [0.0] * self.dimensions
        tokens = re.findall(r"[a-z0-9]+", text.lower())
        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % self.dimensions
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[index] += sign
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]


class SentenceTransformerEmbeddingModel:
    def __init__(self, model_name: str | None = None) -> None:
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name or os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2"))

    def embed(self, texts: Iterable[str]) -> List[List[float]]:
        vectors = self.model.encode(list(texts), normalize_embeddings=True)
        return [vector.tolist() for vector in vectors]


def get_embedding_model() -> EmbeddingModel:
    if os.getenv("USE_HASH_EMBEDDINGS", "").lower() in {"1", "true", "yes"}:
        return HashingEmbeddingModel()
    try:
        return SentenceTransformerEmbeddingModel()
    except Exception as exc:
        if os.getenv("ALLOW_HASH_EMBEDDINGS_FALLBACK", "false").lower() in {"1", "true", "yes"}:
            LOGGER.warning("[RAG] SentenceTransformer unavailable; using local hashing fallback: %s", exc)
            return HashingEmbeddingModel()
        raise
