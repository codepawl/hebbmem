"""Flat vector search baseline — naive cosine similarity, no bio-inspired mechanisms."""

from __future__ import annotations

import time

import numpy as np

from hebbmem.encoders import EncoderBackend


class FlatVectorSearch:
    """Naive vector search baseline.

    Stores memories as (content, embedding, importance, timestamp).
    Retrieves by cosine similarity only. No decay, no Hebbian, no spreading.
    This represents what ChromaDB/LanceDB do at their core.
    """

    def __init__(self, encoder: EncoderBackend) -> None:
        self.encoder = encoder
        self.memories: list[tuple[str, np.ndarray, float, float]] = []

    def store(self, content: str, importance: float = 0.5) -> None:
        emb = self.encoder.encode(content)
        self.memories.append((content, emb, importance, time.time()))

    def recall(self, query: str, top_k: int = 5) -> list[str]:
        if not self.memories:
            return []
        query_emb = self.encoder.encode(query)
        q_norm = query_emb / max(float(np.linalg.norm(query_emb)), 1e-10)

        scored = []
        for content, emb, _importance, _ts in self.memories:
            e_norm = emb / max(float(np.linalg.norm(emb)), 1e-10)
            sim = float(q_norm @ e_norm)
            scored.append((content, sim))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [content for content, _ in scored[:top_k]]

    def step(self, n: int = 1) -> None:
        pass  # flat search doesn't decay
