"""HebbMem — public API for bio-inspired memory."""

from __future__ import annotations

import uuid
from typing import Any

from hebbmem.encoders import (
    EncoderBackend,
    HashEncoder,
    SentenceTransformerEncoder,
    auto_select_encoder,
)
from hebbmem.graph import MemoryGraph
from hebbmem.node import MemoryNode
from hebbmem.types import Config, RecallResult


class HebbMem:
    """Bio-inspired memory for AI agents.

    Uses decay, Hebbian learning, and spreading activation
    to model how human memory works.
    """

    # TODO v0.2: save(path) / HebbMem.load(path) — persistence
    # TODO v0.2: store_batch(contents) — batch ingestion
    # TODO v0.2: threading.RLock — thread safety

    def __init__(
        self,
        encoder: str | EncoderBackend = "auto",
        config: Config | None = None,
    ) -> None:
        self.config = config or Config()
        self._encoder = self._resolve_encoder(encoder)
        self._graph = MemoryGraph(self.config)
        self._time_step: int = 0

    @staticmethod
    def _resolve_encoder(encoder: str | EncoderBackend) -> EncoderBackend:
        if isinstance(encoder, EncoderBackend):
            return encoder
        if encoder == "auto":
            return auto_select_encoder()
        if encoder == "hash":
            return HashEncoder()
        if encoder == "sentence-transformer":
            return SentenceTransformerEncoder()
        raise ValueError(f"Unknown encoder: {encoder}")

    def store(
        self,
        content: str,
        importance: float = 0.5,
        metadata: dict[str, Any] | None = None,
    ) -> uuid.UUID:
        """Encode and store content. Returns memory_id."""
        embedding = self._encoder.encode(content)
        node = MemoryNode(
            content=content,
            embedding=embedding,
            importance=importance,
            metadata=metadata or {},
        )
        self._graph.add_node(node)
        return node.memory_id

    def recall(self, query: str, top_k: int = 5) -> list[RecallResult]:
        """Full recall pipeline: encode → similarity → spread → hebbian → rank."""
        query_embedding = self._encoder.encode(query)

        # Find seed candidates (fetch extra to allow spreading to surface more)
        candidates = self._graph.cosine_similarity(query_embedding, top_k=top_k * 2)
        if not candidates:
            return []

        # Activate seeds proportional to similarity
        seeds = [(nid, sim) for nid, sim in candidates if sim > 0]

        # Spread activation through graph
        activated_ids = self._graph.spread_activation(seeds)

        # Hebbian update on co-activated nodes
        self._graph.hebbian_update(activated_ids)

        # Rank by weighted score
        weights = self.config.scoring_weights
        sim_map = dict(candidates)
        results: list[RecallResult] = []

        for nid in activated_ids:
            node = self._graph.get_node(nid)
            if node is None:
                continue
            node.touch()
            sim = sim_map.get(nid, 0.0)
            values = {
                "activation": node.activation,
                "similarity": sim,
                "strength": node.base_strength,
                "importance": node.importance,
            }
            score = sum(weights[k] * values[k] for k in weights)
            results.append(
                RecallResult(
                    memory_id=node.memory_id,
                    content=node.content,
                    score=score,
                    activation=node.activation,
                    similarity=sim,
                    strength=node.base_strength,
                    importance=node.importance,
                    metadata=node.metadata,
                )
            )

        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_k]

    def step(self, n: int = 1) -> None:
        """Advance time by n steps, applying decay each step."""
        for _ in range(n):
            self._time_step += 1
            self._graph.decay_all()

    def forget(self, memory_id: uuid.UUID) -> bool:
        """Explicitly remove a memory. Returns True if found and removed."""
        node = self._graph.get_node(memory_id)
        if node is None:
            return False
        self._graph.remove_node(memory_id)
        return True

    def stats(self) -> dict[str, Any]:
        """Return introspection statistics."""
        return {
            "node_count": self._graph.node_count,
            "edge_count": self._graph.edge_count,
            "time_step": self._time_step,
            "encoder": type(self._encoder).__name__,
        }
