"""HebbMem — public API for bio-inspired memory."""

from __future__ import annotations

import logging
import threading
import uuid
from pathlib import Path
from typing import Any

from hebbmem.encoders import (
    EncoderBackend,
    HashEncoder,
    SentenceTransformerEncoder,
    auto_select_encoder,
)
from hebbmem.exceptions import EncoderError, MemoryNotFoundError
from hebbmem.graph import MemoryGraph
from hebbmem.node import MemoryNode
from hebbmem.persistence import load_from_sqlite, save_to_sqlite
from hebbmem.types import Config, RecallResult

logger = logging.getLogger("hebbmem")


class HebbMem:
    """Bio-inspired memory for AI agents.

    Implements three neuroscience mechanisms:

    - **Decay**: memories fade over time unless reinforced.
    - **Hebbian learning**: co-recalled memories strengthen their connection.
    - **Spreading activation**: recalling one memory activates related ones.

    Thread-safe — all public methods are protected by an RLock.

    Args:
        encoder: Encoder backend or string name. Options:
            ``"auto"`` (sentence-transformers if available, else hash),
            ``"hash"`` (zero-dependency fallback),
            ``"sentence-transformer"`` (high-quality semantic embeddings),
            or a custom ``EncoderBackend`` instance.
        config: Configuration parameters. Uses defaults if None.

    Example:
        >>> mem = HebbMem(encoder="hash")
        >>> mem.store("Python is great for data science", importance=0.8)
        >>> mem.step(5)  # time passes, memories decay
        >>> results = mem.recall("data science tools")
        >>> results[0].content
        'Python is great for data science'
    """

    def __init__(
        self,
        encoder: str | EncoderBackend = "auto",
        config: Config | None = None,
    ) -> None:
        self.config = config or Config()
        self._encoder = self._resolve_encoder(encoder)
        self._graph = MemoryGraph(self.config)
        self._time_step: int = 0
        self._lock = threading.RLock()

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
        raise EncoderError(f"Unknown encoder: {encoder}")

    def store(
        self,
        content: str,
        importance: float = 0.5,
        metadata: dict[str, Any] | None = None,
    ) -> uuid.UUID:
        """Store a memory.

        Encodes the content, creates a node, and auto-connects it to
        existing similar memories in the graph.

        Args:
            content: Text content to store.
            importance: Importance weight (0-1). Higher values boost
                this memory's recall score. Does not decay over time.
            metadata: Arbitrary key-value data to attach.

        Returns:
            UUID of the stored memory.
        """
        with self._lock:
            embedding = self._encoder.encode(content)
            node = MemoryNode(
                content=content,
                embedding=embedding,
                importance=importance,
                metadata=metadata or {},
            )
            self._graph.add_node(node)
            logger.debug(
                "Stored memory %s (importance=%.2f)", node.memory_id, importance
            )
            return node.memory_id

    def store_batch(
        self,
        contents: list[str],
        importances: list[float] | float = 0.5,
        metadata: list[dict[str, Any]] | None = None,
    ) -> list[uuid.UUID]:
        """Store multiple memories at once.

        Uses ``encode_batch()`` for faster embedding than calling
        ``store()`` in a loop.

        Args:
            contents: List of text strings to store.
            importances: Single float (applied to all) or list of
                floats (one per content). Must match length of contents.
            metadata: List of dicts (one per content), or None for
                empty dicts.

        Returns:
            List of UUIDs for the stored memories.

        Raises:
            ValueError: If importances or metadata length doesn't match contents.
        """
        n = len(contents)
        if isinstance(importances, (int, float)):
            importances = [float(importances)] * n
        if metadata is None:
            metadata = [{} for _ in range(n)]
        if len(importances) != n:
            raise ValueError(
                f"importances length ({len(importances)}) != contents length ({n})"
            )
        if len(metadata) != n:
            raise ValueError(
                f"metadata length ({len(metadata)}) != contents length ({n})"
            )

        with self._lock:
            embeddings = self._encoder.encode_batch(contents)
            ids: list[uuid.UUID] = []
            for i in range(n):
                node = MemoryNode(
                    content=contents[i],
                    embedding=embeddings[i],
                    importance=importances[i],
                    metadata=metadata[i],
                )
                self._graph.add_node(node)
                ids.append(node.memory_id)
            logger.debug("Batch stored %d memories", n)
            return ids

    def recall(self, query: str, top_k: int = 5) -> list[RecallResult]:
        """Recall memories relevant to a query.

        Runs the full bio-inspired recall pipeline:

        1. Encode query and find semantically similar memories (seeds).
        2. Activate seed nodes proportional to their similarity.
        3. Spread activation through graph edges (associative recall).
        4. Hebbian update: strengthen edges between co-activated nodes.
        5. Rank by weighted score and return top-k.

        The scoring formula is:
        ``score = w_act*act + w_sim*sim + w_str*str + w_imp*imp``

        Args:
            query: Natural language query string.
            top_k: Maximum number of results to return.

        Returns:
            List of RecallResult sorted by score descending.
            Empty list if no memories are stored.
        """
        with self._lock:
            query_embedding = self._encoder.encode(query)

            candidates = self._graph.cosine_similarity(query_embedding, top_k=top_k * 2)
            if not candidates:
                return []

            seeds = [(nid, sim) for nid, sim in candidates if sim > 0]
            activated_ids = self._graph.spread_activation(seeds)
            self._graph.hebbian_update(activated_ids)

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
            logger.debug("Recall '%s' -> %d results", query[:50], len(results[:top_k]))
            return results[:top_k]

    def step(self, n: int = 1) -> None:
        """Advance time by n steps, applying decay each step.

        Each step multiplies node activations by ``activation_decay``,
        node strengths by ``strength_decay``, and edge weights by
        ``edge_decay``. Edges below 0.01 are pruned.

        Call this to simulate time passing between agent interactions.

        Args:
            n: Number of time steps to advance.
        """
        with self._lock:
            for _ in range(n):
                self._time_step += 1
                pruned = self._graph.decay_all()
                logger.debug(
                    "Step %d: decayed %d nodes, pruned %d edges",
                    self._time_step,
                    self._graph.node_count,
                    pruned,
                )

    def forget(self, memory_id: uuid.UUID) -> None:
        """Explicitly remove a memory and its edges.

        Unlike natural decay (which weakens memories gradually), this
        immediately and permanently removes a memory from the graph.

        Args:
            memory_id: UUID of the memory to remove.

        Raises:
            MemoryNotFoundError: If the memory_id doesn't exist.
        """
        with self._lock:
            node = self._graph.get_node(memory_id)
            if node is None:
                raise MemoryNotFoundError(f"No memory with id: {memory_id}")
            self._graph.remove_node(memory_id)

    def save(self, path: str | Path) -> None:
        """Save full memory state to a SQLite file.

        Persists all nodes, edges, activations, config, and time step
        into a single ``.hebb`` file. Overwrites if the file exists.

        Args:
            path: File path (e.g., ``"agent_memory.hebb"``).
        """
        with self._lock:
            save_to_sqlite(path, self._graph, self.config, self._time_step)

    @classmethod
    def load(cls, path: str | Path, encoder: str | EncoderBackend = "auto") -> HebbMem:
        """Load memory state from a SQLite file.

        Restores the full graph, config, and time step. The encoder
        must be compatible with the embeddings that were saved (same
        dimension).

        Args:
            path: Path to a ``.hebb`` file created by ``save()``.
            encoder: Encoder to use for future store/recall operations.

        Returns:
            A new HebbMem instance with restored state.

        Raises:
            FileNotFoundError: If the file doesn't exist.
            PersistenceError: If the file is corrupt or unreadable.
        """
        graph, config, time_step = load_from_sqlite(path)
        instance = cls.__new__(cls)
        instance.config = config
        instance._encoder = cls._resolve_encoder(encoder)
        instance._graph = graph
        instance._time_step = time_step
        instance._lock = threading.RLock()
        return instance

    def stats(self) -> dict[str, Any]:
        """Return introspection statistics.

        Returns:
            Dict with keys: ``node_count``, ``edge_count``,
            ``time_step``, ``encoder``.
        """
        with self._lock:
            return {
                "node_count": self._graph.node_count,
                "edge_count": self._graph.edge_count,
                "time_step": self._time_step,
                "encoder": type(self._encoder).__name__,
            }
