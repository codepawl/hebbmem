"""MemoryGraph — weighted graph with spreading activation and Hebbian learning."""

from __future__ import annotations

import uuid
from collections import deque

import numpy as np

from hebbmem.node import MemoryNode
from hebbmem.types import Config, Edge


class MemoryGraph:
    """Graph of memory nodes connected by weighted edges (synapses).

    Implements spreading activation, Hebbian reinforcement, and temporal decay.
    """

    def __init__(self, config: Config | None = None) -> None:
        self.config = config or Config()
        self._nodes: dict[uuid.UUID, MemoryNode] = {}
        self._edges: dict[tuple[uuid.UUID, uuid.UUID], Edge] = {}
        # Embedding cache for vectorized cosine similarity
        self._embedding_matrix: np.ndarray | None = None
        self._embedding_ids: list[uuid.UUID] = []
        self._cache_dirty: bool = True

    # --- Node operations ---

    def add_node(self, node: MemoryNode) -> None:
        """Add a node and auto-connect to similar existing nodes."""
        self._nodes[node.memory_id] = node
        self._cache_dirty = True
        self._auto_connect(node)

    def remove_node(self, memory_id: uuid.UUID) -> None:
        """Remove a node and all its edges."""
        self._nodes.pop(memory_id, None)
        dead = [k for k in self._edges if memory_id in k]
        for k in dead:
            del self._edges[k]
        self._cache_dirty = True

    def get_node(self, memory_id: uuid.UUID) -> MemoryNode | None:
        return self._nodes.get(memory_id)

    @property
    def node_count(self) -> int:
        return len(self._nodes)

    @property
    def edge_count(self) -> int:
        return len(self._edges) // 2  # undirected, stored both directions

    # --- Embedding cache ---

    def _rebuild_cache(self) -> None:
        if not self._nodes:
            self._embedding_matrix = None
            self._embedding_ids = []
            self._cache_dirty = False
            return
        self._embedding_ids = list(self._nodes.keys())
        self._embedding_matrix = np.stack(
            [self._nodes[nid].embedding for nid in self._embedding_ids]
        )
        norms = np.linalg.norm(self._embedding_matrix, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)
        self._embedding_matrix /= norms
        self._cache_dirty = False

    def cosine_similarity(
        self, query_embedding: np.ndarray, top_k: int = 5
    ) -> list[tuple[uuid.UUID, float]]:
        """Vectorized cosine similarity against all nodes.

        Returns list of (memory_id, similarity) sorted descending.
        """
        if self._cache_dirty:
            self._rebuild_cache()
        if self._embedding_matrix is None:
            return []
        query_norm = query_embedding / max(float(np.linalg.norm(query_embedding)), 1e-10)
        scores = self._embedding_matrix @ query_norm
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(self._embedding_ids[i], float(scores[i])) for i in top_indices]

    # --- Auto-connect ---

    def _auto_connect(self, node: MemoryNode) -> None:
        """Connect new node to existing nodes above similarity threshold."""
        if self._cache_dirty:
            self._rebuild_cache()
        if self._embedding_matrix is None or len(self._embedding_ids) <= 1:
            return
        query_norm = node.embedding / max(float(np.linalg.norm(node.embedding)), 1e-10)
        scores = self._embedding_matrix @ query_norm
        threshold = self.config.auto_connect_threshold
        for i, sim in enumerate(scores):
            nid = self._embedding_ids[i]
            if nid != node.memory_id and sim >= threshold:
                self._set_edge(node.memory_id, nid, float(sim))

    def _set_edge(self, a: uuid.UUID, b: uuid.UUID, weight: float) -> None:
        """Set undirected edge weight."""
        self._edges[(a, b)] = Edge(weight=weight)
        self._edges[(b, a)] = Edge(weight=weight)

    def get_neighbors(self, memory_id: uuid.UUID) -> list[tuple[uuid.UUID, float]]:
        """Return [(neighbor_id, edge_weight), ...] for a node."""
        result = []
        for (src, dst), edge in self._edges.items():
            if src == memory_id:
                result.append((dst, edge.weight))
        return result

    # --- Spreading Activation (BFS) ---

    def spread_activation(
        self, seeds: list[tuple[uuid.UUID, float]]
    ) -> list[uuid.UUID]:
        """BFS spread from seed nodes. Returns all activated node IDs.

        seeds: [(memory_id, initial_activation), ...]
        """
        activated: set[uuid.UUID] = set()
        queue: deque[tuple[uuid.UUID, int]] = deque()
        threshold = self.config.activation_threshold

        for nid, act in seeds:
            node = self._nodes.get(nid)
            if node:
                node.activate(act)
                activated.add(nid)
                queue.append((nid, 0))

        while queue:
            current_id, hop = queue.popleft()
            if hop >= self.config.max_hops:
                continue
            current_node = self._nodes[current_id]
            for neighbor_id, edge_weight in self.get_neighbors(current_id):
                spread_amount = (
                    current_node.activation * edge_weight * self.config.spread_factor
                )
                if spread_amount < threshold:
                    continue
                neighbor = self._nodes.get(neighbor_id)
                if neighbor:
                    neighbor.activate(spread_amount)
                    if neighbor_id not in activated:
                        activated.add(neighbor_id)
                        queue.append((neighbor_id, hop + 1))

        return list(activated)

    # --- Hebbian Reinforcement ---

    def hebbian_update(self, activated_ids: list[uuid.UUID]) -> None:
        """Strengthen edges between co-activated nodes."""
        lr = self.config.hebbian_lr
        ids = set(activated_ids)
        for (a, b), edge in self._edges.items():
            if a in ids and b in ids:
                edge.weight = min(1.0, edge.weight + lr * (1.0 - edge.weight))
                edge.co_activations += 1

    # --- Temporal Decay ---

    def decay_all(self) -> None:
        """Apply one time-step of decay to all nodes and edges."""
        for node in self._nodes.values():
            node.decay(self.config.activation_decay, self.config.strength_decay)

        dead_edges = []
        for key, edge in self._edges.items():
            edge.weight *= self.config.edge_decay
            if edge.weight < 0.01:
                dead_edges.append(key)
        for key in dead_edges:
            del self._edges[key]
