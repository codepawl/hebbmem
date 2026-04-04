"""MemoryGraph — weighted graph with spreading activation and Hebbian learning.

The graph is the core data structure of hebbmem. Memory nodes are vertices,
weighted edges (synapses) connect semantically related memories, and three
algorithms operate on the graph:

1. **Spreading activation** — BFS from seed nodes, propagating energy
   through edges to surface associatively related memories.
2. **Hebbian reinforcement** — co-activated edges get stronger
   ("neurons that fire together wire together").
3. **Temporal decay** — activation, strength, and edge weights fade
   each time step unless reinforced.
"""

from __future__ import annotations

import logging
import uuid
from collections import deque

import numpy as np

from hebbmem.node import MemoryNode
from hebbmem.types import Config, Edge

logger = logging.getLogger("hebbmem")


class MemoryGraph:
    """Weighted graph of memory nodes with bio-inspired dynamics.

    Nodes are ``MemoryNode`` instances. Edges are undirected, weighted
    connections stored in both directions. An embedding cache enables
    vectorized cosine similarity queries.

    Args:
        config: Configuration parameters. Uses defaults if None.
    """

    def __init__(self, config: Config | None = None) -> None:
        self.config = config or Config()
        self._nodes: dict[uuid.UUID, MemoryNode] = {}
        self._edges: dict[tuple[uuid.UUID, uuid.UUID], Edge] = {}
        self._embedding_matrix: np.ndarray | None = None
        self._embedding_ids: list[uuid.UUID] = []
        self._cache_dirty: bool = True

    # --- Node operations ---

    def add_node(self, node: MemoryNode) -> None:
        """Add a node and auto-connect to similar existing nodes.

        New edges are created to all existing nodes whose cosine
        similarity exceeds ``config.auto_connect_threshold``.

        Args:
            node: The memory node to add.
        """
        self._nodes[node.memory_id] = node
        self._cache_dirty = True
        self._auto_connect(node)

    def remove_node(self, memory_id: uuid.UUID) -> None:
        """Remove a node and all its edges.

        Args:
            memory_id: UUID of the node to remove.
        """
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

        Args:
            query_embedding: Query vector (same dimension as stored embeddings).
            top_k: Number of top results to return.

        Returns:
            List of ``(memory_id, similarity)`` sorted descending.
        """
        if self._cache_dirty:
            self._rebuild_cache()
        if self._embedding_matrix is None:
            return []
        norm = max(float(np.linalg.norm(query_embedding)), 1e-10)
        query_norm = query_embedding / norm
        scores = self._embedding_matrix @ query_norm
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(self._embedding_ids[i], float(scores[i])) for i in top_indices]

    # --- Auto-connect ---

    def _auto_connect(self, node: MemoryNode) -> None:
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
        self._edges[(a, b)] = Edge(weight=weight)
        self._edges[(b, a)] = Edge(weight=weight)

    def get_neighbors(self, memory_id: uuid.UUID) -> list[tuple[uuid.UUID, float]]:
        """Return neighbors and edge weights for a node.

        Args:
            memory_id: UUID of the node.

        Returns:
            List of ``(neighbor_id, edge_weight)`` tuples.
        """
        result = []
        for (src, dst), edge in self._edges.items():
            if src == memory_id:
                result.append((dst, edge.weight))
        return result

    # --- Spreading Activation (BFS) ---

    def spread_activation(
        self, seeds: list[tuple[uuid.UUID, float]]
    ) -> list[uuid.UUID]:
        """Spread activation energy from seed nodes through the graph.

        Uses breadth-first search. Each neighbor receives energy equal to
        ``source.activation * edge_weight * spread_factor``. Propagation
        stops at ``max_hops`` depth or when energy falls below
        ``activation_threshold``.

        Args:
            seeds: List of ``(memory_id, initial_activation)`` tuples.
                Typically the top cosine-similarity matches from a query.

        Returns:
            List of all activated node IDs (seeds + spread targets).
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

        logger.debug(
            "Spread activation: %d nodes activated from %d seeds",
            len(activated),
            len(seeds),
        )
        return list(activated)

    # --- Hebbian Reinforcement ---

    def hebbian_update(self, activated_ids: list[uuid.UUID]) -> None:
        """Strengthen edges between co-activated nodes (Hebb's rule).

        For every edge where both endpoints are in the activated set,
        the weight is updated: ``w_new = min(1.0, w + lr * (1 - w))``.
        This asymptotically approaches 1.0, preventing unbounded growth.

        Args:
            activated_ids: List of node IDs activated in this recall.
        """
        lr = self.config.hebbian_lr
        ids = set(activated_ids)
        pairs = 0
        for (a, b), edge in self._edges.items():
            if a in ids and b in ids:
                edge.weight = min(1.0, edge.weight + lr * (1.0 - edge.weight))
                edge.co_activations += 1
                pairs += 1
        logger.debug("Hebbian update: %d co-activated pairs", pairs // 2)

    # --- Temporal Decay ---

    def decay_all(self) -> int:
        """Apply one time-step of decay to all nodes and edges.

        Node activation is multiplied by ``activation_decay``.
        Node strength is multiplied by ``strength_decay``.
        Edge weights are multiplied by ``edge_decay``.
        Edges with weight below 0.01 are pruned (removed).

        Returns:
            Number of edges pruned.
        """
        for node in self._nodes.values():
            node.decay(self.config.activation_decay, self.config.strength_decay)

        dead_edges = []
        for key, edge in self._edges.items():
            edge.weight *= self.config.edge_decay
            if edge.weight < 0.01:
                dead_edges.append(key)
        for key in dead_edges:
            del self._edges[key]
        return len(dead_edges) // 2
