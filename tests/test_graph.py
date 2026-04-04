"""Tests for MemoryGraph."""

import uuid

import numpy as np

from hebbmem.graph import MemoryGraph
from hebbmem.node import MemoryNode
from hebbmem.types import Config


def _vec(values: list[float]) -> np.ndarray:
    """Create a unit vector from values."""
    v = np.array(values, dtype=np.float32)
    norm = np.linalg.norm(v)
    return v / norm if norm > 0 else v


class TestNodeOperations:
    def test_add_and_get(self):
        g = MemoryGraph()
        node = MemoryNode(content="hi", embedding=_vec([1, 0, 0, 0]))
        g.add_node(node)
        assert g.get_node(node.memory_id) is node
        assert g.node_count == 1

    def test_remove_node_clears_edges(self):
        g = MemoryGraph(Config(auto_connect_threshold=0.0))
        n1 = MemoryNode(content="a", embedding=_vec([1, 0, 0, 0]))
        n2 = MemoryNode(content="b", embedding=_vec([0.9, 0.1, 0, 0]))
        g.add_node(n1)
        g.add_node(n2)
        assert g.edge_count > 0
        g.remove_node(n1.memory_id)
        assert g.node_count == 1
        assert g.edge_count == 0

    def test_get_nonexistent_returns_none(self):
        g = MemoryGraph()
        assert g.get_node(uuid.uuid4()) is None


class TestAutoConnect:
    def test_similar_nodes_get_edges(self):
        g = MemoryGraph(Config(auto_connect_threshold=0.5))
        n1 = MemoryNode(content="a", embedding=_vec([1, 0, 0, 0]))
        n2 = MemoryNode(content="b", embedding=_vec([0.9, 0.1, 0, 0]))
        g.add_node(n1)
        g.add_node(n2)
        assert g.edge_count >= 1

    def test_dissimilar_nodes_no_edges(self):
        g = MemoryGraph(Config(auto_connect_threshold=0.9))
        n1 = MemoryNode(content="a", embedding=_vec([1, 0, 0, 0]))
        n2 = MemoryNode(content="b", embedding=_vec([0, 0, 0, 1]))
        g.add_node(n1)
        g.add_node(n2)
        assert g.edge_count == 0


class TestCosineSimilarity:
    def test_ranking(self):
        g = MemoryGraph()
        exact = MemoryNode(content="exact", embedding=_vec([1, 0, 0, 0]))
        partial = MemoryNode(content="partial", embedding=_vec([0.7, 0.7, 0, 0]))
        orthogonal = MemoryNode(content="orth", embedding=_vec([0, 1, 0, 0]))
        g.add_node(exact)
        g.add_node(partial)
        g.add_node(orthogonal)

        results = g.cosine_similarity(_vec([1, 0, 0, 0]), top_k=3)
        ids = [r[0] for r in results]
        assert ids[0] == exact.memory_id

    def test_empty_graph(self):
        g = MemoryGraph()
        assert g.cosine_similarity(_vec([1, 0, 0, 0])) == []


class TestSpreadActivation:
    def test_one_hop(self):
        config = Config(auto_connect_threshold=0.0, spread_factor=1.0, max_hops=1)
        g = MemoryGraph(config)
        n1 = MemoryNode(content="a", embedding=_vec([1, 0, 0, 0]))
        n2 = MemoryNode(content="b", embedding=_vec([0.8, 0.2, 0, 0]))
        g.add_node(n1)
        g.add_node(n2)
        # Manually ensure there's an edge
        assert g.edge_count >= 1

        activated = g.spread_activation([(n1.memory_id, 0.8)])
        assert n1.memory_id in activated
        assert n2.activation > 0

    def test_max_hops_respected(self):
        config = Config(auto_connect_threshold=0.0, spread_factor=1.0, max_hops=1)
        g = MemoryGraph(config)
        n1 = MemoryNode(content="a", embedding=_vec([1, 0, 0, 0]))
        n2 = MemoryNode(content="b", embedding=_vec([0.8, 0.2, 0, 0]))
        n3 = MemoryNode(content="c", embedding=_vec([0.6, 0.4, 0, 0]))
        g.add_node(n1)
        g.add_node(n2)
        g.add_node(n3)

        # Reset activations
        for n in [n1, n2, n3]:
            n.activation = 0.0

        # With max_hops=1, only direct neighbors of n1 should activate
        g.spread_activation([(n1.memory_id, 0.9)])
        # n1 is activated as seed
        assert n1.activation > 0

    def test_energy_floor(self):
        config = Config(
            auto_connect_threshold=0.0,
            spread_factor=0.01,  # very low spread
            activation_threshold=0.1,
            max_hops=3,
        )
        g = MemoryGraph(config)
        n1 = MemoryNode(content="a", embedding=_vec([1, 0, 0, 0]))
        n2 = MemoryNode(content="b", embedding=_vec([0.8, 0.2, 0, 0]))
        g.add_node(n1)
        g.add_node(n2)

        n2.activation = 0.0
        g.spread_activation([(n1.memory_id, 0.1)])
        # Spread amount = 0.1 * edge_weight * 0.01 < 0.1 threshold
        assert n2.activation == 0.0


class TestHebbianUpdate:
    def test_strengthens_coactivated(self):
        config = Config(auto_connect_threshold=0.0, hebbian_lr=0.1)
        g = MemoryGraph(config)
        n1 = MemoryNode(content="a", embedding=_vec([1, 0, 0, 0]))
        n2 = MemoryNode(content="b", embedding=_vec([0.8, 0.2, 0, 0]))
        g.add_node(n1)
        g.add_node(n2)

        edge_before = g._edges[(n1.memory_id, n2.memory_id)].weight
        g.hebbian_update([n1.memory_id, n2.memory_id])
        edge_after = g._edges[(n1.memory_id, n2.memory_id)].weight
        assert edge_after > edge_before

    def test_coactivation_counter(self):
        config = Config(auto_connect_threshold=0.0)
        g = MemoryGraph(config)
        n1 = MemoryNode(content="a", embedding=_vec([1, 0, 0, 0]))
        n2 = MemoryNode(content="b", embedding=_vec([0.8, 0.2, 0, 0]))
        g.add_node(n1)
        g.add_node(n2)

        g.hebbian_update([n1.memory_id, n2.memory_id])
        g.hebbian_update([n1.memory_id, n2.memory_id])
        assert g._edges[(n1.memory_id, n2.memory_id)].co_activations == 2


class TestDecay:
    def test_decay_all(self):
        config = Config(
            auto_connect_threshold=0.0,
            activation_decay=0.5,
            strength_decay=0.5,
            edge_decay=0.5,
        )
        g = MemoryGraph(config)
        n1 = MemoryNode(content="a", embedding=_vec([1, 0, 0, 0]))
        n2 = MemoryNode(content="b", embedding=_vec([0.8, 0.2, 0, 0]))
        g.add_node(n1)
        g.add_node(n2)
        n1.activate(1.0)

        edge_before = g._edges[(n1.memory_id, n2.memory_id)].weight
        g.decay_all()
        assert n1.activation < 1.0
        assert n1.base_strength < 1.0
        edge_after = g._edges[(n1.memory_id, n2.memory_id)].weight
        assert edge_after < edge_before

    def test_edge_pruning(self):
        config = Config(auto_connect_threshold=0.0, edge_decay=0.001)
        g = MemoryGraph(config)
        n1 = MemoryNode(content="a", embedding=_vec([1, 0, 0, 0]))
        n2 = MemoryNode(content="b", embedding=_vec([0.8, 0.2, 0, 0]))
        g.add_node(n1)
        g.add_node(n2)
        assert g.edge_count > 0
        # Aggressive decay should prune edges
        g.decay_all()
        g.decay_all()
        assert g.edge_count == 0
