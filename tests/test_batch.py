"""Tests for store_batch."""

import pytest

from hebbmem import HebbMem


class TestStoreBatch:
    def test_basic(self):
        mem = HebbMem(encoder="hash")
        ids = mem.store_batch(["alpha", "bravo", "charlie"])
        assert len(ids) == 3
        assert mem.stats()["node_count"] == 3

    def test_single_importance(self):
        mem = HebbMem(encoder="hash")
        ids = mem.store_batch(["a", "b", "c"], importances=0.9)
        for mid in ids:
            node = mem._graph.get_node(mid)
            assert node.importance == 0.9

    def test_per_item_importance(self):
        mem = HebbMem(encoder="hash")
        imps = [0.1, 0.5, 0.9]
        ids = mem.store_batch(["a", "b", "c"], importances=imps)
        for mid, expected in zip(ids, imps, strict=True):
            node = mem._graph.get_node(mid)
            assert node.importance == expected

    def test_with_metadata(self):
        mem = HebbMem(encoder="hash")
        metas = [{"k": 1}, {"k": 2}, {"k": 3}]
        ids = mem.store_batch(["a", "b", "c"], metadata=metas)
        for mid, expected in zip(ids, metas, strict=True):
            node = mem._graph.get_node(mid)
            assert node.metadata == expected

    def test_length_mismatch_importances(self):
        mem = HebbMem(encoder="hash")
        with pytest.raises(ValueError, match="importances length"):
            mem.store_batch(["a", "b"], importances=[0.1])

    def test_length_mismatch_metadata(self):
        mem = HebbMem(encoder="hash")
        with pytest.raises(ValueError, match="metadata length"):
            mem.store_batch(["a", "b"], metadata=[{}])
