"""Tests for save/load persistence."""

import pytest

from hebbmem import Config, HebbMem


class TestSaveLoadRoundtrip:
    def test_roundtrip(self, tmp_path):
        path = tmp_path / "test.hebb"
        mem = HebbMem(encoder="hash")
        mem.store("alpha bravo charlie")
        mem.store("delta echo foxtrot")
        mem.store("golf hotel india")
        mem.save(path)

        mem2 = HebbMem.load(path, encoder="hash")
        assert mem2.stats()["node_count"] == 3
        assert mem2.stats()["edge_count"] == mem.stats()["edge_count"]
        results = mem2.recall("alpha bravo", top_k=3)
        contents = {r.content for r in results}
        assert "alpha bravo charlie" in contents


class TestPreservesActivations:
    def test_activations_after_decay(self, tmp_path):
        path = tmp_path / "test.hebb"
        mem = HebbMem(encoder="hash")
        mid = mem.store("test activation")
        mem.recall("test activation")  # activates the node
        mem.step(5)

        # Get activation before save
        node_before = mem._graph.get_node(mid)
        act_before = node_before.activation
        strength_before = node_before.base_strength

        mem.save(path)
        mem2 = HebbMem.load(path, encoder="hash")
        node_after = mem2._graph.get_node(mid)

        assert abs(node_after.activation - act_before) < 1e-7
        assert abs(node_after.base_strength - strength_before) < 1e-7


class TestPreservesEdges:
    def test_edge_weights_and_coactivations(self, tmp_path):
        path = tmp_path / "test.hebb"
        mem = HebbMem(encoder="hash")
        mem.store("machine learning algorithms")
        mem.store("machine learning with python")
        # Recall triggers Hebbian update
        for _ in range(3):
            mem.recall("machine learning")

        edges_before = {
            k: (e.weight, e.co_activations) for k, e in mem._graph._edges.items()
        }
        mem.save(path)

        mem2 = HebbMem.load(path, encoder="hash")
        for k, (w, co) in edges_before.items():
            edge = mem2._graph._edges[k]
            assert abs(edge.weight - w) < 1e-7
            assert edge.co_activations == co


class TestPreservesConfig:
    def test_custom_config_roundtrip(self, tmp_path):
        path = tmp_path / "test.hebb"
        config = Config(
            activation_decay=0.8,
            hebbian_lr=0.2,
            max_hops=5,
            scoring_weights={
                "activation": 0.5,
                "similarity": 0.3,
                "strength": 0.1,
                "importance": 0.1,
            },
        )
        mem = HebbMem(encoder="hash", config=config)
        mem.store("test")
        mem.save(path)

        mem2 = HebbMem.load(path, encoder="hash")
        assert mem2.config.activation_decay == 0.8
        assert mem2.config.hebbian_lr == 0.2
        assert mem2.config.max_hops == 5
        assert mem2.config.scoring_weights["activation"] == 0.5


class TestRecallAfterLoad:
    def test_recall_works(self, tmp_path):
        path = tmp_path / "test.hebb"
        mem = HebbMem(encoder="hash")
        mem.store("python programming language", importance=0.9)
        mem.store("cooking pasta recipes", importance=0.3)
        mem.save(path)

        mem2 = HebbMem.load(path, encoder="hash")
        results = mem2.recall("python programming", top_k=2)
        assert len(results) > 0
        assert results[0].content == "python programming language"


class TestSaveOverwrite:
    def test_overwrite_no_error(self, tmp_path):
        path = tmp_path / "test.hebb"
        mem = HebbMem(encoder="hash")
        mem.store("first")
        mem.save(path)
        mem.store("second")
        mem.save(path)  # should overwrite without error

        mem2 = HebbMem.load(path, encoder="hash")
        assert mem2.stats()["node_count"] == 2


class TestLoadFileNotFound:
    def test_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            HebbMem.load(tmp_path / "nonexistent.hebb")


class TestSaveLoadEmpty:
    def test_empty_roundtrip(self, tmp_path):
        path = tmp_path / "test.hebb"
        mem = HebbMem(encoder="hash")
        mem.save(path)

        mem2 = HebbMem.load(path, encoder="hash")
        assert mem2.stats()["node_count"] == 0
        assert mem2.stats()["edge_count"] == 0


class TestStepCountPreserved:
    def test_step_count(self, tmp_path):
        path = tmp_path / "test.hebb"
        mem = HebbMem(encoder="hash")
        mem.step(10)
        mem.save(path)

        mem2 = HebbMem.load(path, encoder="hash")
        assert mem2.stats()["time_step"] == 10
