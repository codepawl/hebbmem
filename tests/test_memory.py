"""Tests for HebbMem public API."""

import uuid

import pytest

from hebbmem import Config, EncoderError, HebbMem, MemoryNotFoundError


class TestStore:
    def test_returns_uuid(self):
        mem = HebbMem(encoder="hash")
        mid = mem.store("hello world")
        assert isinstance(mid, uuid.UUID)

    def test_with_metadata(self):
        mem = HebbMem(encoder="hash")
        mem.store("hello", metadata={"source": "test"})
        results = mem.recall("hello", top_k=1)
        assert results[0].metadata == {"source": "test"}


class TestRecall:
    def test_returns_results(self):
        mem = HebbMem(encoder="hash")
        mem.store("the cat sat on the mat")
        mem.store("dogs are loyal animals")
        results = mem.recall("cat", top_k=2)
        assert len(results) > 0

    def test_score_ordering(self):
        mem = HebbMem(encoder="hash")
        mem.store("python programming language")
        mem.store("java programming language")
        mem.store("cooking recipes for dinner")
        results = mem.recall("python programming", top_k=3)
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_empty_memory(self):
        mem = HebbMem(encoder="hash")
        results = mem.recall("anything")
        assert results == []

    def test_top_k_limits(self):
        mem = HebbMem(encoder="hash")
        for i in range(10):
            mem.store(f"memory number {i}")
        results = mem.recall("memory", top_k=3)
        assert len(results) <= 3


class TestStep:
    def test_applies_decay(self):
        mem = HebbMem(encoder="hash")
        mem.store("test content")
        mem.recall("test content", top_k=1)
        mem.step(10)
        assert mem.stats()["time_step"] == 10


class TestForget:
    def test_removes_memory(self):
        mem = HebbMem(encoder="hash")
        mid = mem.store("forget me")
        assert mem.stats()["node_count"] == 1
        mem.forget(mid)
        assert mem.stats()["node_count"] == 0

    def test_nonexistent(self):
        mem = HebbMem(encoder="hash")
        with pytest.raises(MemoryNotFoundError):
            mem.forget(uuid.uuid4())


class TestStats:
    def test_returns_correct_info(self):
        mem = HebbMem(encoder="hash")
        mem.store("a")
        mem.store("b")
        stats = mem.stats()
        assert stats["node_count"] == 2
        assert stats["time_step"] == 0
        assert stats["encoder"] == "HashEncoder"


class TestConfig:
    def test_custom_config(self):
        config = Config(
            scoring_weights={
                "activation": 0.0,
                "similarity": 1.0,
                "strength": 0.0,
                "importance": 0.0,
            }
        )
        mem = HebbMem(encoder="hash", config=config)
        mem.store("exact match test")
        mem.store("completely different topic")
        results = mem.recall("exact match test", top_k=2)
        # With only similarity weight, the exact match should dominate
        assert results[0].content == "exact match test"

    def test_encoder_string_hash(self):
        mem = HebbMem(encoder="hash")
        assert mem.stats()["encoder"] == "HashEncoder"

    def test_encoder_invalid_raises(self):
        with pytest.raises(EncoderError):
            HebbMem(encoder="nonexistent")
