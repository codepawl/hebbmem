"""Tests for MemoryNode."""

import time
import uuid

import numpy as np

from hebbmem.node import MemoryNode


def _make_node(**kwargs) -> MemoryNode:
    defaults = {"content": "test", "embedding": np.zeros(64, dtype=np.float32)}
    defaults.update(kwargs)
    return MemoryNode(**defaults)


class TestMemoryNodeCreation:
    def test_defaults(self):
        node = _make_node()
        assert isinstance(node.memory_id, uuid.UUID)
        assert node.activation == 0.0
        assert node.base_strength == 1.0
        assert node.importance == 0.5
        assert node.access_count == 0
        assert node.metadata == {}

    def test_custom_values(self):
        mid = uuid.uuid4()
        node = _make_node(importance=0.9, memory_id=mid, metadata={"tag": "a"})
        assert node.memory_id == mid
        assert node.importance == 0.9
        assert node.metadata == {"tag": "a"}


class TestActivation:
    def test_activate_adds(self):
        node = _make_node()
        node.activate(0.5)
        assert node.activation == 0.5

    def test_activate_clamps_to_one(self):
        node = _make_node()
        node.activate(0.7)
        node.activate(0.5)
        assert node.activation == 1.0


class TestDecay:
    def test_decay_reduces_activation(self):
        node = _make_node()
        node.activate(1.0)
        node.decay(activation_rate=0.9, strength_rate=1.0)
        assert node.activation == 0.9

    def test_decay_reduces_strength(self):
        node = _make_node()
        node.decay(activation_rate=1.0, strength_rate=0.5)
        assert node.base_strength == 0.5

    def test_custom_decay_rate_multiplier(self):
        node = _make_node(decay_rate=0.5)
        node.activate(1.0)
        node.decay(activation_rate=0.9, strength_rate=1.0)
        assert abs(node.activation - 0.45) < 1e-7


class TestTouch:
    def test_touch_updates_access(self):
        node = _make_node()
        before = node.last_accessed
        time.sleep(0.01)
        node.touch()
        assert node.last_accessed > before
        assert node.access_count == 1

    def test_touch_increments_count(self):
        node = _make_node()
        node.touch()
        node.touch()
        assert node.access_count == 2
