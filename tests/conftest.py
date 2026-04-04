"""Shared fixtures for hebbmem tests."""

import pytest

from hebbmem.encoders import HashEncoder
from hebbmem.node import MemoryNode
from hebbmem.types import Config


@pytest.fixture
def hash_encoder():
    return HashEncoder(dimension=64)


@pytest.fixture
def config():
    return Config()


@pytest.fixture
def make_node(hash_encoder):
    """Factory fixture to create nodes with pre-computed embeddings."""

    def _make(content: str, **kwargs) -> MemoryNode:
        embedding = hash_encoder.encode(content)
        defaults = {"content": content, "embedding": embedding}
        defaults.update(kwargs)
        return MemoryNode(**defaults)

    return _make
