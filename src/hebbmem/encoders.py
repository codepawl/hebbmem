"""Pluggable encoder backends for hebbmem."""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod

import numpy as np


class EncoderBackend(ABC):
    """Base class for text-to-embedding encoders."""

    @abstractmethod
    def encode(self, text: str) -> np.ndarray:
        """Encode text to a fixed-dimension float32 vector."""
        ...

    @abstractmethod
    def encode_batch(self, texts: list[str]) -> np.ndarray:
        """Encode multiple texts. Returns shape (n, dim)."""
        ...

    @property
    @abstractmethod
    def dimension(self) -> int:
        ...


class HashEncoder(EncoderBackend):
    """Zero-dependency encoder using the hashing trick.

    Deterministic: same input always produces same vector.
    Uses multiple hash seeds to fill a fixed-dimension vector,
    then L2-normalizes so cosine similarity works correctly.
    """

    def __init__(self, dimension: int = 256, num_hashes: int = 4) -> None:
        self._dimension = dimension
        self._num_hashes = num_hashes

    @property
    def dimension(self) -> int:
        return self._dimension

    def encode(self, text: str) -> np.ndarray:
        vec = np.zeros(self._dimension, dtype=np.float32)
        tokens = text.lower().split()
        for token in tokens:
            for seed in range(self._num_hashes):
                h = int(hashlib.md5(f"{seed}:{token}".encode()).hexdigest(), 16)
                idx = h % self._dimension
                sign = 1.0 if (h // self._dimension) % 2 == 0 else -1.0
                vec[idx] += sign
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec

    def encode_batch(self, texts: list[str]) -> np.ndarray:
        return np.stack([self.encode(t) for t in texts])


class SentenceTransformerEncoder(EncoderBackend):
    """Quality encoder using sentence-transformers (optional dependency)."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name)
        self._dimension: int = self._model.get_sentence_embedding_dimension()

    @property
    def dimension(self) -> int:
        return self._dimension

    def encode(self, text: str) -> np.ndarray:
        return self._model.encode(text, convert_to_numpy=True).astype(np.float32)

    def encode_batch(self, texts: list[str]) -> np.ndarray:
        return self._model.encode(texts, convert_to_numpy=True).astype(np.float32)


def auto_select_encoder() -> EncoderBackend:
    """Return SentenceTransformerEncoder if available, else HashEncoder."""
    try:
        return SentenceTransformerEncoder()
    except ImportError:
        return HashEncoder()
