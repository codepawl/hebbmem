"""Pluggable encoder backends for hebbmem.

Encoders convert text into fixed-dimension vectors (embeddings) used for
cosine similarity in the memory graph. Two backends are provided:

- ``SentenceTransformerEncoder``: High-quality semantic embeddings.
  Requires ``pip install hebbmem[ml]``.
- ``HashEncoder``: Zero-dependency fallback using the hashing trick.
  Deterministic but not semantic — suitable for testing and prototyping.
"""

from __future__ import annotations

import hashlib
import logging
from abc import ABC, abstractmethod

import numpy as np

from hebbmem.exceptions import EncoderError

logger = logging.getLogger("hebbmem")


class EncoderBackend(ABC):
    """Abstract base class for text-to-embedding encoders.

    Subclass this to create a custom encoder. Must implement
    ``encode()``, ``encode_batch()``, and the ``dimension`` property.
    """

    @abstractmethod
    def encode(self, text: str) -> np.ndarray:
        """Encode text to a fixed-dimension float32 vector.

        Args:
            text: Input text string.

        Returns:
            Numpy float32 array of shape ``(dimension,)``.
        """
        ...

    @abstractmethod
    def encode_batch(self, texts: list[str]) -> np.ndarray:
        """Encode multiple texts in one call.

        Args:
            texts: List of input strings.

        Returns:
            Numpy float32 array of shape ``(len(texts), dimension)``.
        """
        ...

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Embedding vector dimension."""
        ...


class HashEncoder(EncoderBackend):
    """Zero-dependency encoder using the hashing trick.

    Produces deterministic, L2-normalized vectors by hashing tokens
    into a fixed-dimension space. Not semantic — "cat" and "kitten"
    will have unrelated vectors. Best for testing and environments
    where sentence-transformers cannot be installed.

    Args:
        dimension: Output vector dimension. Higher = fewer hash collisions.
        num_hashes: Hash seeds per token. More = denser vectors.
    """

    def __init__(self, dimension: int = 256, num_hashes: int = 4) -> None:
        self._dimension = dimension
        self._num_hashes = num_hashes
        logger.info("Using encoder: HashEncoder (dimension=%d)", dimension)

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
    """Semantic encoder using sentence-transformers.

    Produces high-quality embeddings where semantically similar texts
    have high cosine similarity. Uses the ``all-MiniLM-L6-v2`` model
    by default (384 dimensions, fast, good general-purpose quality).

    Requires: ``pip install hebbmem[ml]``

    Args:
        model_name: HuggingFace model name or path.

    Raises:
        EncoderError: If sentence-transformers is not installed.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise EncoderError(
                "sentence-transformers is required for SentenceTransformerEncoder. "
                "Install with: pip install hebbmem[ml]"
            ) from e

        self._model = SentenceTransformer(model_name)
        dim = self._model.get_sentence_embedding_dimension()
        assert dim is not None
        self._dimension: int = dim
        logger.info(
            "Using encoder: SentenceTransformerEncoder (model=%s, dimension=%d)",
            model_name,
            self._dimension,
        )

    @property
    def dimension(self) -> int:
        return self._dimension

    def encode(self, text: str) -> np.ndarray:
        raw = self._model.encode(text, convert_to_numpy=True)
        result: np.ndarray = raw.astype(np.float32)
        return result

    def encode_batch(self, texts: list[str]) -> np.ndarray:
        raw = self._model.encode(texts, convert_to_numpy=True)
        result: np.ndarray = raw.astype(np.float32)
        return result


def auto_select_encoder() -> EncoderBackend:
    """Return the best available encoder.

    Tries ``SentenceTransformerEncoder`` first. If sentence-transformers
    is not installed, falls back to ``HashEncoder`` with a warning.
    """
    try:
        return SentenceTransformerEncoder()
    except EncoderError:
        logger.warning("sentence-transformers not found, falling back to HashEncoder")
        return HashEncoder()
