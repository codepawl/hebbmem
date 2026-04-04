"""Tests for encoder backends."""

from unittest.mock import patch

import numpy as np

from hebbmem.encoders import EncoderBackend, HashEncoder, auto_select_encoder


class TestHashEncoder:
    def test_deterministic(self):
        enc = HashEncoder(dimension=128)
        v1 = enc.encode("hello world")
        v2 = enc.encode("hello world")
        np.testing.assert_array_equal(v1, v2)

    def test_unit_vector(self):
        enc = HashEncoder(dimension=128)
        v = enc.encode("some text here")
        norm = float(np.linalg.norm(v))
        assert abs(norm - 1.0) < 1e-6

    def test_different_inputs_differ(self):
        enc = HashEncoder(dimension=128)
        v1 = enc.encode("cats are great")
        v2 = enc.encode("quantum physics")
        assert not np.array_equal(v1, v2)

    def test_dimension(self):
        enc = HashEncoder(dimension=64)
        assert enc.dimension == 64
        v = enc.encode("test")
        assert v.shape == (64,)

    def test_batch(self):
        enc = HashEncoder(dimension=64)
        texts = ["hello", "world", "foo"]
        batch = enc.encode_batch(texts)
        assert batch.shape == (3, 64)
        np.testing.assert_array_equal(batch[0], enc.encode("hello"))

    def test_empty_string_returns_zero_vector(self):
        enc = HashEncoder(dimension=64)
        v = enc.encode("")
        assert float(np.linalg.norm(v)) == 0.0


class TestEncoderProtocol:
    def test_hash_encoder_is_backend(self):
        enc = HashEncoder()
        assert isinstance(enc, EncoderBackend)


class TestAutoSelect:
    def test_fallback_to_hash(self):
        with patch.dict("sys.modules", {"sentence_transformers": None}):
            enc = auto_select_encoder()
            assert isinstance(enc, HashEncoder)
