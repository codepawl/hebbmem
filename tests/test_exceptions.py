"""Tests for custom exceptions."""

import uuid
from unittest.mock import patch

import pytest

from hebbmem import EncoderError, HebbMem, MemoryNotFoundError, PersistenceError
from hebbmem.encoders import SentenceTransformerEncoder


class TestMemoryNotFoundError:
    def test_forget_invalid_id(self):
        mem = HebbMem(encoder="hash")
        with pytest.raises(MemoryNotFoundError):
            mem.forget(uuid.uuid4())


class TestEncoderError:
    def test_sentence_transformer_without_package(self):
        with (
            patch.dict("sys.modules", {"sentence_transformers": None}),
            pytest.raises(EncoderError, match="sentence-transformers"),
        ):
            SentenceTransformerEncoder()

    def test_unknown_encoder_string(self):
        with pytest.raises(EncoderError, match="Unknown encoder"):
            HebbMem(encoder="nonexistent")


class TestPersistenceError:
    def test_corrupt_file(self, tmp_path):
        bad_file = tmp_path / "corrupt.hebb"
        bad_file.write_text("this is not sqlite")
        with pytest.raises(PersistenceError, match="not a valid"):
            HebbMem.load(bad_file, encoder="hash")
