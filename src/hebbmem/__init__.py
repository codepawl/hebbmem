"""hebbmem - Bio-inspired memory for AI agents."""

# TODO v0.3: hebbmem/integrations/ — framework adapters (LangChain, OpenPawl, OpenCode)

from hebbmem.encoders import EncoderBackend, HashEncoder, SentenceTransformerEncoder
from hebbmem.exceptions import (
    ConfigError,
    EncoderError,
    HebbMemError,
    MemoryNotFoundError,
    PersistenceError,
)
from hebbmem.memory import HebbMem
from hebbmem.node import MemoryNode
from hebbmem.types import Config, Edge, RecallResult

__version__ = "0.2.0"
__all__ = [
    "Config",
    "ConfigError",
    "Edge",
    "EncoderBackend",
    "EncoderError",
    "HashEncoder",
    "HebbMem",
    "HebbMemError",
    "MemoryNode",
    "MemoryNotFoundError",
    "PersistenceError",
    "RecallResult",
    "SentenceTransformerEncoder",
]
