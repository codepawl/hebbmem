"""hebbmem - Bio-inspired memory for AI agents."""

# TODO v0.2: py.typed — PEP 561 marker
# TODO v0.2: hebbmem/integrations/ — framework adapters (LangChain, OpenPawl, OpenCode)

from hebbmem.encoders import EncoderBackend, HashEncoder, SentenceTransformerEncoder
from hebbmem.memory import HebbMem
from hebbmem.node import MemoryNode
from hebbmem.types import Config, Edge, RecallResult

__version__ = "0.1.0"
__all__ = [
    "HebbMem",
    "Config",
    "Edge",
    "RecallResult",
    "MemoryNode",
    "EncoderBackend",
    "HashEncoder",
    "SentenceTransformerEncoder",
]
