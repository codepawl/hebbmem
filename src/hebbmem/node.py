"""MemoryNode — a single unit of memory in the graph."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class MemoryNode:
    """A single memory node with content, embedding, and activation dynamics."""

    content: str
    embedding: np.ndarray
    importance: float = 0.5
    metadata: dict[str, Any] = field(default_factory=dict)
    memory_id: uuid.UUID = field(default_factory=uuid.uuid4)
    activation: float = 0.0
    base_strength: float = 1.0
    decay_rate: float = 1.0  # per-node multiplier (1.0 = use global defaults)
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    access_count: int = 0

    def activate(self, amount: float) -> None:
        """Add activation energy, clamped to [0, 1]."""
        self.activation = min(1.0, self.activation + amount)

    def decay(self, activation_rate: float, strength_rate: float) -> None:
        """Apply one step of temporal decay to activation and strength."""
        self.activation *= activation_rate * self.decay_rate
        self.base_strength *= strength_rate

    def touch(self) -> None:
        """Record an access (updates last_accessed, increments count)."""
        self.last_accessed = time.time()
        self.access_count += 1
