"""MemoryNode — a single unit of memory in the graph."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class MemoryNode:
    """The fundamental unit of memory in the hebbmem graph.

    Each node holds text content, its vector embedding, and dynamic
    properties (activation, strength) that change over time through
    decay, recall, and Hebbian reinforcement.

    Attributes:
        content: The original text stored in this memory.
        embedding: Vector embedding of the content (numpy float32 array).
        importance: User-assigned importance (0-1). Higher values contribute
            more to recall scoring. Does not decay.
        metadata: Arbitrary key-value data attached to this memory.
        memory_id: Unique identifier (UUID4, auto-generated).
        activation: Current activation level (0-1). Rises on recall,
            decays each time step. Represents short-term salience.
        base_strength: Long-term memory strength (starts at 1.0).
            Decays slowly over time. Represents consolidation.
        decay_rate: Per-node multiplier for activation decay (default 1.0).
            Set below 1.0 to make specific memories decay faster.
        created_at: Unix timestamp when the memory was created.
        last_accessed: Unix timestamp of last recall or touch.
        access_count: Total number of times this memory was accessed.
    """

    content: str
    embedding: np.ndarray
    importance: float = 0.5
    metadata: dict[str, Any] = field(default_factory=dict)
    memory_id: uuid.UUID = field(default_factory=uuid.uuid4)
    activation: float = 0.0
    base_strength: float = 1.0
    decay_rate: float = 1.0
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    access_count: int = 0

    def activate(self, amount: float) -> None:
        """Add activation energy, clamped to [0, 1].

        Activation represents short-term salience. It rises when a memory
        is recalled (directly or via spreading activation) and decays
        each time step.

        Args:
            amount: Energy to add. Values above 1.0 are clamped.
        """
        self.activation = min(1.0, self.activation + amount)

    def decay(self, activation_rate: float, strength_rate: float) -> None:
        """Apply one step of multiplicative temporal decay.

        Both activation and strength are multiplied by their respective
        rates each step. The node's ``decay_rate`` further scales
        activation decay (e.g., 0.5 means twice as fast).

        Args:
            activation_rate: Global activation decay rate from Config.
            strength_rate: Global strength decay rate from Config.
        """
        self.activation *= activation_rate * self.decay_rate
        self.base_strength *= strength_rate

    def touch(self) -> None:
        """Record an access event.

        Called automatically during recall. Updates ``last_accessed``
        timestamp and increments ``access_count``.
        """
        self.last_accessed = time.time()
        self.access_count += 1
