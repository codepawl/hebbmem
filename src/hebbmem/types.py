"""Shared data structures for hebbmem."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Edge:
    """A weighted connection (synapse) between two memory nodes."""

    weight: float = 0.1
    co_activations: int = 0


@dataclass
class Config:
    """Configuration for the hebbmem memory system."""

    activation_decay: float = 0.95
    strength_decay: float = 0.999
    edge_decay: float = 0.99
    hebbian_lr: float = 0.1
    spread_factor: float = 0.5
    max_hops: int = 3
    auto_connect_threshold: float = 0.5
    activation_threshold: float = 0.1
    scoring_weights: dict[str, float] = field(default_factory=lambda: {
        "activation": 0.4,
        "similarity": 0.35,
        "strength": 0.15,
        "importance": 0.1,
    })


@dataclass
class RecallResult:
    """A single result from a recall query, with full score breakdown."""

    memory_id: uuid.UUID
    content: str
    score: float
    activation: float
    similarity: float
    strength: float
    importance: float
    metadata: dict[str, Any] = field(default_factory=dict)
