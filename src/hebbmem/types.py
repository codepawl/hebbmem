"""Shared data structures for hebbmem."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Edge:
    """A weighted connection (synapse) between two memory nodes.

    Attributes:
        weight: Connection strength between 0 and 1. Higher means the
            two memories are more strongly associated. Starts at initial
            cosine similarity, grows via Hebbian learning, decays over time.
        co_activations: Number of times both connected nodes were activated
            in the same recall. Useful for debugging and visualization.
    """

    weight: float = 0.1
    co_activations: int = 0


@dataclass
class Config:
    """Configuration for the hebbmem memory system.

    All decay rates are per-step multipliers applied by ``step()``.
    Values closer to 1.0 mean slower decay.

    Attributes:
        activation_decay: Per-step multiplier for node activation (0-1).
            Lower values make activation fade faster after recall.
        strength_decay: Per-step multiplier for node base_strength (0-1).
            Controls long-term memory fade. 0.999 means very slow decay.
        edge_decay: Per-step multiplier for edge weights (0-1).
            Edges below 0.01 are pruned automatically.
        hebbian_lr: Learning rate for Hebbian reinforcement (0-1).
            Higher values strengthen co-activated edges faster.
        spread_factor: Multiplier for activation spread per hop (0-1).
            Higher values propagate more energy through the graph.
        max_hops: Maximum BFS depth for spreading activation.
            More hops = wider associative recall but slower.
        auto_connect_threshold: Cosine similarity threshold (0-1) for
            automatically creating edges when storing new memories.
        activation_threshold: Minimum activation energy to propagate
            during spreading activation. Prevents noisy low-energy spread.
        scoring_weights: Weights for the recall scoring formula.
            Keys: "activation", "similarity", "strength", "importance".
            Values should sum to 1.0 for normalized scores.
    """

    activation_decay: float = 0.95
    strength_decay: float = 0.999
    edge_decay: float = 0.99
    hebbian_lr: float = 0.1
    spread_factor: float = 0.5
    max_hops: int = 3
    auto_connect_threshold: float = 0.5
    activation_threshold: float = 0.1
    scoring_weights: dict[str, float] = field(
        default_factory=lambda: {
            "activation": 0.4,
            "similarity": 0.35,
            "strength": 0.15,
            "importance": 0.1,
        }
    )


@dataclass
class RecallResult:
    """A single result from a recall query, with full score breakdown.

    Attributes:
        memory_id: Unique identifier of the recalled memory.
        content: The original text that was stored.
        score: Final weighted score used for ranking. Computed as
            ``sum(scoring_weights[k] * component[k])``.
        activation: Current activation level (0-1). High when recently
            recalled or reached via spreading activation.
        similarity: Cosine similarity to the query embedding (0-1).
        strength: Base memory strength (starts at 1.0, decays over time).
        importance: User-assigned importance when stored (0-1).
        metadata: Arbitrary dict attached when the memory was stored.
    """

    memory_id: uuid.UUID
    content: str
    score: float
    activation: float
    similarity: float
    strength: float
    importance: float
    metadata: dict[str, Any] = field(default_factory=dict)
