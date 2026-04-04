"""Sample memories for visualization demo."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from hebbmem import HebbMem


def load_sample_memories(mem: HebbMem) -> None:
    """Load sample memories that showcase all 3 mechanisms."""
    # Cluster 1: Work project
    mem.store("Working on Project Atlas — a recommendation engine", importance=0.8)
    mem.store("Project Atlas uses collaborative filtering", importance=0.7)
    mem.store("Atlas deadline is end of Q2", importance=0.9)
    mem.store("Team meeting every Tuesday at 10am for Atlas", importance=0.6)

    # Cluster 2: Personal preferences
    mem.store("User loves Vietnamese coffee, ca phe sua da", importance=0.7)
    mem.store("User prefers dark mode in all apps", importance=0.5)
    mem.store("User birthday is March 15", importance=0.8)

    # Cluster 3: Learning
    mem.store("User is studying Rust programming language", importance=0.6)
    mem.store("User bookmarked article about Rust async runtime", importance=0.4)
    mem.store("User wants to rewrite a Python CLI tool in Rust", importance=0.7)

    # Trivial / noise
    mem.store("User mentioned it is raining today", importance=0.1)
    mem.store("User said good morning", importance=0.05)
    mem.store("User asked about the wifi password", importance=0.1)

    # Simulate co-recall to create Hebbian bonds
    for _ in range(3):
        mem.recall("Project Atlas progress")

    for _ in range(3):
        mem.recall("Rust programming")

    # Simulate time passing
    mem.step(5)
