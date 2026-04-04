"""Benchmark metrics for comparing retrieval systems."""

from __future__ import annotations


def precision_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Of the top-k retrieved, how many are relevant?"""
    top = retrieved[:k]
    if not top:
        return 0.0
    return sum(1 for r in top if r in relevant) / len(top)


def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Of all relevant items, how many appear in top-k?"""
    if not relevant:
        return 0.0
    top = retrieved[:k]
    return sum(1 for r in top if r in relevant) / len(relevant)


def mrr(retrieved: list[str], relevant: set[str]) -> float:
    """Mean Reciprocal Rank — how high is the first relevant result?"""
    for i, r in enumerate(retrieved):
        if r in relevant:
            return 1.0 / (i + 1)
    return 0.0


def associative_hit_rate(
    hebbmem_retrieved: list[str],
    baseline_retrieved: list[str],
    relevant: set[str],
) -> float:
    """Fraction of relevant memories hebbmem found but baseline missed."""
    baseline_hits = {r for r in baseline_retrieved if r in relevant}
    hebbmem_hits = {r for r in hebbmem_retrieved if r in relevant}
    unique_to_hebbmem = hebbmem_hits - baseline_hits
    if not relevant:
        return 0.0
    return len(unique_to_hebbmem) / len(relevant)
