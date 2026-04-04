"""Scenario 4: Contradiction Handling.

When a user updates a preference, does the memory system reflect the latest state?
"""

from __future__ import annotations

from benchmarks.baseline import FlatVectorSearch
from benchmarks.metrics import mrr, precision_at_k
from hebbmem import Config, HebbMem
from hebbmem.encoders import EncoderBackend

# (old_memory, new_memory) — both share key words so baseline can find either
PREFERENCE_PAIRS = [
    (
        "user food preference Italian pasta pizza cuisine likes",
        "user food preference Japanese sushi ramen cuisine changed updated",
    ),
    (
        "user lives District 1 Ho Chi Minh City home address",
        "user moved District 7 Ho Chi Minh City home address new",
    ),
    (
        "user editor IDE VS Code preferred development tool coding",
        "user editor Neovim switched from VS Code development tool updated",
    ),
    (
        "user commute Honda Civic car drives work transportation",
        "user commute motorbike sold car drives work transportation changed",
    ),
    (
        "user phone number 0901-234-567 contact mobile",
        "user phone number 0912-876-543 contact mobile changed new",
    ),
    (
        "user work arrangement office five days week onsite",
        "user work arrangement remote home three days week hybrid changed",
    ),
    (
        "user team communication Slack messaging chat tool",
        "user team communication Microsoft Teams messaging chat tool migrated",
    ),
    (
        "user drink preference black coffee no sugar morning beverage",
        "user drink preference matcha latte instead coffee morning beverage changed",
    ),
    (
        "user reading physical books paper before bed nighttime",
        "user reading Kindle ebooks switched digital before bed nighttime",
    ),
    (
        "user cloud provider AWS infrastructure primary services",
        "user cloud provider Google Cloud Platform migrating from AWS infrastructure",
    ),
]

# Queries — each should find the NEW preference ranked higher
QUERIES = [
    ("user food preference cuisine likes", PREFERENCE_PAIRS[0][1]),
    ("user lives home address city district", PREFERENCE_PAIRS[1][1]),
    ("user editor IDE development tool coding", PREFERENCE_PAIRS[2][1]),
    ("user commute transportation work drives", PREFERENCE_PAIRS[3][1]),
    ("user phone number contact mobile", PREFERENCE_PAIRS[4][1]),
    ("user work arrangement office remote days", PREFERENCE_PAIRS[5][1]),
    ("user team communication messaging chat", PREFERENCE_PAIRS[6][1]),
    ("user drink preference morning beverage coffee", PREFERENCE_PAIRS[7][1]),
    ("user reading books bed nighttime", PREFERENCE_PAIRS[8][1]),
    ("user cloud provider infrastructure services", PREFERENCE_PAIRS[9][1]),
]


def run(encoder: EncoderBackend) -> dict[str, dict[str, float]]:
    """Run contradiction handling scenario."""
    config = Config(
        activation_decay=0.85,
        strength_decay=0.990,
        auto_connect_threshold=0.4,
        activation_threshold=0.15,
    )
    mem = HebbMem(encoder=encoder, config=config)
    baseline = FlatVectorSearch(encoder)

    # Store old preferences
    for old, _new in PREFERENCE_PAIRS:
        mem.store(old, importance=0.7)
        baseline.store(old, importance=0.7)

    # Time passes — old memories decay
    mem.step(40)

    # Store updated preferences
    for _old, new in PREFERENCE_PAIRS:
        mem.store(new, importance=0.8)
        baseline.store(new, importance=0.8)

    mem.step(5)

    # Reinforce new preferences (user references them)
    for _old, new in PREFERENCE_PAIRS:
        mem.recall(new, top_k=3)

    mem.step(5)

    # Run queries
    h_prec1_sum = 0.0
    b_prec1_sum = 0.0
    h_mrr_sum = 0.0
    b_mrr_sum = 0.0

    for query, correct_new in QUERIES:
        relevant = {correct_new}
        h_results = [r.content for r in mem.recall(query, top_k=5)]
        b_results = baseline.recall(query, top_k=5)

        h_prec1_sum += precision_at_k(h_results, relevant, 1)
        b_prec1_sum += precision_at_k(b_results, relevant, 1)
        h_mrr_sum += mrr(h_results, relevant)
        b_mrr_sum += mrr(b_results, relevant)

    n = len(QUERIES)
    return {
        "hebbmem": {
            "precision@1": h_prec1_sum / n,
            "mrr": h_mrr_sum / n,
        },
        "baseline": {
            "precision@1": b_prec1_sum / n,
            "mrr": b_mrr_sum / n,
        },
    }
