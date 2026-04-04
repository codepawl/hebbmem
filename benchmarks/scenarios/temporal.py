"""Scenario 1: Temporal Relevance.

After time passes, does hebbmem correctly prioritize important old memories
over trivial recent ones?
"""

from __future__ import annotations

from benchmarks.baseline import FlatVectorSearch
from benchmarks.metrics import mrr, precision_at_k
from hebbmem import Config, HebbMem
from hebbmem.encoders import EncoderBackend

# (content, importance) — Day 1-5: important user info
EARLY_IMPORTANT = [
    ("user allergy shellfish peanuts severe carry EpiPen", 1.0),
    ("user medication blood pressure pills every morning 8am", 1.0),
    ("user project Phoenix deadline end of Q2 very important", 0.9),
    ("user performance review scheduled March 15 annual", 0.9),
    ("user prefers morning meetings before 10am scheduling", 0.8),
    ("user daughter Emma piano recital April 3rd family event", 0.8),
    ("user training marathon May running exercise fitness", 0.7),
    ("user programming language Python preferred developer", 0.8),
    ("user home address 42 Nguyen Hue Boulevard District 1", 0.9),
    ("user wedding anniversary June 12th celebration dinner", 0.9),
    ("user manager Sarah Chen reports to weekly sync", 0.7),
    ("user weekly sync London team Tuesday meetings recurring", 0.8),
    ("user learning Japanese Duolingo language study hobby", 0.6),
    ("user favorite restaurant Quan Bui Ngo Duc Ke street food", 0.7),
    ("user window seat preference flights travel airplane", 0.6),
]

# (content, importance) — Day 25-30: trivial chatter with overlapping words
LATE_TRIVIAL = [
    ("user said weather nice outside sunny morning today", 0.1),
    ("user asked lunch options food nearby restaurants quick", 0.1),
    ("user mentioned traffic bad morning commute slow drive", 0.1),
    ("user said coffee already had breakfast morning drink", 0.1),
    ("user asked time meeting room schedule quick check", 0.1),
    ("user mentioned tired today afternoon energy low feeling", 0.1),
    ("user said thanks reminder helpful noted appreciated", 0.1),
    ("user mentioned rain later today weather forecast umbrella", 0.1),
    ("user said back five minutes break stepping away quick", 0.1),
    ("user asked background music play something office ambient", 0.1),
    ("user said food delivery arrived lunch order package", 0.1),
    ("user asked parking options nearby spots available cars", 0.1),
    ("user said office cold today temperature thermostat chilly", 0.1),
    ("user asked cafeteria menu today lunch options choices", 0.1),
    ("user mentioned colleague birthday tomorrow celebration cake", 0.2),
    ("user said elevator slow today building wait lobby floor", 0.1),
    ("user mentioned forgot umbrella rain today weather outside", 0.1),
    ("user asked building closing time evening late schedule", 0.1),
    ("user said enjoyed team lunch food nice group gathering", 0.1),
    ("user mentioned new coffee machine great office kitchen", 0.1),
    ("user asked dress code Friday casual policy attire", 0.1),
    ("user said need water desk plant office green care", 0.1),
    ("user mentioned sunset beautiful today evening view sky", 0.1),
    ("user asked recycling bins location office where dispose", 0.1),
    ("user mentioned walked 8000 steps today fitness exercise", 0.1),
    ("user asked nearest ATM cash withdraw bank location", 0.1),
    ("user said project demo went okay presentation showed", 0.2),
    ("user mentioned catching up emails inbox messages replies", 0.1),
    ("user said feeling energetic after lunch afternoon boost", 0.1),
    ("user asked about printer paper supplies office needs", 0.1),
    ("user mentioned funny video watched online entertainment", 0.1),
    ("user said heading quick break away returning soon step", 0.1),
    ("user mentioned directions meeting room lost building map", 0.1),
    ("user asked wifi password network connection internet", 0.1),
    ("user said good morning greeting hello start day hi", 0.1),
]

# Queries sharing key words with important memories
QUERIES = [
    (
        "user allergy medication health medical",
        {
            "user allergy shellfish peanuts severe carry EpiPen",
            "user medication blood pressure pills every morning 8am",
        },
    ),
    (
        "user project deadline work important",
        {
            "user project Phoenix deadline end of Q2 very important",
            "user performance review scheduled March 15 annual",
        },
    ),
    (
        "user meetings schedule preferences",
        {
            "user prefers morning meetings before 10am scheduling",
            "user weekly sync London team Tuesday meetings recurring",
        },
    ),
    (
        "user family events daughter anniversary",
        {
            "user daughter Emma piano recital April 3rd family event",
            "user wedding anniversary June 12th celebration dinner",
        },
    ),
    (
        "user hobby learning fitness training",
        {
            "user training marathon May running exercise fitness",
            "user learning Japanese Duolingo language study hobby",
        },
    ),
]


def run(encoder: EncoderBackend) -> dict[str, dict[str, float]]:
    """Run temporal relevance scenario."""
    config = Config(
        activation_decay=0.90,
        strength_decay=0.997,
        auto_connect_threshold=0.4,
        activation_threshold=0.15,
        scoring_weights={
            "activation": 0.3,
            "similarity": 0.3,
            "strength": 0.2,
            "importance": 0.2,
        },
    )
    mem = HebbMem(encoder=encoder, config=config)
    baseline = FlatVectorSearch(encoder)

    # Day 1-5: store important memories
    for content, importance in EARLY_IMPORTANT:
        mem.store(content, importance=importance)
        baseline.store(content, importance=importance)

    # Simulate 20 days
    mem.step(20)

    # Reinforce important memories
    reinforce_queries = [
        "allergy medication health",
        "project deadline work",
        "meetings schedule",
        "family daughter anniversary",
        "hobby training learning",
    ]
    for q in reinforce_queries:
        mem.recall(q, top_k=3)
    mem.step(5)

    # Day 25-30: store trivial noise
    for content, importance in LATE_TRIVIAL:
        mem.store(content, importance=importance)
        baseline.store(content, importance=importance)

    mem.step(5)

    # Run queries
    h_prec_sum = 0.0
    b_prec_sum = 0.0
    h_mrr_sum = 0.0
    b_mrr_sum = 0.0

    for query, relevant in QUERIES:
        h_results = [r.content for r in mem.recall(query, top_k=5)]
        b_results = baseline.recall(query, top_k=5)

        h_prec_sum += precision_at_k(h_results, relevant, 5)
        b_prec_sum += precision_at_k(b_results, relevant, 5)
        h_mrr_sum += mrr(h_results, relevant)
        b_mrr_sum += mrr(b_results, relevant)

    n = len(QUERIES)
    return {
        "hebbmem": {
            "precision@5": h_prec_sum / n,
            "mrr": h_mrr_sum / n,
        },
        "baseline": {
            "precision@5": b_prec_sum / n,
            "mrr": b_mrr_sum / n,
        },
    }
