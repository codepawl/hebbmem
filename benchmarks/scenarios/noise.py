"""Scenario 3: Noise Filtering.

After many decay cycles, does hebbmem surface important memories
while burying trivial ones?
"""

from __future__ import annotations

import random

from benchmarks.baseline import FlatVectorSearch
from benchmarks.metrics import mrr, precision_at_k
from hebbmem import Config, HebbMem
from hebbmem.encoders import EncoderBackend

# Important memories with distinctive keywords
IMPORTANT = [
    ("user allergy shellfish peanuts severe EpiPen medical", 0.9),
    ("project Phoenix launch date April 30th deadline critical", 0.9),
    ("user medication blood pressure pills 8am daily health", 1.0),
    ("user daughter Emma school starts 7:30am morning family", 0.8),
    ("quarterly board presentation 15th each month work report", 0.8),
    ("user flight Tokyo booked May 10th travel international", 0.9),
    ("cloud services subscription renewal June 1st annual payment", 0.8),
    ("user vegetarian meals lunch preference diet food choice", 0.7),
    ("client Acme Corp contract expires end Q3 important deal", 0.9),
    ("user car insurance renewal July payment vehicle coverage", 0.7),
    ("team offsite retreat August Da Lat planning trip event", 0.8),
    ("user passport expires September needs renewal travel document", 0.9),
    ("user saving house monthly budget 30M VND finance goal", 0.8),
    ("user dentist appointment every 6 months dental health checkup", 0.7),
    ("user mentors junior engineers Thursdays teaching coaching team", 0.7),
    ("critical security patch deploy before Friday urgent server", 0.9),
    ("user gym membership auto-renews October fitness exercise", 0.6),
    ("tax filing deadline March 31st financial important annual", 0.9),
    ("user mother birthday November 5th family celebration gift", 0.8),
    ("backup server maintenance window Saturday 2-4am ops schedule", 0.8),
    ("user book club meeting first Monday monthly reading group", 0.6),
    ("user annual health checkup December medical appointment exam", 0.7),
    ("sprint retrospective every other Friday agile team meeting", 0.7),
    ("user expense tracking spreadsheet review weekly finance budget", 0.6),
    ("user wants learn Rust programming language goal development", 0.6),
    ("production database backup midnight UTC ops infrastructure", 0.8),
    ("user wedding anniversary dinner reservation 7pm celebration", 0.8),
    ("team hiring goal 3 engineers end Q2 recruitment staffing", 0.7),
    ("user aisle seat preference long flights travel comfort", 0.6),
    ("code freeze release v2.0 starts May 1st deployment important", 0.9),
]

# Trivial noise with some overlapping common words
_NOISE_TEMPLATES = [
    "user said {} today casual chat small talk",
    "user mentioned {} this morning conversation brief",
    "user asked about {} in passing question quick",
    "user commented {} during break idle chat",
    "user noted {} briefly passing remark offhand",
]
_NOISE_TOPICS = [
    "weather nice sunny outside warm",
    "traffic bad commute slow roads",
    "funny meme online internet humor",
    "coffee machine new office kitchen",
    "elevator music lobby building boring",
    "podcast episode interesting listened audio",
    "office temperature cold warm thermostat",
    "new restaurant nearby opened food",
    "parking situation lot full spaces",
    "TV show last night watched entertainment",
    "vending machine snacks selection options",
    "wifi slow internet connection network",
    "neighbor dog barking noise pets",
    "sunset yesterday evening beautiful colors",
    "sale mall shopping discount deals",
    "gym crowded evening busy workout",
    "recipe tried cooking dinner homemade",
    "new intern started team office junior",
    "sports game score results match",
    "printer jamming again paper stuck",
    "bird window outside nature chirping",
    "water cooler chat conversation gossip",
    "delivery delay package late shipping",
    "cleaning schedule office tidy maintenance",
    "road construction detour traffic route",
    "lobby renovation building upgrade update",
    "coworker haircut new look style",
    "lunch special today cafeteria menu",
    "phone notification sound alert ring",
    "building fire drill practice safety",
    "stain shirt accident clothes messy",
    "bus late commute public transport",
    "power outage night electricity blackout",
    "news headlines today current events",
]

# Queries with keywords matching important memories
QUERIES = [
    (
        "user health medical allergy medication",
        {
            "user allergy shellfish peanuts severe EpiPen medical",
            "user medication blood pressure pills 8am daily health",
            "user dentist appointment every 6 months dental health checkup",
            "user annual health checkup December medical appointment exam",
        },
    ),
    (
        "project deadline launch important date",
        {
            "project Phoenix launch date April 30th deadline critical",
            "quarterly board presentation 15th each month work report",
            "client Acme Corp contract expires end Q3 important deal",
            "tax filing deadline March 31st financial important annual",
            "code freeze release v2.0 starts May 1st deployment important",
        },
    ),
    (
        "user family daughter mother celebration",
        {
            "user daughter Emma school starts 7:30am morning family",
            "user mother birthday November 5th family celebration gift",
            "user wedding anniversary dinner reservation 7pm celebration",
        },
    ),
    (
        "user travel flight passport trip",
        {
            "user flight Tokyo booked May 10th travel international",
            "team offsite retreat August Da Lat planning trip event",
            "user aisle seat preference long flights travel comfort",
            "user passport expires September needs renewal travel document",
        },
    ),
    (
        "team meeting sprint schedule recurring",
        {
            "sprint retrospective every other Friday agile team meeting",
            "user mentors junior engineers Thursdays teaching coaching team",
            "backup server maintenance window Saturday 2-4am ops schedule",
            "production database backup midnight UTC ops infrastructure",
        },
    ),
]


def _generate_trivial(n: int) -> list[tuple[str, float]]:
    """Generate n trivial noise memories."""
    rng = random.Random(42)
    result = []
    for i in range(n):
        template = rng.choice(_NOISE_TEMPLATES)
        topic = _NOISE_TOPICS[i % len(_NOISE_TOPICS)]
        content = template.format(topic)
        importance = round(rng.uniform(0.0, 0.2), 2)
        result.append((content, importance))
    return result


def run(encoder: EncoderBackend) -> dict[str, dict[str, float]]:
    """Run noise filtering scenario."""
    config = Config(
        activation_decay=0.88,
        strength_decay=0.996,
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

    trivial = _generate_trivial(170)

    # Store important memories
    for content, importance in IMPORTANT:
        mem.store(content, importance=importance)
        baseline.store(content, importance=importance)

    mem.step(10)

    # Reinforce important memories
    reinforce = [
        "allergy medication health medical",
        "deadline project launch date",
        "family daughter mother events",
        "travel flight passport trip",
        "team meeting sprint schedule",
        "finance budget payment renewal",
        "server infrastructure ops backup",
    ]
    for q in reinforce:
        mem.recall(q, top_k=5)

    mem.step(10)

    # Store trivial noise
    for content, importance in trivial:
        mem.store(content, importance=importance)
        baseline.store(content, importance=importance)

    # Heavy decay
    mem.step(80)

    # Reinforce important memories again
    for q in reinforce:
        mem.recall(q, top_k=5)

    mem.step(10)

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
