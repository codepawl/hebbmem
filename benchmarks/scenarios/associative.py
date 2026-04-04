"""Scenario 2: Associative Recall.

Can hebbmem find memories related through co-recall patterns,
not just keyword similarity? This is the key differentiator.
"""

from __future__ import annotations

from benchmarks.baseline import FlatVectorSearch
from benchmarks.metrics import associative_hit_rate, mrr, precision_at_k
from hebbmem import Config, HebbMem
from hebbmem.encoders import EncoderBackend

# Memories across different topics
MEMORIES = [
    # Project cluster
    ("Project Alpha deadline is end of March", 0.8),
    ("Project Alpha uses a microservices architecture", 0.6),
    ("Team standup notes from this week about Alpha progress", 0.7),
    ("Budget approval is pending for Q2 infrastructure upgrade", 0.7),
    ("Infrastructure costs for cloud servers are rising", 0.5),
    # People cluster
    ("Sarah is the lead engineer on Project Alpha", 0.7),
    ("Sarah recommended using Kubernetes for deployment", 0.6),
    ("David handles budget approvals for engineering", 0.7),
    ("David mentioned cost optimization is a priority this quarter", 0.6),
    # Meeting cluster
    ("Monday all-hands meeting covered Q1 results", 0.5),
    ("Q1 results showed 15% growth in user engagement", 0.6),
    ("User engagement metrics are tracked in the analytics dashboard", 0.5),
    # Client cluster
    ("Client Acme Corp requested a demo of the new feature", 0.8),
    ("The new feature includes real-time collaboration", 0.7),
    ("Real-time collaboration requires WebSocket infrastructure", 0.6),
    # Hiring cluster
    ("We are hiring two senior backend engineers", 0.6),
    ("Backend engineer job posting closes next Friday", 0.5),
    ("Interview panel includes Sarah and David", 0.6),
    # Distractors — topically similar but not co-recalled
    ("Alpha testing methodology for QA processes", 0.3),
    ("Budget spreadsheet template was updated last month", 0.3),
    ("March Madness basketball tournament schedule", 0.2),
    ("Deadline management best practices article", 0.3),
    ("Infrastructure as code tutorial for beginners", 0.3),
    ("Team building event at the bowling alley", 0.3),
    ("Weekly grocery budget planning tips", 0.2),
    ("Alpha radiation safety guidelines for lab work", 0.2),
    ("Cloud computing certification study guide", 0.3),
    ("Architecture portfolio review for design team", 0.3),
    ("Standup comedy show tickets for Saturday", 0.2),
    ("Progress bar UI component library update", 0.3),
]

# Co-recall patterns to simulate (pairs recalled together)
CO_RECALL_PATTERNS = [
    # Project Alpha <-> Team standup -> Budget approval (chain)
    (
        "Project Alpha deadline is end of March",
        "Team standup notes from this week about Alpha progress",
    ),
    (
        "Team standup notes from this week about Alpha progress",
        "Budget approval is pending for Q2 infrastructure upgrade",
    ),
    # Sarah <-> Project Alpha <-> Kubernetes
    (
        "Sarah is the lead engineer on Project Alpha",
        "Project Alpha uses a microservices architecture",
    ),
    (
        "Sarah recommended using Kubernetes for deployment",
        "Project Alpha uses a microservices architecture",
    ),
    # David <-> Budget <-> Cost
    (
        "David handles budget approvals for engineering",
        "Budget approval is pending for Q2 infrastructure upgrade",
    ),
    (
        "David mentioned cost optimization is a priority this quarter",
        "Infrastructure costs for cloud servers are rising",
    ),
    # Client <-> Feature <-> WebSocket
    (
        "Client Acme Corp requested a demo of the new feature",
        "The new feature includes real-time collaboration",
    ),
    (
        "Real-time collaboration requires WebSocket infrastructure",
        "Infrastructure costs for cloud servers are rising",
    ),
    # Hiring <-> Sarah and David
    (
        "Interview panel includes Sarah and David",
        "We are hiring two senior backend engineers",
    ),
    (
        "Interview panel includes Sarah and David",
        "Sarah is the lead engineer on Project Alpha",
    ),
]

# Queries with relevant sets including association-only targets
QUERIES = [
    (
        "What's happening with Project Alpha?",
        {
            "Project Alpha deadline is end of March",
            "Project Alpha uses a microservices architecture",
            "Team standup notes from this week about Alpha progress",
            "Sarah is the lead engineer on Project Alpha",
            # Association-only (connected through co-recall chain):
            "Budget approval is pending for Q2 infrastructure upgrade",
        },
    ),
    (
        "What does David work on?",
        {
            "David handles budget approvals for engineering",
            "David mentioned cost optimization is a priority this quarter",
            "Interview panel includes Sarah and David",
            # Association-only:
            "Budget approval is pending for Q2 infrastructure upgrade",
            "We are hiring two senior backend engineers",
        },
    ),
    (
        "What infrastructure work is needed?",
        {
            "Infrastructure costs for cloud servers are rising",
            "Real-time collaboration requires WebSocket infrastructure",
            "Sarah recommended using Kubernetes for deployment",
            # Association-only:
            "Budget approval is pending for Q2 infrastructure upgrade",
            "The new feature includes real-time collaboration",
        },
    ),
    (
        "Tell me about the client demo",
        {
            "Client Acme Corp requested a demo of the new feature",
            "The new feature includes real-time collaboration",
            # Association-only:
            "Real-time collaboration requires WebSocket infrastructure",
        },
    ),
    (
        "What's the hiring status?",
        {
            "We are hiring two senior backend engineers",
            "Backend engineer job posting closes next Friday",
            "Interview panel includes Sarah and David",
            # Association-only:
            "Sarah is the lead engineer on Project Alpha",
        },
    ),
]


def run(encoder: EncoderBackend) -> dict[str, dict[str, float]]:
    """Run associative recall scenario."""
    config = Config(
        hebbian_lr=0.15,
        spread_factor=0.6,
        max_hops=3,
        auto_connect_threshold=0.3,
        activation_threshold=0.05,
    )
    mem = HebbMem(encoder=encoder, config=config)
    baseline = FlatVectorSearch(encoder)

    # Store all memories
    for content, importance in MEMORIES:
        mem.store(content, importance=importance)
        baseline.store(content, importance=importance)

    # Simulate co-recall patterns (Hebbian bonding)
    for _ in range(5):
        for mem_a, mem_b in CO_RECALL_PATTERNS:
            # Recall both together to form Hebbian bonds
            mem.recall(mem_a, top_k=5)
            mem.recall(mem_b, top_k=5)
        mem.step(1)

    # Run queries
    h_prec_sum = 0.0
    b_prec_sum = 0.0
    h_mrr_sum = 0.0
    b_mrr_sum = 0.0
    h_assoc_sum = 0.0

    for query, relevant in QUERIES:
        h_results = [r.content for r in mem.recall(query, top_k=5)]
        b_results = baseline.recall(query, top_k=5)

        h_prec_sum += precision_at_k(h_results, relevant, 5)
        b_prec_sum += precision_at_k(b_results, relevant, 5)
        h_mrr_sum += mrr(h_results, relevant)
        b_mrr_sum += mrr(b_results, relevant)
        h_assoc_sum += associative_hit_rate(h_results, b_results, relevant)

    n = len(QUERIES)
    return {
        "hebbmem": {
            "precision@5": h_prec_sum / n,
            "mrr": h_mrr_sum / n,
            "assoc_hit_rate": h_assoc_sum / n,
        },
        "baseline": {
            "precision@5": b_prec_sum / n,
            "mrr": b_mrr_sum / n,
            "assoc_hit_rate": 0.0,
        },
    }
