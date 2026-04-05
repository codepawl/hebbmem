# LoCoMo Retrieval Benchmark

Evaluates hebbmem's retrieval quality on the [LoCoMo dataset](https://github.com/snap-research/locomo) (ACL 2024, Snap Research) — the standard benchmark for long-term conversational memory.

## What is LoCoMo

LoCoMo contains 10 very long conversations (~600 turns, ~16K tokens, up to 32 sessions each) with QA annotations across 4 categories:

- **Single-hop**: answer from one session
- **Multi-hop**: answer requires connecting info from multiple sessions
- **Temporal**: time-based reasoning
- **Open-domain**: general knowledge grounded in conversation

## What we measure

hebbmem is a retrieval layer, not a QA system. We measure **retrieval quality**: given a question, does hebbmem retrieve the conversation turns that contain the answer?

Metrics: Recall@5, Recall@10, MRR

## How it works

1. Store each conversation turn as a memory (with `step()` between sessions)
2. For each QA question, `recall(question, top_k=10)`
3. Check if retrieved memories contain the ground-truth evidence turns
4. Compare hebbmem (full) vs flat baseline vs ablations (no decay, no Hebbian, no spreading)

## Run

```bash
# Quick test (1 conversation, hash encoder)
uv run python benchmarks/run_benchmark.py --locomo --conversations 1

# Full run (all 10 conversations, sentence-transformer recommended)
uv run python benchmarks/run_benchmark.py --locomo --encoder sentence-transformer
```

First run downloads ~2MB dataset file (cached locally).

## Ablations

- **no decay**: proves decay helps prioritize recent/reinforced info
- **no Hebbian**: proves co-recall bonding helps multi-hop
- **no spreading**: proves activation propagation finds indirect connections

## Limitations

- Hash encoder produces noisy results on real conversation text — use sentence-transformer for published numbers
- hebbmem doesn't do date parsing, so temporal category measures decay-based recency, not calendar reasoning
- Fuzzy text matching for evidence overlap may miss partial matches
