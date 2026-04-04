# hebbmem Benchmarks

Compares hebbmem against a flat vector search baseline (pure cosine similarity, no decay/Hebbian/spreading) across 4 scenarios.

## Run

```bash
uv run python benchmarks/run_benchmark.py
uv run python benchmarks/run_benchmark.py --encoder sentence-transformer
uv run python benchmarks/run_benchmark.py --scenario associative
```

## Scenarios

### 1. Temporal Relevance
50 memories across 30 simulated days. Important memories stored early and reinforced; trivial memories stored recently. Tests whether reinforced important memories survive decay.

### 2. Associative Recall
30 memories with simulated co-recall patterns forming Hebbian bonds. Tests whether hebbmem finds indirectly connected memories through spreading activation — memories that share zero keywords but are linked through co-recall chains.

### 3. Noise Filtering
200 memories (30 important, 170 trivial). Heavy decay (100+ steps). Important memories reinforced periodically. Tests whether decay + importance weighting naturally surfaces signal over noise.

### 4. Contradiction Handling
10 preference pairs (old + updated). Tests whether recent, reinforced memories rank above older contradicting ones.

## Methodology

- **Same encoder** for both systems (HashEncoder by default, deterministic)
- **Same data** — identical memories, queries, and ground truth
- **Only difference** is the retrieval mechanism: hebbmem's full pipeline (decay + Hebbian + spreading activation + weighted scoring) vs baseline's pure cosine similarity
- Each scenario uses a tuned Config to highlight the relevant mechanism

## Metrics

- **Precision@k**: fraction of top-k results that are relevant
- **MRR**: reciprocal rank of the first relevant result
- **Associative Hit Rate**: fraction of relevant memories hebbmem found that baseline missed (unique to associative scenario)

## Limitations

- Synthetic data, not real-world agent conversations
- HashEncoder uses word overlap, not semantic similarity — advantages are smaller than with sentence-transformers
- Scenarios use tuned configs; real-world usage may need different tuning
