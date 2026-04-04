# hebbmem

Hebbian memory for AI agents — memories that fire together wire together.

## Install

```bash
pip install hebbmem
```

For higher-quality embeddings (recommended):

```bash
pip install hebbmem[ml]
```

## Quick Start

```python
from hebbmem import HebbMem

mem = HebbMem()

# Store memories
mem.store("Python is great for data science", importance=0.8)
mem.store("JavaScript runs in the browser", importance=0.5)
mem.store("Neural networks learn from data", importance=0.7)

# Time passes, memories decay
mem.step(5)

# Recall activates related memories through the graph
results = mem.recall("machine learning with Python", top_k=3)
for r in results:
    print(f"{r.content} (score={r.score:.3f})")
```

## How It Works

hebbmem replaces flat vector storage with three neuroscience mechanisms:

**Decay** — Memories fade over time unless reinforced, following the Ebbinghaus forgetting curve. Recent and frequently accessed memories stay strong.

**Hebbian Learning** — Memories recalled together strengthen their connections. "Neurons that fire together wire together." Over time, the graph learns which memories are related through usage, not just embedding similarity.

**Spreading Activation** — Recalling one memory activates related ones through the graph, surfacing connections that keyword or vector search alone would miss.

## Links

- [GitHub](https://github.com/codepawl/hebbmem)
- [Codepawl](https://github.com/codepawl)
