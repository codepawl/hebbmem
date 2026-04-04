# hebbmem

Hebbian memory for AI agents — memories that fire together wire together.

## Install

```bash
pip install hebbmem
```

For higher-quality semantic embeddings:

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

## Persistence

Save and restore memory state across sessions:

```python
mem.save("agent_memory.hebb")

# Later...
mem = HebbMem.load("agent_memory.hebb", encoder="hash")
```

Uses SQLite internally — single file, zero dependencies, crash-safe.

## Configuration

```python
from hebbmem import HebbMem, Config

config = Config(
    activation_decay=0.9,    # how fast activation fades (0-1)
    strength_decay=0.999,    # how fast long-term strength fades (0-1)
    hebbian_lr=0.2,          # learning rate for co-activation (0-1)
    spread_factor=0.5,       # energy spread per hop (0-1)
    max_hops=3,              # BFS depth for spreading activation
    scoring_weights={        # recall ranking formula
        "activation": 0.4,
        "similarity": 0.35,
        "strength": 0.15,
        "importance": 0.1,
    },
)
mem = HebbMem(encoder="hash", config=config)
```

## Batch Store

```python
ids = mem.store_batch(
    ["memory one", "memory two", "memory three"],
    importances=[0.9, 0.5, 0.3],
)
```

## Thread Safety

All public methods are thread-safe (protected by `threading.RLock`). Safe to use from multiple threads in agent frameworks.

## Logging

hebbmem uses stdlib `logging`. Enable debug output:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Examples

See [`examples/`](examples/) for runnable scripts:

- [`basic_usage.py`](examples/basic_usage.py) — store, decay, recall, forget
- [`agent_integration.py`](examples/agent_integration.py) — using hebbmem as an agent's memory backend
- [`custom_config.py`](examples/custom_config.py) — tuning decay, Hebbian learning, and scoring weights

## Links

- [GitHub](https://github.com/codepawl/hebbmem)
- [Changelog](CHANGELOG.md)
- [Codepawl](https://github.com/codepawl)
