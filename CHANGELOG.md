# Changelog

All notable changes to hebbmem will be documented in this file.

Format follows [Keep a Changelog](https://keepachangelog.com/).

## [0.4.1] - 2026-04-05

### Added
- LoCoMo benchmark (ACL 2024): real-world long-term conversation retrieval evaluation
- Ablation configs: no-decay, no-Hebbian, no-spreading to prove each mechanism
- Per-category breakdown: single-hop, multi-hop, temporal, open-domain

### Fixed
- graph_example.png: label overlap resolved with alternating above/below placement
- x_thread_3.png: text overlap at bottom fixed, switched to horizontal bar layout
- .gitignore: added `benchmarks/locomo/data/` for downloaded datasets

## [0.4.0] - 2026-04-04

### Added
- Interactive D3.js visualization demo in `demo/`
- FastAPI server exposing HebbMem graph state for visualization
- Real-time animation of spreading activation, Hebbian reinforcement, and decay
- Sample data loader for demo walkthrough
- Dark theme with glowing node effects
- Integration demos: Ollama, LangChain, OpenAI, CrewAI, smolagents

## [0.3.0] - 2026-04-04

### Added
- Benchmark suite in `benchmarks/` comparing hebbmem vs flat vector search
- 4 scenarios: temporal relevance, associative recall, noise filtering, contradiction handling
- Metrics: precision@k, MRR, associative hit rate
- Entry point: `uv run python benchmarks/run_benchmark.py`
- Benchmark README with methodology and reproduction instructions
- Results table in project README

## [0.2.0] - 2026-04-04

### Added
- Persistence: `save()` and `HebbMem.load()` with single-file SQLite backend
- Thread safety: all public methods are now thread-safe via RLock
- `store_batch()` for bulk memory ingestion with vectorized encoding
- Structured logging via stdlib `logging` module (logger name: `hebbmem`)
- Custom exceptions: `HebbMemError`, `EncoderError`, `PersistenceError`, `MemoryNotFoundError`, `ConfigError`
- `py.typed` PEP 561 marker for type checker support
- Examples: `basic_usage.py`, `agent_integration.py`, `custom_config.py`
- Comprehensive Google-style docstrings on all public APIs
- CHANGELOG.md

### Changed
- `forget()` now raises `MemoryNotFoundError` instead of returning `False`
- Unknown encoder strings now raise `EncoderError` instead of `ValueError`
- `decay_all()` now returns the number of pruned edges

## [0.1.0] - 2026-04-04

### Added
- Core memory system with three bio-inspired mechanisms:
  - Temporal decay (Ebbinghaus forgetting curve)
  - Hebbian learning (co-recalled memories strengthen connections)
  - Spreading activation (recall propagates through memory graph)
- `HebbMem` public API: `store()`, `recall()`, `step()`, `forget()`, `stats()`
- Pluggable encoders: `SentenceTransformerEncoder`, `HashEncoder`
- Auto-connect: new memories automatically link to similar existing ones
- Configurable scoring weights, decay rates, and spread parameters
- 48 tests, 97% coverage
- CI/CD: GitHub Actions for testing (Python 3.10-3.12) and PyPI publishing
