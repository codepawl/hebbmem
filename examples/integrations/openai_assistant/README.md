# OpenAI GPT + hebbmem Memory

Persistent memory layer for stateless GPT API calls.
GPT has no memory between calls — hebbmem adds it.

## Setup

1. Set your API key: `export OPENAI_API_KEY=sk-...`
2. Install deps: `pip install -r requirements.txt`
3. Run: `python assistant.py`

Memory persists across sessions in `assistant_memory.hebb`.
