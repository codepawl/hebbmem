# CrewAI Multi-Agent with Shared hebbmem Memory

Three agents share one hebbmem instance: researcher, writer, reviewer.
When the researcher finds info, the writer can recall it through
spreading activation — even topics the writer never directly saw.

## Setup

1. Install deps: `pip install -r requirements.txt`
2. Configure your LLM (set API key or use Ollama)
3. Run: `python team.py`
