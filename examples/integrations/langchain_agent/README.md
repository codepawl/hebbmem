# LangChain Agent with hebbmem Memory

Drop-in replacement for LangChain's ConversationBufferMemory.
Instead of storing all conversation forever, hebbmem decays old memories,
strengthens frequently co-recalled topics, and finds related memories
through spreading activation.

## Setup

1. Install deps: `pip install -r requirements.txt`
2. Have Ollama running (`ollama serve`) or set `OPENAI_API_KEY`
3. Run: `python agent.py`
