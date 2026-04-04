# hebbmem Integration Demos

Drop-in memory backend for any AI agent framework.

| Demo | Framework | Model | Setup Time | API Key? |
|------|-----------|-------|------------|----------|
| [ollama_chat](ollama_chat/) | Raw HTTP | Ollama (local) | 2 min | No |
| [langchain_agent](langchain_agent/) | LangChain | Ollama or OpenAI | 3 min | Optional |
| [openai_assistant](openai_assistant/) | OpenAI SDK | GPT-4o-mini | 1 min | Yes |
| [crewai_team](crewai_team/) | CrewAI | Any | 3 min | Depends |
| [smolagents_tool](smolagents_tool/) | smolagents | HF Inference | 2 min | No |

## The Pattern

Every integration follows the same 3-line pattern:

```python
mem = HebbMem(encoder="hash")
mem.store("something to remember")
results = mem.recall("related query")
mem.step(1)
```

hebbmem doesn't know about LangChain, CrewAI, or any framework.
It just provides store/recall/step. You write a thin adapter.

## Start Here

If you have Ollama installed: `ollama_chat/` (zero API keys, runs locally).
If you have an OpenAI key: `openai_assistant/` (simplest code).
If you use LangChain: `langchain_agent/` (drop-in memory replacement).
