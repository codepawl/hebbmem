# Ollama Chat with hebbmem Memory

A local chatbot that remembers like a brain. No API keys, no cloud.

## Setup

1. Install Ollama: https://ollama.com
2. Pull a small model: `ollama pull qwen2.5:3b`
3. Install deps: `pip install -r requirements.txt`
4. Run: `python chat.py`

## What to try

- Chat about a few topics, then quit and restart — it remembers
- Talk about Topic A and Topic B together several times, then only mention A — it recalls B (Hebbian)
- Wait (or run many steps) — old trivial memories fade, important ones stay (decay)

## Works with any Ollama model

Change MODEL in chat.py: `llama3.2:3b`, `mistral:7b`, `phi3:mini`, etc.
