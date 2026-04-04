"""Chatbot with brain-like memory, powered by Ollama + hebbmem.

Memories decay over time, frequently discussed topics bond together,
and recalling one memory activates related ones — just like your brain.

Usage:
    ollama pull qwen2.5:3b
    pip install hebbmem httpx
    python chat.py

Memory persists across sessions in chat_memory.hebb
"""

from __future__ import annotations

import httpx

from hebbmem import HebbMem

MEMORY_FILE = "chat_memory.hebb"
MODEL = "qwen2.5:3b"
OLLAMA_URL = "http://localhost:11434/api/generate"


def load_or_create_memory() -> HebbMem:
    try:
        return HebbMem.load(MEMORY_FILE, encoder="hash")
    except FileNotFoundError:
        return HebbMem(encoder="hash")


def ask_ollama(prompt: str, context: str = "") -> str:
    system = (
        "You are a helpful assistant with memory. "
        "Here are relevant memories from past conversations:\n"
        f"{context if context else '(no relevant memories yet)'}\n\n"
        "Use these memories naturally. Don't list them — just be helpful."
    )

    response = httpx.post(
        OLLAMA_URL,
        json={"model": MODEL, "prompt": prompt, "system": system, "stream": False},
        timeout=60,
    )
    response.raise_for_status()
    return response.json()["response"]


def main() -> None:
    mem = load_or_create_memory()
    stats = mem.stats()
    print(f"hebbmem chatbot | {stats['node_count']} memories loaded")
    print("Type 'quit' to exit, 'stats' for memory stats, 'forget' to reset\n")

    turn = 0
    while True:
        user_input = input("You: ").strip()
        if not user_input:
            continue
        if user_input == "quit":
            mem.save(MEMORY_FILE)
            print(f"Saved {mem.stats()['node_count']} memories. Bye!")
            break
        if user_input == "stats":
            print(mem.stats())
            continue
        if user_input == "forget":
            mem = HebbMem(encoder="hash")
            print("Memory wiped.")
            continue

        # Recall relevant memories
        results = mem.recall(user_input, top_k=5)
        context = "\n".join(f"- {r.content}" for r in results)

        # Get response
        response = ask_ollama(user_input, context)
        print(f"Bot: {response}\n")

        # Store both user message and bot response as memories
        mem.store(f"User said: {user_input}", importance=0.5)
        mem.store(f"Bot replied about: {user_input[:50]}", importance=0.3)

        # Time passes
        mem.step(1)
        turn += 1

        # Auto-save every 10 turns
        if turn % 10 == 0:
            mem.save(MEMORY_FILE)


if __name__ == "__main__":
    main()
