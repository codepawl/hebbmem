"""OpenAI GPT + hebbmem memory.

Shows hebbmem as a persistent memory layer for stateless API calls.
GPT has no memory between calls — hebbmem adds it.

Usage:
    export OPENAI_API_KEY=sk-...
    pip install hebbmem openai
    python assistant.py
"""

from __future__ import annotations

import os

from openai import OpenAI

from hebbmem import HebbMem

MEMORY_FILE = "assistant_memory.hebb"


def main() -> None:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Set OPENAI_API_KEY environment variable first.")
        return

    client = OpenAI(api_key=api_key)

    try:
        mem = HebbMem.load(MEMORY_FILE, encoder="hash")
    except FileNotFoundError:
        mem = HebbMem(encoder="hash")

    print(f"GPT + hebbmem | {mem.stats()['node_count']} memories loaded\n")

    while True:
        user_input = input("You: ").strip()
        if not user_input or user_input == "quit":
            mem.save(MEMORY_FILE)
            print("Saved. Bye!")
            break

        # Recall relevant context
        results = mem.recall(user_input, top_k=5)
        context = (
            "\n".join(f"- {r.content}" for r in results)
            if results
            else "No relevant memories yet."
        )

        # Call GPT with memory context
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You have memory of past conversations:\n"
                        f"{context}\n\nUse these naturally."
                    ),
                },
                {"role": "user", "content": user_input},
            ],
        )
        reply = response.choices[0].message.content or ""
        print(f"GPT: {reply}\n")

        # Store and step
        mem.store(f"User: {user_input}", importance=0.5)
        mem.store(f"GPT: {reply[:200]}", importance=0.3)
        mem.step(1)


if __name__ == "__main__":
    main()
