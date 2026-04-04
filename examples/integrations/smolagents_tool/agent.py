"""smolagents + hebbmem — lightweight agent with brain-like memory.

smolagents is HuggingFace's minimal agent framework.
hebbmem adds persistent, associative memory as a tool.

Usage:
    pip install hebbmem smolagents
    python agent.py
"""

from __future__ import annotations

from smolagents import CodeAgent, HfApiModel, tool

from hebbmem import HebbMem

mem = HebbMem(encoder="hash")


@tool
def store_memory(content: str, importance: float = 0.5) -> str:
    """Store information in agent memory for later recall.

    Args:
        content: Text to remember.
        importance: How important (0.0 to 1.0). Higher resists forgetting.
    """
    mem.store(content, importance=importance)
    return f"Stored memory: {content[:80]}"


@tool
def recall_memory(query: str) -> str:
    """Recall relevant memories. Finds similar AND associated memories.

    Args:
        query: What to search for in memory.
    """
    results = mem.recall(query, top_k=5)
    mem.step(1)
    if not results:
        return "No relevant memories."
    return "\n".join(f"[{r.score:.2f}] {r.content}" for r in results)


@tool
def memory_stats() -> str:
    """Check current memory statistics."""
    s = mem.stats()
    return (
        f"Memories: {s['node_count']}, "
        f"Connections: {s['edge_count']}, "
        f"Steps: {s['time_step']}"
    )


def main() -> None:
    model = HfApiModel()

    agent = CodeAgent(
        tools=[store_memory, recall_memory, memory_stats],
        model=model,
    )

    print("smolagents + hebbmem | Type 'quit' to exit\n")
    while True:
        user_input = input("You: ").strip()
        if user_input == "quit":
            break
        result = agent.run(user_input)
        print(f"Agent: {result}\n")


if __name__ == "__main__":
    main()
