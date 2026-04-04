"""Agent integration pattern: using hebbmem as an agent's memory backend.

Shows how hebbmem is framework-agnostic — just wrap store/recall/step
in whatever agent class you want. Works with LangChain, CrewAI, or custom.

Run with: python examples/agent_integration.py
"""

from hebbmem import HebbMem


class SimpleAgent:
    """A minimal agent with bio-inspired memory."""

    def __init__(self):
        self.memory = HebbMem(encoder="hash")
        self.turn = 0

    def observe(self, text: str) -> None:
        """Store an observation in memory."""
        self.memory.store(text, importance=0.6)

    def think(self, query: str, context_size: int = 3) -> list[str]:
        """Recall relevant context for a query."""
        results = self.memory.recall(query, top_k=context_size)
        return [r.content for r in results]

    def tick(self) -> None:
        """Advance one time step (call between turns)."""
        self.turn += 1
        self.memory.step(1)


def main():
    agent = SimpleAgent()
    print("=== Agent with hebbmem memory ===\n")

    # Simulate a conversation
    observations = [
        "The user's name is Alice",
        "Alice is working on a Python project",
        "The project uses machine learning for text classification",
        "Alice prefers scikit-learn over TensorFlow",
        "The deadline is next Friday",
        "Alice mentioned she likes coffee",
        "The dataset has 10,000 labeled examples",
        "They discussed switching to PyTorch for the next phase",
        "Alice asked about GPU pricing on cloud providers",
        "The model accuracy is currently 87%",
    ]

    for i, obs in enumerate(observations):
        agent.observe(obs)
        agent.tick()  # time passes between each observation
        print(f"Turn {i+1}: observed '{obs}'")

    print(f"\nMemory stats: {agent.memory.stats()}")
    print()

    # Now the agent thinks about different topics
    queries = [
        "What is Alice working on?",
        "What tools does she prefer?",
        "What about the deadline?",
    ]

    for query in queries:
        context = agent.think(query)
        print(f"Query: '{query}'")
        for c in context:
            print(f"  -> {c}")
        print()

    # After many more turns, old memories fade
    agent.memory.step(50)
    print("After 50 more time steps...")
    context = agent.think("Alice's project")
    print("Query: 'Alice's project'")
    for c in context:
        print(f"  -> {c}")


if __name__ == "__main__":
    main()
