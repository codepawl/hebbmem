"""Basic hebbmem usage: store, decay, recall, forget.

Demonstrates the core API without any framework integration.
Run with: python examples/basic_usage.py
"""

from hebbmem import HebbMem


def main():
    mem = HebbMem(encoder="hash")
    print("=== hebbmem basic usage ===\n")

    # Store memories with different importance levels
    id1 = mem.store("Python is great for data science", importance=0.9)
    id2 = mem.store("JavaScript powers the web", importance=0.7)
    id3 = mem.store("The weather is sunny today", importance=0.2)
    id4 = mem.store("Machine learning uses Python heavily", importance=0.8)
    print(f"Stored 4 memories. Stats: {mem.stats()}\n")

    # Recall before any decay
    print("--- Recall 'Python programming' (fresh) ---")
    for r in mem.recall("Python programming", top_k=3):
        print(f"  [{r.score:.3f}] {r.content}")
    print()

    # Simulate time passing — memories decay
    mem.step(10)
    print(f"Advanced 10 time steps. Stats: {mem.stats()}\n")

    # Recall after decay — scores are lower, but ranking holds
    print("--- Recall 'Python programming' (after decay) ---")
    for r in mem.recall("Python programming", top_k=3):
        print(f"  [{r.score:.3f}] {r.content} (strength={r.strength:.3f})")
    print()

    # Forget a specific memory
    mem.forget(id3)
    print(f"Forgot 'weather' memory. Stats: {mem.stats()}\n")

    # Final recall
    print("--- Recall 'data science' ---")
    for r in mem.recall("data science", top_k=3):
        print(f"  [{r.score:.3f}] {r.content}")


if __name__ == "__main__":
    main()
