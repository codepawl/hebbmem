"""Custom configuration: tuning decay, Hebbian learning, and scoring.

Demonstrates side-by-side comparison of different Config settings
to show how parameters affect memory behavior.

Run with: python examples/custom_config.py
"""

from hebbmem import Config, HebbMem


def compare_decay_rates():
    """Fast decay vs slow decay."""
    print("=== Decay Rate Comparison ===\n")

    fast = HebbMem(encoder="hash", config=Config(activation_decay=0.5))
    slow = HebbMem(encoder="hash", config=Config(activation_decay=0.99))

    content = "important meeting notes about the project roadmap"
    fast.store(content)
    slow.store(content)

    for steps in [0, 5, 10, 20]:
        if steps > 0:
            fast.step(steps)
            slow.step(steps)

        r_fast = fast.recall("meeting", top_k=1)
        r_slow = slow.recall("meeting", top_k=1)
        s_fast = r_fast[0].strength if r_fast else 0
        s_slow = r_slow[0].strength if r_slow else 0
        print(f"  After {steps:2d} steps: fast={s_fast:.4f} slow={s_slow:.4f}")


def compare_hebbian_lr():
    """High vs low Hebbian learning rate."""
    print("\n=== Hebbian Learning Rate Comparison ===\n")

    high = HebbMem(encoder="hash", config=Config(hebbian_lr=0.3))
    low = HebbMem(encoder="hash", config=Config(hebbian_lr=0.01))

    for mem in [high, low]:
        mem.store("machine learning algorithms")
        mem.store("machine learning with Python")

    # Recall multiple times to trigger Hebbian reinforcement
    for _ in range(5):
        high.recall("machine learning")
        low.recall("machine learning")

    h_edges = high._graph._edges
    l_edges = low._graph._edges

    if h_edges and l_edges:
        h_max = max(e.weight for e in h_edges.values())
        l_max = max(e.weight for e in l_edges.values())
        print(f"  High LR (0.3): max edge weight = {h_max:.4f}")
        print(f"  Low  LR (0.01): max edge weight = {l_max:.4f}")


def compare_scoring_weights():
    """Scoring that favors similarity vs importance."""
    print("\n=== Scoring Weight Comparison ===\n")

    sim_focused = HebbMem(
        encoder="hash",
        config=Config(
            scoring_weights={
                "activation": 0.1,
                "similarity": 0.8,
                "strength": 0.05,
                "importance": 0.05,
            }
        ),
    )
    imp_focused = HebbMem(
        encoder="hash",
        config=Config(
            scoring_weights={
                "activation": 0.1,
                "similarity": 0.1,
                "strength": 0.1,
                "importance": 0.7,
            }
        ),
    )

    for mem in [sim_focused, imp_focused]:
        mem.store("Python data science tools", importance=0.3)
        mem.store("cooking pasta recipes", importance=0.9)

    query = "Python programming"
    print(f"  Query: '{query}'")
    print()

    r_sim = sim_focused.recall(query, top_k=2)
    print("  Similarity-focused ranking:")
    for r in r_sim:
        sim, imp = r.similarity, r.importance
        print(f"    [{r.score:.3f}] {r.content} (sim={sim:.2f}, imp={imp:.1f})")

    r_imp = imp_focused.recall(query, top_k=2)
    print("  Importance-focused ranking:")
    for r in r_imp:
        sim, imp = r.similarity, r.importance
        print(f"    [{r.score:.3f}] {r.content} (sim={sim:.2f}, imp={imp:.1f})")


def main():
    compare_decay_rates()
    compare_hebbian_lr()
    compare_scoring_weights()


if __name__ == "__main__":
    main()
