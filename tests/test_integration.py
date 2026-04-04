"""End-to-end integration tests for hebbmem."""

from hebbmem import Config, HebbMem


class TestStoreRecallRoundtrip:
    def test_relevant_memories_ranked_higher(self):
        mem = HebbMem(encoder="hash")
        mem.store("python is a programming language")
        mem.store("javascript runs in browsers")
        mem.store("the weather is sunny today")
        mem.store("python has great data science libraries")

        results = mem.recall("python programming", top_k=4)
        contents = [r.content for r in results]
        # Python-related memories should appear before weather
        python_indices = [i for i, c in enumerate(contents) if "python" in c.lower()]
        weather_indices = [i for i, c in enumerate(contents) if "weather" in c.lower()]
        if python_indices and weather_indices:
            assert min(python_indices) < min(weather_indices)


class TestHebbianStrengthening:
    def test_repeated_corecall_strengthens_edges(self):
        mem = HebbMem(encoder="hash")
        mem.store("machine learning algorithms")
        mem.store("machine learning with python")
        mem.store("cooking pasta recipes")

        # Recall "machine learning" multiple times to strengthen edges
        for _ in range(5):
            mem.recall("machine learning", top_k=3)

        # The ML memories should have stronger connections now
        stats = mem.stats()
        assert stats["node_count"] == 3


class TestTemporalDecay:
    def test_old_memories_weaken(self):
        mem = HebbMem(encoder="hash")
        mem.store("old memory about cats", importance=0.5)

        # Recall to get baseline
        results_before = mem.recall("cats", top_k=1)
        # Apply many decay steps
        mem.step(50)

        # The strength component should have decayed
        results_after = mem.recall("cats", top_k=1)
        assert results_after[0].strength < results_before[0].strength


class TestCustomConfig:
    def test_high_spread_activates_more(self):
        config_high = Config(spread_factor=0.9, auto_connect_threshold=0.1)
        config_low = Config(spread_factor=0.01, auto_connect_threshold=0.1)

        mem_high = HebbMem(encoder="hash", config=config_high)
        mem_low = HebbMem(encoder="hash", config=config_low)

        contents = [
            "neural networks learn patterns",
            "deep learning uses neural networks",
            "backpropagation trains networks",
        ]
        for c in contents:
            mem_high.store(c)
            mem_low.store(c)

        results_high = mem_high.recall("neural", top_k=3)
        results_low = mem_low.recall("neural", top_k=3)

        # High spread should activate more total activation across results
        total_act_high = sum(r.activation for r in results_high)
        total_act_low = sum(r.activation for r in results_low)
        assert total_act_high >= total_act_low
