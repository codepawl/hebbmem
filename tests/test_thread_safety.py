"""Tests for thread safety."""

from concurrent.futures import ThreadPoolExecutor, as_completed

from hebbmem import HebbMem


class TestConcurrentStoreRecall:
    def test_no_crash_or_corruption(self):
        mem = HebbMem(encoder="hash")
        errors = []

        def store_task(i: int):
            try:
                mem.store(f"memory number {i}", importance=0.5)
            except Exception as e:
                errors.append(e)

        def recall_task(i: int):
            try:
                mem.recall(f"memory {i}", top_k=3)
            except Exception as e:
                errors.append(e)

        with ThreadPoolExecutor(max_workers=20) as pool:
            futures = []
            for i in range(10):
                futures.append(pool.submit(store_task, i))
                futures.append(pool.submit(recall_task, i))
            for f in as_completed(futures):
                f.result()

        assert errors == []
        assert mem.stats()["node_count"] == 10


class TestConcurrentStep:
    def test_no_crash(self):
        mem = HebbMem(encoder="hash")
        for i in range(5):
            mem.store(f"content {i}")

        errors = []

        def step_task():
            try:
                mem.step(1)
            except Exception as e:
                errors.append(e)

        with ThreadPoolExecutor(max_workers=5) as pool:
            futures = [pool.submit(step_task) for _ in range(10)]
            for f in as_completed(futures):
                f.result()

        assert errors == []


class TestStoreDuringRecall:
    def test_no_crash(self):
        mem = HebbMem(encoder="hash")
        for i in range(5):
            mem.store(f"initial memory {i}")

        errors = []

        def store_task():
            try:
                mem.store("new concurrent memory")
            except Exception as e:
                errors.append(e)

        def recall_task():
            try:
                mem.recall("memory", top_k=5)
            except Exception as e:
                errors.append(e)

        with ThreadPoolExecutor(max_workers=10) as pool:
            futures = []
            for _ in range(5):
                futures.append(pool.submit(store_task))
                futures.append(pool.submit(recall_task))
            for f in as_completed(futures):
                f.result()

        assert errors == []
