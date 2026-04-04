"""HebbMem visualization server.

Usage:
    uv pip install fastapi uvicorn
    uv run python demo/server.py
    # Opens http://localhost:8765
"""

from __future__ import annotations

import sys
import uuid
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.responses import FileResponse

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hebbmem import HebbMem

app = FastAPI(title="hebbmem visualization")
mem = HebbMem(encoder="hash")

DEMO_DIR = Path(__file__).resolve().parent


def get_graph_state() -> dict:
    """Serialize full graph state for D3.js visualization."""
    nodes = []
    for nid, node in mem._graph._nodes.items():
        nodes.append(
            {
                "id": str(nid),
                "content": node.content,
                "activation": round(node.activation, 4),
                "strength": round(node.base_strength, 4),
                "importance": node.importance,
                "access_count": node.access_count,
            }
        )

    edges = []
    seen: set[tuple[str, str]] = set()
    for (src, tgt), edge in mem._graph._edges.items():
        key = tuple(sorted([str(src), str(tgt)]))
        if key not in seen:
            seen.add(key)
            edges.append(
                {
                    "source": str(src),
                    "target": str(tgt),
                    "weight": round(edge.weight, 4),
                    "co_activations": edge.co_activations,
                }
            )

    return {
        "nodes": nodes,
        "edges": edges,
        "stats": {
            "node_count": mem._graph.node_count,
            "edge_count": mem._graph.edge_count,
            "time_step": mem._time_step,
        },
    }


@app.get("/")
def index() -> FileResponse:
    return FileResponse(DEMO_DIR / "visualization.html")


@app.get("/graph")
def graph() -> dict:
    return get_graph_state()


@app.post("/store")
def store(content: str, importance: float = 0.5) -> dict:
    memory_id = mem.store(content, importance=importance)
    return {"memory_id": str(memory_id), "graph": get_graph_state()}


@app.post("/recall")
def recall(query: str, top_k: int = 5) -> dict:
    # Snapshot activations before recall
    pre = {
        str(nid): round(node.activation, 4) for nid, node in mem._graph._nodes.items()
    }

    results = mem.recall(query, top_k=top_k)

    # Snapshot activations after recall (includes spreading)
    post = {
        str(nid): round(node.activation, 4) for nid, node in mem._graph._nodes.items()
    }

    return {
        "results": [
            {
                "id": str(r.memory_id),
                "content": r.content,
                "score": round(r.score, 4),
                "activation": round(r.activation, 4),
                "similarity": round(r.similarity, 4),
            }
            for r in results
        ],
        "pre_activations": pre,
        "post_activations": post,
        "graph": get_graph_state(),
    }


@app.post("/step")
def step(n: int = 1) -> dict:
    mem.step(n)
    return {"graph": get_graph_state()}


@app.post("/reset")
def reset() -> dict:
    global mem
    mem = HebbMem(encoder="hash")
    return {"graph": get_graph_state()}


@app.post("/load-sample")
def load_sample() -> dict:
    global mem
    mem = HebbMem(encoder="hash")
    from demo.sample_data import load_sample_memories

    load_sample_memories(mem)
    return {"graph": get_graph_state()}


@app.post("/forget")
def forget(memory_id: str) -> dict:
    mem.forget(uuid.UUID(memory_id))
    return {"graph": get_graph_state()}


if __name__ == "__main__":
    print("hebbmem visualization: http://localhost:8765")
    uvicorn.run(app, host="0.0.0.0", port=8765)
