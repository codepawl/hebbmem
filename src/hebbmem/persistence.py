"""SQLite-based persistence for hebbmem graphs."""

from __future__ import annotations

import json
import logging
import sqlite3
import uuid
import warnings
from dataclasses import asdict
from pathlib import Path

import numpy as np

from hebbmem.exceptions import PersistenceError
from hebbmem.graph import MemoryGraph
from hebbmem.node import MemoryNode
from hebbmem.types import Config, Edge

logger = logging.getLogger("hebbmem")


def save_to_sqlite(
    path: str | Path,
    graph: MemoryGraph,
    config: Config,
    time_step: int,
) -> None:
    """Save full graph state to a single SQLite file."""
    path = Path(path)
    if path.exists():
        path.unlink()

    try:
        conn = sqlite3.connect(str(path))
    except sqlite3.Error as e:
        raise PersistenceError(f"Cannot create file {path}: {e}") from e

    try:
        cur = conn.cursor()

        cur.execute("CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT)")
        cur.execute(
            "CREATE TABLE nodes ("
            "  id TEXT PRIMARY KEY,"
            "  content TEXT,"
            "  embedding BLOB,"
            "  embedding_dim INTEGER,"
            "  activation REAL,"
            "  base_strength REAL,"
            "  importance REAL,"
            "  decay_rate REAL,"
            "  created_at REAL,"
            "  last_accessed REAL,"
            "  access_count INTEGER,"
            "  metadata TEXT"
            ")"
        )
        cur.execute(
            "CREATE TABLE edges ("
            "  source_id TEXT,"
            "  target_id TEXT,"
            "  weight REAL,"
            "  co_activations INTEGER,"
            "  PRIMARY KEY (source_id, target_id)"
            ")"
        )

        from hebbmem import __version__

        cur.execute(
            "INSERT INTO meta VALUES (?, ?)",
            ("hebbmem_version", __version__),
        )
        cur.execute(
            "INSERT INTO meta VALUES (?, ?)",
            ("config", json.dumps(asdict(config))),
        )
        cur.execute(
            "INSERT INTO meta VALUES (?, ?)",
            ("time_step", str(time_step)),
        )

        for node in graph._nodes.values():
            cur.execute(
                "INSERT INTO nodes VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    node.memory_id.hex,
                    node.content,
                    node.embedding.astype(np.float32).tobytes(),
                    node.embedding.shape[0],
                    node.activation,
                    node.base_strength,
                    node.importance,
                    node.decay_rate,
                    node.created_at,
                    node.last_accessed,
                    node.access_count,
                    json.dumps(node.metadata),
                ),
            )

        seen: set[tuple[str, str]] = set()
        for (src, dst), edge in graph._edges.items():
            key = (min(src.hex, dst.hex), max(src.hex, dst.hex))
            if key not in seen:
                seen.add(key)
                cur.execute(
                    "INSERT INTO edges VALUES (?, ?, ?, ?)",
                    (src.hex, dst.hex, edge.weight, edge.co_activations),
                )

        conn.commit()
        logger.info(
            "Saved %d nodes, %d edges to %s",
            graph.node_count,
            graph.edge_count,
            path,
        )
    except sqlite3.Error as e:
        raise PersistenceError(f"Failed to save to {path}: {e}") from e
    finally:
        conn.close()


def load_from_sqlite(path: str | Path) -> tuple[MemoryGraph, Config, int]:
    """Load graph state from a SQLite file.

    Returns (graph, config, time_step).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No hebbmem file at: {path}")

    try:
        conn = sqlite3.connect(str(path))
    except sqlite3.Error as e:
        raise PersistenceError(f"Cannot open file {path}: {e}") from e

    try:
        cur = conn.cursor()

        try:
            meta = dict(cur.execute("SELECT key, value FROM meta").fetchall())
        except sqlite3.Error as e:
            raise PersistenceError(
                f"File is not a valid hebbmem database: {path}"
            ) from e

        from hebbmem import __version__

        saved_version = meta.get("hebbmem_version", "unknown")
        if saved_version != __version__:
            warnings.warn(
                f"File was saved with hebbmem {saved_version}, "
                f"current version is {__version__}. Loading anyway.",
                stacklevel=2,
            )

        config = Config(**json.loads(meta["config"]))
        time_step = int(meta["time_step"])

        graph = MemoryGraph(config)

        rows = cur.execute(
            "SELECT id, content, embedding, embedding_dim, activation, "
            "base_strength, importance, decay_rate, created_at, last_accessed, "
            "access_count, metadata FROM nodes"
        ).fetchall()

        for row in rows:
            (
                id_hex,
                content,
                emb_blob,
                _emb_dim,
                activation,
                base_strength,
                importance,
                decay_rate,
                created_at,
                last_accessed,
                access_count,
                metadata_json,
            ) = row
            embedding = np.frombuffer(emb_blob, dtype=np.float32).copy()
            node = MemoryNode(
                content=content,
                embedding=embedding,
                importance=importance,
                metadata=json.loads(metadata_json),
                memory_id=uuid.UUID(hex=id_hex),
                activation=activation,
                base_strength=base_strength,
                decay_rate=decay_rate,
                created_at=created_at,
                last_accessed=last_accessed,
                access_count=access_count,
            )
            graph._nodes[node.memory_id] = node

        edge_rows = cur.execute(
            "SELECT source_id, target_id, weight, co_activations FROM edges"
        ).fetchall()

        for src_hex, dst_hex, weight, co_activations in edge_rows:
            src_id = uuid.UUID(hex=src_hex)
            dst_id = uuid.UUID(hex=dst_hex)
            edge_ab = Edge(weight=weight, co_activations=co_activations)
            edge_ba = Edge(weight=weight, co_activations=co_activations)
            graph._edges[(src_id, dst_id)] = edge_ab
            graph._edges[(dst_id, src_id)] = edge_ba

        graph._cache_dirty = True
        logger.info(
            "Loaded %d nodes, %d edges from %s",
            graph.node_count,
            graph.edge_count,
            path,
        )
        return graph, config, time_step
    except sqlite3.Error as e:
        raise PersistenceError(f"Failed to load from {path}: {e}") from e
    finally:
        conn.close()
