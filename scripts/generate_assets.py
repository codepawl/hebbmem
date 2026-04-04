"""Generate all visual assets for hebbmem.

Usage:
    uv pip install hebbmem[assets]
    uv run python scripts/generate_assets.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hebbmem import HebbMem

matplotlib.use("Agg")

ASSETS = Path(__file__).resolve().parent.parent / "assets"

C = {
    "bg": "#0a0a1a",
    "bg_card": "#12122a",
    "text": "#e0e0f0",
    "text_muted": "#6a6a8a",
    "gold": "#f0b429",
    "blue": "#4a9eff",
    "orange": "#ff6b35",
    "green": "#2dd4a0",
    "red": "#ff4757",
    "edge": "#2a2a4a",
    "edge_active": "#f0b429",
    "node_inactive": "#1a1a3a",
}

plt.rcParams.update(
    {
        "font.family": "monospace",
        "text.color": C["text"],
        "axes.labelcolor": C["text"],
        "xtick.color": C["text_muted"],
        "ytick.color": C["text_muted"],
    }
)


def _save(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  saved {path.relative_to(ASSETS.parent)}")


def _draw_graph_on_ax(ax: plt.Axes, n_nodes: int = 8, seed: int = 42) -> None:
    """Draw a small memory graph on an axes."""
    rng = np.random.RandomState(seed)
    graph = nx.watts_strogatz_graph(n_nodes, 3, 0.4, seed=seed)
    pos = nx.spring_layout(graph, seed=seed, k=1.5)

    activations = rng.uniform(0.1, 1.0, n_nodes)
    node_colors = [
        plt.cm.YlOrBr(a * 0.8 + 0.1)
        for a in activations  # type: ignore[attr-defined]
    ]
    node_sizes = [200 + a * 400 for a in activations]

    nx.draw_networkx_edges(
        graph, pos, ax=ax, edge_color=C["edge"], width=1.5, alpha=0.5
    )
    nx.draw_networkx_nodes(
        graph,
        pos,
        ax=ax,
        node_color=node_colors,
        node_size=node_sizes,
        edgecolors=C["gold"],
        linewidths=[a * 2 for a in activations],
    )
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.axis("off")


# --- Asset generators ---


def gen_hero_banner() -> None:
    fig = plt.figure(figsize=(12, 4), facecolor=C["bg"])

    # Text side
    ax_text = fig.add_axes([0.02, 0.05, 0.5, 0.9])
    ax_text.set_facecolor(C["bg"])
    ax_text.text(
        0.05,
        0.65,
        "hebbmem",
        fontsize=48,
        fontweight="bold",
        color=C["blue"],
        transform=ax_text.transAxes,
    )
    ax_text.text(
        0.05,
        0.35,
        "memories that fire together\nwire together",
        fontsize=18,
        color=C["text_muted"],
        transform=ax_text.transAxes,
        linespacing=1.5,
    )
    ax_text.text(
        0.05,
        0.10,
        "pip install hebbmem",
        fontsize=14,
        color=C["gold"],
        transform=ax_text.transAxes,
        family="monospace",
    )
    ax_text.axis("off")

    # Graph side
    ax_graph = fig.add_axes([0.55, 0.05, 0.42, 0.9])
    ax_graph.set_facecolor(C["bg"])
    _draw_graph_on_ax(ax_graph, n_nodes=10, seed=42)

    _save(fig, ASSETS / "readme" / "hero_banner.png")


def gen_how_it_works() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12, 5), facecolor=C["bg"])

    for ax in axes:
        ax.set_facecolor(C["bg"])
        ax.axis("off")

    # Panel 1: Decay
    ax = axes[0]
    ax.text(
        0.5,
        0.92,
        "Decay",
        fontsize=22,
        fontweight="bold",
        color=C["text"],
        ha="center",
        transform=ax.transAxes,
    )
    ax.text(
        0.5,
        0.82,
        "old memories fade",
        fontsize=13,
        color=C["text_muted"],
        ha="center",
        transform=ax.transAxes,
    )
    for i in range(6):
        alpha = 1.0 - i * 0.17
        color = plt.cm.YlOrBr(alpha * 0.8 + 0.1)  # type: ignore[attr-defined]
        circle = plt.Circle(
            (0.12 + i * 0.15, 0.45),
            0.06,
            color=color,
            alpha=max(alpha, 0.15),
            transform=ax.transAxes,
        )
        ax.add_patch(circle)
    ax.annotate(
        "",
        xy=(0.85, 0.28),
        xytext=(0.12, 0.28),
        xycoords="axes fraction",
        arrowprops=dict(arrowstyle="->", color=C["text_muted"], lw=1.5),
    )
    ax.text(
        0.5,
        0.18,
        "time →",
        fontsize=11,
        color=C["text_muted"],
        ha="center",
        transform=ax.transAxes,
    )

    # Panel 2: Hebbian
    ax = axes[1]
    ax.text(
        0.5,
        0.92,
        "Hebbian",
        fontsize=22,
        fontweight="bold",
        color=C["text"],
        ha="center",
        transform=ax.transAxes,
    )
    ax.text(
        0.5,
        0.82,
        "co-recalled memories bond",
        fontsize=13,
        color=C["text_muted"],
        ha="center",
        transform=ax.transAxes,
    )
    # Before
    ax.add_patch(
        plt.Circle((0.25, 0.55), 0.06, color=C["blue"], transform=ax.transAxes)
    )
    ax.add_patch(
        plt.Circle((0.45, 0.55), 0.06, color=C["blue"], transform=ax.transAxes)
    )
    ax.plot([0.31, 0.39], [0.55, 0.55], color=C["edge"], lw=1, transform=ax.transAxes)
    ax.text(
        0.35,
        0.40,
        "weak",
        fontsize=10,
        color=C["text_muted"],
        ha="center",
        transform=ax.transAxes,
    )
    # Arrow
    ax.annotate(
        "",
        xy=(0.62, 0.55),
        xytext=(0.52, 0.55),
        xycoords="axes fraction",
        arrowprops=dict(arrowstyle="->", color=C["gold"], lw=2),
    )
    # After
    ax.add_patch(
        plt.Circle(
            (0.70, 0.55), 0.07, color=C["gold"], alpha=0.9, transform=ax.transAxes
        )
    )
    ax.add_patch(
        plt.Circle(
            (0.90, 0.55), 0.07, color=C["gold"], alpha=0.9, transform=ax.transAxes
        )
    )
    ax.plot([0.77, 0.83], [0.55, 0.55], color=C["gold"], lw=4, transform=ax.transAxes)
    ax.text(
        0.80,
        0.40,
        "strong",
        fontsize=10,
        color=C["gold"],
        ha="center",
        transform=ax.transAxes,
    )
    ax.text(
        0.5,
        0.18,
        "recall together x5",
        fontsize=11,
        color=C["text_muted"],
        ha="center",
        transform=ax.transAxes,
    )

    # Panel 3: Spreading
    ax = axes[2]
    ax.text(
        0.5,
        0.92,
        "Spreading",
        fontsize=22,
        fontweight="bold",
        color=C["text"],
        ha="center",
        transform=ax.transAxes,
    )
    ax.text(
        0.5,
        0.82,
        "recall activates neighbors",
        fontsize=13,
        color=C["text_muted"],
        ha="center",
        transform=ax.transAxes,
    )
    center = (0.5, 0.48)
    ax.add_patch(
        plt.Circle(center, 0.07, color=C["gold"], transform=ax.transAxes, zorder=5)
    )
    angles = np.linspace(0, 2 * np.pi, 5, endpoint=False)
    for _i, a in enumerate(angles):
        r1 = 0.18
        pos1 = (center[0] + r1 * np.cos(a), center[1] + r1 * np.sin(a))
        ax.plot(
            [center[0], pos1[0]],
            [center[1], pos1[1]],
            color=C["orange"],
            lw=2,
            transform=ax.transAxes,
            alpha=0.7,
        )
        ax.add_patch(
            plt.Circle(
                pos1,
                0.05,
                color=C["orange"],
                alpha=0.7,
                transform=ax.transAxes,
                zorder=4,
            )
        )
        r2 = 0.32
        pos2 = (center[0] + r2 * np.cos(a + 0.3), center[1] + r2 * np.sin(a + 0.3))
        ax.plot(
            [pos1[0], pos2[0]],
            [pos1[1], pos2[1]],
            color=C["orange"],
            lw=1,
            transform=ax.transAxes,
            alpha=0.3,
        )
        ax.add_patch(
            plt.Circle(
                pos2,
                0.035,
                color=C["orange"],
                alpha=0.3,
                transform=ax.transAxes,
                zorder=3,
            )
        )
    ax.text(
        0.5,
        0.18,
        "energy spreads per hop",
        fontsize=11,
        color=C["text_muted"],
        ha="center",
        transform=ax.transAxes,
    )

    fig.subplots_adjust(wspace=0.05)
    _save(fig, ASSETS / "readme" / "how_it_works.png")


def gen_benchmark_results() -> None:
    scenarios = ["Associative", "Contradiction", "Noise", "Temporal"]
    # Best metric per scenario from v0.3.0 benchmarks
    hebbmem_vals = [0.56, 0.50, 0.60, 0.36]
    baseline_vals = [0.44, 0.00, 0.56, 0.36]
    metrics = ["Precision@5", "Precision@1", "Precision@5", "Precision@5"]

    fig, ax = plt.subplots(figsize=(10, 6), facecolor=C["bg"])
    ax.set_facecolor(C["bg"])

    y = np.arange(len(scenarios))
    h = 0.35

    ax.barh(y + h / 2, hebbmem_vals, h, label="hebbmem", color=C["green"], alpha=0.9)
    ax.barh(y - h / 2, baseline_vals, h, label="Baseline", color=C["red"], alpha=0.5)

    # Delta labels
    for i, (hv, bv) in enumerate(zip(hebbmem_vals, baseline_vals, strict=True)):
        if bv > 0:
            delta = (hv - bv) / bv * 100
            label = f"+{delta:.0f}%" if delta > 0 else "tied"
        elif hv > 0:
            label = "unique"
        else:
            label = ""
        ax.text(
            hv + 0.02,
            y[i] + h / 2,
            label,
            va="center",
            fontsize=11,
            color=C["gold"],
            fontweight="bold",
        )

    ax.set_yticks(y)
    ax.set_yticklabels(
        [f"{s}\n({m})" for s, m in zip(scenarios, metrics, strict=True)], fontsize=12
    )
    ax.set_xlabel("Score", fontsize=13)
    ax.set_xlim(0, 0.85)
    ax.set_title(
        "hebbmem vs Flat Vector Search", fontsize=20, fontweight="bold", pad=15
    )
    ax.legend(
        loc="lower right", fontsize=12, facecolor=C["bg_card"], edgecolor=C["edge"]
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_color(C["edge"])
    ax.spines["left"].set_color(C["edge"])
    ax.grid(axis="x", color=C["edge"], alpha=0.3)

    _save(fig, ASSETS / "readme" / "benchmark_results.png")


def gen_recall_pipeline() -> None:
    fig, ax = plt.subplots(figsize=(10, 3), facecolor=C["bg"])
    ax.set_facecolor(C["bg"])
    ax.axis("off")

    steps = [
        ("Query", "input text", C["text_muted"]),
        ("Encode", "embed query", C["blue"]),
        ("Find Seeds", "cosine sim", C["blue"]),
        ("Spread", "BFS graph", C["gold"]),
        ("Hebbian", "strengthen", C["orange"]),
        ("Rank", "score top-k", C["green"]),
    ]

    x_start = 0.03
    box_w = 0.13
    gap = 0.025

    for i, (title, subtitle, color) in enumerate(steps):
        x = x_start + i * (box_w + gap)
        rect = mpatches.FancyBboxPatch(
            (x, 0.35),
            box_w,
            0.4,
            boxstyle="round,pad=0.02",
            facecolor=color if i == 3 else C["bg_card"],
            edgecolor=color,
            linewidth=2,
            alpha=0.9 if i == 3 else 0.7,
            transform=ax.transAxes,
        )
        ax.add_patch(rect)
        ax.text(
            x + box_w / 2,
            0.62,
            title,
            ha="center",
            va="center",
            fontsize=12,
            fontweight="bold",
            color=C["bg"] if i == 3 else C["text"],
            transform=ax.transAxes,
        )
        ax.text(
            x + box_w / 2,
            0.45,
            subtitle,
            ha="center",
            va="center",
            fontsize=9,
            color=C["bg"] if i == 3 else C["text_muted"],
            transform=ax.transAxes,
        )

        if i < len(steps) - 1:
            ax.annotate(
                "",
                xy=(x + box_w + gap * 0.8, 0.55),
                xytext=(x + box_w + gap * 0.2, 0.55),
                xycoords="axes fraction",
                arrowprops=dict(arrowstyle="->", color=C["text_muted"], lw=1.5),
            )

    ax.text(
        0.5,
        0.92,
        "Recall Pipeline",
        fontsize=16,
        fontweight="bold",
        color=C["text"],
        ha="center",
        transform=ax.transAxes,
    )

    _save(fig, ASSETS / "readme" / "recall_pipeline.png")


def gen_graph_example() -> None:
    mem = HebbMem(encoder="hash")
    memories = [
        ("Project Atlas deadline March 15", 0.9),
        ("Atlas uses collaborative filtering", 0.7),
        ("Team meeting every Tuesday", 0.6),
        ("User loves Vietnamese coffee", 0.7),
        ("User is studying Rust language", 0.6),
        ("Rust async runtime article saved", 0.4),
        ("Rewrite Python CLI tool in Rust", 0.7),
        ("Weather is nice today", 0.1),
        ("User said good morning", 0.05),
        ("Budget review end of quarter", 0.8),
    ]
    for content, imp in memories:
        mem.store(content, importance=imp)

    for _ in range(3):
        mem.recall("Atlas project deadline")
    for _ in range(3):
        mem.recall("Rust programming language")
    mem.step(5)

    # Build networkx graph
    graph = nx.Graph()
    node_data = {}
    for nid, node in mem._graph._nodes.items():
        sid = str(nid)[:8]
        graph.add_node(sid)
        node_data[sid] = {
            "label": node.content[:28],
            "activation": node.activation,
            "strength": node.base_strength,
            "importance": node.importance,
        }

    seen = set()
    for (src, tgt), edge in mem._graph._edges.items():
        key = tuple(sorted([str(src)[:8], str(tgt)[:8]]))
        if key not in seen and key[0] != key[1]:
            seen.add(key)
            graph.add_edge(key[0], key[1], weight=edge.weight)

    fig, ax = plt.subplots(figsize=(8, 8), facecolor=C["bg"])
    ax.set_facecolor(C["bg"])
    ax.axis("off")

    pos = nx.spring_layout(graph, seed=42, k=2.0)

    # Edges
    for u, v, d in graph.edges(data=True):
        w = d.get("weight", 0.1)
        color = C["edge_active"] if w > 0.3 else C["edge"]
        ax.plot(
            [pos[u][0], pos[v][0]],
            [pos[u][1], pos[v][1]],
            color=color,
            lw=1 + w * 5,
            alpha=0.3 + w * 0.5,
            zorder=1,
        )

    # Nodes
    for node_id in graph.nodes():
        d = node_data[node_id]
        act = d["activation"]
        strength = d["strength"]
        imp = d["importance"]
        color = plt.cm.YlOrBr(max(act * 2, 0.15))  # type: ignore[attr-defined]
        size = 8 + strength * 12 + imp * 5
        ax.scatter(
            pos[node_id][0],
            pos[node_id][1],
            s=size * 40,
            c=[color],
            edgecolors=C["gold"] if act > 0.1 else C["edge"],
            linewidths=1 + imp * 2,
            zorder=3,
        )
        ax.text(
            pos[node_id][0],
            pos[node_id][1] - 0.12,
            d["label"],
            fontsize=7,
            ha="center",
            color=C["text_muted"],
            zorder=4,
        )

    ax.set_title(
        "Memory Graph Example", fontsize=18, fontweight="bold", color=C["text"], pad=15
    )
    _save(fig, ASSETS / "readme" / "graph_example.png")


def gen_og_image() -> None:
    fig = plt.figure(figsize=(12, 6.3), facecolor=C["bg"])

    ax_text = fig.add_axes([0.03, 0.05, 0.55, 0.9])
    ax_text.set_facecolor(C["bg"])
    ax_text.text(
        0.05,
        0.7,
        "hebbmem",
        fontsize=44,
        fontweight="bold",
        color=C["blue"],
        transform=ax_text.transAxes,
    )
    ax_text.text(
        0.05,
        0.50,
        "Hebbian memory\nfor AI agents",
        fontsize=22,
        color=C["text"],
        transform=ax_text.transAxes,
        linespacing=1.4,
    )
    ax_text.text(
        0.05,
        0.25,
        "pip install hebbmem",
        fontsize=16,
        color=C["gold"],
        transform=ax_text.transAxes,
        family="monospace",
    )
    ax_text.text(
        0.05,
        0.08,
        "github.com/codepawl/hebbmem",
        fontsize=12,
        color=C["text_muted"],
        transform=ax_text.transAxes,
    )
    ax_text.axis("off")

    ax_graph = fig.add_axes([0.58, 0.1, 0.38, 0.8])
    ax_graph.set_facecolor(C["bg"])
    _draw_graph_on_ax(ax_graph, n_nodes=10, seed=77)

    _save(fig, ASSETS / "social" / "og_image.png")


def _gen_x_image(filename: str, title: str, subtitle: str, draw_fn=None) -> None:
    fig = plt.figure(figsize=(16, 9), facecolor=C["bg"])
    ax = fig.add_axes([0.05, 0.05, 0.9, 0.9])
    ax.set_facecolor(C["bg"])
    ax.axis("off")

    ax.text(
        0.5,
        0.75,
        title,
        fontsize=36,
        fontweight="bold",
        color=C["text"],
        ha="center",
        transform=ax.transAxes,
    )
    ax.text(
        0.5,
        0.60,
        subtitle,
        fontsize=18,
        color=C["text_muted"],
        ha="center",
        transform=ax.transAxes,
    )

    if draw_fn:
        draw_fn(ax)

    ax.text(
        0.95,
        0.05,
        "hebbmem",
        fontsize=14,
        color=C["blue"],
        ha="right",
        transform=ax.transAxes,
        alpha=0.6,
    )
    _save(fig, ASSETS / "social" / filename)


def gen_x_thread_1() -> None:
    def draw(ax: plt.Axes) -> None:
        icons = [
            ("brain", 0.25),
            ("->", 0.40),
            ("code", 0.55),
            ("->", 0.70),
            ("agent", 0.85),
        ]
        for text, x in icons:
            color = C["gold"] if text not in ("->",) else C["text_muted"]
            ax.text(
                x,
                0.38,
                text,
                fontsize=22,
                ha="center",
                color=color,
                transform=ax.transAxes,
            )

    _gen_x_image(
        "x_thread_1.png",
        "What if AI memory worked\nlike your brain?",
        "pip install hebbmem",
        draw,
    )


def gen_x_thread_2() -> None:
    def draw(ax: plt.Axes) -> None:
        center = (0.5, 0.32)
        ax.add_patch(
            plt.Circle(center, 0.04, color=C["gold"], transform=ax.transAxes, zorder=5)
        )
        ax.text(
            center[0],
            center[1] - 0.07,
            "seed",
            fontsize=10,
            color=C["gold"],
            ha="center",
            transform=ax.transAxes,
        )
        angles = np.linspace(0, 2 * np.pi, 6, endpoint=False)
        for a in angles:
            p1 = (center[0] + 0.12 * np.cos(a), center[1] + 0.08 * np.sin(a))
            ax.plot(
                [center[0], p1[0]],
                [center[1], p1[1]],
                color=C["orange"],
                lw=2.5,
                transform=ax.transAxes,
                alpha=0.7,
            )
            ax.add_patch(
                plt.Circle(
                    p1,
                    0.03,
                    color=C["orange"],
                    alpha=0.7,
                    transform=ax.transAxes,
                    zorder=4,
                )
            )
            p2 = (
                center[0] + 0.22 * np.cos(a + 0.4),
                center[1] + 0.15 * np.sin(a + 0.4),
            )
            ax.plot(
                [p1[0], p2[0]],
                [p1[1], p2[1]],
                color=C["orange"],
                lw=1,
                transform=ax.transAxes,
                alpha=0.3,
            )
            ax.add_patch(
                plt.Circle(
                    p2,
                    0.02,
                    color=C["orange"],
                    alpha=0.3,
                    transform=ax.transAxes,
                    zorder=3,
                )
            )

    _gen_x_image(
        "x_thread_2.png",
        "Spreading Activation",
        'Query "Project Atlas" → activates related memories',
        draw,
    )


def gen_x_thread_3() -> None:
    """Benchmark comparison for X thread."""
    fig = plt.figure(figsize=(16, 9), facecolor=C["bg"])
    ax = fig.add_axes([0.12, 0.15, 0.78, 0.65])
    ax.set_facecolor(C["bg"])

    scenarios = ["Associative", "Contradiction", "Noise", "Temporal"]
    h_vals = [0.56, 0.50, 0.60, 0.36]
    b_vals = [0.44, 0.00, 0.56, 0.36]

    x = np.arange(len(scenarios))
    w = 0.35
    ax.bar(x - w / 2, h_vals, w, label="hebbmem", color=C["green"], alpha=0.9)
    ax.bar(x + w / 2, b_vals, w, label="Baseline", color=C["red"], alpha=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(scenarios, fontsize=14)
    ax.set_ylabel("Precision", fontsize=14)
    ax.legend(fontsize=13, facecolor=C["bg_card"], edgecolor=C["edge"])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_color(C["edge"])
    ax.spines["left"].set_color(C["edge"])

    fig.text(
        0.5,
        0.88,
        "hebbmem vs Flat Vector Search",
        fontsize=28,
        fontweight="bold",
        color=C["text"],
        ha="center",
    )
    fig.text(
        0.5,
        0.06,
        "pip install hebbmem",
        fontsize=16,
        color=C["gold"],
        ha="center",
        family="monospace",
    )
    fig.text(0.95, 0.03, "hebbmem", fontsize=12, color=C["blue"], ha="right", alpha=0.6)

    _save(fig, ASSETS / "social" / "x_thread_3.png")


def gen_devto_cover() -> None:
    fig = plt.figure(figsize=(10, 4.2), facecolor=C["bg"])

    ax_text = fig.add_axes([0.03, 0.05, 0.55, 0.9])
    ax_text.set_facecolor(C["bg"])
    ax_text.text(
        0.05,
        0.65,
        "hebbmem",
        fontsize=38,
        fontweight="bold",
        color=C["blue"],
        transform=ax_text.transAxes,
    )
    ax_text.text(
        0.05,
        0.40,
        "Hebbian memory for AI agents",
        fontsize=16,
        color=C["text"],
        transform=ax_text.transAxes,
    )
    ax_text.text(
        0.05,
        0.15,
        "pip install hebbmem",
        fontsize=14,
        color=C["gold"],
        transform=ax_text.transAxes,
        family="monospace",
    )
    ax_text.axis("off")

    ax_graph = fig.add_axes([0.58, 0.05, 0.38, 0.9])
    ax_graph.set_facecolor(C["bg"])
    _draw_graph_on_ax(ax_graph, n_nodes=8, seed=99)

    _save(fig, ASSETS / "social" / "devto_cover.png")


def gen_architecture() -> None:
    fig, ax = plt.subplots(figsize=(10, 6), facecolor=C["bg"])
    ax.set_facecolor(C["bg"])
    ax.axis("off")

    boxes = [
        # (x, y, w, h, label, sublabel, color)
        (
            0.1,
            0.78,
            0.8,
            0.15,
            "HebbMem (API)",
            "store / recall / step / save / load",
            C["blue"],
        ),
        (0.1, 0.55, 0.25, 0.18, "Encoder", "hash / sentence-\ntransformer", C["green"]),
        (0.38, 0.55, 0.25, 0.18, "MemoryGraph", "spread / hebbian\n/ decay", C["gold"]),
        (0.66, 0.55, 0.24, 0.18, "Persistence", "SQLite .hebb", C["orange"]),
        (
            0.1,
            0.28,
            0.8,
            0.2,
            "MemoryNode",
            "content | embedding | activation | strength | importance",
            C["text_muted"],
        ),
    ]

    for x, y, w, h, label, sublabel, color in boxes:
        rect = mpatches.FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.015",
            facecolor=C["bg_card"],
            edgecolor=color,
            linewidth=2,
            transform=ax.transAxes,
        )
        ax.add_patch(rect)
        ax.text(
            x + w / 2,
            y + h * 0.7,
            label,
            fontsize=13,
            fontweight="bold",
            color=color,
            ha="center",
            transform=ax.transAxes,
        )
        ax.text(
            x + w / 2,
            y + h * 0.25,
            sublabel,
            fontsize=9,
            color=C["text_muted"],
            ha="center",
            va="center",
            transform=ax.transAxes,
        )

    # Arrows
    for src_x in [0.22, 0.50, 0.78]:
        ax.annotate(
            "",
            xy=(src_x, 0.73),
            xytext=(src_x, 0.78),
            xycoords="axes fraction",
            arrowprops=dict(arrowstyle="->", color=C["text_muted"], lw=1.2),
        )
    ax.annotate(
        "",
        xy=(0.5, 0.48),
        xytext=(0.5, 0.55),
        xycoords="axes fraction",
        arrowprops=dict(arrowstyle="->", color=C["text_muted"], lw=1.2),
    )

    ax.set_title(
        "Architecture", fontsize=20, fontweight="bold", color=C["text"], pad=15
    )
    _save(fig, ASSETS / "diagrams" / "architecture.png")


def gen_comparison() -> None:
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(12, 6), facecolor=C["bg"])

    for ax in (ax_l, ax_r):
        ax.set_facecolor(C["bg"])
        ax.axis("off")

    # Left: flat vector search (grayscale)
    ax_l.set_title(
        "Flat Vector Search",
        fontsize=18,
        fontweight="bold",
        color=C["text_muted"],
        pad=10,
    )
    rng = np.random.RandomState(42)
    pts = rng.randn(12, 2) * 0.3
    query = np.array([0.0, 0.0])
    ax_l.scatter(
        pts[:, 0],
        pts[:, 1],
        s=150,
        c="#3a3a5a",
        edgecolors="#4a4a6a",
        linewidths=1.5,
        zorder=3,
    )
    ax_l.scatter([query[0]], [query[1]], s=200, c="#6a6a8a", marker="*", zorder=5)
    dists = np.linalg.norm(pts - query, axis=1)
    top3 = np.argsort(dists)[:3]
    for i in top3:
        ax_l.plot(
            [query[0], pts[i, 0]],
            [query[1], pts[i, 1]],
            color="#5a5a7a",
            lw=1.5,
            ls="--",
        )
    labels = ["no decay", "no association", "cosine only"]
    for i, lab in enumerate(labels):
        ax_l.text(
            0.5,
            0.08 - i * 0.06,
            lab,
            ha="center",
            fontsize=11,
            color=C["text_muted"],
            transform=ax_l.transAxes,
        )

    # Right: hebbmem (vibrant)
    ax_r.set_title("hebbmem", fontsize=18, fontweight="bold", color=C["blue"], pad=10)
    graph = nx.watts_strogatz_graph(12, 3, 0.4, seed=42)
    pos = nx.spring_layout(graph, seed=42)
    activations = rng.uniform(0.2, 1.0, 12)
    colors = [plt.cm.YlOrBr(a * 0.8 + 0.1) for a in activations]  # type: ignore[attr-defined]
    nx.draw_networkx_edges(
        graph, pos, ax=ax_r, edge_color=C["edge"], width=1.5, alpha=0.4
    )
    nx.draw_networkx_nodes(
        graph,
        pos,
        ax=ax_r,
        node_color=colors,
        node_size=[200 + a * 300 for a in activations],
        edgecolors=C["gold"],
        linewidths=[a * 2 for a in activations],
    )
    labels_r = ["memories decay", "co-recall bonds", "activation spreads"]
    for i, lab in enumerate(labels_r):
        ax_r.text(
            0.5,
            0.08 - i * 0.06,
            lab,
            ha="center",
            fontsize=11,
            color=C["green"],
            transform=ax_r.transAxes,
        )

    fig.suptitle(
        "Why hebbmem?", fontsize=22, fontweight="bold", color=C["text"], y=0.98
    )
    _save(fig, ASSETS / "diagrams" / "comparison.png")


def main() -> None:
    print("Generating hebbmem visual assets...\n")

    gen_hero_banner()
    gen_how_it_works()
    gen_benchmark_results()
    gen_recall_pipeline()
    gen_graph_example()
    gen_og_image()
    gen_x_thread_1()
    gen_x_thread_2()
    gen_x_thread_3()
    gen_devto_cover()
    gen_architecture()
    gen_comparison()

    print(f"\nDone! {sum(1 for _ in ASSETS.rglob('*.png'))} images generated.")


if __name__ == "__main__":
    main()
