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
    "text2": "#9090b0",
    "text_dim": "#6a6a8a",
    "gold": "#f0b429",
    "blue": "#4a9eff",
    "orange": "#ff6b35",
    "green": "#2dd4a0",
    "red": "#ff4757",
    "edge": "#2a2a4a",
    "edge_hi": "#f0b429",
    "node_dim": "#1a1a3a",
}

plt.rcParams.update(
    {
        "font.family": "monospace",
        "text.color": C["text"],
        "axes.labelcolor": C["text"],
        "xtick.color": C["text2"],
        "ytick.color": C["text2"],
    }
)


def _save(fig: plt.Figure, path: Path) -> None:
    fig.savefig(
        path,
        dpi=200,
        bbox_inches="tight",
        facecolor=fig.get_facecolor(),
        pad_inches=0.3,
    )
    plt.close(fig)
    print(f"  {path.relative_to(ASSETS.parent)}")


def _trunc(s: str, n: int = 20) -> str:
    return s[:n] + "..." if len(s) > n else s


def _draw_graph(
    ax: plt.Axes,
    n: int = 8,
    seed: int = 42,
) -> None:
    """Draw a memory graph with circular nodes."""
    rng = np.random.RandomState(seed)
    g = nx.watts_strogatz_graph(n, 3, 0.4, seed=seed)
    pos = nx.spring_layout(g, seed=seed, k=1.8)
    acts = rng.uniform(0.1, 1.0, n)
    colors = [
        plt.cm.YlOrBr(a * 0.8 + 0.1)  # type: ignore[attr-defined]
        for a in acts
    ]
    sizes = [200 + a * 400 for a in acts]
    nx.draw_networkx_edges(
        g,
        pos,
        ax=ax,
        edge_color=C["edge"],
        width=1.5,
        alpha=0.5,
    )
    nx.draw_networkx_nodes(
        g,
        pos,
        ax=ax,
        node_color=colors,
        node_size=sizes,
        edgecolors=C["gold"],
        linewidths=[a * 2 for a in acts],
    )
    ax.set_aspect("equal")
    ax.axis("off")


# ── Hero Banner ──────────────────────────────────────────────


def gen_hero_banner() -> None:
    fig = plt.figure(figsize=(12, 4), facecolor=C["bg"])

    ax_t = fig.add_axes([0.03, 0.05, 0.48, 0.9])
    ax_t.set_facecolor(C["bg"])
    ax_t.text(
        0.05,
        0.65,
        "hebbmem",
        fontsize=48,
        fontweight="bold",
        color=C["blue"],
        transform=ax_t.transAxes,
    )
    ax_t.text(
        0.05,
        0.30,
        "memories that fire together\nwire together",
        fontsize=20,
        color=C["text2"],
        transform=ax_t.transAxes,
        linespacing=1.5,
    )
    ax_t.text(
        0.05,
        0.08,
        "pip install hebbmem",
        fontsize=16,
        color=C["gold"],
        transform=ax_t.transAxes,
        family="monospace",
    )
    ax_t.axis("off")

    ax_g = fig.add_axes([0.55, 0.05, 0.42, 0.9])
    ax_g.set_facecolor(C["bg"])
    _draw_graph(ax_g, n=10, seed=42)
    # glow on brightest node
    ax_g.scatter(
        [0],
        [0],
        s=1200,
        c=C["gold"],
        alpha=0.08,
        zorder=0,
        edgecolors="none",
    )

    _save(fig, ASSETS / "readme" / "hero_banner.png")


# ── How It Works (3 panels) ─────────────────────────────────


def gen_how_it_works() -> None:
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(16, 6),
        facecolor=C["bg"],
    )
    fig.subplots_adjust(wspace=0.15)

    for ax in axes:
        ax.set_facecolor(C["bg"])
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.axis("off")

    # Panel 1 — Decay
    ax = axes[0]
    ax.text(
        0.5,
        0.92,
        "Decay",
        fontsize=28,
        fontweight="bold",
        color=C["text"],
        ha="center",
        transform=ax.transAxes,
    )
    ax.text(
        0.5,
        0.82,
        "old memories fade",
        fontsize=18,
        color=C["text2"],
        ha="center",
        transform=ax.transAxes,
    )
    for i in range(7):
        alpha = 1.0 - i * 0.14
        col = plt.cm.YlOrBr(  # type: ignore[attr-defined]
            alpha * 0.8 + 0.1,
        )
        ax.add_patch(
            plt.Circle(
                (0.10 + i * 0.12, 0.50),
                0.04,
                color=col,
                alpha=max(alpha, 0.12),
                transform=ax.transAxes,
            )
        )
    ax.annotate(
        "",
        xy=(0.88, 0.35),
        xytext=(0.10, 0.35),
        xycoords="axes fraction",
        arrowprops=dict(arrowstyle="->", color=C["text2"], lw=1.5),
    )
    ax.text(
        0.5,
        0.25,
        "time",
        fontsize=16,
        color=C["text2"],
        ha="center",
        transform=ax.transAxes,
    )

    # Panel 2 — Hebbian
    ax = axes[1]
    ax.text(
        0.5,
        0.92,
        "Hebbian",
        fontsize=28,
        fontweight="bold",
        color=C["text"],
        ha="center",
        transform=ax.transAxes,
    )
    ax.text(
        0.5,
        0.82,
        "co-recalled memories bond",
        fontsize=18,
        color=C["text2"],
        ha="center",
        transform=ax.transAxes,
    )
    # before
    ax.add_patch(
        plt.Circle(
            (0.18, 0.52),
            0.05,
            color=C["blue"],
            transform=ax.transAxes,
        )
    )
    ax.add_patch(
        plt.Circle(
            (0.38, 0.52),
            0.05,
            color=C["blue"],
            transform=ax.transAxes,
        )
    )
    ax.plot(
        [0.23, 0.33],
        [0.52, 0.52],
        color=C["edge"],
        lw=1,
        transform=ax.transAxes,
    )
    ax.text(
        0.28,
        0.40,
        "weak",
        fontsize=14,
        color=C["text_dim"],
        ha="center",
        transform=ax.transAxes,
    )
    # arrow
    ax.annotate(
        "",
        xy=(0.56, 0.52),
        xytext=(0.46, 0.52),
        xycoords="axes fraction",
        arrowprops=dict(arrowstyle="->", color=C["gold"], lw=2),
    )
    # after
    ax.add_patch(
        plt.Circle(
            (0.66, 0.52),
            0.06,
            color=C["gold"],
            alpha=0.9,
            transform=ax.transAxes,
        )
    )
    ax.add_patch(
        plt.Circle(
            (0.86, 0.52),
            0.06,
            color=C["gold"],
            alpha=0.9,
            transform=ax.transAxes,
        )
    )
    ax.plot(
        [0.72, 0.80],
        [0.52, 0.52],
        color=C["gold"],
        lw=4,
        transform=ax.transAxes,
    )
    ax.text(
        0.76,
        0.40,
        "strong",
        fontsize=14,
        color=C["gold"],
        ha="center",
        transform=ax.transAxes,
    )
    ax.text(
        0.5,
        0.25,
        "recall together x5",
        fontsize=16,
        color=C["text2"],
        ha="center",
        transform=ax.transAxes,
    )

    # Panel 3 — Spreading
    ax = axes[2]
    ax.text(
        0.5,
        0.92,
        "Spreading",
        fontsize=28,
        fontweight="bold",
        color=C["text"],
        ha="center",
        transform=ax.transAxes,
    )
    ax.text(
        0.5,
        0.82,
        "recall activates neighbors",
        fontsize=18,
        color=C["text2"],
        ha="center",
        transform=ax.transAxes,
    )
    cx, cy = 0.5, 0.50
    ax.add_patch(
        plt.Circle(
            (cx, cy),
            0.06,
            color=C["gold"],
            transform=ax.transAxes,
            zorder=5,
        )
    )
    angles = np.linspace(0, 2 * np.pi, 6, endpoint=False)
    for a in angles:
        p1 = (cx + 0.16 * np.cos(a), cy + 0.16 * np.sin(a))
        ax.plot(
            [cx, p1[0]],
            [cy, p1[1]],
            color=C["orange"],
            lw=2,
            transform=ax.transAxes,
            alpha=0.7,
        )
        ax.add_patch(
            plt.Circle(
                p1,
                0.04,
                color=C["orange"],
                alpha=0.7,
                transform=ax.transAxes,
                zorder=4,
            )
        )
        p2x = cx + 0.28 * np.cos(a + 0.4)
        p2y = cy + 0.28 * np.sin(a + 0.4)
        ax.plot(
            [p1[0], p2x],
            [p1[1], p2y],
            color=C["orange"],
            lw=1,
            transform=ax.transAxes,
            alpha=0.3,
        )
        ax.add_patch(
            plt.Circle(
                (p2x, p2y),
                0.025,
                color=C["orange"],
                alpha=0.3,
                transform=ax.transAxes,
                zorder=3,
            )
        )
    ax.text(
        0.5,
        0.25,
        "energy spreads per hop",
        fontsize=16,
        color=C["text2"],
        ha="center",
        transform=ax.transAxes,
    )

    _save(fig, ASSETS / "readme" / "how_it_works.png")


# ── Benchmark Results ────────────────────────────────────────


def gen_benchmark_results() -> None:
    scenarios = [
        "Associative",
        "Contradiction",
        "Noise",
        "Temporal",
    ]
    hv = [0.56, 0.50, 0.60, 0.36]
    bv = [0.44, 0.00, 0.56, 0.36]
    met = ["P@5", "P@1", "P@5", "P@5"]
    deltas = ["+27%", "unique", "+7%", "tied"]

    fig, ax = plt.subplots(figsize=(10, 6), facecolor=C["bg"])
    ax.set_facecolor(C["bg"])

    y = np.arange(len(scenarios))
    h = 0.35

    ax.barh(
        y + h / 2,
        hv,
        h,
        label="hebbmem",
        color=C["green"],
        alpha=0.9,
    )
    ax.barh(
        y - h / 2,
        bv,
        h,
        label="Baseline",
        color=C["red"],
        alpha=0.5,
    )

    # Value + delta labels
    for i in range(len(scenarios)):
        ax.text(
            hv[i] + 0.02,
            y[i] + h / 2,
            f"{hv[i]:.2f}  {deltas[i]}",
            va="center",
            fontsize=14,
            color=C["gold"],
            fontweight="bold",
        )
        if bv[i] > 0:
            ax.text(
                bv[i] + 0.02,
                y[i] - h / 2,
                f"{bv[i]:.2f}",
                va="center",
                fontsize=12,
                color=C["text_dim"],
            )

    ax.set_yticks(y)
    ax.set_yticklabels(
        [f"{s}\n({m})" for s, m in zip(scenarios, met, strict=True)],
        fontsize=14,
    )
    ax.set_xlabel("Score", fontsize=16)
    ax.set_xlim(0, 0.90)
    ax.set_title(
        "hebbmem vs Flat Vector Search",
        fontsize=24,
        fontweight="bold",
        pad=20,
    )
    ax.text(
        0.5,
        -0.08,
        "Precision@K across 4 synthetic scenarios (HashEncoder)",
        fontsize=14,
        color=C["text2"],
        ha="center",
        transform=ax.transAxes,
    )
    ax.legend(
        loc="lower right",
        fontsize=14,
        facecolor=C["bg_card"],
        edgecolor=C["edge"],
    )
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    for s in ["bottom", "left"]:
        ax.spines[s].set_color(C["edge"])
    ax.grid(axis="x", color=C["edge"], alpha=0.3)

    _save(fig, ASSETS / "readme" / "benchmark_results.png")


# ── Recall Pipeline ──────────────────────────────────────────


def gen_recall_pipeline() -> None:
    steps = [
        ("Query", "input text", C["text_dim"]),
        ("Encode", "embed query", C["blue"]),
        ("Seeds", "cosine sim", C["blue"]),
        ("Spread", "BFS graph", C["gold"]),
        ("Hebbian", "strengthen", C["orange"]),
        ("Rank", "top-k", C["green"]),
    ]
    n = len(steps)
    bw = 1.4  # box width in data coords
    gap = 0.5
    total_w = n * bw + (n - 1) * gap
    fig_w = total_w + 2  # padding

    fig, ax = plt.subplots(
        figsize=(fig_w * 1.3, 3.5),
        facecolor=C["bg"],
    )
    ax.set_facecolor(C["bg"])
    ax.set_xlim(-0.5, total_w + 0.5)
    ax.set_ylim(-0.5, 2.5)
    ax.set_aspect("equal")
    ax.axis("off")

    ax.text(
        total_w / 2,
        2.2,
        "Recall Pipeline",
        fontsize=24,
        fontweight="bold",
        color=C["text"],
        ha="center",
    )

    for i, (title, sub, col) in enumerate(steps):
        x = i * (bw + gap)
        is_spread = i == 3
        fc = col if is_spread else C["bg_card"]
        tc = C["bg"] if is_spread else C["text"]
        sc = C["bg"] if is_spread else C["text2"]

        rect = mpatches.FancyBboxPatch(
            (x, 0.2),
            bw,
            1.2,
            boxstyle="round,pad=0.15",
            facecolor=fc,
            edgecolor=col,
            linewidth=2.5,
            alpha=0.95 if is_spread else 0.75,
        )
        ax.add_patch(rect)
        ax.text(
            x + bw / 2,
            1.0,
            title,
            fontsize=16,
            fontweight="bold",
            color=tc,
            ha="center",
            va="center",
        )
        ax.text(
            x + bw / 2,
            0.55,
            sub,
            fontsize=12,
            color=sc,
            ha="center",
            va="center",
        )

        if i < n - 1:
            ax.annotate(
                "",
                xy=(x + bw + gap * 0.7, 0.8),
                xytext=(x + bw + gap * 0.1, 0.8),
                arrowprops=dict(
                    arrowstyle="->",
                    color=C["text2"],
                    lw=2,
                ),
            )

    _save(fig, ASSETS / "readme" / "recall_pipeline.png")


# ── Graph Example ────────────────────────────────────────────


def gen_graph_example() -> None:
    from hebbmem import Config

    mem = HebbMem(
        encoder="hash",
        config=Config(auto_connect_threshold=0.25),
    )
    mems = [
        ("Atlas deadline March 15", 0.9),
        ("Atlas collaborative filtering", 0.7),
        ("Team meeting every Tuesday", 0.6),
        ("User loves Vietnamese coffee", 0.7),
        ("User studying Rust", 0.6),
        ("Rust async runtime saved", 0.4),
        ("Rewrite CLI tool in Rust", 0.7),
        ("Weather is nice today", 0.1),
        ("User said good morning", 0.05),
        ("Budget review end of Q", 0.8),
    ]
    for content, imp in mems:
        mem.store(content, importance=imp)

    for _ in range(4):
        mem.recall("Atlas project deadline")
    for _ in range(4):
        mem.recall("Rust programming language")
    mem.step(5)

    # Build nx graph
    g = nx.Graph()
    ndata: dict[str, dict] = {}
    for nid, node in mem._graph._nodes.items():
        sid = str(nid)[:8]
        g.add_node(sid)
        ndata[sid] = {
            "label": _trunc(node.content, 20),
            "act": node.activation,
            "str": node.base_strength,
            "imp": node.importance,
        }

    seen: set[tuple[str, str]] = set()
    for (src, tgt), edge in mem._graph._edges.items():
        a, b = str(src)[:8], str(tgt)[:8]
        key = (min(a, b), max(a, b))
        if key not in seen and a != b:
            seen.add(key)
            g.add_edge(a, b, weight=edge.weight)

    fig, ax = plt.subplots(figsize=(9, 9), facecolor=C["bg"])
    ax.set_facecolor(C["bg"])
    ax.set_aspect("equal")
    ax.axis("off")

    pos = nx.spring_layout(g, seed=42, k=2.5, iterations=80)

    # Edges
    for u, v, d in g.edges(data=True):
        w = d.get("weight", 0.1)
        col = C["edge_hi"] if w > 0.3 else C["edge"]
        ax.plot(
            [pos[u][0], pos[v][0]],
            [pos[u][1], pos[v][1]],
            color=col,
            lw=0.5 + w * 3,
            alpha=0.3 + w * 0.5,
            zorder=1,
        )

    # Nodes + labels
    for nid in g.nodes():
        d = ndata[nid]
        act = d["act"]
        col = plt.cm.YlOrBr(  # type: ignore[attr-defined]
            max(act * 2, 0.15),
        )
        sz = 8 + d["str"] * 12 + d["imp"] * 5
        ec = C["gold"] if act > 0.1 else C["edge"]
        ax.scatter(
            pos[nid][0],
            pos[nid][1],
            s=sz * 50,
            c=[col],
            edgecolors=ec,
            linewidths=1 + d["imp"] * 2,
            zorder=3,
        )
        ax.text(
            pos[nid][0],
            pos[nid][1] - 0.14,
            d["label"],
            fontsize=11,
            ha="center",
            color=C["text2"],
            zorder=4,
        )

    ax.set_title(
        "Memory Graph Example",
        fontsize=22,
        fontweight="bold",
        color=C["text"],
        pad=15,
    )
    _save(fig, ASSETS / "readme" / "graph_example.png")


# ── OG Image ─────────────────────────────────────────────────


def gen_og_image() -> None:
    fig = plt.figure(figsize=(12, 6.3), facecolor=C["bg"])

    ax_t = fig.add_axes([0.04, 0.08, 0.52, 0.84])
    ax_t.set_facecolor(C["bg"])
    ax_t.text(
        0.05,
        0.70,
        "hebbmem",
        fontsize=44,
        fontweight="bold",
        color=C["blue"],
        transform=ax_t.transAxes,
    )
    ax_t.text(
        0.05,
        0.48,
        "Hebbian memory\nfor AI agents",
        fontsize=22,
        color=C["text"],
        transform=ax_t.transAxes,
        linespacing=1.4,
    )
    ax_t.text(
        0.05,
        0.25,
        "pip install hebbmem",
        fontsize=18,
        color=C["gold"],
        transform=ax_t.transAxes,
        family="monospace",
    )
    # divider
    ax_t.plot(
        [0.05, 0.90],
        [0.18, 0.18],
        color=C["gold"],
        lw=0.5,
        alpha=0.5,
        transform=ax_t.transAxes,
    )
    ax_t.text(
        0.05,
        0.06,
        "github.com/codepawl/hebbmem",
        fontsize=14,
        color=C["text2"],
        transform=ax_t.transAxes,
    )
    ax_t.axis("off")

    ax_g = fig.add_axes([0.58, 0.1, 0.38, 0.8])
    ax_g.set_facecolor(C["bg"])
    _draw_graph(ax_g, n=10, seed=77)

    _save(fig, ASSETS / "social" / "og_image.png")


# ── X Thread Images ──────────────────────────────────────────


def gen_x_thread_1() -> None:
    fig, ax = plt.subplots(figsize=(16, 9), facecolor=C["bg"])
    ax.set_facecolor(C["bg"])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.axis("off")

    ax.text(
        0.5,
        0.78,
        "What if AI memory\nworked like your brain?",
        fontsize=40,
        fontweight="bold",
        color=C["text"],
        ha="center",
        transform=ax.transAxes,
        linespacing=1.3,
    )

    # Three visual icons: graph cluster, code, agent
    icons = [
        (0.25, "brain", C["gold"]),
        (0.50, "{ code }", C["blue"]),
        (0.75, "agent", C["green"]),
    ]
    for x, label, col in icons:
        ax.add_patch(
            plt.Circle(
                (x, 0.42),
                0.06,
                color=col,
                alpha=0.2,
                transform=ax.transAxes,
                zorder=3,
            )
        )
        ax.text(
            x,
            0.42,
            label,
            fontsize=18,
            ha="center",
            va="center",
            color=col,
            fontweight="bold",
            transform=ax.transAxes,
        )

    # Arrows
    for x in [0.35, 0.60]:
        ax.annotate(
            "",
            xy=(x + 0.06, 0.42),
            xytext=(x, 0.42),
            xycoords="axes fraction",
            arrowprops=dict(
                arrowstyle="->",
                color=C["gold"],
                lw=3,
            ),
        )

    ax.text(
        0.5,
        0.20,
        "pip install hebbmem",
        fontsize=24,
        color=C["gold"],
        ha="center",
        transform=ax.transAxes,
        family="monospace",
    )
    ax.text(
        0.95,
        0.05,
        "hebbmem",
        fontsize=16,
        color=C["blue"],
        ha="right",
        transform=ax.transAxes,
        alpha=0.5,
    )

    _save(fig, ASSETS / "social" / "x_thread_1.png")


def gen_x_thread_2() -> None:
    fig, ax = plt.subplots(figsize=(16, 9), facecolor=C["bg"])
    ax.set_facecolor(C["bg"])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.axis("off")

    ax.text(
        0.5,
        0.88,
        "Spreading Activation",
        fontsize=36,
        fontweight="bold",
        color=C["text"],
        ha="center",
        transform=ax.transAxes,
    )
    ax.text(
        0.5,
        0.80,
        'Query "Project Atlas" activates related memories',
        fontsize=18,
        color=C["text2"],
        ha="center",
        transform=ax.transAxes,
    )

    cx, cy = 0.5, 0.45
    # Seed node
    ax.add_patch(
        plt.Circle(
            (cx, cy),
            0.05,
            color=C["gold"],
            transform=ax.transAxes,
            zorder=5,
        )
    )
    ax.text(
        cx,
        cy - 0.08,
        "Project Atlas",
        fontsize=14,
        color=C["gold"],
        ha="center",
        fontweight="bold",
        transform=ax.transAxes,
    )

    # Hop 1 nodes with labels
    hop1 = [
        (0.18, "Atlas deadline", 0.8),
        (0.50, "Team meeting", 0.7),
        (0.82, "Atlas filtering", 0.7),
    ]
    angles_h1 = np.linspace(
        np.pi * 0.2,
        np.pi * 0.8,
        len(hop1),
    )
    for angle, (_, label, _alpha) in zip(
        angles_h1,
        hop1,
        strict=True,
    ):
        r = 0.18
        px = cx + r * np.cos(angle)
        py = cy + r * np.sin(angle)
        ax.plot(
            [cx, px],
            [cy, py],
            color=C["orange"],
            lw=2.5,
            transform=ax.transAxes,
            alpha=0.7,
        )
        ax.add_patch(
            plt.Circle(
                (px, py),
                0.035,
                color=C["orange"],
                alpha=0.7,
                transform=ax.transAxes,
                zorder=4,
            )
        )
        ax.text(
            px,
            py - 0.06,
            label,
            fontsize=11,
            color=C["text"],
            ha="center",
            transform=ax.transAxes,
        )

    # Hop 2 dimmer
    hop2_labels = ["Budget review", "Q2 planning"]
    angles_h2 = [np.pi * 0.35, np.pi * 0.65]
    for angle, label in zip(angles_h2, hop2_labels, strict=True):
        r2 = 0.32
        px = cx + r2 * np.cos(angle)
        py = cy + r2 * np.sin(angle)
        # connect to nearest hop1
        r1 = 0.18
        p1x = cx + r1 * np.cos(angle)
        p1y = cy + r1 * np.sin(angle)
        ax.plot(
            [p1x, px],
            [p1y, py],
            color=C["orange"],
            lw=1,
            transform=ax.transAxes,
            alpha=0.3,
        )
        ax.add_patch(
            plt.Circle(
                (px, py),
                0.025,
                color=C["orange"],
                alpha=0.3,
                transform=ax.transAxes,
                zorder=3,
            )
        )
        ax.text(
            px,
            py - 0.05,
            label,
            fontsize=10,
            color=C["text_dim"],
            ha="center",
            transform=ax.transAxes,
        )

    ax.text(
        0.95,
        0.05,
        "hebbmem",
        fontsize=16,
        color=C["blue"],
        ha="right",
        transform=ax.transAxes,
        alpha=0.5,
    )
    _save(fig, ASSETS / "social" / "x_thread_2.png")


def gen_x_thread_3() -> None:
    """Horizontal bar chart for X thread — same style as benchmark_results."""
    scenarios = [
        "Associative",
        "Contradiction",
        "Noise",
        "Temporal",
    ]
    hv = [0.56, 0.50, 0.60, 0.36]
    bv = [0.44, 0.00, 0.56, 0.36]
    met = ["P@5", "P@1", "P@5", "P@5"]
    deltas = ["+27%", "unique", "+7%", "tied"]

    fig, ax = plt.subplots(figsize=(16, 9), facecolor=C["bg"])
    ax.set_facecolor(C["bg"])

    y = np.arange(len(scenarios))
    h = 0.35

    ax.barh(
        y + h / 2,
        hv,
        h,
        label="hebbmem",
        color=C["green"],
        alpha=0.9,
    )
    ax.barh(
        y - h / 2,
        bv,
        h,
        label="Baseline",
        color=C["red"],
        alpha=0.5,
    )

    for i in range(len(scenarios)):
        ax.text(
            hv[i] + 0.02,
            y[i] + h / 2,
            f"{hv[i]:.2f}  {deltas[i]}",
            va="center",
            fontsize=18,
            color=C["gold"],
            fontweight="bold",
        )
        if bv[i] > 0:
            ax.text(
                bv[i] + 0.02,
                y[i] - h / 2,
                f"{bv[i]:.2f}",
                va="center",
                fontsize=14,
                color=C["text_dim"],
            )

    ax.set_yticks(y)
    ax.set_yticklabels(
        [f"{s}\n({m})" for s, m in zip(scenarios, met, strict=True)],
        fontsize=18,
    )
    ax.set_xlabel("Score", fontsize=18)
    ax.set_xlim(0, 0.90)
    ax.set_title(
        "hebbmem vs Flat Vector Search",
        fontsize=28,
        fontweight="bold",
        pad=20,
    )
    ax.text(
        0.5,
        -0.10,
        "Precision@K across 4 synthetic scenarios",
        fontsize=16,
        color=C["text2"],
        ha="center",
        transform=ax.transAxes,
    )
    ax.legend(
        loc="lower right",
        fontsize=16,
        facecolor=C["bg_card"],
        edgecolor=C["edge"],
    )
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    for s in ["bottom", "left"]:
        ax.spines[s].set_color(C["edge"])
    ax.grid(axis="x", color=C["edge"], alpha=0.3)

    fig.text(
        0.5,
        0.02,
        "pip install hebbmem",
        fontsize=22,
        color=C["gold"],
        ha="center",
        family="monospace",
    )
    fig.text(
        0.95,
        0.02,
        "hebbmem",
        fontsize=14,
        color=C["blue"],
        ha="right",
        alpha=0.5,
    )
    _save(fig, ASSETS / "social" / "x_thread_3.png")


# ── Dev.to Cover ─────────────────────────────────────────────


def gen_devto_cover() -> None:
    fig = plt.figure(figsize=(10, 4.2), facecolor=C["bg"])

    ax_t = fig.add_axes([0.04, 0.08, 0.50, 0.84])
    ax_t.set_facecolor(C["bg"])
    ax_t.text(
        0.05,
        0.65,
        "hebbmem",
        fontsize=38,
        fontweight="bold",
        color=C["blue"],
        transform=ax_t.transAxes,
    )
    ax_t.text(
        0.05,
        0.40,
        "Hebbian memory for AI agents",
        fontsize=18,
        color=C["text"],
        transform=ax_t.transAxes,
    )
    ax_t.text(
        0.05,
        0.15,
        "pip install hebbmem",
        fontsize=16,
        color=C["gold"],
        transform=ax_t.transAxes,
        family="monospace",
    )
    ax_t.axis("off")

    ax_g = fig.add_axes([0.60, 0.08, 0.36, 0.84])
    ax_g.set_facecolor(C["bg"])
    _draw_graph(ax_g, n=8, seed=99)

    _save(fig, ASSETS / "social" / "devto_cover.png")


# ── Architecture ─────────────────────────────────────────────


def gen_architecture() -> None:
    fig, ax = plt.subplots(figsize=(10, 6), facecolor=C["bg"])
    ax.set_facecolor(C["bg"])
    ax.axis("off")

    boxes = [
        (
            0.1,
            0.78,
            0.8,
            0.14,
            "HebbMem (API)",
            "store / recall / step / save / load",
            C["blue"],
        ),
        (0.1, 0.56, 0.24, 0.16, "Encoder", "hash / sentence-\ntransformer", C["green"]),
        (0.38, 0.56, 0.24, 0.16, "MemoryGraph", "spread / hebbian\n/ decay", C["gold"]),
        (0.66, 0.56, 0.24, 0.16, "Persistence", "SQLite .hebb", C["orange"]),
        (
            0.1,
            0.32,
            0.8,
            0.16,
            "MemoryNode",
            "content | embedding | activation\nstrength | importance | metadata",
            C["text2"],
        ),
    ]

    for x, y, w, h, label, sub, col in boxes:
        rect = mpatches.FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.015",
            facecolor=C["bg_card"],
            edgecolor=col,
            linewidth=2,
            transform=ax.transAxes,
        )
        ax.add_patch(rect)
        ax.text(
            x + w / 2,
            y + h * 0.68,
            label,
            fontsize=14,
            fontweight="bold",
            color=col,
            ha="center",
            transform=ax.transAxes,
        )
        ax.text(
            x + w / 2,
            y + h * 0.25,
            sub,
            fontsize=10,
            color=C["text2"],
            ha="center",
            va="center",
            transform=ax.transAxes,
        )

    # Arrows
    for sx in [0.22, 0.50, 0.78]:
        ax.annotate(
            "",
            xy=(sx, 0.72),
            xytext=(sx, 0.78),
            xycoords="axes fraction",
            arrowprops=dict(arrowstyle="->", color=C["text2"], lw=1.2),
        )
    ax.annotate(
        "",
        xy=(0.5, 0.48),
        xytext=(0.5, 0.56),
        xycoords="axes fraction",
        arrowprops=dict(arrowstyle="->", color=C["text2"], lw=1.2),
    )

    ax.set_title(
        "Architecture",
        fontsize=22,
        fontweight="bold",
        color=C["text"],
        pad=15,
    )
    _save(fig, ASSETS / "diagrams" / "architecture.png")


# ── Comparison ───────────────────────────────────────────────


def gen_comparison() -> None:
    """Side-by-side: flat vector search vs hebbmem. Rebuilt from scratch."""
    fig, (ax_l, ax_r) = plt.subplots(
        1,
        2,
        figsize=(14, 7),
        facecolor=C["bg"],
    )
    fig.subplots_adjust(wspace=0.08)

    for ax in (ax_l, ax_r):
        ax.set_facecolor(C["bg"])
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.axis("off")

    # ── Left half: Flat Vector Search ──
    ax_l.text(
        0.5,
        0.93,
        "Flat Vector Search",
        fontsize=22,
        fontweight="bold",
        color=C["text_dim"],
        ha="center",
        transform=ax_l.transAxes,
    )

    rng = np.random.RandomState(42)
    pts = rng.uniform(0.15, 0.85, (12, 2))
    pts[:, 1] = pts[:, 1] * 0.55 + 0.25  # keep in mid area

    ax_l.scatter(
        pts[:, 0],
        pts[:, 1],
        s=120,
        c="#4a4a6a",
        edgecolors="#5a5a7a",
        linewidths=1,
        zorder=3,
        transform=ax_l.transAxes,
    )

    # Query star
    qx, qy = 0.45, 0.52
    ax_l.scatter(
        [qx],
        [qy],
        s=400,
        c="#b0b0d0",
        marker="*",
        zorder=5,
        transform=ax_l.transAxes,
    )
    ax_l.text(
        qx + 0.05,
        qy + 0.03,
        "query",
        fontsize=14,
        color=C["text2"],
        transform=ax_l.transAxes,
    )

    # Dashed lines to 3 nearest
    dists = np.sqrt((pts[:, 0] - qx) ** 2 + (pts[:, 1] - qy) ** 2)
    top3 = np.argsort(dists)[:3]
    for idx in top3:
        ax_l.plot(
            [qx, pts[idx, 0]],
            [qy, pts[idx, 1]],
            color="#7a7a9a",
            lw=2,
            ls="--",
            transform=ax_l.transAxes,
            zorder=2,
        )

    # Labels
    left_labels = ["no decay", "no association", "cosine only"]
    for i, lab in enumerate(left_labels):
        ax_l.text(
            0.08,
            0.12 - i * 0.05,
            lab,
            fontsize=14,
            color=C["text_dim"],
            transform=ax_l.transAxes,
        )

    # ── Vertical divider ──
    fig.add_artist(
        plt.Line2D(
            [0.50, 0.50],
            [0.08, 0.92],
            color=C["edge"],
            lw=1,
            ls="--",
            transform=fig.transFigure,
        )
    )

    # ── Right half: hebbmem ──
    ax_r.text(
        0.5,
        0.93,
        "hebbmem",
        fontsize=22,
        fontweight="bold",
        color=C["blue"],
        ha="center",
        transform=ax_r.transAxes,
    )

    # Cluster A (4 nodes)
    ca = [(0.25, 0.65), (0.40, 0.72), (0.32, 0.52), (0.48, 0.58)]
    # Cluster B (4 nodes)
    cb = [(0.65, 0.68), (0.78, 0.60), (0.72, 0.48), (0.82, 0.72)]
    # Scattered (4 nodes)
    cs = [(0.20, 0.35), (0.55, 0.35), (0.75, 0.30), (0.45, 0.45)]

    all_pos = ca + cb + cs
    acts = [0.9, 0.7, 0.5, 0.4, 0.6, 0.5, 0.3, 0.4, 0.15, 0.2, 0.1, 0.25]

    # Intra-cluster edges
    edges = [
        (0, 1),
        (0, 2),
        (1, 3),
        (2, 3),  # cluster A
        (4, 5),
        (4, 6),
        (5, 7),
        (6, 7),  # cluster B
        (3, 9),
        (9, 4),  # bridge
    ]

    # Draw edges
    for i, j in edges:
        ax_r.plot(
            [all_pos[i][0], all_pos[j][0]],
            [all_pos[i][1], all_pos[j][1]],
            color=C["edge"],
            lw=1.5,
            alpha=0.5,
            transform=ax_r.transAxes,
            zorder=1,
        )

    # Draw nodes
    for (px, py), act in zip(all_pos, acts, strict=True):
        col = plt.cm.YlOrBr(  # type: ignore[attr-defined]
            act * 0.8 + 0.1,
        )
        sz = 80 + act * 200
        ec = C["gold"] if act > 0.5 else C["edge"]
        lw = 1 + act * 2
        ax_r.scatter(
            [px],
            [py],
            s=sz,
            c=[col],
            edgecolors=ec,
            linewidths=lw,
            zorder=3,
            transform=ax_r.transAxes,
        )

    # Labels
    right_labels = [
        "memories decay",
        "co-recall bonds",
        "activation spreads",
    ]
    for i, lab in enumerate(right_labels):
        ax_r.text(
            0.08,
            0.12 - i * 0.05,
            lab,
            fontsize=14,
            color=C["green"],
            transform=ax_r.transAxes,
        )

    fig.suptitle(
        "Why hebbmem?",
        fontsize=28,
        fontweight="bold",
        color=C["text"],
        y=0.98,
    )
    _save(fig, ASSETS / "diagrams" / "comparison.png")


# ── Main ─────────────────────────────────────────────────────


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
    total = sum(1 for _ in ASSETS.rglob("*.png"))
    print(f"\nDone! {total} images generated.")


if __name__ == "__main__":
    main()
