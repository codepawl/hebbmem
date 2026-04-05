"""Run hebbmem on LoCoMo retrieval benchmark."""

from __future__ import annotations

from collections import defaultdict
from datetime import date

from benchmarks.baseline import FlatVectorSearch
from benchmarks.locomo.loader import Conversation, QAPair, Turn, parse_conversations
from hebbmem import Config, HebbMem
from hebbmem.encoders import EncoderBackend


def _evidence_match(
    retrieved_contents: list[str],
    evidence_turns: list[Turn],
) -> list[bool]:
    """Check which evidence turns appear in retrieved contents."""
    matches = []
    for ev in evidence_turns:
        found = any(
            ev.text in content or content in ev.text for content in retrieved_contents
        )
        matches.append(found)
    return matches


def _recall_at_k(
    retrieved: list[str],
    evidence_turns: list[Turn],
    k: int,
) -> float:
    """Of ground-truth evidence, how many in top-k retrieved?"""
    if not evidence_turns:
        return 0.0
    top = retrieved[:k]
    hits = sum(
        1 for ev in evidence_turns if any(ev.text in c or c in ev.text for c in top)
    )
    return hits / len(evidence_turns)


def _mrr(retrieved: list[str], evidence_turns: list[Turn]) -> float:
    """Reciprocal rank of first relevant result."""
    ev_texts = {t.text for t in evidence_turns}
    for i, content in enumerate(retrieved):
        for ev in ev_texts:
            if ev in content or content in ev:
                return 1.0 / (i + 1)
    return 0.0


def _ingest_conversation(
    mem: HebbMem | None,
    baseline: FlatVectorSearch | None,
    conv: Conversation,
) -> dict[str, Turn]:
    """Store all turns, return turn_id -> Turn mapping."""
    turn_map: dict[str, Turn] = {}
    prev_session = -1
    for turn in conv.turns:
        content = f"{turn.speaker}: {turn.text}"
        if turn.session_num != prev_session and prev_session >= 0 and mem:
            mem.step(10)
        if mem:
            mem.store(content, importance=0.5)
        if baseline:
            baseline.store(content, importance=0.5)
        turn_map[turn.turn_id] = turn
        prev_session = turn.session_num

    # Simulate natural conversation replay for Hebbian bonds
    if mem and len(conv.turns) > 10:
        sample_turns = conv.turns[:: max(1, len(conv.turns) // 8)]
        for t in sample_turns[:8]:
            mem.recall(t.text[:80], top_k=3)

    return turn_map


def _eval_qa(
    mem: HebbMem | None,
    baseline: FlatVectorSearch | None,
    qa: QAPair,
    turn_map: dict[str, Turn],
    top_k: int,
) -> dict[str, dict[str, float]]:
    """Evaluate one QA pair. Returns metrics for each system."""
    evidence_turns = [turn_map[tid] for tid in qa.evidence_turn_ids if tid in turn_map]
    if not evidence_turns:
        return {}

    results: dict[str, dict[str, float]] = {}

    if mem:
        h_retrieved = [r.content for r in mem.recall(qa.question, top_k=top_k)]
        results["hebbmem"] = {
            "recall@5": _recall_at_k(h_retrieved, evidence_turns, 5),
            "recall@10": _recall_at_k(
                h_retrieved,
                evidence_turns,
                10,
            ),
            "mrr": _mrr(h_retrieved, evidence_turns),
        }

    if baseline:
        b_retrieved = baseline.recall(qa.question, top_k=top_k)
        results["baseline"] = {
            "recall@5": _recall_at_k(b_retrieved, evidence_turns, 5),
            "recall@10": _recall_at_k(
                b_retrieved,
                evidence_turns,
                10,
            ),
            "mrr": _mrr(b_retrieved, evidence_turns),
        }

    return results


def run_locomo_benchmark(
    encoder: EncoderBackend,
    top_k: int = 10,
    conversations: int | None = None,
    verbose: bool = False,
) -> dict:
    """Run LoCoMo retrieval benchmark.

    Returns results per system, per category, and overall.
    """
    convs = parse_conversations(limit=conversations)
    if not convs:
        print("No conversations loaded. Check LoCoMo data.")
        return {}

    configs = {
        "hebbmem_full": Config(),
        "no_decay": Config(
            activation_decay=1.0,
            strength_decay=1.0,
            edge_decay=1.0,
        ),
        "no_hebbian": Config(hebbian_lr=0.0),
        "no_spread": Config(spread_factor=0.0),
    }

    # Collect per-category metrics for each system
    all_metrics: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list),
    )

    for ci, conv in enumerate(convs):
        if verbose:
            print(
                f"  Conv {ci + 1}/{len(convs)}: "
                f"{len(conv.turns)} turns, "
                f"{conv.num_sessions} sessions, "
                f"{len(conv.qa_pairs)} QAs",
            )

        # Run each config
        for cfg_name, cfg in configs.items():
            mem = HebbMem(encoder=encoder, config=cfg)
            turn_map = _ingest_conversation(mem, None, conv)

            for qa in conv.qa_pairs:
                res = _eval_qa(mem, None, qa, turn_map, top_k)
                if "hebbmem" in res:
                    cat = qa.category
                    for metric, val in res["hebbmem"].items():
                        key = f"{cfg_name}/{cat}/{metric}"
                        all_metrics[cfg_name][key].append(val)
                        key_overall = f"{cfg_name}/overall/{metric}"
                        all_metrics[cfg_name][key_overall].append(val)

        # Flat baseline
        baseline = FlatVectorSearch(encoder)
        turn_map = _ingest_conversation(None, baseline, conv)
        for qa in conv.qa_pairs:
            res = _eval_qa(None, baseline, qa, turn_map, top_k)
            if "baseline" in res:
                cat = qa.category
                for metric, val in res["baseline"].items():
                    key = f"baseline/{cat}/{metric}"
                    all_metrics["baseline"][key].append(val)
                    key_overall = f"baseline/overall/{metric}"
                    all_metrics["baseline"][key_overall].append(val)

    return _format_results(all_metrics, encoder, len(convs))


def _avg(vals: list[float]) -> float:
    return sum(vals) / len(vals) if vals else 0.0


def _format_results(
    all_metrics: dict[str, dict[str, list[float]]],
    encoder: EncoderBackend,
    n_convs: int,
) -> dict:
    """Format raw metrics into final results dict and print table."""
    systems = [
        "hebbmem_full",
        "no_decay",
        "no_hebbian",
        "no_spread",
        "baseline",
    ]
    labels = {
        "hebbmem_full": "hebbmem (full)",
        "no_decay": "  - no decay",
        "no_hebbian": "  - no Hebbian",
        "no_spread": "  - no spreading",
        "baseline": "Flat baseline",
    }
    metrics_list = ["recall@5", "recall@10", "mrr"]
    categories = [
        "single_hop",
        "multi_hop",
        "temporal",
        "open_domain",
    ]

    enc_name = type(encoder).__name__

    lines = [
        "## LoCoMo Retrieval Benchmark Results\n",
        f"Dataset: LoCoMo ({n_convs} conversations)",
        f"Encoder: {enc_name} | Date: {date.today()}\n",
        "### Overall",
        "| System | Recall@5 | Recall@10 | MRR |",
        "|--------|----------|-----------|-----|",
    ]

    result_data: dict[str, dict[str, float]] = {}

    for sys_name in systems:
        row_data: dict[str, float] = {}
        vals = []
        for m in metrics_list:
            key = f"{sys_name}/overall/{m}"
            v = _avg(all_metrics.get(sys_name, {}).get(key, []))
            row_data[m] = v
            vals.append(f"{v:.3f}")
        result_data[sys_name] = row_data
        label = labels[sys_name]
        lines.append(f"| {label:<18} | {vals[0]:<8} | {vals[1]:<9} | {vals[2]:<5}|")

    lines.extend(
        [
            "",
            "### By Category (Recall@5)",
            "| Category | hebbmem | Baseline | Delta |",
            "|----------|---------|----------|-------|",
        ]
    )

    for cat in categories:
        h_key = f"hebbmem_full/{cat}/recall@5"
        b_key = f"baseline/{cat}/recall@5"
        h_val = _avg(
            all_metrics.get("hebbmem_full", {}).get(h_key, []),
        )
        b_val = _avg(
            all_metrics.get("baseline", {}).get(b_key, []),
        )
        if b_val > 0:
            delta = (h_val - b_val) / b_val * 100
            d_str = f"+{delta:.0f}%" if delta > 0 else f"{delta:.0f}%"
        elif h_val > 0:
            d_str = "---"
        else:
            d_str = "0"
        lines.append(f"| {cat:<12} | {h_val:.3f}   | {b_val:.3f}    | {d_str:<5} |")

    table = "\n".join(lines)
    print(table)

    return {
        "overall": result_data,
        "table": table,
        "n_conversations": n_convs,
        "encoder": enc_name,
    }
