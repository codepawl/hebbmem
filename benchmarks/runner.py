"""Run all benchmark scenarios and collect results."""

from __future__ import annotations

from benchmarks.scenarios import associative, contradiction, noise, temporal
from hebbmem.encoders import EncoderBackend

SCENARIOS = {
    "temporal": temporal,
    "associative": associative,
    "noise": noise,
    "contradiction": contradiction,
}


def run_all_scenarios(
    encoder: EncoderBackend,
    scenario_filter: str | None = None,
) -> dict[str, dict[str, dict[str, float]]]:
    """Run scenarios and return results.

    Returns:
        {scenario_name: {"hebbmem": {metrics}, "baseline": {metrics}}}
    """
    results = {}
    targets = SCENARIOS
    if scenario_filter and scenario_filter in SCENARIOS:
        targets = {scenario_filter: SCENARIOS[scenario_filter]}

    for name, module in targets.items():
        results[name] = module.run(encoder)

    return results


def format_results_table(
    results: dict[str, dict[str, dict[str, float]]],
    encoder_name: str,
) -> str:
    """Format results as a markdown table."""
    from datetime import date

    lines = [
        f"Encoder: {encoder_name} | Date: {date.today()}",
        "",
        "| Scenario | Metric | hebbmem | Baseline | Delta |",
        "|----------|--------|---------|----------|-------|",
    ]

    for scenario, data in results.items():
        h = data["hebbmem"]
        b = data["baseline"]
        all_keys = list(dict.fromkeys(list(h.keys()) + list(b.keys())))
        for metric in all_keys:
            h_val = h.get(metric, 0.0)
            b_val = b.get(metric, 0.0)
            if b_val > 0:
                pct = (h_val - b_val) / b_val * 100
                delta_str = f"+{pct:.1f}%" if pct > 0 else f"{pct:.1f}%"
            elif h_val > 0:
                delta_str = "—"
            else:
                delta_str = "0"
            row = (
                f"| {scenario:<14} | {metric:<14} "
                f"| {h_val:.2f}    | {b_val:.2f}     "
                f"| {delta_str:<5} |"
            )
            lines.append(row)

    return "\n".join(lines)
