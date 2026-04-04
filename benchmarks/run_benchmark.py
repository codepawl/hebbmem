"""Run hebbmem benchmark suite.

Usage:
    uv run python benchmarks/run_benchmark.py
    uv run python benchmarks/run_benchmark.py --encoder sentence-transformer
    uv run python benchmarks/run_benchmark.py --scenario associative
    uv run python benchmarks/run_benchmark.py --output results.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Add project root to path so benchmarks/ can import hebbmem
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.runner import format_results_table, run_all_scenarios
from hebbmem.encoders import HashEncoder, SentenceTransformerEncoder


def main() -> None:
    parser = argparse.ArgumentParser(description="hebbmem benchmark suite")
    parser.add_argument(
        "--encoder",
        choices=["hash", "sentence-transformer"],
        default="hash",
        help="Encoder to use (default: hash)",
    )
    parser.add_argument(
        "--scenario",
        choices=["temporal", "associative", "noise", "contradiction"],
        default=None,
        help="Run single scenario (default: all)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Save results to JSON file",
    )
    args = parser.parse_args()

    if args.encoder == "sentence-transformer":
        encoder = SentenceTransformerEncoder()
    else:
        encoder = HashEncoder()

    encoder_name = type(encoder).__name__
    print(f"Running benchmarks with {encoder_name}...\n")

    results = run_all_scenarios(encoder, scenario_filter=args.scenario)

    print("## hebbmem Benchmark Results\n")
    print(format_results_table(results, encoder_name))

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
