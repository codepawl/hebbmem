"""Run hebbmem benchmark suite.

Usage:
    uv run python benchmarks/run_benchmark.py
    uv run python benchmarks/run_benchmark.py --encoder sentence-transformer
    uv run python benchmarks/run_benchmark.py --scenario associative
    uv run python benchmarks/run_benchmark.py --locomo
    uv run python benchmarks/run_benchmark.py --locomo --conversations 1
    uv run python benchmarks/run_benchmark.py --all
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


def _make_encoder(name: str) -> HashEncoder | SentenceTransformerEncoder:
    if name == "sentence-transformer":
        return SentenceTransformerEncoder()
    return HashEncoder()


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
        help="Run single synthetic scenario (default: all)",
    )
    parser.add_argument(
        "--locomo",
        action="store_true",
        help="Run LoCoMo retrieval benchmark",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run both synthetic and LoCoMo benchmarks",
    )
    parser.add_argument(
        "--conversations",
        type=int,
        default=None,
        help="Limit LoCoMo to N conversations (for quick test)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed per-query results",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Save results to JSON file",
    )
    args = parser.parse_args()

    encoder = _make_encoder(args.encoder)
    encoder_name = type(encoder).__name__
    all_results: dict = {}

    run_synthetic = not args.locomo or args.all
    run_locomo = args.locomo or args.all

    if run_synthetic:
        print(f"Running synthetic benchmarks with {encoder_name}...\n")
        results = run_all_scenarios(encoder, scenario_filter=args.scenario)
        print("## hebbmem Benchmark Results\n")
        print(format_results_table(results, encoder_name))
        all_results["synthetic"] = results

    if run_locomo:
        from benchmarks.locomo.benchmark import run_locomo_benchmark

        print(f"\nRunning LoCoMo benchmark with {encoder_name}...\n")
        locomo_results = run_locomo_benchmark(
            encoder=encoder,
            conversations=args.conversations,
            verbose=args.verbose,
        )
        all_results["locomo"] = locomo_results

    if args.output:
        # Strip non-serializable table string
        output = {}
        for k, v in all_results.items():
            if isinstance(v, dict):
                output[k] = {dk: dv for dk, dv in v.items() if dk != "table"}
            else:
                output[k] = v
        with open(args.output, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
