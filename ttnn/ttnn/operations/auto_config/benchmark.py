# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Benchmark harness for matmul auto-config.

Collects telemetry for all valid configs across representative shapes,
measures device latency, and optionally trains the DNN scorer.

Usage:
    python -m ttnn.operations.auto_config.benchmark --op matmul --shapes shapes.json
    python -m ttnn.operations.auto_config.benchmark --op matmul --sweep

Example shapes.json:
    [
        {"M": 1024, "K": 1024, "N": 1024},
        {"M": 2048, "K": 4096, "N": 4096},
        {"M": 128, "K": 768, "N": 3072}
    ]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# Representative shapes for sweep mode
DEFAULT_SWEEP_SHAPES = [
    # Small shapes
    {"M": 32, "K": 32, "N": 32},
    {"M": 64, "K": 64, "N": 64},
    {"M": 128, "K": 128, "N": 128},
    # Medium shapes
    {"M": 256, "K": 256, "N": 256},
    {"M": 512, "K": 512, "N": 512},
    {"M": 1024, "K": 1024, "N": 1024},
    # Large shapes
    {"M": 2048, "K": 2048, "N": 2048},
    {"M": 4096, "K": 4096, "N": 4096},
    # Tall shapes (M >> N)
    {"M": 2048, "K": 1024, "N": 32},
    {"M": 4096, "K": 512, "N": 64},
    # Wide shapes (N >> M)
    {"M": 32, "K": 1024, "N": 2048},
    {"M": 64, "K": 512, "N": 4096},
    # Common LLM shapes
    {"M": 128, "K": 4096, "N": 4096},
    {"M": 128, "K": 4096, "N": 11008},
    {"M": 128, "K": 11008, "N": 4096},
    {"M": 32, "K": 4096, "N": 4096},
    {"M": 32, "K": 4096, "N": 4096},
    # Attention shapes
    {"M": 2048, "K": 128, "N": 2048},
    {"M": 128, "K": 2048, "N": 128},
    # Non-power-of-2
    {"M": 384, "K": 1024, "N": 1024},
    {"M": 768, "K": 3072, "N": 768},
]


def run_benchmark(
    shapes: List[Dict[str, int]],
    dtype: str = "bfloat16",
    num_warmup: int = 2,
    num_runs: int = 5,
    output_file: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Run benchmark on the given shapes.

    Requires TT hardware. Each shape is tested with all valid config candidates.

    Args:
        shapes: List of shape dicts with M, K, N keys.
        dtype: Data type to use.
        num_warmup: Number of warmup runs per candidate.
        num_runs: Number of timed runs per candidate (takes median).
        output_file: Path to save benchmark results as JSON.

    Returns:
        List of benchmark result dicts.
    """
    import torch
    import ttnn
    from ttnn.operations.auto_config.matmul_auto import MatmulAutoConfig

    results = []

    # Open device
    device = ttnn.open_device(device_id=0)

    try:
        dt = ttnn.bfloat16 if dtype == "bfloat16" else ttnn.bfloat8_b

        for shape_idx, shape in enumerate(shapes):
            M, K, N = shape["M"], shape["K"], shape["N"]

            # Pad to tile-aligned
            M_padded = ((M + 31) // 32) * 32
            K_padded = ((K + 31) // 32) * 32
            N_padded = ((N + 31) // 32) * 32

            logger.info(f"\n[{shape_idx+1}/{len(shapes)}] Shape: M={M_padded}, K={K_padded}, N={N_padded}")

            # Create tensors
            torch_a = torch.randn(1, M_padded, K_padded)
            torch_b = torch.randn(1, K_padded, N_padded)

            input_a = ttnn.from_torch(torch_a, dtype=dt, layout=ttnn.TILE_LAYOUT, device=device)
            input_b = ttnn.from_torch(torch_b, dtype=dt, layout=ttnn.TILE_LAYOUT, device=device)

            # Get auto-config selection
            selector = MatmulAutoConfig()
            result = selector.select(input_a, input_b)

            valid_candidates = [c for c in result.all_candidates if c.is_valid]
            logger.info(f"  {len(valid_candidates)} valid candidates")

            # Benchmark each valid candidate
            for cand_idx, candidate in enumerate(valid_candidates):
                try:
                    # Warmup
                    for _ in range(num_warmup):
                        if candidate.backend == "matmul" and candidate.config is not None:
                            out = ttnn.matmul(
                                input_a, input_b,
                                program_config=candidate.config,
                            )
                            ttnn.synchronize_device(device)
                            ttnn.deallocate(out)

                    # Timed runs
                    times = []
                    for _ in range(num_runs):
                        ttnn.synchronize_device(device)
                        start = time.perf_counter()

                        if candidate.backend == "matmul" and candidate.config is not None:
                            out = ttnn.matmul(
                                input_a, input_b,
                                program_config=candidate.config,
                            )
                        else:
                            out = ttnn.matmul(input_a, input_b)

                        ttnn.synchronize_device(device)
                        elapsed = (time.perf_counter() - start) * 1e6
                        times.append(elapsed)
                        ttnn.deallocate(out)

                    median_time = sorted(times)[len(times) // 2]
                    candidate.measured_latency_us = median_time

                    result_entry = {
                        "shape": {"M": M_padded, "K": K_padded, "N": N_padded},
                        "config_family": candidate.config_family,
                        "backend": candidate.backend,
                        "params": candidate.params,
                        "score": candidate.score,
                        "latency_us": median_time,
                        "all_times_us": times,
                    }
                    results.append(result_entry)

                    logger.info(
                        f"  [{cand_idx+1}/{len(valid_candidates)}] "
                        f"{candidate.config_family}: {median_time:.0f} us"
                    )

                except Exception as e:
                    logger.warning(
                        f"  [{cand_idx+1}/{len(valid_candidates)}] "
                        f"{candidate.config_family} FAILED: {e}"
                    )

            # Cleanup
            ttnn.deallocate(input_a)
            ttnn.deallocate(input_b)

    finally:
        ttnn.close_device(device)

    # Save results
    if output_file:
        os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"Benchmark results saved to {output_file}")

    return results


def train_dnn_scorer(benchmark_results: List[Dict[str, Any]], model_path: Optional[str] = None) -> None:
    """Train the DNN scorer on benchmark results."""
    from ttnn.operations.auto_config.scorer.dnn_scorer import DNNScorer

    # Convert benchmark results to training data
    training_data = []
    for entry in benchmark_results:
        if entry.get("latency_us") is not None:
            training_data.append({
                "features": {
                    "M": entry["shape"]["M"],
                    "K": entry["shape"]["K"],
                    "N": entry["shape"]["N"],
                    "batch_size_a": 1,
                    "batch_size_b": 1,
                    "M_tiles": entry["shape"]["M"] // 32,
                    "K_tiles": entry["shape"]["K"] // 32,
                    "N_tiles": entry["shape"]["N"] // 32,
                    "grid_x": 8,
                    "grid_y": 8,
                    "num_cores": 64,
                    "num_devices": 1,
                },
                "config_params": entry["params"],
                "config_family": entry["config_family"],
                "latency_us": entry["latency_us"],
            })

    if not training_data:
        logger.warning("No valid training data from benchmark results")
        return

    scorer = DNNScorer(model_path=model_path)
    scorer.train(training_data, epochs=100)
    logger.info("DNN scorer training complete")


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Benchmark harness for matmul auto-config selection"
    )
    parser.add_argument(
        "--op", default="matmul", help="Operation to benchmark (currently only 'matmul')"
    )
    parser.add_argument(
        "--shapes", type=str, default=None, help="Path to shapes JSON file"
    )
    parser.add_argument(
        "--sweep", action="store_true", help="Run default sweep shapes"
    )
    parser.add_argument(
        "--dtype", default="bfloat16", choices=["bfloat16", "bfloat8_b"],
        help="Data type"
    )
    parser.add_argument(
        "--warmup", type=int, default=2, help="Number of warmup runs"
    )
    parser.add_argument(
        "--runs", type=int, default=5, help="Number of timed runs (takes median)"
    )
    parser.add_argument(
        "--output", type=str, default="benchmark_results.json",
        help="Output file for results"
    )
    parser.add_argument(
        "--train-dnn", action="store_true",
        help="Train DNN scorer on benchmark results"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Verbose logging"
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    # Load shapes
    if args.shapes:
        with open(args.shapes, "r") as f:
            shapes = json.load(f)
    elif args.sweep:
        shapes = DEFAULT_SWEEP_SHAPES
    else:
        logger.error("Must specify --shapes <file> or --sweep")
        sys.exit(1)

    logger.info(f"Benchmarking {len(shapes)} shapes with dtype={args.dtype}")

    results = run_benchmark(
        shapes,
        dtype=args.dtype,
        num_warmup=args.warmup,
        num_runs=args.runs,
        output_file=args.output,
    )

    # Print summary
    if results:
        logger.info(f"\n{'='*60}")
        logger.info(f"Benchmark complete: {len(results)} configs measured")

        # Group by shape
        by_shape = {}
        for r in results:
            key = f"M={r['shape']['M']},K={r['shape']['K']},N={r['shape']['N']}"
            if key not in by_shape:
                by_shape[key] = []
            by_shape[key].append(r)

        for shape_key, entries in by_shape.items():
            entries.sort(key=lambda e: e["latency_us"])
            best = entries[0]
            logger.info(
                f"  {shape_key}: Best={best['config_family']} "
                f"({best['latency_us']:.0f} us), "
                f"{len(entries)} configs tested"
            )

    # Train DNN scorer if requested
    if args.train_dnn and results:
        train_dnn_scorer(results)


if __name__ == "__main__":
    main()
