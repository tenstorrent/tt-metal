# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Benchmark harness for matmul auto-config.

Collects telemetry for all valid configs across representative shapes,
measures device latency, and optionally trains the DNN scorer.

Usage:
    python -m ttnn._experimental.auto_config.benchmark --op matmul --shapes shapes.json
    python -m ttnn._experimental.auto_config.benchmark --op matmul --sweep

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

# Allowed base directories for file I/O operations (Cycode SAST compliance)
_ALLOWED_BASE_DIRS = [
    os.path.realpath(os.getcwd()),
    os.path.realpath(os.path.join(os.path.expanduser("~"), ".ttnn")),
    os.path.realpath(os.environ.get("TTNN_AUTO_CONFIG_CACHE_DIR", os.path.join(os.path.expanduser("~"), ".ttnn"))),
]


def _sanitize_path(user_path: str) -> str:
    """Sanitize and validate a user-provided file path.

    Ensures the resolved path is within an allowed base directory,
    preventing directory traversal attacks.

    Raises:
        ValueError: If the resolved path is outside all allowed directories.
    """
    resolved = os.path.realpath(user_path)
    for base in _ALLOWED_BASE_DIRS:
        if resolved.startswith(base + os.sep) or resolved == base:
            return resolved
    raise ValueError(f"Path '{resolved}' is outside allowed directories. " f"Allowed: {_ALLOWED_BASE_DIRS}")


# Representative shapes for sweep mode — 100+ shapes covering all asymmetric
# M/K/N families (per @sankarmanoj-tt's feedback for DNN training diversity)
DEFAULT_SWEEP_SHAPES = [
    # ── Square baselines ────────────────────────────────────────────
    {"M": 32, "K": 32, "N": 32},
    {"M": 64, "K": 64, "N": 64},
    {"M": 128, "K": 128, "N": 128},
    {"M": 256, "K": 256, "N": 256},
    {"M": 512, "K": 512, "N": 512},
    {"M": 1024, "K": 1024, "N": 1024},
    {"M": 2048, "K": 2048, "N": 2048},
    {"M": 4096, "K": 4096, "N": 4096},
    # ── M large, K & N small ────────────────────────────────────────
    {"M": 2048, "K": 128, "N": 128},
    {"M": 2048, "K": 256, "N": 64},
    {"M": 4096, "K": 64, "N": 64},
    {"M": 4096, "K": 128, "N": 32},
    {"M": 8192, "K": 32, "N": 32},
    {"M": 8192, "K": 64, "N": 64},
    {"M": 2048, "K": 64, "N": 128},
    {"M": 4096, "K": 32, "N": 128},
    {"M": 1024, "K": 128, "N": 64},
    {"M": 1024, "K": 64, "N": 32},
    # ── K large, M & N small ────────────────────────────────────────
    {"M": 32, "K": 4096, "N": 32},
    {"M": 64, "K": 8192, "N": 64},
    {"M": 128, "K": 4096, "N": 128},
    {"M": 32, "K": 2048, "N": 64},
    {"M": 64, "K": 4096, "N": 32},
    {"M": 128, "K": 8192, "N": 32},
    {"M": 64, "K": 2048, "N": 128},
    {"M": 32, "K": 1024, "N": 128},
    {"M": 128, "K": 2048, "N": 64},
    {"M": 64, "K": 1024, "N": 64},
    # ── N large, M & K small ────────────────────────────────────────
    {"M": 32, "K": 128, "N": 4096},
    {"M": 64, "K": 64, "N": 8192},
    {"M": 128, "K": 256, "N": 4096},
    {"M": 32, "K": 64, "N": 2048},
    {"M": 64, "K": 128, "N": 4096},
    {"M": 128, "K": 64, "N": 8192},
    {"M": 32, "K": 256, "N": 2048},
    {"M": 64, "K": 32, "N": 4096},
    # ── M & N large, K small ───────────────────────────────────────
    {"M": 2048, "K": 128, "N": 2048},
    {"M": 4096, "K": 64, "N": 4096},
    {"M": 2048, "K": 64, "N": 4096},
    {"M": 4096, "K": 128, "N": 2048},
    {"M": 1024, "K": 128, "N": 2048},
    {"M": 2048, "K": 256, "N": 1024},
    # ── M & K large, N small ───────────────────────────────────────
    {"M": 2048, "K": 4096, "N": 32},
    {"M": 4096, "K": 2048, "N": 64},
    {"M": 2048, "K": 2048, "N": 128},
    {"M": 4096, "K": 4096, "N": 32},
    {"M": 1024, "K": 4096, "N": 64},
    {"M": 2048, "K": 1024, "N": 32},
    # ── K & N large, M small ───────────────────────────────────────
    {"M": 32, "K": 4096, "N": 4096},
    {"M": 64, "K": 8192, "N": 2048},
    {"M": 32, "K": 2048, "N": 4096},
    {"M": 128, "K": 4096, "N": 2048},
    {"M": 64, "K": 4096, "N": 4096},
    {"M": 32, "K": 8192, "N": 2048},
    # ── LLM decode (M=1, tile-padded to 32) ────────────────────────
    {"M": 32, "K": 4096, "N": 4096},  # LLaMA/GPT-J decode
    {"M": 32, "K": 4096, "N": 11008},  # LLaMA MLP up
    {"M": 32, "K": 11008, "N": 4096},  # LLaMA MLP down
    {"M": 32, "K": 4544, "N": 4672},  # Falcon-7B QKV
    {"M": 32, "K": 4544, "N": 18176},  # Falcon-7B MLP up
    {"M": 32, "K": 18176, "N": 4544},  # Falcon-7B MLP down
    {"M": 32, "K": 8192, "N": 8192},  # LLaMA-70B decode
    {"M": 32, "K": 8192, "N": 28672},  # LLaMA-70B MLP
    # ── LLM prefill ────────────────────────────────────────────────
    {"M": 128, "K": 4096, "N": 4096},
    {"M": 512, "K": 4096, "N": 11008},
    {"M": 2048, "K": 4096, "N": 4096},
    {"M": 128, "K": 4096, "N": 11008},
    {"M": 128, "K": 11008, "N": 4096},
    {"M": 256, "K": 4096, "N": 4096},
    {"M": 1024, "K": 4096, "N": 4096},
    {"M": 2048, "K": 4096, "N": 11008},
    {"M": 512, "K": 8192, "N": 8192},
    {"M": 2048, "K": 8192, "N": 8192},
    # ── Attention shapes (seq²) ────────────────────────────────────
    {"M": 2048, "K": 128, "N": 2048},  # QK^T
    {"M": 128, "K": 2048, "N": 128},  # V projection
    {"M": 4096, "K": 128, "N": 4096},  # Long-ctx QK^T
    {"M": 512, "K": 64, "N": 512},  # Small head attn
    {"M": 1024, "K": 128, "N": 1024},  # Mid-size attn
    {"M": 2048, "K": 64, "N": 2048},  # 64-head attn
    {"M": 256, "K": 128, "N": 256},  # Short-ctx attn
    {"M": 4096, "K": 64, "N": 4096},  # Long-ctx 64-head
    # ── Non-power-of-2 ─────────────────────────────────────────────
    {"M": 384, "K": 1024, "N": 768},
    {"M": 768, "K": 3072, "N": 768},
    {"M": 1536, "K": 2048, "N": 512},
    {"M": 192, "K": 768, "N": 3072},
    {"M": 1536, "K": 1536, "N": 1536},
    {"M": 384, "K": 384, "N": 1536},
    {"M": 768, "K": 768, "N": 3072},
    {"M": 1024, "K": 768, "N": 3072},
    # ── Tall shapes (M >> N) ───────────────────────────────────────
    {"M": 2048, "K": 1024, "N": 32},
    {"M": 4096, "K": 512, "N": 64},
    {"M": 8192, "K": 256, "N": 128},
    {"M": 4096, "K": 1024, "N": 128},
    # ── Wide shapes (N >> M) ───────────────────────────────────────
    {"M": 32, "K": 1024, "N": 2048},
    {"M": 64, "K": 512, "N": 4096},
    {"M": 128, "K": 256, "N": 8192},
    {"M": 64, "K": 1024, "N": 4096},
    # ── Batch/extra shapes ─────────────────────────────────────────
    {"M": 256, "K": 2048, "N": 256},
    {"M": 512, "K": 1024, "N": 512},
    {"M": 1024, "K": 512, "N": 1024},
    {"M": 2048, "K": 256, "N": 2048},
    {"M": 4096, "K": 256, "N": 4096},
    {"M": 1024, "K": 2048, "N": 1024},
]

# Import GPT/LLM attention training shapes for expanded coverage
try:
    from ttnn._experimental.auto_config.math_fidelity import GPT_ATTENTION_SHAPES

    for m, k, n, _desc in GPT_ATTENTION_SHAPES:
        shape = {"M": ((m + 31) // 32) * 32, "K": ((k + 31) // 32) * 32, "N": ((n + 31) // 32) * 32}
        if shape not in DEFAULT_SWEEP_SHAPES:
            DEFAULT_SWEEP_SHAPES.append(shape)
except ImportError:
    pass  # math_fidelity module not available


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
    from ttnn._experimental.auto_config.matmul_auto import MatmulAutoConfig

    import ttnn

    results = []

    # Open device
    device = ttnn.open_device(device_id=0)

    try:
        dt = ttnn.bfloat16 if dtype == "bfloat16" else ttnn.bfloat8_b

        # Build compute_kernel_config with appropriate math fidelity
        try:
            from ttnn._experimental.auto_config.math_fidelity import default_fidelity, fidelity_to_ttnn_string

            fidelity_str = fidelity_to_ttnn_string(default_fidelity(str(dt), str(dt)))
            math_fidelity = getattr(ttnn.MathFidelity, fidelity_str.split(".")[-1], ttnn.MathFidelity.HiFi4)
        except (ImportError, AttributeError):
            math_fidelity = ttnn.MathFidelity.HiFi4

        compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=math_fidelity,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

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
                                input_a,
                                input_b,
                                program_config=candidate.config,
                                compute_kernel_config=compute_kernel_config,
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
                                input_a,
                                input_b,
                                program_config=candidate.config,
                                compute_kernel_config=compute_kernel_config,
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
                        f"  [{cand_idx+1}/{len(valid_candidates)}] " f"{candidate.config_family}: {median_time:.0f} us"
                    )

                except Exception as e:
                    logger.warning(
                        f"  [{cand_idx+1}/{len(valid_candidates)}] " f"{candidate.config_family} FAILED: {e}"
                    )

            # Cleanup
            ttnn.deallocate(input_a)
            ttnn.deallocate(input_b)

    finally:
        ttnn.close_device(device)

    # Save results
    if output_file:
        # Path validated against allowed base directories (Cycode SAST compliant)
        output_file = _sanitize_path(output_file)
        os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"Benchmark results saved to {output_file}")

    return results


def train_dnn_scorer(benchmark_results: List[Dict[str, Any]], model_path: Optional[str] = None) -> None:
    """Train the DNN scorer on benchmark results."""
    from ttnn._experimental.auto_config.scorer.dnn_scorer import DNNScorer

    # Convert benchmark results to training data
    training_data = []
    for entry in benchmark_results:
        if entry.get("latency_us") is not None:
            training_data.append(
                {
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
                }
            )

    if not training_data:
        logger.warning("No valid training data from benchmark results")
        return

    scorer = DNNScorer(model_path=model_path)
    scorer.train(training_data, epochs=100)
    logger.info("DNN scorer training complete")


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Benchmark harness for matmul auto-config selection")
    parser.add_argument("--op", default="matmul", help="Operation to benchmark (currently only 'matmul')")
    parser.add_argument("--shapes", type=str, default=None, help="Path to shapes JSON file")
    parser.add_argument("--sweep", action="store_true", help="Run default sweep shapes")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "bfloat8_b"], help="Data type")
    parser.add_argument("--warmup", type=int, default=2, help="Number of warmup runs")
    parser.add_argument("--runs", type=int, default=5, help="Number of timed runs (takes median)")
    parser.add_argument("--output", type=str, default="benchmark_results.json", help="Output file for results")
    parser.add_argument("--train-dnn", action="store_true", help="Train DNN scorer on benchmark results")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    # Load shapes
    if args.shapes:
        # Path validated against allowed base directories (Cycode SAST compliant)
        shapes_path = _sanitize_path(args.shapes)
        with open(shapes_path, "r") as f:
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
